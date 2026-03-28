import cv2
import os
import torch
import whisperx
import numpy as np
import re
import logging
from paddleocr import PaddleOCR
from Levenshtein import ratio as levenshtein_ratio
from datetime import timedelta
import ffmpeg
from deep_translator import GoogleTranslator

class ProfessionalSubtitleSystem:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Sistem Subtitle Fusion Profesional (V2):
        - WhisperX (Audio) + PaddleOCR (Visual).
        - Generasi File Original & File Terjemahan terpisah.
        - Fallback & Error Handling Robust.
        """
        self.device = device
        print(f"Loading WhisperX (device={device})...")
        try:
            self.whisper_model = whisperx.load_model("medium", device, compute_type="float16" if device=="cuda" else "int8")
        except Exception as e:
            print(f"WhisperX load failed: {e}. Falling back to 'base' model.")
            self.whisper_model = whisperx.load_model("base", device)
        
        print("Loading PaddleOCR...")
        # Inisialisasi tanpa use_gpu untuk kompatibilitas versi
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        
        self.translator = GoogleTranslator(source='auto', target='id')
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def translate_contextual(self, text, previous_context=""):
        """Terjemahan natural berbasis konteks."""
        if not text or len(text) < 2: return ""
        try:
            full_input = f"{previous_context}\n{text}" if previous_context else text
            translated = self.translator.translate(full_input)
            if "\n" in translated:
                translated = translated.split("\n")[-1]
            return translated.strip()
        except:
            return text

    def wrap_text(self, text, max_chars=42):
        """Line-wrapping profesional untuk subtitle."""
        if not text or len(text) <= max_chars: return text
        mid = len(text) // 2
        split_idx = text.rfind(" ", 0, mid + 10)
        if split_idx == -1: split_idx = mid
        return f"{text[:split_idx].strip()}\n{text[split_idx:].strip()}"

    def extract_audio(self, video_path, output_audio="temp_audio.wav"):
        """Ekstrak audio WAV 16kHz mono."""
        if os.path.exists(output_audio): os.remove(output_audio)
        try:
            (
                ffmpeg
                .input(video_path)
                .output(output_audio, acodec='pcm_s16le', ac=1, ar='16k')
                .run(quiet=True, overwrite_output=True)
            )
            return output_audio
        except Exception as e:
            print(f"Audio extraction failed: {e}")
            return None

    def align_audio(self, video_path, audio_path):
        """Transkripsi & Alignment dengan Fallback Bahasa."""
        try:
            result = self.whisper_model.transcribe(audio_path, batch_size=16)
            detected_lang = result.get("language", "en") # Fallback ke 'en'
            
            # Coba alignment
            try:
                model_a, metadata = whisperx.load_align_model(language_code=detected_lang, device=self.device)
                result = whisperx.align(result["segments"], model_a, metadata, audio_path, self.device, return_char_alignments=False)
                return result["segments"], detected_lang
            except:
                # Jika alignment gagal, gunakan segmen asli dari transcribe
                return result["segments"], detected_lang
        except Exception as e:
            print(f"Transcription failed: {e}")
            return [], "unknown"

    def run_ocr_on_segment(self, cap, start_sec, end_sec, fps):
        """Jalankan OCR dengan try-except."""
        try:
            mid_time = (start_sec + end_sec) / 2
            frame_no = int(mid_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if not ret: return ""
            
            # Preprocessing
            h, w = frame.shape[:2]
            crop = frame[int(h * 0.6):h, :]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            enhanced = self.clahe.apply(gray)
            processed = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            result = self.ocr.ocr(processed, cls=True)
            texts = []
            if result and result[0]:
                for line in result[0]:
                    if line[1][1] > 0.45:
                        texts.append(line[1][0])
            return " ".join(texts)
        except:
            return ""

    def process_full_subtitle(self, video_path, progress_callback=None):
        """
        Menghasilkan data subtitle Original & Terjemahan.
        Returns: (original_subs, translated_subs, language)
        """
        audio_path = self.extract_audio(video_path)
        if not audio_path:
            # Jika audio gagal, coba OCR-only (perlu dikembangkan/fallback)
            return [], [], "unknown"

        segments, lang = self.align_audio(video_path, audio_path)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_segs = len(segments)
        
        orig_subs = []
        trans_subs = []
        previous_text = ""
        
        for i, seg in enumerate(segments):
            start, end = seg.get('start', 0), seg.get('end', 0)
            audio_text = seg.get('text', '').strip()
            
            # Fusion: Cek OCR untuk hardsub
            ocr_text = self.run_ocr_on_segment(cap, start, end, fps)
            
            # Gunakan OCR jika ada, jika tidak pakai audio (Fallback)
            final_orig = ocr_text if len(ocr_text) > 3 else audio_text
            final_orig = " ".join(final_orig.split()) # Clean spaces
            
            if not final_orig: continue
            
            # 1. Simpan Original
            orig_subs.append({
                'index': len(orig_subs) + 1,
                'start': self.format_time(start),
                'end': self.format_time(end),
                'content': self.wrap_text(final_orig)
            })
            
            # 2. Terjemahkan ke Indonesia
            final_trans = self.translate_contextual(final_orig, previous_context=previous_text)
            previous_text = final_orig
            
            trans_subs.append({
                'index': len(trans_subs) + 1,
                'start': self.format_time(start),
                'end': self.format_time(end),
                'content': self.wrap_text(final_trans)
            })
            
            if progress_callback:
                progress_callback((i + 1) / total_segs * 100)
        
        cap.release()
        if os.path.exists(audio_path): os.remove(audio_path)
        
        return orig_subs, trans_subs, lang

    def format_time(self, seconds):
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds_part = divmod(remainder, 60)
        milliseconds = int((td.total_seconds() - total_seconds) * 1000)
        return f"{hours:02}:{minutes:02}:{seconds_part:02},{milliseconds:03}"

    def save_srt(self, subtitles, output_path):
        """Simpan list subtitle ke file .srt."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for sub in subtitles:
                f.write(f"{sub['index']}\n")
                f.write(f"{sub['start']} --> {sub['end']}\n")
                f.write(f"{sub['content']}\n\n")
        return output_path
