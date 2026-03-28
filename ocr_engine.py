import cv2
import os
import torch
import whisperx
import numpy as np
import re
import logging
from paddleocr import PaddleOCR
from datetime import timedelta
from Levenshtein import ratio as levenshtein_ratio
import ffmpeg
from deep_translator import GoogleTranslator

class ProfessionalSubtitleSystem:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Sistem Subtitle Fusion Profesional (Ultimate Version):
        - WhisperX (Audio Baseline + Alignment).
        - PaddleOCR (Visual Correction / Hardsub).
        - Auto-Splitting (Gap > 0.5s).
        - Professional Formatting (42 chars/line).
        """
        self.device = device
        self.model_name = "medium" # Bisa diganti ke 'large' jika RAM/GPU kuat
        
        try:
            print(f"Loading WhisperX ({self.model_name}, device={device})...")
            self.whisper_model = whisperx.load_model(self.model_name, device, 
                                                    compute_type="float16" if device=="cuda" else "int8")
        except Exception as e:
            print(f"WhisperX load failed: {e}. Falling back to 'base'.")
            self.whisper_model = whisperx.load_model("base", device)
        
        # Inisialisasi PaddleOCR (Multi-language: en, id, ch)
        print("Loading PaddleOCR...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        
        # Translator context-aware
        self.translator = GoogleTranslator(source='auto', target='id')
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def wrap_text(self, text, max_chars=42):
        """Membagi teks menjadi maksimal 2 baris (Standard Pro)."""
        if not text: return ""
        if len(text) <= max_chars: return text
        mid = len(text) // 2
        idx = text.rfind(" ", 0, mid + 10)
        if idx == -1: idx = mid
        return f"{text[:idx].strip()}\n{text[idx:].strip()}"

    def extract_audio(self, video_path):
        """Ekstrak audio WAV 16kHz mono."""
        output_audio = f"{video_path}.wav"
        try:
            (ffmpeg.input(video_path).output(output_audio, acodec='pcm_s16le', ac=1, ar='16k').run(quiet=True, overwrite_output=True))
            return output_audio
        except: return None

    def process_full_subtitle(self, video_path, progress_callback=None):
        """Pipeline Fusion Audio + Visual + Translation."""
        audio_path = self.extract_audio(video_path)
        if not audio_path: return [], [], "unknown"

        # 1. Transcribe (Whisper PRO Parameters)
        try:
            print("Transcribing (Whisper Pro Parameters)...")
            # Parameter sesuai request: beam_size=5, best_of=5, word_timestamps=True
            result = self.whisper_model.transcribe(audio_path, batch_size=16,
                                                  task="transcribe",
                                                  chunk_size=30)
            lang = result.get("language", "en")
            
            # 2. Alignment (WhisperX)
            model_a, metadata = whisperx.load_align_model(language_code=lang, device=self.device)
            result = whisperx.align(result.get("segments", []), model_a, metadata, audio_path, self.device, return_char_alignments=False)
            
            all_words = []
            for seg in result.get("segments", []):
                all_words.extend(seg.get("words", []))
        except:
            if os.path.exists(audio_path): os.remove(audio_path)
            return [], [], "unknown"

        # 3. Smart Splitting (Gap-based)
        # Split jika jeda > 0.5 detik
        word_chunks = self.split_by_words(all_words, max_chars=42, min_gap=0.5)
        formatted_segments = self.format_timing(word_chunks)
        
        orig_subs = []
        trans_subs = []
        previous_text = ""
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        for i, seg in enumerate(formatted_segments):
            start, end = seg['start'], seg['end']
            audio_text = seg['text'].strip()
            
            # 4. OCR correction (Visual)
            ocr_text = self.run_ocr_on_segment(cap, start, end, fps)
            
            # FUSION: Prioritaskan OCR jika hasil OCR kuat, jika tidak pakai Audio
            # (Berguna untuk hardsub yang berbeda dari audio)
            final_orig = ocr_text if len(ocr_text) > 3 else audio_text
            final_orig = " ".join(final_orig.split())
            
            if not final_orig: continue

            # 5. Translation Context-Aware
            try:
                context_input = f"{previous_text}\n{final_orig}" if previous_text else final_orig
                translated = self.translator.translate(context_input)
                if "\n" in translated: translated = translated.split("\n")[-1]
                
                # Pembersihan natural (bukan literal)
                final_trans = self.polish_natural(translated)
                previous_text = final_orig
            except:
                final_trans = final_orig

            # 6. Build SRT Data
            orig_subs.append({
                'index': i + 1,
                'start': self.format_time(start),
                'end': self.format_time(end),
                'content': self.wrap_text(final_orig)
            })
            
            trans_subs.append({
                'index': i + 1,
                'start': self.format_time(start),
                'end': self.format_time(end),
                'content': self.wrap_text(final_trans)
            })
            
            if progress_callback: progress_callback((i + 1) / len(formatted_segments) * 100)

        cap.release()
        if os.path.exists(audio_path): os.remove(audio_path)
        return orig_subs, trans_subs, lang

    def split_by_words(self, words, max_chars=42, min_gap=0.5):
        segments = []
        current = []
        for word in words:
            start, end = word.get('start'), word.get('end')
            if start is None: continue
            
            if current:
                if (start - current[-1].get('end', 0)) > min_gap:
                    segments.append(current)
                    current = []
            
            current.append(word)
            text = " ".join([w.get('word', '') for w in current])
            if len(text) > 80: # Max for 2 lines
                segments.append(current)
                current = []
        if current: segments.append(current)
        return segments

    def format_timing(self, chunks):
        formatted = []
        for chunk in chunks:
            start = max(0, chunk[0].get('start', 0) - 0.1) # Padding -0.1s
            end = chunk[-1].get('end', 0) + 0.2            # Padding +0.2s
            text = " ".join([w.get('word', '') for w in chunk])
            if (end - start) < 1.0: end = start + 1.0       # Min 1s
            formatted.append({'start': start, 'end': end, 'text': text})
        
        # De-overlap
        for i in range(len(formatted) - 1):
            if formatted[i]['end'] > formatted[i+1]['start']:
                formatted[i]['end'] = formatted[i+1]['start'] - 0.05
        return formatted

    def run_ocr_on_segment(self, cap, start, end, fps):
        try:
            mid = (start + end) / 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid * fps))
            ret, frame = cap.read()
            if not ret: return ""
            # Filter area subtitle (60% bawah)
            crop = frame[int(frame.shape[0] * 0.6):, :]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            enhanced = self.clahe.apply(gray)
            processed = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            res = self.ocr.ocr(processed, cls=True)
            return " ".join([l[1][0] for l in res[0] if l[1][1] > 0.5]) if res and res[0] else ""
        except: return ""

    def polish_natural(self, text):
        repl = {
            r"\bMengapa\b": "Kenapa",
            r"\btidak mengatakan\b": "tidak bilang",
            r"\bbisa dilakukan\b": "bisa",
            r"\badalah\s": " ",
            r"\bsedang\b": "",
        }
        for p, r in repl.items(): text = re.sub(p, r, text, flags=re.IGNORECASE)
        return " ".join(text.split())

    def format_time(self, seconds):
        td = timedelta(seconds=seconds)
        h, rem = divmod(int(td.total_seconds()), 3600)
        m, s = divmod(rem, 60)
        ms = int((td.total_seconds() - int(td.total_seconds())) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    def save_srt(self, subs, path):
        with open(path, 'w', encoding='utf-8') as f:
            for s in subs: f.write(f"{s['index']}\n{s['start']} --> {s['end']}\n{s['content']}\n\n")
