import cv2
import os
import torch
import whisperx
import numpy as np
import re
from paddleocr import PaddleOCR
from Levenshtein import ratio as levenshtein_ratio
from datetime import timedelta
from pydub import AudioSegment
import ffmpeg
from deep_translator import GoogleTranslator

class ProfessionalSubtitleSystem:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Sistem Subtitle Profesional Ultimate:
        - WhisperX (Audio Alignment) + PaddleOCR (Hardsub Fusion).
        - Auto-Translation ke Indonesia Formal.
        - Dual-Subtitle Support.
        """
        self.device = device
        print(f"Loading WhisperX (device={device})...")
        self.whisper_model = whisperx.load_model("medium", device, compute_type="float16" if device=="cuda" else "int8")
        
        print("Loading PaddleOCR...")
        # Hilangkan use_gpu agar PaddleOCR mendeteksi perangkat secara otomatis (lebih kompatibel)
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        
        # Inisialisasi Translator: Multi-source -> Indonesian
        self.translator = GoogleTranslator(source='auto', target='id')
        
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def translate_contextual(self, text, previous_context=""):
        """Menerjemahkan teks dengan mempertimbangkan konteks kalimat sebelumnya."""
        if not text or len(text) < 2: return ""
        try:
            # Gunakan gabungan konteks untuk membantu mesin terjemahan memahami makna
            # Misalnya: "Hello" bisa jadi "Halo" atau "Salam". Dengan konteks, lebih akurat.
            full_input = f"{previous_context}\n{text}" if previous_context else text
            
            # Terjemahkan blok teks sekaligus
            translated = self.translator.translate(full_input)
            
            # Jika ada konteks, ambil hanya bagian terakhir (teks yang dimaksud)
            if "\n" in translated:
                translated = translated.split("\n")[-1]
                
            # --- Polishing Naturalness (Penghalusan Bahasa) ---
            # Menghapus beberapa literal translation yang kaku
            translated = translated.replace("Apa yang sedang terjadi?", "Apa yang terjadi?") # Lebih natural
            translated = translated.replace("mempunyai", "memiliki") # Lebih formal/elegan
            translated = translated.replace("saya adalah", "saya") # Seringkali redundancy
            
            return translated.strip()
        except:
            return text

    def wrap_text(self, text, max_chars=42):
        """Membagi teks menjadi maksimal 2 baris agar enak dibaca."""
        if len(text) <= max_chars: return text
        
        # Cari spasi terdekat dari tengah sebagai titik potong
        mid = len(text) // 2
        
        # Coba cari spasi ke arah kiri dan kanan dari titik tengah
        left_space = text.rfind(" ", 0, mid + 10)
        right_space = text.find(" ", mid - 5)
        
        # Prioritaskan titik potong yang paling dekat dengan tengah
        if left_space != -1 and (mid - left_space) < 5:
            split_idx = left_space
        elif right_space != -1:
            split_idx = right_space
        else:
            split_idx = mid
            
        line1 = text[:split_idx].strip()
        line2 = text[split_idx:].strip()
        
        # Pastikan tidak lebih dari 2 baris (jika masih terlalu panjang, potong lagi)
        if len(line2) > max_chars:
            line2 = line2[:max_chars-3] + "..."
            
        return f"{line1}\n{line2}"

    def extract_audio(self, video_path, output_audio="temp_audio.wav"):
        """Ekstrak audio WAV 16kHz mono."""
        if os.path.exists(output_audio): os.remove(output_audio)
        (
            ffmpeg
            .input(video_path)
            .output(output_audio, acodec='pcm_s16le', ac=1, ar='16k')
            .run(quiet=True)
        )
        return output_audio

    def align_audio(self, video_path, audio_path):
        """Transkripsi & Alignment menggunakan WhisperX."""
        result = self.whisper_model.transcribe(audio_path, batch_size=16)
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio_path, self.device, return_char_alignments=False)
        return result["segments"], result["language"]

    def preprocess_frame(self, frame):
        h, w = frame.shape[:2]
        crop = frame[int(h * 0.6):h, :]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)
        return cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    def run_ocr_on_segment(self, cap, start_sec, end_sec, fps):
        mid_time = (start_sec + end_sec) / 2
        frame_no = int(mid_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret: return ""
        
        processed = self.preprocess_frame(frame)
        result = self.ocr.ocr(processed, cls=True)
        
        texts = []
        if result and result[0]:
            for line in result[0]:
                if line[1][1] > 0.5:
                    texts.append(line[1][0])
        return " ".join(texts)

    def process_full_subtitle(self, video_path, dual_sub=True, progress_callback=None):
        """Pipeline Fusion Audio + OCR + Translation."""
        audio_path = self.extract_audio(video_path)
        segments, lang = self.align_audio(video_path, audio_path)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_segs = len(segments)
        final_subs = []
        previous_text = ""
        
        for i, seg in enumerate(segments):
            start, end = seg['start'], seg['end']
            audio_text = seg['text'].strip()
            
            # OCR correction
            ocr_text = self.run_ocr_on_segment(cap, start, end, fps)
            original_text = ocr_text if len(ocr_text) > 3 else audio_text
            original_text = " ".join(original_text.split())
            
            # 1. Terjemahkan secara kontekstual (mengacu pada kalimat sebelumnya)
            translated_text = self.translate_contextual(original_text, previous_context=previous_text)
            previous_text = original_text # Simpan untuk konteks berikutnya
            
            # 2. Format hasil
            orig_wrapped = self.wrap_text(original_text)
            trans_wrapped = self.wrap_text(translated_text)
            
            if dual_sub:
                # Ambil baris pertama original dan baris pertama terjemahan
                # (atau biarkan wrap logic-nya)
                content = f"{orig_wrapped}\n{trans_wrapped}"
            else:
                content = trans_wrapped

            final_subs.append({
                'index': i + 1,
                'start': self.format_time(start),
                'end': self.format_time(end),
                'content': content
            })
            
            if progress_callback:
                progress_callback((i + 1) / total_segs * 100)
        
        cap.release()
        if os.path.exists(audio_path): os.remove(audio_path)
        return final_subs

    def format_time(self, seconds):
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds_part = divmod(remainder, 60)
        milliseconds = int((td.total_seconds() - total_seconds) * 1000)
        return f"{hours:02}:{minutes:02}:{seconds_part:02},{milliseconds:03}"

    def save_srt(self, subtitles, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            for sub in subtitles:
                f.write(f"{sub['index']}\n")
                f.write(f"{sub['start']} --> {sub['end']}\n")
                f.write(f"{sub['content']}\n\n")
        return output_path
