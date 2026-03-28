import cv2
import os
import torch
import whisperx
import numpy as np
import re
import logging
from paddleocr import PaddleOCR
from datetime import timedelta
import ffmpeg
from deep_translator import GoogleTranslator

class ProfessionalSubtitleSystem:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Sistem Subtitle Fusion Profesional (V3 - Word-Level Accuracy):
        - WhisperX (Word-Level Timestamps) + PaddleOCR.
        - Auto-Splitting Cerdas (Gap > 0.5s).
        - Natural Timing & Compression.
        """
        self.device = device
        self.model_name = "medium"
        try:
            print(f"Loading WhisperX ({self.model_name}, device={device})...")
            self.whisper_model = whisperx.load_model(self.model_name, device, compute_type="float16" if device=="cuda" else "int8")
        except:
            self.whisper_model = whisperx.load_model("base", device)
        
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        self.translator = GoogleTranslator(source='auto', target='id')
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def shorten_id_text(self, text):
        """Menyederhanakan kalimat Indonesia agar lebih 'rapi' & pendek (Conciseness)."""
        if not text: return ""
        # Kamus penyederhanaan (Formal tapi pendek)
        replacements = {
            r"\bMengapa\b": "Kenapa",
            r"\bAnda\b": "Kamu", # Opsional, tergantung konteks, tapi 'Anda' sering kaku
            r"\btidak mengatakan saja\b": "bilang saja",
            r"\bmengatakan bahwa\b": "bilang kalau",
            r"\bseseorang saat mengemudi\b": "orang saat menyetir",
            r"\bmelihat saja\b": "lihat saja",
            r"\bakan baik-baik saja\b": "akan aman-aman saja",
            r"\bmempunyai\b": "memiliki",
            r"\badalah\s": " ", 
            r"\byang mana\b": "yang",
            r"\bsedang\b": "", # Hilangkan 'sedang' jika berlebihan
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return " ".join(text.split())

    def split_by_words(self, words, max_chars=42, max_lines=2, min_gap=0.5):
        """
        Logika pemotongan subtitle berdasarkan WORD-LEVEL TIMESTAMPS.
        """
        segments = []
        current_chunk = []
        
        for i, word in enumerate(words):
            w_text = word.get('word', '')
            w_start = word.get('start')
            w_end = word.get('end')
            
            if w_start is None or w_end is None: continue
            
            # 1. Cek Jeda (Gap > 0.5s -> New Subtitle)
            if current_chunk:
                prev_end = current_chunk[-1].get('end', 0)
                if (w_start - prev_end) > min_gap:
                    segments.append(current_chunk)
                    current_chunk = []
            
            current_chunk.append(word)
            
            # 2. Cek Panjang (Max Chars)
            chunk_text = " ".join([w.get('word', '') for w in current_chunk])
            if len(chunk_text) > (max_chars * max_lines):
                segments.append(current_chunk)
                current_chunk = []
                
        if current_chunk:
            segments.append(current_chunk)
            
        return segments

    def format_segments(self, word_segments):
        """Ubah word chunks menjadi subtitle object dengan timing profesional."""
        formatted = []
        for i, chunk in enumerate(word_segments):
            if not chunk: continue
            start = chunk[0].get('start', 0) - 0.1 # Padding awal -0.1s
            end = chunk[-1].get('end', 0) + 0.2    # Padding akhir +0.2s
            text = " ".join([w.get('word', '') for w in chunk]).strip()
            
            # Minimal durasi 1 detik
            if (end - start) < 1.0:
                end = start + 1.0
                
            formatted.append({
                'start': max(0, start),
                'end': end,
                'text': text
            })
            
        # Hindari Overlap
        for j in range(len(formatted) - 1):
            if formatted[j]['end'] > formatted[j+1]['start']:
                formatted[j]['end'] = formatted[j+1]['start'] - 0.05
                
        return formatted

    def extract_audio(self, video_path):
        output_audio = f"{video_path}.wav"
        try:
            (ffmpeg.input(video_path).output(output_audio, acodec='pcm_s16le', ac=1, ar='16k').run(quiet=True, overwrite_output=True))
            return output_audio
        except: return None

    def process_full_subtitle(self, video_path, progress_callback=None):
        audio_path = self.extract_audio(video_path)
        if not audio_path: return [], [], "unknown"

        # 1. Whisper Transcribe
        try:
            result = self.whisper_model.transcribe(audio_path, batch_size=16, 
                                                  task="transcribe", 
                                                  chunk_size=30)
            lang = result.get("language", "en")
            
            # 2. Align (Word-Level)
            model_a, metadata = whisperx.load_align_model(language_code=lang, device=self.device)
            result = whisperx.align(result.get("segments", []), model_a, metadata, audio_path, self.device, return_char_alignments=False)
            
            all_words = []
            for seg in result.get("segments", []):
                all_words.extend(seg.get("words", []))
                
        except Exception as e:
            print(f"Whisper Error: {e}")
            return [], [], "unknown"

        # 3. Smart Splitting
        word_chunks = self.split_by_words(all_words)
        formatted_segments = self.format_segments(word_chunks)
        
        orig_subs = []
        trans_subs = []
        previous_trans = ""
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        for i, seg in enumerate(formatted_segments):
            text = seg['text']
            start, end = seg['start'], seg['end']
            
            # 4. OCR correction (Optional but recommended for hardsubs)
            ocr_text = self.run_ocr_on_segment(cap, start, end, fps)
            final_orig = ocr_text if len(ocr_text) > 3 else text
            
            # 5. Translation + Shortening (Agar 'Rapih')
            try:
                # Context buffer
                context_input = f"{previous_trans}\n{final_orig}" if previous_trans else final_orig
                translated = self.translator.translate(context_input)
                if "\n" in translated: translated = translated.split("\n")[-1]
                
                # Pemangkasan kalimat agar lebih pendek/rapih
                final_trans = self.shorten_id_text(translated)
                previous_trans = final_orig
            except:
                final_trans = ""

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

    def run_ocr_on_segment(self, cap, start, end, fps):
        try:
            mid = (start + end) / 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid * fps))
            ret, frame = cap.read()
            if not ret: return ""
            crop = frame[int(frame.shape[0] * 0.6):, :]
            res = self.ocr.ocr(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cls=True)
            return " ".join([l[1][0] for l in res[0] if l[1][1] > 0.5]) if res and res[0] else ""
        except: return ""

    def wrap_text(self, text, max_chars=42):
        if not text: return ""
        if len(text) <= max_chars: return text
        mid = len(text) // 2
        idx = text.rfind(" ", 0, mid + 10)
        if idx == -1: idx = mid
        return f"{text[:idx].strip()}\n{text[idx:].strip()}"

    def format_time(self, seconds):
        td = timedelta(seconds=seconds)
        h, rem = divmod(int(td.total_seconds()), 3600)
        m, s = divmod(rem, 60)
        ms = int((td.total_seconds() - int(td.total_seconds())) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    def save_srt(self, subs, path):
        with open(path, 'w', encoding='utf-8') as f:
            for s in subs:
                f.write(f"{s['index']}\n{s['start']} --> {s['end']}\n{s['content']}\n\n")
