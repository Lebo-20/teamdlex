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
        Sistem Subtitle Fusion Profesional (Ultimate V5 - Sentence Flow):
        - WhisperX (Audio Baseline + Alignment).
        - Intelligent Sentence-based Merging.
        - OCR Correction & Noise Filtering.
        - Natural Timing (Padding & De-overlap).
        """
        self.device = device
        self.model_name = "medium"
        
        try:
            print(f"Loading WhisperX ({self.model_name}, device={device})...")
            self.whisper_model = whisperx.load_model(self.model_name, device, 
                                                    compute_type="float16" if device=="cuda" else "int8")
        except:
            self.whisper_model = whisperx.load_model("base", device)
        
        print("Loading PaddleOCR...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
        self.translator = GoogleTranslator(source='auto', target='id')
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def filter_ocr_noise(self, text):
        """Menghapus teks watermark/UI yang sering muncul di OCR."""
        noise_keywords = ["快手搜索", "快手", "抖音", "douyin", "kuaishou", "www.", ".com", "ID:"]
        for nw in noise_keywords:
            if nw.lower() in text.lower(): return ""
        # Bersihkan karakter aneh
        text = re.sub(r'[~|`\\“”„»«]', '', text)
        return text.strip()

    def merge_segments_intelligently(self, segments, max_chars=80, gap_threshold=0.5):
        """
        Menggabungkan segmen-segmen pendek menjadi kalimat utuh yang rapi.
        Aturan: Gabung jika gap kecil dan bukan akhir kalimat.
        """
        merged = []
        if not segments: return []
        
        current = segments[0]
        
        for next_seg in segments[1:]:
            # Hilangkan durasi 0
            if next_seg['start'] == next_seg['end']: continue
            
            # Kriteria Penggabungan:
            # 1. Gap antar segmen < 0.5 detik
            gap = next_seg['start'] - current['end']
            # 2. Panjang teks gabungan masih masuk akal (max 2 baris standar)
            combined_text = f"{current['text']} {next_seg['text']}"
            # 3. Apakah segmen sebelumnya berakhir dengan tanda baca penutup?
            ends_sentence = any(current['text'].strip().endswith(p) for p in [".", "?", "!", "。", "？", "！"])
            
            if gap < gap_threshold and len(combined_text) < max_chars and not ends_sentence:
                # GABUNGKAN
                current['text'] = combined_text
                current['end'] = next_seg['end']
            else:
                # SIMPAN & MULAI BARU
                merged.append(current)
                current = next_seg
                
        merged.append(current)
        return merged

    def format_professional_timing(self, segments):
        """Memastikan durasi minimal dan padding waktu."""
        for seg in segments:
            seg['start'] = max(0, seg['start'] - 0.1) # -0.1s
            seg['end'] = seg['end'] + 0.2             # +0.2s
            # Minimal durasi 1.2 detik agar mata tidak lelah
            if (seg['end'] - seg['start']) < 1.2:
                seg['end'] = seg['start'] + 1.2
        return segments

    def process_full_subtitle(self, video_path, progress_callback=None):
        """Pipeline Utama V5 - High Quality Sentence Flow."""
        audio_path = f"{video_path}.wav"
        try:
            (ffmpeg.input(video_path).output(audio_path, acodec='pcm_s16le', ac=1, ar='16k').run(quiet=True, overwrite_output=True))
        except: return [], [], "unknown"

        # 1. Transcribe & Align
        try:
            result = self.whisper_model.transcribe(audio_path, batch_size=16)
            lang = result.get("language", "en")
            model_a, metadata = whisperx.load_align_model(language_code=lang, device=self.device)
            result = whisperx.align(result.get("segments", []), model_a, metadata, audio_path, self.device, return_char_alignments=False)
            raw_segments = result.get("segments", [])
        except:
            if os.path.exists(audio_path): os.remove(audio_path)
            return [], [], "unknown"

        # 2. Gabungkan Kalimat Pendek (Merge)
        merged_segments = self.merge_segments_intelligently(raw_segments)
        
        # 3. Format Timing (Padding & Duration)
        pro_segments = self.format_professional_timing(merged_segments)
        
        orig_subs = []
        trans_subs = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        for i, seg in enumerate(pro_segments):
            start, end = seg['start'], seg['end']
            audio_text = seg['text'].strip()
            
            # 4. OCR correction (Visual)
            ocr_text = self.run_ocr_on_segment(cap, start, end, fps)
            ocr_text = self.filter_ocr_noise(ocr_text)
            
            # FUSION: Audio sebagai struktur, OCR sebagai koreksi kata layar
            final_orig = ocr_text if len(ocr_text) > 4 else audio_text
            final_orig = " ".join(final_orig.split())
            
            if not final_orig: continue

            # 5. Translation Professional
            try:
                translated = self.translator.translate(final_orig)
                # Sederhanakan bahasa agar 'rapih'
                final_trans = self.polish_natural(translated)
            except:
                final_trans = final_orig

            # 6. Save SRT Format
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
            
            if progress_callback: progress_callback((i + 1) / len(pro_segments) * 100)

        cap.release()
        if os.path.exists(audio_path): os.remove(audio_path)
        return orig_subs, trans_subs, lang

    def run_ocr_on_segment(self, cap, start, end, fps):
        try:
            mid = (start + end) / 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid * fps))
            ret, frame = cap.read()
            if not ret: return ""
            # Filter area subtitle (65% bawah)
            crop = frame[int(frame.shape[0] * 0.65):, :]
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

    def polish_natural(self, text):
        repl = {
            r"\bMengapa\b": "Kenapa",
            r"\btidak mengatakan\b": "tidak bilang",
            r"\bseseorang saat mengemudi\b": "orang saat menyetir",
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
