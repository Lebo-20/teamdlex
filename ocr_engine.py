import cv2
import os
import torch
import whisperx
import numpy as np
import re
from paddleocr import PaddleOCR
from datetime import timedelta
import ffmpeg
from deep_translator import GoogleTranslator

class ProfessionalSubtitleSystem:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Sistem Subtitle Dinamis (Ultimate V6 - TikTok/Shorts Style):
        - WhisperX (Word-Level Timing).
        - Segmen Pendek (1-5 Kata).
        - Single Line Format (Rapi & Modern).
        - Sinkronisasi Audio Presisi.
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

    def split_into_dynamic_chunks(self, words, max_words=5, gap_threshold=0.35):
        """
        Memecah transkripsi menjadi potongan pendek bergaya TikTok/Shorts.
        - Maksimal 5 kata.
        - Potong jika ada jeda > 0.35 detik.
        """
        chunks = []
        current_chunk = []
        
        for word in words:
            w_start = word.get('start')
            w_end = word.get('end')
            w_text = word.get('word', '').strip()
            
            if w_start is None: continue
            
            should_split = False
            if current_chunk:
                # 1. Cek Kapasitas Kata (Max 5)
                if len(current_chunk) >= max_words: should_split = True
                
                # 2. Cek Jeda Bicara (Berdasarkan ritme audio)
                gap = w_start - current_chunk[-1].get('end', 0)
                if gap > gap_threshold: should_split = True
                
                # 3. Cek Tanda Baca (., ?, !)
                prev_word = current_chunk[-1].get('word', '')
                if any(p in prev_word for p in [".", "?", "!", "。", "？", "！"]): should_split = True

            if should_split and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                
            current_chunk.append(word)
            
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def format_dynamic_timing(self, chunks):
        """Memformat timing subtitle agar dinamis & sinkron."""
        formatted = []
        for i, chunk in enumerate(chunks):
            start = max(0, chunk[0]['start'] - 0.05) # Delay minimal agar pas
            end = chunk[-1]['end'] + 0.15           # Sedikit perpanjangan agar terbaca
            text = " ".join([w['word'] for w in chunk]).strip()
            
            # Gabungkan jika 1 kata terlalu pendek durasinya
            # Durasi minimal 0.8 detik (Standar baca cepat)
            if (end - start) < 0.8:
                end = start + 0.8
                
            formatted.append({'start': start, 'end': end, 'text': text})
            
        # De-overlap
        for j in range(len(formatted) - 1):
            if formatted[j]['end'] > formatted[j+1]['start']:
                formatted[j]['end'] = formatted[j+1]['start'] - 0.02
        return formatted

    def process_full_subtitle(self, video_path, progress_callback=None):
        """Pipeline Utama V6 - Dynamic Short Subtitles."""
        audio_path = f"{video_path}.wav"
        try:
            (ffmpeg.input(video_path).output(audio_path, acodec='pcm_s16le', ac=1, ar='16k').run(quiet=True, overwrite_output=True))
        except: return [], [], "unknown"

        # 1. Transcribe & Align (Word-Level)
        try:
            result = self.whisper_model.transcribe(audio_path, batch_size=16)
            lang = result.get("language", "en")
            model_a, metadata = whisperx.load_align_model(language_code=lang, device=self.device)
            result = whisperx.align(result.get("segments", []), model_a, metadata, audio_path, self.device, return_char_alignments=False)
            
            all_words = []
            for seg in result.get("segments", []):
                all_words.extend(seg.get("words", []))
        except:
            if os.path.exists(audio_path): os.remove(audio_path)
            return [], [], "unknown"

        # 2. Pecah Jadi Potongan Pendek (TikTok Style)
        dynamic_chunks = self.split_into_dynamic_chunks(all_words)
        
        # 3. Format Waktu
        pro_segments = self.format_dynamic_timing(dynamic_chunks)
        
        orig_subs = []
        trans_subs = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        for i, seg in enumerate(pro_segments):
            start, end = seg['start'], seg['end']
            audio_text = seg['text']
            
            # 4. OCR correction (Visual)
            ocr_text = self.run_ocr_on_segment(cap, start, end, fps)
            final_orig = ocr_text if (ocr_text and len(ocr_text.split()) < 6) else audio_text
            final_orig = " ".join(final_orig.split())
            
            if not final_orig: continue

            # 5. Translation Spontan & Modern
            try:
                final_trans = self.translator.translate(final_orig)
                # Bersihkan karakter aneh
                final_trans = re.sub(r'[~|`\\“”„»«]', '', final_trans).strip()
            except:
                final_trans = final_orig

            # 6. Save (SINGLE LINE)
            orig_subs.append({
                'index': i + 1,
                'start': self.format_time(start),
                'end': self.format_time(end),
                'content': final_orig # Single line, no wrap
            })
            
            trans_subs.append({
                'index': i + 1,
                'start': self.format_time(start),
                'end': self.format_time(end),
                'content': final_trans # Single line, no wrap
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
            # Filter area subtitle (Sedikit lebih ke atas agar tidak tertutup objek bawah)
            crop = frame[int(frame.shape[0] * 0.55):int(frame.shape[0] * 0.9), :]
            res = self.ocr.ocr(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cls=True)
            return " ".join([l[1][0] for l in res[0] if l[1][1] > 0.5]) if res and res[0] else ""
        except: return ""

    def format_time(self, seconds):
        td = timedelta(seconds=seconds)
        h, rem = divmod(int(td.total_seconds()), 3600)
        m, s = divmod(rem, 60)
        ms = int((td.total_seconds() - int(td.total_seconds())) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    def save_srt(self, subs, path):
        with open(path, 'w', encoding='utf-8') as f:
            for s in subs: f.write(f"{s['index']}\n{s['start']} --> {s['end']}\n{s['content']}\n\n")
