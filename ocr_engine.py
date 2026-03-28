import cv2
import os
import torch
import whisperx
import numpy as np
import re
import difflib
from paddleocr import PaddleOCR
from datetime import timedelta
import ffmpeg
from deep_translator import GoogleTranslator
from PIL import Image
import logging

# Disable unnecessary logging
logging.getLogger("ppocr").setLevel(logging.ERROR)

class ProfessionalSubtitleSystem:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        AI Subtitle System V8: Dual-Engine Architecture
        - Mode 1: High-Precision Reconstruction (Perfect Sync + Context AI)
        - Mode 2: Fast Extraction (OCR Optimized + Speed)
        """
        self.device = device
        self.model_name = "medium"
        self.compute_type = "float16" if device == "cuda" else "int8"
        
        print(f"🚀 Initializing AI System on {device.upper()}...")
        
        try:
            self.whisper_model = whisperx.load_model(
                self.model_name, device, compute_type=self.compute_type
            )
        except Exception as e:
            print(f"⚠️ Whisper load failed: {e}")
            self.whisper_model = whisperx.load_model("base", device)
        
        # Multilingual OCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
        self.translator = GoogleTranslator(source='auto', target='id')
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def preprocess_frame(self, frame, mode="precision"):
        """Image Enhancement for OCR."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if mode == "fast":
                return gray # Skip heavy filters for speed
                
            denoised = cv2.bilateralFilter(gray, 7, 50, 50)
            enhanced = self.clahe.apply(denoised)
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            if np.mean(thresh) > 127: thresh = cv2.bitwise_not(thresh)
            return thresh
        except:
            return frame

    def resolve_conflicts(self, ocr_text, audio_text):
        """Dual Validation: Resolves differences between OCR and Audio."""
        if not ocr_text: return audio_text
        if not audio_text: return ocr_text
        
        # If texts are very similar, prioritize OCR (visual truth for hardsubs)
        similarity = difflib.SequenceMatcher(None, ocr_text, audio_text).ratio()
        if similarity > 0.7:
            return ocr_text
            
        # If completely different, OCR might be noise or watermark
        # or audio might be music/background.
        # Heuristic: Subtitles usually have 1-12 words.
        if len(ocr_text.split()) > 15: return audio_text # Likely noise
        
        return ocr_text # Trust the eyes for hardsubs

    def clean_reconstruction(self, text):
        """Zero Noise Policy: Strict artifacts removal."""
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\'\"-ー]', '', text)
        text = re.sub(r'([.,!?])\1+', r'\1', text) # Remove duplicates
        return " ".join(text.split()).strip()

    def run_ocr_on_segment(self, cap, start, end, fps, mode="precision"):
        """Visual detection with mode-specific sampling."""
        results = []
        # Precision: 5 samples, Fast: 2 samples
        count = 5 if mode == "precision" else 2
        timestamps = np.linspace(start + 0.05, end - 0.05, count)
        
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
            ret, frame = cap.read()
            if not ret: continue
            
            # Masking (Ignore watermarks/UI: bottom 25% focus)
            crop = frame[int(h_orig * 0.70):int(h_orig * 0.95), :]
            
            processed = self.preprocess_frame(crop, mode)
            res = self.ocr.ocr(processed, cls=True)
            
            if res and res[0]:
                text = " ".join([l[1][0] for l in res[0] if l[1][1] > 0.65])
                if text: results.append(text)

        if not results: return ""
        return max(results, key=len)

    def process_full_subtitle(self, video_path, mode="precision", progress_callback=None):
        """Main Pipeline: Supports Reconstruction (High-Res) and Extraction (Fast)."""
        audio_path = f"{video_path}.wav"
        if progress_callback: progress_callback(0)
        
        # 1. Extraction (Always needed for timing in Precision mode)
        try:
            (ffmpeg.input(video_path).output(audio_path, acodec='pcm_s16le', ac=1, ar='16k').run(quiet=True, overwrite_output=True))
        except: pass

        segments = []
        lang = "en"
        
        if mode == "precision":
            # PRECISISON MODE: WhisperX Word-Level + Multi-Sampling
            try:
                result = self.whisper_model.transcribe(audio_path, batch_size=16)
                lang = result.get("language", "en")
                model_a, metadata = whisperx.load_align_model(language_code=lang, device=self.device)
                result = whisperx.align(result.get("segments", []), model_a, metadata, audio_path, self.device)
                
                all_words = []
                for seg in result.get("segments", []):
                    all_words.extend(seg.get("words", []))
                
                # Split based on speech rhythm (Instruction 5)
                chunks = self.split_into_dynamic_chunks(all_words, max_words=8, gap_threshold=0.6)
                segments = self.format_dynamic_timing(chunks)
            except: mode = "fast" # Fallback if audio AI fails

        if mode == "fast" or not segments:
            # FAST MODE: Frame-based timing estimation
            # Use Whisper segments without word-alignment for speed
            try:
                result = self.whisper_model.transcribe(audio_path, batch_size=32)
                segments = [{'start': s['start'], 'end': s['end'], 'text': s['text']} for s in result['segments']]
            except: return [], [], "unknown"

        orig_subs, trans_subs = [], []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = len(segments)

        for i, seg in enumerate(segments):
            start, end = seg['start'], seg['end']
            audio_text = seg['text']
            
            visual_text = self.run_ocr_on_segment(cap, start, end, fps, mode)
            
            if mode == "precision":
                # STRICT RULES for Precision
                final_text = self.resolve_conflicts(visual_text, audio_text)
                final_text = self.clean_reconstruction(final_text)
            else:
                # FAST MODE: OCR priority
                final_text = visual_text if visual_text else audio_text
                final_text = " ".join(final_text.split())

            if not final_text: continue

            # Subtitle Intelligence: Ensure max 2 lines (simple wrap)
            # (Note: actually SRT viewers handle wrap, but we keep it tidy)
            
            try:
                translated = self.translator.translate(final_text)
            except: translated = final_text

            orig_subs.append({'index': i + 1, 'start': self.format_time(start), 'end': self.format_time(end), 'content': final_text})
            trans_subs.append({'index': i + 1, 'start': self.format_time(start), 'end': self.format_time(end), 'content': translated})
            
            if progress_callback: progress_callback((i + 1) / total * 100)

        cap.release()
        if os.path.exists(audio_path): os.remove(audio_path)
        return orig_subs, trans_subs, lang

    def split_into_dynamic_chunks(self, words, max_words=8, gap_threshold=0.6):
        chunks, current = [], []
        for w in words:
            if 'start' not in w or 'end' not in w: continue
            if current:
                gap = w['start'] - current[-1]['end']
                if len(current) >= max_words or gap > gap_threshold:
                    chunks.append(current)
                    current = []
            current.append(w)
        if current: chunks.append(current)
        return chunks

    def format_dynamic_timing(self, chunks):
        return [{'start': c[0]['start'], 'end': c[-1]['end'], 'text': " ".join([w['word'] for w in c])} for c in chunks]

    def format_time(self, seconds):
        td = timedelta(seconds=max(0, seconds))
        h, m, s = str(td).split(':')
        if '.' in s:
            s, ms = s.split('.')
            ms = ms[:3].ljust(3, '0')
        else: ms = "000"
        return f"{int(h):02}:{int(m):02}:{int(s):02},{ms}"

    def save_srt(self, subs, path):
        with open(path, 'w', encoding='utf-8') as f:
            for s in subs:
                f.write(f"{s['index']}\n{s['start']} --> {s['end']}\n{s['content']}\n\n")
