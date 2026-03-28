# 🖥️ Telegram Video Subtitle OCR Bot

Bot Telegram ini dapat mengekstrak subtitle hardsub dari video menggunakan OCR (Optical Character Recognition). Video akan di-proses per-frame, di-crop di bagian posisi subtitle, dan di-ekstrak menjadi teks SRT.

## Teknologi yang Digunakan
- **[python-telegram-bot](https://python-telegram-bot.org/)**: Framework bot Telegram.
- **[pytesseract](https://github.com/madmaze/pytesseract)**: Wrapper untuk Tesseract OCR Engine.
- **[opencv-python](https://opencv.org/)**: Untuk manipulasi frame video dan preprocessing image.
- **FFmpeg**: Untuk pembacaan video stream.

## Persyaratan Sistem

1. **Python 3.8+**
2. **Tesseract OCR**: 
   - Windows: Download [Tesseract Installer](https://github.com/UB-Mannheim/tesseract/wiki).
   - Pastikan path Tesseract ada di Environment Variable (PATH) atau atur path manual di `ocr_engine.py`.
3. **FFmpeg**: 
   - Pastikan FFmpeg terinstall di sistem dan bisa diakses via terminal.

## Cara Instalasi

1. Clone atau download proyek ini.
2. Install dependensi lewat terminal:
   ```bash
   pip install -r requirements.txt
   ```
3. Edit `bot.py` dan masukkan **API TOKEN** dari [@BotFather](https://t.me/BotFather).
4. (Opsional) Jika Tesseract tidak di PATH, buka `ocr_engine.py` dan uncomment baris:
   ```python
   # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

## Cara Menjalankan

Jalankan bot dengan perintah:
```bash
python bot.py
```

## Cara Penggunaan
1. Kirim `/start` ke bot Anda.
2. Kirim berkas video atau pesan video (hardsub).
3. Tunggu proses OCR selesai.
4. Bot akan mengirimkan file `.srt` hasil ekstraksi.

## Penting
- Kecepatan pemrosesan tergantung pada durasi video dan spesifikasi PC/Server Anda.
- Hasil OCR sangat bergantung pada kontras teks subtitle dengan latar belakang video.
- Area cropping saat ini diset di bagian bawah (75-95% tinggi video). Jika posisi subtitle berbeda, ubah parameter di `ocr_engine.py` pada fungsi `preprocess_frame`.
