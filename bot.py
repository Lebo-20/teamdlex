import os
import asyncio
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from ocr_engine import ProfessionalSubtitleSystem
from dotenv import load_dotenv

# Load kredensial dari file .env (Sangat disarankan untuk GitHub upload)
load_dotenv()

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

API_TOKEN = os.getenv('BOT_TOKEN')
ADMIN_IDS = [int(i) for i in os.getenv('ADMIN_IDS', '').split() if i]

def get_progress_bar(percentage):
    """Visual progress bar dengan emoji (TikTok style)."""
    filled = int(percentage // 10)
    bar = "██" * filled + "░░" * (10 - filled)
    return f"[{bar}] {int(percentage)}%"

async def edit_status(status_msg, stage, percentage):
    """Update pesan progres secara aman (edit_text)."""
    try:
        bar = get_progress_bar(percentage)
        text = f"⚙️ {stage}...\nProgres: {bar}"
        await status_msg.edit_text(text)
    except: pass # Abaikan jika pesan sama atau error rate limit

def cleanup_temp():
    if os.path.exists('temp'):
        for f in os.listdir('temp'):
            try: os.remove(os.path.join('temp', f))
            except: pass

class SubtitleBot:
    def __init__(self, token):
        cleanup_temp()
        self.application = ApplicationBuilder().token(token).build()
        self.extractor = ProfessionalSubtitleSystem()
        
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("update", self.update_bot))
        self.application.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO, self.handle_video))
        
    async def is_admin(self, user_id):
        return user_id in ADMIN_IDS

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self.is_admin(update.effective_user.id): return
        await update.message.reply_text("👋 Halo Admin! Bot Subtitle Dinamis (V6) Siap Digunakan.")

    async def handle_video(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self.is_admin(update.effective_user.id): return

        message = update.message
        video_file = message.video or message.document
        if not video_file: return
            
        status_msg = await message.reply_text("📥 Sedang mengunduh video...")
        
        video_path = f"temp/{video_file.file_id}.mp4"
        srt_path_orig = f"temp/{video_file.file_id}_original.srt"
        srt_path_trans = f"temp/{video_file.file_id}_id.srt"
        
        try:
            # 1. DOWNLOAD (10%)
            file = await context.bot.get_file(video_file.file_id)
            await file.download_to_drive(video_path)
            await edit_status(status_msg, "Download Video", 10)
            
            # 2. INISIALISASI (20%)
            await edit_status(status_msg, "Inisialisasi Sistem AI", 20)
            
            # Progress callback untuk pemrosesan (Mapping 20% -> 95%)
            last_p = [20]
            loop = asyncio.get_event_loop()
            
            def sync_progress_callback(p_internal):
                # Map 0-100 internal ke 20-95 bot progress
                current_p = int(20 + (p_internal * 0.75)) 
                if current_p > last_p[0]:
                    last_p[0] = current_p
                    asyncio.run_coroutine_threadsafe(
                        edit_status(status_msg, "Sedang Memproses AI (Whisper + OCR)", current_p),
                        loop
                    )
            
            # 3. PROSES UTAMA (20% - 95%)
            from functools import partial
            orig_subs, trans_subs, detected_lang = await loop.run_in_executor(
                None, 
                partial(self.extractor.process_full_subtitle, video_path, progress_callback=sync_progress_callback)
            )
            
            if not orig_subs:
                await status_msg.edit_text("⚠️ Gagal mengekstrak subtitle.")
                return
            
            # 4. GENERATE & SEND (95% - 100%)
            await edit_status(status_msg, "Menyusun File Subtitle", 95)
            self.extractor.save_srt(orig_subs, srt_path_orig)
            self.extractor.save_srt(trans_subs, srt_path_trans)
            
            await edit_status(status_msg, "Selesai! Mengirim Ke Anda", 100)
            await message.reply_document(document=srt_path_orig, filename=f"SUB_ORIGINAL_{video_file.file_id[:5]}.srt")
            await message.reply_document(document=srt_path_trans, filename=f"SUB_INDONESIA_{video_file.file_id[:5]}.srt")
            await status_msg.edit_text("✅ Selesai! Subtitle berhasil dibuat.")
            
        except Exception as e:
            logging.error(f"Error: {e}")
            await status_msg.edit_text(f"❌ Terjadi kesalahan: {str(e)}")
            
        finally:
            for p in [video_path, srt_path_orig, srt_path_trans]:
                if os.path.exists(p):
                    try: os.remove(p)
                    except: pass
            if os.path.exists(f"{video_path}.wav"):
                try: os.remove(f"{video_path}.wav")
                except: pass

    async def update_bot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self.is_admin(update.effective_user.id): return
        cleanup_temp()
        status_msg = await update.message.reply_text("🔄 Memeriksa pembaruan...")
        
        try:
            import subprocess
            import sys
            
            # 1. Jalankan git pull
            print("Menjalankan git pull...")
            pull_result = subprocess.run(['git', 'pull', 'origin', 'main'], capture_output=True, text=True)
            
            if "Already up to date." in pull_result.stdout:
                await status_msg.edit_text("✅ Bot sudah menggunakan versi terbaru (Up to date).")
                return
                
            await status_msg.edit_text(f"📦 Pembaruan ditemukan!\n\n`{pull_result.stdout[:200]}`\n\n⚙️ Melakukan restart...")
            
            # 2. Restart Proses
            # Ini akan mengakhiri proses saat ini dan menjalankan perintah python yang sama lagi
            print("Melakukan restart sistem...")
            args = [sys.executable] + sys.argv
            os.execv(sys.executable, args)
            
        except Exception as e:
            logging.error(f"Update failed: {e}")
            await status_msg.edit_text(f"❌ Gagal melakukan update: {str(e)}")

    def run(self):
        self.application.run_polling()

if __name__ == '__main__':
    if not API_TOKEN:
        print("PERINGATAN: BOT_TOKEN tidak ditemukan di file .env!")
    else:
        bot = SubtitleBot(API_TOKEN)
        bot.run()
