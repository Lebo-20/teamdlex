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

# Pastikan folder temp ada
os.makedirs('temp', exist_ok=True)

class SubtitleBot:
    def __init__(self, token):
        self.application = ApplicationBuilder().token(token).build()
        # Inisialisasi mesin subtitle profesional
        self.extractor = ProfessionalSubtitleSystem()
        
        # Register handlers
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("update", self.update_bot))
        self.application.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO, self.handle_video))
        
    async def is_admin(self, user_id):
        return user_id in ADMIN_IDS

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self.is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ Maaf, Anda tidak memiliki akses ke bot ini.")
            return

        await update.message.reply_text(
            "👋 Halo Admin! Kirimkan video (Inggris/Mandarin) yang ingin diterjemahkan.\n"
            "Saya akan mengekstrak hardsub/audio dan menghasilkan subtitle Dual-Language (Indo Formal)."
        )

    async def handle_video(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self.is_admin(update.effective_user.id):
            return

        message = update.message
        video_file = message.video or message.document
        
        if not video_file:
            return
            
        status_message = await message.reply_text("📥 Sedang mengunduh video...")
        
        # Download video
        file = await context.bot.get_file(video_file.file_id)
        video_path = f"temp/{video_file.file_id}.mp4"
        await file.download_to_drive(video_path)
        
        await status_message.edit_text("⚙️ Sedang melakukan Transkripsi & Terjemahan... Ini butuh beberapa menit.")
        
        try:
            srt_path = f"temp/{video_file.file_id}_id.srt"
            
            # Progress callback logic
            last_progress = [0]
            loop = asyncio.get_event_loop()
            
            def sync_progress_callback(percentage):
                if percentage - last_progress[0] >= 10:
                    last_progress[0] = percentage
                    asyncio.run_coroutine_threadsafe(
                        status_message.edit_text(f"⚙️ Memproses... {int(percentage)}%"),
                        loop
                    )
            
            # Ekstrak & Terjemahkan
            from functools import partial
            subtitles = await loop.run_in_executor(
                None, 
                partial(self.extractor.process_full_subtitle, video_path, dual_sub=True, progress_callback=sync_progress_callback)
            )
            
            if not subtitles:
                await status_message.edit_text("⚠️ Tidak ditemukan subtitle/audio yang terbaca.")
                return
                
            self.extractor.save_srt(subtitles, srt_path)
            
            # Kirim balik dokumen
            await status_message.edit_text("✅ Selesai! Mengirim file Subtitle Dual-Sub...")
            await message.reply_document(
                document=srt_path,
                filename=f"subtitle_{video_file.file_id[:5]}_ID.srt",
                caption="Berikut file dual-subtitle (Original & Indo Formal)."
            )
            
            # Hapus file SRT setelah dikirim
            if os.path.exists(srt_path):
                os.remove(srt_path)
            
        except Exception as e:
            logging.error(f"Error processing video: {e}")
            await status_message.edit_text(f"❌ Terjadi kesalahan: {str(e)}")
            
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)

    async def update_bot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler untuk perintah /update - Melakukan git pull dan restart."""
        if not await self.is_admin(update.effective_user.id):
            return

        status_msg = await update.message.reply_text("🔄 Sedang memeriksa pembaruan di GitHub...")
        
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
