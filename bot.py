import os
import asyncio
import logging
import time
from telegram import Update, constants, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from ocr_engine import ProfessionalSubtitleSystem
from dotenv import load_dotenv

# Load credentials
load_dotenv()

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

API_TOKEN = os.getenv('BOT_TOKEN')
ADMIN_IDS = [int(i) for i in os.getenv('ADMIN_IDS', '').split() if i]
MAX_CONCURRENT_JOBS = 2 # Change based on your CPU/GPU specs

# Shared instance with semaphore for concurrency control
class BotManager:
    # Semaphor used for limiting heavy AI processes
    process_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
    user_modes = {}

def get_progress_bar(percentage):
    """Visual progress bar with premium aesthetics."""
    filled = int(percentage // 10)
    bar = "🟢" * filled + "⚪" * (10 - filled)
    return f"{bar} {int(percentage)}%"

async def edit_status(status_msg, stage, percentage, mode_text, queue_pos=None):
    """Update progress message with safe throttle."""
    try:
        bar = get_progress_bar(percentage)
        queue_text = f"⏳ **Queue Position:** `{queue_pos}`\n" if queue_pos else ""
        text = (
            f"🎬 **ULTIMATE SUBTITLE AI V8**\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🛠 **Mode:** `{mode_text.upper()}`\n"
            f"{queue_text}"
            f"🚀 **Stage:** `{stage}`\n"
            f"📊 **Progress:** {bar}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚡ _Multi-User Engine Active_"
        )
        await status_msg.edit_text(text, parse_mode=constants.ParseMode.MARKDOWN)
    except: pass

def cleanup_temp():
    if not os.path.exists('temp'):
        os.makedirs('temp', exist_ok=True)
        return
    for f in os.listdir('temp'):
        try: os.remove(os.path.join('temp', f))
        except: pass

class SubtitleBot:
    def __init__(self, token):
        os.makedirs('temp', exist_ok=True)
        cleanup_temp()
        self.application = ApplicationBuilder().token(token).build()
        self.extractor = ProfessionalSubtitleSystem()
        
        # Handlers
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("mode", self.mode_command))
        self.application.add_handler(CommandHandler("update", self.update_bot))
        self.application.add_handler(CallbackQueryHandler(self.button_callback))
        self.application.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO, self.handle_video))
        
    async def is_admin(self, user_id):
        return user_id in ADMIN_IDS

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        welcome_text = (
            "🌟 **Welcome to Ultimate Subtitle AI V8** 🌟\n\n"
            "This bot is designed for high-precision hardsub reconstruction and fast extraction.\n\n"
            "✅ **Multi-User Processing**: You can send videos while others are processing!\n"
            "💎 **Precision (RECONSTRUCTION)**\n"
            "⚡ **Fast (EXTRACTION)**\n\n"
            "📌 **How to use:** Send me a video, or use /mode to switch engine!"
        )
        await update.message.reply_text(welcome_text, parse_mode=constants.ParseMode.MARKDOWN)

    async def mode_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        current = BotManager.user_modes.get(update.effective_user.id, "precision")
        keyboard = [[InlineKeyboardButton("💎 Precision", callback_data="set_precision"), InlineKeyboardButton("⚡ Fast", callback_data="set_fast")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(f"🛠 **Engine Selection**\n\nCurrent active mode: `{current.upper()}`", reply_markup=reply_markup, parse_mode=constants.ParseMode.MARKDOWN)

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        user_id = query.from_user.id
        await query.answer()
        
        if query.data == "set_precision":
            BotManager.user_modes[user_id] = "precision"
            await query.edit_message_text("✅ Engine set to **PRECISION**", parse_mode=constants.ParseMode.MARKDOWN)
        elif query.data == "set_fast":
            BotManager.user_modes[user_id] = "fast"
            await query.edit_message_text("✅ Engine set to **FAST**", parse_mode=constants.ParseMode.MARKDOWN)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = (
            "📖 **Expert Help Guide**\n\n"
            "1️⃣ **Mode Selection:** Use /mode to choose between Quality or Speed.\n"
            "2️⃣ **Real-Time Queuing:** If the bot is busy, you will be placed in a fair queue.\n"
            "3️⃣ **Automatic Clean-up:** All temporary data is deleted after your subtitles are ready.\n\n"
            "💡 *Tip: Sending videos while in the queue is okay!*"
        )
        await update.message.reply_text(help_text, parse_mode=constants.ParseMode.MARKDOWN)

    async def handle_video(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        message = update.message
        video_file = message.video or message.document
        if not video_file: return
            
        mode = BotManager.user_modes.get(user_id, "precision")
        status_msg = await message.reply_text(f"📥 **Added to Queue ({mode.upper()})**", parse_mode=constants.ParseMode.MARKDOWN)
        
        file_id = video_file.file_id
        # Use user_id and timestamp for unique file naming
        safe_id = f"{user_id}_{file_id[:10]}_{int(time.time())}"
        video_path = f"temp/{safe_id}.mp4"
        srt_path_orig = f"temp/{safe_id}_orig.srt"
        srt_path_trans = f"temp/{safe_id}_id.srt"
        
        async with BotManager.process_semaphore: # Ensuring concurrency control
            try:
                # 1. DOWNLOAD
                file = await context.bot.get_file(file_id)
                await file.download_to_drive(video_path)
                await edit_status(status_msg, "Media Initialized", 15, mode)
                
                # 2. AI PROCESSING
                last_update = [0]
                loop = asyncio.get_event_loop()
                
                def sync_progress_callback(p_internal):
                    now = time.time()
                    if now - last_update[0] < 1.0: return
                    last_update[0] = now
                    current_p = int(20 + (p_internal * 0.75)) 
                    asyncio.run_coroutine_threadsafe(edit_status(status_msg, "Deep AI Engine", current_p, mode), loop)
                
                from functools import partial
                orig_subs, trans_subs, detected_lang = await loop.run_in_executor(
                    None, 
                    partial(self.extractor.process_full_subtitle, video_path, mode=mode, progress_callback=sync_progress_callback)
                )
                
                if not orig_subs:
                    await status_msg.edit_text("❌ **AI Failure:** Zero text detected in this file.")
                    return
                
                # 3. FINALIZING
                await edit_status(status_msg, "Finalizing Subtitles", 95, mode)
                self.extractor.save_srt(orig_subs, srt_path_orig)
                self.extractor.save_srt(trans_subs, srt_path_trans)
                
                await status_msg.delete()
                
                # Send Files
                await message.reply_document(document=open(srt_path_orig, 'rb'), filename=f"SUB_{mode.upper()}.srt", caption=f"✅ **Done!** Extracted Lang: `{detected_lang.upper()}`", parse_mode=constants.ParseMode.MARKDOWN)
                await message.reply_document(document=open(srt_path_trans, 'rb'), filename="SUB_ID_TRANSLATION.srt", caption="🇮🇩 AI-Translated Subtitles", parse_mode=constants.ParseMode.MARKDOWN)
                
            except Exception as e:
                logging.error(f"Error: {e}")
                await status_msg.edit_text(f"💢 **Process Error:** `{str(e)}`", parse_mode=constants.ParseMode.MARKDOWN)
                
            finally: # Ensuring files are cleaned up per session
                for p in [video_path, srt_path_orig, srt_path_trans]:
                    if os.path.exists(p):
                        try: os.remove(p)
                        except: pass
                if os.path.exists(f"{video_path}.wav"):
                    try: os.remove(f"{video_path}.wav")
                    except: pass

    async def update_bot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self.is_admin(update.effective_user.id):
            await update.message.reply_text("🚫 Admin command only.")
            return
            
        status_msg = await update.message.reply_text("🔄 **Checking for updates...**", parse_mode=constants.ParseMode.MARKDOWN)
        
        try:
            import subprocess
            import sys
            pull_result = subprocess.run(['git', 'pull', 'origin', 'main'], capture_output=True, text=True)
            if "Already up to date." in pull_result.stdout:
                await status_msg.edit_text("✅ **Bot is up to date.**", parse_mode=constants.ParseMode.MARKDOWN)
                return
            await status_msg.edit_text(f"🚀 **Pembaruan ditemukan!**\n\n`{pull_result.stdout[:200]}`\n\n⚙️ Restarting bot...")
            args = [sys.executable] + sys.argv
            os.execv(sys.executable, args)
        except Exception as e:
            await status_msg.edit_text(f"❌ **Update failed:** `{str(e)}`", parse_mode=constants.ParseMode.MARKDOWN)

    def run(self):
        print("🤖 Bot is starting...")
        self.application.run_polling()

if __name__ == '__main__':
    if not API_TOKEN:
        print("FATAL: BOT_TOKEN not found in .env!")
    else:
        bot = SubtitleBot(API_TOKEN)
        bot.run()
