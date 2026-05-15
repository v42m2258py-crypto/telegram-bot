import os
import base64
import logging
import traceback
from io import BytesIO
import httpx
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ChatAction

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_BASE_URL = os.environ["AI_INTEGRATIONS_OPENAI_BASE_URL"]
OPENAI_API_KEY = os.environ["AI_INTEGRATIONS_OPENAI_API_KEY"]

openai_client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
)

conversation_history: dict[int, list[dict]] = {}

SYSTEM_PROMPT = (
    "You are a helpful, friendly AI assistant. "
    "Answer questions clearly and concisely. "
    "You can help with writing, analysis, coding, math, creative tasks, and general knowledge. "
    "Be conversational and engaging."
)

MAX_HISTORY = 20


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    conversation_history.pop(update.effective_chat.id, None)
    await update.message.reply_text(
        f"Hi {user.first_name}! I'm your AI assistant powered by GPT. "
        "Ask me anything — I can help with writing, coding, math, analysis, and more.\n\n"
        "Commands:\n"
        "/start — start a new conversation\n"
        "/clear — clear conversation history\n"
        "/image <prompt> — generate an image\n"
        "/help — show this message"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "I'm an AI assistant. Just send me any message and I'll respond!\n\n"
        "Commands:\n"
        "/start — start a new conversation\n"
        "/clear — clear conversation history\n"
        "/image <prompt> — generate an image\n"
        "/help — show this message"
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    conversation_history.pop(update.effective_chat.id, None)
    await update.message.reply_text("Conversation cleared! Let's start fresh.")


async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    prompt = " ".join(context.args).strip()

    if not prompt:
        await update.message.reply_text(
            "Please provide a description after /image.\n"
            "Example: /image a sunset over the ocean with purple clouds"
        )
        return

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)
    status_msg = await update.message.reply_text("Generating your image...")

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{OPENAI_BASE_URL}/images/generations",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-image-1",
                    "prompt": prompt,
                    "n": 1,
                    "size": "1024x1024",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        image_b64 = data["data"][0]["b64_json"]
        image_bytes = base64.b64decode(image_b64)
        image_file = BytesIO(image_bytes)
        image_file.seek(0)

        try:
            await status_msg.delete()
        except Exception:
            pass

        await update.message.reply_photo(
            photo=image_file,
            caption=f'"{prompt}"',
        )

    except Exception as e:
        logger.error(f"Image generation error: {e}\n{traceback.format_exc()}")
        try:
            await status_msg.edit_text(
                "Sorry, I couldn't generate that image. Please try a different prompt."
            )
        except Exception:
            await update.message.reply_text(
                "Sorry, I couldn't generate that image. Please try a different prompt."
            )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_message = update.message.text

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    if chat_id not in conversation_history:
        conversation_history[chat_id] = []

    conversation_history[chat_id].append({"role": "user", "content": user_message})

    if len(conversation_history[chat_id]) > MAX_HISTORY:
        conversation_history[chat_id] = conversation_history[chat_id][-MAX_HISTORY:]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history[chat_id]

    try:
        response = openai_client.chat.completions.create(
            model="gpt-5.4",
            messages=messages,
            max_completion_tokens=8192,
        )
        assistant_reply = response.choices[0].message.content

        conversation_history[chat_id].append({"role": "assistant", "content": assistant_reply})

        await update.message.reply_text(assistant_reply)

    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        await update.message.reply_text(
            "Sorry, I ran into an error. Please try again in a moment."
        )


def main() -> None:
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("image", image_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot is running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
