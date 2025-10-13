import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web

from config import API_TOKEN, WEBHOOK_PATH, WEBHOOK_URL, APP_HOST, APP_PORT, DEV_MODE

logging.basicConfig(level=logging.INFO)

bot = Bot(
    token=API_TOKEN,
    default=DefaultBotProperties(parse_mode="HTML"),
)
dp = Dispatcher(storage=MemoryStorage())

from bot.router import router as bot_router
dp.include_router(bot_router)


async def on_startup(app):
    if DEV_MODE == "webhook":
        logging.info(f"🌐 Setting webhook: {WEBHOOK_URL}")
        await bot.set_webhook(WEBHOOK_URL)
    else:
        logging.info("💡 Skipping webhook setup (polling mode)")


async def on_shutdown(app):
    if DEV_MODE == "webhook":
        logging.info("🔻 Removing webhook")
        await bot.delete_webhook()


def main():
    if DEV_MODE == "polling":
        logging.info("🟢 Starting bot in polling mode...")
        asyncio.run(dp.start_polling(bot))
        return

    # === webhook mode ===
    app = web.Application()
    SimpleRequestHandler(dispatcher=dp, bot=bot).register(app, path=WEBHOOK_PATH)
    setup_application(app, dp, bot=bot)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    logging.info(f"🚀 Starting webhook server on http://{APP_HOST}:{APP_PORT}")
    web.run_app(app, host=APP_HOST, port=APP_PORT)


if __name__ == "__main__":
    main()
