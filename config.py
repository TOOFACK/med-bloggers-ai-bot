import os

from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")
API_TOKEN_LOCAL = os.getenv("TELEGRAM_API_TOKEN_LOCAL")
WEBHOOK_PATH = f"/bot/{API_TOKEN}"
APP_HOST = "0.0.0.0"
APP_PORT = 8080
WEBHOOK_BASE = os.getenv("WEBHOOK_URL", "")
WEBHOOK_URL = WEBHOOK_BASE + WEBHOOK_PATH

BOT_MODE = os.getenv("BOT_MODE", "test").lower()
DEV_MODE = os.getenv("DEV_MODE", "webhook").lower()


DB_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://bot:bot@db:5432/botdb")

# PROMPT_SUGGESTION_COUNT = int(os.getenv("PROMPT_SUGGESTION_COUNT", "3"))

COMET_API_KEY = os.getenv("COMET_API_KEY")
COMET_BASE_URL = os.getenv("COMET_BASE_URL", "https://api.cometapi.com/v1beta")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

PROMPT_MODEL = os.getenv("PROMPT_MODEL", "google/gemini-2.0-flash")

VERTEX_CREDENTIALS_PATH = os.getenv("VERTEX_CREDENTIALS_PATH")
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_IMAGE_MODEL = os.getenv("VERTEX_IMAGE_MODEL", "gemini-2.5-flash-image")
VERTEX_ASPECT_RATIO = os.getenv("VERTEX_ASPECT_RATIO")

S3_BUCKET = os.getenv("S3_BUCKET")
S3_REGION = os.getenv("S3_REGION", "")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_PUBLIC_BASE_URL = os.getenv("S3_PUBLIC_BASE_URL")
S3_MEDIA_PREFIX = os.getenv("S3_MEDIA_PREFIX", "user-media")


# === SALEBOT =====

SALEBOT_API_KEY = os.getenv('SALEBOT_API_KEY')
SALEBOT_ADMIN_CHAT_ID = os.getenv('SALEBOT_ADMIN_CHAT_ID')


PAYMENTS_ACTIVE = os.getenv("PAYMENTS_ACTIVE") or False
