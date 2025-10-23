import requests
import logging
from config import SALEBOT_API_KEY, SALEBOT_ADMIN_CHAT_ID

logger = logging.getLogger(__name__)


class SaleBotClient:
    def __init__(self):
        self.api_key = SALEBOT_API_KEY
        self.admin_chat_id = SALEBOT_ADMIN_CHAT_ID
        self.SALEBOT_API_URL = f"https://chatter.salebot.pro/api/{self.api_key}/callback"

    def send_message(self, data):
        """Отправка сообщения ОБЯЗАТЕЛЬНО в data должны быть "message" и "client_id" """
        try:
            response = requests.post(self.SALEBOT_API_URL, data=data)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise Exception(e)

    def send_error_message(self, error_text: str, error_place: str):
        """Отправка сообщения об ошибке"""
        data = {
            "error_place": error_place,
            "error_message": error_text,
            "message": "med_bloggers_ai_pavel",
            "client_id": self.admin_chat_id
        }
        logger.info(f"Sending error message: {data}")

        try:
            response = requests.post(self.SALEBOT_API_URL, data=data)
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")

            try:
                response_json = response.json()
                logger.debug(f"Response JSON: {response_json}")
            except ValueError:
                logger.debug(f"Response text: {response.text}")

            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            logger.error(f"Request to {self.SALEBOT_API_URL} failed: {e}")
            if 'response' in locals() and response is not None:
                logger.error(f"Raw response (if any): {getattr(response, 'text', '<no text>')}")
            raise