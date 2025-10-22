import requests

import os



class SaleBotClient:
    def __init__(self):
        self.api_key = os.getenv("SALEBOT_API_KEY")
        self.admin_chat_id = os.getenv("SALEBOT_ADMIN_CHAT_ID")
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

        try:
            response = requests.post(self.SALEBOT_API_URL, data=data)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise Exception(e)
 