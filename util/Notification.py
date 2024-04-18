import requests
import os
import datetime


def send_notification(config, message: str):
    print(message)
    if config.telebot:
        try:
            baseUrl = f"https://api.telegram.org/bot{config.token}"
            url = f"{baseUrl}/sendMessage?chat_id={config.chat_id}&text={message}"
            requests.get(url)
        except Exception as e:
            print(f"Error send notification: {e}")
    if config.log:
        logs_path = os.path.join(os.getcwd(), "logs")
        if not os.path.exists(logs_path):
            os.mkdir(logs_path)
        filename = datetime.datetime.now().strftime("%Y%m%d.log")
        with open(os.path.join(logs_path, filename), "a") as f:
            f.write(f"{datetime.datetime.now()} - {message}\n")
