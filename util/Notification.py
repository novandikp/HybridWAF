import requests


def send_notification(config, message):
    print(message)
    if config.telebot:
        baseUrl = f"https://api.telegram.org/bot{config.token}"
        url = f"{baseUrl}/sendMessage?chat_id={config.chat_id}&text={message}"
        requests.get(url)

