# logger_ui.py
from IPython.display import display, HTML
from .client import send_message, fetch_messages
import uuid

SERVER_URL = "https://mc.blahaj.sg"


def log(string, user="sys"):
    user_id = f"{user}_{uuid.uuid4().hex[:6]}"
    send_message(user_id, string)


def display_log():
    return fetch_messages()