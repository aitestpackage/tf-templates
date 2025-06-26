# client.py
import requests

SERVER_URL = "https://mc.blahaj.sg"

def send_message(user, msg):
    try:
        r = requests.post(f"{SERVER_URL}/send", json={"user": user, "msg": msg}, timeout=3)
        r.raise_for_status()
    except Exception as e:
        print(f"[Logger] Send failed: {e}")


def fetch_messages():
    try:
        r = requests.get(f"{SERVER_URL}/recv", timeout=3)
        r.raise_for_status()
        return r.json().get("msg", "")
    except Exception as e:
        return f"[Logger] Fetch failed: {e}"
