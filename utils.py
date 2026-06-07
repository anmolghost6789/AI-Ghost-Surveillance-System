import time
import cv2
import os
import threading
import pandas as pd

# --------------- Thread-safe stores ---------------
_alerts = []
_alerts_lock = threading.Lock()

_metrics_history = []
_metrics_lock = threading.Lock()

# --------------- TTS (lazy, optional) ---------------
_tts_engine = None
_tts_lock = threading.Lock()
last_alert_time = 0


def _get_tts():
    global _tts_engine
    if _tts_engine is None:
        try:
            import pyttsx3
            _tts_engine = pyttsx3.init()
        except Exception:
            pass
    return _tts_engine


def speak(text, cooldown):
    """Speak text with cooldown (backward-compatible). Respects TTS_ENABLED."""
    global last_alert_time
    try:
        import config
        if not getattr(config, 'TTS_ENABLED', True):
            return
    except ImportError:
        pass
    if time.time() - last_alert_time > cooldown:
        print("ALERT:", text)
        with _tts_lock:
            engine = _get_tts()
            if engine:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception:
                    pass
        last_alert_time = time.time()


# --------------- Alert store ---------------

def add_alert(alert_type, camera_id, message):
    """Record an alert event (thread-safe)."""
    alert = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": time.time(),
        "type": alert_type,
        "camera": camera_id,
        "message": message,
    }
    with _alerts_lock:
        _alerts.append(alert)
        if len(_alerts) > 500:
            _alerts.pop(0)
    print(f"[ALERT] [{alert_type.upper()}] {camera_id}: {message}")
    return alert


def get_alerts(limit=50):
    """Return the latest alerts (newest first)."""
    with _alerts_lock:
        return list(reversed(_alerts[-limit:]))


# --------------- Metrics history ---------------

def add_metrics_point(person_count):
    """Append an aggregate metrics snapshot."""
    with _metrics_lock:
        _metrics_history.append({
            "time": time.strftime("%H:%M:%S"),
            "timestamp": time.time(),
            "person_count": person_count,
        })
        if len(_metrics_history) > 2000:
            _metrics_history[:] = _metrics_history[-1000:]


def get_metrics_history(limit=120):
    with _metrics_lock:
        return list(_metrics_history[-limit:])


# --------------- Screenshots ---------------

def save_screenshot(frame, label, camera_id="cam"):
    os.makedirs("outputs/screenshots", exist_ok=True)
    filename = f"outputs/screenshots/{camera_id}_{label}_{int(time.time())}.jpg"
    cv2.imwrite(filename, frame)
    return filename


def get_screenshots(limit=30):
    """Return screenshot filenames sorted newest-first."""
    path = "outputs/screenshots"
    if not os.path.exists(path):
        return []
    files = sorted(
        [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png'))],
        key=lambda f: os.path.getmtime(os.path.join(path, f)),
        reverse=True,
    )
    return files[:limit]


# --------------- CSV logging ---------------

def log_data(person_count, phone_detected):
    data = {
        "time": time.strftime("%H:%M:%S"),
        "person_count": person_count,
        "phone": int(phone_detected),
    }
    df = pd.DataFrame([data])
    os.makedirs("outputs", exist_ok=True)
    log_path = "outputs/log.csv"
    df.to_csv(log_path, mode='a',
              header=not os.path.exists(log_path),
              index=False)


# --------------- Telegram ---------------

def send_telegram(image_path, message, token, chat_id):
    import requests
    url = f"https://api.telegram.org/bot{8185919621:AAHTh_5NM3VNnUDfpjaINFxIF0sYOYFYi0w}/sendPhoto"
    with open(image_path, "rb") as img:
        requests.post(url,
                      data={"chat_id": 525545537, "caption": message},
                      files={"photo": img})