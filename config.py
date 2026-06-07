# Camera sources: int (USB index), str (RTSP/HTTP URL), or file path
# Each entry: {"id": unique_name, "source": int_or_url, "label": display_name}
CAMERA_SOURCES = [
    {"id": "cam0", "source": 0, "label": "Default Webcam"},
    # {"id": "cam1", "source": "rtsp://admin:pass@192.168.1.100:554/stream1", "label": "IP Cam 1"},
    # {"id": "cam2", "source": "http://192.168.1.101:8080/video", "label": "HTTP Cam"},
    # {"id": "cam3", "source": 1, "label": "USB Camera 2"},
]

# Detection
CONFIDENCE_THRESHOLD = 0.5
CROWD_THRESHOLD = 5

# Alerts
ALERT_COOLDOWN = 10
CRIMINAL_ALERT_COOLDOWN = 30   # longer cooldown for criminal re-alerts

# Criminal / Face Recognition
CRIMINAL_FACES_DIR = "known_faces"
FACE_MATCH_TOLERANCE = 0.6     # lower = stricter matching

# Restricted Zone (x1, y1, x2, y2)
RESTRICTED_ZONE = (100, 100, 400, 400)

# Entry line
LINE_Y = 300

# Dashboard
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 5000

# Text-to-speech (disable for web-only mode)
TTS_ENABLED = True

# Telegram (optional)
TELEGRAM_TOKEN = "8185919621:AAHTh_5NM3VNnUDfpjaINFxIF0sYOYFYi0w"
CHAT_ID = "525545537"
