import cv2
import numpy as np
import threading
import time
from deep_sort_realtime.deepsort_tracker import DeepSort

import config
import utils


class CameraStream:
    """Threaded camera with integrated YOLO detection + DeepSort tracking."""

    def __init__(self, cam_id, source, label, engine, criminal_detector=None):
        self.cam_id = cam_id
        self.source = source
        self.label = label
        self.engine = engine
        self.criminal_detector = criminal_detector

        self.cap = None
        self.tracker = DeepSort()

        self.annotated_frame = None
        self.heatmap = None
        self._lock = threading.Lock()

        self.running = False
        self.person_count = 0
        self.phone_detected = False

        # Alert cooldowns (per type)
        self._last_alert = {"phone": 0, "crowd": 0, "intrusion": 0,
                            "criminal": 0}

        # Per-name criminal cooldowns
        self._criminal_cooldowns: dict[str, float] = {}

        # Logging throttle
        self._last_log_time = 0

        # Face scan throttle (face_recognition is slow)
        self._last_face_scan = 0
        self._face_scan_interval = 2.0  # seconds between scans

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            return False
        self.running = True
        threading.Thread(target=self._process_loop, daemon=True).start()
        return True

    def _process_loop(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                self._process_frame(frame)
            except Exception as e:
                print(f"[{self.cam_id}] Error: {e}")
                time.sleep(0.1)

    def _process_frame(self, frame):
        h, w = frame.shape[:2]
        if self.heatmap is None:
            self.heatmap = np.zeros((h, w), dtype=np.float32)

        # --- Detection ---
        annotated, detections, phone = self.engine.detect(frame.copy())
        self.phone_detected = phone

        # --- Tracking ---
        tracks = self.tracker.update_tracks(detections, frame=frame)

        ids = set()
        for track in tracks:
            if not track.is_confirmed():
                continue

            tid = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            cx, cy = (l + r) // 2, (t + b) // 2
            ids.add(tid)

            # Heatmap (clamped to frame bounds)
            self.heatmap[max(0, min(cy, h - 1)), max(0, min(cx, w - 1))] += 1

            # --- Restricted zone ---
            zx1, zy1, zx2, zy2 = config.RESTRICTED_ZONE
            if zx1 < cx < zx2 and zy1 < cy < zy2:
                self._fire_alert("intrusion",
                                 f"Zone breach by ID {tid}", annotated)

            # Draw tracking box
            cv2.rectangle(annotated, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(annotated, f"ID {tid}", (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        self.person_count = len(ids)

        # --- Alerts ---
        if phone:
            self._fire_alert("phone", "Phone detected", annotated)

        if self.person_count > config.CROWD_THRESHOLD:
            self._fire_alert("crowd",
                             f"Crowd: {self.person_count} people", annotated)

        # --- Criminal face scanning (throttled) ---
        now_face = time.time()
        if (self.criminal_detector
                and now_face - self._last_face_scan >= self._face_scan_interval):
            self._last_face_scan = now_face
            crim_matches = self.criminal_detector.scan_frame(frame)
            cooldown = getattr(config, "CRIMINAL_ALERT_COOLDOWN",
                               config.ALERT_COOLDOWN)
            for m in crim_matches:
                name = m["name"]
                top, right, bottom, left = m["box"]

                # Red bounding box + label
                cv2.rectangle(annotated, (left, top), (right, bottom),
                              (0, 0, 255), 3)
                label_text = f"CRIMINAL: {name}"
                cv2.putText(annotated, label_text, (left, top - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Per-name cooldown check
                last = self._criminal_cooldowns.get(name, 0)
                if now_face - last > cooldown:
                    self._criminal_cooldowns[name] = now_face
                    msg = f"CRIMINAL DETECTED: {name} (dist {m['distance']})"
                    utils.add_alert("criminal", self.cam_id, msg)
                    path = utils.save_screenshot(annotated, f"criminal_{name}",
                                                 self.cam_id)
                    utils.speak(f"Criminal detected: {name}",
                                cooldown)
                    # Telegram for criminal alerts
                    if (config.TELEGRAM_TOKEN and config.CHAT_ID
                            and config.TELEGRAM_TOKEN != "YOUR_TOKEN"):
                        try:
                            utils.send_telegram(
                                path, f"🚨 CRIMINAL: {name} on {self.cam_id}",
                                config.TELEGRAM_TOKEN, config.CHAT_ID)
                        except Exception:
                            pass

        # --- Overlays ---
        cv2.putText(annotated, f"{self.label} | Count: {self.person_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw restricted zone outline
        zx1, zy1, zx2, zy2 = config.RESTRICTED_ZONE
        cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), (0, 0, 255), 1)
        cv2.putText(annotated, "RESTRICTED", (zx1, zy1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        with self._lock:
            self.annotated_frame = annotated

        # --- Throttled logging (every 2 s) ---
        now = time.time()
        if now - self._last_log_time > 2.0:
            utils.log_data(self.person_count, int(phone))
            self._last_log_time = now

    # ---- helpers ----

    def _fire_alert(self, alert_type, message, frame):
        now = time.time()
        if now - self._last_alert.get(alert_type, 0) > config.ALERT_COOLDOWN:
            self._last_alert[alert_type] = now
            utils.add_alert(alert_type, self.cam_id, message)
            path = utils.save_screenshot(frame, alert_type, self.cam_id)
            # Send Telegram notification
            if (config.TELEGRAM_TOKEN and config.CHAT_ID
                    and config.TELEGRAM_TOKEN != "YOUR_TOKEN"):
                try:
                    utils.send_telegram(
                        path,
                        f"[{alert_type.upper()}] {self.cam_id}: {message}",
                        config.TELEGRAM_TOKEN, config.CHAT_ID)
                except Exception:
                    pass

    def get_frame(self, with_heatmap=False):
        with self._lock:
            if self.annotated_frame is None:
                return None
            frame = self.annotated_frame.copy()

        if with_heatmap and self.heatmap is not None:
            hm = cv2.GaussianBlur(self.heatmap, (15, 15), 0)
            hm = cv2.normalize(hm, None, 0, 255,
                               cv2.NORM_MINMAX).astype(np.uint8)
            hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
            if hm_color.shape[:2] != frame.shape[:2]:
                hm_color = cv2.resize(hm_color,
                                      (frame.shape[1], frame.shape[0]))
            frame = cv2.addWeighted(frame, 0.7, hm_color, 0.3, 0)

        return frame

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def status(self):
        return {
            "id": self.cam_id,
            "label": self.label,
            "source": str(self.source),
            "active": (self.running and self.cap is not None
                       and self.cap.isOpened()),
            "person_count": self.person_count,
            "phone_detected": self.phone_detected,
        }
