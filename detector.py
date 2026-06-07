import threading
import cv2
from ultralytics import YOLO
import config


class DetectionEngine:
    """Thread-safe YOLOv8 detection engine (shared across cameras)."""

    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self._lock = threading.Lock()

    def detect(self, frame):
        """Run detection on a frame.

        Returns:
            (annotated_frame, detections_for_deepsort, phone_detected)
            detections_for_deepsort: list of ([x, y, w, h], confidence, 'person')
        """
        with self._lock:
            results = self.model(frame, verbose=False)[0]

        detections = []
        phone_detected = False

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf < config.CONFIDENCE_THRESHOLD:
                continue

            if cls == 0:  # person
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

            if cls == 67:  # cell phone
                phone_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Phone {conf:.0%}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame, detections, phone_detected
