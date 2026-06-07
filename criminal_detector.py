import os
import threading
import cv2
import numpy as np

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("[WARNING] face_recognition module not available. Criminal detection disabled.")

import config


class CriminalDetector:
    """Load known criminal face encodings and match against video frames."""

    def __init__(self, faces_dir=None):
        self._lock = threading.Lock()
        self.faces_dir = faces_dir or getattr(config, "CRIMINAL_FACES_DIR",
                                              "known_faces")
        self.known_encodings = []   # list of numpy arrays
        self.known_names = []       # parallel list of names
        self._load_faces()

    def _load_faces(self):
        """Scan faces_dir for images and compute 128-d encodings."""
        if not FACE_RECOGNITION_AVAILABLE:
            print("[CriminalDetector] Skipping face loading - face_recognition not available")
            return
            
        if not os.path.isdir(self.faces_dir):
            print(f"[CriminalDetector] Directory not found: {self.faces_dir}")
            return

        for fname in os.listdir(self.faces_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            path = os.path.join(self.faces_dir, fname)
            # Load as guaranteed 8-bit RGB via PIL
            from PIL import Image
            pil_img = Image.open(path).convert("RGB")
            img = np.array(pil_img)
            encs = face_recognition.face_encodings(img)
            if encs:
                self.known_encodings.append(encs[0])
                name = os.path.splitext(fname)[0].replace("_", " ").title()
                self.known_names.append(name)
                print(f"[CriminalDetector] Loaded: {name}")
            else:
                print(f"[CriminalDetector] No face found in {fname}, skipped")

        print(f"[CriminalDetector] {len(self.known_names)} criminal face(s) loaded")

    def reload(self):
        """Re-scan the faces directory (hot-reload)."""
        with self._lock:
            self.known_encodings.clear()
            self.known_names.clear()
            self._load_faces()

    def scan_frame(self, frame):
        """Detect and match faces in a BGR frame.

        Returns list of dicts:
            [{"name": str, "distance": float, "box": (top, right, bottom, left)}]
        """
        if not FACE_RECOGNITION_AVAILABLE or not self.known_encodings:
            return []

        tolerance = getattr(config, "FACE_MATCH_TOLERANCE", 0.6)

        # Downscale for speed
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        matches = []
        with self._lock:
            for enc, (top, right, bottom, left) in zip(encodings, locations):
                distances = face_recognition.face_distance(
                    self.known_encodings, enc)
                best_idx = int(np.argmin(distances))
                best_dist = distances[best_idx]

                if best_dist <= tolerance:
                    # Scale back to original size
                    matches.append({
                        "name": self.known_names[best_idx],
                        "distance": round(float(best_dist), 3),
                        "box": (top * 2, right * 2, bottom * 2, left * 2),
                    })

        return matches
