import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import face_recognition
import os
import time


import config
import utils

# Load models
model = YOLO("yolov8n.pt")
tracker = DeepSort()

# Load faces (FINAL FIX)
known_encodings = []
known_names = []

for file in os.listdir("known_faces"):
    path = os.path.join("known_faces", file)

    # Read image using OpenCV
    img = cv2.imread(path)

    if img is None:
        print(f"[ERROR] Cannot load {file}")
        continue

    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Ensure correct type
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Check shape (must be 3 channel)
    if len(img.shape) != 3 or img.shape[2] != 3:
        print(f"[ERROR] Invalid image format: {file}")
        continue

    try:
        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(file)[0])
        else:
            print(f"[WARNING] No face found in {file}")

    except Exception as e:
        print(f"[ERROR] Face encoding failed for {file}: {e}")


# Camera
caps = [cv2.VideoCapture(src["source"]) for src in config.CAMERA_SOURCES]

heatmap = None
last_phone_alert = 0
last_crowd_alert = 0

while True:
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        if heatmap is None:
            heatmap = np.zeros((h, w), dtype=np.float32)

        results = model(frame)[0]

        detections = []
        phone_detected = False

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf < config.CONFIDENCE_THRESHOLD:
                continue

            if cls == 0:
                detections.append(([x1, y1, x2-x1, y2-y1], conf, 'person'))

            if cls == 67:
                phone_detected = True

        tracks = tracker.update_tracks(detections, frame=frame)

        ids = set()

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, w_box, h_box = map(int, track.to_ltrb())

            cx = int(l + w_box/2)
            cy = int(t + h_box/2)

            ids.add(track_id)

            heatmap[cy, cx] += 1

            # Restricted zone
            x1, y1, x2, y2 = config.RESTRICTED_ZONE
            if x1 < cx < x2 and y1 < cy < y2:
                utils.speak("Restricted area breach", config.ALERT_COOLDOWN)

            # Entry/Exit
            if not hasattr(track, "prev_y"):
                track.prev_y = cy

            if track.prev_y < config.LINE_Y and cy >= config.LINE_Y:
                print("Entered")

            track.prev_y = cy

            cv2.rectangle(frame,(l,t),(l+w_box,t+h_box),(0,255,0),2)
            cv2.putText(frame,f"ID {track_id}",(l,t-10),0,0.6,(0,255,0),2)

        person_count = len(ids)

        # Face recognition (SAFE VERSION)
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if rgb.dtype != np.uint8:
                rgb = rgb.astype(np.uint8)

            faces = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, faces)

            for (top, right, bottom, left), enc in zip(faces, encs):
                name = "Unknown"

                if len(known_encodings) > 0:
                    matches = face_recognition.compare_faces(known_encodings, enc)
                    face_distances = face_recognition.face_distance(known_encodings, enc)

                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_names[best_match_index]

                cv2.putText(frame, name, (left, top-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                cv2.rectangle(frame, (left, top), (right, bottom), (255,0,0), 2)

        except Exception as e:
            print("[Face Error]", e)

        # Phone alert
        if phone_detected:
            if time.time() - last_phone_alert > config.ALERT_COOLDOWN:
                path = utils.save_screenshot(frame, "phone")
                utils.speak("Phone detected", config.ALERT_COOLDOWN)
                last_phone_alert = time.time()

        # Crowd alert
        if person_count > config.CROWD_THRESHOLD:
            if time.time() - last_crowd_alert > config.ALERT_COOLDOWN:
                path = utils.save_screenshot(frame, "crowd")
                utils.speak("Crowd detected", config.ALERT_COOLDOWN)
                last_crowd_alert = time.time()

        # Logging
        utils.log_data(person_count, phone_detected)

        # Heatmap overlay
        heatmap_blur = cv2.GaussianBlur(heatmap, (15,15), 0)
        heatmap_color = cv2.applyColorMap(
            cv2.normalize(heatmap_blur,None,0,255,cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(frame,0.7,heatmap_color,0.3,0)

        cv2.putText(overlay,f"Count: {person_count}",(20,40),0,1,(0,255,255),2)
        cv2.imshow("AI Surveillance", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()