from flask import (Flask, Response, jsonify, request,
                   render_template, send_from_directory)
import cv2
import threading
import time
import os

import config
import utils
from detector import DetectionEngine
from camera import CameraStream
from criminal_detector import CriminalDetector

app = Flask(__name__)
engine = DetectionEngine()
criminal = CriminalDetector()
cameras: dict[str, CameraStream] = {}


# ==================== Init ====================

def init_cameras():
    for entry in config.CAMERA_SOURCES:
        cam = CameraStream(entry["id"], entry["source"],
                           entry["label"], engine,
                           criminal_detector=criminal)
        if cam.start():
            cameras[entry["id"]] = cam
            print(f"[OK] Camera '{entry['label']}' ({entry['id']}) started")
        else:
            print(f"[ERROR] Cannot open '{entry['label']}' "
                  f"({entry['source']})")


def _metrics_collector():
    """Background thread: sample aggregate metrics once per second."""
    while True:
        total = sum(c.person_count for c in cameras.values())
        utils.add_metrics_point(total)
        time.sleep(1)


# ==================== Video streaming ====================

def _generate_mjpeg(cam_id, heatmap=False):
    cam = cameras.get(cam_id)
    if cam is None:
        return
    while cam.running:
        frame = cam.get_frame(with_heatmap=heatmap)
        if frame is None:
            time.sleep(0.05)
            continue
        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')


# ==================== Pages ====================

@app.route('/')
def dashboard():
    return render_template('dashboard.html',
                           cameras=[c.status() for c in cameras.values()])


@app.route('/video_feed/<cam_id>')
def video_feed(cam_id):
    if cam_id not in cameras:
        return jsonify({"error": "Camera not found"}), 404
    heatmap = request.args.get('heatmap', '0') == '1'
    return Response(_generate_mjpeg(cam_id, heatmap),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/screenshots/<path:filename>')
def serve_screenshot(filename):
    return send_from_directory('outputs/screenshots', filename)


# ==================== REST API ====================

@app.route('/api/cameras', methods=['GET'])
def api_cameras():
    return jsonify([c.status() for c in cameras.values()])


@app.route('/api/cameras/add', methods=['POST'])
def api_add_camera():
    data = request.get_json(force=True)
    cam_id = data.get("id")
    source = data.get("source")
    label = data.get("label", cam_id)

    if not cam_id or source is None:
        return jsonify({"error": "id and source are required"}), 400
    if cam_id in cameras:
        return jsonify({"error": f"Camera '{cam_id}' already exists"}), 409

    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cam = CameraStream(cam_id, source, label, engine,
                        criminal_detector=criminal)
    if not cam.start():
        return jsonify({"error": f"Cannot open source: {source}"}), 500

    cameras[cam_id] = cam
    return jsonify({"message": "Camera added", "status": cam.status()}), 201


@app.route('/api/cameras/<cam_id>/remove', methods=['POST'])
def api_remove_camera(cam_id):
    cam = cameras.pop(cam_id, None)
    if cam is None:
        return jsonify({"error": "Camera not found"}), 404
    cam.stop()
    return jsonify({"message": f"Camera '{cam_id}' removed"})


@app.route('/api/cameras/<cam_id>/snapshot', methods=['GET'])
def api_snapshot(cam_id):
    cam = cameras.get(cam_id)
    if cam is None:
        return jsonify({"error": "Camera not found"}), 404
    frame = cam.get_frame()
    if frame is None:
        return jsonify({"error": "No frame available"}), 503
    ret, buf = cv2.imencode('.jpg', frame)
    if not ret:
        return jsonify({"error": "Encoding failed"}), 500
    return Response(buf.tobytes(), mimetype='image/jpeg')


@app.route('/api/alerts', methods=['GET'])
def api_alerts():
    limit = request.args.get('limit', 50, type=int)
    return jsonify(utils.get_alerts(limit))


@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    return jsonify({
        "total_people": sum(c.person_count for c in cameras.values()),
        "total_alerts": len(utils.get_alerts(10000)),
        "cameras": [c.status() for c in cameras.values()],
        "history": utils.get_metrics_history(120),
    })


@app.route('/api/screenshots', methods=['GET'])
def api_screenshots():
    return jsonify(utils.get_screenshots())


@app.route('/api/criminal_alerts', methods=['GET'])
def api_criminal_alerts():
    """Return only criminal-type alerts."""
    limit = request.args.get('limit', 30, type=int)
    all_alerts = utils.get_alerts(10000)
    crim = [a for a in all_alerts if a["type"] == "criminal"]
    return jsonify(crim[:limit])


@app.route('/api/criminal/reload', methods=['POST'])
def api_reload_criminals():
    """Hot-reload criminal face encodings from disk."""
    criminal.reload()
    return jsonify({"message": "Reloaded",
                    "count": len(criminal.known_names),
                    "names": criminal.known_names})


# ==================== Entry point ====================

if __name__ == '__main__':
    os.makedirs('outputs/screenshots', exist_ok=True)
    init_cameras()
    threading.Thread(target=_metrics_collector, daemon=True).start()
    print(f"\n  Dashboard: http://localhost:{config.DASHBOARD_PORT}\n")
    app.run(host=config.DASHBOARD_HOST,
            port=config.DASHBOARD_PORT,
            debug=False, threaded=True)