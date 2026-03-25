"""
Age Verification System — Flask + AWS EC2
Routes:
  /                   → Main verification tool
  /api/predict        → POST image → JSON result
  /api/predict/video  → POST video → JSON result + download
  /api/stream/frame   → POST base64 frame → annotated base64 frame
  /adultvault         → AdultVault landing
  /adultvault/verify  → AdultVault verification widget
  /royalbet           → RoyalBet age verification
  /spiritshop         → SpiritShop landing
  /spiritshop/signin  → SpiritShop sign-in with age gate
"""

import os
import cv2
import base64
import logging
import tempfile
import numpy as np
import torch
from flask import (
    Flask, render_template, request, jsonify,
    send_file, redirect, url_for
)
from werkzeug.utils import secure_filename

from core.config import Config
from core.models import (
    load_cnn_model, load_yolo_model, gh_download,
    _is_valid_model_file
)
from core.predictor import (
    MultiFaceDetector, predict_age, annotate_image,
    get_vlm_reasoning, compute_group_summary
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = Config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH

# ── Model registry — loaded eagerly at startup, shared to workers via --preload
_models: dict = {}
_detectors: dict = {}
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_yolo_model = None


def _get_yolo():
    return _yolo_model


def _get_model(backbone: str):
    return _models.get(backbone)


def _get_detector(backbone: str):
    return _detectors.get(backbone)


def _ensure_models():
    """Download any missing model files."""
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    for fname, url in Config.MODEL_URLS.items():
        dest = os.path.join(Config.MODEL_DIR, fname)
        if not _is_valid_model_file(dest):
            logger.info(f"Downloading {fname}…")
            ok = gh_download(url, dest, fname)
            logger.info(f"{'OK' if ok else 'FAILED'}: {fname}")


def _warmup_models():
    """Load all models into memory before gunicorn forks workers."""
    global _yolo_model

    logger.info("Loading YOLO model…")
    yolo_path = os.path.join(Config.MODEL_DIR, Config.YOLO_FILENAME)
    _yolo_model = load_yolo_model(yolo_path)
    logger.info("YOLO ready.")

    for backbone, fname in Config.BACKBONE_FILES.items():
        model_path = os.path.join(Config.MODEL_DIR, fname)
        if not os.path.exists(model_path):
            continue
        # Skip duplicate ViT entry (vit-tiny reuses vit_base weights)
        if backbone in _models:
            continue
        logger.info(f"Loading {backbone}…")
        try:
            _models[backbone] = load_cnn_model(backbone, model_path, _device)
            _detectors[backbone] = MultiFaceDetector(yolo_model=_yolo_model)
            logger.info(f"{backbone} ready.")
        except Exception as e:
            logger.warning(f"Could not load {backbone}: {e}")

    logger.info("All models loaded. Ready to serve requests.")


# ── Startup: download then load all models ─────────────────────────────────
_ensure_models()
_warmup_models()

ALLOWED_IMAGE = {"png", "jpg", "jpeg", "webp", "bmp"}
ALLOWED_VIDEO = {"mp4", "avi", "mov", "mkv", "webm"}


def _allowed(filename, allowed_set):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN TOOL
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", device=str(_device).upper())


# ─────────────────────────────────────────────────────────────────────────────
#  API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    POST /api/predict
    Form fields:
      file        — image file
      backbone    — model backbone name (default: densenet)
      conf        — detection confidence (default: 0.35)
      use_vlm     — "true"/"false"
      openai_key  — OpenAI API key (optional)
    Returns JSON with faces, summary, annotated image (base64).
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not _allowed(f.filename, ALLOWED_IMAGE):
        return jsonify({"error": "Invalid file type"}), 400

    backbone = request.form.get("backbone", Config.DEFAULT_BACKBONE)
    conf = float(request.form.get("conf", 0.35))
    use_vlm = request.form.get("use_vlm", "false").lower() == "true"
    api_key = request.form.get("openai_key", Config.OPENAI_API_KEY)

    raw = f.read()
    if not raw:
        return jsonify({"error": "Empty file received"}), 400
    file_bytes = np.frombuffer(raw, np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Could not decode image — unsupported format or corrupt file"}), 400

    model = _get_model(backbone)
    detector = _get_detector(backbone)
    if model is None or detector is None:
        backbone = Config.DEFAULT_BACKBONE
        model = _get_model(backbone)
        detector = _get_detector(backbone)
    if model is None or detector is None:
        return jsonify({"error": "No models are loaded. Check server logs."}), 500

    raw_faces = detector.detect(img_bgr, conf=conf)
    if not raw_faces:
        return jsonify({"faces": [], "summary": None, "message": "No faces detected"})

    faces_data = []
    for i, face in enumerate(raw_faces):
        result = predict_age(face["crop_bgr"], model, _device)
        vlm = get_vlm_reasoning(face["crop_bgr"], result, i + 1, api_key) if use_vlm else None

        _, crop_buf = cv2.imencode(".jpg", face["crop_bgr"])
        crop_b64 = base64.b64encode(crop_buf).decode()

        faces_data.append({
            "id": i + 1,
            "bbox": list(face["bbox"]),
            "detection_method": face["method"],
            "detection_confidence": round(face["confidence"], 3),
            "crop_b64": crop_b64,
            "result": result,
            "vlm": vlm,
        })

    annotated = annotate_image(img_bgr, [
        {"id": fd["id"], "bbox": fd["bbox"], "result": fd["result"]}
        for fd in faces_data
    ])
    _, ann_buf = cv2.imencode(".jpg", annotated)
    ann_b64 = base64.b64encode(ann_buf).decode()

    summary = compute_group_summary([
        {"result": fd["result"]} for fd in faces_data
    ])

    # Strip crop from response faces to keep JSON lean if needed
    faces_out = []
    for fd in faces_data:
        faces_out.append({
            "id": fd["id"],
            "bbox": fd["bbox"],
            "detection_method": fd["detection_method"],
            "detection_confidence": fd["detection_confidence"],
            "crop_b64": fd["crop_b64"],
            "predicted_age": fd["result"]["predicted_age"],
            "age_group": fd["result"]["age_group"],
            "confidence": fd["result"]["confidence"],
            "group_probs": fd["result"]["group_probs"],
            "decision": fd["result"]["decision"],
            "vlm": fd["vlm"],
        })

    return jsonify({
        "faces": faces_out,
        "summary": summary,
        "annotated_b64": ann_b64,
        "backbone": backbone,
        "device": str(_device),
    })


@app.route("/api/predict/video", methods=["POST"])
def api_predict_video():
    """
    POST /api/predict/video
    Form fields:
      file        — video file
      backbone    — model backbone
      conf        — confidence threshold
      max_frames  — max frames to process (default: 150)
    Returns JSON with per-frame stats + download URL for annotated video.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not _allowed(f.filename, ALLOWED_VIDEO):
        return jsonify({"error": "Invalid file type"}), 400

    backbone = request.form.get("backbone", Config.DEFAULT_BACKBONE)
    conf = float(request.form.get("conf", 0.35))
    max_frames = int(request.form.get("max_frames", 150))

    safe_name = secure_filename(f.filename)
    video_path = os.path.join(Config.UPLOAD_DIR, safe_name)
    f.save(video_path)

    model = _get_model(backbone)
    detector = _get_detector(backbone)
    if model is None or detector is None:
        backbone = Config.DEFAULT_BACKBONE
        model = _get_model(backbone)
        detector = _get_detector(backbone)
    if model is None or detector is None:
        return jsonify({"error": "No models are loaded. Check server logs."}), 500

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Cannot open video"}), 400

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_name = "annotated_" + safe_name
    out_path = os.path.join(Config.UPLOAD_DIR, out_name)
    writer = None
    for fc in ("avc1", "mp4v", "XVID", "MJPG"):
        try:
            w = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fc), fps, (width, height))
            if w.isOpened():
                writer = w
                break
            w.release()
        except Exception:
            pass

    if writer is None:
        cap.release()
        return jsonify({"error": "Could not initialize video writer"}), 500

    frame_count = 0
    all_ages = []
    group_counts = {"child": 0, "teen": 0, "adult": 0}

    try:
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 2 == 0:
                raw_faces = detector.detect(frame, conf=conf)
                faces_data = []
                for i, face in enumerate(raw_faces):
                    result = predict_age(face["crop_bgr"], model, _device)
                    faces_data.append({"id": i + 1, "bbox": face["bbox"], "result": result})
                    all_ages.append(result["predicted_age"])
                    group_counts[result["age_group"]] = group_counts.get(result["age_group"], 0) + 1

                annotated = annotate_image(frame, faces_data)
                cv2.putText(annotated, f"Faces: {len(raw_faces)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                writer.write(annotated)
            else:
                writer.write(frame)

            frame_count += 1
    finally:
        cap.release()
        writer.release()

    total = len(all_ages)
    summary = {
        "total_frames_processed": frame_count,
        "total_face_detections": total,
        "avg_age": round(sum(all_ages) / total, 1) if total else 0,
        "min_age": round(min(all_ages), 1) if total else 0,
        "max_age": round(max(all_ages), 1) if total else 0,
        "group_counts": group_counts,
        "group_percentages": {
            g: round(group_counts[g] / total * 100, 1) if total else 0
            for g in group_counts
        },
        "has_minors": group_counts["child"] + group_counts["teen"] > 0,
    }

    return jsonify({
        "summary": summary,
        "download_url": f"/api/download/{out_name}",
        "backbone": backbone,
    })


@app.route("/api/download/<filename>")
def api_download(filename):
    safe = secure_filename(filename)
    path = os.path.join(Config.UPLOAD_DIR, safe)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True, download_name=safe)


@app.route("/api/stream/frame", methods=["POST"])
def api_stream_frame():
    """
    POST /api/stream/frame
    JSON body:
      frame_b64   — base64 JPEG frame from browser
      backbone    — model backbone
      conf        — confidence threshold
    Returns JSON:
      annotated_b64  — base64 annotated frame
      faces          — list of face results
    """
    data = request.get_json(silent=True) or {}
    frame_b64 = data.get("frame_b64", "")
    backbone = data.get("backbone", Config.DEFAULT_BACKBONE)
    conf = float(data.get("conf", 0.35))

    if not frame_b64:
        return jsonify({"error": "No frame data"}), 400

    try:
        img_bytes = base64.b64decode(frame_b64.split(",")[-1])
        file_bytes = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"Frame decode failed: {e}"}), 400

    if img_bgr is None:
        return jsonify({"error": "Could not decode frame"}), 400

    model = _get_model(backbone)
    detector = _get_detector(backbone)
    if model is None or detector is None:
        backbone = Config.DEFAULT_BACKBONE
        model = _get_model(backbone)
        detector = _get_detector(backbone)
    if model is None or detector is None:
        return jsonify({"error": "No models are loaded. Check server logs."}), 500

    raw_faces = detector.detect(img_bgr, conf=conf)
    faces_data = []
    face_results = []
    for i, face in enumerate(raw_faces):
        result = predict_age(face["crop_bgr"], model, _device)
        faces_data.append({"id": i + 1, "bbox": face["bbox"], "result": result})
        face_results.append({
            "id": i + 1,
            "predicted_age": result["predicted_age"],
            "age_group": result["age_group"],
            "confidence": result["confidence"],
            "decision": result["decision"],
        })

    annotated = annotate_image(img_bgr, faces_data)
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
    ann_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    return jsonify({
        "annotated_b64": ann_b64,
        "faces": face_results,
        "face_count": len(face_results),
    })


@app.route("/api/models/status")
def api_models_status():
    statuses = {}
    for fname in Config.MODEL_URLS:
        path = os.path.join(Config.MODEL_DIR, fname)
        statuses[fname] = {
            "exists": os.path.exists(path),
            "valid": _is_valid_model_file(path) if os.path.exists(path) else False,
        }
    return jsonify({"models": statuses, "device": str(_device)})


@app.route("/api/models/download", methods=["POST"])
def api_models_download():
    data = request.get_json(silent=True) or {}
    fname = data.get("filename", "")
    if fname not in Config.MODEL_URLS:
        return jsonify({"error": "Unknown model filename"}), 400
    dest = os.path.join(Config.MODEL_DIR, fname)
    ok = gh_download(Config.MODEL_URLS[fname], dest, fname)
    return jsonify({"success": ok, "filename": fname})


# ─────────────────────────────────────────────────────────────────────────────
#  ADULTVAULT PAGES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/adultvault")
def adultvault_landing():
    return render_template("adultvault/landing.html")


@app.route("/adultvault/verify")
def adultvault_verify():
    return render_template("adultvault/verify.html")


# ─────────────────────────────────────────────────────────────────────────────
#  ROYALBET PAGES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/royalbet")
def royalbet_landing():
    return render_template("royalbet/landing.html")


@app.route("/royalbet/verify")
def royalbet_verify():
    return render_template("royalbet/verify.html")


# ─────────────────────────────────────────────────────────────────────────────
#  SPIRITSHOP PAGES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/spiritshop")
def spiritshop_landing():
    return render_template("spiritshop/landing.html")


@app.route("/spiritshop/signin")
def spiritshop_signin():
    return render_template("spiritshop/signin.html")


# ─────────────────────────────────────────────────────────────────────────────
#  ERROR HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large (max 500 MB)"}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
