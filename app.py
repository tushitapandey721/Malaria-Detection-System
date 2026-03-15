"""
app.py
======
Flask web application for MalariaNet — Malaria Detection Demo.
Serves a full-featured UI for uploading cell images and getting CNN predictions.

Usage:
    python app.py

Then open: http://localhost:5000
"""

import os
import io
import json
import time
import uuid
import base64
import numpy as np
from pathlib import Path
from datetime import datetime
from functools import lru_cache

from flask import (
    Flask, request, jsonify, render_template,
    send_from_directory
)
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image

# ── CONFIG ──────────────────────────────────────────────────────────────────
IMAGE_SIZE     = (224, 224)
MODELS_DIR     = "models"
UPLOAD_FOLDER  = "static/uploads"
ALLOWED_EXTS   = {"png", "jpg", "jpeg", "bmp", "tiff"}
MAX_CONTENT_MB = 16
RESULTS_FILE   = "results/training_summary.json"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024
app.secret_key = os.environ.get("SECRET_KEY", "malaria-det-2024")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── MODEL LOADING ────────────────────────────────────────────────────────────
_models = {}

def load_keras_model(model_name):
    """Load model once and cache it."""
    if model_name in _models:
        return _models[model_name]

    for suffix in ["_best.keras", "_final.keras", "_best.h5", "_final.h5"]:
        path = os.path.join(MODELS_DIR, model_name + suffix)
        if os.path.exists(path):
            print(f"[app] Loading {model_name} from {path} ...")
            _models[model_name] = load_model(path)
            print(f"[app] {model_name} loaded.")
            return _models[model_name]

    return None   # not trained yet


def preprocess_image(img_bytes):
    """Convert raw image bytes → normalized numpy array (1,224,224,3)."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


# ── ROUTES ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST /api/predict
    Accepts: multipart/form-data with 'file' and optional 'model' fields.
    Returns: JSON prediction result.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file       = request.files["file"]
    model_name = request.form.get("model", "MobileNetV2")

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Use: {', '.join(ALLOWED_EXTS)}"}), 400

    img_bytes = file.read()

    # Load model
    model = load_keras_model(model_name)
    if model is None:
        return jsonify({
            "error": f"Model '{model_name}' not found. Run train.py first.",
            "hint":  "python train.py"
        }), 503

    # Inference
    t0  = time.perf_counter()
    arr = preprocess_image(img_bytes)
    prob = float(model.predict(arr, verbose=0)[0][0])
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    label      = "Parasitized" if prob > 0.5 else "Uninfected"
    confidence = prob if prob > 0.5 else 1.0 - prob

    # Save uploaded file
    filename  = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    with open(save_path, "wb") as f_out:
        f_out.write(img_bytes)

    return jsonify({
        "prediction":      label,
        "confidence":      round(confidence, 4),
        "parasite_prob":   round(prob, 4),
        "uninfected_prob": round(1.0 - prob, 4),
        "model":           model_name,
        "inference_ms":    elapsed_ms,
        "image_url":       f"/static/uploads/{filename}",
        "timestamp":       datetime.now().isoformat(),
    })


@app.route("/api/predict_base64", methods=["POST"])
def predict_base64():
    """
    POST /api/predict_base64
    Accepts JSON: { "image": "<base64 string>", "model": "MobileNetV2" }
    Returns: JSON prediction result.
    """
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image data provided"}), 400

    model_name = data.get("model", "MobileNetV2")

    try:
        img_bytes = base64.b64decode(data["image"])
    except Exception:
        return jsonify({"error": "Invalid base64 image data"}), 400

    model = load_keras_model(model_name)
    if model is None:
        return jsonify({
            "error": f"Model '{model_name}' not found. Run train.py first.",
            "hint":  "python train.py"
        }), 503

    t0  = time.perf_counter()
    arr = preprocess_image(img_bytes)
    prob = float(model.predict(arr, verbose=0)[0][0])
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    label      = "Parasitized" if prob > 0.5 else "Uninfected"
    confidence = prob if prob > 0.5 else 1.0 - prob

    return jsonify({
        "prediction":      label,
        "confidence":      round(confidence, 4),
        "parasite_prob":   round(prob, 4),
        "uninfected_prob": round(1.0 - prob, 4),
        "model":           model_name,
        "inference_ms":    elapsed_ms,
        "timestamp":       datetime.now().isoformat(),
    })


@app.route("/api/models", methods=["GET"])
def list_models():
    """Return which models are available (trained and saved)."""
    available = []
    for name in ["MobileNetV2", "EfficientNetB0"]:
        for suffix in ["_best.keras", "_final.keras", "_best.h5", "_final.h5"]:
            if os.path.exists(os.path.join(MODELS_DIR, name + suffix)):
                available.append(name)
                break

    return jsonify({"available_models": available})


@app.route("/api/results", methods=["GET"])
def get_results():
    """Return training summary JSON if it exists."""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "No training results found. Run train.py first."}), 404


@app.route("/health")
def health():
    return jsonify({"status": "ok", "tensorflow": tf.__version__})


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"\n MalariaNet Flask App")
    print(f" Running on http://localhost:{port}")
    print(f" TensorFlow {tf.__version__}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
