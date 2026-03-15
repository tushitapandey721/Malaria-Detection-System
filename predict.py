"""
predict.py
==========
Standalone inference script — run prediction on a single image or folder.

Usage:
    python predict.py --image path/to/cell.png
    python predict.py --folder path/to/folder/
    python predict.py --image cell.png --model MobileNetV2
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# ── CONFIG ──────────────────────────────────────────────────────────────────
IMAGE_SIZE  = (224, 224)
MODELS_DIR  = "models"
CLASS_NAMES = {0: "Uninfected", 1: "Parasitized"}
SUPPORTED   = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def load_best_model(model_name="MobileNetV2"):
    """Load the saved Keras model."""
    # Try best first, then final
    for suffix in ["_best.keras", "_final.keras", "_best.h5", "_final.h5"]:
        path = os.path.join(MODELS_DIR, model_name + suffix)
        if os.path.exists(path):
            print(f"Loading model: {path}")
            return load_model(path)

    raise FileNotFoundError(
        f"No saved model found for '{model_name}' in ./{MODELS_DIR}/\n"
        f"Run train.py first to train and save the model."
    )


def preprocess(img_path):
    """Load and preprocess a single image."""
    img = keras_image.load_img(img_path, target_size=IMAGE_SIZE)
    arr = keras_image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)   # shape: (1, 224, 224, 3)


def predict_single(model, img_path):
    """Return prediction dict for one image."""
    arr  = preprocess(img_path)
    prob = float(model.predict(arr, verbose=0)[0][0])

    label      = "Parasitized" if prob > 0.5 else "Uninfected"
    confidence = prob if prob > 0.5 else 1.0 - prob

    return {
        "image":              os.path.basename(img_path),
        "prediction":         label,
        "confidence":         round(confidence, 4),
        "parasite_prob":      round(prob,        4),
        "uninfected_prob":    round(1.0 - prob,  4),
    }


def predict_folder(model, folder_path):
    """Run predictions on all images in a folder."""
    folder = Path(folder_path)
    imgs   = [f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED]

    if not imgs:
        print(f"No supported images found in {folder_path}")
        return []

    results = []
    for i, img_path in enumerate(sorted(imgs), 1):
        res = predict_single(model, str(img_path))
        results.append(res)
        verdict = "🔴 PARASITIZED" if res["prediction"] == "Parasitized" else "🟢 Uninfected"
        print(f"  [{i:3d}/{len(imgs)}] {img_path.name:<40} {verdict}  ({res['confidence']*100:.1f}%)")

    # Summary
    n_para  = sum(1 for r in results if r["prediction"] == "Parasitized")
    n_uninf = len(results) - n_para
    print(f"\n  Total  : {len(results)}")
    print(f"  🔴 Parasitized : {n_para}  ({n_para/len(results)*100:.1f}%)")
    print(f"  🟢 Uninfected  : {n_uninf}  ({n_uninf/len(results)*100:.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Malaria Detection — Inference")
    parser.add_argument("--image",  type=str, help="Path to a single cell image")
    parser.add_argument("--folder", type=str, help="Path to a folder of images")
    parser.add_argument("--model",  type=str, default="MobileNetV2",
                        choices=["MobileNetV2", "EfficientNetB0"],
                        help="Which model to use (default: MobileNetV2)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to this JSON file")
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.print_help()
        sys.exit(1)

    model = load_best_model(args.model)

    results = []

    if args.image:
        if not os.path.exists(args.image):
            print(f"File not found: {args.image}")
            sys.exit(1)
        res = predict_single(model, args.image)
        results.append(res)
        print(f"\n{'='*45}")
        print(f"  Image      : {res['image']}")
        print(f"  Prediction : {res['prediction']}")
        print(f"  Confidence : {res['confidence']*100:.2f}%")
        print(f"  Parasite   : {res['parasite_prob']*100:.2f}%")
        print(f"  Uninfected : {res['uninfected_prob']*100:.2f}%")
        print(f"{'='*45}\n")

    elif args.folder:
        print(f"\nRunning predictions on: {args.folder}\n")
        results = predict_folder(model, args.folder)

    if args.output and results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
