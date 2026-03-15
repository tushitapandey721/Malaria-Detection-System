"""
train.py
========
Malaria Detection using Deep Learning — Training Script
Trains MobileNetV2 and EfficientNetB0 on the NIH cell image dataset.

Usage:
    python train.py

Requirements:
    - cell_images/train/Parasitized/
    - cell_images/train/Uninfected/
    - cell_images/validation/Parasitized/
    - cell_images/validation/Uninfected/

Run prepare_data.py first if your dataset is not split yet.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ── CONFIG ──────────────────────────────────────────────────────────────────
IMAGE_SIZE   = (224, 224)
BATCH_SIZE   = 32
EPOCHS       = 10
LEARNING_RATE = 1e-4
TRAIN_DIR    = "cell_images/train"
VAL_DIR      = "cell_images/validation"
MODELS_DIR   = "models"
RESULTS_DIR  = "results"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── DATA GENERATORS ─────────────────────────────────────────────────────────
def build_generators():
    """Build train and validation image data generators."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    print(f"Train samples   : {train_gen.samples}")
    print(f"Val samples     : {val_gen.samples}")
    print(f"Class indices   : {train_gen.class_indices}")
    return train_gen, val_gen


# ── MODEL BUILDERS ───────────────────────────────────────────────────────────
def build_mobilenet(input_shape=(224, 224, 3)):
    """MobileNetV2 with custom binary classification head."""
    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = True

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    print(f"\nMobileNetV2 — Total params: {model.count_params():,}")
    return model


def build_efficientnet(input_shape=(224, 224, 3)):
    """
    EfficientNetB0 with correct preprocessing.
    NOTE: EfficientNetB0 expects raw [0,255] pixel values — NOT normalized [0,1].
    We add a Lambda layer to rescale back before the base model.
    """
    base = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = True

    # EfficientNet has its own internal preprocessing, so we pass
    # the normalized input through a rescale-back layer.
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(255.0)(inputs)   # [0,1] → [0,255]
    x = base(x, training=True)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    print(f"\nEfficientNetB0 — Total params: {model.count_params():,}")
    return model


# ── CALLBACKS ────────────────────────────────────────────────────────────────
def get_callbacks(model_name):
    return [
        ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, f"{model_name}_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        CSVLogger(os.path.join(RESULTS_DIR, f"{model_name}_training_log.csv")),
    ]


# ── TRAINING ─────────────────────────────────────────────────────────────────
def train_model(model, model_name, train_gen, val_gen):
    print(f"\n{'='*55}")
    print(f"  Training {model_name}")
    print(f"{'='*55}")

    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_gen),
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=len(val_gen),
        callbacks=get_callbacks(model_name),
        verbose=1,
    )

    # Save final model
    model.save(os.path.join(MODELS_DIR, f"{model_name}_final.keras"))
    print(f"\nModel saved → models/{model_name}_final.keras")
    return history


# ── EVALUATION ───────────────────────────────────────────────────────────────
def evaluate_model(model, model_name, val_gen):
    print(f"\n── Evaluating {model_name} ──")

    val_gen.reset()
    true_labels = val_gen.classes

    pred_probs = model.predict(val_gen, steps=len(val_gen), verbose=1)
    predictions = (pred_probs.flatten() > 0.5).astype(int)

    acc  = accuracy_score(true_labels, predictions)
    prec = precision_score(true_labels, predictions, zero_division=0)
    rec  = recall_score(true_labels, predictions, zero_division=0)
    f1   = f1_score(true_labels, predictions, zero_division=0)
    cm   = confusion_matrix(true_labels, predictions)

    metrics = {
        "model": model_name,
        "accuracy":  round(float(acc),  4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec),  4),
        "f1_score":  round(float(f1),   4),
        "confusion_matrix": cm.tolist(),
    }

    print(f"\n{classification_report(true_labels, predictions, target_names=['Uninfected','Parasitized'])}")

    # Save metrics JSON
    with open(os.path.join(RESULTS_DIR, f"{model_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics, cm


# ── PLOTTING ─────────────────────────────────────────────────────────────────
def plot_history(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} — Training History", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train Acc",  color="#00c465", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Acc",    color="#00c465", linewidth=2, linestyle="--")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss", color="#ff4d6d", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss",   color="#ff4d6d", linewidth=2, linestyle="--")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{model_name}_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {path}")


def plot_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=["Uninfected", "Parasitized"],
        yticklabels=["Uninfected", "Parasitized"],
        ax=ax,
    )
    ax.set_title(f"{model_name} — Confusion Matrix", fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved → {path}")


def plot_comparison(mn_metrics, en_metrics):
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    mn_vals = [mn_metrics["accuracy"], mn_metrics["precision"], mn_metrics["recall"], mn_metrics["f1_score"]]
    en_vals = [en_metrics["accuracy"], en_metrics["precision"], en_metrics["recall"], en_metrics["f1_score"]]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, mn_vals, width, label="MobileNetV2",  color="#00c465", alpha=0.85)
    bars2 = ax.bar(x + width / 2, en_vals, width, label="EfficientNetB0", color="#ff4d6d", alpha=0.85)

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Classification Metrics", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars1:
        ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison chart saved → {path}")


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  MALARIA DETECTION — CNN Training Pipeline")
    print("  TensorFlow", tf.__version__)
    print("="*55)

    # GPU check
    gpus = tf.config.list_physical_devices("GPU")
    print(f"  GPUs available: {len(gpus)}")

    # Data
    train_gen, val_gen = build_generators()

    # ── MobileNetV2 ──
    mn_model   = build_mobilenet(IMAGE_SIZE + (3,))
    mn_history = train_model(mn_model, "MobileNetV2", train_gen, val_gen)
    plot_history(mn_history, "MobileNetV2")
    mn_metrics, mn_cm = evaluate_model(mn_model, "MobileNetV2", val_gen)
    plot_confusion_matrix(mn_cm, "MobileNetV2")

    # ── EfficientNetB0 ──
    en_model   = build_efficientnet(IMAGE_SIZE + (3,))
    en_history = train_model(en_model, "EfficientNetB0", train_gen, val_gen)
    plot_history(en_history, "EfficientNetB0")
    en_metrics, en_cm = evaluate_model(en_model, "EfficientNetB0", val_gen)
    plot_confusion_matrix(en_cm, "EfficientNetB0")

    # ── Comparison ──
    plot_comparison(mn_metrics, en_metrics)

    # ── Summary ──
    print("\n" + "="*55)
    print("  TRAINING COMPLETE — SUMMARY")
    print("="*55)
    for name, m in [("MobileNetV2", mn_metrics), ("EfficientNetB0", en_metrics)]:
        print(f"\n  {name}")
        print(f"    Accuracy  : {m['accuracy']:.4f}")
        print(f"    Precision : {m['precision']:.4f}")
        print(f"    Recall    : {m['recall']:.4f}")
        print(f"    F1-Score  : {m['f1_score']:.4f}")

    # Save combined summary
    summary = {
        "trained_at": datetime.now().isoformat(),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "image_size": IMAGE_SIZE,
        "MobileNetV2": mn_metrics,
        "EfficientNetB0": en_metrics,
    }
    with open(os.path.join(RESULTS_DIR, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved → results/training_summary.json")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
