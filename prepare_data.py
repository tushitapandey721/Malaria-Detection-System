"""
prepare_data.py
===============
Splits the raw NIH malaria dataset into train/validation sets.

Expected input structure:
    cell_images/
        Parasitized/    ← raw images
        Uninfected/     ← raw images

Output structure:
    cell_images/
        train/
            Parasitized/
            Uninfected/
        validation/
            Parasitized/
            Uninfected/

Usage:
    python prepare_data.py
"""

import os
import shutil
import random
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────────
RAW_DIR      = "cell_images"
TRAIN_SPLIT  = 0.80          # 80% train, 20% validation
RANDOM_SEED  = 42
CLASSES      = ["Parasitized", "Uninfected"]


def prepare():
    random.seed(RANDOM_SEED)

    train_dir = os.path.join(RAW_DIR, "train")
    val_dir   = os.path.join(RAW_DIR, "validation")

    # Check if already split
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        print("Dataset already split. Skipping.")
        print(f"  Train : {train_dir}")
        print(f"  Val   : {val_dir}")
        return

    print("Splitting dataset into train / validation ...\n")

    total_train = 0
    total_val   = 0

    for cls in CLASSES:
        src = os.path.join(RAW_DIR, cls)

        if not os.path.exists(src):
            raise FileNotFoundError(
                f"Expected folder not found: {src}\n"
                f"Make sure cell_images/Parasitized and cell_images/Uninfected exist."
            )

        # Collect all image files
        exts  = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        files = [f for f in os.listdir(src)
                 if Path(f).suffix.lower() in exts]
        random.shuffle(files)

        n_train = int(len(files) * TRAIN_SPLIT)
        train_files = files[:n_train]
        val_files   = files[n_train:]

        # Create destination dirs
        train_cls_dir = os.path.join(train_dir, cls)
        val_cls_dir   = os.path.join(val_dir,   cls)
        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(val_cls_dir,   exist_ok=True)

        # Copy files
        for f in train_files:
            shutil.copy2(os.path.join(src, f), os.path.join(train_cls_dir, f))
        for f in val_files:
            shutil.copy2(os.path.join(src, f), os.path.join(val_cls_dir, f))

        print(f"  {cls}")
        print(f"    Total  : {len(files)}")
        print(f"    Train  : {len(train_files)}")
        print(f"    Val    : {len(val_files)}")

        total_train += len(train_files)
        total_val   += len(val_files)

    print(f"\n  Grand total  — Train: {total_train}  |  Val: {total_val}")
    print("\nDataset split complete.")


if __name__ == "__main__":
    prepare()
