# MalariaNet — Malaria Detection using Deep Learning

Automated classification of microscopic blood smear cell images as **Parasitized** or **Uninfected** using transfer learning with MobileNetV2 and EfficientNetB0.

---

## Project Structure

```
malaria_detection/
├── prepare_data.py       # Split raw dataset → train/validation
├── train.py              # Train MobileNetV2 + EfficientNetB0
├── predict.py            # CLI inference on single image or folder
├── app.py                # Flask web app (demo link)
├── requirements.txt
├── cell_images/          # Dataset (not included in repo)
│   ├── Parasitized/      # Raw images (OR pre-split below)
│   ├── Uninfected/
│   ├── train/
│   └── validation/
├── models/               # Saved .keras models (after training)
├── results/              # Metrics JSON, plots, CSV logs
├── static/uploads/       # Flask uploaded images
└── templates/index.html  # Flask UI
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare dataset
```bash
# If you have raw cell_images/Parasitized/ and cell_images/Uninfected/:
python prepare_data.py

# If already split into train/validation — skip this step
```

### 3. Train the models
```bash
python train.py
```
This trains **MobileNetV2** and **EfficientNetB0**, saves models to `models/`, and outputs plots and metrics to `results/`.

### 4. Run the web demo
```bash
python app.py
# Open http://localhost:5000
```

### 5. CLI inference
```bash
# Single image
python predict.py --image path/to/cell.png

# Entire folder
python predict.py --folder path/to/images/ --output results.json

# Use EfficientNetB0
python predict.py --image cell.png --model EfficientNetB0
```

---

## Dataset

- **Source:** [NIH Cell Image Library](https://ceb.nlm.nih.gov/proj/malaria/cell_images.zip)
- **Size:** 27,558 images — 13,779 Parasitized + 13,779 Uninfected
- **Split:** 80% train / 20% validation

---

## Model Architecture

### MobileNetV2 (Best Model)
- Base: MobileNetV2 pretrained on ImageNet
- Head: GlobalAveragePooling2D → Dropout(0.3) → Dense(1, sigmoid)
- Optimizer: Adam (lr=1e-4)
- Loss: Binary cross-entropy

### EfficientNetB0
- Base: EfficientNetB0 pretrained on ImageNet
- Fix applied: Rescaling(255.0) layer before base model (EfficientNet expects [0,255])
- Same head and optimizer as MobileNetV2

---

## Results

| Model | Train Acc | Val Acc | Val Loss |
|---|---|---|---|
| MobileNetV2 | 90.47% | **87.12%** | 0.3370 |
| EfficientNetB0 | ~50% | ~50% | ~0.693 |

MobileNetV2 converged successfully. EfficientNetB0 failed in the original notebook due to incorrect input preprocessing (the fix is applied in this codebase).

---

## API Endpoints (Flask)

| Method | Route | Description |
|---|---|---|
| GET | `/` | Web UI |
| POST | `/api/predict` | Predict from uploaded file |
| POST | `/api/predict_base64` | Predict from base64 image |
| GET | `/api/models` | List available models |
| GET | `/api/results` | Get training summary |
| GET | `/health` | Health check |

---

## Tech Stack

- Python 3.10+
- TensorFlow / Keras
- Flask
- NumPy, Pillow
- scikit-learn
- Matplotlib, Seaborn

---

## References

- Rajaraman et al. (2018). Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection. *PeerJ*.
- NIH National Library of Medicine — [Malaria Datasets](https://ceb.nlm.nih.gov/proj/malaria/)
