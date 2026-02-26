# DesertNet-X

> CPU-optimized semantic segmentation for the **Hack For Green Bharat – Duality AI Offroad Segmentation** challenge.

DeepLabV3+ with ResNet18 backbone · Weighted CE + Dice loss · Horizontal-flip TTA · ~8 FPS on M1 CPU

---

## Architecture

```
Input (3×256×256)
  → ResNet18 Encoder (ImageNet pretrained)
    → ASPP (rates 1, 6, 12, 18 + global pooling)
      → Decoder (low-level fusion from layer1)
        → 10-class segmentation map
```

| Property | Value |
|----------|-------|
| Parameters | 13,515,690 |
| Model Size | 51.56 MB |
| Backbone | ResNet18 |
| Input Size | 256 × 256 |
| Inference (CPU) | ~120 ms / image |

---

## Dataset Structure

Place the competition dataset as follows:

```
data/
├── train/
│   ├── images/        # RGB .png/.jpg
│   └── masks/         # Segmentation masks (pixel values: 100–10000)
├── val/
│   ├── images/
│   └── masks/
└── testImages/        # Test RGB images (NO masks, NOT used in training)
```

**Classes:** Trees (100) · Lush Bushes (200) · Dry Grass (300) · Dry Bushes (500) · Ground Clutter (550) · Flowers (600) · Logs (700) · Rocks (800) · Landscape (7100) · Sky (10000)

---

## Setup

```bash
# 1. Clone
git clone <repo-url> && cd DesertNet-X

# 2. Virtual environment (recommended)
python3 -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place data
#    Copy dataset into data/ following the structure above.
#    Update DATA_ROOT in config.py if path differs.
```

**Requirements:** Python 3.9+ · PyTorch ≥ 2.0 · torchvision · albumentations · matplotlib · numpy · Pillow

---

## Training

```bash
python train.py                            # Default: 20 epochs, batch 4, lr 1e-3
python train.py --epochs 30 --lr 0.0005    # Custom
python train.py --resume checkpoints/latest_model.pth   # Resume
```

- Seeds are fixed for reproducibility.
- Best model saved to `checkpoints/best_model.pth` on highest val mIoU.
- Early stopping (patience 5) halts if no improvement.
- Loss curve saved to `results/loss_curve.png`.

**Expected training time:** ~60–120 minutes on M1 CPU (20 epochs).

---

## Testing

```bash
python test.py                 # Val metrics + test predictions + TTA
python test.py --no_tta        # Without TTA
python test.py --val_only      # Validation only
```

Outputs:
- Validation metrics (mIoU, pixel accuracy, per-class IoU)
- Confusion matrix → `results/confusion_matrix.png`
- Test predictions → `results/test_masks/` (raw-value masks for submission)
- Sample overlays → `results/sample_predictions/`
- Failure cases → `results/sample_predictions/failure_*.png`

---

## Inference Benchmark

```bash
python inference.py            # Latency, FPS, model size
```

---

## Analysis

```bash
python analyze.py              # Class distribution, per-class IoU, confusion matrix
```

---

## How to Reproduce Results

```bash
pip install -r requirements.txt
# Place dataset in data/
python train.py --epochs 20
python test.py
python analyze.py
python inference.py
# All results in results/
```

---

## Expected Hardware

| Component | Spec |
|-----------|------|
| Machine | MacBook Air M1 |
| RAM | 8 GB |
| GPU | None (CPU only) |
| Training time | ~60–120 min (20 epochs) |

---

## Results

See [results/metrics_summary.md](results/metrics_summary.md) for full metrics table.

| Output | Location |
|--------|----------|
| Metrics summary | `results/metrics_summary.md` |
| Loss curve | `results/loss_curve.png` |
| Confusion matrix | `results/confusion_matrix.png` |
| Per-class IoU | `results/per_class_iou.png` |
| Sample predictions | `results/sample_predictions/` |
| Training log | `results/training_log.json` |
| Inference benchmark | `results/inference_bench.json` |

## How to Regenerate Metrics

```bash
python train.py --epochs 20     # Train
python test.py                  # Validate + predict
python analyze.py               # Visualize
python inference.py             # Benchmark
```

All results are written to `results/`. Checkpoints are saved locally to `checkpoints/` (excluded from repo).

---

## Project Structure

```
DesertNet-X/
├── models/
│   └── deeplabv3_resnet18.py    # DeepLabV3+ architecture
├── datasets/
│   └── desert_dataset.py        # Dataset, augmentations, remapping
├── utils/
│   ├── losses.py                # Weighted CE + Dice
│   ├── metrics.py               # IoU, pixel accuracy, confusion matrix
│   ├── visualization.py         # Overlays, plots, curves
│   └── helpers.py               # Seeding, checkpoints, early stopping
├── results/
│   ├── metrics_summary.md       # Full metrics table
│   ├── confusion_matrix.png     # Generated confusion matrix
│   ├── loss_curve.png           # Training/val loss curves
│   └── sample_predictions/      # Visual outputs
├── config.py                    # Central configuration
├── train.py                     # Training pipeline
├── test.py                      # Validation + TTA inference
├── inference.py                 # Speed benchmark
├── analyze.py                   # Post-training analysis
├── requirements.txt             # Dependencies
├── README.md                    # This file
└── report_summary.md            # Technical report
```
