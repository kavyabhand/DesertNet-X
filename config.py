"""
DesertNet-X — Central Configuration
====================================
All hyperparameters, paths, and class definitions.
Optimized for CPU training on MacBook Air M1 (8GB RAM).
"""

import os

# ── Paths ────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train", "images")
TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "train", "masks")
VAL_IMG_DIR = os.path.join(DATA_ROOT, "val", "images")
VAL_MASK_DIR = os.path.join(DATA_ROOT, "val", "masks")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "testImages")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# ── Class Definitions ────────────────────────────────────────
CLASS_MAPPING = {
    100: 0,      # Trees
    200: 1,      # Lush Bushes
    300: 2,      # Dry Grass
    500: 3,      # Dry Bushes
    550: 4,      # Ground Clutter
    600: 5,      # Flowers
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9,    # Sky
}

CLASS_NAMES = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs", "Rocks",
    "Landscape", "Sky",
]

NUM_CLASSES = 10

CLASS_COLORS = [
    (34, 139, 34),    (0, 200, 0),      (210, 180, 80),
    (139, 90, 43),    (160, 120, 90),   (255, 105, 180),
    (101, 67, 33),    (128, 128, 128),  (210, 180, 140),
    (135, 206, 235),
]

# ── Training Hyperparameters ─────────────────────────────────
IMG_SIZE = 256
BATCH_SIZE = 4
NUM_WORKERS = 0
MAX_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5
EARLY_STOP_PATIENCE = 5

DICE_WEIGHT = 0.5
CE_WEIGHT = 0.5

CLASS_WEIGHTS = [
    1.0, 1.5, 0.8, 1.2, 1.0,
    3.0, 3.0, 1.5, 0.6, 0.5,
]

# ── Model ────────────────────────────────────────────────────
BACKBONE = "resnet18"
PRETRAINED = True

# ── Inference ────────────────────────────────────────────────
TTA_ENABLED = True

# ── Reproducibility ──────────────────────────────────────────
SEED = 42

# ── Normalization (ImageNet) ─────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
