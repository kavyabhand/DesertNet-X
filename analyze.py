#!/usr/bin/env python3
"""
DesertNet-X — Post-training Analysis
======================================
Class distribution, per-class IoU, confusion matrix, histograms.

Usage:  python analyze.py
"""

import os, json, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image

import config as cfg
from models.deeplabv3_resnet18 import build_model
from datasets.desert_dataset import DesertSegDataset, get_val_transforms, remap_mask
from utils.metrics import SegmentationMetrics
from utils.helpers import set_seed, load_checkpoint
from utils.visualization import plot_confusion_matrix, save_comparison, plot_training_curves


def class_distribution():
    if not os.path.isdir(cfg.TRAIN_MASK_DIR):
        print("[Skip] Training masks not found")
        return
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = sorted(f for f in os.listdir(cfg.TRAIN_MASK_DIR)
                   if os.path.splitext(f)[1].lower() in exts)
    counts = np.zeros(cfg.NUM_CLASSES, dtype=np.int64)
    for fn in files:
        m = remap_mask(np.array(Image.open(os.path.join(cfg.TRAIN_MASK_DIR, fn))))
        for c in range(cfg.NUM_CLASSES):
            counts[c] += (m == c).sum()
    total = counts.sum()
    print("\nClass Distribution (train)")
    for i, n in enumerate(cfg.CLASS_NAMES):
        print(f"  {n:<20s} {counts[i]:>10,d}  ({100*counts[i]/max(total,1):.1f}%)")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(cfg.CLASS_NAMES, counts,
           color=[np.array(c)/255 for c in cfg.CLASS_COLORS])
    ax.set_ylabel("Pixels"); ax.set_title("Class Distribution")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(cfg.RESULTS_DIR, "class_dist.png"), dpi=150)
    plt.close()


def model_analysis(model, device):
    ds = DesertSegDataset(cfg.VAL_IMG_DIR, cfg.VAL_MASK_DIR,
                          get_val_transforms())
    dl = DataLoader(ds, cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS)
    met = SegmentationMetrics()
    model.eval()
    with torch.no_grad():
        for imgs, masks in dl:
            met.update(model(imgs.to(device)).argmax(1), masks.to(device))
    print(met.summary())
    iou = met.iou_per_class()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(cfg.CLASS_NAMES, iou,
           color=[np.array(c)/255 for c in cfg.CLASS_COLORS])
    ax.axhline(met.mean_iou(), color="red", ls="--", label=f"mIoU={met.mean_iou():.4f}")
    ax.set_ylim(0, 1); ax.set_ylabel("IoU"); ax.legend()
    ax.set_title("Per-Class IoU")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(cfg.RESULTS_DIR, "per_class_iou.png"), dpi=150)
    plt.close()
    plot_confusion_matrix(met.cm,
        os.path.join(cfg.RESULTS_DIR, "confusion_matrix.png"))


def replay_curves():
    lp = os.path.join(cfg.RESULTS_DIR, "training_log.json")
    if not os.path.exists(lp):
        return
    d = json.load(open(lp))
    plot_training_curves(d["train_losses"], d["val_losses"], d["val_ious"],
                         os.path.join(cfg.RESULTS_DIR, "loss_curve.png"))


def main():
    set_seed()
    device = torch.device("cpu")
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    class_distribution()
    ckpt = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(ckpt):
        model = build_model(cfg.NUM_CLASSES, False).to(device)
        load_checkpoint(model, ckpt)
        model_analysis(model, device)
    else:
        print("[Skip] No checkpoint found")
    replay_curves()
    print(f"\nAll outputs in {cfg.RESULTS_DIR}")


if __name__ == "__main__":
    main()
