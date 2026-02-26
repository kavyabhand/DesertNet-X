#!/usr/bin/env python3
"""
DesertNet-X — Test / Prediction Script
=======================================
Validates on val set, generates predictions on testImages with TTA.

Usage:
    python test.py
    python test.py --no_tta
"""

import os, sys, time, argparse
import numpy as np, torch
from torch.utils.data import DataLoader
from PIL import Image

import config as cfg
from models.deeplabv3_resnet18 import build_model
from datasets.desert_dataset import build_datasets
from utils.metrics import SegmentationMetrics
from utils.helpers import set_seed, load_checkpoint
from utils.visualization import (plot_confusion_matrix, save_comparison,
                                  save_prediction_overlay, denormalize,
                                  mask_to_color)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=os.path.join(cfg.CHECKPOINT_DIR,
                                                         "best_model.pth"))
    p.add_argument("--no_tta", action="store_true")
    p.add_argument("--val_only", action="store_true")
    return p.parse_args()


@torch.no_grad()
def predict_tta(model, img, device):
    """Horizontal-flip TTA — average two predictions."""
    p1 = torch.softmax(model(img.to(device)), 1)
    p2 = torch.softmax(model(torch.flip(img, [3]).to(device)), 1)
    return (p1 + torch.flip(p2, [3])) / 2


@torch.no_grad()
def run_val(model, loader, device):
    model.eval()
    met = SegmentationMetrics()
    imgs_all, gts, preds, img_ious = [], [], [], []
    for images, masks in loader:
        logits = model(images.to(device))
        pred = logits.argmax(1)
        met.update(pred, masks.to(device))
        for i in range(images.shape[0]):
            imgs_all.append(images[i].cpu())
            gts.append(masks[i].cpu().numpy())
            preds.append(pred[i].cpu().numpy())
            m = SegmentationMetrics()
            m.update(pred[i:i+1].cpu(), masks[i:i+1])
            img_ious.append(m.mean_iou())
    print(met.summary())
    return met, imgs_all, gts, preds, img_ious


@torch.no_grad()
def run_test(model, loader, device, tta):
    model.eval()
    pred_dir = os.path.join(cfg.RESULTS_DIR, "test_masks")
    vis_dir = os.path.join(cfg.RESULTS_DIR, "sample_predictions")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    inv = {v: k for k, v in cfg.CLASS_MAPPING.items()}
    count, t0 = 0, time.time()
    for images, fnames in loader:
        if tta:
            probs = predict_tta(model, images, device)
        else:
            probs = torch.softmax(model(images.to(device)), 1)
        pred = probs.argmax(1).cpu().numpy()
        for i in range(len(fnames)):
            base = os.path.splitext(fnames[i])[0]
            # Save raw-value mask for submission
            raw = np.zeros_like(pred[i], dtype=np.uint16)
            for ci, rv in inv.items():
                raw[pred[i] == ci] = rv
            Image.fromarray(raw).save(os.path.join(pred_dir, f"{base}.png"))
            # Save overlay
            if count < 20:
                save_prediction_overlay(images[i], pred[i],
                    os.path.join(vis_dir, f"{base}.png"), fnames[i])
            count += 1
    dt = time.time() - t0
    print(f"[Test] {count} images in {dt:.1f}s "
          f"({dt/max(count,1)*1000:.1f} ms/img)")


def main():
    args = parse_args()
    set_seed()
    device = torch.device("cpu")

    model = build_model(cfg.NUM_CLASSES, pretrained=False).to(device)
    if not os.path.exists(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}\nRun train.py first.")
    load_checkpoint(model, args.checkpoint)
    model.eval()

    _, val_ds, test_ds = build_datasets()

    # ── Validation ──
    val_dl = DataLoader(val_ds, cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS)
    met, imgs, gts, preds, img_ious = run_val(model, val_dl, device)

    # Confusion matrix
    plot_confusion_matrix(met.cm,
                          os.path.join(cfg.RESULTS_DIR, "confusion_matrix.png"))

    # Sample comparisons
    comp_dir = os.path.join(cfg.RESULTS_DIR, "sample_predictions")
    os.makedirs(comp_dir, exist_ok=True)
    for i in range(min(10, len(imgs))):
        save_comparison(imgs[i], gts[i], preds[i],
                        os.path.join(comp_dir, f"val_{i+1}.png"),
                        f"Val {i+1} | IoU={img_ious[i]:.4f}")

    # Failures (bottom 5)
    worst = np.argsort(img_ious)[:5]
    for rank, idx in enumerate(worst):
        save_comparison(imgs[idx], gts[idx], preds[idx],
            os.path.join(comp_dir, f"failure_{rank+1}.png"),
            f"Failure #{rank+1} IoU={img_ious[idx]:.4f}")

    # ── Test predictions ──
    if not args.val_only:
        tta = cfg.TTA_ENABLED and not args.no_tta
        if tta:
            print("[TTA] horizontal-flip enabled")
        test_dl = DataLoader(test_ds, 1, num_workers=cfg.NUM_WORKERS)
        run_test(model, test_dl, device, tta)

    print(f"\nFinal mIoU = {met.mean_iou():.4f}")


if __name__ == "__main__":
    main()
