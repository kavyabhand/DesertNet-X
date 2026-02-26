#!/usr/bin/env python3
"""
DesertNet-X — Training Script
==============================
Usage:
    python train.py
    python train.py --epochs 30 --lr 0.0005
"""

import os, sys, time, json, argparse
import numpy as np, torch
from torch.utils.data import DataLoader

import config as cfg
from models.deeplabv3_resnet18 import build_model
from datasets.desert_dataset import build_datasets
from utils.losses import CombinedLoss
from utils.metrics import SegmentationMetrics
from utils.helpers import set_seed, EarlyStopping, save_checkpoint
from utils.visualization import plot_training_curves


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=cfg.MAX_EPOCHS)
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, n = 0.0, len(loader)
    for i, (imgs, masks) in enumerate(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        loss = criterion(model(imgs), masks)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
        if (i + 1) % max(1, n // 5) == 0:
            print(f"  E{epoch} [{i+1}/{n}] loss={loss.item():.4f}")
    return total_loss / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    met = SegmentationMetrics()
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, masks).item()
        met.update(logits.argmax(1), masks)
    return total_loss / len(loader), met.mean_iou(), met.pixel_accuracy(), met.iou_per_class(), met


def main():
    args = parse_args()
    set_seed()
    device = torch.device("cpu")
    print(f"[Device] {device}  [Epochs] {args.epochs}  "
          f"[BS] {args.batch_size}  [LR] {args.lr}")

    train_ds, val_ds, _ = build_datasets()
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True,
                          num_workers=cfg.NUM_WORKERS, drop_last=True)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False,
                        num_workers=cfg.NUM_WORKERS)
    print(f"[Data] train={len(train_ds)}  val={len(val_ds)}")

    model = build_model(cfg.NUM_CLASSES, cfg.PRETRAINED).to(device)
    cw = torch.tensor(cfg.CLASS_WEIGHTS, dtype=torch.float32)
    criterion = CombinedLoss(cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=cfg.SCHEDULER_PATIENCE,
        factor=cfg.SCHEDULER_FACTOR)
    es = EarlyStopping()
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    if args.resume:
        from utils.helpers import load_checkpoint
        load_checkpoint(model, args.resume, optimizer)

    best_iou = 0.0
    t_losses, v_losses, v_ious = [], [], []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_t = time.time()
        tl = train_epoch(model, train_dl, criterion, optimizer, device, epoch)
        vl, miou, pa, cls_iou, met = validate(model, val_dl, criterion, device)
        t_losses.append(tl); v_losses.append(vl); v_ious.append(miou)
        dt = time.time() - ep_t
        print(f"Epoch {epoch}/{args.epochs}  TL={tl:.4f}  VL={vl:.4f}  "
              f"PA={pa:.4f}  mIoU={miou:.4f}  ({dt:.0f}s)")
        for i, n in enumerate(cfg.CLASS_NAMES):
            print(f"  {n:<20s} {cls_iou[i]:.4f}")

        if miou > best_iou:
            best_iou = miou
            save_checkpoint(model, optimizer, epoch, miou,
                            os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth"))

        save_checkpoint(model, optimizer, epoch, miou,
                        os.path.join(cfg.CHECKPOINT_DIR, "latest_model.pth"))
        scheduler.step(miou)
        if es(miou):
            print(f"Early stopping at epoch {epoch}")
            break

    total_min = (time.time() - t0) / 60
    print(f"\nDone — {total_min:.1f} min, best mIoU={best_iou:.4f}")

    plot_training_curves(t_losses, v_losses, v_ious,
                         os.path.join(cfg.RESULTS_DIR, "loss_curve.png"))

    log = {"total_minutes": round(total_min, 2), "best_iou": round(best_iou, 4),
           "epochs_trained": len(t_losses),
           "train_losses": [round(x, 4) for x in t_losses],
           "val_losses": [round(x, 4) for x in v_losses],
           "val_ious": [round(x, 4) for x in v_ious],
           "config": {"lr": args.lr, "bs": args.batch_size,
                      "backbone": "resnet18", "img_size": 256,
                      "loss": "WCE+Dice"}}
    with open(os.path.join(cfg.RESULTS_DIR, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print("Run `python test.py` next.")


if __name__ == "__main__":
    main()
