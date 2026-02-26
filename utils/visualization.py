"""
Visualization utilities — overlays, confusion matrix, loss curves.
"""

import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import config as cfg


def denormalize(t):
    """(C,H,W) tensor → (H,W,C) uint8 numpy."""
    img = t.numpy() if hasattr(t, "numpy") else t.copy()
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    img = img * np.array(cfg.STD) + np.array(cfg.MEAN)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def mask_to_color(mask):
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(cfg.NUM_CLASSES):
        out[mask == i] = cfg.CLASS_COLORS[i]
    return out


def _legend():
    return [mpatches.Patch(color=np.array(cfg.CLASS_COLORS[i]) / 255,
                           label=cfg.CLASS_NAMES[i])
            for i in range(cfg.NUM_CLASSES)]


def save_comparison(img, gt, pred, path, title=""):
    img_np = denormalize(img) if (img.max() <= 1 or
                                   (hasattr(img, "shape") and img.shape[0] == 3)) else img
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(img_np); ax[0].set_title("Input"); ax[0].axis("off")
    ax[1].imshow(mask_to_color(gt)); ax[1].set_title("GT"); ax[1].axis("off")
    ax[2].imshow(mask_to_color(pred)); ax[2].set_title("Pred"); ax[2].axis("off")
    overlay = (img_np * 0.5 + mask_to_color(pred) * 0.5).astype(np.uint8)
    ax[3].imshow(overlay); ax[3].set_title("Overlay"); ax[3].axis("off")
    plt.legend(handles=_legend(), bbox_to_anchor=(1.05, 1),
               loc="upper left", fontsize=7)
    plt.suptitle(title, fontsize=13)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def save_prediction_overlay(img, pred, path, title=""):
    img_np = denormalize(img) if (img.max() <= 1 or
                                   (hasattr(img, "shape") and img.shape[0] == 3)) else img
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img_np); ax[0].set_title("Input"); ax[0].axis("off")
    ax[1].imshow(mask_to_color(pred)); ax[1].set_title("Pred"); ax[1].axis("off")
    overlay = (img_np * 0.5 + mask_to_color(pred) * 0.5).astype(np.uint8)
    ax[2].imshow(overlay); ax[2].set_title("Overlay"); ax[2].axis("off")
    plt.legend(handles=_legend(), bbox_to_anchor=(1.05, 1),
               loc="upper left", fontsize=7)
    plt.suptitle(title, fontsize=13)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def plot_confusion_matrix(cm, path):
    cm_n = cm.astype(float)
    rs = cm_n.sum(1, keepdims=True)
    cm_n = np.divide(cm_n, rs, where=rs > 0)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(cm_n, cmap="Blues")
    ax.set_xticks(range(cfg.NUM_CLASSES))
    ax.set_yticks(range(cfg.NUM_CLASSES))
    ax.set_xticklabels(cfg.CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cfg.CLASS_NAMES, fontsize=8)
    for i in range(cfg.NUM_CLASSES):
        for j in range(cfg.NUM_CLASSES):
            ax.text(j, i, f"{cm_n[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_n[i, j] > 0.5 else "black", fontsize=7)
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    ax.set_title("Normalized Confusion Matrix")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def plot_training_curves(train_losses, val_losses, val_ious, path):
    epochs = range(1, len(train_losses) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.plot(epochs, train_losses, "b-o", ms=4, label="Train")
    a1.plot(epochs, val_losses, "r-o", ms=4, label="Val")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Loss")
    a1.set_title("Loss Curves"); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(epochs, val_ious, "g-o", ms=4, label="Val mIoU")
    best_e = int(np.argmax(val_ious)) + 1
    a2.axvline(best_e, color="red", ls="--", alpha=0.5)
    a2.set_xlabel("Epoch"); a2.set_ylabel("mIoU")
    a2.set_title("Validation Mean IoU"); a2.legend(); a2.grid(alpha=0.3)
    plt.suptitle("DesertNet-X Training", fontsize=14)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
