"""
Segmentation Metrics — Pixel Accuracy, Mean IoU, Per-class IoU
===============================================================
All computed via confusion matrix for efficiency.
"""

import numpy as np
import config as cfg


class SegmentationMetrics:
    def __init__(self, num_classes=cfg.NUM_CLASSES, ignore=255):
        self.C = num_classes
        self.ignore = ignore
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.cm.fill(0)

    def update(self, preds, targets):
        """preds, targets: torch tensors (B,H,W)."""
        p = preds.cpu().numpy().ravel()
        t = targets.cpu().numpy().ravel()
        valid = t != self.ignore
        p, t = np.clip(p[valid], 0, self.C - 1), np.clip(t[valid], 0, self.C - 1)
        self.cm += np.bincount(
            t * self.C + p, minlength=self.C ** 2).reshape(self.C, self.C)

    def pixel_accuracy(self):
        return np.diag(self.cm).sum() / (self.cm.sum() + 1e-10)

    def iou_per_class(self):
        tp = np.diag(self.cm)
        return tp / (self.cm.sum(0) + self.cm.sum(1) - tp + 1e-10)

    def mean_iou(self):
        return self.iou_per_class().mean()

    def summary(self):
        iou = self.iou_per_class()
        lines = ["=" * 50, "SEGMENTATION METRICS", "=" * 50,
                 f"Pixel Accuracy : {self.pixel_accuracy():.4f}",
                 f"Mean IoU       : {self.mean_iou():.4f}",
                 "-" * 50, f"{'Class':<20s}{'IoU':>8s}", "-" * 50]
        for i, n in enumerate(cfg.CLASS_NAMES):
            lines.append(f"{n:<20s}{iou[i]:>8.4f}")
        lines.append("=" * 50)
        return "\n".join(lines)
