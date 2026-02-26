"""
Combined Weighted Cross-Entropy + Dice Loss
=============================================
CE handles pixel-wise classification with class weighting.
Dice directly maximises region overlap (IoU proxy).
Together they yield higher Mean IoU than either alone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


class DiceLoss(nn.Module):
    def __init__(self, num_classes=cfg.NUM_CLASSES, smooth=1.0):
        super().__init__()
        self.C = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        valid = (targets != 255).unsqueeze(1).float()
        t_clean = targets.clone()
        t_clean[targets == 255] = 0
        onehot = F.one_hot(t_clean, self.C).permute(0, 3, 1, 2).float() * valid
        probs = F.softmax(logits, dim=1) * valid
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        card = probs.sum(dims) + onehot.sum(dims)
        return 1 - ((2 * inter + self.smooth) / (card + self.smooth)).mean()


class CombinedLoss(nn.Module):
    def __init__(self, class_weights=None, dice_w=cfg.DICE_WEIGHT,
                 ce_w=cfg.CE_WEIGHT):
        super().__init__()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
        self.dw = dice_w
        self.cw = ce_w

    def forward(self, logits, targets):
        return self.cw * self.ce(logits, targets) + self.dw * self.dice(logits, targets)
