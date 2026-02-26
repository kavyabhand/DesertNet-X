"""
Helper utilities — seeding, early stopping, checkpoints.
"""

import os, random, numpy as np, torch
import config as cfg


def set_seed(seed=cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience=cfg.EARLY_STOP_PATIENCE, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best = -float("inf")
        self.counter = 0
        self.stop = False

    def __call__(self, score):
        if score > self.best + self.delta:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


def save_checkpoint(model, optimizer, epoch, iou, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": iou}, path)


def load_checkpoint(model, path, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("val_iou", 0.0)
