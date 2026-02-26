"""
Desert Offroad Segmentation Dataset
=====================================
Handles RGB+mask loading, class remapping, augmentations.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config as cfg


# ── Augmentation pipelines ───────────────────────────────────

def get_train_transforms():
    return A.Compose([
        A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.HueSaturationValue(10, 20, 20, p=0.3),
        A.RandomScale(scale_limit=0.1, p=0.3),
        A.PadIfNeeded(cfg.IMG_SIZE, cfg.IMG_SIZE, border_mode=0),
        A.CenterCrop(cfg.IMG_SIZE, cfg.IMG_SIZE),
        A.Normalize(cfg.MEAN, cfg.STD),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
        A.Normalize(cfg.MEAN, cfg.STD),
        ToTensorV2(),
    ])


# ── Mask remapping ───────────────────────────────────────────

def remap_mask(mask_np):
    """Raw pixel values → contiguous 0..9. Unknown → 255 (ignore)."""
    out = np.full(mask_np.shape[:2], 255, dtype=np.uint8)
    for raw, idx in cfg.CLASS_MAPPING.items():
        if mask_np.ndim == 3:
            out[mask_np[:, :, 0] == raw] = idx
        else:
            out[mask_np == raw] = idx
    return out


# ── Dataset ──────────────────────────────────────────────────

class DesertSegDataset(Dataset):
    """
    Args:
        img_dir:   path to RGB images
        mask_dir:  path to masks (None for test)
        transform: albumentations pipeline
        is_test:   if True, return (image, filename) with no mask
    """

    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, img_dir, mask_dir=None, transform=None, is_test=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_test = is_test
        self.files = sorted(
            f for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in self.EXTS
        )
        assert self.files, f"No images in {img_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = np.array(Image.open(
            os.path.join(self.img_dir, fname)).convert("RGB"))

        if self.is_test:
            if self.transform:
                img = self.transform(image=img)["image"]
            return img, fname

        # Find matching mask
        base = os.path.splitext(fname)[0]
        mask_path = None
        for ext in [".png", ".jpg", ".bmp", ".tif", ".tiff"]:
            p = os.path.join(self.mask_dir, base + ext)
            if os.path.exists(p):
                mask_path = p
                break
        if mask_path is None:
            mask_path = os.path.join(self.mask_dir, fname)

        mask = remap_mask(np.array(Image.open(mask_path)))

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        mask = torch.as_tensor(mask, dtype=torch.long)
        return img, mask


def build_datasets():
    """Return (train_ds, val_ds, test_ds)."""
    train = DesertSegDataset(cfg.TRAIN_IMG_DIR, cfg.TRAIN_MASK_DIR,
                             get_train_transforms())
    val = DesertSegDataset(cfg.VAL_IMG_DIR, cfg.VAL_MASK_DIR,
                           get_val_transforms())
    test = DesertSegDataset(cfg.TEST_IMG_DIR, transform=get_val_transforms(),
                            is_test=True)
    return train, val, test
