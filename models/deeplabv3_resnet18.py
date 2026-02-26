"""
DeepLabV3+ with ResNet18 Backbone
==================================
Lightweight semantic segmentation model.

ResNet18: 11.7M params (vs 44.5M ResNet50) — 2-3x faster on CPU.
ImageNet pretraining compensates for smaller capacity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling — multi-scale context."""

    def __init__(self, in_ch, out_ch=128):
        super().__init__()
        self.b0 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout(0.1))

    def forward(self, x):
        h, w = x.shape[2:]
        feats = [self.b0(x), self.b1(x), self.b2(x), self.b3(x),
                 F.interpolate(self.gap(x), (h, w), mode="bilinear",
                               align_corners=False)]
        return self.project(torch.cat(feats, dim=1))


class Decoder(nn.Module):
    """DeepLabV3+ decoder — fuse ASPP with low-level features."""

    def __init__(self, low_ch, aspp_ch=128, num_classes=10):
        super().__init__()
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_ch, 48, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(
            nn.Conv2d(aspp_ch + 48, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Dropout(0.05))
        self.cls = nn.Conv2d(128, num_classes, 1)

    def forward(self, aspp_feat, low_feat, target_size):
        low = self.low_proj(low_feat)
        up = F.interpolate(aspp_feat, low.shape[2:], mode="bilinear",
                           align_corners=False)
        fused = self.fuse(torch.cat([up, low], dim=1))
        return self.cls(F.interpolate(fused, target_size, mode="bilinear",
                                      align_corners=False))


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ with ResNet18 encoder.
    Input:  (B, 3, H, W)   Output: (B, num_classes, H, W) logits
    """

    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        bb = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1   # 64ch, stride 4
        self.layer2 = bb.layer2   # 128ch
        self.layer3 = bb.layer3   # 256ch
        self.layer4 = bb.layer4   # 512ch, stride 32
        self.aspp = ASPPModule(512, 128)
        self.decoder = Decoder(64, 128, num_classes)

    def forward(self, x):
        size = x.shape[2:]
        x0 = self.stem(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return self.decoder(self.aspp(x4), x1, size)


def build_model(num_classes=10, pretrained=True):
    model = DeepLabV3Plus(num_classes, pretrained)
    n = sum(p.numel() for p in model.parameters())
    print(f"[Model] DeepLabV3+ ResNet18 — {n:,} params")
    return model
