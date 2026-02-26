#!/usr/bin/env python3
"""
DesertNet-X — Inference Benchmark
==================================
Measures latency, FPS, model size on CPU.

Usage:  python inference.py
"""

import os, time, argparse, numpy as np, torch
import config as cfg
from models.deeplabv3_resnet18 import build_model
from utils.helpers import set_seed, load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=os.path.join(cfg.CHECKPOINT_DIR,
                                                         "best_model.pth"))
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()
    set_seed()

    model = build_model(cfg.NUM_CLASSES, pretrained=False)
    if os.path.exists(args.checkpoint):
        load_checkpoint(model, args.checkpoint)
    model.eval()

    x = torch.randn(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE)
    with torch.no_grad():
        for _ in range(args.warmup):
            model(x)

    lats = []
    with torch.no_grad():
        for _ in range(args.runs):
            t = time.perf_counter()
            model(x)
            lats.append((time.perf_counter() - t) * 1000)

    lats = np.array(lats)
    n_params = sum(p.numel() for p in model.parameters())
    mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

    print("=" * 55)
    print("INFERENCE BENCHMARK")
    print("=" * 55)
    print(f"Input        : {cfg.IMG_SIZE}×{cfg.IMG_SIZE}")
    print(f"Runs         : {len(lats)}")
    print(f"Mean latency : {lats.mean():.2f} ms")
    print(f"Median       : {np.median(lats):.2f} ms")
    print(f"Std          : {lats.std():.2f} ms")
    print(f"P95          : {np.percentile(lats, 95):.2f} ms")
    fps = 1000 / lats.mean()
    print(f"FPS          : {fps:.2f}")
    print(f"Params       : {n_params:,}")
    print(f"Size         : {mb:.2f} MB")
    print("=" * 55)

    # Write results for downstream report
    out = {"mean_ms": round(lats.mean(), 2),
           "median_ms": round(np.median(lats), 2),
           "fps": round(fps, 2),
           "params": n_params,
           "size_mb": round(mb, 2)}
    import json
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(cfg.RESULTS_DIR, "inference_bench.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
