"""Parameters, FP16 weight size and inference latency of the five models (Table VI).

Batch 1, 1x3x224x224 input, FP32 weights with FP16 autocast; median and 90th percentile of 60 timed passes
after 15 warm-up passes. Weights do not affect these numbers, so the architectures are instantiated without
checkpoints. Close other programs that use the GPU before running: they inflate the latency of small models.
Writes: <RESULTS_DIR>/efficiency.json
"""
import os
import json
import time

import numpy as np
import torch

from common import build_model
from config import RESULTS_DIR
from models import RETFoundSeg, DINOv2Seg, ViTUNetr

MODELS = ["U-Net (ResNet-34)", "RETFound + SETR-PUP", "DINOv2 + SETR-PUP", "RETFound + UNETR", "DINOv2 + UNETR"]


def params_M(model):
    tot = sum(p.numel() for p in model.parameters()) / 1e6
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return tot, tr


@torch.no_grad()
def bench(model, n_warm=15, n_run=60):
    model.eval()
    x = torch.randn(1, 3, 224, 224, device="cuda")
    for _ in range(n_warm):
        with torch.cuda.amp.autocast():
            model(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_run):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.cuda.amp.autocast():
            model(x)
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1000), float(np.percentile(ts, 90) * 1000)


def make(name):
    if name == "U-Net (ResNet-34)":
        return build_model(n_classes=6)
    if name == "RETFound + SETR-PUP":
        return RETFoundSeg()
    if name == "DINOv2 + SETR-PUP":
        return DINOv2Seg()
    if name == "RETFound + UNETR":
        return ViTUNetr("ret")
    if name == "DINOv2 + UNETR":
        return ViTUNetr("dino")
    raise ValueError(name)


def main():
    dev = "cuda"
    out = {}
    for name in MODELS:
        model = make(name).to(dev)
        tot, tr = params_M(model)
        weights_mb = sum(p.numel() for p in model.parameters()) * 2 / 1e6
        med, p90 = bench(model)
        out[name] = {"params_total_M": tot, "params_trainable_M": tr, "weights_fp16_MB": weights_mb,
                     "latency_ms_median": med, "latency_ms_p90": p90}
        print(f"{name:20s} | total {tot:7.2f} M | trainable {tr:6.2f} M | {weights_mb:6.1f} MB | "
              f"{med:6.2f} ms (p90 {p90:.2f})")
        del model
        torch.cuda.empty_cache()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "efficiency.json"), "w", encoding="utf-8") as f:
        json.dump({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                   "input": "1x3x224x224, batch 1, FP16 autocast", "models": out}, f, indent=2)


if __name__ == "__main__":
    main()
