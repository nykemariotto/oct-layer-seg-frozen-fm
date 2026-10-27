"""Train the U-Net baseline on one OCT5k fold.

Usage: python train_unet_oct5k.py <fold> [size]     size 256 = main model, 224 = input-resolution control
Writes: <WORK_DIR>/checkpoints/oct5k_unet<size>_fold<fold>.pth, <RESULTS_DIR>/folds/oct5k_unet<size>_fold<fold>.json
"""
import os
import sys
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import set_seed, build_model, CEDiceLoss
from config import CKPT_DIR, RESULTS_DIR, FIG_DIR
from oct5k import get_fold, items_for, OCT5kDS, compute_w, ev, di, bound, N_CLASSES, CLASS_NAMES

MAX_EPOCHS = 100
PATIENCE = 20


def main():
    fold = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    tag = f"unet{size}"
    umpx = (512.0 / size) * 3.5
    bs = 16 if size <= 256 else (8 if size <= 384 else 4)   # fits an 8 GB GPU
    set_seed(42)
    dev = "cuda"
    sp = get_fold(fold)
    print(f"OCT5k fold {fold} size {size}: train {len(sp['train'])} / val {len(sp['val'])} / test {len(sp['test'])} volumes")

    tr_items = items_for(sp["train"])
    trl = DataLoader(OCT5kDS(tr_items, True, size), batch_size=bs, shuffle=True, num_workers=0)
    val = DataLoader(OCT5kDS(items_for(sp["val"]), False, size), batch_size=bs, shuffle=False, num_workers=0)
    te_ds = OCT5kDS(items_for(sp["test"]), False, size)
    tel = DataLoader(te_ds, batch_size=bs, shuffle=False, num_workers=0)

    model = build_model(n_classes=N_CLASSES).to(dev)
    w = compute_w(tr_items).to(dev)
    print("class_weights:", {CLASS_NAMES[c]: round(float(w[c]), 2) for c in range(N_CLASSES)})
    loss_fn = CEDiceLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    os.makedirs(CKPT_DIR, exist_ok=True)
    ck = os.path.join(CKPT_DIR, f"oct5k_{tag}_fold{fold}.pth")
    best, bep, wait = -1.0, 0, 0
    hist = {"loss": [], "vm": []}
    t0 = time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        tot, nb = 0.0, 0
        for imgs, masks in trl:
            imgs, masks = imgs.to(dev), masks.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                l = loss_fn(model(imgs), masks)
            scaler.scale(l).backward()
            scaler.step(opt)
            scaler.update()
            tot += l.item(); nb += 1
        sched.step()
        inter, ps, ts = ev(model, val, dev)
        vd, _ = di(inter, ps, ts)
        vm = float(np.nanmean(vd))
        hist["loss"].append(tot / nb); hist["vm"].append(vm)
        if vm > best:
            best, bep, wait = vm, ep, 0
            torch.save(model.state_dict(), ck)
        else:
            wait += 1
        if ep == 1 or ep % 5 == 0:
            print(f"ep {ep:3d} | loss {tot/nb:.4f} | val_mDice {vm:.4f} | best {best:.4f}@{bep}")
        if wait >= PATIENCE:
            print(f"early stop @ ep {ep} (best {best:.4f}@{bep})")
            break
    dt = time.time() - t0

    model.load_state_dict(torch.load(ck))
    inter, ps, ts = ev(model, tel, dev)
    td, tiou = di(inter, ps, ts)
    bd = bound(model, tel, dev, umpx)
    nz = lambda x: round(float(x), 4) if not np.isnan(x) else None
    res = {
        "model": "U-Net", "dataset": "OCT5k", "fold": fold, "size": size, "um_per_px": round(umpx, 3),
        "train_time_s": round(dt, 1), "best_epoch": bep, "best_val_mdice": round(best, 4),
        "test_dice_per_class": {CLASS_NAMES[c]: nz(td[c]) for c in range(N_CLASSES)},
        "test_mdice": round(float(np.nanmean(td)), 4), "test_miou": round(float(np.nanmean(tiou)), 4),
        "test_mdice_layers_only": round(float(np.nanmean(td[1:5])), 4),
        "test_boundary_error_um": bd,
        "class_weights": {CLASS_NAMES[c]: round(float(w[c]), 3) for c in range(N_CLASSES)},
        "subjects": sp,
    }
    os.makedirs(os.path.join(RESULTS_DIR, "folds"), exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "folds", f"oct5k_{tag}_fold{fold}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("\n" + json.dumps(res, indent=2))

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["loss"]); ax[0].set_title("train loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(hist["vm"]); ax[1].axvline(bep - 1, ls="--", c="r")
    ax[1].set_title(f"val mDice (best {best:.3f}@{bep})"); ax[1].set_xlabel("epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"curves_oct5k_{tag}_fold{fold}.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
