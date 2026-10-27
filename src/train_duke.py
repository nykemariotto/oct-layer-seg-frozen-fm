"""Train one model on one Duke Chiu 2015 fold (low-data regime, 224x224, 10 classes).

Usage: python train_duke.py <fold> <unet|unet_frozen|ret|dino>
  unet         U-Net, ResNet-34 encoder fully trained
  unet_frozen  U-Net with the ImageNet ResNet-34 encoder frozen (only the decoder is trained)
  ret / dino   frozen RETFound / DINOv2 + SETR-PUP decoder
Writes: <WORK_DIR>/checkpoints/duke_<model>_fold<fold>.pth, <RESULTS_DIR>/folds/duke_<model>_fold<fold>.json
"""
import os
import sys
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import set_seed, build_model, CEDiceLoss
from config import CKPT_DIR, RESULTS_DIR
from duke import (get_fold_subjects, items_for, DukeDS, compute_class_weights, ev, di, bound,
                  N_CLASSES, SIZE, CLASS_NAMES)
from models import RETFoundSeg, DINOv2Seg

MAX_EPOCHS = 120
PATIENCE = 30
NAMES = {"unet": "unet", "unet_frozen": "unet_frozen", "ret": "retfound_setr", "dino": "dinov2_setr"}


def main():
    fold = int(sys.argv[1])
    which = sys.argv[2]
    assert which in NAMES, __doc__
    set_seed(42)
    dev = "cuda"
    sp = get_fold_subjects(fold)
    tr = items_for(sp["train"])
    trl = DataLoader(DukeDS(tr, True), batch_size=8, shuffle=True, num_workers=0)
    val = DataLoader(DukeDS(items_for(sp["val"]), False), batch_size=8, shuffle=False, num_workers=0)
    tel = DataLoader(DukeDS(items_for(sp["test"]), False), batch_size=8, shuffle=False, num_workers=0)

    if which == "ret":
        model = RETFoundSeg(n_classes=N_CLASSES).to(dev)
        params = model.decoder.parameters()
        lr = 1e-3
    elif which == "dino":
        model = DINOv2Seg(n_classes=N_CLASSES).to(dev)
        params = model.decoder.parameters()
        lr = 1e-3
    elif which == "unet_frozen":
        model = build_model(n_classes=N_CLASSES).to(dev)
        for p in model.encoder.parameters():
            p.requires_grad = False
        params = [p for p in model.parameters() if p.requires_grad]
        lr = 1e-3
    else:
        model = build_model(n_classes=N_CLASSES).to(dev)
        params = model.parameters()
        lr = 3e-4
    w = compute_class_weights(tr).to(dev)
    loss_fn = CEDiceLoss(weight=w)
    opt = torch.optim.AdamW(params, lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    os.makedirs(CKPT_DIR, exist_ok=True)
    ck = os.path.join(CKPT_DIR, f"duke_{NAMES[which]}_fold{fold}.pth")
    best, bep, wait = -1.0, 0, 0
    t0 = time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()   # the frozen ViT encoders stay in eval mode
        if which == "unet_frozen":
            model.encoder.eval()   # also freezes the BatchNorm statistics of the ImageNet encoder
        for imgs, masks in trl:
            imgs, masks = imgs.to(dev), masks.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = loss_fn(model(imgs), masks)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        inter, ps, ts = ev(model, val, dev)
        vd, _ = di(inter, ps, ts)
        vm = float(np.nanmean(vd))
        if vm > best:
            best, bep, wait = vm, ep, 0
            torch.save((model.decoder if which in ("ret", "dino") else model).state_dict(), ck)
        else:
            wait += 1
        if wait >= PATIENCE:
            break
    dt = time.time() - t0

    if which in ("ret", "dino"):
        model.decoder.load_state_dict(torch.load(ck))
    else:
        model.load_state_dict(torch.load(ck))
    inter, ps, ts = ev(model, tel, dev)
    td, tiou = di(inter, ps, ts)
    bd = bound(model, tel, dev)
    nz = lambda x: round(float(x), 4) if not np.isnan(x) else None
    res = {
        "model": NAMES[which], "dataset": "Duke", "fold": fold, "size": SIZE,
        "train_time_s": round(dt, 1), "best_epoch": bep, "best_val_mdice": round(best, 4),
        "test_mdice": round(float(np.nanmean(td)), 4),
        "test_mdice_layers_only": round(float(np.nanmean(td[1:8])), 4),
        "test_dice_fluid": nz(td[9]), "test_boundary_um": bd,
        "test_dice_per_class": {CLASS_NAMES[c]: nz(td[c]) for c in range(N_CLASSES)},
        "subjects_test": sp["test"],
    }
    os.makedirs(os.path.join(RESULTS_DIR, "folds"), exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "folds", f"duke_{NAMES[which]}_fold{fold}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"[{which} fold{fold}] mDice {res['test_mdice']} | layers {res['test_mdice_layers_only']} | "
          f"fluid {res['test_dice_fluid']} | boundary {bd} um | {round(dt)} s (best@{bep})")


if __name__ == "__main__":
    main()
