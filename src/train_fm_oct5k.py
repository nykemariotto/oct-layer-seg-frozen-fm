"""Train the decoder of a frozen foundation-model encoder on one OCT5k fold (224x224 input).

Usage: python train_fm_oct5k.py <ret|dino> <setr|unetr> <fold>
Writes: <WORK_DIR>/checkpoints/oct5k_<enc>_<dec>_fold<fold>.pth (decoder weights only),
        <RESULTS_DIR>/folds/oct5k_<enc>_<dec>_fold<fold>.json
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

from common import set_seed, CEDiceLoss
from config import CKPT_DIR, RESULTS_DIR, FIG_DIR
from models import RETFoundSeg, DINOv2Seg, ViTUNetr, n_dec_params, SIZE
from oct5k import get_fold, items_for, OCT5kDS, compute_w, ev, di, bound, N_CLASSES, CLASS_NAMES

UM_PER_PX = (512.0 / SIZE) * 3.5
MAX_EPOCHS = 80
PATIENCE = 20
ENCODERS = {"ret": "RETFound", "dino": "DINOv2"}
DECODERS = {"setr": "SETR-PUP", "unetr": "UNETR"}


def build(enc, dec):
    if dec == "setr":
        return RETFoundSeg() if enc == "ret" else DINOv2Seg()
    return ViTUNetr(enc)


def main():
    enc, dec, fold = sys.argv[1], sys.argv[2], int(sys.argv[3])
    assert enc in ENCODERS and dec in DECODERS, __doc__
    tag = f"{ENCODERS[enc].lower()}_{dec}"
    dev = "cuda"
    set_seed(42)
    sp = get_fold(fold)
    print(f"{ENCODERS[enc]} + {DECODERS[dec]}, OCT5k fold {fold} @{SIZE}: "
          f"train {len(sp['train'])} / val {len(sp['val'])} / test {len(sp['test'])} volumes")

    tr_items = items_for(sp["train"])
    trl = DataLoader(OCT5kDS(tr_items, True, SIZE), batch_size=8, shuffle=True, num_workers=0)
    val = DataLoader(OCT5kDS(items_for(sp["val"]), False, SIZE), batch_size=8, shuffle=False, num_workers=0)
    te_ds = OCT5kDS(items_for(sp["test"]), False, SIZE)
    tel = DataLoader(te_ds, batch_size=8, shuffle=False, num_workers=0)

    model = build(enc, dec).to(dev)
    print("decoder params:", round(n_dec_params(model) / 1e6, 2), "M (encoder frozen)")
    w = compute_w(tr_items).to(dev)
    loss_fn = CEDiceLoss(weight=w)
    opt = torch.optim.AdamW(model.decoder.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckp = os.path.join(CKPT_DIR, f"oct5k_{tag}_fold{fold}.pth")
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
                loss = loss_fn(model(imgs), masks)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item(); nb += 1
        sched.step()
        inter, ps, ts = ev(model, val, dev)
        vd, _ = di(inter, ps, ts)
        vm = float(np.nanmean(vd))
        hist["loss"].append(tot / nb); hist["vm"].append(vm)
        if vm > best:
            best, bep, wait = vm, ep, 0
            torch.save(model.decoder.state_dict(), ckp)
        else:
            wait += 1
        if ep == 1 or ep % 5 == 0:
            print(f"ep {ep:3d} | loss {tot/nb:.4f} | val_mDice {vm:.4f} | best {best:.4f}@{bep}")
        if wait >= PATIENCE:
            print(f"early stop @ ep {ep} (best {best:.4f}@{bep})")
            break
    dt = time.time() - t0

    model.decoder.load_state_dict(torch.load(ckp))
    inter, ps, ts = ev(model, tel, dev)
    td, tiou = di(inter, ps, ts)
    bd = bound(model, tel, dev, UM_PER_PX)
    nz = lambda x: round(float(x), 4) if not np.isnan(x) else None
    res = {
        "model": f"{ENCODERS[enc]} (frozen) + {DECODERS[dec]}", "dataset": "OCT5k", "fold": fold, "size": SIZE,
        "decoder_params_M": round(n_dec_params(model) / 1e6, 3),
        "train_time_s": round(dt, 1), "best_epoch": bep, "best_val_mdice": round(best, 4),
        "test_dice_per_class": {CLASS_NAMES[c]: nz(td[c]) for c in range(N_CLASSES)},
        "test_mdice": round(float(np.nanmean(td)), 4), "test_miou": round(float(np.nanmean(tiou)), 4),
        "test_mdice_layers_only": round(float(np.nanmean(td[1:5])), 4),
        "test_boundary_error_um": bd, "subjects": sp,
    }
    os.makedirs(os.path.join(RESULTS_DIR, "folds"), exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "folds", f"oct5k_{tag}_fold{fold}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("\n" + json.dumps({k: res[k] for k in ["model", "decoder_params_M", "test_mdice", "test_mdice_layers_only",
                                                  "test_boundary_error_um", "best_epoch"]}, indent=2))

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["loss"]); ax[0].set_title(f"train loss ({tag})"); ax[0].set_xlabel("epoch")
    ax[1].plot(hist["vm"]); ax[1].axvline(bep - 1, ls="--", c="r")
    ax[1].set_title(f"val mDice (best {best:.3f}@{bep})"); ax[1].set_xlabel("epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"curves_oct5k_{tag}_fold{fold}.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
