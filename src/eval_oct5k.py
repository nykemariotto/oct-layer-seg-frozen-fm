"""Per-patient metrics on OCT5k for the six models of the paper (inference only, from the fold checkpoints).

OCT5k has one volume per subject, so each row is one patient tested by the fold that held it out.
Metrics per volume, pooled over its B-scans:
  dice_c0..c5, mdice (6 classes), mdice_layers (classes 1-4)
  surf1..surf5_px   mean absolute interface displacement per A-scan, in native mask pixels (512 rows)
  boundary_mean_px  mean of the five interfaces
  thickness_mae_px  mean absolute error of the ILM-to-OBRPE thickness (classes 1-4), column by column
Pixels are converted to micrometres in stats_oct5k.py.

Usage: python eval_oct5k.py [model ...]        (default: all six)
Writes: <RESULTS_DIR>/oct5k_per_patient_metrics.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import set_seed, build_model
from config import CKPT_DIR, RESULTS_DIR
from models import RETFoundSeg, DINOv2Seg, ViTUNetr
from oct5k import SEG_DIR, SPLITS, OCT5kDS, N_CLASSES, N_SURF

OUT_CSV = os.path.join(RESULTS_DIR, "oct5k_per_patient_metrics.csv")
MASK_SIZE = 512
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# name -> (checkpoint pattern, working resolution, constructor).
# "U-Net@224" is the input-resolution control: the U-Net trained at the frozen encoders' 224x224.
MODELS = {
    "U-Net":             ("oct5k_unet256_fold{f}.pth", 256, "unet"),
    "U-Net@224":         ("oct5k_unet224_fold{f}.pth", 224, "unet"),
    "RETFound+SETR-PUP": ("oct5k_retfound_setr_fold{f}.pth", 224, "ret_setr"),
    "DINOv2+SETR-PUP":   ("oct5k_dinov2_setr_fold{f}.pth", 224, "dino_setr"),
    "RETFound+UNETR":    ("oct5k_retfound_unetr_fold{f}.pth", 224, "ret_unetr"),
    "DINOv2+UNETR":      ("oct5k_dinov2_unetr_fold{f}.pth", 224, "dino_unetr"),
}


def build(kind):
    """Built once per model: the frozen encoder is identical across folds, only the decoder changes."""
    if kind == "unet":
        return build_model(n_classes=N_CLASSES, encoder="resnet34").to(DEV)
    if kind == "ret_setr":
        return RETFoundSeg(n_classes=N_CLASSES).to(DEV)
    if kind == "dino_setr":
        return DINOv2Seg(n_classes=N_CLASSES).to(DEV)
    if kind in ("ret_unetr", "dino_unetr"):
        return ViTUNetr("ret" if kind == "ret_unetr" else "dino", n_classes=N_CLASSES).to(DEV)
    raise ValueError(kind)


def load_fold_weights(model, kind, ckpt):
    """U-Net checkpoints hold the whole model; foundation-model checkpoints hold only the decoder."""
    sd = torch.load(ckpt, map_location=DEV, weights_only=False)
    if kind == "unet":
        model.load_state_dict(sd)
    else:
        model.decoder.load_state_dict(sd)
    return model.eval()


@torch.no_grad()
def eval_volume(model, items, size):
    loader = DataLoader(OCT5kDS(items, train=False, size=size), batch_size=8, shuffle=False, num_workers=0)
    inter = np.zeros(N_CLASSES); ps = np.zeros(N_CLASSES); ts = np.zeros(N_CLASSES)
    surf_err = {i: [] for i in range(1, N_SURF + 1)}
    thick_err = []
    n = 0

    for imgs, masks in loader:
        with torch.cuda.amp.autocast(enabled=(DEV == "cuda")):
            pred = model(imgs.to(DEV)).argmax(1).cpu()
        pred_np, gt_np = pred.numpy(), masks.numpy()
        n += gt_np.shape[0]

        for c in range(N_CLASSES):
            p, t = pred == c, masks == c
            inter[c] += (p & t).sum().item(); ps[c] += p.sum().item(); ts[c] += t.sum().item()

        for b in range(gt_np.shape[0]):
            g, p = gt_np[b], pred_np[b]
            # interface i separates class i-1 from class i; its position is the number of pixels above it
            for i in range(1, N_SURF + 1):
                surf_err[i].extend(np.abs((p < i).sum(0) - (g < i).sum(0)).tolist())
            tp = ((p >= 1) & (p <= 4)).sum(0)
            tg = ((g >= 1) & (g <= 4)).sum(0)
            thick_err.extend(np.abs(tp - tg).tolist())

    den = ps + ts
    dice = np.where(den > 0, 2 * inter / np.maximum(den, 1), np.nan)
    k = MASK_SIZE / size          # working pixels -> native mask pixels
    row = {"n_bscans": n,
           "mdice": float(np.nanmean(dice)),
           "mdice_layers": float(np.nanmean(dice[1:5])),
           "thickness_mae_px": float(np.mean(thick_err)) * k}
    row.update({f"dice_c{c}": float(dice[c]) for c in range(N_CLASSES)})
    row.update({f"surf{i}_px": float(np.mean(surf_err[i])) * k for i in range(1, N_SURF + 1)})
    row["boundary_mean_px"] = float(np.mean([row[f"surf{i}_px"] for i in range(1, N_SURF + 1)]))
    return row


def main():
    targets = sys.argv[1:] or list(MODELS)
    unknown = [m for m in targets if m not in MODELS]
    if unknown:
        sys.exit(f"unknown model(s): {unknown}; available: {list(MODELS)}")

    set_seed(42)
    splits = pd.read_csv(SPLITS)
    meta = pd.read_csv(os.path.join(SEG_DIR, "metadata.csv"))
    disease = dict(zip(splits.vol_id, splits.disease))

    rows = []
    for name in targets:
        pattern, size, kind = MODELS[name]
        t0 = time.time()
        print(f"\n=== {name} (@{size}) ===", flush=True)
        model = build(kind)
        for fold in range(5):
            ckpt = os.path.join(CKPT_DIR, pattern.format(f=fold))
            if not os.path.exists(ckpt):
                print(f"  fold {fold}: missing {os.path.basename(ckpt)}, skipped", flush=True)
                continue
            load_fold_weights(model, kind, ckpt)
            vols = splits[(splits.fold == fold) & (splits.role == "test")].vol_id.tolist()
            for vid in vols:
                sub = meta[meta.vol_id == vid]
                r = eval_volume(model, list(zip(sub.img_path, sub.mask_path)), size)
                r.update({"model": name, "fold": fold, "vol_id": vid, "disease": disease.get(vid), "size": size})
                rows.append(r)
            print(f"  fold {fold}: {len(vols)} volumes "
                  f"(mean mDice {np.mean([l['mdice'] for l in rows[-len(vols):]]):.4f})", flush=True)
        del model
        torch.cuda.empty_cache()
        print(f"  -> {time.time() - t0:.0f} s", flush=True)

    df = pd.DataFrame(rows)
    cols = ["model", "fold", "vol_id", "disease", "size", "n_bscans", "mdice", "mdice_layers"]
    df = df[cols + [c for c in df.columns if c not in cols]]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwritten: {OUT_CSV} ({len(df)} rows = {df.model.nunique()} models x {df.vol_id.nunique()} volumes)")
    print(df.groupby("model")[["mdice", "mdice_layers", "boundary_mean_px", "thickness_mae_px"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
