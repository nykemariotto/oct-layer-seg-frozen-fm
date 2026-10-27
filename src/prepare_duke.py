"""Duke Chiu 2015: dense 10-class masks from grader 1 and 256x512 PNGs of the 110 annotated B-scans.

Mask classes: 0 above the ILM, 1-7 the seven layers between consecutive boundaries, 8 below the eighth
boundary, 9 fluid (manualFluid1 > 0, drawn over the layers), 255 ignore (columns without the eight boundaries).
Geometry: vertical crop of rows 64-384, horizontal crop to the columns annotated in most B-scans
(10th/90th percentile of the first/last annotated column over all B-scans), resize to 256x512
(area interpolation for images, nearest neighbour for masks).
Writes: <WORK_DIR>/processed/duke_chiu2015_seg/{images,masks}/<Subject>_b<NN>.png + metadata.csv
"""
import os
import glob
import csv

import numpy as np
import scipy.io as sio
import cv2

from config import DUKE_DIR, DUKE_SEG_DIR

MATDIR = DUKE_DIR
OUTDIR = DUKE_SEG_DIR
CROP_Y0, CROP_Y1 = 64, 384
OUT_H, OUT_W = 256, 512
FLUID_CLASS = 9
IGNORE = 255
N_CLASSES = 10


def build_mask(L, Fbin, H, W):
    """L: (8, W) grader-1 boundaries (NaN where not annotated); Fbin: (H, W) binary fluid."""
    mask = np.full((H, W), IGNORE, dtype=np.uint8)
    valid = np.isfinite(L).all(axis=0)
    if valid.any():
        Lv = L[:, valid]
        y = np.arange(H)[None, :, None]
        reg = (Lv[:, None, :] <= y).sum(axis=0)       # number of boundaries above each pixel: 0..8
        mask[:, valid] = reg.astype(np.uint8)
    mask[Fbin > 0] = FLUID_CLASS
    return mask


def compute_xrange(mats):
    firsts, lasts = [], []
    for mp in mats:
        L_all = sio.loadmat(mp)["manualLayers1"]
        for n in range(L_all.shape[2]):
            L = L_all[:, :, n]
            if not np.isfinite(L).any():
                continue
            vc = np.where(np.isfinite(L).all(axis=0))[0]
            if len(vc):
                firsts.append(int(vc.min()))
                lasts.append(int(vc.max()))
    firsts, lasts = np.array(firsts), np.array(lasts)
    x0 = int(np.percentile(firsts, 90))
    x1 = int(np.percentile(lasts, 10)) + 1
    print(f"horizontal crop: columns [{x0}:{x1}] (width {x1 - x0})")
    return x0, x1


def main():
    mats = sorted(glob.glob(os.path.join(MATDIR, "*.mat")))
    assert mats, f"no .mat files in {MATDIR}"
    x0, x1 = compute_xrange(mats)
    os.makedirs(os.path.join(OUTDIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTDIR, "masks"), exist_ok=True)
    rows = []
    class_px = np.zeros(N_CLASSES + 1, dtype=np.int64)
    n_total = 0
    for mp in mats:
        subj = os.path.splitext(os.path.basename(mp))[0]
        d = sio.loadmat(mp)
        imgs, L_all, F_all = d["images"], d["manualLayers1"], d["manualFluid1"]
        H, W, N = imgs.shape
        annot = [n for n in range(N) if np.isfinite(L_all[:, :, n]).any()]
        for n in annot:
            img = imgs[:, :, n]
            L = L_all[:, :, n]
            Fbin = (np.nan_to_num(F_all[:, :, n], nan=0.0) > 0)
            mask = build_mask(L, Fbin, H, W)
            img_c = img[CROP_Y0:CROP_Y1, x0:x1]
            mask_c = mask[CROP_Y0:CROP_Y1, x0:x1]
            img_r = cv2.resize(img_c, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
            mask_r = cv2.resize(mask_c, (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST)
            for c in range(N_CLASSES):
                class_px[c] += int((mask_r == c).sum())
            class_px[N_CLASSES] += int((mask_r == IGNORE).sum())
            ip = os.path.join(OUTDIR, "images", f"{subj}_b{n:02d}.png")
            xp = os.path.join(OUTDIR, "masks", f"{subj}_b{n:02d}.png")
            cv2.imwrite(ip, img_r)
            cv2.imwrite(xp, mask_r)
            rows.append([subj, n, ip, xp,
                         int((mask_r == FLUID_CLASS).sum() > 0),
                         int(np.isfinite(L).all(axis=0).sum())])
            n_total += 1
    meta = os.path.join(OUTDIR, "metadata.csv")
    with open(meta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", "bscan_idx", "img_path", "mask_path", "has_fluid", "n_valid_cols"])
        w.writerows(rows)
    tot = class_px.sum()
    print(f"B-scans written: {n_total} (expected 110) -> {OUTDIR}")
    print(f"ignore {100 * class_px[N_CLASSES] / tot:.2f}% | fluid {100 * class_px[9] / tot:.2f}% of pixels")


if __name__ == "__main__":
    main()
