"""Pair each OCT5k annotation (Grading_1) with its B-scan and write 512x512 PNGs.

Pairs come from OCT5k's Scripts/paths/manual_paths.csv. B-scans (512x496 TIFF) are resampled to 512x512 with
area interpolation to match the annotation grid; masks are resampled (nearest neighbour) only if needed.
6 dense classes (0-5), no crop.
Writes: <WORK_DIR>/processed/oct5k_seg/{images,masks}/<vol_id>__b<N>.png + metadata.csv
"""
import os
import re
import csv

import numpy as np
import cv2

from config import OCT5K_DIR, RASTI_DIR, OCT5K_SEG_DIR

MASKS_G1 = os.path.join(OCT5K_DIR, "Masks", "Masks_Manual", "Grading_1")
CSVPATH = os.path.join(OCT5K_DIR, "Scripts", "paths", "manual_paths.csv")
OUTDIR = OCT5K_SEG_DIR
OUT = 512
N_CLASSES = 6


def resolve_from(fp):
    rel = fp.strip().replace("/", os.sep)
    while rel.startswith("." + os.sep):
        rel = rel[2:]
    return os.path.join(RASTI_DIR, rel)


def resolve_mask(tp):
    sub = tp.split("Images_Manual/", 1)[1].strip()
    return os.path.join(MASKS_G1, sub.replace("/", os.sep)), sub


def main():
    os.makedirs(os.path.join(OUTDIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTDIR, "masks"), exist_ok=True)
    rows = [r for r in csv.reader(open(CSVPATH, encoding="utf-8")) if len(r) >= 2]
    meta = []
    class_px = np.zeros(N_CLASSES, dtype=np.int64)
    n = 0
    for tp, fp in rows:
        ip = resolve_from(fp)
        mp, sub = resolve_mask(tp)
        if not (os.path.exists(ip) and os.path.exists(mp)):
            continue
        parts = sub.split("/")
        classe, volume = parts[0], parts[1]
        disease = classe.split(" Part")[0]                 # AMD / DME / Normal
        vol_id = re.sub(r"[^A-Za-z0-9]+", "_", f"{classe}__{volume}".replace(".E2E", "")).strip("_")
        bscan = re.sub(r"\D", "", parts[-1]) or str(n)

        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        img_r = cv2.resize(img, (OUT, OUT), interpolation=cv2.INTER_AREA)
        if mask.shape != (OUT, OUT):
            mask = cv2.resize(mask, (OUT, OUT), interpolation=cv2.INTER_NEAREST)
        for c in range(N_CLASSES):
            class_px[c] += int((mask == c).sum())

        op = os.path.join(OUTDIR, "images", f"{vol_id}__b{bscan}.png")
        xp = os.path.join(OUTDIR, "masks", f"{vol_id}__b{bscan}.png")
        cv2.imwrite(op, img_r)
        cv2.imwrite(xp, mask)
        meta.append([vol_id, disease, volume, bscan, op, xp])
        n += 1

    metap = os.path.join(OUTDIR, "metadata.csv")
    with open(metap, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["vol_id", "disease", "volume", "bscan", "img_path", "mask_path"])
        w.writerows(meta)

    tot = class_px.sum()
    print(f"pairs written: {n} (expected 1672) -> {metap}")
    names = ["0 above-ILM", "1 NFL-OPLh", "2 OPLh-ISOS", "3 ISOS-IBRPE", "4 IBRPE-OBRPE", "5 below-OBRPE"]
    for c in range(N_CLASSES):
        print(f"  {names[c]:16s}: {100 * class_px[c] / tot:5.2f}% of pixels")
    print(f"volumes: {len(set(m[0] for m in meta))} (expected 60)")


if __name__ == "__main__":
    main()
