"""Agreement between the three manual annotation sets of OCT5k (Grading_1/2/3).

Pairwise Dice per volume over the same images, with the class definition used for training (6 classes,
512x512 masks, nearest-neighbour resampling). The sets do not correspond to individual annotators: the
dataset distributes images among graders within each set and does not publish grader identity per image.
Writes: <RESULTS_DIR>/oct5k_annotation_agreement.json
"""
import os
import re
import csv
import json
import sys
from itertools import combinations

import numpy as np
import cv2

from config import OCT5K_DIR, RESULTS_DIR

MASK_BASE = os.path.join(OCT5K_DIR, "Masks", "Masks_Manual")
CSVPATH = os.path.join(OCT5K_DIR, "Scripts", "paths", "manual_paths.csv")
OUT = os.path.join(RESULTS_DIR, "oct5k_annotation_agreement.json")
GRADINGS = ["Grading_1", "Grading_2", "Grading_3"]
N_CLASSES = 6
SIZE = 512


def load(grading, sub):
    m = cv2.imread(os.path.join(MASK_BASE, grading, sub.replace("/", os.sep)), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape != (SIZE, SIZE):
        m = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    return m


def main():
    acc = {p: {} for p in combinations(GRADINGS, 2)}
    n_imgs = 0

    for row in csv.reader(open(CSVPATH, encoding="utf-8")):
        if len(row) < 2:
            continue
        tp = row[0]
        if "Images_Manual/" not in tp:
            continue
        sub = tp.split("Images_Manual/", 1)[1].strip()
        parts = sub.split("/")
        classe, volume = parts[0], parts[1]
        vol_id = re.sub(r"[^A-Za-z0-9]+", "_", f"{classe}__{volume}".replace(".E2E", "")).strip("_")

        masks = {g: load(g, sub) for g in GRADINGS}
        if any(m is None for m in masks.values()):
            continue
        n_imgs += 1

        for pair in acc:
            a, b = masks[pair[0]], masks[pair[1]]
            d = acc[pair].setdefault(vol_id, {"inter": np.zeros(N_CLASSES), "sa": np.zeros(N_CLASSES),
                                              "sb": np.zeros(N_CLASSES), "n": 0})
            for c in range(N_CLASSES):
                pa, pb = a == c, b == c
                d["inter"][c] += (pa & pb).sum()
                d["sa"][c] += pa.sum()
                d["sb"][c] += pb.sum()
            d["n"] += 1

    if not n_imgs:
        sys.exit("no image matched: check OCT5K_DIR")

    result = {"n_images": n_imgs, "n_volumes": len(acc[list(acc)[0]]),
              "definition": "Dice between two manual annotation sets, per volume, 6 classes, 512x512 masks",
              "pairs": {}}
    all_mdice = []
    for pair, vols in acc.items():
        per_vol, per_vol_layers, per_class = [], [], np.zeros((len(vols), N_CLASSES))
        for i, (vid, d) in enumerate(sorted(vols.items())):
            den = d["sa"] + d["sb"]
            dice = np.where(den > 0, 2 * d["inter"] / np.maximum(den, 1), np.nan)
            per_class[i] = dice
            per_vol.append(float(np.nanmean(dice)))
            per_vol_layers.append(float(np.nanmean(dice[1:5])))
        result["pairs"][f"{pair[0]}_vs_{pair[1]}"] = {
            "mdice_mean": float(np.mean(per_vol)),
            "mdice_std": float(np.std(per_vol, ddof=1)),
            "mdice_layers_mean": float(np.mean(per_vol_layers)),
            "per_class_mean": [float(np.nanmean(per_class[:, c])) for c in range(N_CLASSES)],
        }
        all_mdice.extend(per_vol)

    m = float(np.mean(all_mdice))
    result["summary"] = {"mdice_mean_over_pairs": m}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2)

    print(f"images: {n_imgs} | volumes: {result['n_volumes']}")
    for name, v in result["pairs"].items():
        print(f"  {name}: mDice {v['mdice_mean']:.4f} +- {v['mdice_std']:.4f} (layers {v['mdice_layers_mean']:.4f})")
    print(f"mean over pairs: {m:.4f}\nwritten: {OUT}")


if __name__ == "__main__":
    main()
