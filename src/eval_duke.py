"""Per-subject mDice on Duke Chiu 2015 from the fold checkpoints (inference only).

For each subject, intersections and areas per class are pooled over its test B-scans; mDice is the mean of the
class Dice values (a class absent from both reference and prediction is left out), as in train_duke.py.
Control: pooling the two subjects of each fold must reproduce the fold's test_mdice written by train_duke.py
(tolerance 0.001), when those files are present.
Writes: <RESULTS_DIR>/duke_per_subject_metrics.csv (mDice rounded to 4 decimals, as used by stats_duke.py)
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import build_model
from config import CKPT_DIR, RESULTS_DIR
from duke import N_CLASSES, DukeDS, get_fold_subjects, items_for, ev, di
from models import RETFoundSeg, DINOv2Seg

MODELS = ["unet", "retfound_setr", "dinov2_setr", "unet_frozen"]
TOL = 0.001


def load(name, fold, dev):
    ck = os.path.join(CKPT_DIR, f"duke_{name}_fold{fold}.pth")
    if name in ("retfound_setr", "dinov2_setr"):
        model = (RETFoundSeg if name == "retfound_setr" else DINOv2Seg)(n_classes=N_CLASSES).to(dev)
        model.decoder.load_state_dict(torch.load(ck, map_location=dev))
    else:
        model = build_model(n_classes=N_CLASSES).to(dev)
        model.load_state_dict(torch.load(ck, map_location=dev))
    return model.eval()


def mdice(inter, ps, ts):
    d, _ = di(inter, ps, ts)
    return float(np.nanmean(d))


def main():
    dev = "cuda"
    rows, control = [], []
    for name in MODELS:
        for fold in range(5):
            subs = get_fold_subjects(fold)["test"]
            model = load(name, fold, dev)
            pooled = [np.zeros(N_CLASSES) for _ in range(3)]
            for s in subs:
                dl = DataLoader(DukeDS(items_for([s]), False), batch_size=8, shuffle=False, num_workers=0)
                inter, ps, ts = ev(model, dl, dev)
                rows.append({"model": name, "fold": fold, "subject": s, "mdice": round(mdice(inter, ps, ts), 4)})
                for acc, v in zip(pooled, (inter, ps, ts)):
                    acc += v
            ref_path = os.path.join(RESULTS_DIR, "folds", f"duke_{name}_fold{fold}.json")
            if os.path.exists(ref_path):
                ref = json.load(open(ref_path, encoding="utf-8"))
                assert sorted(subs) == sorted(ref["subjects_test"]), (name, fold)
                fold_md = mdice(*pooled)
                control.append(abs(fold_md - ref["test_mdice"]))
                assert control[-1] <= TOL, (name, fold, fold_md, ref["test_mdice"])
            del model
            torch.cuda.empty_cache()
    if control:
        print(f"control: per-fold mDice reproduced in {len(control)} folds (largest difference {max(control):.4f})")
    out = os.path.join(RESULTS_DIR, "duke_per_subject_metrics.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
