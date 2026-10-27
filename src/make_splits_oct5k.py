"""OCT5k 5-fold cross-validation, one volume per subject, stratified by disease group (seed 42).

Each fold tests 12 volumes; the remaining 48 are split into 42 train / 6 validation, also stratified.
Every volume is tested exactly once. Reads the metadata written by prepare_oct5k.py.
Writes: <SPLITS_DIR>/oct5k_splits_seed42.csv (vol_id, disease, fold, role)
"""
import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import OCT5K_SEG_DIR, OCT5K_SPLITS

META = os.path.join(OCT5K_SEG_DIR, "metadata.csv")
OUT = OCT5K_SPLITS
SEED = 42
N_VAL = 6


def main():
    meta = pd.read_csv(META)
    vols = meta[["vol_id", "disease"]].drop_duplicates().reset_index(drop=True)
    print(f"{len(vols)} volumes: {vols.disease.value_counts().to_dict()}")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    summary = []
    for fold, (tr, te) in enumerate(skf.split(vols.vol_id, vols.disease)):
        tv = vols.iloc[tr].reset_index(drop=True)
        test = vols.iloc[te]
        train, val = train_test_split(tv, test_size=N_VAL, stratify=tv.disease, random_state=SEED + fold)
        for _, r in train.iterrows():
            rows.append((r.vol_id, r.disease, fold, "train"))
        for _, r in val.iterrows():
            rows.append((r.vol_id, r.disease, fold, "val"))
        for _, r in test.iterrows():
            rows.append((r.vol_id, r.disease, fold, "test"))
        summary.append((fold, len(train), len(val), len(test), test.disease.value_counts().to_dict()))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = pd.DataFrame(rows, columns=["vol_id", "disease", "fold", "role"])
    df.to_csv(OUT, index=False)
    print(f"written: {OUT} ({len(rows)} rows)")
    for f, nt, nv, nte, td in summary:
        print(f"fold {f}: train {nt} / val {nv} / test {nte} | test: {td}")
    tc = df[df.role == "test"].vol_id.value_counts()
    assert (tc == 1).all() and len(tc) == len(vols), "every volume must be tested exactly once"


if __name__ == "__main__":
    main()
