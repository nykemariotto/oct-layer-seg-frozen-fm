"""Duke Chiu 2015 5-fold cross-validation at the subject level (seed 42).

Each fold: 2 test subjects, 1 validation subject (drawn with seed 42 + fold), 7 training subjects.
Every subject is tested exactly once.
Writes: <SPLITS_DIR>/duke_chiu2015_splits_seed42.csv (subject, fold, role)
"""
import os
import csv

import numpy as np
from sklearn.model_selection import KFold

from config import DUKE_SPLITS

SUBJECTS = [f"Subject_{i:02d}" for i in range(1, 11)]
OUT = DUKE_SPLITS
SEED = 42


def main():
    subs = np.array(SUBJECTS)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for fold, (trainval_idx, test_idx) in enumerate(kf.split(subs)):
        test = subs[test_idx]
        trainval = subs[trainval_idx]
        rng = np.random.default_rng(SEED + fold)
        tv = trainval.copy()
        rng.shuffle(tv)
        val = tv[:1]
        train = sorted(tv[1:].tolist())
        for s in train:
            rows.append((s, fold, "train"))
        for s in sorted(val.tolist()):
            rows.append((s, fold, "val"))
        for s in sorted(test.tolist()):
            rows.append((s, fold, "test"))
        print(f"fold {fold}: test={sorted(test.tolist())} val={sorted(val.tolist())} train={train}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", "fold", "role"])
        w.writerows(rows)
    print(f"written: {OUT} ({len(rows)} rows)")
    test_counts = {}
    for s, fold, role in rows:
        if role == "test":
            test_counts[s] = test_counts.get(s, 0) + 1
    assert all(v == 1 for v in test_counts.values()) and len(test_counts) == 10, "every subject tested exactly once"


if __name__ == "__main__":
    main()
