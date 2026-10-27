"""Subject-level statistics on Duke Chiu 2015 (Table V and the Duke paragraph of the paper).

Unit of analysis: the subject (n = 10).
  - per model: mean mDice with a 95% percentile bootstrap CI over subjects (10,000 resamples, seed 42)
  - paired contrasts per subject, with bootstrap CI and the number of subjects favouring the first term:
      D1 RETFound - DINOv2            (both frozen, same SETR-PUP decoder)
      D2 U-Net - RETFound
      D3 U-Net - U-Net with frozen ImageNet encoder
  - sensitivity analysis (post hoc): Student t intervals (paired for the contrasts, 9 degrees of freedom),
    because the percentile bootstrap tends to understate the uncertainty with ten subjects.
One random generator is shared by all intervals, in the order below.
Reads: <RESULTS_DIR>/duke_per_subject_metrics.csv     Writes: <RESULTS_DIR>/duke_stats.json
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from config import RESULTS_DIR

MODELS = {"unet": "U-Net", "retfound_setr": "RETFound", "dinov2_setr": "DINOv2",
          "unet_frozen": "U-Net, frozen ImageNet encoder"}
CONTRASTS = {"D1 RETFound - DINOv2": ("retfound_setr", "dinov2_setr"),
             "D2 U-Net - RETFound": ("unet", "retfound_setr"),
             "D3 U-Net - U-Net with frozen encoder": ("unet", "unet_frozen")}
N_BOOT, SEED = 10000, 42


def ci(v, rng):
    v = np.asarray(v)
    b = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def ci_t(v):
    v = np.asarray(v)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return [float(v.mean() - h), float(v.mean() + h)]


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "duke_per_subject_metrics.csv"))
    per_subject = {m: dict(zip(g.subject, g.mdice)) for m, g in df.groupby("model")}
    subjects = sorted(per_subject["unet"])
    rng = np.random.default_rng(SEED)
    res = {"n_subjects": len(subjects), "bootstrap": {"resamples": N_BOOT, "seed": SEED, "type": "percentile, over subjects"},
           "per_model": {}, "contrasts": {}}
    for key, name in MODELS.items():
        v = [per_subject[key][s] for s in subjects]
        res["per_model"][name] = {"mdice_mean": float(np.mean(v)), "ci95": ci(v, rng),
                                  "ci95_t_sensitivity": ci_t(v)}
    for name, (a, b) in CONTRASTS.items():
        d = [per_subject[a][s] - per_subject[b][s] for s in subjects]
        res["contrasts"][name] = {"delta": float(np.mean(d)), "ci95": ci(d, rng),
                                  "ci95_paired_t_sensitivity": ci_t(d),
                                  "subjects_favouring_first": f"{sum(x > 0 for x in d)}/{len(d)}"}
    out = os.path.join(RESULTS_DIR, "duke_stats.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for k, v in res["per_model"].items():
        print(f"{k:32s} {v['mdice_mean']:.4f} {v['ci95']}")
    for k, v in res["contrasts"].items():
        print(f"{k:38s} {v['delta']:+.4f} {v['ci95']}  t {v['ci95_paired_t_sensitivity']}  {v['subjects_favouring_first']}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
