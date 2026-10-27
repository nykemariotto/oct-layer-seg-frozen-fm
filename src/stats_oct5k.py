"""Patient-level statistics on OCT5k (Tables I-IV and the text of the paper).

Unit of analysis: the patient (n = 60; one volume per subject; every model measured on the same 60).
  - per model: mean with a 95% percentile bootstrap CI over patients (10,000 resamples, seed 42)
  - paired contrasts on the per-patient mDice differences: bootstrap CI, Wilcoxon signed-rank test with Holm
    correction over the five contrasts, equivalence by two one-sided tests (TOST) at delta = 0.01 mDice
      C1 RETFound+SETR-PUP - DINOv2+SETR-PUP
      C2 RETFound+UNETR    - DINOv2+UNETR
      C3 U-Net             - RETFound+UNETR      (best frozen FM)
      C4 (DINOv2+UNETR - DINOv2+SETR-PUP) - (RETFound+UNETR - RETFound+SETR-PUP)
      C5 U-Net@224         - RETFound+UNETR      (matched input resolution)
  - by disease group, per class, per interface
Boundary and thickness values are converted from native mask pixels with 3.5 um/px, the axial figure reported
for the source acquisitions (a common factor: no comparison depends on it).

Reads: <RESULTS_DIR>/oct5k_per_patient_metrics.csv (and oct5k_annotation_agreement.json, if present)
Writes: <RESULTS_DIR>/oct5k_stats.json
"""
import os
import sys
import json

import numpy as np
import pandas as pd
from scipy import stats

from config import RESULTS_DIR

CSV = os.path.join(RESULTS_DIR, "oct5k_per_patient_metrics.csv")
AGREEMENT = os.path.join(RESULTS_DIR, "oct5k_annotation_agreement.json")
OUT = os.path.join(RESULTS_DIR, "oct5k_stats.json")

UM_PER_PX = 3.5
DELTA_TOST = 0.01
N_BOOT = 10000
SEED = 42

UNET, RS, DS, RU, DU = "U-Net", "RETFound+SETR-PUP", "DINOv2+SETR-PUP", "RETFound+UNETR", "DINOv2+UNETR"
UNET224 = "U-Net@224"
CONTRASTS = [
    ("C1", RS, DS, "medical - generic, skip-free decoder (SETR-PUP)"),
    ("C2", RU, DU, "medical - generic, skip decoder (UNETR)"),
    ("C3", UNET, RU, "U-Net (256) - best frozen FM"),
    ("C5", UNET224, RU, "U-Net (224) - best frozen FM, matched input resolution"),
]


def boot_ci(x, n=N_BOOT, seed=SEED):
    """95% percentile bootstrap CI of the mean, resampling patients."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def tost(d, delta=DELTA_TOST):
    """Two one-sided tests; H1: |mean paired difference| < delta."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return {"p": 0.0 if abs(d.mean()) < delta else 1.0, "equivalent": bool(abs(d.mean()) < delta)}
    t_low = (d.mean() + delta) / se
    t_up = (d.mean() - delta) / se
    p = max(stats.t.sf(t_low, n - 1), stats.t.cdf(t_up, n - 1))
    return {"delta": delta, "p": float(p), "equivalent": bool(p < 0.05)}


def holm(ps):
    """Holm-Bonferroni adjusted p values, in input order."""
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    current = 0.0
    for i, k in enumerate(order):
        current = max(current, (m - i) * ps[k])
        adj[k] = min(1.0, current)
    return adj.tolist()


def main():
    if not os.path.exists(CSV):
        sys.exit(f"missing {CSV}: run eval_oct5k.py first")
    df = pd.read_csv(CSV)
    out = {"n_patients": int(df.vol_id.nunique()),
           "bootstrap": {"resamples": N_BOOT, "seed": SEED, "type": "percentile, over patients"},
           "um_per_px": UM_PER_PX}

    metrics = {"mdice": 1.0, "mdice_layers": 1.0, "boundary_mean_px": UM_PER_PX, "thickness_mae_px": UM_PER_PX}
    per_model = {}
    for model, g in df.groupby("model"):
        g = g.sort_values("vol_id")
        d = {}
        for met, k in metrics.items():
            v = g[met].to_numpy()
            lo, hi = boot_ci(v)
            name = met.replace("_px", "_um") if k != 1.0 else met
            d[name] = {"mean": float(v.mean()) * k, "ci95": [lo * k, hi * k]}
        d["dice_per_class"] = [float(g[f"dice_c{c}"].mean()) for c in range(6)]
        d["surface_distance_um"] = [float(g[f"surf{i}_px"].mean()) * UM_PER_PX for i in range(1, 6)]
        d["n_volumes"] = int(len(g))
        per_model[model] = d
    out["per_model"] = per_model

    piv = df.pivot(index="vol_id", columns="model", values="mdice")
    contrasts, p_raw = {}, []
    for cid, a, b, desc in CONTRASTS:
        d = (piv[a] - piv[b]).to_numpy()
        lo, hi = boot_ci(d)
        w = stats.wilcoxon(d)
        contrasts[cid] = {"description": desc, "a": a, "b": b, "n_pairs": int(len(d)),
                          "delta_mean": float(d.mean()), "ci95_bootstrap": [lo, hi],
                          "patients_favouring_a": f"{int((d > 0).sum())}/{len(d)}",
                          "wilcoxon_p_raw": float(w.pvalue), "tost": tost(d)}
        p_raw.append(float(w.pvalue))

    d4 = ((piv[DU] - piv[DS]) - (piv[RU] - piv[RS])).to_numpy()
    lo, hi = boot_ci(d4)
    w4 = stats.wilcoxon(d4)
    contrasts["C4"] = {"description": "decoder gain (UNETR - SETR-PUP) of DINOv2 minus that of RETFound",
                       "n_pairs": int(len(d4)), "delta_mean": float(d4.mean()),
                       "ci95_bootstrap": [lo, hi],
                       "patients_favouring_a": f"{int((d4 > 0).sum())}/{len(d4)}",
                       "wilcoxon_p_raw": float(w4.pvalue)}
    p_raw.append(float(w4.pvalue))

    for k, p in zip([c[0] for c in CONTRASTS] + ["C4"], holm(p_raw)):
        contrasts[k]["wilcoxon_p_holm"] = float(p)
    out["contrasts"] = contrasts

    per_disease = {}
    for (model, disease), g in df.groupby(["model", "disease"]):
        lo, hi = boot_ci(g.mdice.to_numpy())
        per_disease.setdefault(model, {})[disease] = {
            "n": int(len(g)), "mdice": float(g.mdice.mean()), "ci95": [lo, hi],
            "boundary_um": float(g.boundary_mean_px.mean()) * UM_PER_PX,
            "thickness_mae_um": float(g.thickness_mae_px.mean()) * UM_PER_PX}
    out["per_disease"] = per_disease

    if os.path.exists(AGREEMENT):
        ag = json.load(open(AGREEMENT, encoding="utf-8"))
        out["annotation_agreement"] = {"mdice_mean_over_pairs": ag["summary"]["mdice_mean_over_pairs"],
                                       "pairs": {k: v["mdice_mean"] for k, v in ag["pairs"].items()}}

    with open(OUT, "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2)

    print(f"n = {out['n_patients']} patients")
    for m, d in sorted(per_model.items(), key=lambda kv: -kv[1]["mdice"]["mean"]):
        print(f"  {m:18s} mDice {d['mdice']['mean']:.4f} {d['mdice']['ci95']}  "
              f"surface {d['boundary_mean_um']['mean']:.2f} um  thickness {d['thickness_mae_um']['mean']:.2f} um")
    for cid, c in contrasts.items():
        eq = c.get("tost", {}).get("equivalent")
        print(f"  {cid}: delta {c['delta_mean']:+.4f} {c['ci95_bootstrap']}  p_Holm {c['wilcoxon_p_holm']:.1e}  "
              f"{c['patients_favouring_a']}" + ("  equivalent" if eq else ""))
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
