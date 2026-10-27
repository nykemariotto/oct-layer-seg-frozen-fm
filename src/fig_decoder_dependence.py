"""Fig. 2: medical vs generic frozen encoder under the two decoders (OCT5k, patient level, 95% bootstrap CI).

Reads: <RESULTS_DIR>/oct5k_per_patient_metrics.csv     Writes: <WORK_DIR>/figures/fig_decoder_dependence.png
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RESULTS_DIR, FIG_DIR

CSV = os.path.join(RESULTS_DIR, "oct5k_per_patient_metrics.csv")
OUT = os.path.join(FIG_DIR, "fig_decoder_dependence.png")
COL_RET, COL_DINO, COL_UNET = "#1f3a5f", "#16a085", "#7f8c8d"
N_BOOT, SEED = 10000, 42


def mean_ci(v):
    rng = np.random.default_rng(SEED)
    v = np.asarray(v, dtype=float)
    m = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    if not os.path.exists(CSV):
        sys.exit(f"missing {CSV}: run eval_oct5k.py first")
    df = pd.read_csv(CSV)
    val = {m: mean_ci(g.mdice.to_numpy()) for m, g in df.groupby("model")}

    ret = {"SETR-PUP": val["RETFound+SETR-PUP"], "UNETR": val["RETFound+UNETR"]}
    dino = {"SETR-PUP": val["DINOv2+SETR-PUP"], "UNETR": val["DINOv2+UNETR"]}
    unet_mean = val["U-Net"][0]

    def err(t):
        return np.array([[t[0] - t[1]], [t[2] - t[0]]])

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    x = np.array([0.0, 1.0]); w = 0.34
    ax.bar(x[0] - w/2, ret["SETR-PUP"][0], w, yerr=err(ret["SETR-PUP"]), capsize=4,
           color=COL_RET, label="SETR-PUP (skip-free)")
    ax.bar(x[0] + w/2, ret["UNETR"][0], w, yerr=err(ret["UNETR"]), capsize=4,
           color=COL_RET, hatch="///", edgecolor="white", label="UNETR (skip)")
    ax.bar(x[1] - w/2, dino["SETR-PUP"][0], w, yerr=err(dino["SETR-PUP"]), capsize=4,
           color=COL_DINO)
    ax.bar(x[1] + w/2, dino["UNETR"][0], w, yerr=err(dino["UNETR"]), capsize=4,
           color=COL_DINO, hatch="///", edgecolor="white")

    for xi, d in zip(x, [ret, dino]):
        for off, k in [(-w/2, "SETR-PUP"), (w/2, "UNETR")]:
            ax.text(xi + off, d[k][2] + 0.003, f"{d[k][0]:.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

    # U-Net reference line, interrupted over the bars so that it does not cross labels or error bars
    ax.set_xlim(-0.55, 1.68)
    for a, b in [(-0.55, -0.34), (0.34, 1.0), (1.34, 1.68)]:
        ax.plot([a, b], [unet_mean, unet_mean], ls="--", color=COL_UNET, lw=1.8,
                zorder=1, solid_capstyle="butt")
    ax.text(1.66, unet_mean + 0.011, f"U-Net {unet_mean:.3f}", color=COL_UNET, fontsize=8.5,
            ha="right", va="bottom", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["RETFound\n(medical FM)", "DINOv2\n(generic FM)"], fontsize=9)
    ax.set_ylabel("Mean Dice (patient level, $n$=60)", fontsize=10)
    ax.set_ylim(0.86, 0.972)
    ax.set_title("Medical vs. generic frozen FM under two decoders (OCT5k)", fontsize=10.5, pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(OUT, dpi=600, bbox_inches="tight")
    plt.close()
    print("written:", OUT)


if __name__ == "__main__":
    main()
