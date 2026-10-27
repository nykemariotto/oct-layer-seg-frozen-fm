"""Print every table of the paper, and the numbers quoted in its text, from the files in <RESULTS_DIR>.

Run after stats_oct5k.py and stats_duke.py (efficiency.py and memory_energy.py for Table VI).
The statistics files store unrounded values, so every number below is rounded once (half-up). The Duke
per-subject mDice values are the exception: they are stored with 4 decimals and the statistics use them as such.
Memory is printed in decimal MB (1 MB = 10^6 bytes).
"""
import glob
import itertools
import json
import os
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

from config import RESULTS_DIR

MIB_TO_MB = 1024 ** 2 / 1e6
ORDER = ["U-Net", "U-Net@224", "RETFound+UNETR", "DINOv2+UNETR", "RETFound+SETR-PUP", "DINOv2+SETR-PUP"]


def r(x, places):
    q = Decimal(1).scaleb(-places)
    return str(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))


def sr(x, places=3):
    s = r(x, places)
    return s if s.startswith("-") else "+" + s


def ci(v, places=3, signed=False):
    f = sr if signed else r
    return f"[{f(v[0], places)}, {f(v[1], places)}]"


def load(name):
    path = os.path.join(RESULTS_DIR, name)
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def table_1(st):
    print("\nTable I - OCT5k, patient level (n = 60)")
    for m in ORDER:
        d = st["per_model"][m]
        print(f"  {m:18s} mDice {r(d['mdice']['mean'], 3)} {ci(d['mdice']['ci95'])}  "
              f"surface {r(d['boundary_mean_um']['mean'], 1)} um  thickness {r(d['thickness_mae_um']['mean'], 1)} um")
    if "annotation_agreement" in st:
        ag = st["annotation_agreement"]
        print(f"  annotation agreement: mean {r(ag['mdice_mean_over_pairs'], 3)}; pairs "
              + ", ".join(r(v, 3) for v in ag["pairs"].values()))


def table_2(st):
    print("\nTable II - paired contrasts (Wilcoxon, Holm over five; TOST at 0.01)")
    for cid in ["C1", "C2", "C3", "C4", "C5"]:
        c = st["contrasts"][cid]
        eq = c.get("tost", {}).get("equivalent")
        print(f"  {cid} {sr(c['delta_mean'])} {ci(c['ci95_bootstrap'], signed=True)}  "
              f"p_Holm {c['wilcoxon_p_holm']:.0e}  {c['patients_favouring_a']}  "
              f"equivalent: {'n/a' if eq is None else ('yes' if eq else 'no')}  ({c['description']})")


def table_3(st):
    print("\nTable III - Dice by class and surface distance (um), patient-level mean")
    models = ["U-Net", "RETFound+UNETR", "DINOv2+UNETR"]
    names = ["above-ILM", "ILM-OPLh", "OPLh-IS/OS", "IS/OS-IBRPE", "IBRPE-OBRPE", "below-OBRPE"]
    for c, n in enumerate(names):
        print(f"  {n:12s} " + "  ".join(r(st["per_model"][m]["dice_per_class"][c], 3) for m in models))
    for i, n in enumerate(["ILM", "OPL-Henle", "IS/OS", "IBRPE", "OBRPE"]):
        print(f"  {n:12s} " + "  ".join(r(st["per_model"][m]["surface_distance_um"][i], 1) for m in models))
    print("  mDice_layers: " + ", ".join(f"{m} {r(st['per_model'][m]['mdice_layers']['mean'], 3)}" for m in ORDER))


def table_4(st):
    print("\nTable IV - mDice by disease group [95% CI]")
    for m in ["U-Net", "RETFound+UNETR", "DINOv2+UNETR", "RETFound+SETR-PUP", "DINOv2+SETR-PUP"]:
        p = st["per_disease"][m]
        print(f"  {m:18s} " + "  ".join(f"{g} {r(p[g]['mdice'], 3)} {ci(p[g]['ci95'])}" for g in ["AMD", "DME", "Normal"]))
        print(f"  {'':18s} normal - DME: {r(p['Normal']['mdice'] - p['DME']['mdice'], 3)}")


def table_5(du):
    print("\nTable V - Duke, subject level (n = 10)")
    for m, v in du["per_model"].items():
        print(f"  {m:32s} {r(v['mdice_mean'], 3)} {ci(v['ci95'])}  (t: {ci(v['ci95_t_sensitivity'])})")
    for k, v in du["contrasts"].items():
        print(f"  {k:38s} {sr(v['delta'])} {ci(v['ci95'], signed=True)}  "
              f"t {ci(v['ci95_paired_t_sensitivity'], signed=True)}  {v['subjects_favouring_first']}")


def table_6(st, eff, mem):
    print("\nTable VI - efficiency (batch 1, 224x224)")
    acc = {"U-Net (ResNet-34)": "U-Net@224", "RETFound + SETR-PUP": "RETFound+SETR-PUP",
           "DINOv2 + SETR-PUP": "DINOv2+SETR-PUP", "RETFound + UNETR": "RETFound+UNETR", "DINOv2 + UNETR": "DINOv2+UNETR"}
    for m, key in acc.items():
        line = f"  {m:20s}"
        if eff:
            e = eff["models"][m]
            line += (f" total {r(e['params_total_M'], 1)} M  trainable {r(e['params_trainable_M'], 1)} M  "
                     f"weights {r(e['weights_fp16_MB'], 0)} MB  latency {r(e['latency_ms_median'], 1)} "
                     f"({r(e['latency_ms_p90'], 1)}) ms")
        if mem:
            g = mem["models"][m]
            line += (f"  peak {r(g['autocast']['memory_peak_MiB'] * MIB_TO_MB, 0)} MB "
                     f"(FP16 weights {r(g['fp16']['memory_peak_MiB'] * MIB_TO_MB, 0)} MB)  "
                     f"energy {r(g['energy_autocast']['median']['energy_net_mJ'], 0)} mJ")
        line += f"  mDice {r(st['per_model'][key]['mdice']['mean'], 3)}"
        print(line)


def text_numbers():
    print("\nNumbers quoted in the text")
    sds = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "folds", "oct5k_*_fold0.json"))):
        tag = os.path.basename(path)[len("oct5k_"):-len("_fold0.json")]
        v = [json.load(open(path.replace("_fold0.json", f"_fold{f}.json"), encoding="utf-8"))["test_mdice"]
             for f in range(5)]
        sds[tag] = float(np.std(v, ddof=1))
    if sds:
        print("  between-fold SD of the fold mDice: "
              + ", ".join(f"{k} {r(v, 4)}" for k, v in sds.items())
              + f"  -> range {r(min(sds.values()), 3)}-{r(max(sds.values()), 3)}")
    df = pd.read_csv(os.path.join(RESULTS_DIR, "oct5k_per_patient_metrics.csv"))
    per_fold = df.groupby(["model", "fold"]).mdice.mean().unstack()
    gain = per_fold.loc["U-Net"] - per_fold.loc["U-Net@224"]
    print(f"  U-Net 256 - 224, per fold: {[r(x, 4) for x in gain]}  mean {r(gain.mean(), 3)}")
    m = df.groupby("model")[["mdice", "boundary_mean_px"]].mean()
    px = [abs(m.loc[a, "boundary_mean_px"] - m.loc[b, "boundary_mean_px"]) / abs(m.loc[a, "mdice"] - m.loc[b, "mdice"]) * 0.01
          for a, b in itertools.combinations(m.index, 2)]
    print(f"  boundary displacement per 0.01 mDice, median over the {len(px)} model pairs: {r(np.median(px), 2)} px")


def main():
    st = load("oct5k_stats.json")
    du = load("duke_stats.json")
    table_1(st)
    table_2(st)
    table_3(st)
    table_4(st)
    if du:
        table_5(du)
    table_6(st, load("efficiency.json"), load("memory_energy.json"))
    text_numbers()


if __name__ == "__main__":
    main()
