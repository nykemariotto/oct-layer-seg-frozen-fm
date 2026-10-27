"""Fig. 3: reference and predictions on three OCT5k test B-scans of fold 0 (one AMD, one DME, one normal).

The middle B-scan of the first fold-0 test volume of each group. The U-Net shown is the one trained at 224x224
(input-resolution control), so that all panels share the same input size.
Writes: <WORK_DIR>/figures/fig_qualitative.png
"""
import os

import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from common import build_model
from config import CKPT_DIR, FIG_DIR
from models import ViTUNetr
from oct5k import get_fold, N_CLASSES, SEG_DIR, COLORS

OUT = os.path.join(FIG_DIR, "fig_qualitative.png")
SIZE = 224
CLS_EN = ["above-ILM (bg)", "NFL–OPL", "OPL–ISOS", "ISOS–IBRPE", "IBRPE–OBRPE", "below-OBRPE (bg)"]
COLS = ["B-scan", "Ground truth", "U-Net", "RETFound+UNETR", "DINOv2+UNETR"]


def col(m):
    rgb = np.zeros((*m.shape, 3))
    for c in range(N_CLASSES):
        rgb[m == c] = COLORS[c]
    return rgb


def pick_cases():
    m = pd.read_csv(os.path.join(SEG_DIR, "metadata.csv"))
    m = m[m.vol_id.isin(get_fold(0)["test"])]
    items = []
    for dis in ["AMD", "DME", "Normal"]:
        sub = m[m.disease == dis]
        vol = sub.vol_id.iloc[0]
        sv = sub[sub.vol_id == vol].reset_index(drop=True)
        rec = sv.iloc[len(sv) // 2]
        items.append((dis, rec.img_path, rec.mask_path))
    return items


@torch.no_grad()
def predict(model, img01, dev):
    rgb = np.stack([img01] * 3, -1)
    x = ((rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225]))
    x = torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float32)).to(dev)
    with torch.cuda.amp.autocast():
        return model(x).argmax(1)[0].cpu().numpy()


def main():
    dev = "cuda"
    cases = pick_cases()
    imgs, gts = [], []
    for dis, ip, xp in cases:
        u8 = cv2.resize(cv2.imread(ip, cv2.IMREAD_GRAYSCALE), (SIZE, SIZE))
        imgs.append(u8.astype(np.float32) / 255.0)
        gts.append(cv2.resize(cv2.imread(xp, cv2.IMREAD_GRAYSCALE), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST))

    preds = {}
    un = build_model(n_classes=N_CLASSES).to(dev).eval()
    un.load_state_dict(torch.load(os.path.join(CKPT_DIR, "oct5k_unet224_fold0.pth")))
    preds["U-Net"] = [predict(un, im, dev) for im in imgs]
    del un; torch.cuda.empty_cache()
    rm = ViTUNetr("ret").to(dev).eval()
    rm.decoder.load_state_dict(torch.load(os.path.join(CKPT_DIR, "oct5k_retfound_unetr_fold0.pth")))
    preds["RETFound+UNETR"] = [predict(rm, im, dev) for im in imgs]
    del rm; torch.cuda.empty_cache()
    dm = ViTUNetr("dino").to(dev).eval()
    dm.decoder.load_state_dict(torch.load(os.path.join(CKPT_DIR, "oct5k_dinov2_unetr_fold0.pth")))
    preds["DINOv2+UNETR"] = [predict(dm, im, dev) for im in imgs]
    del dm; torch.cuda.empty_cache()

    nrow = len(cases)
    fig, axes = plt.subplots(nrow, 5, figsize=(12.5, 2.6 * nrow))
    for r, (dis, _, _) in enumerate(cases):
        axes[r, 0].imshow(imgs[r], cmap="gray")
        axes[r, 1].imshow(col(gts[r]))
        axes[r, 2].imshow(col(preds["U-Net"][r]))
        axes[r, 3].imshow(col(preds["RETFound+UNETR"][r]))
        axes[r, 4].imshow(col(preds["DINOv2+UNETR"][r]))
        axes[r, 0].set_ylabel(dis, fontsize=17, fontweight="bold")
        for c in range(5):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(COLS[c], fontsize=16)
    handles = [Patch(color=COLORS[c], label=CLS_EN[c]) for c in range(N_CLASSES)]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=13, frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(OUT, dpi=600)
    plt.close()
    print("written:", OUT)


if __name__ == "__main__":
    main()
