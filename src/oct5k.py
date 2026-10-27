"""OCT5k: folds, dataset, class weights and the fold-level metrics written by the training scripts.

6 classes (0-5, no ignore label), 5 interfaces. Boundary errors are converted to micrometres with the
3.5 um axial figure reported for the source acquisitions, rescaled from the 512-row mask to the working size.
"""
import os

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from common import IMAGENET_MEAN, IMAGENET_STD
from config import OCT5K_SEG_DIR, OCT5K_SPLITS

N_CLASSES = 6
N_SURF = 5
SEG_DIR = OCT5K_SEG_DIR
SPLITS = OCT5K_SPLITS
CLASS_NAMES = ["above-ILM", "NFL-OPLh", "OPLh-ISOS", "ISOS-IBRPE", "IBRPE-OBRPE", "below-OBRPE"]
COLORS = np.array([(26, 26, 26), (31, 119, 180), (255, 128, 13), (44, 160, 44), (214, 39, 40), (255, 255, 0)]) / 255.0


def get_fold(fold):
    df = pd.read_csv(SPLITS)
    df = df[df.fold == fold]
    return {r: df[df.role == r].vol_id.tolist() for r in ["train", "val", "test"]}


def items_for(vids):
    m = pd.read_csv(os.path.join(SEG_DIR, "metadata.csv"))
    m = m[m.vol_id.isin(vids)]
    return list(zip(m.img_path.tolist(), m.mask_path.tolist()))


def make_tf(train, size):
    aug = [A.Resize(size, size)]
    if train:
        aug += [A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.3)]
    return A.Compose(aug + [A.Normalize(IMAGENET_MEAN, IMAGENET_STD), ToTensorV2()])


class OCT5kDS(Dataset):
    def __init__(self, items, train, size):
        self.items = list(items)
        self.tf = make_tf(train, size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ip, xp = self.items[i]
        img = cv2.cvtColor(cv2.imread(ip, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB)
        mask = cv2.imread(xp, cv2.IMREAD_GRAYSCALE)
        o = self.tf(image=img, mask=mask)
        return o["image"], o["mask"].long()


def compute_w(items):
    """Median-frequency class weights from the training masks."""
    cnt = np.zeros(N_CLASSES)
    for _, xp in items:
        m = cv2.imread(xp, cv2.IMREAD_GRAYSCALE)
        for c in range(N_CLASSES):
            cnt[c] += (m == c).sum()
    fr = cnt / cnt.sum()
    med = np.median(fr[fr > 0])
    return torch.tensor(np.where(fr > 0, med / np.maximum(fr, 1e-8), 1.0), dtype=torch.float32)


@torch.no_grad()
def ev(model, loader, dev):
    """Per-class intersections and areas pooled over all B-scans of the loader."""
    model.eval()
    inter = np.zeros(N_CLASSES); ps = np.zeros(N_CLASSES); ts = np.zeros(N_CLASSES)
    for imgs, masks in loader:
        with torch.cuda.amp.autocast():
            pred = model(imgs.to(dev)).argmax(1).cpu()
        for c in range(N_CLASSES):
            p = pred == c; t = masks == c
            inter[c] += (p & t).sum().item(); ps[c] += p.sum().item(); ts[c] += t.sum().item()
    return inter, ps, ts


def di(inter, ps, ts):
    """Dice and IoU per class (NaN when the class is absent from both prediction and reference)."""
    den = ps + ts; un = ps + ts - inter
    return (np.where(den > 0, 2 * inter / np.maximum(den, 1), np.nan),
            np.where(un > 0, inter / np.maximum(un, 1), np.nan))


@torch.no_grad()
def bound(model, loader, dev, umpx):
    """Mean absolute interface displacement per A-scan; an interface position is the number of pixels above it."""
    errs = {i: [] for i in range(1, N_SURF + 1)}
    for imgs, masks in loader:
        with torch.cuda.amp.autocast():
            pred = model(imgs.to(dev)).argmax(1).cpu().numpy()
        gt = masks.numpy()
        for b in range(gt.shape[0]):
            g, p = gt[b], pred[b]
            for i in range(1, N_SURF + 1):
                errs[i].extend(np.abs((p < i).sum(0) - (g < i).sum(0)).tolist())
    out = {f"surface_{i}": round(float(np.mean(errs[i])) * umpx, 2) for i in range(1, N_SURF + 1)}
    out["mean_all_um"] = round(float(np.mean([e for i in range(1, N_SURF + 1) for e in errs[i]])) * umpx, 2)
    return out
