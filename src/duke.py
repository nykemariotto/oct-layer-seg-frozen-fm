"""Duke Chiu 2015: folds, dataset, class weights and metrics.

10 classes: 0 above the ILM, 1-7 retinal layers, 8 below Bruch's membrane, 9 fluid; 255 = ignore (columns
without all eight annotated boundaries). Every model works at 224x224.
"""
import os

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from common import IGNORE, IMAGENET_MEAN, IMAGENET_STD
from config import DUKE_SEG_DIR, DUKE_SPLITS

N_CLASSES = 10
N_SURF = 8
SIZE = 224
UM_PER_PX = (320.0 / SIZE) * 3.87   # 320-row crop resized to 224; 3.87 um/px axial (Spectralis)
SEG_DIR = DUKE_SEG_DIR
SPLITS = DUKE_SPLITS
CLASS_NAMES = ["above-ILM", "NFL", "GCL-IPL", "INL", "OPL", "ONL-ISM", "ISE", "OS-RPE", "below-Bruch", "fluid"]


def get_fold_subjects(fold):
    df = pd.read_csv(SPLITS)
    df = df[df.fold == fold]
    return {role: sorted(df[df.role == role].subject.tolist()) for role in ["train", "val", "test"]}


def get_meta(subjects=None):
    meta = pd.read_csv(os.path.join(SEG_DIR, "metadata.csv"))
    if subjects is not None:
        meta = meta[meta.subject.isin(subjects)].reset_index(drop=True)
    return meta


def items_for(subs):
    m = get_meta(subs)
    return list(zip(m.img_path.tolist(), m.mask_path.tolist()))


def compute_class_weights(items, n_classes=N_CLASSES):
    """Median-frequency class weights from the training masks (ignore pixels excluded)."""
    counts = np.zeros(n_classes, dtype=np.float64)
    for _, xp in items:
        m = cv2.imread(xp, cv2.IMREAD_GRAYSCALE)
        for c in range(n_classes):
            counts[c] += (m == c).sum()
    freqs = counts / max(counts.sum(), 1.0)
    med = np.median(freqs[freqs > 0])
    w = np.where(freqs > 0, med / np.maximum(freqs, 1e-8), 1.0)
    return torch.tensor(w, dtype=torch.float32)


def make_tf(train):
    aug = [A.Resize(SIZE, SIZE)]
    if train:
        aug += [A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.3)]
    return A.Compose(aug + [A.Normalize(IMAGENET_MEAN, IMAGENET_STD), ToTensorV2()])


class DukeDS(Dataset):
    def __init__(self, items, train):
        self.items = list(items)
        self.tf = make_tf(train)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ip, xp = self.items[i]
        img = cv2.cvtColor(cv2.imread(ip, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB)
        mask = cv2.imread(xp, cv2.IMREAD_GRAYSCALE)
        o = self.tf(image=img, mask=mask)
        return o["image"], o["mask"].long()


@torch.no_grad()
def ev(model, loader, dev):
    """Per-class intersections and areas pooled over the loader, ignore pixels excluded."""
    model.eval()
    inter = np.zeros(N_CLASSES); ps = np.zeros(N_CLASSES); ts = np.zeros(N_CLASSES)
    for imgs, masks in loader:
        with torch.cuda.amp.autocast():
            pred = model(imgs.to(dev)).argmax(1).cpu()
        valid = masks != IGNORE
        for c in range(N_CLASSES):
            p = (pred == c) & valid; t = (masks == c) & valid
            inter[c] += (p & t).sum().item(); ps[c] += p.sum().item(); ts[c] += t.sum().item()
    return inter, ps, ts


def di(inter, ps, ts):
    den = ps + ts; un = ps + ts - inter
    return (np.where(den > 0, 2 * inter / np.maximum(den, 1), np.nan),
            np.where(un > 0, inter / np.maximum(un, 1), np.nan))


@torch.no_grad()
def bound(model, loader, dev):
    """Mean interface displacement (um) over columns without fluid or ignore pixels."""
    errs = {i: [] for i in range(1, N_SURF + 1)}
    for imgs, masks in loader:
        with torch.cuda.amp.autocast():
            pred = model(imgs.to(dev)).argmax(1).cpu().numpy()
        gt = masks.numpy()
        for b in range(gt.shape[0]):
            g, p = gt[b], pred[b]
            clean = ~(((g == 9) | (g == IGNORE)).any(0))
            for i in range(1, N_SURF + 1):
                errs[i].extend(np.abs((p < i).sum(0)[clean] - (g < i).sum(0)[clean]).tolist())
    allv = [e for i in range(1, N_SURF + 1) for e in errs[i]]
    return round(float(np.mean(allv)) * UM_PER_PX, 2) if allv else None
