"""Paths used by every script. Set the environment variables (or edit the defaults below).

OCT5K_DIR   folder extracted from OCT5k.zip (contains Masks/ and Scripts/)
RASTI_DIR   folder that contains Macular-Dataset-R.Rasti_old/ (the OCT5k B-scan images)
DUKE_DIR    folder with Subject_01.mat ... Subject_10.mat (Duke Chiu 2015)
WORK_DIR    pre-processed images, checkpoints and figures (default: <repo>/work)
RESULTS_DIR metrics and statistics (default: <repo>/results, the values reported in the paper)
SPLITS_DIR  cross-validation splits (default: <repo>/splits, the splits used in the paper)
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(name, default):
    return os.environ.get(name, default)


OCT5K_DIR = _env("OCT5K_DIR", os.path.join(ROOT, "data", "OCT5k"))
RASTI_DIR = _env("RASTI_DIR", os.path.join(ROOT, "data", "rasti"))
DUKE_DIR = _env("DUKE_DIR", os.path.join(ROOT, "data", "2015_BOE_Chiu"))
WORK_DIR = _env("WORK_DIR", os.path.join(ROOT, "work"))
RESULTS_DIR = _env("RESULTS_DIR", os.path.join(ROOT, "results"))
SPLITS_DIR = _env("SPLITS_DIR", os.path.join(ROOT, "splits"))

OCT5K_SEG_DIR = os.path.join(WORK_DIR, "processed", "oct5k_seg")
DUKE_SEG_DIR = os.path.join(WORK_DIR, "processed", "duke_chiu2015_seg")
CKPT_DIR = os.path.join(WORK_DIR, "checkpoints")
FIG_DIR = os.path.join(WORK_DIR, "figures")

OCT5K_SPLITS = os.path.join(SPLITS_DIR, "oct5k_splits_seed42.csv")
DUKE_SPLITS = os.path.join(SPLITS_DIR, "duke_chiu2015_splits_seed42.csv")
