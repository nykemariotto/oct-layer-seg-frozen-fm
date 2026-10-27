# Lightweight networks vs. frozen foundation models for retinal OCT layer segmentation

Code, cross-validation splits and derived results for the paper *Lightweight Networks Match Frozen
Foundation Models for Embedded Retinal OCT Layer Segmentation* (CBEB 2026). The full citation will be
added once the proceedings are published.

The study compares a fully trained U-Net (ResNet-34 encoder) with two frozen ViT-L foundation-model
encoders, RETFound (retina-specific) and DINOv2 (generic), each paired with a skip-free decoder
(SETR-PUP) and a skip decoder (UNETR), on OCT5k (60 subjects) and Duke Chiu 2015 (10 subjects), with
subject-disjoint 5-fold cross-validation and patient-level statistics.

## What is included

| Folder | Content |
|---|---|
| `src/` | pre-processing, splits, training, evaluation, statistics, efficiency measurements, figures |
| `splits/` | the cross-validation splits used in the paper (seed 42) |
| `results/` | per-patient metrics, per-fold test metrics, statistics and efficiency measurements reported in the paper |

Not included: images, annotations and model weights (pre-trained or trained). They have to be obtained
from their sources, below.

## Data

- **OCT5k annotations** (CC0): Arikan et al., *Sci. Data* 12:267 (2025); files at
  [doi:10.5522/04/22128671.v4](https://doi.org/10.5522/04/22128671.v4). Extract `OCT5k.zip`; the folder
  with `Masks/` and `Scripts/` is `OCT5K_DIR`.
- **OCT5k B-scans**: OCT5k does not redistribute the images. They belong to the macular OCT dataset of
  Rasti et al., *IEEE Trans. Med. Imag.* 37(4):1024-1034 (2018). `RASTI_DIR` is the folder that contains
  `Macular-Dataset-R.Rasti_old/`, the path referenced in OCT5k's `Scripts/paths/manual_paths.csv`.
- **Duke Chiu 2015**: Chiu et al., *Biomed. Opt. Express* 6(4):1172-1194 (2015);
  [2015_BOE_Chiu2.zip](https://people.duke.edu/~sf59/Datasets/2015_BOE_Chiu2.zip). `DUKE_DIR` is the folder
  with `Subject_01.mat` ... `Subject_10.mat`.

Pre-trained encoders are downloaded on first use: RETFound from
[YukunZhou/RETFound_mae_natureOCT](https://huggingface.co/YukunZhou/RETFound_mae_natureOCT) (accept the terms
on the model page and log in with `huggingface-cli login`), DINOv2 ViT-L/14 through `timm`, and the
ImageNet ResNet-34 through `segmentation-models-pytorch`.

## Environment

Python 3.11, PyTorch 2.3.1 with CUDA 12.1; see `requirements.txt` for the other versions. The experiments ran
on a single 8 GB GPU (NVIDIA RTX 4060 Ti) under Windows 11.

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Paths are read from environment variables (`OCT5K_DIR`, `RASTI_DIR`, `DUKE_DIR`, `WORK_DIR`, `RESULTS_DIR`,
`SPLITS_DIR`; see `src/config.py`). Run the scripts from `src/`.

## Pipeline

| Step | Script | Output |
|---|---|---|
| Pre-processing | `prepare_oct5k.py`, `prepare_duke.py` | 512x512 (OCT5k) and 256x512 (Duke) image/mask PNGs |
| Splits | `make_splits_oct5k.py`, `make_splits_duke.py` | `splits/*.csv` (subject-disjoint, seed 42) |
| Training, OCT5k | `train_unet_oct5k.py <fold> [256\|224]`, `train_fm_oct5k.py <ret\|dino> <setr\|unetr> <fold>` | checkpoints, `results/folds/` |
| Training, Duke | `train_duke.py <fold> <unet\|unet_frozen\|ret\|dino>` | checkpoints, `results/folds/` |
| Evaluation | `eval_oct5k.py`, `eval_duke.py` | `results/oct5k_per_patient_metrics.csv`, `results/duke_per_subject_metrics.csv` |
| Annotation agreement | `annotation_agreement_oct5k.py` | `results/oct5k_annotation_agreement.json` |
| Statistics | `stats_oct5k.py`, `stats_duke.py` | `results/oct5k_stats.json`, `results/duke_stats.json` |
| Efficiency | `efficiency.py`, `memory_energy.py` | `results/efficiency.json`, `results/memory_energy.json` |
| Figures | `fig_decoder_dependence.py`, `fig_qualitative.py` | Figs. 2 and 3 |
| Tables | `paper_numbers.py` | Tables I-VI and the numbers quoted in the text |

Training settings: U-Net at 256x256 (and 224x224 for the input-resolution control), batch 16, AdamW 3e-4,
up to 100 epochs with early stopping (patience 20); frozen-encoder decoders at 224x224, batch 8, AdamW 1e-3,
80 epochs (patience 20); Duke, 224x224, batch 8, 120 epochs (patience 30). Cross-entropy + Dice loss with
median-frequency class weights, cosine schedule, mixed precision; the checkpoint with the best validation mean
Dice is kept. Each configuration was trained once (seed 42). GPU training is not bit-for-bit deterministic, so
retraining gives close but not identical numbers; evaluation of a given checkpoint is.

## Reproducing the tables from the included results

The statistics need only the files in `results/`:

```bash
cd src
python stats_oct5k.py
python stats_duke.py
python paper_numbers.py
```

| File | Content |
|---|---|
| `oct5k_per_patient_metrics.csv` | one row per model and test patient: Dice per class, mDice, interface displacement, thickness error |
| `duke_per_subject_metrics.csv` | one row per model and test subject: mDice |
| `folds/*.json` | per-fold test metrics written by the training scripts |
| `oct5k_annotation_agreement.json` | agreement between the three OCT5k annotation sets |
| `oct5k_stats.json`, `duke_stats.json` | means, bootstrap intervals, paired contrasts, equivalence tests |
| `efficiency.json` | parameters, FP16 weight size and latency (RTX 4060 Ti; latency measured in June 2026) |
| `memory_energy.json` | peak GPU memory and energy per B-scan (RTX 4060 Ti, October 2026); its latency is within 2.4% of `efficiency.json` |

Statistics are stored unrounded, and `paper_numbers.py` rounds each value once (half up).

## Notes

- Boundary and thickness errors are computed in pixels of the native 512-row OCT5k mask and converted with
  3.5 um/px, the axial figure reported for the source acquisitions; it is a resolution figure, not a
  calibrated pixel spacing, and it is common to all models.
- OCT5k ships three annotation sets; *Grading_1* is the reference for every model and fold.
- Efficiency figures depend on the hardware; close other programs that use the GPU before measuring.

## License

Code under the MIT License (see `LICENSE`). The datasets and the pre-trained encoders keep their own terms.
