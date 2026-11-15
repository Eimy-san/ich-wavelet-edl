# Wavelet-Enhanced Evidential Deep Learning for Uncertainty-Aware Intracranial Hemorrhage Triage on Non-Contrast Head CT

Code, patient-level data partitions, and per-series rescale metadata for the study
"Wavelet-Enhanced Evidential Deep Learning for Uncertainty-Aware Clinical Decision
Support in Intracranial Hemorrhage Triage on Non-Contrast Head CT"
(submitted to *Scientific Reports*).

## Contents

```
src/   Training, evaluation, examination-level aggregation, and analysis code
data/  Patient-level partition lists (train/val/test) and per-series rescale metadata
```

### Source code (`src/`)

| File | Purpose |
|------|---------|
| `config.py` | Experiment configuration (seeds, hyperparameters, paths) |
| `data_prep.py` | Archive extraction, label matching, patient-level split construction |
| `dataset.py` | CT slice dataset: HU conversion, brain windowing, augmentation |
| `model.py` | ResNet-50 backbone + wavelet attention + evidential head (+ parameter-matched SE-only variant) |
| `wavelet_attention.py` | Haar DWT/IDWT, SE blocks, wavelet attention module |
| `edl_head.py` | Evidential (Dirichlet) head: losses and predictions |
| `train.py` | Training loop (50 epochs, AdamW, cosine annealing, balanced sampler, seed control) |
| `evaluate.py` | Risk-coverage, ROC, calibration, and figure generation |
| `gradcam.py` | Grad-CAM visualization |
| `run_experiments_v4.sh` | Factorial ablation used for the reported results (4 configurations) |
| `run_seeds.sh` | Multi-seed replication protocol (5 independent seeds per configuration) |

### Data files (`data/`)

| File | Content |
|------|---------|
| `train.csv`, `val.csv`, `test.csv` | Patient-level partition lists: image path, patient ID, slice-level hemorrhage labels (`any` and five subtypes). 47,368/10,204/9,560 slices from 1,179/253/253 patients. |
| `rsna_rescale_values.csv` | Per-series rescale slope and intercept (from DICOM headers of the RSNA stage-2 training set) keyed by patient/study/series; used to convert 16-bit PNG archive values to Hounsfield units. |

The RSNA-ICH images themselves are available from the
[RSNA 2019 Brain CT Hemorrhage Challenge](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection).

## Reproducing the experiments

```bash
# factorial ablation (single primary run, seed 42)
bash src/run_experiments_v4.sh

# multi-seed replication (seeds 1-4; seed 42 = primary run)
MAXPAR=1 bash src/run_seeds.sh

# figures and risk-coverage/calibration summaries
python src/evaluate.py
```

Requirements: Python >= 3.10, PyTorch 2.5.1, torchvision 0.20.1, scikit-learn, numpy, pandas, opencv-python, matplotlib, scipy.

## Citation

If you use this code, please cite the corresponding manuscript.

## License

MIT
