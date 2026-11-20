# Wavelet-Enhanced Evidential Deep Learning for Uncertainty-Aware Intracranial Hemorrhage Triage on Non-Contrast Head CT

Code, patient-level data partitions, and per-series rescale metadata for the study
"Wavelet-Enhanced Evidential Deep Learning for Uncertainty-Aware Clinical Decision
Support in Intracranial Hemorrhage Triage on Non-Contrast Head CT"
(submitted to *Scientific Reports*).

## Contents

```
src/        Training, evaluation, examination-level aggregation, and analysis code
analysis/   Multi-seed, calibration, and bootstrap analysis scripts with precomputed outputs
data/       Patient-level partition lists (train/val/test) and per-series rescale metadata
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
| `evaluate.py` | Risk-coverage, ROC, calibration, examination-level aggregation, and figure generation |
| `gradcam.py` | Grad-CAM visualization |
| `run_experiments_v4.sh` | Factorial ablation used for the reported results (4 configurations) |
| `run_seeds.sh` | Multi-seed replication protocol (5 independent seeds per configuration) |

### Analysis scripts (`analysis/`)

| File | Purpose |
|------|---------|
| `analyze.py` | Main test-set evaluation: Table 2 metrics, risk--coverage (Table 7), figures |
| `analyze2c.py` | Examination-level evaluation (Table 8) |
| `analyze3.py` | Calibration assessment: standard and quantile-binned ECE, Brier, NLL, prior correction, temperature scaling (Table 6) |
| `analyze4.py` | Multi-seed aggregation: mean ± SD and range (Table 3), SE-only and deep-ensemble rows (Table 4, Table 7) |
| `analyze5_supp_ci.py` | Patient-clustered bootstrap confidence intervals around the multi-seed means (Supplementary Table S1) |
| `generate_val_predictions.py` | Validation-set predictions for temperature-scaling fits |
| `make_fig6.py` | Examination-level evaluation figure |
| `results/` | Precomputed outputs of the above analyses (Tables 3, 4, 6, 7, 8, 10, 11, S1) |

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

The analysis scripts read the per-run prediction files (`probs`, `labels`, `unc`)
produced by `src/evaluate.py`. `analyze4.py` expects the primary runs under
`~/analysis_data/runs/<config>/` and the seed replications under
`~/analysis_data/runs_seeds/<config>_s<seed>/`; `analyze5_supp_ci.py` reads them
from the directory given by the `ICH_ANALYSIS_DATA` environment variable.
The precomputed JSON files under `analysis/results/` reproduce every derived
number in the manuscript without retraining.

Requirements: Python >= 3.10, PyTorch 2.5.1, torchvision 0.20.1, scikit-learn, numpy, pandas, opencv-python, matplotlib, scipy.

## Reproducibility notes (revision)

- **Portable paths.** All scripts resolve data, run, and output directories
  through `ICH_*` environment variables (see `src/config.py`); no absolute
  paths from the original research machine remain.
- **Exact Haar inverse transform.** The original implementation reconstructed
  the IDWT with a fixed permutation of the four subband contributions within
  each 2x2 block. The exact inverse Haar transform is now the default in
  `src/wavelet_attention.py` (verified to machine precision); the original
  reconstruction is preserved behind the `--idwt_legacy` flag so that the
  reported results can be reproduced exactly. Retraining with the exact
  transform is reported in the manuscript's robustness analysis.
- **SE-only configuration.** The original `run_seeds.sh`/`train.py` did not
  pass the SE-only switch to the model constructor, so runs labeled
  `se_only` in fact instantiated the full wavelet module. This is fixed;
  the SE-only results in the manuscript come from retrained runs with the
  corrected `WaveletModuleSEOnly` module. The five affected earlier runs are
  retained as `analysis/results/legacy_se_only_runs_as_wavelet_reps.json`
  (five additional replications of the wavelet configuration).
- **Inference cost.** `src/bench_latency.py` measures per-slice forward-pass
  latency and throughput of every configuration on GPU (outputs
  `latency_results.json`).

## Citation

If you use this code, please cite the corresponding manuscript.

## License

MIT
