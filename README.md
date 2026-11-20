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
| `dump_evidence.py` | Re-evaluates a checkpoint on validation and test splits and exports per-slice probabilities and Dirichlet evidence (alphas) for the deferral-rule study |
| `analyze8_exam.py` | Validation-set examination-level deferral-rule ablation: prespecified, decision-coupled, Dirichlet-decomposition, and learned rules |
| `analyze9_learned_deferral.py` | Validation-fitted logistic deferral-rule exploration |
| `analyze10_exam_final.py` | Per-seed rule-selection sensitivity analysis |
| `analyze11_frozen.py` | Frozen-rule protocol: validation-selected decision-coupled rule applied unchanged to all configurations and seeds (Table 12) |
| `results/` | Precomputed outputs of the above analyses (Tables 3, 4, 6, 7, 8, 10, 11, 12, S1) |

### Data files (`data/`)

| File | Content |
|------|---------|
| `train.csv`, `val.csv`, `test.csv` | Patient-level partition lists: image path, patient ID, slice-level hemorrhage labels (`any` and five subtypes). 47,368/10,204/9,560 slices from 1,179/253/253 patients. |
| `rsna_rescale_values.csv` | Per-series rescale slope and intercept (from DICOM headers of the RSNA stage-2 training set) keyed by patient/study/series; used to convert 16-bit PNG archive values to Hounsfield units. |

The RSNA-ICH images themselves are available from the
[RSNA 2019 Brain CT Hemorrhage Challenge](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection).

## Reproducing the experiments

The two inverse-Haar-transform variants used by the reported runs:

| Reported results | Transform used | Flag |
|---|---|---|
| Primary ablation (Table 2, seed 42) and multi-seed replications (Tables 3, 7, 8) of baseline / +Wavelet / +EDL / full | original (permutation) IDWT | `--idwt_legacy` |
| Retrained SE-only ablation (Table 4, five seeds) | exact IDWT | default (no flag) |
| Exact-transform robustness runs (Section 2.4, seed 42) | exact IDWT | default (no flag) |

```bash
# primary ablation (Table 2, seed 42) + SE-only ablation (Table 4, seed 42):
# reproduces the exact transforms used for each reported configuration
bash src/run_experiments_v4.sh

# multi-seed replication (main configurations: seeds 1-4, legacy IDWT;
# SE-only: seeds 42-4, exact IDWT)
MAXPAR=1 bash src/run_seeds.sh

# exact-transform robustness runs quoted in Section 2.4 (seed 42)
python src/train.py --exp_name wavelet_idwtfix --use_wavelet --wavelet_pos layer2 --se_reduction 16 --seed 42
python src/train.py --exp_name wavelet_edl_idwtfix --use_wavelet --wavelet_pos layer2 --se_reduction 16 --use_edl --edl_loss log --edl_coeff 0.03 --seed 42

# figures and risk-coverage/calibration summaries
python src/evaluate.py
```

`src/run_experiments_v4.sh` and `src/run_seeds.sh` pass `--idwt_legacy` to the
four main configurations and leave the SE-only runs on the exact-transform
default, matching the transforms of the reported runs; the seeds 1-4
replications of the main configurations therefore reproduce Table 3 exactly.
`src/dump_evidence.py` re-evaluates a checkpoint on the validation
and test splits and stores the Dirichlet evidence (alphas) for the
examination-level deferral-rule study (Table 12): `analyze8_exam.py` compares
the prespecified rule with decision-coupled and Dirichlet-decomposition
alternatives on the validation split, and `analyze11_frozen.py` applies the
validation-selected decision-coupled rule (top-3-mean probability aggregation;
mean uncertainty of the three most suspicious slices), frozen, to all
configurations and seeds. The precomputed outputs are
`analysis/results/exam_rule_ablation.json`, `exam_final.json`,
`exam_frozen.json`, and `exam_margin_multiseed.json`.

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
