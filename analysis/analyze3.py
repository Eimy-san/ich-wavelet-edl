"""Stage 3: consolidated calibration table (standard 15-bin ECE, Brier, NLL,
prior-corrected variants, temperature scaling) with robust clipping."""

import json
import numpy as np
from scipy.optimize import minimize_scalar

EXPS = ["baseline", "wavelet_only", "edl_only", "wavelet_edl"]
PRETTY = {
    "baseline": "Baseline (ResNet-50)",
    "wavelet_only": "+Wavelet",
    "edl_only": "+EDL",
    "wavelet_edl": "+Wavelet+EDL (full)",
}


def metrics(p, y):
    conf = np.maximum(p, 1 - p)
    correct = ((p > 0.5).astype(float) == y).astype(float)
    bins = np.linspace(0, 1, 16)
    ece = 0.0
    for i in range(15):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum():
            ece += abs(conf[m].mean() - correct[m].mean()) * m.mean()
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    nll = float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean())
    return dict(ece=float(ece), brier=float(((p - y) ** 2).mean()), nll=nll)


def prior_correct(p, pi_train=0.5, pi_test=0.5):
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    odds = (pc / (1 - pc)) * (pi_test / pi_train) * ((1 - pi_train) / (1 - pi_test))
    return odds / (1 + odds)


def fit_T(p, y):
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(pc / (1 - pc))

    def nll(T):
        z = logit / T
        pr = 1 / (1 + np.exp(-z))
        return -(y * np.log(pr) + (1 - y) * np.log(1 - pr)).mean()

    r = minimize_scalar(nll, bounds=(0.2, 5.0), method="bounded")
    return float(r.x)


out = {}
for e in EXPS:
    d_te = np.load(f"preds/{e}/test_predictions.npz")
    d_va = np.load(f"preds/{e}/val_predictions.npz")
    p_te, y = d_te["probs"].astype(np.float64), d_te["labels"]
    row = {"raw": metrics(p_te, y)}
    pi_te = y.mean()
    row["prior_corrected"] = metrics(prior_correct(p_te, 0.5, pi_te), y)
    if e in ("baseline", "wavelet_only"):
        T = fit_T(d_va["probs"].astype(np.float64), d_va["labels"])
        logit = np.log(
            np.clip(p_te, 1e-6, 1 - 1e-6) / np.clip(1 - p_te, 1e-6, 1 - 1e-6)
        )
        pT = 1 / (1 + np.exp(-logit / T))
        row["temp_scaled"] = metrics(pT, y)
        row["T"] = T
    out[e] = row
    print(PRETTY[e])
    for k in ("raw", "prior_corrected", "temp_scaled"):
        if k in row:
            print(
                f"  {k:16s}",
                {m: round(v, 4) for m, v in row[k].items()},
                ("T=%.3f" % row["T"]) if k == "temp_scaled" else "",
            )

json.dump(out, open("analysis3_results.json", "w"), indent=2)
print("saved")
