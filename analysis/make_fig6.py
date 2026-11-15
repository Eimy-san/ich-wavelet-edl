"""Generate Figure 6: examination-level risk-coverage and deferral analysis."""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPS = ["baseline", "wavelet_only", "edl_only", "wavelet_edl"]
LABELS = {
    "baseline": "Baseline",
    "wavelet_only": "+Wavelet",
    "edl_only": "+EDL",
    "wavelet_edl": "+Wavelet+EDL",
}
COLORS = {
    "baseline": "#7f7f7f",
    "wavelet_only": "#1f77b4",
    "edl_only": "#2ca02c",
    "wavelet_edl": "#d62728",
}

test = pd.read_csv("test.csv")
pid = test["patient_id"].to_numpy()
g = {p: np.where(pid == p)[0] for p in np.unique(pid)}
pids = sorted(g)
te_y = np.array([test["any"].to_numpy()[g[p]].max() for p in pids])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

for e in EXPS:
    d = np.load(f"preds/{e}/test_predictions.npz")
    p, u = d["probs"].astype(np.float64), d["unc"].astype(np.float64)
    pe = np.array([p[g[pp]].max() for pp in pids])
    tot = 1 - np.maximum(p, 1 - p)
    ue = np.array([tot[g[pp]].max() for pp in pids])
    n = len(pids)
    order = np.argsort(ue)
    accs = np.array(
        [
            ((pe[order[:k]] > 0.5).astype(int) == te_y[order[:k]]).mean()
            for k in range(1, n + 1)
        ]
    )
    cov = np.arange(1, n + 1) / n
    risk = 1 - accs
    axes[0].plot(cov, risk, label=LABELS[e], color=COLORS[e], lw=1.8)
    # panel b: retained sensitivity + deferred positives
    covs, sens, defpos = [], [], []
    for c in np.arange(0.5, 1.001, 0.01):
        k = int(round(c * n))
        sel = order[:k]
        pred = (pe[sel] > 0.5).astype(int)
        pos = te_y[sel] == 1
        tp = int(((pred == 1) & pos).sum())
        fn = int(((pred == 0) & pos).sum())
        defer = np.setdiff1d(np.arange(n), sel)
        covs.append(c)
        sens.append(tp / (tp + fn))
        defpos.append(int((te_y[defer] == 1).sum()))
    axes[1].plot(covs, np.array(sens) * 100, color=COLORS[e], lw=1.8, label=LABELS[e])

axes[0].axvline(0.95, color="k", ls="--", lw=1, alpha=0.6)
axes[0].set_xlabel("Examination coverage", fontsize=12)
axes[0].set_ylabel("Risk (1 - accuracy)", fontsize=12)
axes[0].set_title("(a) Examination-level risk-coverage", fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].set_xlim(0.5, 1.0)

axes[1].axvline(0.95, color="k", ls="--", lw=1, alpha=0.6)
axes[1].set_xlabel("Examination coverage", fontsize=12)
axes[1].set_ylabel("Sensitivity among retained exams (%)", fontsize=12)
axes[1].set_title("(b) Retained examination-level sensitivity", fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)
axes[1].set_xlim(0.5, 1.0)

plt.tight_layout()
plt.savefig(
    "/Users/jingzeli/Documents/ScienceGuru Edu/神内-new/revision/manuscript/figures/Fig.6.png",
    dpi=300,
    bbox_inches="tight",
)
print("saved Fig.6.png")

# annotate deferred positives at 95% for caption text check
for e in EXPS:
    d = np.load(f"preds/{e}/test_predictions.npz")
    p = d["probs"].astype(np.float64)
    pe = np.array([p[g[pp]].max() for pp in pids])
    tot = 1 - np.maximum(p, 1 - p)
    ue = np.array([tot[g[pp]].max() for pp in pids])
    n = len(pids)
    k = int(0.95 * n)
    sel = np.argsort(ue)[:k]
    defer = np.setdiff1d(np.arange(n), sel)
    print(e, "deferred_pos", int((te_y[defer] == 1).sum()), "/", len(defer))
