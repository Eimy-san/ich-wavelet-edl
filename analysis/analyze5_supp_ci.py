#!/usr/bin/env python3
"""Patient-clustered bootstrap CIs around multi-seed mean slice metrics (Supplementary Table S1).

For each bootstrap replicate: resample test patients with replacement, compute each
run's metric on the resampled slice set, average across the 5 runs, and take 2.5/97.5 percentiles.
"""

import numpy as np, csv, json, os

# DATA holds the per-run prediction archives:
#   DATA/preds/<config>/test_predictions.npz   primary runs (seed 42)
#   DATA/multiseed/<config>_s<seed>.npz        seed replications (seeds 1-4)
#   DATA/test.csv                              test partition (order defines the label alignment)
DATA = os.environ.get("ICH_ANALYSIS_DATA", ".")
MS = os.path.join(DATA, "multiseed")
METRICS = ["auc", "f1", "acc", "sens", "spec"]


def metric_on(idx, labels, preds, hem):
    l = labels[idx]
    p = preds[idx]
    h = hem[idx]
    pos = l == 1
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0 or len(np.unique(l)) < 2:
        return None
    try:
        auc = roc_auc_local(l, h)
    except Exception:
        return None
    return {
        "auc": auc,
        "f1": f1_local(l, p),
        "acc": (p == l).mean(),
        "sens": (p[pos] == 1).mean(),
        "spec": (p[neg] == 0).mean(),
    }


def roc_auc_local(l, h):
    # rank-based AUC with tie handling
    order = np.argsort(h, kind="mergesort")
    ranks = np.empty(len(h), float)
    sh = h[order]
    i = 0
    r = 1
    while i < len(sh):
        j = i
        while j + 1 < len(sh) and sh[j + 1] == sh[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n1 = (l == 1).sum()
    n0 = (l == 0).sum()
    return (ranks[l == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def f1_local(l, p):
    tp = ((p == 1) & (l == 1)).sum()
    fp = ((p == 1) & (l == 0)).sum()
    fn = ((p == 0) & (l == 1)).sum()
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def main():
    rows = list(csv.DictReader(open(os.path.join(DATA, "test.csv"))))
    pids = np.array([r["patient_id"] for r in rows])
    uniq = np.unique(pids)
    labels_all, preds_all, hem_all, runs_meta = [], [], [], []
    cfg_runs = {}
    for cfg in ["baseline", "wavelet_only", "edl_only", "wavelet_edl", "se_only"]:
        runs = []
        if cfg != "se_only":
            d = np.load(os.path.join(DATA, "preds", cfg, "test_predictions.npz"))
            runs.append((d["probs"], d["labels"]))
        for s in [1, 2, 3, 4]:
            d = np.load(os.path.join(MS, f"{cfg}_s{s}.npz"))
            runs.append((d["probs"], d["labels"]))
        p42 = os.path.join(MS, f"{cfg}_s42.npz")
        if os.path.exists(p42):
            d = np.load(p42)
            runs.append((d["probs"], d["labels"]))
        cfg_runs[cfg] = runs
    # verify all label arrays identical & aligned with test.csv order
    l0 = cfg_runs["baseline"][0][1]
    for cfg, runs in cfg_runs.items():
        for p, l in runs:
            assert (l == l0).all(), cfg
    d = np.load(os.path.join(DATA, "preds", "baseline", "test_predictions.npz"))
    hem0 = d["probs"]
    # sanity: labels vs csv 'any'
    csv_lab = np.array([int(float(r["any"])) for r in rows])
    print("label alignment:", (csv_lab == l0).mean())

    rng = np.random.RandomState(2026)
    B = 1000
    out = {}
    for cfg, runs in cfg_runs.items():
        preds = [(p > 0.5).astype(int) for p, l in runs]
        hems = [p for p, l in runs]
        stats = {m: [] for m in METRICS}
        for b in range(B):
            sel_pat = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.where(pids == x)[0] for x in sel_pat])
            vals = [metric_on(idx, l0, preds[r], hems[r]) for r in range(len(runs))]
            vals = [v for v in vals if v]
            if not vals:
                continue
            for m in METRICS:
                stats[m].append(np.mean([v[m] for v in vals]))
        out[cfg] = {
            m: [
                float(np.percentile(stats[m], 2.5)),
                float(np.percentile(stats[m], 97.5)),
                float(np.mean(stats[m])),
            ]
            for m in METRICS
        }
        print(cfg, {m: ["%.4f" % x for x in out[cfg][m]] for m in METRICS})
    json.dump(out, open("supp_s1_cis.json", "w"), indent=1)


if __name__ == "__main__":
    main()
