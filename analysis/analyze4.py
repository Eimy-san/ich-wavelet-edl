#!/usr/bin/env python3
"""Multi-seed aggregation: Table 3 (mean+/-SD+range), Table 4 (SE-only, ensemble), Table 7 ensemble row.

Reads test_predictions.npz per run. npz keys: probs (N,) hemorrhage probability, labels (N,) 0/1, unc (N,) model uncertainty.
"""

import numpy as np, json, os, glob
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

ROOT = os.path.expanduser(
    "~/analysis_data"
)  # local dir holding runs/<cfg>/ and runs_seeds/<cfg>_s<N>/
CFGS = ["baseline", "wavelet_only", "edl_only", "wavelet_edl"]
SEEDS = [42, 1, 2, 3, 4]


def load(path):
    d = np.load(path)
    return d["probs"], d["labels"], d["unc"]


def slice_metrics(probs, labels, unc=None):
    hem = probs
    pred = (hem >= 0.5).astype(int)
    out = {}
    out["auc"] = roc_auc_score(labels, hem)
    out["f1"] = f1_score(labels, pred, pos_label=1, zero_division=0)
    out["acc"] = accuracy_score(labels, pred)
    pos = labels == 1
    neg = ~pos
    out["sens"] = (pred[pos] == 1).mean() if pos.any() else float("nan")
    out["spec"] = (pred[neg] == 0).mean() if neg.any() else float("nan")
    # selective metrics (slice level): uncertainty = 1 - max prob
    if unc is None:
        unc = 1.0 - np.maximum(hem, 1.0 - hem)
    order = np.argsort(unc)  # most confident first
    labels_o = labels[order]
    correct = (pred == labels)[order].astype(float)
    N = len(labels)
    risk = 1 - np.cumsum(correct) / np.arange(1, N + 1)
    out["aurc"] = risk.mean()
    out["acc95"] = correct[: int(0.95 * N)].mean()
    # ECE 15 equal-width bins on confidence
    conf = np.maximum(hem, 1.0 - hem)
    bins = np.linspace(0, 1, 16)
    ece = 0.0
    for b in range(15):
        m = (conf > bins[b]) & (conf <= bins[b + 1])
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    out["ece"] = ece
    return out


def collect():
    res = {}
    for cfg in CFGS:
        runs = []
        for s in SEEDS:
            if s == 42:
                p = os.path.join(ROOT, "runs", cfg, "test_predictions.npz")
            else:
                p = os.path.join(
                    ROOT, "runs_seeds", f"{cfg}_s{s}", "test_predictions.npz"
                )
            if not os.path.exists(p):
                print("missing:", p)
                continue
            probs, labels, unc = load(p)
            runs.append((s, slice_metrics(probs, labels, unc), probs, labels, unc))
        res[cfg] = runs
    return res


def fmt_table3(res):
    print("=== Table 3: mean +/- sd (range) over", len(res["baseline"]), "runs")
    for cfg in CFGS:
        runs = res[cfg]
        row = [cfg]
        for k in ["auc", "f1", "acc", "sens", "spec"]:
            v = np.array([r[1][k] for r in runs])
            row.append(
                f"{v.mean():.4f}±{v.std(ddof=1):.4f} [{v.min():.4f},{v.max():.4f}]"
            )
        print(" & ".join(row[:1] + [f"{x.split('±')[0]}" for x in row[1:]]))
        for k, v in zip(["auc", "f1", "acc", "sens", "spec"], row[1:]):
            print(f"   {k}: {v}")
    return res


def ensemble(res, out_json):
    base = res["baseline"]
    if len(base) < 5:
        print("ensemble needs 5 baseline runs, have", len(base))
        return
    probs = np.mean(np.stack([r[2] for r in base]), axis=0)
    labels = base[0][3]
    m = slice_metrics(probs, labels)
    print("=== Ensemble slice metrics:", {k: round(v, 4) for k, v in m.items()})
    # exam level for ensemble
    import csv

    test_csv = os.path.join(ROOT, "test.csv")
    rows = list(csv.DictReader(open(test_csv)))
    exam_ids = [r["patient"] if "patient" in r else r["series"] for r in rows]
    assert len(exam_ids) == len(labels), (len(exam_ids), len(labels))
    ex = {}
    for i, e in enumerate(exam_ids):
        ex.setdefault(e, []).append(i)
    ytrue, ypred, yunc = [], [], []
    for e, idx in ex.items():
        # aggregate rule: maximum slice probability (Section 2.7 of the manuscript)
        pmax = probs[idx].max()
        ytrue.append(int(labels[idx].max()))
        ypred.append(int(pmax >= 0.5))
        yunc.append(1.0 - np.maximum(pmax, 1.0 - pmax))
    ytrue = np.array(ytrue)
    ypred = np.array(ypred)
    yunc = np.array(yunc)
    exauc = roc_auc_score(
        ytrue, np.array([probs[idx].max() for idx in ex.values()])
    )
    pos = ytrue == 1
    neg = ~pos
    exsens = (ypred[pos] == 1).mean()
    exspec = (ypred[neg] == 0).mean()
    exacc = (ypred == ytrue).mean()
    order = np.argsort(yunc)  # confident first
    keep = order[: int(0.95 * len(ytrue))]
    exacc95 = (ypred[keep] == ytrue[keep]).mean()
    k = int(0.05 * len(ytrue))
    deferred_idx = np.argsort(yunc)[-k:]
    deferred_pos = int(ytrue[deferred_idx].sum())
    print(
        "=== Ensemble exam metrics: AUC %.4f sens %.4f spec %.4f acc %.4f acc95 %.4f deferred+ %d/%d"
        % (exauc, exsens, exspec, exacc, exacc95, deferred_pos, k)
    )
    json.dump(
        {
            "ensemble_slice": m,
            "ensemble_exam": {
                "auc": exauc,
                "sens": exsens,
                "spec": exspec,
                "acc": exacc,
                "acc95": exacc95,
                "deferred_pos": deferred_pos,
                "k": k,
            },
        },
        open(out_json, "w"),
        indent=1,
    )


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else ROOT
    ROOT = root
    res = collect()
    fmt_table3(res)
    # SE-only handled separately if present in runs_seeds
    se_runs = []
    for s in SEEDS:
        p = (
            os.path.join(root, "runs_seeds", f"se_only_s{s}", "test_predictions.npz")
            if s != 42
            else os.path.join(root, "runs", "se_only", "test_predictions.npz")
        )
        if os.path.exists(p):
            probs, labels = load(p)
            se_runs.append(slice_metrics(probs, labels))
    if se_runs:
        print(
            "=== SE-only mean:",
            {
                k: round(np.mean([r[k] for r in se_runs]), 4)
                for k in ["auc", "f1", "acc", "sens", "spec"]
            },
            "n=",
            len(se_runs),
        )
    json.dump(
        {cfg: {str(s): m for s, m, _, _, _ in runs} for cfg, runs in res.items()},
        open(os.path.join(root, "multiseed_results.json"), "w"),
        indent=1,
    )
    ensemble(res, os.path.join(root, "ensemble_results.json"))
