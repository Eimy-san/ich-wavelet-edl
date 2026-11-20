#!/usr/bin/env python3
"""Post-fix analysis for the corrected SE-only runs and the IDWT robustness runs.

Inputs (downloaded from the server):
  fixed_runs/se_only_s{42,1,2,3,4}.npz       corrected SE-only test predictions
  fixed_runs/wavelet_idwtfix_s42.npz         wavelet retrained with exact IDWT
  fixed_runs/wavelet_edl_idwtfix_s42.npz     wavelet+EDL retrained with exact IDWT
  multiseed/{cfg}_s{42,1,2,3,4}.npz          four main configurations, five seeds
  test.csv                                   test partition (patient_id per row)

Outputs:
  se_only_corrected.json   per-seed metrics, mean+/-SD+range, paired permutation
                           tests vs baseline/wavelet, S1-style cluster bootstrap
  supp_s1_cis.json         REBUILT with 2,000 replications for all five configs
  idwtfix_results.json     slice metrics of the IDWT-robustness runs with references

Replicates analyze5_supp_ci.py conventions exactly: patients resampled with
replacement (a patient drawn twice contributes its slices twice),
RandomState(2026), per-replicate metric per run averaged across runs.
"""

import csv
import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
FIXED = os.path.join(HERE, "fixed_runs")
MS = os.path.join(HERE, "multiseed")
SEEDS = [42, 1, 2, 3, 4]
METRICS = ["auc", "f1", "acc", "sens", "spec"]


def load(path):
    d = np.load(path)
    return d["probs"], d["labels"], d["unc"]


def metric_on(idx, labels, preds, hem):
    l = labels[idx]
    h = hem[idx]
    p = preds[idx]
    pos = l == 1
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    try:
        auc = roc_auc_score(l, h)
    except Exception:
        return None
    return {
        "auc": auc,
        "f1": f1_score(l, p, pos_label=1, zero_division=0),
        "acc": (p == l).mean(),
        "sens": p[pos].mean(),
        "spec": (p[neg] == 0).mean(),
    }


def slice_metrics(probs, labels, unc=None):
    hem = probs
    pred = (hem >= 0.5).astype(int)
    out = {}
    out["auc"] = roc_auc_score(labels, hem)
    out["f1"] = f1_score(labels, pred, pos_label=1, zero_division=0)
    out["acc"] = accuracy_score(labels, pred)
    pos = labels == 1
    neg = ~pos
    out["sens"] = (pred[pos] == 1).mean()
    out["spec"] = (pred[neg] == 0).mean()
    if unc is None:
        unc = 1.0 - np.maximum(hem, 1.0 - hem)
    order = np.argsort(unc)
    correct = (pred == labels)[order].astype(float)
    N = len(labels)
    risk = 1 - np.cumsum(correct) / np.arange(1, N + 1)
    out["aurc"] = risk.mean()
    out["acc95"] = correct[: int(0.95 * N)].mean()
    conf = np.maximum(hem, 1.0 - hem)
    bins = np.linspace(0, 1, 16)
    ece = 0.0
    for b in range(15):
        m = (conf > bins[b]) & (conf <= bins[b + 1])
        if m.sum():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    out["ece"] = ece
    return out


def summarize(runs):
    out = {}
    for k in ["auc", "f1", "acc", "sens", "spec", "aurc", "acc95", "ece"]:
        v = [runs[s][k] for s in SEEDS]
        out[k] = {
            "mean": float(np.mean(v)),
            "sd": float(np.std(v, ddof=1)),
            "min": float(min(v)),
            "max": float(max(v)),
            "per_seed": {str(s): float(runs[s][k]) for s in SEEDS},
        }
    return out


def perm_paired(diffs):
    d = np.asarray(diffs, float)
    n = len(d)
    idx = np.arange(1 << n)
    signs = ((idx[:, None] >> np.arange(n)) & 1) * 2 - 1
    stats = np.abs((signs * d).sum(1))
    obs = abs(d.sum())
    return float((stats >= obs - 1e-12).mean()), float(d.mean())


def s1_bootstrap(cfg_runs, labels, pids, uniq, B=2000, seed=2026):
    rng = np.random.RandomState(seed)
    preds = [(p > 0.5).astype(int) for p, l, u in cfg_runs]
    hems = [p for p, l, u in cfg_runs]
    stats = {m: [] for m in METRICS}
    for b in range(B):
        sel_pat = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(pids == x)[0] for x in sel_pat])
        vals = [metric_on(idx, labels, preds[r], hems[r]) for r in range(len(cfg_runs))]
        vals = [v for v in vals if v]
        if not vals:
            continue
        for m in METRICS:
            stats[m].append(np.mean([v[m] for v in vals]))
    return {
        m: [
            float(np.percentile(stats[m], 2.5)),
            float(np.percentile(stats[m], 97.5)),
            float(np.mean(stats[m])),
        ]
        for m in METRICS
    }


def main_idwtfix():
    pass


def idwtfix_section():
    idwt = {}
    msf = json.load(open(os.path.join(HERE, "multiseed_final.json")))
    for name, refkey in (
        ("wavelet_idwtfix", "wavelet_only"),
        ("wavelet_edl_idwtfix", "wavelet_edl"),
    ):
        fp = os.path.join(FIXED, f"{name}_s42.npz")
        if not os.path.exists(fp):
            print(f"skip {name}: npz not yet available")
            continue
        p, l, u = load(fp)
        assert np.array_equal(labels_ref, l), f"label mismatch {name}"
        idwt[name] = slice_metrics(p, l, u)
        idwt[f"ref_{refkey}_s42"] = msf[refkey]["42"]
    if idwt:
        with open(os.path.join(HERE, "idwtfix_results.json"), "w") as f:
            json.dump(idwt, f, indent=2)
        print("idwtfix_results.json written")


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "test.csv"))))
    pids = np.array([r["patient_id"] for r in rows])
    uniq = np.unique(pids)

    labels = None
    se_runs = []
    for s in SEEDS:
        p, l, u = load(os.path.join(FIXED, f"se_only_s{s}.npz"))
        if labels is None:
            labels = l
        assert np.array_equal(labels, l), f"label mismatch se_only_s{s}"
        se_runs.append((p, l, u))

    m = {s: slice_metrics(p, l, u) for s, (p, l, u) in zip(SEEDS, se_runs)}
    results = {"per_seed": {str(s): m[s] for s in SEEDS}, "summary": summarize(m)}

    msf = json.load(open(os.path.join(HERE, "multiseed_final.json")))
    tests = {}
    for other in ("baseline", "wavelet_only"):
        for k in ("auc", "sens", "spec"):
            diffs = [m[s][k] - msf[other][str(s)][k] for s in SEEDS]
            pp, md = perm_paired(diffs)
            tests[f"se_only_vs_{other}_{k}"] = {
                "per_seed_diffs": [round(x, 4) for x in diffs],
                "mean_diff": round(md, 4),
                "perm_p": pp,
                "n_positive_diffs": int(sum(d > 0 for d in diffs)),
            }
    results["paired_tests"] = tests

    with open(os.path.join(HERE, "se_only_corrected.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("se_only_corrected.json written")

    # Rebuild supplementary Table S1 for all five configs at B=2000
    s1 = {}
    for cfg in ["baseline", "wavelet_only", "edl_only", "wavelet_edl", "se_only"]:
        runs = []
        for s in SEEDS:
            if cfg == "se_only":
                runs.append(se_runs[SEEDS.index(s)])
            else:
                p, l, u = load(os.path.join(MS, f"{cfg}_s{s}.npz"))
                assert np.array_equal(labels, l), f"label mismatch {cfg}_s{s}"
                runs.append((p, l, u))
        s1[cfg] = s1_bootstrap(runs, labels, pids, uniq, B=2000)
        print(cfg, {k: ["%.4f" % x for x in v] for k, v in s1[cfg].items()})
    with open(os.path.join(HERE, "supp_s1_cis.json"), "w") as f:
        json.dump(s1, f, indent=1)
    print("supp_s1_cis.json rebuilt (B=2000)")

    # IDWT robustness runs
    global labels_ref
    labels_ref = labels
    idwtfix_section()
    print(json.dumps(results["summary"], indent=1)[:1500])


if __name__ == "__main__":
    main()
