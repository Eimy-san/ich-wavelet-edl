"""Transform-matched attribution: wavelet (exact IDWT) vs SE-only (exact IDWT).

Computes per-seed test AUC, Acc@95, AURC for wavelet_idwtfix seeds and
se_only seeds, plus legacy wavelet/baseline for context, with paired
permutation tests on matched seeds. Saves JSON to results.
"""

import csv
import glob
import json
import os
import sys

import numpy as np

A = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(A, "idwtfix_seeds")


def load_csv_labels(split):
    rows = list(csv.DictReader(open(os.path.join(A, f"{split}.csv"))))
    return np.array([int(float(r["any"])) for r in rows])


def auc_mw(probs, y):
    from scipy.stats import mannwhitneyu

    pos, neg = probs[y == 1], probs[y == 0]
    return mannwhitneyu(pos, neg, alternative="greater").statistic / (
        len(pos) * len(neg)
    )


def selective(probs, unc, y):
    order = np.argsort(unc, kind="stable")
    n = len(y)
    risk, accs = [], []
    for frac in np.linspace(0.05, 1.0, 20):
        keep = int(round(frac * n))
        kept = order[:keep]
        pred = (probs[kept] >= 0.5).astype(int)
        risk.append(1.0 - (pred == y[kept]).mean())
    risk = np.array(risk)
    aurc = risk.mean()
    keep = int(round(0.95 * n))
    kept = order[:keep]
    acc95 = ((probs[kept] >= 0.5).astype(int) == y[kept]).mean()
    return acc95, aurc


def get(path, labels):
    z = np.load(path)
    assert (z["labels"] == labels).all(), path
    probs, unc = z["probs"], z["unc"]
    return auc_mw(probs, labels), *selective(probs, unc, labels)


def paired_perm(a, b, n=100000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    obs = d.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(n, len(d)))
    null = (signs * d).mean(axis=1)
    return obs, (np.abs(null) >= abs(obs)).mean()


def main():
    y = load_csv_labels("test")
    res = {}

    def collect(pattern, run_label):
        import re

        vals = {}
        for path in sorted(glob.glob(pattern)):
            m = re.search(r"_s(\d+)", os.path.basename(path))
            seed = int(m.group(1)) if m else -1
            auc, acc95, aurc = get(path, y)
            vals[seed] = {
                "auc": round(auc, 4),
                "acc95": round(acc95, 4),
                "aurc": round(aurc, 4),
            }
        if vals:
            res[run_label] = vals
            a = np.array([[v["auc"], v["acc95"], v["aurc"]] for v in vals.values()])
            res[run_label + "_mean"] = {
                "auc": round(a[:, 0].mean(), 4),
                "acc95": round(a[:, 1].mean(), 4),
                "aurc": round(a[:, 2].mean(), 4),
                "sd_auc": round(a[:, 0].std(ddof=1), 4),
                "sd_acc95": round(a[:, 1].std(ddof=1), 4),
                "n": len(vals),
            }

    collect(os.path.join(OUT, "wavelet_idwtfix_s*.npz"), "wavelet_idwtfix")
    collect(os.path.join(A, "seeds", "se_only_s*_test.npz"), "se_only")
    collect(os.path.join(A, "seeds", "wavelet_s*_test.npz"), "wavelet_legacy")
    collect(os.path.join(OUT, "wavelet_edl_idwtfix_s*.npz"), "wavelet_edl_idwtfix")
    collect(os.path.join(A, "seeds", "baseline_s*_test.npz"), "baseline_legacy")

    if "wavelet_idwtfix" in res and "se_only" in res:
        common = sorted(set(res["wavelet_idwtfix"]) & set(res["se_only"]))
        for metric in ["auc", "acc95", "aurc"]:
            obs, p = paired_perm(
                [res["wavelet_idwtfix"][s][metric] for s in common],
                [res["se_only"][s][metric] for s in common],
            )
            res[f"paired_wavelet_idwtfix_vs_se_only_{metric}"] = {
                "diff_mean": round(obs, 4),
                "p": round(p, 4),
                "n_pairs": len(common),
                "seeds": common,
                "note": "paired by seed, preliminary until 5 seeds",
            }

    out = os.path.join(A, "idwt_attribution.json")
    json.dump(res, open(out, "w"), indent=1)
    print("saved", out)
    for k, v in res.items():
        print(k, v)


if __name__ == "__main__":
    main()
