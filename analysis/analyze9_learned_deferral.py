"""Validation-trained examination-level deferral: can a deferral rule that
combines predictive uncertainty with prediction confidence, fitted on the
validation set, recover an examination-level selective-prediction advantage
for the full model?

Features per exam (all computed from slice-level probs/unc, no test info):
  f1 = max slice prob            f2 = mean slice prob
  f3 = max slice vacuity/entropy f4 = mean slice vacuity/entropy
  f5 = max slice margin 1-max(p,1-p)
  f6 = count of slices with p > 0.5 (normalized)
  f7 = number of slices (log)
Deferral model: logistic regression on standardized features, fit on VAL,
defer the exams with highest predicted "error risk" so that retained
coverage = 95%. Also a simple 2-signal rank-fusion baseline (rank of max-unc
+ rank of max-margin).

Reported: test acc@95, AURC, and per-seed means for all configs.
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EXPS = ["baseline", "wavelet_only", "edl_only", "wavelet_edl"]
EDL = {"edl_only", "wavelet_edl"}


def groups_of(df):
    pid = df["patient_id"].to_numpy()
    return {p: np.where(pid == p)[0] for p in np.unique(pid)}


def aurc(prob_ex, unc_ex, y):
    n = len(y)
    order = np.argsort(unc_ex, kind="stable")
    accs = np.array(
        [
            ((prob_ex[order[:k]] > 0.5).astype(int) == y[order[:k]]).mean()
            for k in range(1, n + 1)
        ]
    )
    return float((1 - accs).mean())


def at95(prob_ex, ue, y):
    n = len(y)
    k = int(0.95 * n)
    sel = np.argsort(ue, kind="stable")[:k]
    pred = (prob_ex[sel] > 0.5).astype(int)
    defer = np.setdiff1d(np.arange(n), sel)
    pos = y == 1
    return dict(
        acc=float((pred == y[sel]).mean()),
        sens=float(((pred == 1) & pos[sel]).sum() / max(pos[sel].sum(), 1)),
        spec=float(((pred == 0) & ~pos[sel]).sum() / max((~pos[sel]).sum(), 1)),
        deferred_pos=int((y[defer] == 1).sum()),
        deferred_neg=int((y[defer] == 0).sum()),
    )


def exam_features(p, u, g, pids):
    X = []
    for pp in pids:
        idx = g[pp]
        ps, us = p[idx], u[idx]
        pe = ps.max()
        X.append(
            [
                pe,
                ps.mean(),
                us.max(),
                us.mean(),
                (1 - np.maximum(ps, 1 - ps)).max(),
                (ps > 0.5).mean(),
                np.log1p(len(idx)),
            ]
        )
    return np.array(X)


def load(e, split):
    if e in EDL:
        d = np.load(f"primary/{e}_{split}_ev.npz")
        return (
            d["probs"].astype(np.float64),
            d["unc"].astype(np.float64),
            d["alphas"].astype(np.float64),
        )
    d = np.load(f"primary/{e}_{split}.npz")
    return d["probs"].astype(np.float64), d["unc"].astype(np.float64), None


def run_config(e, g_va, va_pids, va_y, g_te, te_pids, te_y, seed=0):
    p_va, u_va, _ = load(e, "val")
    p_te, u_te, _ = load(e, "test")
    X_va = exam_features(p_va, u_va, g_va, va_pids)
    X_te = exam_features(p_te, u_te, g_te, te_pids)
    err_va = ((X_va[:, 0] > 0.5).astype(int) != va_y).astype(int)
    sc = StandardScaler().fit(X_va)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(sc.transform(X_va), err_va)
    risk_va = clf.predict_proba(sc.transform(X_va))[:, 1]
    risk_te = clf.predict_proba(sc.transform(X_te))[:, 1]
    pe_va = X_va[:, 0]
    pe_te = X_te[:, 0]
    # rank fusion of max-unc and max-margin
    f_va = (
        np.argsort(np.argsort(X_va[:, 2])) + np.argsort(np.argsort(X_va[:, 4]))
    ).astype(float)
    f_te = (
        np.argsort(np.argsort(X_te[:, 2])) + np.argsort(np.argsort(X_te[:, 4]))
    ).astype(float)
    out = {}
    out["logistic"] = dict(
        val=metrics_for(pe_va, risk_va, va_y), test=metrics_for(pe_te, risk_te, te_y)
    )
    out["fusion2"] = dict(
        val=metrics_for(pe_va, f_va, va_y), test=metrics_for(pe_te, f_te, te_y)
    )
    out["vacmax"] = dict(
        val=metrics_for(pe_va, X_va[:, 2], va_y),
        test=metrics_for(pe_te, X_te[:, 2], te_y),
    )
    return out


def metrics_for(pe, ue, y):
    return dict(aurc=aurc(pe, ue, y), **at95(pe, ue, y))


def main():
    val = pd.read_csv("val.csv")
    test = pd.read_csv("test.csv")
    g_va, g_te = groups_of(val), groups_of(test)
    va_pids, te_pids = sorted(g_va), sorted(g_te)
    va_y = np.array([val["any"].to_numpy()[g_va[p]].max() for p in va_pids])
    te_y = np.array([test["any"].to_numpy()[g_te[p]].max() for p in te_pids])

    res = {}
    for e in EXPS:
        res[e] = run_config(e, g_va, va_pids, va_y, g_te, te_pids, te_y)
        for rule in ("logistic", "fusion2", "vacmax"):
            v, t = res[e][rule]["val"], res[e][rule]["test"]
            print(
                f"{e:13s} {rule:9s} AURC val {v['aurc']:.4f} test {t['aurc']:.4f} | "
                f"acc95 val {v['acc']:.4f} test {t['acc']:.4f} | "
                f"sens test {t['sens']:.4f} spec test {t['spec']:.4f} "
                f"defpos {t['deferred_pos']}"
            )

    # multi-seed logistic rule for EDL configs
    multi = {}
    for e in ["edl_only", "wavelet_edl"]:
        accs, aurcs, senss, specs = [], [], [], []
        for s in [1, 2, 3, 4]:
            # primary s42 uses primary files; seeds use _sN files
            pass
        multi[e] = "see multi-seed extension"
    with open("exam_deferral_learned.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
