"""Final examination-level analysis for the revision.

Protocol (no test-set selection):
  1. Exam-level classifier aggregation: candidates {max, top3-mean, noisy-OR};
     selected per configuration on the VALIDATION set by AUC.
  2. Exam-level deferral score: candidates {max-uncertainty (prespecified),
     decision-coupled (uncertainty of the argmax-prob slice),
     decision-coupled top-3 variants, rank fusion, validation-fitted logistic};
     selected per configuration on the VALIDATION set by AURC.
  3. Selected choices are applied unchanged to the test split at 95% coverage.

Run for the primary runs (seed 42) and the four replication seeds of every
configuration; reports mean +/- SD across the five seeds.

Inputs: primary/{cfg}_{split}.npz (+ _ev.npz with alphas for EDL configs),
        seeds/{cfg}_s{seed}_{split}[_ev].npz
Output: exam_final.json
"""

import json
import os
import numpy as np
import pandas as pd
from scipy.special import digamma
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EXPS = ["baseline", "wavelet_only", "edl_only", "wavelet_edl", "se_only"]
EDL = {"edl_only", "wavelet_edl"}
SEEDS = [42, 1, 2, 3, 4]


def groups_of(df):
    pid = df["patient_id"].to_numpy()
    return {p: np.where(pid == p)[0] for p in np.unique(pid)}


def path_for(e, seed, split):
    if e == "se_only" and seed == 42:
        # primary se_only lives under runs_seeds/se_only_s42
        pass
    d = "primary" if seed == 42 and e != "se_only" else "seeds"
    tag = "" if seed == 42 and e != "se_only" else f"_s{seed}"
    ev = "_ev" if e in EDL else ""
    return f"{d}/{e}{tag}_{split}{ev}.npz"


def aurc(prob_ex, ue, y):
    n = len(y)
    order = np.argsort(ue, kind="stable")
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
        fn_retained=int(((pred == 0) & pos[sel]).sum()),
        deferred_pos=int((y[defer] == 1).sum()),
        deferred_neg=int((y[defer] == 0).sum()),
    )


def dirichlet_terms(alphas):
    S = alphas.sum(axis=1, keepdims=True)
    p = alphas / S
    ale = -(p * (digamma(S + 1.0) - digamma(alphas + 1.0))).sum(axis=1)
    he = -(p * np.log(p + 1e-12)).sum(axis=1)
    return ale, he


def exam_arrays(p, u, alphas, g, pids):
    """Return per-exam classifier probs (dict of aggregations) and deferral
    scores (dict of rules)."""
    top1 = np.array([p[g[pp]].max() for pp in pids])
    top3m = np.array([np.sort(p[g[pp]])[::-1][:3].mean() for pp in pids])
    nor = np.array([1.0 - np.prod(1.0 - p[g[pp]]) for pp in pids])
    clf = {"max": top1, "top3": top3m, "noisyor": nor}
    mean_p = np.array([p[g[pp]].mean() for pp in pids])
    t = 1 - np.maximum(p, 1 - p)
    scores = {}
    scores["maxunc"] = np.array([u[g[pp]].max() for pp in pids])
    amax = np.array([np.argmax(p[g[pp]]) for pp in pids])
    scores["dec"] = np.array([u[g[pp]][a] for pp, a in zip(pids, amax)])
    scores["dec3max"] = np.array(
        [np.sort(u[g[pp]][np.argsort(p[g[pp]])[::-1][:3]]).max() for pp in pids]
    )
    scores["dec3mean"] = np.array(
        [np.sort(u[g[pp]][np.argsort(p[g[pp]])[::-1][:3]]).mean() for pp in pids]
    )
    dm = np.array([t[g[pp]][a] for pp, a in zip(pids, amax)])
    rv = np.argsort(np.argsort(scores["dec"]))
    rm = np.argsort(np.argsort(dm))
    scores["decfusion"] = (rv + rm).astype(float)
    if alphas is not None:
        ale, he = dirichlet_terms(alphas)
        scores["tothe"] = np.array([he[g[pp]].max() for pp in pids])
        scores["dectot"] = np.array([he[g[pp]][a] for pp, a in zip(pids, amax)])
        scores["decfusion"] = None  # recomputed below with tothe+decmar
        r1 = np.argsort(np.argsort(scores["dec"]))
        r2 = np.argsort(np.argsort(scores["dectot"]))
        scores["decfusion"] = (r1 + r2).astype(float)
    # features for the validation-fitted logistic rule
    feats = np.stack(
        [
            top1,
            mean_p,
            scores["maxunc"],
            np.array([u[g[pp]].mean() for pp in pids]),
            np.array([t[g[pp]].max() for pp in pids]),
            np.array([(p[g[pp]] > 0.5).mean() for pp in pids]),
            np.log1p(np.array([len(g[pp]) for pp in pids])),
            scores["dec"],
        ],
        axis=1,
    )
    return clf, scores, feats


def metrics(pe, ue, y):
    return dict(aurc=aurc(pe, ue, y), **at95(pe, ue, y))


def eval_run(e, seed, val, test, g_va, g_te, va_pids, te_pids, va_y, te_y):
    d_va = np.load(path_for(e, seed, "val"))
    d_te = np.load(path_for(e, seed, "test"))
    p_va, u_va = d_va["probs"].astype(np.float64), d_va["unc"].astype(np.float64)
    p_te, u_te = d_te["probs"].astype(np.float64), d_te["unc"].astype(np.float64)
    a_va = d_va["alphas"].astype(np.float64) if "alphas" in d_va else None
    a_te = d_te["alphas"].astype(np.float64) if "alphas" in d_te else None
    clf_va, sc_va, X_va = exam_arrays(p_va, u_va, a_va, g_va, va_pids)
    clf_te, sc_te, X_te = exam_arrays(p_te, u_te, a_te, g_te, te_pids)

    # 1) classifier aggregation selected on validation by AUC
    from sklearn.metrics import roc_auc_score

    auc_va = {k: roc_auc_score(va_y, v) for k, v in clf_va.items()}
    best_clf = max(auc_va, key=auc_va.get)
    pe_va, pe_te = clf_va[best_clf], clf_te[best_clf]

    # 2) deferral rule selected on validation by AURC (under the selected
    #    classifier's exam probabilities)
    cand = {k: v for k, v in sc_va.items() if v is not None}
    aurc_va = {k: aurc(pe_va, v, va_y) for k, v in cand.items()}
    # logistic rule fitted on validation errors of the selected classifier
    err_va = ((pe_va > 0.5).astype(int) != va_y).astype(int)
    scaler = StandardScaler().fit(X_va)
    lr = LogisticRegression(max_iter=5000, C=1.0, solver="liblinear")
    lr.fit(scaler.transform(X_va), err_va)
    log_va = lr.predict_proba(scaler.transform(X_va))[:, 1]
    log_te = lr.predict_proba(scaler.transform(X_te))[:, 1]
    aurc_va["logistic"] = aurc(pe_va, log_va, va_y)
    best_rule = min(aurc_va, key=aurc_va.get)
    ue_te = log_te if best_rule == "logistic" else sc_te[best_rule]

    return dict(
        best_clf=best_clf,
        val_clf_auc={k: float(v) for k, v in auc_va.items()},
        best_rule=best_rule,
        val_rule_aurc={k: float(v) for k, v in aurc_va.items()},
        val=metrics(
            pe_va, log_va if best_rule == "logistic" else sc_va[best_rule], va_y
        ),
        test=metrics(pe_te, ue_te, te_y),
        # prespecified reference (max aggregation + max uncertainty)
        prespec_test=metrics(clf_te["max"], sc_te["maxunc"], te_y),
        prespec_val=metrics(clf_va["max"], sc_va["maxunc"], va_y),
    )


def main():
    val = pd.read_csv("val.csv")
    test = pd.read_csv("test.csv")
    g_va, g_te = groups_of(val), groups_of(test)
    va_pids, te_pids = sorted(g_va), sorted(g_te)
    va_y = np.array([val["any"].to_numpy()[g_va[p]].max() for p in va_pids])
    te_y = np.array([test["any"].to_numpy()[g_te[p]].max() for p in te_pids])

    out = {}
    for e in EXPS:
        runs = {}
        for seed in SEEDS:
            fn = path_for(e, seed, "test")
            if not os.path.exists(fn):
                print("missing", fn)
                continue
            runs[str(seed)] = eval_run(
                e, seed, val, test, g_va, g_te, va_pids, te_pids, va_y, te_y
            )
        # aggregate across seeds
        agg = {}
        for metric in ("aurc", "acc", "sens", "spec", "fn_retained", "deferred_pos"):
            vals = [r["test"][metric] for r in runs.values()]
            pre = [r["prespec_test"][metric] for r in runs.values()]
            agg[metric] = dict(
                mean=float(np.mean(vals)),
                sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                pre_mean=float(np.mean(pre)),
                pre_sd=float(np.std(pre, ddof=1)) if len(pre) > 1 else 0.0,
            )
        rules = [r["best_rule"] for r in runs.values()]
        clfs = [r["best_clf"] for r in runs.values()]
        out[e] = dict(runs=runs, agg=agg, rules=rules, clfs=clfs)
        print(f"\n=== {e} (rules: {rules}; clf: {clfs})")
        for metric in agg:
            a = agg[metric]
            print(
                f"  {metric:13s} selected {a['mean']:.4f}±{a['sd']:.4f}   prespecified {a['pre_mean']:.4f}±{a['pre_sd']:.4f}"
            )

    with open("exam_final.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote exam_final.json")


if __name__ == "__main__":
    main()
