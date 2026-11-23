"""Frozen-protocol examination-level evaluation.

Choices frozen on the primary run's (seed 42) validation split:
  exam classifier aggregation = top3-mean of slice probabilities
  deferral score              = dec3mean (mean uncertainty of the top-3
                                probability slices; decision-coupled)
Applied unchanged to every configuration and all five seeds on test.
Reference: prespecified rule (max aggregation + max slice uncertainty).
Output: exam_frozen.json
"""

import json
import numpy as np
import pandas as pd

EXPS = ["baseline", "wavelet_only", "edl_only", "wavelet_edl", "se_only"]
SEEDS = [42, 1, 2, 3, 4]


def groups_of(df):
    pid = df["patient_id"].to_numpy()
    return {p: np.where(pid == p)[0] for p in np.unique(pid)}


def path_for(e, seed, split):
    d = "primary" if seed == 42 and e != "se_only" else "seeds"
    tag = "" if seed == 42 and e != "se_only" else f"_s{seed}"
    ev = "_ev" if e in {"edl_only", "wavelet_edl"} else ""
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


def metrics(pe, ue, y):
    return dict(aurc=aurc(pe, ue, y), **at95(pe, ue, y))


def exam_vectors(p, u, g, pids):
    top1 = np.array([p[g[pp]].max() for pp in pids])
    top3m = np.array([np.sort(p[g[pp]])[::-1][:3].mean() for pp in pids])
    # probability-based total uncertainty 1 - max_k p_k (the paper's
    # prespecified deferral score for every configuration)
    u_margin = 1.0 - np.maximum(p, 1.0 - p)
    maxunc = np.array([u_margin[g[pp]].max() for pp in pids])
    # per-exam mean of the raw npz uncertainty (entropy for softmax, Dirichlet
    # vacuity for evidential models) over the top-3 probability slices; this is
    # the validation-selected decision-coupled score
    dec3mean = np.array(
        [u[g[pp]][np.argsort(p[g[pp]])[::-1][:3]].mean() for pp in pids]
    )
    # raw npz unc maximised over slices (kept for auditability; for softmax
    # models it coincides with maxunc, for evidential models it is the
    # vacuity-max alternative, not the paper's prespecified rule)
    maxunc_npz = np.array([u[g[pp]].max() for pp in pids])
    return top1, top3m, maxunc, dec3mean, maxunc_npz


def main():
    val = pd.read_csv("val.csv")
    test = pd.read_csv("test.csv")
    g_va, g_te = groups_of(val), groups_of(test)
    va_pids, te_pids = sorted(g_va), sorted(g_te)
    va_y = np.array([val["any"].to_numpy()[g_va[p]].max() for p in va_pids])
    te_y = np.array([test["any"].to_numpy()[g_te[p]].max() for p in te_pids])

    out = {}
    for e in EXPS:
        per = {}
        for seed in SEEDS:
            f_te = path_for(e, seed, "test")
            f_va = path_for(e, seed, "val")
            try:
                d_te = np.load(f_te)
                d_va = np.load(f_va)
            except FileNotFoundError:
                print("missing", f_te)
                continue
            top1_va, top3_va, mu_va, d3m_va, munp_va = exam_vectors(
                d_va["probs"].astype(float), d_va["unc"].astype(float), g_va, va_pids
            )
            top1_te, top3_te, mu_te, d3m_te, munp_te = exam_vectors(
                d_te["probs"].astype(float), d_te["unc"].astype(float), g_te, te_pids
            )
            per[str(seed)] = dict(
                frozen=metrics(top3_te, d3m_te, te_y),
                prespec=metrics(top1_te, mu_te, te_y),
                prespec_npz_unc=metrics(top1_te, munp_te, te_y),
                frozen_val=metrics(top3_va, d3m_va, va_y),
                prespec_val=metrics(top1_va, mu_va, va_y),
            )
        agg = {}
        for metric in (
            "aurc",
            "acc",
            "sens",
            "spec",
            "fn_retained",
            "deferred_pos",
            "deferred_neg",
        ):
            fz = [r["frozen"][metric] for r in per.values()]
            ps = [r["prespec"][metric] for r in per.values()]
            agg[metric] = dict(
                frozen_mean=float(np.mean(fz)),
                frozen_sd=float(np.std(fz, ddof=1)) if len(fz) > 1 else 0.0,
                prespec_mean=float(np.mean(ps)),
                prespec_sd=float(np.std(ps, ddof=1)) if len(ps) > 1 else 0.0,
            )
        out[e] = dict(per_seed=per, agg=agg)
        print(f"=== {e}")
        for metric in agg:
            a = agg[metric]
            print(
                f"  {metric:13s} frozen {a['frozen_mean']:.4f}±{a['frozen_sd']:.4f}   prespec {a['prespec_mean']:.4f}±{a['prespec_sd']:.4f}"
            )

    with open("exam_frozen.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote exam_frozen.json")


if __name__ == "__main__":
    main()
