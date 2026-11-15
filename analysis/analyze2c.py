"""Stage 2c: exam-level deferral scores for EDL models — vacuity vs
probability-based total uncertainty (1 - max p) vs Dirichlet entropy."""

import json
import numpy as np
import pandas as pd

EXPS = ["baseline", "wavelet_only", "edl_only", "wavelet_edl"]
test = pd.read_csv("test.csv")
val = pd.read_csv("val.csv")


def groups_of(df):
    pid = df["patient_id"].to_numpy()
    return {p: np.where(pid == p)[0] for p in np.unique(pid)}


g_te, g_va = groups_of(test), groups_of(val)
te_pids, va_pids = sorted(g_te), sorted(g_va)
te_y = np.array([test["any"].to_numpy()[g_te[p]].max() for p in te_pids])
va_y = np.array([val["any"].to_numpy()[g_va[p]].max() for p in va_pids])


def aurc(prob_ex, unc_ex, y):
    n = len(y)
    order = np.argsort(unc_ex)
    accs = np.array(
        [
            ((prob_ex[order[:k]] > 0.5).astype(int) == y[order[:k]]).mean()
            for k in range(1, n + 1)
        ]
    )
    return float((1 - accs).mean())


def at95(prob_ex, unc_ex, y):
    n = len(y)
    k = int(0.95 * n)
    sel = np.argsort(unc_ex)[:k]
    pred = (prob_ex[sel] > 0.5).astype(int)
    pos = y[sel] == 1
    tp = int(((pred == 1) & pos).sum())
    fn = int(((pred == 0) & pos).sum())
    defer = np.setdiff1d(np.arange(n), sel)
    return dict(
        acc=float((pred == y[sel]).mean()),
        sens=tp / (tp + fn),
        fn=fn,
        deferred_pos=int((y[defer] == 1).sum()),
        deferred=len(defer),
    )


scores = {}
for e in EXPS:
    scores[e] = {}
    for split, g, pids, y in [
        ("val", g_va, va_pids, va_y),
        ("test", g_te, te_pids, te_y),
    ]:
        d = np.load(f"preds/{e}/{split}_predictions.npz")
        p, u = d["probs"].astype(np.float64), d["unc"].astype(np.float64)
        pe = np.array([p[g[pp]].max() for pp in pids])
        variants = {
            "vac_max": lambda sl_u: sl_u.max(),
            "vac_mean": lambda sl_u: sl_u.mean(),
            "conf_max": None,  # 1 - max prob per slice, then max
            "conf_mean": None,
            "vac_p95": lambda sl_u: np.quantile(sl_u, 0.95),
        }
        res = {}
        # vacuity variants
        for name, fn in [
            ("vac_max", lambda s: s.max()),
            ("vac_mean", lambda s: s.mean()),
            ("vac_p95", lambda s: np.quantile(s, 0.95)),
        ]:
            ue = np.array([fn(u[g[pp]]) for pp in pids])
            res[name] = dict(val_or_test=aurc(pe, ue, y))
        # probability-based total uncertainty per slice: 1 - max(p_k)
        tot = 1 - np.maximum(p, 1 - p)
        for name, fn in [
            ("conf_max", lambda s: s.max()),
            ("conf_mean", lambda s: s.mean()),
            ("conf_p95", lambda s: np.quantile(s, 0.95)),
        ]:
            ue = np.array([fn(tot[g[pp]]) for pp in pids])
            res[name] = dict(val_or_test=aurc(pe, ue, y))
        scores[e][split] = res
    print(f"\n{e}:")
    for k in scores[e]["val"]:
        print(
            f"  {k}: val {scores[e]['val'][k]['val_or_test']:.4f}  test {scores[e]['test'][k]['val_or_test']:.4f}"
        )

# detail at 95% coverage with conf-based score for all models (uniform rule)
print("\nUniform rule: exam unc = max over slices of (1 - max_k p_k); coverage 95%")
for e in EXPS:
    d = np.load(f"preds/{e}/test_predictions.npz")
    p = d["probs"].astype(np.float64)
    pe = np.array([p[g_te[pp]].max() for pp in te_pids])
    tot = 1 - np.maximum(p, 1 - p)
    ue = np.array([tot[g_te[pp]].max() for pp in te_pids])
    print(f"  {e}: AURC {aurc(pe, ue, te_y):.4f} @95: {at95(pe, ue, te_y)}")
