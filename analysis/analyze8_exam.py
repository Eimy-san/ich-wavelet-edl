"""Examination-level deferral scores: epistemic (vacuity) vs total predictive
uncertainty from the Dirichlet posterior.

For EDL runs, per-slice Dirichlet alphas are dumped by dump_evidence.py. For a
Dirichlet posterior with parameters alpha over K classes:

  E[p_k]      = alpha_k / S,  S = sum_k alpha_k
  aleatoric   = -sum_k (alpha_k/S) * (psi(S+1) - psi(alpha_k+1))   (expected entropy)
  H(E[p])     = -sum_k (alpha_k/S) * log(alpha_k/S)
  MI (epist.) = H(E[p]) - aleatoric
  total       = aleatoric + MI = H(E[p]) - E[H[p]]

Softmax models: their saved `unc` is the predictive entropy, which already is
the total predictive uncertainty of a point-estimate model.

Exam-level scores tested (select on VALIDATION, report test):
  vac_max    max over slices of vacuity u = K/S            (original prespecified)
  tot_max    max over slices of total uncertainty
  ale_max    max over slices of aleatoric component
  mi_max     max over slices of MI component
  mar_max    max over slices of 1 - max(p, 1-p)            (probability margin)
  fus_max    mean of ranks of vac_max and mar_max          (two-signal fusion)
  vacm_max   max over slices of (vacuity * margin)
plus mean-aggregation variants of vac and tot for completeness.

Outputs exam_rule_ablation.json with val/test AURC and @95 metrics per score
per configuration (primary runs), plus the multi-seed test metrics of the
selected rule for the EDL configurations.
"""

import json
import numpy as np
import pandas as pd
from scipy.special import gammaln, digamma

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


def at95(prob_ex, unc_ex, y):
    n = len(y)
    k = int(0.95 * n)
    sel = np.argsort(unc_ex, kind="stable")[:k]
    pred = (prob_ex[sel] > 0.5).astype(int)
    correct = pred == y[sel]
    defer = np.setdiff1d(np.arange(n), sel)
    pos = y == 1
    return dict(
        acc=float((pred == y[sel]).mean()),
        sens=float(((pred == 1) & pos[sel]).sum() / max(pos[sel].sum(), 1)),
        spec=float(((pred == 0) & ~pos[sel]).sum() / max((~pos[sel]).sum(), 1)),
        deferred_pos=int((y[defer] == 1).sum()),
        deferred_neg=int((y[defer] == 0).sum()),
        aurc=None,
    )


def dirichlet_terms(alphas):
    """alphas: (N, K). Returns (aleatoric, MI, total) per slice."""
    S = alphas.sum(axis=1, keepdims=True)
    p = alphas / S
    K = alphas.shape[1]
    # E[H[p]] = -sum p_k (psi(S+1) - psi(alpha_k+1))
    ale = -(p * (digamma(S + 1.0) - digamma(alphas + 1.0))).sum(axis=1)
    # H(E[p])
    he = -(p * np.log(p + 1e-12)).sum(axis=1)
    mi = he - ale
    return ale, mi, he


def exam_scores(p, u, alphas, g, pids):
    """Return dict score_name -> exam-level uncertainty vector."""
    out = {}
    pe = np.array([p[g[pp]].max() for pp in pids])
    out["_pe"] = pe
    if alphas is not None:
        ale, mi, he = dirichlet_terms(alphas)
        for nm, comp in [("vac", u), ("ale", ale), ("mi", mi), ("tot", he)]:
            out[nm + "_max"] = np.array([comp[g[pp]].max() for pp in pids])
            out[nm + "_mean"] = np.array([comp[g[pp]].mean() for pp in pids])
    else:
        # softmax: unc (entropy) is total uncertainty; also expose vac alias
        out["tot_max"] = np.array([u[g[pp]].max() for pp in pids])
        out["tot_mean"] = np.array([u[g[pp]].mean() for pp in pids])
        out["dec_tot"] = np.array([u[g[pp]][np.argmax(p[g[pp]])] for pp in pids])
    t = 1 - np.maximum(p, 1 - p)
    out["mar_max"] = np.array([t[g[pp]].max() for pp in pids])
    out["mar_mean"] = np.array([t[g[pp]].mean() for pp in pids])
    if alphas is not None:
        rv = np.argsort(np.argsort(out["vac_max"]))
        rm = np.argsort(np.argsort(out["mar_max"]))
        out["fus_max"] = (rv + rm).astype(float)
        out["vacm_max"] = np.array([(u[g[pp]] * t[g[pp]]).max() for pp in pids])
        # decision-coupled scores: uncertainty attached to the slice that
        # drives the exam decision (argmax prob)
        out["dec_vac"] = np.array([u[g[pp]][np.argmax(p[g[pp]])] for pp in pids])
        out["dec_mar"] = np.array([t[g[pp]][np.argmax(p[g[pp]])] for pp in pids])
        out["dec_tot"] = np.array([he[g[pp]][np.argmax(p[g[pp]])] for pp in pids])
        out["dec_mi"] = np.array([mi[g[pp]][np.argmax(p[g[pp]])] for pp in pids])
        # noisy-OR exam probability and its distance from the decision boundary
        nor = np.array([1.0 - np.prod(1.0 - p[g[pp]]) for pp in pids])
        out["_pe_nor"] = nor
        out["dec2_vac"] = np.array(
            [u[g[pp]][np.argmax(np.abs(p[g[pp]] - 0.5))] for pp in pids]
        )
    return out


def metrics_for(prob_ex, ue, y):
    r = at95(prob_ex, ue, y)
    r["aurc"] = aurc(prob_ex, ue, y)
    return r


def main():
    val = pd.read_csv("val.csv")
    test = pd.read_csv("test.csv")
    g_va, g_te = groups_of(val), groups_of(test)
    va_pids, te_pids = sorted(g_va), sorted(g_te)
    va_y = np.array([val["any"].to_numpy()[g_va[p]].max() for p in va_pids])
    te_y = np.array([test["any"].to_numpy()[g_te[p]].max() for p in te_pids])

    results = {}
    for e in EXPS:
        res = {}
        for split, g, pids, y in [
            ("val", g_va, va_pids, va_y),
            ("test", g_te, te_pids, te_y),
        ]:
            if e in EDL:
                fn = f"primary/{e}_{split}_ev.npz"
            else:
                fn = f"primary/{e}_{split}.npz"
            d = np.load(fn)
            p, u = d["probs"].astype(np.float64), d["unc"].astype(np.float64)
            alphas = d["alphas"].astype(np.float64) if "alphas" in d else None
            sc = exam_scores(p, u, alphas, g, pids)
            pe = sc.pop("_pe")
            m = {}
            for name, ue in sc.items():
                m[name] = metrics_for(pe, ue, y)
            res[split] = m
        results[e] = res

    with open("exam_rule_ablation.json", "w") as f:
        json.dump(results, f, indent=1)

    # summary print
    keys = [
        "vac_max",
        "tot_max",
        "ale_max",
        "mi_max",
        "mar_max",
        "fus_max",
        "vacm_max",
        "vac_mean",
        "tot_mean",
        "mar_mean",
        "dec_vac", "dec_mar", "dec_tot", "dec_mi", "dec2_vac",
    ]
    for e in EXPS:
        print(f"\n{e}:")
        for k in keys:
            if k in results[e]["val"]:
                v = results[e]["val"][k]["aurc"]
                t = results[e]["test"][k]["aurc"]
                v95 = results[e]["val"][k]["acc"]
                t95 = results[e]["test"][k]["acc"]
                print(
                    f"  {k:9s} AURC val {v:.4f} test {t:.4f} | acc95 val {v95:.4f} test {t95:.4f}"
                )


if __name__ == "__main__":
    main()
