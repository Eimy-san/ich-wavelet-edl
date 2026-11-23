"""Decompose the Table 12 rule change into classification vs deferral components.

Four rule cells per (config, seed), all on the test split:

  A  prespec-prespec : exam classification = max slice probability;
                       deferral score      = max over slices of (1 - max p)
                       (the paper's prespecified probability-based rule)
  B  top3cls-prespec : classification = mean of top-3 slice probabilities;
                       deferral score   = as A
  C  prespec-cls-top3score : classification = as A;
                       deferral score      = mean uncertainty of the top-3
                       probability slices (decision-coupled)
  D  frozen (top3-top3) : classification and deferral score as in the frozen
                       decision-coupled rule

Metrics per cell: AURC, Acc@95 (retained accuracy at 95% coverage), retained
sensitivity, retained false negatives, deferred positives/negatives (of the
13 deferred), and examination-level sensitivity/specificity of the aggregated
classifier. Writes results/rule_decomposition.json.
"""

import json
import os

import numpy as np
import pandas as pd

A = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 1, 2, 3, 4]
EXPS = {
    "baseline": "baseline",
    "wavelet_only": "wavelet",
    "edl_only": "edl",
    "wavelet_edl": "full",
    "se_only": "se_only",
}


def groups_of(df):
    g = {}
    for pid, grp in df.groupby("patient_id", sort=False):
        g[pid] = grp.index.to_numpy()
    return g


def path_for(exp_key, seed, split):
    if exp_key == 'se_only':
        return os.path.join(A, 'seeds', f'se_only_s{seed}_{split}.npz')
    if seed == 42:
        d = os.path.join(A, "primary")
        if exp_key == "edl_only":
            f = os.path.join(d, f"{exp_key}_{split}_ev.npz")
            if os.path.exists(f):
                return f
            return os.path.join(d, f"{exp_key}_{split}.npz")
        if exp_key == "wavelet_edl":
            f = os.path.join(d, f"wavelet_edl_{split}_ev.npz")
            if os.path.exists(f):
                return f
            return os.path.join(d, f"wavelet_edl_{split}.npz")
        return os.path.join(d, f"{exp_key}_{split}.npz")
    d = os.path.join(A, "seeds")
    if exp_key in ("edl_only", "wavelet_edl"):
        f = os.path.join(d, f"{exp_key}_s{seed}_{split}_ev.npz")
        if os.path.exists(f):
            return f
    return os.path.join(d, f"{exp_key}_s{seed}_{split}.npz")


def aurc(prob_ex, ue, y):
    order = np.argsort(ue, kind="stable")
    n = len(y)
    risks = []
    for k in range(1, n + 1):
        pred = (prob_ex[order[:k]] > 0.5).astype(int)
        risks.append(1.0 - (pred == y[order[:k]]).mean())
    return float(np.mean(risks))


def at95(prob_ex, ue, y):
    order = np.argsort(ue, kind="stable")
    n = len(y)
    keep = int(round(0.95 * n))
    sel = order[:keep]
    pred = (prob_ex[sel] > 0.5).astype(int)
    acc = float((pred == y[sel]).mean())
    pos = y == 1
    retained_sens = float(pred[pos[sel]].mean()) if pos[sel].any() else float("nan")
    fn_ret = int(((pred == 0) & (y[sel] == 1)).sum())
    deferred = order[keep:]
    dpos = int((y[deferred] == 1).sum())
    dneg = int((y[deferred] == 0).sum())
    return acc, retained_sens, fn_ret, dpos, dneg


def metrics(pe, ue, y):
    acc, rs, fnr, dpos, dneg = at95(pe, ue, y)
    return {
        "aurc": round(aurc(pe, ue, y), 4),
        "acc95": round(acc, 4),
        "ret_sens": round(rs, 4),
        "fn_retained": fnr,
        "deferred_pos": dpos,
        "deferred_neg": dneg,
        "exam_sens": round(float(((pe > 0.5) & (y == 1)).sum() / (y == 1).sum()), 4),
        "exam_spec": round(float(((pe <= 0.5) & (y == 0)).sum() / (y == 0).sum()), 4),
    }


def main():
    val = pd.read_csv(os.path.join(A, "val.csv"))
    test = pd.read_csv(os.path.join(A, "test.csv"))
    g_te = groups_of(test)
    te_pids = sorted(g_te)
    te_y = np.array([test["any"].to_numpy()[g_te[p]].max() for p in te_pids])

    out = {}
    for exp_key, short in EXPS.items():
        per_seed = {}
        for seed in SEEDS:
            f_te = path_for(exp_key, seed, "test")
            if not os.path.exists(f_te):
                print("missing", f_te)
                continue
            d_te = np.load(f_te)
            p = d_te["probs"].astype(float)
            u_ev = d_te["unc"].astype(float)
            # probability-based total uncertainty 1 - max p (paper prespecified)
            u_margin = 1.0 - np.maximum(p, 1.0 - p)
            top1 = np.array([p[g_te[pp]].max() for pp in te_pids])
            top3m = np.array([np.sort(p[g_te[pp]])[::-1][:3].mean() for pp in te_pids])
            maxunc = np.array([u_margin[g_te[pp]].max() for pp in te_pids])
            top3score = np.array(
                [
                    u_margin[g_te[pp]][np.argsort(p[g_te[pp]])[::-1][:3]].mean()
                    for pp in te_pids
                ]
            )
            top3uncvac = np.array(
                [
                    u_ev[g_te[pp]][np.argsort(p[g_te[pp]])[::-1][:3]].mean()
                    for pp in te_pids
                ]
            )
            per_seed[str(seed)] = {
                "A_prespec": metrics(top1, maxunc, te_y),
                "B_top3cls": metrics(top3m, maxunc, te_y),
                "C_top3score": metrics(top1, top3score, te_y),
                "D_frozen": metrics(top3m, top3uncvac, te_y),
            }
        if not per_seed:
            continue
        agg = {}
        for cell in ["A_prespec", "B_top3cls", "C_top3score", "D_frozen"]:
            for m in [
                "aurc",
                "acc95",
                "ret_sens",
                "fn_retained",
                "deferred_pos",
                "exam_sens",
                "exam_spec",
            ]:
                vals = [per_seed[s][cell][m] for s in per_seed]
                agg[f"{cell}_{m}"] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "sd": round(float(np.std(vals, ddof=1)), 4),
                }
        out[short] = {"per_seed": per_seed, "agg": agg}
    dest = os.path.join(A, "rule_decomposition.json")
    json.dump(out, open(dest, "w"), indent=1)
    print("saved", dest)
    for short, v in out.items():
        a = v["agg"]
        print(
            short,
            "A aurc %.4f acc95 %.4f rsens %.4f fnr %.1f dpos %.1f esens %.4f espec %.4f"
            % (
                a["A_prespec_aurc"]["mean"],
                a["A_prespec_acc95"]["mean"],
                a["A_prespec_ret_sens"]["mean"],
                a["A_prespec_fn_retained"]["mean"],
                a["A_prespec_deferred_pos"]["mean"],
                a["A_prespec_exam_sens"]["mean"],
                a["A_prespec_exam_spec"]["mean"],
            ),
        )
        print(
            "      B aurc %.4f acc95 %.4f rsens %.4f fnr %.1f dpos %.1f esens %.4f espec %.4f"
            % (
                a["B_top3cls_aurc"]["mean"],
                a["B_top3cls_acc95"]["mean"],
                a["B_top3cls_ret_sens"]["mean"],
                a["B_top3cls_fn_retained"]["mean"],
                a["B_top3cls_deferred_pos"]["mean"],
                a["B_top3cls_exam_sens"]["mean"],
                a["B_top3cls_exam_spec"]["mean"],
            )
        )
        print(
            "      C aurc %.4f acc95 %.4f rsens %.4f fnr %.1f dpos %.1f"
            % (
                a["C_top3score_aurc"]["mean"],
                a["C_top3score_acc95"]["mean"],
                a["C_top3score_ret_sens"]["mean"],
                a["C_top3score_fn_retained"]["mean"],
                a["C_top3score_deferred_pos"]["mean"],
            )
        )
        print(
            "      D aurc %.4f acc95 %.4f rsens %.4f fnr %.1f dpos %.1f esens %.4f espec %.4f"
            % (
                a["D_frozen_aurc"]["mean"],
                a["D_frozen_acc95"]["mean"],
                a["D_frozen_ret_sens"]["mean"],
                a["D_frozen_fn_retained"]["mean"],
                a["D_frozen_deferred_pos"]["mean"],
                a["D_frozen_exam_sens"]["mean"],
                a["D_frozen_exam_spec"]["mean"],
            )
        )


if __name__ == "__main__":
    main()
