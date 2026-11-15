"""Post-hoc analyses for the revision: patient-level evaluation, clustered bootstrap CIs,
calibration metrics, prior-probability correction, coverage sweep, FN analysis,
and subset representativeness."""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

RNG = np.random.default_rng(2026)
EXPS = ["baseline", "wavelet_only", "edl_only", "wavelet_edl"]
PRETTY = {
    "baseline": "Baseline (ResNet-50)",
    "wavelet_only": "+Wavelet",
    "edl_only": "+EDL",
    "wavelet_edl": "+Wavelet+EDL (full)",
}

test = pd.read_csv("test.csv")
out = {}

# ---------- sanity: npz order matches test.csv ----------
data0 = np.load("preds/baseline/test_predictions.npz")
assert len(test) == len(data0["labels"]) == 9560, (len(test), len(data0["labels"]))
assert (data0["labels"] == test["any"].values).all(), "label order mismatch"
print(
    "order check OK; test ICH+ slices:",
    test["any"].sum(),
    "/",
    len(test),
    f"({test['any'].mean() * 100:.1f}%)",
    "| patients:",
    test["patient_id"].nunique(),
)

# patient -> slice indices
pid_arr = test["patient_id"].values
groups = {p: np.where(pid_arr == p)[0] for p in np.unique(pid_arr)}
pid_list = sorted(groups)
npos_pid = np.array([test["any"].values[groups[p]].max() for p in pid_list])
print(
    "exams:",
    len(pid_list),
    "| ICH+ exams:",
    npos_pid.sum(),
    f"({npos_pid.mean() * 100:.1f}%)",
)


def slice_metrics(probs, labels, thr=0.5):
    pred = (probs > thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    return dict(
        auc=float(roc_auc_score(labels, probs)),
        f1=float(f1_score(labels, pred)),
        acc=float(accuracy_score(labels, pred)),
        sens=tp / (tp + fn),
        spec=tn / (tn + fp),
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        thr=thr,
    )


def exam_metrics(prob_sl, unc_sl, thr=0.5):
    """Patient-level: presence = any slice positive; uncertainty = max slice uncertainty."""
    prob_ex = np.array([prob_sl[groups[p]].max() for p in pid_list])
    unc_ex = np.array([unc_sl[groups[p]].max() for p in pid_list])
    pred_ex = (prob_ex > thr).astype(int)
    tp = int(((pred_ex == 1) & (npos_pid == 1)).sum())
    fn = int(((pred_ex == 0) & (npos_pid == 1)).sum())
    fp = int(((pred_ex == 1) & (npos_pid == 0)).sum())
    tn = int(((pred_ex == 0) & (npos_pid == 0)).sum())
    return dict(
        auc=float(roc_auc_score(npos_pid, prob_ex)),
        f1=float(f1_score(npos_pid, pred_ex)),
        acc=float(accuracy_score(npos_pid, pred_ex)),
        sens=tp / (tp + fn),
        spec=tn / (tn + fp),
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        prob_ex=prob_ex,
        unc_ex=unc_ex,
    )


def cluster_bootstrap_ci(stat_fn, B=2000):
    """Bootstrap resampling patients (clusters). stat_fn(idx)->metric dict."""
    n = len(pid_list)
    boots = {k: [] for k in stat_fn(np.arange(n))}
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        for k, v in stat_fn(idx).items():
            boots[k].append(v)
    return {
        k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        for k, v in boots.items()
    }


def standard_ece(conf, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += abs(conf[m].mean() - correct[m].mean()) * m.mean()
    return float(ece)


def brier_nll_ece(probs, labels):
    conf = np.maximum(probs, 1 - probs)
    correct = (probs > 0.5).astype(float) == labels
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    nll = float(-(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean())
    return dict(
        brier=float(((probs - labels) ** 2).mean()),
        nll=nll,
        ece=standard_ece(conf, correct.astype(float)),
    )


def prior_correct(p, pi_train=0.5, pi_test=None):
    """Label-shift (prior) correction on positive-class probability."""
    odds = (p / (1 - p)) * (pi_test / pi_train) * ((1 - pi_train) / (1 - pi_test))
    return odds / (1 + odds)


res = {}
for e in EXPS:
    d = np.load(f"preds/{e}/test_predictions.npz")
    probs, labels, unc = (
        d["probs"].astype(np.float64),
        d["labels"],
        d["unc"].astype(np.float64),
    )
    sm = slice_metrics(probs, labels)
    em = exam_metrics(probs, unc)
    cal = brier_nll_ece(probs, labels)

    # prior-corrected calibration (balanced sampler pi_train=0.5 -> test 0.1458)
    pi_test = labels.mean()
    pc = prior_correct(probs, 0.5, pi_test)
    cal_pc = brier_nll_ece(pc, labels)

    # patient-clustered bootstrap CIs for slice metrics and exam metrics
    def sl_stat(idx):
        sel = np.concatenate([groups[pid_list[i]] for i in idx])
        m = slice_metrics(probs[sel], labels[sel])
        return {f"slice_{k}": m[k] for k in ("auc", "f1", "acc", "sens", "spec")}

    def ex_stat(idx):
        pids = [pid_list[i] for i in idx]
        pe = np.array([probs[groups[p]].max() for p in pids])
        ue = np.array([unc[groups[p]].max() for p in pids])
        ye = (
            npos_pid[idx]
            if False
            else np.array([test["any"].values[groups[p]].max() for p in pids])
        )
        pred = (pe > 0.5).astype(int)
        tp = int(((pred == 1) & (ye == 1)).sum())
        fn = int(((pred == 0) & (ye == 1)).sum())
        fp = int(((pred == 1) & (ye == 0)).sum())
        tn = int(((pred == 0) & (ye == 0)).sum())
        return {
            "exam_auc": roc_auc_score(ye, pe),
            "exam_f1": f1_score(ye, pred),
            "exam_acc": accuracy_score(ye, pred),
            "exam_sens": tp / (tp + fn),
            "exam_spec": tn / (tn + fp),
        }

    ci_sl = cluster_bootstrap_ci(sl_stat, B=2000)
    ci_ex = cluster_bootstrap_ci(ex_stat, B=2000)

    # exam-level selective prediction sweep
    sweep = {}
    order = np.argsort(em["unc_ex"])
    n = len(pid_list)
    for cov in (0.80, 0.85, 0.90, 0.95):
        k = int(cov * n)
        sel = order[:k]
        ret_labels, ret_pred = npos_pid[sel], (em["prob_ex"][sel] > 0.5).astype(int)
        tp = int(((ret_pred == 1) & (ret_labels == 1)).sum())
        fn_ = int(((ret_pred == 0) & (ret_labels == 1)).sum())
        fp = int(((ret_pred == 1) & (ret_labels == 0)).sum())
        tn = int(((ret_pred == 0) & (ret_labels == 0)).sum())
        defer = np.setdiff1d(np.arange(n), sel)
        deferred_pos = int((npos_pid[defer] == 1).sum())
        sweep[f"{int(cov * 100)}"] = dict(
            acc=float((ret_pred == ret_labels).mean()),
            sens=tp / (tp + fn_),
            spec=tn / (tn + fp),
            fn_retained=fn_,
            fn_rate_retained=fn_ / max(1, int((ret_labels == 1).sum())),
            pos_deferred=deferred_pos,
            pos_capture=deferred_pos / int(npos_pid.sum()),
        )

    res[e] = dict(
        slice=sm,
        exam={k: v for k, v in em.items() if k not in ("prob_ex", "unc_ex")},
        cal=cal,
        cal_prior_corrected=cal_pc,
        ci_slice=ci_sl,
        ci_exam=ci_ex,
        sweep=sweep,
    )
    print(f"\n=== {PRETTY[e]}")
    print(" slice:", {k: round(v, 4) for k, v in sm.items() if not isinstance(v, int)})
    print(" exam :", {k: round(v, 4) for k, v in res[e]["exam"].items()})
    print(
        " cal  :",
        {k: round(v, 4) for k, v in cal.items()},
        "| prior-corrected:",
        {k: round(v, 4) for k, v in cal_pc.items()},
    )
    print(
        " CI95 AUC:",
        [round(x, 4) for x in ci_sl["slice_auc"]],
        "sens:",
        [round(x, 4) for x in ci_sl["slice_sens"]],
        "spec:",
        [round(x, 4) for x in ci_sl["slice_spec"]],
    )
    print(
        " CI95 exam AUC:",
        [round(x, 4) for x in ci_ex["exam_auc"]],
        "exam sens:",
        [round(x, 4) for x in ci_ex["exam_sens"]],
    )
    for cov, s in sweep.items():
        print(
            f"  cov {cov}%: acc={s['acc']:.4f} sens={s['sens']:.4f} spec={s['spec']:.4f} "
            f"FN_retained={s['fn_retained']} pos_deferred={s['pos_deferred']}"
        )

# ---------- false-negative analysis by subtype (slice + exam level) ----------
sub = {}
subcols = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]
fn_tab = {}
for e in EXPS:
    d = np.load(f"preds/{e}/test_predictions.npz")
    probs, unc = d["probs"].astype(np.float64), d["unc"].astype(np.float64)
    pred = (probs > 0.5).astype(int)
    t = {}
    for c in subcols:
        m = test[c].values == 1
        pos_slices = int(m.sum())
        fn_slices = int((m & (pred == 0)).sum())
        # exam level: patient has this subtype; exam FN = exam predicted negative
        pids_sub = np.unique(pid_arr[m])
        ex_pos = np.array([(test[c].values[groups[p]].max()) for p in pids_sub])
        ex_pred = np.array([int(probs[groups[p]].max() > 0.5) for p in pids_sub])
        ex_fn = int(((ex_pos == 1) & (ex_pred == 0)).sum())
        t[c] = dict(
            slices=pos_slices,
            fn_slices=fn_slices,
            fn_rate=fn_slices / pos_slices,
            patients=len(pids_sub),
            fn_patients=ex_fn,
            fn_patient_rate=ex_fn / len(pids_sub),
            mean_unc_missed=float(unc[m & (pred == 0)].mean()) if fn_slices else None,
            mean_unc_detected=float(unc[m & (pred == 1)].mean())
            if (m & (pred == 1)).sum()
            else None,
        )
    fn_tab[e] = t
    print(f"\nFN by subtype [{PRETTY[e]}]:")
    for c in subcols:
        print(
            f"  {c}: slices {t[c]['slices']} FN {t[c]['fn_slices']} ({t[c]['fn_rate'] * 100:.1f}%) | "
            f"exams {t[c]['patients']} FN {t[c]['fn_patients']} ({t[c]['fn_patient_rate'] * 100:.1f}%)"
            + (
                f" | mean unc missed {t[c]['mean_unc_missed']:.3f}"
                if t[c]["mean_unc_missed"] is not None
                else ""
            )
        )

# ---------- subset representativeness vs full RSNA cohort ----------
full = pd.read_csv("rsna_slice_labels.csv")
print("\nfull cohort cols:", list(full.columns), "rows:", len(full))
pid_col = "patient_ID" if "patient_ID" in full.columns else "patient_id"
sub_all = pd.concat([pd.read_csv(f) for f in ["train.csv", "val.csv", "test.csv"]])
sub_pids = set(sub_all["patient_id"])
full_pids = set(full[pid_col].unique())
overlap = sub_pids & full_pids
print(
    "subset patients in full labels:",
    len(sub_pids),
    "| full patients:",
    len(full_pids),
    "| overlap:",
    len(overlap),
)
rep = {
    "full_slices": int(len(full)),
    "full_patients": int(full[pid_col].nunique()),
    "full_prevalence_any": float(full["any"].mean()),
    "subset_prevalence_any": float(sub_all["any"].mean()),
}
for c in subcols:
    rep[f"full_{c}"] = float(full[c].mean())
    rep[f"subset_{c}"] = float(sub_all[c].mean())
print(json.dumps(rep, indent=2))

out["models"] = res
out["fn_by_subtype"] = fn_tab
out["representativeness"] = rep
out["n_test_slices"] = int(len(test))
out["n_test_exams"] = int(len(pid_list))
out["test_prevalence"] = float(test["any"].mean())
with open("analysis_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nsaved analysis_results.json")
