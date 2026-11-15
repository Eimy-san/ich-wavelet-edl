"""Evaluation: risk-coverage curves, ROC, calibration, and figure generation."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.isotonic import IsotonicRegression


def risk_coverage_curve(probs, labels, uncertainty):
    """Compute accuracy vs. coverage as abstention threshold varies."""
    n = len(labels)
    preds = (probs > 0.5).astype(int)
    correct = (preds == labels).astype(int)
    sorted_idx = np.argsort(uncertainty)
    coverages = []
    accuracies = []
    risks = []
    for k in range(1, n + 1):
        sel = sorted_idx[:k]
        cov = k / n
        acc = correct[sel].mean()
        risk = 1 - acc
        coverages.append(cov)
        accuracies.append(acc)
        risks.append(risk)
    return np.array(coverages), np.array(accuracies), np.array(risks)


def aurc(accuracies):
    """Area Under Risk-Coverage curve (lower is better)."""
    return np.mean(1 - accuracies)


def plot_risk_coverage(results, save_path):
    """Plot risk-coverage curves for multiple methods."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for name, data in results.items():
        cov, acc, risk = risk_coverage_curve(
            data['probs'], data['labels'], data['unc'])
        ax.plot(cov, risk, label=f'{name} (AURC={aurc(acc):.4f})', linewidth=1.5)
    ax.set_xlabel('Coverage', fontsize=12)
    ax.set_ylabel('Risk (1 - Accuracy)', fontsize=12)
    ax.set_title('Risk-Coverage Curves', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc(results, save_path):
    """Plot ROC curves for multiple methods."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for name, data in results.items():
        fpr, tpr, _ = roc_curve(data['labels'], data['probs'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.4f})', linewidth=1.5)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_coverage_accuracy(results, save_path):
    """Plot accuracy vs. coverage curves."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for name, data in results.items():
        cov, acc, _ = risk_coverage_curve(
            data['probs'], data['labels'], data['unc'])
        ax.plot(cov, acc, label=name, linewidth=1.5)
    ax.set_xlabel('Coverage', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Selective Prediction: Accuracy vs. Coverage', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def compute_ece(probs, labels, unc, n_bins=10):
    """Expected Calibration Error using uncertainty-based binning."""
    preds = (probs > 0.5).astype(int)
    correct = (preds == labels).astype(float)
    sorted_idx = np.argsort(unc)
    bin_size = len(sorted_idx) // n_bins
    ece = 0
    for i in range(n_bins):
        if i == n_bins - 1:
            sel = sorted_idx[i * bin_size:]
        else:
            sel = sorted_idx[i * bin_size:(i + 1) * bin_size]
        if len(sel) == 0:
            continue
        conf = 1 - unc[sel].mean()
        acc = correct[sel].mean()
        ece += abs(conf - acc) * len(sel) / len(labels)
    return ece


def selective_metrics(probs, labels, unc, target_coverage=0.95):
    """Compute accuracy at a target coverage level."""
    n = len(labels)
    preds = (probs > 0.5).astype(int)
    correct = (preds == labels).astype(int)
    sorted_idx = np.argsort(unc)
    k = int(target_coverage * n)
    sel = sorted_idx[:k]
    return correct[sel].mean(), 1 - correct[sel].mean()


def main():
    runs_dir = '/root/autodl-tmp/ich_project/runs'
    fig_dir = '/root/autodl-tmp/ich_project/figures'
    os.makedirs(fig_dir, exist_ok=True)

    exp_names = ['baseline', 'wavelet_only', 'edl_only', 'wavelet_edl']
    all_results = {}
    summary = []

    for name in exp_names:
        pred_path = os.path.join(runs_dir, name, 'test_predictions.npz')
        if not os.path.exists(pred_path):
            print(f'Skip {name}: no predictions found')
            continue
        data = np.load(pred_path)
        probs, labels, unc = data['probs'], data['labels'], data['unc']
        all_results[name] = {'probs': probs, 'labels': labels, 'unc': unc}

        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        cov, acc, risk = risk_coverage_curve(probs, labels, unc)
        aurc_val = aurc(acc)
        ece = compute_ece(probs, labels, unc)
        acc95, risk95 = selective_metrics(probs, labels, unc, 0.95)

        summary.append({
            'exp': name, 'auc': roc_auc, 'aurc': aurc_val, 'ece': ece,
            'acc_at_95cov': acc95, 'risk_at_95cov': risk95,
        })
        print(f'{name}: AUC={roc_auc:.4f} AURC={aurc_val:.4f} ECE={ece:.4f} '
              f'Acc@95%={acc95:.4f}')

    if all_results:
        plot_risk_coverage(all_results, os.path.join(fig_dir, 'risk_coverage.png'))
        plot_roc(all_results, os.path.join(fig_dir, 'roc_curves.png'))
        plot_coverage_accuracy(all_results, os.path.join(fig_dir, 'coverage_accuracy.png'))

    with open(os.path.join(fig_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nFigures saved to {fig_dir}')


if __name__ == '__main__':
    main()
