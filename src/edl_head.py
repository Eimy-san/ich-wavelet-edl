"""Evidential Deep Learning (EDL) head and loss functions.

Based on Sensoy et al. (2018): Evidential Deep Learning to Quantify
Classification Uncertainty. NeurIPS 2018.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def evidence_to_alpha(evidence):
    """Convert non-negative evidence to Dirichlet parameters."""
    alpha = evidence + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    return alpha, S


def edl_mse_loss(evidence, target, K, annealing_coeff):
    """Expected MSE loss + KL divergence regularization."""
    alpha, S = evidence_to_alpha(evidence)
    prob = alpha / S
    err = (target - prob) ** 2
    var = prob * (1 - prob) / (S + 1)
    err = (err + var).sum(dim=1)

    # KL: remove evidence for the correct class
    alpha_tilde = target + (1 - target) * alpha
    kl = kl_dirichlet_uniform(alpha_tilde, K)

    return (err + annealing_coeff * kl.squeeze()).mean()


def edl_log_loss(evidence, target, K, annealing_coeff):
    """Expected log loss (digamma-based) + KL."""
    alpha, S = evidence_to_alpha(evidence)
    err = (target * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1)

    alpha_tilde = target + (1 - target) * alpha
    kl = kl_dirichlet_uniform(alpha_tilde, K)

    return (err + annealing_coeff * kl.squeeze()).mean()


def edl_ce_loss(evidence, target, K, annealing_coeff):
    """Expected cross-entropy loss + KL."""
    alpha, S = evidence_to_alpha(evidence)
    err = (target * (torch.log(S) - torch.log(alpha))).sum(dim=1)

    alpha_tilde = target + (1 - target) * alpha
    kl = kl_dirichlet_uniform(alpha_tilde, K)

    return (err + annealing_coeff * kl.squeeze()).mean()


def kl_dirichlet_uniform(alpha, K):
    """KL[Dir(alpha) || Dir(1)] — penalize misplaced evidence."""
    S = alpha.sum(dim=1, keepdim=True)
    lgamma_S = torch.lgamma(S)
    sum_lgamma_alpha = torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    term1 = lgamma_S - sum_lgamma_alpha - torch.lgamma(torch.tensor(float(K), device=alpha.device))
    term2 = torch.sum((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S)), dim=1, keepdim=True)
    return term1 + term2


def get_edl_loss(loss_type='mse'):
    if loss_type == 'mse':
        return edl_mse_loss
    elif loss_type == 'log':
        return edl_log_loss
    elif loss_type == 'ce':
        return edl_ce_loss
    raise ValueError(f'Unknown EDL loss type: {loss_type}')


def edl_predict(evidence):
    """Return predicted probabilities and uncertainty (vacuity)."""
    alpha, S = evidence_to_alpha(evidence)
    K = evidence.shape[1]
    prob = alpha / S
    uncertainty = K / S.squeeze(1)
    return prob, uncertainty
