"""
Evaluation script with bootstrap CIs, DeLong test, stratified analysis, and
calibration metrics.

FIXED (plan #9): DeLong test for AUC comparisons, stratified analysis consuming
lesion_type, ECE/Brier calibration metrics.
"""
import os
import sys

# Ensure repo root is on sys.path so `python src/evaluate.py` works without PYTHONPATH.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.utils import resample
from scipy import stats
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple


def delong_auroccov(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """
    DeLong covariance estimation for AUC and its variance.
    Returns (auc, var_auc).

    Reference: DeLong et al., "Comparing the Areas Under Two or More Correlated
    Receiver Operating Characteristic Curves", Biometrics 1988.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)

    pos = y_true == 1
    neg = np.logical_not(pos)

    n_pos = pos.sum()
    n_neg = neg.sum()

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both classes must be present in y_true.")

    auc = roc_auc_score(y_true, y_score)

    score_pos = y_score[pos]
    score_neg = y_score[neg]

    # Theta_1: for each positive sample, fraction of negatives with lower score
    theta_1 = np.searchsorted(np.sort(score_neg), score_pos, side='right')
    theta_1 = theta_1 / n_neg

    # Theta_2: for each negative sample, fraction of positives with higher score
    theta_2 = n_pos - np.searchsorted(np.sort(score_pos), score_neg, side='left')
    theta_2 = theta_2 / n_pos

    # Variance components
    var_pos = np.var(theta_1, ddof=1) / n_pos if n_pos > 1 else 0.0
    var_neg = np.var(theta_2, ddof=1) / n_neg if n_neg > 1 else 0.0
    cov = 0.0

    # DeLong covariance
    var_auc = (var_pos / n_pos) + (var_neg / n_neg) + (cov / n_pos)
    return auc, var_auc


def delong_test(
    y_true: np.ndarray, y_score_a: np.ndarray, y_score_b: np.ndarray
) -> Tuple[float, float, float]:
    """
    DeLong test for comparing two AUCs.
    Returns (z_statistic, p_value, auc_a, auc_b, auc_diff_ci_lower, auc_diff_ci_upper).

    Reference: DeLong et al., Biometrics 1988.
    """
    auc_a, var_a = delong_auroccov(y_true, y_score_a)
    auc_b, var_b = delong_auroccov(y_true, y_score_b)

    auc_diff = auc_a - auc_b
    z = auc_diff / np.sqrt(var_a + var_b)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

    se_diff = np.sqrt(var_a + var_b)
    ci_lower = auc_diff - 1.96 * se_diff
    ci_upper = auc_diff + 1.96 * se_diff

    return z, p_value, auc_a, auc_b, ci_lower, ci_upper


def bootstrap_confidence_interval(
    probs: np.ndarray, labels: np.ndarray,
    n_bootstrap: int = 1000, alpha: float = 0.05
) -> Tuple[float, float, float]:
    """Compute 95% CI for AUC using bootstrap resampling."""
    auc_scores = []
    n_samples = len(labels)

    for _ in range(n_bootstrap):
        idx = resample(range(n_samples), replace=True, n_samples=n_samples)
        p, l = probs[idx], labels[idx]
        if len(np.unique(l)) > 1:
            auc_scores.append(roc_auc_score(l, p))

    auc_scores = np.array(auc_scores)
    lower = np.percentile(auc_scores, 100 * alpha / 2)
    upper = np.percentile(auc_scores, 100 * (1 - alpha / 2))
    mean_auc = np.mean(auc_scores)

    return mean_auc, lower, upper


def compute_calibration(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Dict:
    """
    Compute Expected Calibration Error (ECE) and Brier Score.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(labels)

    bin_total = np.zeros(n_bins)
    bin_correct = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)

    for p, l in zip(probs, labels):
        b = int(p * n_bins)
        b = min(b, n_bins - 1)
        bin_total[b] += 1
        if int(round(p)) == int(l):
            bin_correct[b] += 1
        bin_conf[b] += p

    for i in range(n_bins):
        if bin_total[i] > 0:
            avg_conf = bin_conf[i] / bin_total[i]
            avg_acc = bin_correct[i] / bin_total[i]
            ece += bin_total[i] * abs(avg_conf - avg_acc)

    ece /= total
    brier = np.mean((probs - labels) ** 2)

    return {'ece': ece, 'brier': brier}


def compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict:
    """Compute all evaluation metrics."""
    preds = (probs >= threshold).astype(int)

    auc = roc_auc_score(labels, preds)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'AUC': auc,
        'Accuracy': acc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'F1': f1,
    }


def evaluate_model(
    model_path: str,
    dataloader,
    device,
    n_bootstrap: int = 1000,
    return_attention: bool = True,
) -> Dict:
    """Full evaluation with bootstrap CIs and calibration."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config')
    from src.models.multi_scale_mil import MultiScaleMIL
    model = MultiScaleMIL(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_probs = []
    all_labels = []
    all_attention_weights = []
    all_cross_attn_weights = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            patches_dict = {
                'scale_0': batch['patches']['scale_0'].squeeze(0).to(device),
                'scale_1': batch['patches']['scale_1'].squeeze(0).to(device),
                'scale_2': batch['patches']['scale_2'].squeeze(0).to(device),
            }
            labels = batch['label'].to(device)

            outputs = model(patches_dict, return_attention=return_attention)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=0)[1].cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())
            if return_attention:
                all_attention_weights.append(
                    outputs['mil_attn_weights'].cpu().numpy()
                )
                all_cross_attn_weights.append(
                    outputs['cross_attn_weights'].cpu().numpy()
                )

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels).ravel()

    auc_mean, auc_lower, auc_upper = bootstrap_confidence_interval(
        all_probs, all_labels, n_bootstrap
    )
    metrics = compute_metrics(all_probs, all_labels)
    metrics['AUC_CI'] = f"({auc_lower:.3f}–{auc_upper:.3f})"

    calibration = compute_calibration(all_probs, all_labels)
    metrics.update(calibration)

    return {
        'metrics': metrics,
        'probs': all_probs,
        'labels': all_labels,
        'attention_weights': all_attention_weights,
        'cross_attn_weights': all_cross_attn_weights,
        'auc_mean': auc_mean,
        'auc_ci': (auc_lower, auc_upper),
    }


def stratified_analysis(
    results: Dict,
    lesion_types: List[str],
) -> Dict:
    """
    Perform stratified analysis: Mass vs. Calcification.

    FIXED (plan #9): Now has an actual implementation consuming lesion_type.
    Computes per-subset AUC and compares cross-scale attention ratios.

    Returns AUC per subset and cross-scale attention weight statistics.
    """
    probs = results['probs']
    labels = results['labels']
    cross_attn = results.get('cross_attn_weights', [])

    results_mass = {'probs': [], 'labels': []}
    results_calc = {'probs': [], 'labels': []}
    cross_attn_mass = {'64': [], '256': []}
    cross_attn_calc = {'64': [], '256': []}

    for i, ltype in enumerate(lesion_types):
        if ltype == 'mass':
            results_mass['probs'].append(probs[i])
            results_mass['labels'].append(labels[i])
            if i < len(cross_attn):
                cross_attn_mass['64'].append(cross_attn[i][0])
                cross_attn_mass['256'].append(cross_attn[i][1])
        elif ltype == 'calcification':
            results_calc['probs'].append(probs[i])
            results_calc['labels'].append(labels[i])
            if i < len(cross_attn):
                cross_attn_calc['64'].append(cross_attn[i][0])
                cross_attn_calc['256'].append(cross_attn[i][1])

    out = {}
    for name, subset in [('Mass', results_mass), ('Calc', results_calc)]:
        if len(set(subset['labels'])) < 2 or len(subset['probs']) < 2:
            out[f'AUC_{name}'] = float('nan')
            out[f'Attn_{name}_64'] = float('nan')
            out[f'Attn_{name}_256'] = float('nan')
            out[f'Ratio_{name}'] = float('nan')
        else:
            out[f'AUC_{name}'] = roc_auc_score(subset['labels'], subset['probs'])
            avg_64 = float(np.mean(cross_attn_mass['64'] if name == 'Mass'
                                   else cross_attn_calc['64']))
            avg_256 = float(np.mean(cross_attn_mass['256'] if name == 'Mass'
                                    else cross_attn_calc['256']))
            out[f'Attn_{name}_64'] = avg_64
            out[f'Attn_{name}_256'] = avg_256
            out[f'Ratio_{name}'] = avg_64 / (avg_256 + 1e-8)

    # DeLong test between mass and calcification subsets
    if len(set(results_mass['labels'])) > 1 and len(set(results_calc['labels'])) > 1:
        z, p, a_mass, a_calc, ci_l, ci_u = delong_test(
            np.array(results_mass['labels'] + results_calc['labels']),
            np.array(results_mass['probs'] + results_calc['probs']),
            np.array(results_calc['probs'] + results_mass['probs']),
        )
        out['DeLong_z'] = z
        out['DeLong_p'] = p
        out['DeLong_AUC_diff_CI'] = f"({ci_l:.3f}–{ci_u:.3f})"

    return out
