"""Stage 3 - GMM Soft Labeling.

Dynamically thresholds patch-wise asymmetry scores using an unsupervised
2-component Gaussian Mixture Model to generate soft pseudo-labels for
the weakly supervised classifier.
"""

import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
import numpy as np


def _compute_z_scores(scores: torch.Tensor, mean_ref: float = None, std_ref: float = None) -> torch.Tensor:
    flat = scores.flatten().cpu().numpy()
    if mean_ref is None:
        mean_val = float(flat.mean())
    else:
        mean_val = mean_ref
    if std_ref is None:
        std_val = float(flat.std()) + 1e-8
    else:
        std_val = std_ref + 1e-8
    z = (flat - mean_val) / std_val
    return torch.from_numpy(z).float()


def _find_gmm_threshold(gmm: GaussianMixture, X: np.ndarray) -> float:
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_).flatten()
    sorted_idx = np.argsort(means)
    m1, s1 = means[sorted_idx[0]], stds[sorted_idx[0]]
    m2, s2 = means[sorted_idx[1]], stds[sorted_idx[1]]
    a = 1 / (2 * s1 ** 2) - 1 / (2 * s2 ** 2)
    b = m2 / (s2 ** 2) - m1 / (s1 ** 2)
    c = -m1 ** 2 / (2 * s1 ** 2) + m2 ** 2 / (2 * s2 ** 2) + np.log(s2 / s1)
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return float(np.mean(means))
    t1 = (-b + np.sqrt(discriminant)) / (2 * a)
    t2 = (-b - np.sqrt(discriminant)) / (2 * a)
    t_gmm = float((t1 + t2) / 2.0)
    return t_gmm


def _apply_scaled_sigmoid(z_scores: torch.Tensor, threshold: float, temp: float = 0.1) -> torch.Tensor:
    return torch.sigmoid((z_scores - threshold) / temp)


def calculate_gmm_pseudo_labels(
    cc_map: torch.Tensor,
    mlo_map: torch.Tensor,
    components: int = 2,
    ref_mean: float = None,
    ref_std: float = None,
    sigmoid_temp: float = 0.1,
) -> dict:
    z_cc = _compute_z_scores(cc_map, ref_mean, ref_std)
    z_mlo = _compute_z_scores(mlo_map, ref_mean, ref_std)

    z_cc_np = z_cc.flatten().unsqueeze(1).cpu().numpy()
    z_mlo_np = z_mlo.flatten().unsqueeze(1).cpu().numpy()

    gmm_cc = GaussianMixture(n_components=components, random_state=42)
    gmm_cc.fit(z_cc_np)
    threshold_cc = _find_gmm_threshold(gmm_cc, z_cc_np)

    gmm_mlo = GaussianMixture(n_components=components, random_state=42)
    gmm_mlo.fit(z_mlo_np)
    threshold_mlo = _find_gmm_threshold(gmm_mlo, z_mlo_np)

    p_cc = _apply_scaled_sigmoid(z_cc, threshold_cc, sigmoid_temp)
    p_mlo = _apply_scaled_sigmoid(z_mlo, threshold_mlo, sigmoid_temp)

    p_cc_2d = p_cc.reshape(cc_map.shape)
    p_mlo_2d = p_mlo.reshape(mlo_map.shape)

    return {
        "cc_map": p_cc_2d,
        "mlo_map": p_mlo_2d,
        "z_cc": z_cc,
        "z_mlo": z_mlo,
        "threshold_cc": threshold_cc,
        "threshold_mlo": threshold_mlo,
    }
