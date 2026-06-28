"""
Mass-Normalized Differentiable Local Extremum Mapping Loss.

Key properties:
1. Fully differentiable via soft peakness indicator (sigmoid on min_diff/τ).
2. Mass-normalized denominator prevents trivial uniform-map solution.
3. Scale-invariant: global feature magnitudes cancel out.
4. Uses ReLU activation to allow true zero suppression of background.

Mathematical formulation:
    S = ReLU(||F||_2)
    M = sigmoid( min_{d in N}(S - S_d) / tau )
    L = - mean(S * M) / (mean(S) + epsilon)

Where M approximates a local maximum indicator. Implementation uses
per-sample ratio then batch-mean (see docstring).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MassNormalizedLEMLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, epsilon: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.epsilon = epsilon

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: [B, C, H, W]
        Returns:
            loss: scalar (mean over batch of per-sample ratios)

        Note: The loss is computed as mean(peakness(S) / mass(S)) per sample,
        then averaged over the batch. This is NOT a single global ratio.
        """
        B, C, H, W = feature_map.shape

        # 1. Response intensity using L2-norm then ReLU
        S_raw = torch.norm(feature_map, dim=1, p=2)
        S = F.relu(S_raw)

        # 2. Pad for 4-neighbor comparison
        S_pad = F.pad(S, (1, 1, 1, 1), mode='replicate')

        center = S_pad[:, 1:H+1, 1:W+1]
        up    = S_pad[:, 0:H,   1:W+1]
        down  = S_pad[:, 2:H+2, 1:W+1]
        left  = S_pad[:, 1:H+1, 0:W]
        right = S_pad[:, 1:H+1, 2:W+2]

        diffs = torch.stack([center - up, center - down,
                             center - left, center - right], dim=1)

        min_diff, _ = torch.min(diffs, dim=1)

        soft_indicator = torch.sigmoid(min_diff / self.temperature)

        # 3. Per-sample ratio, then batch mean
        numerator   = torch.mean(S * soft_indicator, dim=(1, 2))
        denominator = torch.mean(S, dim=(1, 2)) + self.epsilon

        loss = - (numerator / denominator).mean()
        return loss
