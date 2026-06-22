"""Stage 4 - Supervised Contrastive Loss for patch-level features.

Encourages patches with the same soft pseudo-label to cluster together in
embedding space, while pushing patches with different pseudo-labels apart.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B, N, D = features.shape
        if B == 0:
            return features.new_zeros(())

        features = F.normalize(features, p=2, dim=-1)

        features_flat = features.view(B * N, D)
        labels_flat = labels.view(B * N)

        mask = labels_flat.unsqueeze(0) == labels_flat.unsqueeze(1)
        mask = mask.float()
        mask = mask.fill_diagonal_(0)

        logits = torch.matmul(features_flat, features_flat.T) / self.temperature
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits)

        denom = exp_logits.sum(dim=1, keepdim=True) - exp_logits.diag().unsqueeze(1)
        denom = denom.clamp(min=1e-8)

        log_probs = logits - denom.log()
        pos_mask_sum = mask.sum(dim=1).clamp(min=1e-8)
        loss_per_sample = -(mask * log_probs).sum(dim=1) / pos_mask_sum
        loss = loss_per_sample.mean()

        return loss
