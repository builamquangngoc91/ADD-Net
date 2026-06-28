"""
Combined loss function: Classification (CE or Focal) + LEM Regularization.

FIXED (plan #5): LEM loss is always computed in fp32 to avoid epsilon
underflow under mixed-precision training. The feature map is cast to fp32
before entering the LEM computation, which uses F.pad, min, sigmoid, and mean
— all numerically stable in fp32.
"""
import torch
import torch.nn as nn
from src.losses.lem_loss import MassNormalizedLEMLoss
from src.losses.focal_loss import FocalLoss


class TotalLoss(nn.Module):
    def __init__(self, config, warmup_epoch: int = 0, current_epoch: int = 0):
        super().__init__()
        self.config = config
        self.warmup_epoch = warmup_epoch
        self.current_epoch = current_epoch

        if config.loss.use_focal:
            alpha = config.loss.focal_alpha
            if isinstance(alpha, list):
                self.classification_loss = FocalLoss(alpha=alpha)
            else:
                self.classification_loss = FocalLoss(alpha=alpha)
        else:
            self.classification_loss = nn.CrossEntropyLoss()

        self.lem_loss = MassNormalizedLEMLoss(
            temperature=config.loss.lem_temperature,
            epsilon=config.loss.epsilon,
        )

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def forward(self, logits, labels, feature_map, compute_lem: bool = True):
        """
        Args:
            logits:       [num_classes]
            labels:       scalar
            feature_map:  [C, H, W] or [B, C, H, W] (backbone feature map)
            compute_lem:  whether to compute LEM (False during validation)
        Returns:
            total_loss: scalar
            losses:     dict with individual losses for logging
        """
        loss_cls = self.classification_loss(logits.unsqueeze(0), labels)

        loss_lem = torch.tensor(0.0, device=logits.device)
        if compute_lem and self.current_epoch >= self.warmup_epoch:
            # FIXED (plan #5): Cast to fp32 before LEM to avoid epsilon underflow
            # under autocast. F.pad, min, sigmoid, mean are all safe in fp32.
            fm_fp32 = feature_map.float()  # [N, C', H', W'] - per-patch feature map
            loss_lem = self.lem_loss(fm_fp32)

        total_loss = loss_cls + self.config.loss.lambda_lem * loss_lem

        return total_loss, {
            'cls_loss': loss_cls.item(),
            'lem_loss': loss_lem.item() if isinstance(loss_lem, torch.Tensor) else loss_lem,
            'lem_active': 1.0 if (compute_lem and self.current_epoch >= self.warmup_epoch) else 0.0,
        }
