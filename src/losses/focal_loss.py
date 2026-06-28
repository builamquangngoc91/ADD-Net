"""
Focal Loss with multi-class support.

FIXED (plan #3): alpha is now a per-class tensor (or list), not a scalar.
This supports both binary (alpha=[α, 1-α]) and multi-class (BI-RADS) settings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance.
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Args:
        alpha: Per-class weights. Can be:
               - list/tuple of floats (length = num_classes): explicit class weights
               - scalar α: binary FL with class-1 weight α, class-0 weight (1-α)
        gamma: Focusing parameter (2.0 recommended)
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(
        self,
        alpha=None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (list, tuple)):
            self.alpha = nn.Parameter(
                torch.tensor(alpha, dtype=torch.float32), requires_grad=False
            )
        else:
            # Scalar alpha -> binary FL (legacy path)
            self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C] raw logits (before softmax)
            targets: [B] class indices (0..C-1) OR [B, C] one-hot targets
        Returns:
            loss: scalar tensor
        """
        num_classes = logits.size(1)
        B = logits.size(0)

        ce_loss = F.cross_entropy(
            logits, targets, reduction='none', ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce_loss)

        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_tensor = self.alpha.to(logits.device)
            if alpha_tensor.dim() == 0:
                # Scalar alpha -> binary FL
                alpha_t = alpha_tensor * targets.float() + \
                          (1 - alpha_tensor) * (1 - targets.float())
            elif alpha_tensor.numel() == num_classes:
                # Per-class alpha tensor [C]
                if targets.dim() == 1:
                    alpha_t = alpha_tensor[targets]
                else:
                    alpha_t = (alpha_tensor.unsqueeze(0) * targets.float()).sum(dim=1)
            else:
                raise ValueError(
                    f"alpha length {alpha_tensor.numel()} != num_classes {num_classes}"
                )
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
