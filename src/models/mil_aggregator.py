"""
Attention-based Multiple Instance Learning (AbMIL) Aggregator.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMILAggregator(nn.Module):
    def __init__(self, embed_dim: int = 512, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, patch_features: torch.Tensor):
        """
        Args:
            patch_features: [N, D] for one bag, or [B, N, D] for a batched bag stack.
        Returns:
            logits:       [num_classes] or [B, num_classes]
            attn_weights: [N, 1] or [B, N, 1]
        """
        if patch_features.dim() == 2:
            attn_scores = self.attention(patch_features)            # [N, 1]
            attn_weights = F.softmax(attn_scores, dim=0)            # [N, 1]
            bag_feature = torch.sum(attn_weights * patch_features, dim=0)  # [D]
            logits = self.classifier(bag_feature)                   # [num_classes]
            return logits, attn_weights

        # Batched: [B, N, D]
        B, N, D = patch_features.shape
        attn_scores = self.attention(patch_features)                # [B, N, 1]
        attn_weights = F.softmax(attn_scores, dim=1)                # [B, N, 1]
        bag_feature = torch.sum(attn_weights * patch_features, dim=1)  # [B, D]
        logits = self.classifier(bag_feature)                       # [B, num_classes]
        return logits, attn_weights
