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
            patch_features: [N, D] features for all patches in the bag
        Returns:
            logits:       [num_classes]
            attn_weights: [N, 1]
        """
        attn_scores = self.attention(patch_features)
        attn_weights = F.softmax(attn_scores, dim=0)

        bag_feature = torch.sum(attn_weights * patch_features, dim=0)
        logits = self.classifier(bag_feature)

        return logits, attn_weights
