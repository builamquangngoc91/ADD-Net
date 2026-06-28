"""
Asymmetric Cross-Scale Attention Fusion.

FIXED (plan #2 & #6): Assert all three scale feature tensors have identical N
before cross-scale attention. This is guaranteed by the MultiScalePatchExtractor
using aligned anchor centers, but we validate at runtime for safety.
"""
import torch
import torch.nn as nn


class CrossScaleAttention(nn.Module):
    """
    Asymmetric Cross-Scale Attention Fusion.
    Query = middle scale (128×128), Key/Value = {small (64×64), large (256×256)}.

    Clinical Justification: The middle scale (128×128) serves as the anchor,
    corresponding to the median receptive field used in standard mammography models.
    """
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.gate = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, f_64: torch.Tensor, f_128: torch.Tensor, f_256: torch.Tensor):
        """
        Args:
            f_64:  [N, D] features from 64×64 patches
            f_128: [N, D] features from 128×128 patches
            f_256: [N, D] features from 256×256 patches
        Returns:
            fused:       [N, D] fused features
            attn_weights: [N, 2] attention over the two contextual scales
        """
        # FIXED (plan #6): Validate N-alignment from the patch extractor
        assert f_64.shape[0] == f_128.shape[0] == f_256.shape[0], (
            f"N mismatch across scales: "
            f"64={f_64.shape[0]}, 128={f_128.shape[0]}, 256={f_256.shape[0]}. "
            f"Use MultiScalePatchExtractor with aligned anchor centers."
        )
        N = f_64.shape[0]

        # Query = middle scale (anchor)
        Q = self.W_q(f_128).unsqueeze(1)   # [N, 1, D]
        # Key/Value = stack of small and large scales
        KV = torch.stack([self.W_k(f_64), self.W_k(f_256)], dim=1)  # [N, 2, D]
        VV = torch.stack([self.W_v(f_64), self.W_v(f_256)], dim=1)  # [N, 2, D]

        attn_out, attn_weights = self.attention(Q, KV, VV)  # [N,1,D], [N,2]
        attn_out = attn_out.squeeze(1)

        fused = self.norm(f_128 + self.gate * attn_out)

        return fused, attn_weights
