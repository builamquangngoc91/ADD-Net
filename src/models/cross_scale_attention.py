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
            f_64:  [N, D] or [B, N, D] features from 64×64 patches
            f_128: [N, D] or [B, N, D] features from 128×128 patches
            f_256: [N, D] or [B, N, D] features from 256×256 patches
        Returns:
            fused:        [N, D] or [B, N, D] fused features
            attn_weights: [N, 2] or [B, N, 2] attention over the two contextual scales
        """
        # FIXED (plan #6): Validate N-alignment from the patch extractor
        assert f_64.shape[0] == f_128.shape[0] == f_256.shape[0], (
            f"Outer-dim mismatch across scales: "
            f"64={f_64.shape[0]}, 128={f_128.shape[0]}, 256={f_256.shape[0]}. "
            f"Use MultiScalePatchExtractor with aligned anchor centers."
        )
        if f_64.dim() == 2:
            B, N = 1, f_64.shape[0]
        else:
            B, N = f_64.shape[:2]

        # Flatten (B, N, D) -> (B*N, D) so MHA's [B, L, D] input is satisfied
        # by treating each (bag, patch) as its own batch element. Per-patch
        # attention semantics are preserved because KV spans the same patches.
        if f_64.dim() == 3:
            f_64_flat = f_64.reshape(B * N, -1)
            f_128_flat = f_128.reshape(B * N, -1)
            f_256_flat = f_256.reshape(B * N, -1)
        else:
            f_64_flat, f_128_flat, f_256_flat = f_64, f_128, f_256

        # Query = middle scale (anchor)
        Q = self.W_q(f_128_flat).unsqueeze(1)               # [B*N, 1, D]
        # Key/Value = stack of small and large scales
        KV = torch.stack([self.W_k(f_64_flat), self.W_k(f_256_flat)], dim=1)  # [B*N, 2, D]
        VV = torch.stack([self.W_v(f_64_flat), self.W_v(f_256_flat)], dim=1)  # [B*N, 2, D]

        attn_out, attn_weights = self.attention(Q, KV, VV)  # [B*N,1,D], [B*N,1,2]
        attn_out = attn_out.squeeze(1)                      # [B*N, D]
        attn_weights = attn_weights.squeeze(1)              # [B*N, 2]

        # Restore original bag structure
        if B > 1:
            attn_out = attn_out.view(B, N, -1)
            attn_weights = attn_weights.view(B, N, 2)

        fused = self.norm(f_128 + self.gate * attn_out)

        return fused, attn_weights
