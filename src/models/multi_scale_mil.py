"""
Complete Multi-Scale MIL model with Cross-Scale Attention.
"""
import torch
import torch.nn as nn
from src.models.encoder import SharedEncoder
from src.models.cross_scale_attention import CrossScaleAttention
from src.models.mil_aggregator import AttentionMILAggregator


class MultiScaleMIL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = SharedEncoder(
            backbone=config.model.backbone,
            embed_dim=config.model.embed_dim,
            pretrained=True,
            use_checkpointing=config.model.use_checkpointing,
        )

        self.cross_attn = CrossScaleAttention(
            embed_dim=config.model.embed_dim,
            num_heads=config.model.num_heads,
        )

        self.mil = AttentionMILAggregator(
            embed_dim=config.model.embed_dim,
            hidden_dim=config.model.mil_hidden_dim,
            num_classes=config.model.num_classes,
        )

    def forward(self, patches_dict, return_feature_map=False, return_attention=False):
        """
        Args:
            patches_dict: {
                'scale_0': [N, 1, 64, 64] or [B, N, 1, 64, 64],
                'scale_1': [N, 1, 128, 128] or [B, N, 1, 128, 128],
                'scale_2': [N, 1, 256, 256] or [B, N, 1, 256, 256]
            }
            return_feature_map: whether to return feature map for LEM loss
            return_attention: whether to return cross-scale attention weights
        Returns:
            dict with keys: logits, mil_attn_weights, [feature_map], [cross_attn_weights]
        """
        B = None
        s0 = patches_dict['scale_0']
        if s0.dim() == 5:
            B = s0.size(0)
            # Collapse batch and bag dim for the encoder: [B, N, 1, H, W] -> [B*N, 1, H, W]
            s0_in = s0.reshape(B * s0.size(1), *s0.shape[2:])
            s1_in = patches_dict['scale_1'].reshape(B * patches_dict['scale_1'].size(1), *patches_dict['scale_1'].shape[2:])
            s2_in = patches_dict['scale_2'].reshape(B * patches_dict['scale_2'].size(1), *patches_dict['scale_2'].shape[2:])
        else:
            s0_in = s0
            s1_in = patches_dict['scale_1']
            s2_in = patches_dict['scale_2']

        # 1. Encode each scale with shared encoder
        if return_feature_map:
            f_64, _ = self.encoder(s0_in, return_fm=True)
            f_128, fm_128 = self.encoder(s1_in, return_fm=True)
            f_256, _ = self.encoder(s2_in, return_fm=True)
        else:
            f_64 = self.encoder(s0_in)
            f_128 = self.encoder(s1_in)
            f_256 = self.encoder(s2_in)

        if B is not None:
            N = f_64.size(0) // B
            # Restore batch dim: [B*N, D] -> [B, N, D]
            f_64 = f_64.view(B, N, -1)
            f_128 = f_128.view(B, N, -1)
            f_256 = f_256.view(B, N, -1)
            if return_feature_map:
                fm_128 = fm_128.view(
                    B * N, *fm_128.shape[1:]
                )  # LEM sees flat (B*N) patches

        # 2. Cross-Scale Attention fusion
        fused_features, cross_attn_weights = self.cross_attn(f_64, f_128, f_256)

        # 3. MIL aggregation (handles both [N, D] and [B, N, D])
        logits, mil_attn_weights = self.mil(fused_features)

        outputs = {'logits': logits, 'mil_attn_weights': mil_attn_weights}

        if return_feature_map:
            outputs['feature_map'] = fm_128

        if return_attention:
            outputs['cross_attn_weights'] = cross_attn_weights

        return outputs
