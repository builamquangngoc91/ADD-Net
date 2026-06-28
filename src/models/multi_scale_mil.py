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
                'scale_0': [N, 1, 64, 64],
                'scale_1': [N, 1, 128, 128],
                'scale_2': [N, 1, 256, 256]
            }
            return_feature_map: whether to return feature map for LEM loss
            return_attention: whether to return cross-scale attention weights
        Returns:
            dict with keys: logits, mil_attn_weights, [feature_map], [cross_attn_weights]
        """
        # 1. Encode each scale with shared encoder
        if return_feature_map:
            f_64, _ = self.encoder(patches_dict['scale_0'], return_fm=True)
            f_128, fm_128 = self.encoder(patches_dict['scale_1'], return_fm=True)
            f_256, _ = self.encoder(patches_dict['scale_2'], return_fm=True)
        else:
            f_64 = self.encoder(patches_dict['scale_0'])
            f_128 = self.encoder(patches_dict['scale_1'])
            f_256 = self.encoder(patches_dict['scale_2'])

        # 2. Cross-Scale Attention fusion
        fused_features, cross_attn_weights = self.cross_attn(f_64, f_128, f_256)

        # 3. MIL aggregation
        logits, mil_attn_weights = self.mil(fused_features)

        outputs = {'logits': logits, 'mil_attn_weights': mil_attn_weights}

        if return_feature_map:
            outputs['feature_map'] = fm_128

        if return_attention:
            outputs['cross_attn_weights'] = cross_attn_weights

        return outputs
