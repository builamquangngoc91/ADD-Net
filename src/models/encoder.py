"""
Shared-weight encoder for multi-scale patches.
Supports ResNet and ViT backbones.

FIXED: Added gradient checkpointing to save VRAM (plan #5 context).
FIXED: Added shape assertion for ViT feature map reconstruction.
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.checkpoint as checkpoint
import timm


class SharedEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = 'resnet50',
        embed_dim: int = 512,
        pretrained: bool = True,
        use_checkpointing: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.embed_dim = embed_dim
        self.use_checkpointing = use_checkpointing

        if backbone.startswith('resnet'):
            base = getattr(models, backbone)(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(base.fc.in_features, embed_dim)
            self.has_feature_map = True

        elif backbone.startswith('vit'):
            base = timm.create_model(backbone, pretrained=pretrained)
            self.backbone = base
            self.avgpool = nn.Identity()
            self.fc = nn.Linear(base.embed_dim, embed_dim)
            self.has_feature_map = False
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, patch: torch.Tensor, return_fm: bool = False):
        """
        Args:
            patch: [B, C, H, W] (C=1 for grayscale mammography patches)
            return_fm: Whether to return spatial feature map for LEM loss
        Returns:
            embedding: [B, embed_dim]
            feature_map: [B, C', H', W'] (if return_fm=True)
        """
        if self.backbone_name.startswith('resnet'):
            # Tile grayscale to 3 channels so ImageNet-pretrained ResNet weights apply.
            if patch.size(1) == 1:
                patch = patch.repeat(1, 3, 1, 1)

            # Gradient checkpointing saves VRAM when training with large bags
            if self.use_checkpointing and self.training:
                features = checkpoint.checkpoint(
                    self.backbone, patch, use_reentrant=False
                )
            else:
                features = self.backbone(patch)

            if return_fm:
                feature_map = features

            pooled = self.avgpool(features).flatten(1)
            embedding = self.fc(pooled)

        else:  # ViT
            outputs = self.backbone.forward_features(patch)

            cls_out = outputs[:, 0, :]
            embedding = self.fc(cls_out)

            if return_fm:
                spatial = outputs[:, 1:, :]
                B, seq_len, D = spatial.shape
                spatial_dim = int(patch.size(2) / 16)
                expected_len = spatial_dim * spatial_dim
                if seq_len != expected_len:
                    raise RuntimeError(
                        f"ViT spatial tokens {seq_len} != expected {expected_len} "
                        f"(H/16={spatial_dim}). Check ViT variant and image size."
                    )
                feature_map = spatial.permute(0, 2, 1).view(
                    B, D, spatial_dim, spatial_dim
                )
            else:
                feature_map = None

        if return_fm:
            return embedding, feature_map
        return embedding
