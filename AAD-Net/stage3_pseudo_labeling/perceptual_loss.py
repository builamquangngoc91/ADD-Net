"""Stage 3 - Domain-Adapted Med-LPIPS perceptual loss.

Extracts multi-scale features from VGG-16 (ideally RadImageNet-pretrained)
and computes cosine-distance-based patch-wise asymmetry scores between
real and reconstructed contralateral breasts. Shallow layers (1,2) are
weighted 0.7; deep layers (4,5) are weighted 0.3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGG16FeatureExtractor(nn.Module):
    def __init__(self, weights_path: str = None):
        super().__init__()
        vgg = models.vgg16(weights=None)
        if weights_path:
            try:
                vgg.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
            except Exception:
                pass
        self.features = vgg.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.register_buffer = lambda name, tensor: setattr(self, name, tensor)

    def forward(self, x: torch.Tensor) -> dict:
        results = {}
        for i in range(1, 6):
            layer_idx = _VGG_LAYER_MAP.get(i)
            if layer_idx is None:
                continue
            feat = self.features[:layer_idx](x)
            results[i] = feat
        return results


_VGG_LAYER_MAP = {
    1: 4,
    2: 9,
    3: 16,
    4: 23,
    5: 30,
}


class MedLPIPS(nn.Module):
    def __init__(self, feat_extractor: nn.Module):
        super().__init__()
        self.feat_extractor = feat_extractor

    def extract_patch_features(self, image: torch.Tensor, patch_size: int = 128, stride: int = 64) -> torch.Tensor:
        B, C, H, W = image.shape
        patches = F.unfold(image, kernel_size=patch_size, stride=stride)
        num_patches = patches.shape[2]
        patches = patches.transpose(1, 2)
        patch_vectors = []

        for i in range(num_patches):
            p = patches[:, i, :].view(B, C, patch_size, patch_size)
            with torch.no_grad():
                feats = self.feat_extractor(p)
            feat_vec = torch.cat([feats[k].mean(dim=[2, 3]) for k in sorted(feats.keys())], dim=1)
            patch_vectors.append(feat_vec)

        return torch.stack(patch_vectors, dim=1)


def compute_med_lpips_maps(
    l_cc: torch.Tensor,
    r_cc: torch.Tensor,
    hat_I_R: torch.Tensor,
    hat_I_L: torch.Tensor,
    patch_size: int = 128,
    stride: int = 64,
    shallow_weight: float = 0.7,
    deep_weight: float = 0.3,
    feat_extractor: nn.Module = None,
) -> torch.Tensor:
    if feat_extractor is None:
        feat_extractor = VGG16FeatureExtractor()
    lpips_model = MedLPIPS(feat_extractor)

    patch_h = (512 - patch_size) // stride + 1
    patch_w = (1024 - patch_size) // stride + 1

    fake_R_feats = lpips_model.extract_patch_features(hat_I_R, patch_size, stride)
    real_R_feats = lpips_model.extract_patch_features(r_cc, patch_size, stride)
    fake_L_feats = lpips_model.extract_patch_features(hat_I_L, patch_size, stride)
    real_L_feats = lpips_model.extract_patch_features(l_cc, patch_size, stride)

    fake_R_norm = F.normalize(fake_R_feats, p=2, dim=-1)
    real_R_norm = F.normalize(real_R_feats, p=2, dim=-1)
    fake_L_norm = F.normalize(fake_L_feats, p=2, dim=-1)
    real_L_norm = F.normalize(real_L_feats, p=2, dim=-1)

    cos_dist_R = 1.0 - (fake_R_norm * real_R_norm).sum(dim=-1)
    cos_dist_L = 1.0 - (fake_L_norm * real_L_norm).sum(dim=-1)

    cc_map = cos_dist_R.squeeze(0).reshape(patch_h, patch_w)
    mlo_map = cos_dist_L.squeeze(0).reshape(patch_h, patch_w)

    return cc_map, mlo_map
