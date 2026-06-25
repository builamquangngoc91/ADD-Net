"""Stage 4 - Adaptive Multi-View Classifier.

Extracts 128x128 patch features from CC and MLO views using ResNet-50,
weights each view by its maximum pseudo-label score, fuses the representations,
and classifies as Normal (0) or Abnormal (1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PatchFeatureExtractor(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = resnet.fc.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        b = self.backbone(x)
        return b.view(b.size(0), -1)


class AdaptiveMultiViewClassifier(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.patch_extractor = PatchFeatureExtractor(pretrained=True)
        self.feature_dim = self.patch_extractor.feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def _extract_patch_features(self, image: torch.Tensor, patch_size: int = 128, stride: int = 64) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        B, C, H, W = image.shape
        patches = F.unfold(image, kernel_size=patch_size, stride=stride)
        num_patches = patches.shape[2]
        patches = patches.transpose(1, 2).reshape(B * num_patches, C, patch_size, patch_size)
        upsampled = F.interpolate(patches, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.no_grad():
            features = self.patch_extractor(upsampled)
        return features.reshape(B, num_patches, -1)

    def forward(
        self,
        x_cc: torch.Tensor,
        x_mlo: torch.Tensor,
        scores_cc: torch.Tensor,
        scores_mlo: torch.Tensor,
    ) -> torch.Tensor:
        patch_feats_cc = self._extract_patch_features(x_cc)
        patch_feats_mlo = self._extract_patch_features(x_mlo)

        w_cc = scores_cc.view(scores_cc.size(0), -1).max(dim=1)[0]
        w_mlo = scores_mlo.view(scores_mlo.size(0), -1).max(dim=1)[0]

        w_cc = w_cc.unsqueeze(1)
        w_mlo = w_mlo.unsqueeze(1)
        total = w_cc + w_mlo + 1e-8
        alpha_cc = w_cc / total
        alpha_mlo = w_mlo / total

        V_cc = patch_feats_cc.mean(dim=1)
        V_mlo = patch_feats_mlo.mean(dim=1)

        V_final = alpha_cc * V_cc + alpha_mlo * V_mlo

        logits = self.classifier(V_final)
        return logits

    def extract_all_patches(self, x_cc: torch.Tensor, x_mlo: torch.Tensor) -> torch.Tensor:
        f_cc = self._extract_patch_features(x_cc)
        f_mlo = self._extract_patch_features(x_mlo)
        return torch.cat([f_cc, f_mlo], dim=1)
