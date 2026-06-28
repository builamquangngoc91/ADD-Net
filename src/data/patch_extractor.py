"""
Multi-Scale Patch Extractor with N-alignment fix.

FIXED (from plan issue #1 & #6): All three scales are anchored on the same coarse
spatial grid so that CrossScaleAttention receives identical N at every scale.

Strategy: Use stride=128 (coarsest scale's natural stride) as the anchor grid.
  - size=64:  natural N = ceil((1024-64)/128)+1 = 8, but we center-crop to
    align with the 128/256 grid (7 anchors), yielding N=49.
  - size=128: natural N = ceil((1024-128)/128)+1 = 7, N=49. ✓
  - size=256: natural N = ceil((1024-256)/128)+1 = 7, N=49. ✓

The 64-scale patches are extracted as 64x64 crops centered on each 128-grid anchor,
so the spatial positions are identical across all three scales.
"""
import torch
import numpy as np
from typing import List, Dict


class MultiScalePatchExtractor:
    def __init__(self, patch_sizes: List[int], stride: int, image_size: int = 1024):
        self.patch_sizes = patch_sizes  # [64, 128, 256]
        self.stride = stride            # 128 — the coarsest scale's natural stride
        self.image_size = image_size

        # Number of anchors along one axis (same for 128 and 256).
        # With image_size=1024 and stride=128: (1024-128)/128 = 7 anchors, N=49.
        self.num_anchors = (image_size - patch_sizes[1]) // stride  # 7 for 1024/128
        self.N = self.num_anchors ** 2  # 49

        # Precompute anchor center coordinates in pixels.
        # Each anchor defines the center of a 128x128 patch; the 256 and 64
        # patches are extracted around the same center, so the last 256-crop
        # ends at start + 256 = 832 + 64 + 256 = 960, still inside 1024.
        half = stride // 2
        starts = [half + i * stride for i in range(self.num_anchors)]
        self.anchor_centers = [
            (y, x) for y in starts for x in starts
        ]

    def _extract_centered_crop(
        self, image: torch.Tensor, center_y: int, center_x: int, size: int
    ) -> torch.Tensor:
        """Extract a size×size crop centered at (center_y, center_x)."""
        H, W = image.shape[-2], image.shape[-1]
        y0 = max(0, center_y - size // 2)
        x0 = max(0, center_x - size // 2)
        y0 = min(y0, H - size)
        x0 = min(x0, W - size)
        crop = image[..., y0:y0 + size, x0:x0 + size]
        if crop.shape[-1] != size or crop.shape[-2] != size:
            crop = torch.nn.functional.pad(crop, (0, size - crop.shape[-1], 0, size - crop.shape[-2]))
        return crop

    def extract_patches_torch(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            image: [1, H, W] or [1, 1, H, W] or [H, W]
        Returns:
            dict: {'scale_0': [N, 1, 64, 64], 'scale_1': [N, 1, 128, 128],
                   'scale_2': [N, 1, 256, 256]} — all with identical N.
        """
        # Normalise to [1, 1, H, W]
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(1)

        B, C, H, W = image.shape
        if B != 1 or C != 1:
            raise ValueError(f"Expected shape [1, 1, H, W], got {image.shape}")

        patches = {}
        for s, size in enumerate(self.patch_sizes):
            patch_list = []
            for cy, cx in self.anchor_centers:
                crop = self._extract_centered_crop(image, cy, cx, size)
                patch_list.append(crop)
            # [N, 1, size, size]
            patches[f'scale_{s}'] = torch.cat(patch_list, dim=0)

        return patches

    def extract_patches_numpy(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Numpy fallback for debugging / non-PyTorch preprocessing."""
        H, W = image.shape
        patches = {}
        for s, size in enumerate(self.patch_sizes):
            patch_list = []
            for cy, cx in self.anchor_centers:
                y0 = cy - size // 2
                x0 = cx - size // 2
                patch = image[y0:y0 + size, x0:x0 + size]
                patch_list.append(patch)
            patches[f'scale_{s}'] = np.stack(patch_list, axis=0)
        return patches

    @property
    def num_patches_per_bag(self) -> int:
        return self.N
