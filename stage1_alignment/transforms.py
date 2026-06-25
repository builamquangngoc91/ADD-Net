"""Stage 1 - Radiometric Normalization: Otsu masking, Z-score, and CLAHE."""

import numpy as np
import torch
import torch.nn.functional as F


def _otsu_threshold(image: torch.Tensor) -> float:
    lo = float(image.min())
    hi = float(image.max())
    if hi <= lo:
        return 0.0
    img_clamped = ((image - lo) / (hi - lo)).clamp(0.0, 1.0)
    hist = torch.histc(img_clamped, bins=256, min=0.0, max=1.0)
    hist = hist.cpu().numpy()
    total = hist.sum()
    if total == 0:
        return 0.0
    sum_total = np.dot(np.arange(256), hist)
    sum_bg, weight_bg = 0.0, 0.0
    max_variance, threshold = 0.0, 0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > max_variance:
            max_variance = variance
            threshold = t
    return threshold / 255.0


def _zscore_normalize(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    tissue = image * mask
    tissue_flat = tissue[tissue > 0]
    if tissue_flat.numel() == 0:
        return image * mask
    mean = tissue_flat.mean()
    std = tissue_flat.std()
    std = std if std > 1e-6 else 1.0
    normalized = (image - mean) / std
    normalized = normalized * mask
    if normalized.abs().sum() == 0:
        return image
    return normalized


def _apply_clahe(image: torch.Tensor, mask: torch.Tensor, clip_limit=2.0, tile_size=8) -> torch.Tensor:
    result = image.clone()
    tissue_mask = mask.bool()
    tissue = image[tissue_mask]
    if tissue.numel() == 0:
        return result

    img_clamped = image.squeeze(0).clamp(0.0, 1.0).cpu().numpy()
    mask_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)

    try:
        import cv2
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        clahe_np = clahe.apply((img_clamped * 255).astype(np.uint8))
        clahe_tensor = torch.from_numpy(clahe_np.astype(np.float32)).float() / 255.0
        result = clahe_tensor.unsqueeze(0) * mask + image * (1 - mask)
    except Exception:
        min_v, max_v = tissue.min().item(), tissue.max().item()
        if max_v > min_v:
            result = (image - min_v) / (max_v - min_v)
            result = result * mask + image * (1 - mask)

    return result


def run_radiometric_normalization(batch: dict, target_size=None) -> dict:
    result = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor) or value.dim() != 4:
            result[key] = value
            continue
        img = value
        threshold = _otsu_threshold(img)
        mask = (img > threshold).float()
        enhanced = _apply_clahe(img, mask)
        normalized = _zscore_normalize(enhanced, mask)
        result[key] = normalized
    return result
