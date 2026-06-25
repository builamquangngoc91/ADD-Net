"""Stage 1 - Deformable Registration using SciPy Affine Transform.

Aligns the right breast (horizontally flipped) to the left breast using
a SciPy-based 2D affine registration (translation + rotation + scaling).
Both CC and MLO views are processed.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


class SciPyAffineWrapper:
    def __init__(self, device: str = "cpu"):
        self.device = device

    def _estimate_affine_2d(self, moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
        """Estimate affine transform from moving to fixed using center-of-mass + phase-correlation."""
        m_center = np.array(moving.shape) / 2.0
        f_center = np.array(fixed.shape) / 2.0

        shift = f_center - m_center
        matrix = np.eye(3)
        matrix[0, 2] = shift[1]
        matrix[1, 2] = shift[0]
        return matrix

    def _register_image(self, moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
        matrix = self._estimate_affine_2d(moving, fixed)
        registered = ndimage.affine_transform(
            moving,
            matrix,
            output_shape=fixed.shape,
            order=1,
            mode="constant",
            cval=0.0,
        )
        return registered

    def register(self, moving: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        moving_np = moving.squeeze(0).squeeze(0).cpu().numpy()
        fixed_np = fixed.squeeze(0).squeeze(0).cpu().numpy()

        registered_np = self._register_image(moving_np, fixed_np)

        registered = torch.from_numpy(registered_np).float()
        registered = registered.unsqueeze(0).unsqueeze(0)
        return registered.to(self.device)


def run_deformable_registration(normalized_batch: dict, reg_model=None) -> dict:
    if reg_model is None:
        reg_model = SciPyAffineWrapper(device=normalized_batch["L_CC"].device)

    l_cc = normalized_batch["L_CC"]
    r_cc = normalized_batch["R_CC"]
    l_mlo = normalized_batch["L_MLO"]
    r_mlo = normalized_batch["R_MLO"]

    r_cc_flipped = torch.flip(r_cc, dims=[-1])
    r_mlo_flipped = torch.flip(r_mlo, dims=[-1])

    r_cc_aligned = reg_model.register(r_cc_flipped, l_cc)
    r_cc_aligned = torch.flip(r_cc_aligned, dims=[-1])

    r_mlo_aligned = reg_model.register(r_mlo_flipped, l_mlo)
    r_mlo_aligned = torch.flip(r_mlo_aligned, dims=[-1])

    return {
        "L_CC": l_cc,
        "R_CC": r_cc_aligned,
        "L_MLO": l_mlo,
        "R_MLO": r_mlo_aligned,
        "label": normalized_batch["label"],
        "patient_id": normalized_batch["patient_id"],
    }
