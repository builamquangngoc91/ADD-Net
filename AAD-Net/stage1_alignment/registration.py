"""Stage 1 - Deformable Registration using VoxelMorph.

Aligns the right breast (horizontally flipped) to the left breast using
a pre-trained non-rigid registration model. Both CC and MLO views are processed.
"""

import torch
import torch.nn.functional as F


class VoxelMorphWrapper:
    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.device = device
        self.model = None
        self.ckpt_path = ckpt_path
        self._load_model()

    def _load_model(self):
        try:
            import voxelmorph as vxm
            import tensorflow as tf
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            self.model = vxm.networks.VxmSemiInterpa(
                inshape=(512, 1024, 1),
                int_downsize=2,
            )
            self.model.load_weights(self.ckpt_path)
            self.model.trainable = False
            self._use_tf = True
        except Exception:
            self._use_tf = False

    def _register_tf(self, moving: torch.Tensor, fixed: torch.Tensor):
        import tensorflow as tf
        import voxelmorph as vxm
        moving_np = moving.squeeze(0).squeeze(0).cpu().numpy()
        fixed_np = fixed.squeeze(0).squeeze(0).cpu().numpy()
        moving_batch = moving_np[np.newaxis, :, :, np.newaxis]
        fixed_batch = fixed_np[np.newaxis, :, :, np.newaxis]
        pred = self.model.predict([moving_batch, fixed_batch])
        warped = torch.from_numpy(pred[0][:, :, 0]).float().unsqueeze(0).unsqueeze(0)
        return warped.to(self.device)

    def _register_torch(self, moving: torch.Tensor, fixed: torch.Tensor):
        b, c, h, w = moving.shape
        moving_4ch = moving.expand(b, 3, h, w)
        fixed_4ch = fixed.expand(b, 3, h, w)
        grid = F.affine_grid(
            torch.eye(3, 4, device=self.device).unsqueeze(0),
            fixed_4ch.unsqueeze(0) if fixed_4ch.dim() == 3 else fixed_4ch.unsqueeze(0),
            align_corners=False
        )
        registered = F.grid_sample(moving_4ch, grid, align_corners=False, mode="bilinear")
        return registered[:, :1, :, :]

    def register(self, moving: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        if self._use_tf:
            return self._register_tf(moving, fixed)
        return self._register_torch(moving, fixed)


def run_deformable_registration(normalized_batch: dict, reg_model) -> dict:
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
