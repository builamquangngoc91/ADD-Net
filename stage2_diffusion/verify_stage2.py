"""Stage 2 - Verifier: validate the Diffusion U-Net structural integrity."""

import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import PipelineConfig
from stage2_diffusion.unet_modules import CrossLateralDiffusionUNet


def verify_stage2(cfg: PipelineConfig = None):
    if cfg is None:
        cfg = PipelineConfig()

    if not cfg.diffusion_ckpt.exists():
        raise FileNotFoundError(f"Diffusion checkpoint not found: {cfg.diffusion_ckpt}")

    diff_unet = CrossLateralDiffusionUNet(
        in_channels=cfg.latent_channels,
        out_channels=cfg.latent_channels,
        latent_h=cfg.latent_height,
        latent_w=cfg.latent_width,
    ).to(cfg.device)

    state = torch.load(cfg.diffusion_ckpt, map_location=cfg.device, weights_only=True)
    diff_unet.load_state_dict(state)
    diff_unet.eval()

    x = torch.randn(1, cfg.latent_channels, cfg.latent_height, cfg.latent_width, device=cfg.device)
    t = torch.randint(0, 1000, (1,), device=cfg.device).long()
    condition = torch.randn(1, cfg.latent_channels, cfg.latent_height, cfg.latent_width, device=cfg.device)

    with torch.no_grad():
        out = diff_unet(x, t, condition)

    assert out.shape == x.shape, f"Output shape {out.shape} != input shape {x.shape}"

    print("Diffusion U-Net structural integrity verified.")
    return True


if __name__ == "__main__":
    verify_stage2()
