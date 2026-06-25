"""Stage 2 - Runner: train the Diffusion U-Net on Normal patients only.

Filters aligned .pt files to only those with label=0, then trains the
Cross-Lateral Diffusion U-Net using the latent diffusion loss.
"""

import json
import torch
import torch.optim as optim
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import PipelineConfig
from stage2_diffusion.autoencoder import load_vqgan
from stage2_diffusion.unet_modules import CrossLateralDiffusionUNet
from stage2_diffusion.trainer import LatentDiffusionTrainer


def run_stage2(cfg: PipelineConfig = None):
    if cfg is None:
        cfg = PipelineConfig()

    manifest_path = cfg.processed_dir / "manifest_stage1.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Stage 1 manifest not found: {manifest_path}. Run stage 1 first.")

    with open(manifest_path) as f:
        manifest = json.load(f)

    normal_files = []
    for fpath in manifest["aligned_files"]:
        batch = torch.load(fpath, map_location="cpu", weights_only=False)
        if batch["label"].item() == 0:
            normal_files.append(fpath)

    print(f"[Stage 2] Normal patients: {len(normal_files)} / {len(manifest['aligned_files'])}")

    autoencoder = load_vqgan(cfg, device=cfg.device)
    diff_unet = CrossLateralDiffusionUNet(
        in_channels=cfg.latent_channels,
        out_channels=cfg.latent_channels,
        latent_h=cfg.latent_height,
        latent_w=cfg.latent_width,
    ).to(cfg.device)

    optimizer = optim.Adam(diff_unet.parameters(), lr=cfg.lr)
    trainer = LatentDiffusionTrainer(T=cfg.diffusion_timesteps, device=cfg.device)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    total_steps = cfg.diff_epochs * len(normal_files)
    global_pbar = tqdm(
        total=cfg.diff_epochs,
        desc="Stage 2 (epochs)",
        unit="epoch",
        ncols=80,
    )

    for epoch in range(cfg.diff_epochs):
        diff_unet.train()
        epoch_loss = 0.0
        n_batches = 0

        epoch_pbar = tqdm(
            normal_files,
            desc=f"  Epoch {epoch + 1}/{cfg.diff_epochs}",
            unit="patient",
            ncols=80,
            leave=False,
        )

        for fpath in epoch_pbar:
            batch = torch.load(fpath, map_location=cfg.device, weights_only=False)

            l_cc = batch["L_CC"].to(cfg.device).unsqueeze(0)
            r_cc = batch["R_CC"].to(cfg.device).unsqueeze(0)

            loss = trainer.train_step(l_cc, r_cc, autoencoder, diff_unet)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            n_batches += 1

            epoch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_pbar.close()

        avg_loss = epoch_loss / max(n_batches, 1)
        global_pbar.set_postfix({"avg_loss": f"{avg_loss:.6f}"})
        global_pbar.update(1)

    global_pbar.close()

    ckpt_path = cfg.diffusion_ckpt
    torch.save(diff_unet.state_dict(), ckpt_path)
    print(f"[Stage 2] Saved: {ckpt_path}")
    return ckpt_path


if __name__ == "__main__":
    run_stage2()
