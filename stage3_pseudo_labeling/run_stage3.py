"""Stage 3 - Runner: generate and save soft pseudo-labels for all patients.

Iterates through the Stage 1 manifest, runs bilateral DDIM inference to
generate simulated contralateral breasts, computes Med-LPIPS asymmetry maps,
and applies GMM-based soft labeling.
"""

import json
import numpy as np
import torch
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import PipelineConfig
from stage2_diffusion.autoencoder import load_vqgan
from stage2_diffusion.unet_modules import CrossLateralDiffusionUNet
from stage3_pseudo_labeling.inference import run_bilateral_inference
from stage3_pseudo_labeling.perceptual_loss import compute_med_lpips_maps
from stage3_pseudo_labeling.statistical_gmm import calculate_gmm_pseudo_labels


def run_stage3(cfg: PipelineConfig = None):
    if cfg is None:
        cfg = PipelineConfig()

    manifest_path = cfg.processed_dir / "manifest_stage1.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Stage 1 manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    autoencoder = load_vqgan(cfg, device=cfg.device)

    diff_unet = CrossLateralDiffusionUNet(
        in_channels=cfg.latent_channels,
        out_channels=cfg.latent_channels,
        latent_h=cfg.latent_height,
        latent_w=cfg.latent_width,
    ).to(cfg.device)

    if cfg.diffusion_ckpt.exists():
        state = torch.load(cfg.diffusion_ckpt, map_location=cfg.device, weights_only=True)
        diff_unet.load_state_dict(state)
    diff_unet.eval()

    autoencoder.eval()
    cfg.stage3_pseudolabel_dir.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    all_soft_labels = {}
    files = manifest["aligned_files"]

    print(f"[Stage 3] Processing {len(files)} patients...")

    pbar = tqdm(files, desc="Stage 3", unit="patient", ncols=80)
    for idx, fpath in enumerate(pbar):
        batch = torch.load(fpath, map_location=cfg.device, weights_only=False)
        patient_id = batch["patient_id"]
        if isinstance(patient_id, list):
            patient_id = "_".join(patient_id)

        l_cc = batch["L_CC"].to(cfg.device)
        r_cc = batch["R_CC"].to(cfg.device)
        l_mlo = batch["L_MLO"].to(cfg.device)
        r_mlo = batch["R_MLO"].to(cfg.device)

        hat_I_R, hat_I_L, hat_I_R_mlo, hat_I_L_mlo = run_bilateral_inference(
            diff_unet, autoencoder,
            l_cc, r_cc, l_mlo, r_mlo,
            steps=cfg.ddim_steps,
            T=cfg.diffusion_timesteps,
        )

        cc_map, mlo_map = compute_med_lpips_maps(
            l_cc, r_cc, hat_I_R, hat_I_L,
            patch_size=cfg.patch_size,
            stride=cfg.stride,
        )

        labels = calculate_gmm_pseudo_labels(
            cc_map, mlo_map,
            components=cfg.num_gmm_components,
            sigmoid_temp=cfg.gmm_sigmoid_temp,
        )

        all_soft_labels[patient_id] = {
            "cc_map": labels["cc_map"].cpu().numpy(),
            "mlo_map": labels["mlo_map"].cpu().numpy(),
        }

        pbar.set_postfix({"done": idx + 1})

    pbar.close()

    save_path = cfg.stage3_pseudolabel_dir / "pseudo_labels.npy"
    np.save(save_path, all_soft_labels, allow_pickle=True)
    print(f"[Stage 3] Saved {len(all_soft_labels)} entries to {save_path}")
    return save_path


if __name__ == "__main__":
    run_stage3()
