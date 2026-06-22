"""Stage 1 - Orchestrator: radiometric normalization + deformable registration.

Processes the full VinDr mammography dataset through Stage 1 and saves
aligned batches as .pt files alongside a manifest JSON.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config.config import PipelineConfig
from data_loaders.reader import MammographyLoader
from stage1_alignment.transforms import run_radiometric_normalization
from stage1_alignment.registration import VoxelMorphWrapper, run_deformable_registration


def run_stage1(cfg: PipelineConfig = None):
    if cfg is None:
        cfg = PipelineConfig()

    cfg.stage1_aligned_dir.mkdir(parents=True, exist_ok=True)

    loader = MammographyLoader(cfg, split="training")
    data_loader = DataLoader(
        loader, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )

    reg_model = None
    if cfg.voxelmorph_ckpt.exists():
        reg_model = VoxelMorphWrapper(str(cfg.voxelmorph_ckpt), device=cfg.device)
    else:
        print("[WARNING] VoxelMorph checkpoint not found. Registration will use identity transform.")

    saved_files = []

    for batch in data_loader:
        patient_id = batch["patient_id"][0]

        norm_batch = run_radiometric_normalization(batch, cfg.target_size)

        if reg_model is not None:
            aligned_batch = run_deformable_registration(norm_batch, reg_model)
        else:
            aligned_batch = {
                "L_CC": norm_batch["L_CC"],
                "R_CC": norm_batch["R_CC"],
                "L_MLO": norm_batch["L_MLO"],
                "R_MLO": norm_batch["R_MLO"],
                "label": norm_batch["label"],
                "patient_id": patient_id,
            }

        save_path = cfg.stage1_aligned_dir / f"{patient_id}.pt"
        torch.save(aligned_batch, save_path)
        saved_files.append(str(save_path))

        if len(saved_files) % 500 == 0:
            print(f"  Processed {len(saved_files)} patients...")

    manifest = {
        "aligned_files": saved_files,
        "base_path": str(cfg.stage1_aligned_dir),
        "total_count": len(saved_files),
    }
    manifest_path = cfg.processed_dir / "manifest_stage1.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Stage 1] Complete. {len(saved_files)} patients processed.")
    print(f"[Stage 1] Manifest saved to {manifest_path}")
    return manifest


if __name__ == "__main__":
    run_stage1()
