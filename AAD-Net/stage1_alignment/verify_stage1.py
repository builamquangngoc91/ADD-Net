"""Stage 1 - Verifier: sanity-check aligned .pt files before Stage 2."""

import json

import torch

import sys
sys.path.insert(0, str(__file__).rsplit("/", 1)[0] if "/" in __file__ else ".")

from config.config import PipelineConfig


def verify_stage1(cfg: PipelineConfig = None):
    if cfg is None:
        cfg = PipelineConfig()

    manifest_path = cfg.processed_dir / "manifest_stage1.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    files = manifest["aligned_files"]
    if len(files) == 0:
        raise ValueError("No aligned files in manifest.")

    check_count = min(5, len(files))
    errors = []

    for i in range(check_count):
        path = files[i]
        batch = torch.load(path, map_location=cfg.device, weights_only=False)

        try:
            assert batch["L_CC"].shape == (1, 512, 1024), \
                f"L_CC shape {batch['L_CC'].shape} != (1, 512, 1024)"
            assert batch["R_CC"].shape == (1, 512, 1024), \
                f"R_CC shape {batch['R_CC'].shape} != (1, 512, 1024)"
            assert batch["L_MLO"].shape == (1, 512, 1024), \
                f"L_MLO shape {batch['L_MLO'].shape} != (1, 512, 1024)"
            assert batch["R_MLO"].shape == (1, 512, 1024), \
                f"R_MLO shape {batch['R_MLO'].shape} != (1, 512, 1024)"
            assert batch["label"].item() in [0, 1], \
                f"Label {batch['label']} is not binary (0 or 1)"
        except AssertionError as e:
            errors.append(f"  File {path}: {e}")

    if errors:
        print("Stage 1 Verification FAILED:")
        for err in errors:
            print(err)
        raise AssertionError("Stage 1 verification failed.")

    print(f"[Stage 1] Verified {check_count}/{len(files)} files. All assertions passed.")
    print("Stage 1 Data Verified.")
    return True


if __name__ == "__main__":
    verify_stage1()
