"""Stage 3 - Verifier: ensure pseudo-labels have valid dynamic thresholds."""

import numpy as np
import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import PipelineConfig


def verify_stage3(cfg: PipelineConfig = None):
    if cfg is None:
        cfg = PipelineConfig()

    save_path = cfg.stage3_pseudolabel_dir / "pseudo_labels.npy"
    if not save_path.exists():
        raise FileNotFoundError(f"Pseudo-label file not found: {save_path}")

    data = np.load(save_path, allow_pickle=True).item()
    keys = list(data.keys())

    if len(keys) == 0:
        raise ValueError("Pseudo-label dictionary is empty.")

    errors = []
    check_count = min(5, len(keys))

    for i in range(check_count):
        key = keys[i]
        entry = data[key]
        cc_map = entry["cc_map"]
        mlo_map = entry["mlo_map"]

        if not (cc_map.max() > cc_map.min()):
            errors.append(f"  {key}: cc_map max ({cc_map.max()}) <= min ({cc_map.min()})")
        if cc_map.max() > 1.0 or cc_map.min() < 0.0:
            errors.append(f"  {key}: cc_map out of [0,1] range: [{cc_map.min():.4f}, {cc_map.max():.4f}]")
        if not (mlo_map.max() > mlo_map.min()):
            errors.append(f"  {key}: mlo_map max ({mlo_map.max()}) <= min ({mlo_map.min()})")
        if mlo_map.max() > 1.0 or mlo_map.min() < 0.0:
            errors.append(f"  {key}: mlo_map out of [0,1] range: [{mlo_map.min():.4f}, {mlo_map.max():.4f}]")

    if errors:
        print("Stage 3 Verification FAILED:")
        for err in errors:
            print(err)
        raise AssertionError("Stage 3 verification failed.")

    print(f"[Stage 3] Verified {check_count} entries. Pseudo-labels are well-formed.")
    print("Stage 3 Pseudo-labels generated successfully.")
    return True


if __name__ == "__main__":
    verify_stage3()
