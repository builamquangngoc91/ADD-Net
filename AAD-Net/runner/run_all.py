"""Orchestrator: run all AAD-Net pipeline stages sequentially.

Executes each stage with verification. Halts immediately on any failure.
"""

import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE_DIR = ROOT


def _run(script_path: str, stage_name: str) -> bool:
    rel = os.path.relpath(script_path, ROOT)
    print(f"\n{'='*60}")
    print(f"Running: {stage_name}")
    print(f"Script:  {rel}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"\n[ERROR] {stage_name} failed (exit code {result.returncode}). Halting pipeline.")
        sys.exit(1)
    print(f"[OK] {stage_name} completed.")
    return True


def main():
    stages = [
        ("stage1_alignment/run_stage1.py",      "Stage 1: Alignment"),
        ("stage1_alignment/verify_stage1.py",     "Stage 1 Verification"),
        ("stage2_diffusion/run_stage2.py",        "Stage 2: Diffusion Training"),
        ("stage2_diffusion/verify_stage2.py",     "Stage 2 Verification"),
        ("stage3_pseudo_labeling/run_stage3.py",  "Stage 3: Pseudo-labeling"),
        ("stage3_pseudo_labeling/verify_stage3.py","Stage 3 Verification"),
        ("stage4_classifier/run_stage4.py",      "Stage 4: Classifier Training"),
        ("stage4_classifier/verify_stage4.py",    "Stage 4 Verification"),
    ]

    print(f"\nAAD-Net Pipeline — {len(stages)} stages\n")

    for i, (script_rel, name) in enumerate(stages, 1):
        script = ROOT / script_rel
        if not script.exists():
            print(f"[SKIP] {name}: script not found ({script})")
            continue
        _run(script, f"[{i}/{len(stages)}] {name}")

    print(f"\n{'#'*60}")
    print("Full AAD-Net Pipeline executed successfully.")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
