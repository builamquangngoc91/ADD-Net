"""Stage 1 - Orchestrator: parallel radiometric normalization + deformable registration.

Processes the full VinDr mammography dataset through Stage 1 using
ProcessPoolExecutor for embarrassingly-parallel patient processing.
"""

import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import PipelineConfig
from data_loaders.reader import MammographyLoader


# ---------------------------------------------------------------------------
# Per-patient worker (runs in a subprocess)
# ---------------------------------------------------------------------------

def _process_single_patient(record: dict, cfg_kwargs: dict) -> str:
    """Load, normalize, register, and save one patient. Runs in a worker."""
    import torch
    from stage1_alignment.transforms import run_radiometric_normalization
    from stage1_alignment.registration import SciPyAffineWrapper, run_deformable_registration

    target_size = tuple(cfg_kwargs["target_size"])
    output_dir = Path(cfg_kwargs["stage1_aligned_dir"])

    batch = {
        "patient_id": record["patient_id"],
        "L_CC": _load_image_from_path(record["L_CC_path"], target_size),
        "R_CC": _load_image_from_path(record["R_CC_path"], target_size),
        "L_MLO": _load_image_from_path(record["L_MLO_path"], target_size),
        "R_MLO": _load_image_from_path(record["R_MLO_path"], target_size),
        "label": torch.tensor(record["label"], dtype=torch.long),
    }

    norm_batch = run_radiometric_normalization(batch, target_size)
    aligned_batch = run_deformable_registration(norm_batch, None)

    keys = ["L_CC", "R_CC", "L_MLO", "R_MLO"]
    aligned_batch = {k: aligned_batch[k].squeeze(1) for k in keys}
    aligned_batch["label"] = norm_batch["label"]
    aligned_batch["patient_id"] = record["patient_id"]

    save_path = output_dir / f"{record['patient_id']}.pt"
    torch.save(aligned_batch, save_path)
    return str(save_path)


def _load_image_from_path(path: str, target_size: tuple) -> torch.Tensor:
    """Load a single image file into a [1, H, W] tensor. Mirrors reader._load_image."""
    import numpy as np
    import PIL.Image
    import torch

    p = Path(path) if path else None
    h, w = target_size[1], target_size[0]

    if p is None or not p.exists():
        return torch.zeros(1, h, w)

    try:
        if p.suffix.lower() in (".dcm", ".dicom"):
            import pydicom
            ds = pydicom.dcmread(str(p))
            arr = ds.pixel_array
            if arr.ndim == 3:
                arr = arr[0]
            arr = arr.astype(np.float32)
            lo, hi = float(arr.min()), float(arr.max())
            arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
            img = PIL.Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        else:
            img = PIL.Image.open(p)
            if img.mode != "L":
                img = img.convert("L")

        img = img.resize(target_size, PIL.Image.BILINEAR)
        arr = torch.from_numpy(np.asarray(img).copy()).float() / 255.0
        return arr.unsqueeze(0)
    except Exception:
        return torch.zeros(1, h, w)


# ---------------------------------------------------------------------------
# Record collection (runs in main process — reads CSV once)
# ---------------------------------------------------------------------------

def _collect_records(cfg: PipelineConfig) -> list[dict]:
    """Build serializable record list from MammographyLoader, pre-resolving paths."""
    loader = MammographyLoader(cfg, split="training")
    records = []
    for rec in loader.records:
        paths = {
            k: _resolve_path(cfg.raw_data_dir, loader._uid_to_study, rec[k])
            for k in ["L_CC", "R_CC", "L_MLO", "R_MLO"]
        }
        records.append({
            "patient_id": rec["patient_id"],
            "L_CC_path": paths["L_CC"],
            "R_CC_path": paths["R_CC"],
            "L_MLO_path": paths["L_MLO"],
            "R_MLO_path": paths["R_MLO"],
            "label": rec["label"],
        })
    return records


def _resolve_path(raw_data_dir: Path, uid_to_study: dict, image_id: str) -> str | None:
    """Find the on-disk path for a given image_id."""
    study_uid = uid_to_study.get(image_id)
    candidates = []
    if study_uid:
        candidates.append(raw_data_dir / "images" / study_uid)
    candidates.append(raw_data_dir / "images" / image_id[:2])
    candidates.append(raw_data_dir / image_id[:2])
    for folder in candidates:
        if not folder.exists():
            continue
        for ext in [".dicom", ".dcm", ".png", ".jpg", ".tif"]:
            p = folder / f"{image_id}{ext}"
            if p.exists():
                return str(p)
        for p in folder.glob(f"{image_id}.*"):
            if p.is_file():
                return str(p)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_stage1(cfg: PipelineConfig = None, num_workers: int | None = None):
    if cfg is None:
        cfg = PipelineConfig()

    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 16)

    cfg.stage1_aligned_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Stage 1] Loading {len(MammographyLoader(cfg, split='training'))} patients...")
    records = _collect_records(cfg)

    cfg_kwargs = {
        "target_size": cfg.target_size,
        "stage1_aligned_dir": str(cfg.stage1_aligned_dir),
    }

    saved_files = []
    failed = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_process_single_patient, rec, cfg_kwargs): rec["patient_id"]
            for rec in records
        }

        from tqdm import tqdm
        for future in tqdm(as_completed(futures), total=len(records), desc="Stage 1", unit="patient", ncols=80):
            patient_id = futures[future]
            try:
                path = future.result()
                saved_files.append(path)
            except Exception:
                failed.append((patient_id, traceback.format_exc()))

    if failed:
        print(f"\n[Stage 1] {len(failed)} patients failed:")
        for pid, tb in failed:
            print(f"  [{pid}] {tb.strip().splitlines()[-1]}")

    manifest = {
        "aligned_files": saved_files,
        "base_path": str(cfg.stage1_aligned_dir),
        "total_count": len(saved_files),
    }
    manifest_path = cfg.processed_dir / "manifest_stage1.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Stage 1] Done. {len(saved_files)}/{len(records)} patients saved to {cfg.stage1_aligned_dir}")
    print(f"[Stage 1] Manifest: {manifest_path}")
    return manifest


if __name__ == "__main__":
    run_stage1()
