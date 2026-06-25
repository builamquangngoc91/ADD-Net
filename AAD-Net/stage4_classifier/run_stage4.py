"""Stage 4 - Runner: Teacher-Student training of the Adaptive Multi-View Classifier.

Uses Stage 3 soft pseudo-labels, EMA-updated teacher, and a combined loss:
  total_loss = 0.5 * loss_supcon + 0.5 * loss_ce + 0.05 * loss_consistency
with gradient accumulation.
"""

import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import PipelineConfig
from stage4_classifier.patch_supcon import SupConLoss
from stage4_classifier.fusion_network import AdaptiveMultiViewClassifier


def _update_ema(teacher: nn.Module, student: nn.Module, decay: float = 0.999):
    with torch.no_grad():
        for tp, sp in zip(teacher.parameters(), student.parameters()):
            tp.data.mul_(decay).add_(sp.data, alpha=1 - decay)


def _clip_gradients(model: nn.Module, max_norm: float = 1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def _flatten_pseudo_labels(cc_map: torch.Tensor, mlo_map: torch.Tensor, patch_size: int, stride: int) -> torch.Tensor:
    if cc_map.dim() == 2:
        cc_map = cc_map.unsqueeze(0).unsqueeze(0)
    elif cc_map.dim() == 3:
        cc_map = cc_map.unsqueeze(0)
    if mlo_map.dim() == 2:
        mlo_map = mlo_map.unsqueeze(0).unsqueeze(0)
    elif mlo_map.dim() == 3:
        mlo_map = mlo_map.unsqueeze(0)
    cc_patches = F.unfold(cc_map, kernel_size=patch_size, stride=stride)
    mlo_patches = F.unfold(mlo_map, kernel_size=patch_size, stride=stride)
    B, _, num_patches = cc_patches.shape
    cc_labels = cc_patches.mean(dim=1).reshape(B, num_patches)
    mlo_labels = mlo_patches.mean(dim=1).reshape(B, num_patches)
    return torch.cat([cc_labels, mlo_labels], dim=1)


def run_stage4(cfg: PipelineConfig = None):
    if cfg is None:
        cfg = PipelineConfig()

    manifest_path = cfg.processed_dir / "manifest_stage1.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Stage 1 manifest not found: {manifest_path}")

    pseudo_path = cfg.stage3_pseudolabel_dir / "pseudo_labels.npy"
    if not pseudo_path.exists():
        raise FileNotFoundError(f"Stage 3 pseudo-labels not found: {pseudo_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    pseudo_labels = np.load(pseudo_path, allow_pickle=True).item()

    student = AdaptiveMultiViewClassifier(dropout=0.3).to(cfg.device)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(student.parameters(), lr=cfg.lr)
    supcon_loss_fn = SupConLoss(temperature=cfg.temp_supercon)
    ce_loss_fn = nn.CrossEntropyLoss()

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    files = manifest["aligned_files"]
    patch_h = (512 - cfg.patch_size) // cfg.stride + 1
    patch_w = (1024 - cfg.patch_size) // cfg.stride + 1

    print(f"[Stage 4] Training on {len(files)} patients, {cfg.cls_epochs} epochs")

    global_pbar = tqdm(
        total=cfg.cls_epochs,
        desc="Stage 4 (epochs)",
        unit="epoch",
        ncols=80,
    )

    for epoch in range(cfg.cls_epochs):
        student.train()
        epoch_total = 0.0
        epoch_supcon = 0.0
        epoch_ce = 0.0
        epoch_cons = 0.0
        n_batches = 0

        optimizer.zero_grad()

        epoch_pbar = tqdm(
            files,
            desc=f"  Epoch {epoch + 1}/{cfg.cls_epochs}",
            unit="patient",
            ncols=80,
            leave=False,
        )

        for fpath in epoch_pbar:
            batch = torch.load(fpath, map_location=cfg.device, weights_only=False)
            patient_id = batch["patient_id"]
            if isinstance(patient_id, list):
                patient_id = "_".join(patient_id)
            label = batch["label"].to(cfg.device)
            print(f"[DEBUG Stage4] patient={patient_id} label.shape={tuple(label.shape)} label.dtype={label.dtype} label={label.item() if label.numel()==1 else label}")

            pl_entry = pseudo_labels.get(patient_id, None)
            if pl_entry is None:
                continue

            x_cc = batch["L_CC"].to(cfg.device)
            x_mlo = batch["L_MLO"].to(cfg.device)

            cc_map = torch.from_numpy(pl_entry["cc_map"]).float().to(cfg.device)
            mlo_map = torch.from_numpy(pl_entry["mlo_map"]).float().to(cfg.device)

            if cc_map.dim() == 2:
                cc_map = cc_map.unsqueeze(0)
            if mlo_map.dim() == 2:
                mlo_map = mlo_map.unsqueeze(0)

            if cc_map.shape[-2:] != (512, 1024):
                cc_map = F.interpolate(cc_map.unsqueeze(1), size=(512, 1024), mode="bilinear", align_corners=False).squeeze(1)
            if mlo_map.shape[-2:] != (512, 1024):
                mlo_map = F.interpolate(mlo_map.unsqueeze(1), size=(512, 1024), mode="bilinear", align_corners=False).squeeze(1)

            patch_labels = _flatten_pseudo_labels(cc_map, mlo_map, cfg.patch_size, cfg.stride)

            all_patches = student.extract_all_patches(x_cc, x_mlo)

            loss_supcon = supcon_loss_fn(all_patches, patch_labels)

            logits_student = student(x_cc, x_mlo, cc_map, mlo_map)
            logits_teacher = teacher(x_cc, x_mlo, cc_map, mlo_map)

            print(f"[DEBUG Stage4] logits_student.shape={tuple(logits_student.shape)} label.shape={tuple(label.shape)} label.dtype={label.dtype}")
            ce_target = label.long().view(-1)
            loss_ce = ce_loss_fn(logits_student, ce_target)

            log_probs_student = F.log_softmax(logits_student, dim=-1)
            probs_teacher = F.softmax(logits_teacher, dim=-1).clamp(min=1e-8)
            loss_consistency = F.kl_div(log_probs_student, probs_teacher, reduction="batchmean")

            total_loss = (
                cfg.supercon_weight * loss_supcon
                + cfg.ce_weight * loss_ce
                + cfg.consistency_weight * loss_consistency
            )
            scaled_loss = total_loss / cfg.accumulation_steps
            scaled_loss.backward()

            epoch_total += total_loss.item()
            epoch_supcon += loss_supcon.item()
            epoch_ce += loss_ce.item()
            epoch_cons += loss_consistency.item()
            n_batches += 1

            epoch_pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

            if n_batches % cfg.accumulation_steps == 0:
                _clip_gradients(student, cfg.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                _update_ema(teacher, student, cfg.ema_decay)

        epoch_pbar.close()

        if n_batches % cfg.accumulation_steps != 0:
            _clip_gradients(student, cfg.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        avg_total = epoch_total / max(n_batches, 1)
        avg_supcon = epoch_supcon / max(n_batches, 1)
        avg_ce = epoch_ce / max(n_batches, 1)
        avg_cons = epoch_cons / max(n_batches, 1)

        global_pbar.set_postfix({
            "loss": f"{avg_total:.4f}",
            "supcon": f"{avg_supcon:.4f}",
            "ce": f"{avg_ce:.4f}",
            "cons": f"{avg_cons:.4f}",
        })
        global_pbar.update(1)

    global_pbar.close()

    ckpt_path = cfg.classifier_ckpt
    torch.save(student.state_dict(), ckpt_path)
    print(f"[Stage 4] Saved: {ckpt_path}")
    return ckpt_path


if __name__ == "__main__":
    run_stage4()
