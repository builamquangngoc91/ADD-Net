"""Stage 4 - Evaluator: MedIA-standard metrics.

Computes AUC and Sensitivity at Specificity thresholds on the validation set.
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import PipelineConfig
from stage4_classifier.fusion_network import AdaptiveMultiViewClassifier


def compute_AUC(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_pred)


def compute_Sensitivity_At_Specificity(y_pred: np.ndarray, y_true: np.ndarray, target_spec: float = 0.95) -> float:
    thresholds = np.sort(y_pred)
    best_sens = 0.0
    for thresh in thresholds:
        y_binary = (y_pred >= thresh).astype(int)
        tp = ((y_binary == 1) & (y_true == 1)).sum()
        fn = ((y_binary == 0) & (y_true == 1)).sum()
        fp = ((y_binary == 1) & (y_true == 0)).sum()
        tn = ((y_binary == 0) & (y_true == 0)).sum()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        if abs(spec - target_spec) < 0.02:
            best_sens = max(best_sens, sens)
    return best_sens


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray):
    auc = compute_AUC(y_pred, y_true)
    sens_95 = compute_Sensitivity_At_Specificity(y_pred, y_true, target_spec=0.95)
    return {"AUC": auc, "Sensitivity@95%": sens_95}


def evaluate(cfg: PipelineConfig = None):
    if cfg is None:
        cfg = PipelineConfig()

    manifest_path = cfg.processed_dir / "manifest_stage1.json"
    pseudo_path = cfg.stage3_pseudolabel_dir / "pseudo_labels.npy"

    if not manifest_path.exists() or not pseudo_path.exists():
        print("[WARNING] Stage 1/3 outputs not found. Skipping evaluation.")
        return None

    with open(manifest_path) as f:
        manifest = json.load(f)

    pseudo_labels = np.load(pseudo_path, allow_pickle=True).item()

    val_files = [f for f in manifest["aligned_files"]]
    if not val_files:
        return None

    classifier = AdaptiveMultiViewClassifier(dropout=0.3).to(cfg.device)
    if cfg.classifier_ckpt.exists():
        state = torch.load(cfg.classifier_ckpt, map_location=cfg.device, weights_only=True)
        classifier.load_state_dict(state)
    classifier.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for fpath in val_files:
            patient_id = torch.load(fpath, map_location="cpu", weights_only=False)["patient_id"]
            batch = torch.load(fpath, map_location=cfg.device, weights_only=False)
            label = batch["label"].item()

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

            logits = classifier(x_cc, x_mlo, cc_map, mlo_map)
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()

            all_preds.append(probs[0])
            all_labels.append(label)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = compute_metrics(all_preds, all_labels)
    return metrics
