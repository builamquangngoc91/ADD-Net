"""Stage 4 - Evaluator: MedIA-standard + classification metrics.

Computes AUC, Sensitivity at Specificity, Accuracy, Precision, Recall, and F1.
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
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


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    auc = compute_AUC(y_pred, y_true)
    sens_95 = compute_Sensitivity_At_Specificity(y_pred, y_true, target_spec=0.95)

    y_binary = (y_pred >= threshold).astype(int)
    tp = ((y_binary == 1) & (y_true == 1)).sum()
    fn = ((y_binary == 0) & (y_true == 1)).sum()
    fp = ((y_binary == 1) & (y_true == 0)).sum()
    tn = ((y_binary == 0) & (y_true == 0)).sum()

    acc = (tp + tn) / max(tp + fn + fp + tn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2.0 * prec * rec / max(prec + rec, 1e-8)

    return {
        "AUC": auc,
        "Sensitivity@95%": sens_95,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
    }


def evaluate(cfg: PipelineConfig = None, show_progress: bool = False):
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

    iterator = tqdm(val_files, desc="Stage 4 Eval", unit="patient", ncols=80) if show_progress else val_files

    with torch.no_grad():
        for fpath in iterator:
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

    n_pos = int((all_labels == 1).sum())
    n_neg = int((all_labels == 0).sum())
    print(f"[DEBUG Stage4 Eval] label distribution: pos={n_pos}, neg={n_neg}, total={len(all_labels)}")
    print(f"[DEBUG Stage4 Eval] pred prob stats: min={all_preds.min():.4f}, max={all_preds.max():.4f}, mean={all_preds.mean():.4f}, frac>=0.5={(all_preds >= 0.5).mean():.4f}")
    pos_probs = all_preds[all_labels == 1] if n_pos > 0 else all_preds[:0]
    neg_probs = all_preds[all_labels == 0] if n_neg > 0 else all_preds[:0]
    if n_pos > 0:
        print(f"[DEBUG Stage4 Eval] pred probs for positives: min={pos_probs.min():.4f}, max={pos_probs.max():.4f}, mean={pos_probs.mean():.4f}")
    if n_neg > 0:
        print(f"[DEBUG Stage4 Eval] pred probs for negatives: min={neg_probs.min():.4f}, max={neg_probs.max():.4f}, mean={neg_probs.mean():.4f}")

    metrics = compute_metrics(all_preds, all_labels)
    return metrics
