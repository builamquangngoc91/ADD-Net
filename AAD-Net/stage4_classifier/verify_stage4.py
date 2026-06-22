"""Stage 4 - Verifier: load classifier, run validation, and check AUC > 0.5."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import PipelineConfig
from stage4_classifier.evaluator import evaluate


def verify_stage4(cfg: PipelineConfig = None):
    if cfg is None:
        cfg = PipelineConfig()

    if not cfg.classifier_ckpt.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {cfg.classifier_ckpt}")

    metrics = evaluate(cfg)

    if metrics is None:
        print("[WARNING] Could not evaluate. Check that Stage 1, 3, and 4 have been run.")
        return False

    print(f"  AUC: {metrics['AUC']:.4f}")
    print(f"  Sensitivity@95%: {metrics['Sensitivity@95%']:.4f}")

    assert metrics["AUC"] > 0.5, f"AUC {metrics['AUC']:.4f} <= 0.5 (model not learning)"

    print("Stage 4 Model Trained and Validated.")
    return True


if __name__ == "__main__":
    verify_stage4()
