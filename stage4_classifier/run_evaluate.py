"""Stage 4 - Evaluate the trained classifier on the test set."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from config.config import PipelineConfig
from stage4_classifier.evaluator import evaluate


def main():
    cfg = PipelineConfig()
    metrics = evaluate(cfg, show_progress=True)
    if metrics is None:
        print("[Stage 4] Evaluation skipped (missing Stage 1/3 outputs).")
        return
    print("[Stage 4] Test Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
