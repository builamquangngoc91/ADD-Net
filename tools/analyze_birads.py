"""Analyze BI-RADS distribution among patients (study_id).

Reads VinDR/breast-level_annotations.csv, groups by study_id, and reports
counts for BI-RADS 1-5 plus per-label normalized vs raw.

Usage:
    python tools/analyze_birads.py
"""
import sys
import os
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import PipelineConfig


def analyze_birads(cfg: PipelineConfig = None):
    if cfg is None:
        cfg = PipelineConfig()

    csv_path = cfg.breast_annot_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"Breast-level CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    expected_cols = {"study_id", "breast_birads"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")

    df["birads_label"] = df["breast_birads"].str.extract(r"(\d)").astype(int)

    per_patient = df.groupby("study_id")["birads_label"].max().reset_index()

    counts = Counter(per_patient["birads_label"])
    total = len(per_patient)

    print(f"Total unique patients: {total}")
    print("BI-RADS distribution (per-patient, label = max BI-RADS across views):")
    print("-" * 60)
    print(f"{'BI-RADS':<10}{'Patients':>12}{'Percent':>12}")
    print("-" * 60)
    for label in sorted(counts.keys()):
        n = counts[label]
        pct = 100.0 * n / max(total, 1)
        print(f"BI-RADS {label:<3}{n:>12d}{pct:>11.2f}%")
    print("-" * 60)
    print(f"{'TOTAL':<10}{total:>12d}{100.0:>11.2f}%")

    counts_image = Counter(df["birads_label"])
    total_img = len(df)
    print()
    print(f"Total labeled images (rows): {total_img}")
    print(f"{'BI-RADS':<10}{'Images':>12}{'Percent':>12}")
    print("-" * 60)
    for label in sorted(counts_image.keys()):
        n = counts_image[label]
        pct = 100.0 * n / max(total_img, 1)
        print(f"BI-RADS {label:<3}{n:>12d}{pct:>11.2f}%")
    print("-" * 60)
    print(f"{'TOTAL':<10}{total_img:>12d}{100.0:>11.2f}%")

    normal = sum(counts[l] for l in (1, 2))
    abnormal = sum(counts[l] for l in (4, 5))
    ignore = counts.get(3, 0)
    print()
    print("Pipeline mapping (cfg.normal_bi_rads / cfg.abnormal_bi_rads / cfg.ignore_bi_rads):")
    print(f"  Normal  (BI-RADS 1-2): {normal} ({100.0 * normal / max(total, 1):.2f}%)")
    print(f"  Abnormal (BI-RADS 4-5): {abnormal} ({100.0 * abnormal / max(total, 1):.2f}%)")
    print(f"  Ignored (BI-RADS 3):  {ignore} ({100.0 * ignore / max(total, 1):.2f}%)")
    used = normal + abnormal
    print(f"  Used for Stage 1+:    {used} ({100.0 * used / max(total, 1):.2f}%)")

    return {
        "per_patient_counts": dict(counts),
        "per_image_counts": dict(counts_image),
        "total_patients": total,
        "total_images": total_img,
    }


def main():
    cfg = PipelineConfig()
    analyze_birads(cfg)


if __name__ == "__main__":
    main()
