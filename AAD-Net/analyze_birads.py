"""Analyze BI-RADS distribution per patient in the VinDR dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.config import PipelineConfig
import pandas as pd

cfg = PipelineConfig()
ANNOT_CSV = cfg.breast_annot_csv


def analyze_birads():
    df = pd.read_csv(ANNOT_CSV)
    print(f"Total breast images: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Unique patients (study_id)
    patients = df["study_id"].unique()
    print(f"Unique patients (study_id): {len(patients)}")

    # Extract numeric BI-RADS from string (e.g. "BI-RADS 2" -> 2)
    df["birads_num"] = df["breast_birads"].str.extract(r"BI-RADS\s*(\d)")
    df["birads_num"] = pd.to_numeric(df["birads_num"], errors="coerce")

    # Drop rows with no valid BI-RADS
    df_valid = df.dropna(subset=["birads_num"])
    print(f"\nImages with valid BI-RADS: {len(df_valid)}")
    print(f"BI-RADS value counts (images):\n{df_valid['birads_num'].value_counts().sort_index()}")

    # Per-patient: take the most severe BI-RADS if patient has multiple views/series
    # (a patient could have L and R breast, each with a BI-RADS — use max)
    patient_birads = (
        df_valid.groupby("study_id")["birads_num"]
        .max()
        .astype(int)
        .value_counts()
        .sort_index()
    )
    print(f"\nBI-RADS distribution per patient (most severe per patient):")
    for birads in sorted(patient_birads.index):
        count = patient_birads[birads]
        pct = count / len(patients) * 100
        print(f"  BI-RADS {birads}: {count:>5} patients ({pct:.2f}%)")

    # Per-patient: just take the first BI-RADS (L-CC view typically)
    patient_birads_first = (
        df_valid.drop_duplicates(subset=["study_id"])[["study_id", "birads_num"]]
        .set_index("study_id")["birads_num"]
        .astype(int)
        .value_counts()
        .sort_index()
    )
    print(f"\nBI-RADS distribution per patient (first image per patient):")
    for birads in sorted(patient_birads_first.index):
        count = patient_birads_first[birads]
        pct = count / len(patients) * 100
        print(f"  BI-RADS {birads}: {count:>5} patients ({pct:.2f}%)")

    # Config mapping summary
    normal_mask = patient_birads.index.isin([1, 2])
    abnormal_mask = patient_birads.index.isin([4, 5])
    ignore_mask = patient_birads.index == 3
    n_normal = patient_birads[normal_mask].sum()
    n_abnormal = patient_birads[abnormal_mask].sum()
    n_ignored = patient_birads[ignore_mask].sum()
    print(f"\nConfig mapping summary:")
    print(f"  Normal   (BI-RADS 1,2): {n_normal:>5} ({n_normal/len(patients)*100:.2f}%)")
    print(f"  Abnormal (BI-RADS 4,5): {n_abnormal:>5} ({n_abnormal/len(patients)*100:.2f}%)")
    print(f"  Ignored  (BI-RADS 3) : {n_ignored:>5} ({n_ignored/len(patients)*100:.2f}%)")
    print(f"  Imbalance ratio (normal:abnormal): {n_normal/max(n_abnormal,1):.2f}:1")


if __name__ == "__main__":
    analyze_birads()
