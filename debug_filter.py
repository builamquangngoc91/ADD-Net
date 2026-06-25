"""Analyze filter losses in MammographyLoader."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

CSV = "/mnt/c/Users/Tyler/Desktop/Ngoc/ADD-NET/VinDR/breast-level_annotations.csv"
df = pd.read_csv(CSV)
df = df[df["split"] == "training"]
df["birads"] = df["breast_birads"].str.extract(r"BI-RADS\s*(\d)").astype(float)

normal, abnormal = (1, 2), (4, 5)
abn_ids, nrm_ids, missing_view_ids = set(), set(), set()

for study_id, group in df.groupby("study_id"):
    vals = group["birads"].dropna().astype(int).unique()
    has_n = any(b in normal for b in vals)
    has_a = any(b in abnormal for b in vals)
    if not has_n and not has_a:
        continue
    if has_a:
        abn_ids.add(study_id)
    elif has_n:
        nrm_ids.add(study_id)

    views = set(group.apply(lambda r: f"{r['laterality']}_{r['view_position']}", axis=1))
    if not {"L_CC", "R_CC", "L_MLO", "R_MLO"} <= views:
        missing_view_ids.add(study_id)

print(f"Training patients total: {len(abn_ids) + len(nrm_ids)}")
print(f"  Patients with normal views (label=0):  {len(nrm_ids)}")
print(f"  Patients with abnormal views (label=1): {len(abn_ids)}")
print(f"  Patients dropped (missing 4 views): {len(missing_view_ids)}")
print(f"  Normal:Abnormal ratio: {len(nrm_ids)/max(len(abn_ids),1):.2f}:1")