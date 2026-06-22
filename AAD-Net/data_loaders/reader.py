"""Mammography data loader with Bi-Rads weak-supervision label mapping.

Bi-Rads 1,2  -> Normal  (label = 0)
Bi-Rads 4,5  -> Abnormal (label = 1)
Bi-Rads 3   -> STRICTLY DISCARDED (neither Normal nor Abnormal)
"""

import re
from pathlib import Path

import pandas as pd
import PIL.Image
import torch
from torch.utils.data import Dataset


def parse_birads(value) -> int:
    if pd.isna(value):
        return None
    match = re.search(r"BI-RADS\s*(\d)", str(value))
    return int(match.group(1)) if match else None


class MammographyLoader(Dataset):
    def __init__(self, cfg, split: str = "training"):
        self.cfg = cfg
        self.split = split
        self.cfg.stage1_aligned_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(cfg.breast_annot_csv)
        if "split" in df.columns:
            df = df[df["split"] == split]

        df["birads_int"] = df["breast_birads"].apply(parse_birads)
        df = df.dropna(subset=["birads_int"])

        records = []
        for study_id, group in df.groupby("study_id"):
            group = group.copy()
            birads_values = group["birads_int"].unique()

            if cfg.ignore_bi_rads in birads_values:
                continue

            has_normal = any(b in cfg.normal_bi_rads for b in birads_values)
            has_abnormal = any(b in cfg.abnormal_bi_rads for b in birads_values)

            if has_normal and has_abnormal:
                continue

            if has_abnormal:
                label = 1
            elif has_normal:
                label = 0
            else:
                continue

            view_map = {}
            for _, row in group.iterrows():
                laterality = row["laterality"]
                view_pos = row["view_position"]
                key = f"{laterality}_{view_pos}"
                view_map[key] = {
                    "image_id": row["image_id"],
                    "series_id": row["series_id"],
                }

            required_views = ["L_CC", "R_CC", "L_MLO", "R_MLO"]
            if not all(k in view_map for k in required_views):
                continue

            records.append({
                "patient_id": study_id,
                "L_CC": view_map["L_CC"]["image_id"],
                "R_CC": view_map["R_CC"]["image_id"],
                "L_MLO": view_map["L_MLO"]["image_id"],
                "R_MLO": view_map["R_MLO"]["image_id"],
                "label": label,
            })

        self.records = records
        self._build_image_path_cache()

    def _build_image_path_cache(self):
        metadata_df = pd.read_csv(self.cfg.metadata_csv)
        uid_to_path = {}
        for _, row in metadata_df.iterrows():
            uid = row["SOP Instance UID"]
            series_uid = row["Series Instance UID"]
            uid_to_path[uid] = series_uid
        self._uid_to_series = uid_to_path

    def _find_image_file(self, image_id: str) -> Path:
        for ext in [".dcm", ".png", ".jpg", ".tif"]:
            p = self.cfg.raw_data_dir / "images" / image_id[:2] / f"{image_id}{ext}"
            if p.exists():
                return p
            p = self.cfg.raw_data_dir / image_id[:2] / f"{image_id}{ext}"
            if p.exists():
                return p
        return None

    def _load_image(self, image_id: str) -> torch.Tensor:
        path = self._find_image_file(image_id)
        if path is None:
            h, w = self.cfg.target_size[1], self.cfg.target_size[0]
            return torch.zeros(1, h, w)

        try:
            img = PIL.Image.open(path)
            if img.mode != "L":
                img = img.convert("L")
            img = img.resize(self.cfg.target_size, PIL.Image.BILINEAR)
            import numpy as np
            arr = torch.from_numpy(np.asarray(img)).float() / 255.0
            return arr.unsqueeze(0)
        except Exception:
            h, w = self.cfg.target_size[1], self.cfg.target_size[0]
            return torch.zeros(1, h, w)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        return {
            "patient_id": rec["patient_id"],
            "L_CC": self._load_image(rec["L_CC"]),
            "R_CC": self._load_image(rec["R_CC"]),
            "L_MLO": self._load_image(rec["L_MLO"]),
            "R_MLO": self._load_image(rec["R_MLO"]),
            "label": torch.tensor(rec["label"], dtype=torch.long),
        }
