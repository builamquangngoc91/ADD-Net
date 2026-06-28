"""
PyTorch Dataset for multi-scale patch MIL.

FIXED (plan issue #4): Patch-level augmentation now applies a single random
seed per bag so that all patches within a bag receive the same RandomErasing
mask and GaussianBlur kernel — preserving the relative spatial signal of
small calcifications.

Also stores lesion_type for stratified analysis (plan issue #9).
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
from typing import Dict, List

from src.data.preprocessing import preprocess_image
from src.data.patch_extractor import MultiScalePatchExtractor


class MammoMultiScaleDataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'train', config=None):
        self.root_dir = root_dir
        self.split = split
        self.config = config
        self.augment = getattr(config.data, 'augment', False) if config else False

        self.image_paths = []
        self._load_metadata()

        self.patch_extractor = MultiScalePatchExtractor(
            patch_sizes=config.data.patch_sizes,
            stride=config.data.patch_stride,
            image_size=config.data.image_size,
        )

        if self.augment and split == 'train':
            self.bag_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=15, translate=(0.1, 0.1), fill=0
                ),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.2, contrast=0.2)
                ], p=0.3),
            ])
            # Patch-level transform — intentionally lightweight; probability
            # is controlled by augment_patch_prob in config, applied per-bag.
            self.patch_transform_base = transforms.Compose([
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
                ], p=0.2),
            ])
            # RandomErasing applied bag-wide (single seed per bag).
            self.random_erasing = transforms.RandomErasing(
                p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0
            )
        else:
            self.bag_transform = None
            self.patch_transform_base = None
            self.random_erasing = None

    def _load_metadata(self):
        """Load image paths, labels, and lesion types from CSV or folder structure."""
        import pandas as pd

        # Auto-build split CSV from raw source if missing.
        csv_path = os.path.join(self.root_dir, f'{self.split}.csv')
        if not os.path.exists(csv_path):
            dataset_name = (self.config.data.dataset or '').lower() if self.config else ''
            if dataset_name == 'cmmd':
                csv_path = self._build_cmmd_split_csv()
            elif dataset_name == 'vindr':
                csv_path = self._build_vindr_split_csv()
            else:
                # Nothing to load.
                return

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            self.image_paths.append({
                'path': row['image_path'],
                'label': int(row['label']),
                'lesion_type': row.get('lesion_type', 'unknown'),
            })

    def _split_file(self) -> str:
        return os.path.join(self.root_dir, f'{self.split}.csv')

    def _build_cmmd_split_csv(self) -> str:
        """Walk the CMMD folder structure and produce a train/val CSV."""
        import pandas as pd

        raw_root = getattr(self.config.data, 'raw_dir', './CMD')
        out_root = self.root_dir
        os.makedirs(out_root, exist_ok=True)

        xlsx_path = os.path.join(raw_root, 'CMMD_clinicaldata_revision.xlsx')
        if not os.path.exists(xlsx_path):
            raise FileNotFoundError(
                f"CMMD clinical file not found at {xlsx_path}. "
                f"Set data.raw_dir to the folder containing CMMD_clinicaldata_revision.xlsx."
            )

        clinical = pd.read_excel(xlsx_path)
        if 'classification' not in clinical.columns or 'ID1' not in clinical.columns:
            raise ValueError("CMMD clinical xlsx is missing required columns (ID1, classification).")
        label_map = {'Benign': 0, 'Malignant': 1}
        clinical = clinical[['ID1', 'LeftRight', 'classification', 'subtype']].copy()
        clinical['label'] = clinical['classification'].map(label_map)
        if clinical['label'].isna().any():
            raise ValueError(f"Unknown CMMD classifications: {clinical['classification'].unique()}")

        records = []
        for case_id, group in clinical.groupby('ID1'):
            case_dir = os.path.join(raw_root, case_id)
            if not os.path.isdir(case_dir):
                continue
            for _, row in group.iterrows():
                side = row['LeftRight']
                dcm_files = self._find_cmmd_dicoms(case_dir, side)
                if not dcm_files:
                    continue
                rel_path = os.path.abspath(dcm_files[0])
                records.append({
                    'image_path': rel_path,
                    'label': int(row['label']),
                    'lesion_type': str(row.get('subtype', 'unknown')) if not pd.isna(row.get('subtype')) else 'unknown',
                    'case_id': case_id,
                })

        if not records:
            raise RuntimeError("No CMMD samples found. Check data.raw_dir and folder layout.")

        df = pd.DataFrame(records)
        return self._split_and_write(df, split_col=None)

    def _find_cmmd_dicoms(self, case_dir: str, side: str) -> list:
        """Recursively find the first DICOM for a given case / side."""
        # CMMD layout: <case>/<study_uid>/<series_uid>/*.dcm
        candidates = []
        for root, _, files in os.walk(case_dir):
            for f in files:
                if f.lower().endswith('.dcm'):
                    candidates.append(os.path.join(root, f))
        if not candidates:
            return []
        if side:
            side = side.upper()
            preferred = [p for p in candidates if side.lower() in os.path.basename(os.path.dirname(os.path.dirname(p))).lower()]
            if preferred:
                return preferred[:1]
        return candidates[:1]

    def _build_vindr_split_csv(self) -> str:
        """Build a VinDR train/val CSV from the breast-level annotations."""
        import pandas as pd

        raw_root = getattr(self.config.data, 'raw_dir', './VinDR')
        out_root = self.root_dir
        os.makedirs(out_root, exist_ok=True)

        csv_src = os.path.join(raw_root, 'breast-level_annotations.csv')
        if not os.path.exists(csv_src):
            raise FileNotFoundError(f"VinDR breast-level CSV not found at {csv_src}.")

        df = pd.read_csv(csv_src)

        # Map BI-RADS categories to binary labels.
        # BI-RADS 1, 2 -> 0 (negative); 3, 4, 5 -> 1 (suspicious / positive).
        def to_label(b):
            if pd.isna(b):
                return None
            s = str(b).upper().replace('BI-RADS', '').strip()
            try:
                n = int(s.split()[0])
            except Exception:
                return None
            return 0 if n <= 2 else 1

        df['label'] = df['breast_birads'].apply(to_label)
        df = df[df['label'].notna()].copy()
        df['label'] = df['label'].astype(int)

        images_root = os.path.join(os.path.abspath(raw_root), 'images')
        df['image_path'] = df.apply(
            lambda r: os.path.abspath(os.path.join(images_root, r['study_id'], f"{r['image_id']}.dicom")),
            axis=1,
        )
        df['lesion_type'] = df['breast_density'].fillna('unknown')

        keep = ['image_path', 'label', 'lesion_type', 'split']
        df = df[keep].rename(columns={'split': '_source_split'})

        return self._split_and_write(df, split_col='_source_split')

    def _split_and_write(self, df, split_col: str | None) -> str:
        """Split df into train/val deterministically and write the current split's CSV."""
        import pandas as pd
        from sklearn.model_selection import train_test_split

        out_path = self._split_file()
        train_path = os.path.join(self.root_dir, 'train.csv')
        val_path = os.path.join(self.root_dir, 'val.csv')

        if split_col and split_col in df.columns:
            train_df = df[df[split_col] == 'training'].drop(columns=[split_col])
            test_df = df[df[split_col] == 'test'].drop(columns=[split_col])
            if self.split == 'train':
                self._write_csv(train_df, train_path)
                self._write_csv(test_df, val_path)
                return train_path
            else:
                return val_path
        else:
            train_df, val_df = train_test_split(
                df, test_size=0.2, stratify=df['label'], random_state=42
            )
            self._write_csv(train_df, train_path)
            self._write_csv(val_df, val_path)
            return out_path

    @staticmethod
    def _write_csv(df, path: str) -> None:
        import pandas as pd
        cols = ['image_path', 'label', 'lesion_type']
        for c in cols:
            if c not in df.columns:
                df[c] = 'unknown' if c == 'lesion_type' else 0
        df[cols].to_csv(path, index=False)

    def get_all_labels(self) -> np.ndarray:
        """Return all labels for WeightedRandomSampler."""
        return np.array([item['label'] for item in self.image_paths])

    def get_lesion_types(self) -> List[str]:
        """Return lesion types for stratified analysis."""
        return [item.get('lesion_type', 'unknown') for item in self.image_paths]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        item = self.image_paths[idx]
        img_path = item['path']
        label = item['label']
        lesion_type = item.get('lesion_type', 'unknown')

        image = preprocess_image(img_path, self.config.preprocessing, self.config.data.image_size)
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)

        if self.augment and self.split == 'train' and self.bag_transform:
            image_tensor = self.bag_transform(image_tensor)

        patches_dict = self.patch_extractor.extract_patches_torch(image_tensor)

        # FIXED (plan #4): Apply patch augmentation with a shared random state
        # per bag. This ensures all patches in the bag see the same erasing mask,
        # preserving spatial coherence for small calcification signals.
        if self.augment and self.split == 'train' and self.patch_transform_base:
            if random.random() < self.config.data.augment_patch_prob:
                bag_seed = random.randint(0, 2**31 - 1)
                for key in patches_dict:
                    N = patches_dict[key].size(0)
                    augmented = []
                    for i in range(N):
                        torch.manual_seed(bag_seed + i)
                        p = self.patch_transform_base(patches_dict[key][i])
                        torch.manual_seed(bag_seed + i + 100000)
                        p = self.random_erasing(p)
                        augmented.append(p)
                    patches_dict[key] = torch.stack(augmented, dim=0)
                    torch.seed()

        return {
            'patches': patches_dict,
            'label': torch.tensor(label, dtype=torch.long),
            'image_path': img_path,
            'lesion_type': lesion_type,
        }
