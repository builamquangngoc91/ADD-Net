"""Global configuration for the AAD-Net pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT.parent / "VinDR"
PROCESSED_DIR = PROJECT_ROOT / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

TARGET_SIZE = (1024, 512)
PATCH_SIZE = 128
STRIDE = 64

NORMAL_BI_RADS = (1, 2)
ABNORMAL_BI_RADS = (4, 5)
IGNORE_BI_RADS = 3

DIFF_EPOCHS = 100
CLS_EPOCHS = 50
LR = 1e-4
ACCUMULATION_STEPS = 4

DEVICE = "cuda"

METADATA_CSV = RAW_DATA_DIR / "metadata.csv"
BREAST_ANNOT_CSV = RAW_DATA_DIR / "breast-level_annotations.csv"
FINDING_ANNOT_CSV = RAW_DATA_DIR / "finding_annotations.csv"

STAGE1_ALIGNED_DIR = PROCESSED_DIR / "stage1_aligned"
STAGE3_PSEUDOLABEL_DIR = PROCESSED_DIR / "stage3_pseudolabels"

VOXELMORPH_CKPT = PROJECT_ROOT / "weights" / "vxm_breast.h5"
AUTOENCODER_CKPT = PROJECT_ROOT / "weights" / "autoencoder.pth"
RADIMAGE_NET_WEIGHTS = PROJECT_ROOT / "weights" / "radimagenet_vgg16.pth"

DIFFUSION_CKPT = CHECKPOINT_DIR / "diffusion_model.pth"
CLASSIFIER_CKPT = CHECKPOINT_DIR / "classifier_student.pth"

DIFFUSION_TIMESTEPS = 1000
DDIM_STEPS = 20
NUM_GMM_COMPONENTS = 2

SUPERCON_WEIGHT = 0.5
CE_WEIGHT = 0.5
CONSISTENCY_WEIGHT = 0.05
EMA_DECAY = 0.999
EMA_UPDATE = 0.001

TEMP_SUPERCON = 0.07
CLIP_GRAD_NORM = 1.0

GMM_SIGMOID_TEMP = 0.1
LPIPS_SHALLOW_WEIGHT = 0.7
LPIPS_DEEP_WEIGHT = 0.3


@dataclass
class PipelineConfig:
    project_root: Path = PROJECT_ROOT
    raw_data_dir: Path = RAW_DATA_DIR
    processed_dir: Path = PROCESSED_DIR
    checkpoint_dir: Path = CHECKPOINT_DIR

    target_size: tuple = field(default_factory=lambda: TARGET_SIZE)
    patch_size: int = PATCH_SIZE
    stride: int = STRIDE

    normal_bi_rads: tuple = NORMAL_BI_RADS
    abnormal_bi_rads: tuple = ABNORMAL_BI_RADS
    ignore_bi_rads: int = IGNORE_BI_RADS

    diff_epochs: int = DIFF_EPOCHS
    cls_epochs: int = CLS_EPOCHS
    lr: float = LR
    accumulation_steps: int = ACCUMULATION_STEPS

    device: str = DEVICE

    metadata_csv: Path = METADATA_CSV
    breast_annot_csv: Path = BREAST_ANNOT_CSV
    finding_annot_csv: Path = FINDING_ANNOT_CSV

    stage1_aligned_dir: Path = STAGE1_ALIGNED_DIR
    stage3_pseudolabel_dir: Path = STAGE3_PSEUDOLABEL_DIR

    voxelmorph_ckpt: Path = VOXELMORPH_CKPT
    autoencoder_ckpt: Path = AUTOENCODER_CKPT
    radimagenet_weights: Path = RADIMAGE_NET_WEIGHTS

    diffusion_ckpt: Path = DIFFUSION_CKPT
    classifier_ckpt: Path = CLASSIFIER_CKPT

    diffusion_timesteps: int = DIFFUSION_TIMESTEPS
    ddim_steps: int = DDIM_STEPS
    num_gmm_components: int = NUM_GMM_COMPONENTS

    supercon_weight: float = SUPERCON_WEIGHT
    ce_weight: float = CE_WEIGHT
    consistency_weight: float = CONSISTENCY_WEIGHT
    ema_decay: float = EMA_DECAY
    ema_update: float = EMA_UPDATE

    temp_supercon: float = TEMP_SUPERCON
    clip_grad_norm: float = CLIP_GRAD_NORM

    gmm_sigmoid_temp: float = GMM_SIGMOID_TEMP
    lpips_shallow_weight: float = LPIPS_SHALLOW_WEIGHT
    lpips_deep_weight: float = LPIPS_DEEP_WEIGHT

    latent_channels: int = 4
    latent_height: int = 64
    latent_width: int = 128
    latent_dim: int = 2048
