import yaml
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    root_dir: str
    dataset: str
    image_size: int
    patch_sizes: List[int]
    patch_stride: int
    num_patches_per_bag: Optional[int] = None
    augment: bool = True
    augment_bag_prob: float = 0.5
    augment_patch_prob: float = 0.2
    raw_dir: str = ""  # Auto-built dataset sources (VinDR/, CMD/, ...)


@dataclass
class PreprocessingConfig:
    crop_margin: int
    otsu_threshold: float
    clahe_clip_limit: float
    clahe_grid_size: List[int]


@dataclass
class ModelConfig:
    backbone: str
    embed_dim: int
    num_heads: int
    mil_hidden_dim: int
    num_classes: int
    use_checkpointing: bool = True


@dataclass
class LossConfig:
    use_focal: bool = True
    focal_alpha: List[float] = field(default_factory=lambda: [0.75, 0.25])
    focal_gamma: float = 2.0
    lambda_lem: float = 0.05
    lem_temperature: float = 0.1
    epsilon: float = 1e-8


@dataclass
class TrainingConfig:
    epochs: int
    warmup_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    scheduler_patience: int
    scheduler_factor: float
    use_amp: bool = True
    use_weighted_sampler: bool = True
    early_stopping_patience: int = 10


@dataclass
class EvaluationConfig:
    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    random_seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1010])


@dataclass
class LoggingConfig:
    log_interval: int = 10
    save_interval: int = 5
    tensorboard: bool = True


@dataclass
class Config:
    data: DataConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    loss: LossConfig
    training: TrainingConfig
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(config_path: str) -> Config:
    with open(config_path, 'r', newline='') as f:
        raw = yaml.safe_load(f)

    def _strip_cr(d):
        if isinstance(d, dict):
            return {k: _strip_cr(v) for k, v in d.items()}
        if isinstance(d, list):
            return [_strip_cr(v) for v in d]
        if isinstance(d, str):
            return d.replace('\r', '')
        return d

    raw = _strip_cr(raw)

    # Coerce values to the dataclass field types. YAML 1.2 (PyYAML 6+) parses
    # things like "1e-4" as strings, which then break torch/optim type checks.
    _coerce(raw.get('training', {}), {
        'epochs': int,
        'warmup_epochs': int,
        'batch_size': int,
        'learning_rate': float,
        'weight_decay': float,
        'gradient_clip': float,
        'scheduler_patience': int,
        'scheduler_factor': float,
        'use_amp': bool,
        'use_weighted_sampler': bool,
        'early_stopping_patience': int,
    })
    _coerce(raw.get('data', {}), {
        'image_size': int,
        'patch_sizes': list,
        'patch_stride': int,
        'augment': bool,
        'augment_bag_prob': float,
        'augment_patch_prob': float,
    })
    _coerce(raw.get('preprocessing', {}), {
        'crop_margin': int,
        'otsu_threshold': float,
        'clahe_clip_limit': float,
    })
    _coerce(raw.get('loss', {}), {
        'use_focal': bool,
        'focal_gamma': float,
        'lambda_lem': float,
        'lem_temperature': float,
        'epsilon': float,
    })

    return Config(
        data=DataConfig(**raw.get('data', {})),
        preprocessing=PreprocessingConfig(**raw.get('preprocessing', {})),
        model=ModelConfig(**raw.get('model', {})),
        loss=LossConfig(**raw.get('loss', {})),
        training=TrainingConfig(**raw.get('training', {})),
        evaluation=EvaluationConfig(**raw.get('evaluation', {})),
        logging=LoggingConfig(**raw.get('logging', {})),
    )


def _coerce(section: dict, types: dict) -> None:
    for key, typ in types.items():
        if key not in section:
            continue
        val = section[key]
        if val is None:
            continue
        try:
            if typ is bool:
                if isinstance(val, str):
                    section[key] = val.strip().lower() in ('1', 'true', 'yes', 'on')
                else:
                    section[key] = bool(val)
            elif typ is int:
                section[key] = int(float(val)) if isinstance(val, str) else int(val)
            elif typ is float:
                section[key] = float(val) if not isinstance(val, bool) else float(int(val))
            elif typ is list:
                if isinstance(val, str):
                    section[key] = [v.strip() for v in val.split(',') if v.strip()]
                else:
                    section[key] = list(val)
        except (TypeError, ValueError):
            pass
