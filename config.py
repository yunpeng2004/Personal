from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrainConfig:
    seed: int = 42
    image_size: int = 352
    batch_size: int = 2
    num_workers: int = 0

    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    num_epochs: int = 100
    early_stop_patience: int = 14
    use_amp: bool = True
    grad_clip_norm: float = 1.0

    save_dir: str = "checkpoints"
    use_pretrained_backbone: bool = True

    stem_channels: int = 64
    stage_channels: Tuple[int, int, int, int] = (64, 128, 256, 512)
    hidden_channels: int = 128
    state_channels: int = 16
    kan_groups: int = 4
    kan_bases: int = 6

    aux_region_weight: float = 0.18
    aux_boundary_weight: float = 0.06
    aux_confidence_weight: float = 0.06

    use_train_augmentation: bool = True
    boundary_emphasis: float = 2.0
