
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataConfig:
    dataset_name: str = "random"
    data_root: str = "./data"
    image_size: int = 64
    num_classes: int = 10
    num_channels: int = 3
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42

@dataclass
class AugmentationConfig:
    resize_scale: tuple = (0.08, 1.0)
    color_jitter_strength: float = 0.5
    blur_sigma: tuple = (0.1, 2.0)
    blur_kernel_size: int = 23
    gaussian_blur_prob: float = 0.5
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.2
    rotation_degrees: int = 90
    num_views: int = 2

@dataclass
class ModelConfig:
    backbone: str = "resnet18"
    projection_dim: int = 128
    hidden_dim: int = 2048
    num_layers: int = 2
    use_batch_norm: bool = True
    dropout_rate: float = 0.0

@dataclass
class TrainingConfig:
    batch_size: int = 128
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-6
    momentum: float = 0.9
    temperature: float = 0.07
    num_negative_pairs: int = 256
    warmup_epochs: int = 5
    cosine_annealing: bool = True
    checkpoint_freq: int = 10
    log_freq: int = 10
    device: str = "cuda"
    num_workers: int = 0
    pin_memory: bool = False

@dataclass
class EvalConfig:
    linear_probe_epochs: int = 100
    linear_probe_lr: float = 0.1
    linear_probe_batch_size: int = 128
    embedding_dim: int = 2048
    visualization_sample_size: int = 5000

@dataclass
class ExperimentConfig:
    data: DataConfig = None
    augmentation: AugmentationConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    eval: EvalConfig = None
    experiment_name: str = "satellite_ssl_colab"
    seed: int = 42
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"

    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.augmentation is None:
            self.augmentation = AugmentationConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.eval is None:
            self.eval = EvalConfig()
        
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
