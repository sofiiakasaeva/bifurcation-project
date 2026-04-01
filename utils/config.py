# тут разные классы для настройки параметров
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    input_dim: int = 2
    hidden_dim: int = 32
    num_hidden_layers: int = 3
    output_dim: int = 1


@dataclass
class TrainConfig:
    epochs: int = 4000
    lr: float = 1e-3
    bc_weight: float = 10.0
    n_int_train: int = 2000
    n_bnd_train: int = 200
    log_every: int = 2000


@dataclass
class DetectorConfig:
    n_int_detect: int = 100
    n_bnd_detect: int = 40
    svd_tol_factor: float = 10.0
    sigma_abs_threshold: float = 1e-2
    lambda_fd_eps: float = 1e-4


@dataclass
class ContinuationConfig:
    gamma: float = 0.05
    alpha_cont: float = 10.0
    max_steps: int = 300
    norm_target_max: float = 15.0
    beta1: float = 1.0
    beta2: float = 0.0
    delta: float = 0.0
    epochs_per_step: int = 3000
    lr: float = 1e-3
    bc_weight: float = 10.0
    n_int_train: int = 2000
    n_bnd_train: int = 200
    lam_clamp_min: float = 0.01
    lam_clamp_max: float = 12.0
    lam_init_step: float = 0.3


@dataclass
class ExperimentConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lam_start: float = 0.5
    lam_end: float = 7.5
    lam_step: float = 0.3
    device: str = "cpu"
    seed: int = 42
    results_dir: str = "results"
    figures_dir: str = "figures"
    reports_dir: str = "reports"
