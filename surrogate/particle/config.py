"""Настройки обучения по отдельным частицам без выделения пространственных областей."""
from dataclasses import asdict, dataclass, fields


@dataclass
class Config:
    data_dir: str = "data"
    out_dir: str = "checkpoints/particle"
    holdout: str = "phi30_c500,phi40_c2000,phi25_c5000,phi45_c0"
    dt: float = 0.02
    frame_stride: int = 1
    history_frames: int = 8         # предыдущие кадры ПЛЮС текущий кадр
    neighbors: int = 256
    batch: int = 32                  # целевые частицы, каждая со своими историями соседей
    hidden: int = 256
    lr: float = 1e-4
    lr_decay_steps: int = 50_000
    steps: int = 20_000
    stats_frames: int = 40
    val_samples: int = 16            # фиксированные пакеты примеров для проверки прогноза на один шаг
    log_every: int = 100
    val_every: int = 1000
    seed: int = 0
    device: str = "auto"

    def __post_init__(self):
        for name in ("frame_stride", "neighbors", "batch", "hidden", "steps",
                     "lr_decay_steps", "stats_frames", "val_samples", "log_every", "val_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.dt <= 0 or self.lr <= 0:
            raise ValueError("dt and lr must be positive")
        if self.history_frames < 0:
            raise ValueError("history_frames must be nonnegative")

    @property
    def holdout_tags(self):
        return [tag.strip() for tag in self.holdout.split(",") if tag.strip()]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})
