"""All knobs in one place. Every field can be overridden from the train CLI."""
from dataclasses import dataclass, asdict, fields
import json


@dataclass
class Config:
    # ---- data -------------------------------------------------------------
    data_dir: str = "data"
    dt: float = 0.02            # seconds between frames
    spacing: float = 0.02       # SPH particle spacing [m]
    radius: float = 0.04        # neighbour radius [m]; tune with scripts/neighbors.py
    holdout: str = "phi30_c500,phi40_c2000,phi25_c5000,phi45_c0"  # never trained on

    # ---- graph ------------------------------------------------------------
    # Edges whose receiver is a wall/plate marker are useless (we never predict
    # those nodes) and roughly double the edge count. Off by default.
    edges_to_static: bool = False

    # ---- training crops ---------------------------------------------------
    # A full frame is ~72k nodes / ~2-3M edges: does not fit a GPU during
    # backprop. We train on random cubes of half-size `crop_half`. Nodes closer
    # than `crop_halo` to the cube border see a truncated neighbourhood, so
    # they are inputs only, not loss targets.
    crop_half: float = 0.12         # 0 disables cropping (full frames)
    crop_halo: float = 0.08
    crop_near_plate: float = 0.7    # probability a crop is centred under the plate

    # ---- input noise (the autoregressive-stability trick from GNS) -------
    noise_pos: float = 3e-5         # [m]; plate moves 1e-4 m per frame, so ~0.3 of a step
    noise_state: float = 0.3        # fraction of each state feature's per-frame change std

    # ---- model ------------------------------------------------------------
    hidden: int = 128
    layers: int = 6

    # ---- optimisation -----------------------------------------------------
    batch: int = 4                  # crops per step
    lr: float = 1e-4
    lr_decay_steps: int = 50_000    # lr *= 0.1 every this many steps
    steps: int = 20_000
    stats_frames: int = 40          # frames sampled to estimate normalisation
    val_samples: int = 32
    log_every: int = 100
    val_every: int = 1000
    workers: int = 2
    seed: int = 0
    out_dir: str = "checkpoints/gns"

    # ---- helpers ----------------------------------------------------------
    @property
    def holdout_tags(self):
        return [t for t in self.holdout.split(",") if t]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
