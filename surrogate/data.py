"""Loading packed runs. Column layout matches NN_SPEC.md."""
import re
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

N_FEAT = 16
POS = slice(0, 3)
VEL = slice(3, 6)        # solver velocity: ParticleNet uses it; legacy GNS derives velocity from positions
RHO = 6
STRESS = slice(7, 13)
PLAST = slice(13, 16)    # pc, Ev, Sv
STATE = slice(6, 16)     # rho + 6 stress + pc/Ev/Sv: everything with history
N_STATE = 10
P33 = 9                  # normal stress along z, used for plate pressure

SOIL, WALL, PLATE = 0, 1, 2
N_TYPES = 3

TAG_RE = re.compile(r"phi(\d+(?:\.\d+)?)_c(\d+(?:\.\d+)?)")


def parse_tag(tag):
    m = TAG_RE.fullmatch(tag)
    if m is None:
        raise ValueError(f"tag {tag!r} does not look like phi35_c1000")
    return float(m.group(1)), float(m.group(2))


class Run:
    """One simulation. Arrays are memory-mapped lazily so 25 runs cost nothing
    until touched, and the object stays picklable for DataLoader workers."""

    def __init__(self, data_dir, tag):
        self.dir = Path(data_dir)
        self.tag = tag
        self.phi_deg, self.cohesion = parse_tag(tag)

    # -- lazy arrays --------------------------------------------------------
    @cached_property
    def soil(self):
        return np.load(self.dir / f"{self.tag}.npy", mmap_mode="r")

    @cached_property
    def plate(self):
        return np.load(self.dir / f"{self.tag}_plate.npy", mmap_mode="r")

    @cached_property
    def boundary(self):
        return np.load(self.dir / f"{self.tag}_boundary.npy", mmap_mode="r")

    def __getstate__(self):
        # drop memmaps before pickling, workers reopen them
        return {k: v for k, v in self.__dict__.items()
                if k not in ("soil", "plate", "boundary")}

    # -- shapes -------------------------------------------------------------
    @property
    def n_frames(self):
        return min(len(self.soil), len(self.plate))   # guard against the 201-frame run

    @property
    def n_soil(self):
        return self.soil.shape[1]

    @property
    def n_plate(self):
        return self.plate.shape[1]

    @property
    def n_wall(self):
        return self.boundary.shape[0]

    @property
    def n_nodes(self):
        return self.n_soil + self.n_plate + self.n_wall

    @cached_property
    def types(self):
        """Node order everywhere: soil, plate, wall."""
        return np.concatenate([
            np.full(self.n_soil, SOIL, np.int64),
            np.full(self.n_plate, PLATE, np.int64),
            np.full(self.n_wall, WALL, np.int64),
        ])

    def frame(self, t):
        """(n_nodes, 16) float32 for frame t, in the order soil, plate, wall."""
        return np.concatenate([
            np.asarray(self.soil[t], np.float32),
            np.asarray(self.plate[t], np.float32),
            np.asarray(self.boundary, np.float32),
        ])

    # -- reference curve ----------------------------------------------------
    @cached_property
    def reference(self):
        """pressure_sinkage.csv from the solver, or None if not downloaded."""
        candidates = [
            self.dir / self.tag / "pressure_sinkage.csv",
            self.dir / f"{self.tag}_pressure_sinkage.csv",
            self.dir / f"{self.tag}.pressure_sinkage.csv",
        ]
        for c in candidates:
            if c.exists():
                return pd.read_csv(c, skipinitialspace=True)
        return None

    def __repr__(self):
        return f"Run({self.tag}, phi={self.phi_deg}, c={self.cohesion})"


def discover_runs(data_dir):
    """All tags with a soil array present, sorted."""
    data_dir = Path(data_dir)
    tags = []
    for p in data_dir.glob("*.npy"):
        stem = p.stem
        if stem.endswith("_plate") or stem.endswith("_boundary"):
            continue
        if (data_dir / f"{stem}_plate.npy").exists() and (data_dir / f"{stem}_boundary.npy").exists():
            tags.append(stem)
    return [Run(data_dir, t) for t in sorted(tags)]


def solver_wall_seconds(data_dir, tag):
    """wall_seconds from runs_soil.csv for this tag, or None."""
    path = Path(data_dir) / "runs_soil.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, skipinitialspace=True)
    if "wall_seconds" not in df.columns:
        return None
    hit = df[(df.astype(str) == tag).any(axis=1)]
    if hit.empty:
        return None
    return float(hit["wall_seconds"].iloc[0])
