"""Загрузка упакованных запусков. Порядок столбцов соответствует NN_SPEC.md."""
import re
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

N_FEAT = 16
POS = slice(0, 3)
VEL = slice(3, 6)        # скорость из решателя: её использует ParticleNet; прежняя GNS вычисляет скорость по координатам
RHO = 6
STRESS = slice(7, 13)
PLAST = slice(13, 16)    # параметры пластичности: pc, Ev, Sv
STATE = slice(6, 16)     # rho + 6 компонент напряжения + pc/Ev/Sv: все величины с историей
N_STATE = 10
P33 = 9                  # нормальное напряжение вдоль z для расчёта давления под штампом

SOIL, WALL, PLATE = 0, 1, 2
N_TYPES = 3

TAG_RE = re.compile(r"phi(\d+(?:\.\d+)?)_c(\d+(?:\.\d+)?)")


def parse_tag(tag):
    m = TAG_RE.fullmatch(tag)
    if m is None:
        raise ValueError(f"tag {tag!r} does not look like phi35_c1000")
    return float(m.group(1)), float(m.group(2))


class Run:
    """Одна симуляция. Массивы лениво отображаются из файлов в память:
    данные 25 запусков не загружаются до обращения к ним. Объект при этом
    остаётся пригодным для сериализации при передаче рабочим процессам DataLoader."""

    def __init__(self, data_dir, tag):
        self.dir = Path(data_dir)
        self.tag = tag
        self.phi_deg, self.cohesion = parse_tag(tag)

    # -- ленивая загрузка массивов ------------------------------------------
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
        # убираем отображения файлов перед сериализацией; рабочие процессы откроют их заново
        return {k: v for k, v in self.__dict__.items()
                if k not in ("soil", "plate", "boundary")}

    # -- размеры массивов ---------------------------------------------------
    @property
    def n_frames(self):
        return min(len(self.soil), len(self.plate))   # защита от несовпадения длин в запуске с 201 кадром

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
        """Порядок узлов везде одинаков: грунт, штамп, стенка."""
        return np.concatenate([
            np.full(self.n_soil, SOIL, np.int64),
            np.full(self.n_plate, PLATE, np.int64),
            np.full(self.n_wall, WALL, np.int64),
        ])

    def frame(self, t):
        """Массив float32 формы (n_nodes, 16) для кадра t; порядок: грунт, штамп, стенка."""
        return np.concatenate([
            np.asarray(self.soil[t], np.float32),
            np.asarray(self.plate[t], np.float32),
            np.asarray(self.boundary, np.float32),
        ])

    # -- эталонная кривая ----------------------------------------------------
    @cached_property
    def reference(self):
        """Файл pressure_sinkage.csv из решателя или None, если он не загружен."""
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
    """Отсортированные имена всех запусков, для которых есть массив грунта."""
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
    """Значение wall_seconds из runs_soil.csv для данного запуска или None."""
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
