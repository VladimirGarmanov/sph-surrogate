"""Целевая частица, её текущие соседи, их истории и изменение целевой частицы за следующий шаг."""
import numpy as np
import torch

from ..data import N_TYPES, POS, SOIL, STATE, VEL
from .neighbors import nearest_neighbors
from .timing import measure

N_INPUT = 18                         # скорость — 3, состояние — 10, тип — 3, материал — 2
N_EDGE = 7                          # относительные координаты — 3, расстояние — 1, относительная скорость — 3
N_OUTPUT = 16                       # изменение каждого столбца исходных данных
FEATURE_NAMES = ("x", "y", "z", "vx", "vy", "vz", "rho", "p11", "p22", "p33",
                 "shear12", "shear13", "shear23", "pc", "Ev", "Sv")
FEATURE_UNITS = ("m",) * 3 + ("m/s",) * 3 + ("kg/m^3",) + ("Pa",) * 7 + ("1",) * 2


def input_features(frame, types, phi_deg, cohesion):
    """Используем измеренную или предсказанную скорость; координаты задаются относительно, через рёбра.

    Напряжения маркеров BCE — отклик решателя, а не заданное граничное условие.
    Скрываем всё их материальное состояние одинаково при обучении и прогнозе.
    Координаты, скорость и тип остаются: они описывают движущийся штамп и стенки.
    """
    state = frame[:, STATE].copy()
    state[types != SOIL] = 0
    onehot = np.eye(N_TYPES, dtype=np.float32)[types]
    material = np.broadcast_to(np.array([phi_deg, cohesion], np.float32), (len(frame), 2))
    return np.concatenate([frame[:, VEL], state, onehot, material], axis=1).astype(np.float32)


def build_inputs(history, types, phi_deg, cohesion, target_ids, count, tree=None, features=None,
                 timer=None, neighbor_selection=None):
    """Выбираем соседей в момент t и отслеживаем ТЕ ЖЕ ID в кадрах t-8..t.

    history имеет форму (H, N, 16), от старых кадров к новым. Раньше сосед
    мог быть далеко: собираем его собственную траекторию, сохраняя ID,
    а не подменяем его другой частицей, которая тогда была ближе.
    """
    if history.ndim != 3 or history.shape[-1] != 16 or not len(history):
        raise ValueError("history must have shape (H, N, 16), oldest to current")
    frame = history[-1]
    target_ids = np.asarray(target_ids, np.int64)
    if np.any(types[target_ids] != SOIL):
        raise ValueError("only soil particles can be prediction targets")
    if neighbor_selection is None:
        with measure(timer, "neighbors"):
            ids, valid = nearest_neighbors(frame[:, POS], target_ids, count, tree)
    else:
        # При последовательном прогнозе передаются срезы соседей, найденных один раз для этого кадра.
        ids, valid = neighbor_selection
        if ids.shape != (len(target_ids), count) or valid.shape != ids.shape:
            raise ValueError("precomputed neighbours must have shape (targets, count)")
    with measure(timer, "gather"):
        if features is None:
            features = np.stack([input_features(f, types, phi_deg, cohesion) for f in history])
        # Собираем ТЕ ЖЕ ID по времени, затем переносим H на ось последовательности.
        rel = (history[:, ids, POS] - history[:, target_ids, POS][:, :, None, :]).transpose(1, 2, 0, 3)
        distance = np.linalg.norm(rel, axis=-1, keepdims=True)
        relv = (history[:, ids, VEL] - history[:, target_ids, VEL][:, :, None, :]).transpose(1, 2, 0, 3)
        edge = np.concatenate([rel, distance, relv], axis=-1)
        neighbors = features[:, ids].transpose(1, 2, 0, 3).copy()
        neighbors[~valid] = 0
        edge[~valid] = 0
        return {"x": features[:, target_ids].transpose(1, 0, 2),
                "neighbors": neighbors, "e": edge, "valid": valid}


class ParticleDataset:
    """Выбираем целевые частицы по всему кадру и используем этот кадр для всего пакета.

    Общий момент времени внутри пакета нужен только для того, чтобы не загружать
    массивы и не строить деревья повторно. У каждой целевой частицы свои K соседей;
    пространственные области и дополнительные полосы вокруг их границ не выделяются.
    """
    def __init__(self, runs, cfg, seed=0):
        self.runs, self.cfg = list(runs), cfg
        self.rng = np.random.default_rng(seed)
        start = cfg.history_frames * cfg.frame_stride
        last = cfg.prediction_horizon * cfg.frame_stride
        self.index = [(ri, t) for ri, run in enumerate(self.runs)
                      for t in range(start, run.n_frames - last)]
        if not self.index:
            raise ValueError("not enough frames for the configured history and prediction step")

    def build(self, run, t, target_ids):
        start = t - self.cfg.history_frames * self.cfg.frame_stride
        horizon = self.cfg.prediction_horizon
        if start < 0 or t + horizon * self.cfg.frame_stride >= run.n_frames:
            raise ValueError("target frame lacks the requested history or next frame")
        history = np.stack([run.frame(s) for s in range(start, t + 1, self.cfg.frame_stride)])
        current = history[-1]
        sample = build_inputs(history, run.types, run.phi_deg, run.cohesion,
                              target_ids, self.cfg.neighbors)
        # ID частиц сохраняются. Для каждого будущего кадра предсказывается
        # изменение относительно текущего состояния, а не абсолютные координаты
        # и напряжения. Получается один выходной вектор на каждый горизонт.
        future = np.stack([
            np.asarray(run.soil[t + step * self.cfg.frame_stride][target_ids], np.float32)
            for step in range(1, horizon + 1)
        ])
        sample["y"] = future - current[target_ids][None]
        if horizon == 1:
            sample["y"] = sample["y"][0]
        return sample

    def sample(self):
        ri, t = self.index[self.rng.integers(len(self.index))]
        run = self.runs[ri]
        ids = self.rng.choice(run.n_soil, size=min(self.cfg.batch, run.n_soil), replace=False)
        return self.build(run, t, ids)


class _Moments:
    """Объединяем статистические моменты пакетов без хранения всех наборов соседей."""
    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros(dim, np.float64)
        self.m2 = np.zeros(dim, np.float64)

    def add(self, x):
        x = np.asarray(x, np.float64)
        if not len(x):
            return
        mean = x.mean(0)
        delta = mean - self.mean
        total = self.n + len(x)
        self.m2 += ((x - mean) ** 2).sum(0) + delta ** 2 * self.n * len(x) / total
        self.mean += delta * len(x) / total
        self.n = total

    def result(self):
        std = np.sqrt(self.m2 / max(self.n, 1))
        # Постоянные компоненты сохраняют обучаемые нулевые изменения; защита
        # от деления на ноль не должна создавать искусственный шум.
        return self.mean.astype(np.float32), np.where(std < 1e-8, 1., std).astype(np.float32)


class Stats:
    def __init__(self, arrays):
        self.arrays = {key: np.asarray(value, np.float32) for key, value in arrays.items()}

    @classmethod
    def compute(cls, samples):
        moments = {"node": _Moments(N_INPUT), "edge": _Moments(N_EDGE), "target": _Moments(N_OUTPUT)}
        for sample in samples:
            moments["node"].add(sample["x"].reshape(-1, N_INPUT))
            moments["node"].add(sample["neighbors"][sample["valid"]].reshape(-1, N_INPUT))
            moments["edge"].add(sample["e"][sample["valid"]].reshape(-1, N_EDGE))
            moments["target"].add(sample["y"].reshape(-1, N_OUTPUT))
        if not moments["target"].n:
            raise ValueError("normalization requires at least one training example")
        arrays = {}
        for key, accumulator in moments.items():
            arrays[f"{key}_mean"], arrays[f"{key}_std"] = accumulator.result()
        return cls(arrays)

    def norm(self, key, x):
        return (x - self.arrays[f"{key}_mean"]) / self.arrays[f"{key}_std"]

    def denorm(self, key, x):
        return x * self.arrays[f"{key}_std"] + self.arrays[f"{key}_mean"]

    def to_dict(self):
        return {key: value.tolist() for key, value in self.arrays.items()}


def to_tensors(sample, stats):
    values = {"x": stats.norm("node", sample["x"]),
              "neighbors": stats.norm("node", sample["neighbors"]),
              "e": stats.norm("edge", sample["e"])}
    for key in ("neighbors", "e"):
        values[key][~sample["valid"]] = 0
    if "y" in sample:
        values["y"] = stats.norm("target", sample["y"])
    result = {key: torch.from_numpy(np.ascontiguousarray(value, np.float32)) for key, value in values.items()}
    result["valid"] = torch.from_numpy(np.ascontiguousarray(sample["valid"], bool))
    return result


def add_input_noise(batch, std, generator=None):
    """Добавляет шум к действительным нормализованным признакам входа."""
    if std <= 0:
        return batch
    for key in ("x", "neighbors", "e"):
        noise = torch.randn(batch[key].shape, device=batch[key].device,
                            dtype=batch[key].dtype, generator=generator) * std
        if key != "x":
            noise = noise * batch["valid"].unsqueeze(-1).unsqueeze(-1)
        batch[key] = batch[key] + noise
    return batch
