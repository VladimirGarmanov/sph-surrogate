"""Обучающие входы из зафиксированного синхронного прогноза текущей модели.

Из решателя берутся только начальная история и заданное движение границ.
Более поздние истинные состояния грунта используются только как цели обучения,
никогда как признаки или координаты соседей. Собранный прогноз отсоединён
от графа вычислений: градиенты проходят только через следующий переход.
"""
import numpy as np
from scipy.spatial import cKDTree

from ..data import SOIL, STATE
from .data import build_inputs, input_features


def rollout_windows(runs, cfg, steps):
    """Все допустимые пары (запуск, последний известный кадр), включая позднее погружение штампа."""
    if steps < 1:
        raise ValueError("rollout_steps must be positive")
    first = cfg.history_frames * cfg.frame_stride
    windows = [(ri, frame) for ri, run in enumerate(runs)
               for frame in range(first, run.n_frames - steps * cfg.frame_stride)]
    if not windows:
        raise ValueError("no training run has enough frames for the history and rollout_steps")
    return windows


class RolloutReplay:
    """Зафиксированные предсказанные истории с корректными целями восстановления следующего состояния."""
    def __init__(self, run, cfg, result, seed=0):
        if str(result.get("mode", "")) != "rollout":
            raise ValueError("replay requires a synchronous rollout result")
        self.run, self.cfg = run, cfg
        self.rng = np.random.default_rng(seed)
        self.types = run.types
        self.targets = np.flatnonzero(self.types == SOIL)
        np.testing.assert_array_equal(result["particle_ids"], self.targets)
        predicted = np.asarray(result["particle_predicted"], np.float32)
        self.reference = np.asarray(result["particle_reference"], np.float32).copy()
        frames = np.asarray(result["particle_frames"], np.int64)
        if len(frames) < 2 or predicted.shape != (len(frames), len(self.targets), 16):
            raise ValueError("replay predictions do not match frames and soil IDs")
        if self.reference.shape != predicted.shape:
            raise ValueError("replay reference and predictions must have the same shape")
        if not np.isfinite(predicted).all() or not np.isfinite(self.reference).all():
            raise ValueError("non-finite replay predictions or reference")
        initial = int(result["frames"][0])
        expected = initial + cfg.frame_stride * np.arange(1, len(frames) + 1)
        if not np.array_equal(frames, expected):
            raise ValueError("replay frames do not follow the checkpoint frame_stride")
        first = initial - cfg.history_frames * cfg.frame_stride
        if first < 0:
            raise ValueError("replay lacks a complete warm start")
        warm = np.stack([run.frame(t) for t in range(first, initial + 1, cfg.frame_stride)])
        warm[:, self.types != SOIL, STATE] = 0
        following = np.zeros((len(frames), len(self.types), 16), np.float32)
        following[:, self.targets] = predicted
        for index, frame in enumerate(frames):
            following[index, self.types != SOIL, :6] = np.concatenate(
                [run.plate[frame, :, :6], run.boundary[:, :6]])
        self.history = np.concatenate([warm, following])
        self.features = np.stack([input_features(f, self.types, run.phi_deg, run.cohesion)
                                  for f in self.history])
        self.trees = {}
        self.frame_ids = frames
        self.soil_rows = np.full(len(self.types), -1, np.int64)
        self.soil_rows[self.targets] = np.arange(len(self.targets))

    def build(self, index, target_ids):
        if not 0 <= index < len(self.reference):
            raise ValueError("replay transition index out of range")
        ids = np.asarray(target_ids, np.int64)
        if ids.ndim != 1 or not len(ids) or np.any(ids < 0) or np.any(ids >= len(self.types)):
            raise ValueError("invalid replay target IDs")
        if np.any(self.soil_rows[ids] < 0):
            raise ValueError("replay targets must be soil particles")
        stop = index + self.cfg.history_frames + 1
        history = self.history[index:stop]
        if index not in self.trees:
            self.trees[index] = cKDTree(history[-1, :, :3])
        sample = build_inputs(history, self.types, self.run.phi_deg, self.run.cohesion,
                              ids, self.cfg.neighbors, tree=self.trees[index],
                              features=self.features[index:stop],
                              neighbor_history=self.cfg.neighbor_history)
        # Цель восстановления: будущее ИСТИННОЕ состояние минус ТЕКУЩЕЕ ПРЕДСКАЗАННОЕ
        # состояние. Вычитание текущего истинного состояния учило бы неправильному изменению.
        sample["y"] = self.reference[index, self.soil_rows[ids]] - history[-1, ids]
        return sample

    def sample(self):
        # В нулевом переходе вся входная история ещё истинная; такие примеры
        # выбираются отдельно. Каждое обновление по собранному прогнозу должно содержать
        # хотя бы одно предсказанное состояние центральной частицы и её соседей.
        index = int(self.rng.integers(1, len(self.reference)))
        ids = self.rng.choice(self.targets, size=min(self.cfg.batch, len(self.targets)), replace=False)
        return self.build(index, ids)
