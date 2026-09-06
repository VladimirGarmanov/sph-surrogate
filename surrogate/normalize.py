"""Per-feature mean/std for nodes, edges and targets.

Scales span eight orders of magnitude (velocities ~5e-3 m/s, stresses ~1e5 Pa).
Without this the loss is all stress and the net treats acceleration as noise.
"""
import numpy as np


class _Welford:
    def __init__(self, dim):
        self.n = 0
        self.s = np.zeros(dim, np.float64)
        self.ss = np.zeros(dim, np.float64)

    def add(self, x):
        x = np.asarray(x, np.float64)
        self.n += len(x)
        self.s += x.sum(0)
        self.ss += (x * x).sum(0)

    def result(self, floor=1e-8):
        mean = self.s / max(self.n, 1)
        var = self.ss / max(self.n, 1) - mean * mean
        std = np.sqrt(np.maximum(var, 0.0))
        std = np.where(std < floor, 1.0, std)      # constant features (e.g. a one-hot column)
        return mean.astype(np.float32), std.astype(np.float32)


class Stats:
    KEYS = ("node", "edge", "target")

    def __init__(self, **arrays):
        for k in self.KEYS:
            setattr(self, f"{k}_mean", arrays[f"{k}_mean"])
            setattr(self, f"{k}_std", arrays[f"{k}_std"])

    # -- construction -------------------------------------------------------
    @classmethod
    def compute(cls, samples):
        """`samples` yields raw dicts with x, e, y, mask (see dataset.build)."""
        acc = {k: None for k in cls.KEYS}
        for smp in samples:
            for k, arr in (("node", smp["x"]), ("edge", smp["e"]), ("target", smp["y"][smp["mask"]])):
                if acc[k] is None:
                    acc[k] = _Welford(arr.shape[1])
                acc[k].add(arr)
        out = {}
        for k in cls.KEYS:
            out[f"{k}_mean"], out[f"{k}_std"] = acc[k].result()
        return cls(**out)

    # -- apply --------------------------------------------------------------
    def norm(self, kind, arr):
        return (arr - getattr(self, f"{kind}_mean")) / getattr(self, f"{kind}_std")

    def denorm(self, kind, arr):
        return arr * getattr(self, f"{kind}_std") + getattr(self, f"{kind}_mean")

    @property
    def state_step_std(self):
        """Raw std of the per-frame change of the 10 state features (target rate * dt).
        Input noise is scaled by this, not by the state's own std: the noise must be
        comparable to what changes in one step, otherwise the target degenerates into
        'undo the noise' and the net stops learning physics."""
        return self.target_std[3:] * self._dt

    _dt = 0.02  # overwritten by whoever loads the stats with a config

    # -- io -----------------------------------------------------------------
    def as_dict(self):
        return {f"{k}_{m}": getattr(self, f"{k}_{m}") for k in self.KEYS for m in ("mean", "std")}

    def save(self, path):
        np.savez(path, **self.as_dict())

    @classmethod
    def load(cls, path_or_dict):
        d = path_or_dict if isinstance(path_or_dict, dict) else dict(np.load(path_or_dict))
        return cls(**{k: np.asarray(v, np.float32) for k, v in d.items()})
