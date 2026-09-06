"""torch Dataset: (run, frame) -> one graph with noisy inputs and finite-difference targets."""
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import POS, STATE, SOIL, PLATE
from .graph import radius_edges, node_features, edge_features, targets


class FrameDataset(Dataset):
    """One item = one frame t of one run (needs t-1 and t+1), optionally cropped."""

    def __init__(self, runs, cfg, stats=None, train=True, seed=0):
        self.runs = list(runs)
        self.cfg = cfg
        self.stats = stats
        k = cfg.frame_stride if cfg.frame_stride > 0 else 1
        if stats is not None:
            stats._dt = cfg.dt*k
        self.train = train
        k = cfg.frame_stride if cfg.frame_stride > 0 else 1
        self.index = [(ri, t) for ri, run in enumerate(self.runs) for t in range(k, run.n_frames - k)]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.index)

    # ------------------------------------------------------------------ raw
    def build(self, run, t, noise=False, crop=False, rng=None):
        """Raw numpy sample. Kept separate from __getitem__ so stats and rollout can reuse it."""
        cfg = self.cfg
        rng = rng or self.rng
        k = cfg.frame_stride if cfg.frame_stride > 0 else 1
        dt = cfg.dt * k
        f_prev, f_t, f_next = run.frame(t - k), run.frame(t), run.frame(t + k)
        pos_prev, pos_t, pos_next = f_prev[:, POS].copy(), f_t[:, POS].copy(), f_next[:, POS]
        state_t, state_next = f_t[:, STATE].copy(), f_next[:, STATE]
        types = run.types
        soil = types == SOIL

        if noise:
            pos_prev[soil] += rng.normal(0, cfg.noise_pos, pos_prev[soil].shape).astype(np.float32)
            pos_t[soil] += rng.normal(0, cfg.noise_pos, pos_t[soil].shape).astype(np.float32)
            if self.stats is not None and cfg.noise_state > 0:
                sigma = cfg.noise_state * self.stats.state_step_std
                state_t[soil] += (rng.normal(0, 1, state_t[soil].shape) * sigma).astype(np.float32)

        if crop and cfg.crop_half > 0:
            idx, loss_mask = self._crop(pos_t, types, rng)
            pos_prev, pos_t, pos_next = pos_prev[idx], pos_t[idx], pos_next[idx]
            state_t, state_next, types, soil = state_t[idx], state_next[idx], types[idx], soil[idx]
        else:
            loss_mask = soil

        recv = None if cfg.edges_to_static else soil
        s, r = radius_edges(pos_t, cfg.radius, receiver_mask=recv)
        vel = (pos_t - pos_prev) / dt
        # Raw (unnormalized) direction for the Newton's-third-law correction in model.py.
        # Must come from raw positions, not the normalized edge features: dividing x/y/z by
        # their (unequal) per-axis std would distort the direction, not just its scale.
        rel_raw = pos_t[s] - pos_t[r]
        dist_raw = np.linalg.norm(rel_raw, axis=1, keepdims=True)
        unit = (rel_raw / np.clip(dist_raw, 1e-8, None)).astype(np.float32)
        soil_edge = (soil[s] & soil[r])
        return {
            "x": node_features(pos_t, pos_prev, state_t, types, run.phi_deg, run.cohesion, dt),
            "e": edge_features(pos_t, vel, s, r),
            "senders": s,
            "receivers": r,
            "unit": unit,
            "soil_edge": soil_edge,
            "y": targets(pos_next, pos_t, pos_prev, state_next, state_t, dt),
            "mask": loss_mask,
        }

    def _crop(self, pos, types, rng):
        cfg = self.cfg
        if rng.random() < cfg.crop_near_plate:
            plate = np.flatnonzero(types == PLATE)
            center = pos[rng.choice(plate)].copy()
            center[2] -= 0.5 * cfg.crop_half           # box mostly below the plate
        else:
            center = pos[rng.choice(np.flatnonzero(types == SOIL))]
        d = np.abs(pos - center).max(axis=1)
        idx = np.flatnonzero(d <= cfg.crop_half)
        inner = d[idx] <= cfg.crop_half - cfg.crop_halo
        loss_mask = inner & (types[idx] == SOIL)
        if not loss_mask.any():                         # tiny crop at a corner: use everything
            loss_mask = types[idx] == SOIL
        return idx, loss_mask

    # ---------------------------------------------------------------- torch
    def __getitem__(self, i):
        ri, t = self.index[i]
        rng = np.random.default_rng(self.rng.integers(1 << 31) + i)
        smp = self.build(self.runs[ri], t, noise=self.train, crop=True, rng=rng)
        return to_tensors(smp, self.stats)


def to_tensors(smp, stats=None):
    x, e, y = smp["x"], smp["e"], smp["y"]
    if stats is not None:
        x, e, y = stats.norm("node", x), stats.norm("edge", e), stats.norm("target", y)
    return {
        "x": torch.from_numpy(np.ascontiguousarray(x, np.float32)),
        "e": torch.from_numpy(np.ascontiguousarray(e, np.float32)),
        "senders": torch.from_numpy(smp["senders"]),
        "receivers": torch.from_numpy(smp["receivers"]),
        "unit": torch.from_numpy(np.ascontiguousarray(smp["unit"], np.float32)),
        "soil_edge": torch.from_numpy(np.asarray(smp["soil_edge"], bool)),
        "y": torch.from_numpy(np.ascontiguousarray(y, np.float32)),
        "mask": torch.from_numpy(np.asarray(smp["mask"], bool)),
    }


def collate(items):
    """Concatenate graphs into one big disconnected graph (offset edge indices)."""
    out = {k: [] for k in items[0]}
    offset = 0
    for it in items:
        for k, v in it.items():
            out[k].append(v + offset if k in ("senders", "receivers") else v)
        offset += it["x"].shape[0]
    return {k: torch.cat(v) for k, v in out.items()}


def sample_frames(runs, n, rng, stride=1):
    """n random (run, frame) pairs, frames uniformly over the middle of each run."""
    out = []
    for _ in range(n):
        run = runs[rng.integers(len(runs))]
        out.append((run, int(rng.integers(stride, run.n_frames - stride))))
    return out
