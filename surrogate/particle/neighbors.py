"""Nearest neighbours in the entire frame; no cubes or multi-hop expansion."""
import numpy as np
from scipy.spatial import cKDTree


def nearest_neighbors(pos, target_ids, count, tree=None):
    """Return (indices, valid), both (B, count), excluding each target itself.

    Selecting the closest K after trimming a radius or filling it from outside
    gives exactly K-nearest neighbours. A radius is therefore diagnostic, not a
    second independent selection parameter. If the WHOLE frame has fewer than
    K other particles, remaining slots are masked, never duplicate particles.
    """
    pos = np.asarray(pos)
    target_ids = np.asarray(target_ids, dtype=np.int64)
    if pos.ndim != 2 or pos.shape[1] != 3 or not len(pos):
        raise ValueError("pos must be a nonempty (N, 3) array")
    if count < 1 or target_ids.ndim != 1:
        raise ValueError("count must be positive and target_ids one-dimensional")
    if np.any(target_ids < 0) or np.any(target_ids >= len(pos)):
        raise ValueError("target index outside frame")
    out = np.zeros((len(target_ids), count), dtype=np.int64)
    valid = np.zeros_like(out, dtype=bool)
    if not len(target_ids):
        return out, valid
    tree = tree if tree is not None else cKDTree(pos)
    n_query = min(count + 1, len(pos))
    _, candidates = tree.query(pos[target_ids], k=list(range(1, n_query + 1)))
    # Exclude by particle identity, not by dropping the first distance-zero hit:
    # distinct particles can occupy the same position.
    for row, target in enumerate(target_ids):
        ids = candidates[row][candidates[row] != target][:count]
        out[row, :len(ids)] = ids
        valid[row, :len(ids)] = True
    return out, valid
