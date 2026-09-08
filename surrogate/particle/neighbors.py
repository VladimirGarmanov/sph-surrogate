"""Nearest neighbours in the entire frame; no cubes or multi-hop expansion."""
import numpy as np
from scipy.spatial import cKDTree


def nearest_neighbors(pos, target_ids, count, tree=None, workers=1, chunk_size=4096):
    """Return (indices, valid), both (B, count), excluding each target itself.

    Selecting the closest K after trimming a radius or filling it from outside
    gives exactly K-nearest neighbours. A radius is therefore diagnostic, not a
    second independent selection parameter. If the WHOLE frame has fewer than
    K other particles, remaining slots are masked, never duplicate particles.
    Query many targets together; chunks bound temporary query memory. Parallel
    workers change execution only: the search remains exact (eps=0).
    """
    pos = np.asarray(pos)
    target_ids = np.asarray(target_ids, dtype=np.int64)
    if pos.ndim != 2 or pos.shape[1] != 3 or not len(pos):
        raise ValueError("pos must be a nonempty (N, 3) array")
    if count < 1 or target_ids.ndim != 1:
        raise ValueError("count must be positive and target_ids one-dimensional")
    if workers == 0 or workers < -1 or chunk_size < 1:
        raise ValueError("workers must be -1 or positive; chunk_size must be positive")
    if np.any(target_ids < 0) or np.any(target_ids >= len(pos)):
        raise ValueError("target index outside frame")
    out = np.zeros((len(target_ids), count), dtype=np.int64)
    valid = np.zeros_like(out, dtype=bool)
    available = min(count, len(pos) - 1)
    if not len(target_ids) or not available:
        return out, valid
    tree = tree if tree is not None else cKDTree(pos)
    n_query = min(count + 1, len(pos))
    columns = np.arange(available)[None, :]
    for start in range(0, len(target_ids), chunk_size):
        stop = min(start + chunk_size, len(target_ids))
        targets = target_ids[start:stop]
        _, candidates = tree.query(pos[targets], k=n_query, workers=workers)
        # Remove the target wherever it appears, preserving the query order.
        # With many coincident particles the target may not be returned at all.
        is_self = candidates == targets[:, None]
        self_column = np.where(is_self.any(1), is_self.argmax(1), n_query)
        source_columns = columns + (columns >= self_column[:, None])
        out[start:stop, :available] = np.take_along_axis(candidates, source_columns, axis=1)
        valid[start:stop, :available] = True
    return out, valid
