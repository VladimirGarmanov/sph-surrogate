"""Neighbour graph and raw (un-normalised) feature construction."""
import numpy as np
from scipy.spatial import cKDTree

from .data import N_TYPES, N_STATE

N_NODE_FEAT = 3 + N_STATE + N_TYPES + 2   # vel, state, one-hot, (phi, cohesion) = 18
N_EDGE_FEAT = 3 + 1 + 3                   # rel pos, dist, rel vel = 7
N_TARGET = 3 + N_STATE                    # acc + d(state)/dt = 13


def radius_edges(pos, radius, receiver_mask=None):
    """Directed edges (senders, receivers) for all pairs closer than `radius`.
    If `receiver_mask` is given, only edges into masked nodes are kept."""
    tree = cKDTree(pos)
    pairs = tree.query_pairs(radius, output_type="ndarray")      # (M, 2), i < j
    s = np.concatenate([pairs[:, 0], pairs[:, 1]])
    r = np.concatenate([pairs[:, 1], pairs[:, 0]])
    if receiver_mask is not None:
        keep = receiver_mask[r]
        s, r = s[keep], r[keep]
    return s.astype(np.int64), r.astype(np.int64)


def neighbor_counts(pos_query, pos_all, radius):
    """Number of nodes of `pos_all` within `radius` of each query point,
    excluding the point itself when it belongs to `pos_all`."""
    tree = cKDTree(pos_all)
    counts = tree.query_ball_point(pos_query, radius, return_length=True)
    return counts.astype(np.int64)


def node_features(pos_t, pos_prev, state_t, types, phi_deg, cohesion, dt):
    """(N, 18): finite-difference velocity, state, one-hot type, globals.
    Absolute position is deliberately absent."""
    n = len(pos_t)
    vel = (pos_t - pos_prev) / dt
    onehot = np.eye(N_TYPES, dtype=np.float32)[types]
    glob = np.tile(np.array([phi_deg, cohesion], np.float32), (n, 1))
    return np.concatenate([vel, state_t, onehot, glob], axis=1).astype(np.float32)


def edge_features(pos, vel, senders, receivers):
    """(E, 7): pos_j - pos_i, |.|, vel_j - vel_i with i = receiver, j = sender."""
    rel = pos[senders] - pos[receivers]
    dist = np.linalg.norm(rel, axis=1, keepdims=True)
    relv = vel[senders] - vel[receivers]
    return np.concatenate([rel, dist, relv], axis=1).astype(np.float32)


def targets(pos_next, pos_t, pos_prev, state_next, state_t, dt):
    """(N, 13): acceleration from three positions, state rate from two frames.
    Computed from the (possibly noisy) inputs so the net learns to correct."""
    acc = (pos_next - 2.0 * pos_t + pos_prev) / (dt * dt)
    rate = (state_next - state_t) / dt
    return np.concatenate([acc, rate], axis=1).astype(np.float32)
