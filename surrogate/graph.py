"""Построение графа соседей и исходных, ещё не нормализованных признаков."""
import numpy as np
from scipy.spatial import cKDTree

from .data import N_TYPES, N_STATE

N_NODE_FEAT = 3 + N_STATE + N_TYPES + 2   # скорость, состояние, тип в кодировке one-hot, (phi, cohesion) = 18
N_EDGE_FEAT = 3 + 1 + 3                   # относительные координаты, расстояние, относительная скорость = 7
N_TARGET = 3 + N_STATE                    # ускорение + d(state)/dt = 13


def radius_edges(pos, radius, receiver_mask=None):
    """Направленные рёбра (senders, receivers) для всех пар ближе `radius`.
    Если задана `receiver_mask`, оставляем только рёбра, ведущие к отмеченным узлам."""
    tree = cKDTree(pos)
    pairs = tree.query_pairs(radius, output_type="ndarray")      # форма (M, 2), индексы i < j
    s = np.concatenate([pairs[:, 0], pairs[:, 1]])
    r = np.concatenate([pairs[:, 1], pairs[:, 0]])
    if receiver_mask is not None:
        keep = receiver_mask[r]
        s, r = s[keep], r[keep]
    return s.astype(np.int64), r.astype(np.int64)


def neighbor_counts(pos_query, pos_all, radius):
    """Число узлов из `pos_all` в пределах `radius` от каждой точки запроса.
    Саму точку исключаем, если она входит в `pos_all`."""
    tree = cKDTree(pos_all)
    counts = tree.query_ball_point(pos_query, radius, return_length=True)
    return counts.astype(np.int64)


def node_features(pos_t, pos_prev, state_t, types, phi_deg, cohesion, dt):
    """(N, 18): скорость по конечным разностям, состояние, тип в кодировке
    one-hot и общие параметры. Абсолютные координаты намеренно не включены."""
    n = len(pos_t)
    vel = (pos_t - pos_prev) / dt
    onehot = np.eye(N_TYPES, dtype=np.float32)[types]
    glob = np.tile(np.array([phi_deg, cohesion], np.float32), (n, 1))
    return np.concatenate([vel, state_t, onehot, glob], axis=1).astype(np.float32)


def edge_features(pos, vel, senders, receivers):
    """(E, 7): pos_j - pos_i, |.|, vel_j - vel_i; i — получатель, j — отправитель."""
    rel = pos[senders] - pos[receivers]
    dist = np.linalg.norm(rel, axis=1, keepdims=True)
    relv = vel[senders] - vel[receivers]
    return np.concatenate([rel, dist, relv], axis=1).astype(np.float32)


def targets(pos_next, pos_t, pos_prev, state_next, state_t, dt):
    """(N, 13): ускорение по трём положениям и скорость изменения состояния по двум кадрам.
    Вычисляются из входов, возможно зашумлённых, чтобы сеть училась исправлять ошибки."""
    acc = (pos_next - 2.0 * pos_t + pos_prev) / (dt * dt)
    rate = (state_next - state_t) / dt
    return np.concatenate([acc, rate], axis=1).astype(np.float32)
