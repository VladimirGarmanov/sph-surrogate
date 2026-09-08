"""Ближайшие соседи во всём кадре, без кубических областей и расширения по цепочкам связей."""
import numpy as np
from scipy.spatial import cKDTree


def nearest_neighbors(pos, target_ids, count, tree=None, workers=1, chunk_size=4096):
    """Возвращаем (indices, valid), оба формы (B, count), исключая саму целевую частицу.

    Отбор ближайших K с отсечением по радиусу или дополнением снаружи даёт
    ровно K ближайших соседей. Поэтому радиус служит для диагностики,
    а не вторым независимым параметром отбора. Если во ВСЁМ кадре меньше
    K других частиц, лишние места маскируются, а частицы не дублируются.
    Обрабатываем много целевых частиц за один запрос; размер блока ограничивает
    временную память. Параллельные рабочие потоки меняют лишь выполнение:
    поиск остаётся точным (eps=0).
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
        # Удаляем целевую частицу из любой позиции результата, сохраняя порядок соседей.
        # При множестве совпадающих координат целевая частица может вообще не попасть в ответ.
        is_self = candidates == targets[:, None]
        self_column = np.where(is_self.any(1), is_self.argmax(1), n_query)
        source_columns = columns + (columns >= self_column[:, None])
        out[start:stop, :available] = np.take_along_axis(candidates, source_columns, axis=1)
        valid[start:stop, :available] = True
    return out, valid
