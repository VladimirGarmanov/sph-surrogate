"""Замер точного поиска K ближайших соседей на полных локальных или серверных кадрах без нейросети.

    python scripts/benchmark_particle_neighbors.py --tag phi35_c1000 --out checkpoints/neighbors.json
"""
import argparse
import json
import os
import platform
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surrogate.data import POS, Run  # noqa: E402
from surrogate.particle.neighbors import nearest_neighbors  # noqa: E402


def previous_query(pos, targets, count, tree):
    """Прежняя последовательная реализация с перебором целевых частиц, сохранённая только для сравнения."""
    pos = np.asarray(pos)
    targets = np.asarray(targets, np.int64)
    ids = np.zeros((len(targets), count), np.int64)
    valid = np.zeros_like(ids, dtype=bool)
    n_query = min(count + 1, len(pos))
    _, candidates = tree.query(pos[targets], k=list(range(1, n_query + 1)))
    for row, target in enumerate(targets):
        selected = candidates[row][candidates[row] != target][:count]
        ids[row, :len(selected)] = selected
        valid[row, :len(selected)] = True
    return ids, valid


def previous_frame(pos, targets, count, tree, batch):
    ids = np.empty((len(targets), count), np.int64)
    valid = np.empty_like(ids, dtype=bool)
    for start in range(0, len(targets), batch):
        ids[start:start + batch], valid[start:start + batch] = previous_query(
            pos, targets[start:start + batch], count, tree)
    return ids, valid


def measure_query(function, expected, repeats):
    # Прогреваем каждую реализацию перед замером; сравнения результатов не входят в замер времени.
    actual = function()
    for reference, value in zip(expected, actual):
        np.testing.assert_array_equal(reference, value)
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        actual = function()
        seconds.append(time.perf_counter() - start)
        for reference, value in zip(expected, actual):
            np.testing.assert_array_equal(reference, value)
    return {"seconds": seconds, "median_seconds": float(np.median(seconds)), "ids_and_masks_identical": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--tag", default="phi35_c1000")
    parser.add_argument("--frames", type=int, nargs="+", default=[8, 100, 159])
    parser.add_argument("--neighbors", type=int, default=256)
    parser.add_argument("--batch", type=int, default=32, help="GPU batch size simulated by the old query path")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out")
    args = parser.parse_args()
    if min(args.neighbors, args.batch, args.repeats) < 1 or any(w == 0 or w < -1 for w in args.workers):
        parser.error("counts must be positive; workers must be -1 or positive")
    out = Path(args.out) if args.out else None
    if out is not None and out.exists():
        parser.error("choose a new --out file")
    run = Run(args.data_dir, args.tag)
    if min(args.frames) < 0 or max(args.frames) >= run.n_frames:
        parser.error(f"frames must be in 0..{run.n_frames - 1}")
    targets = np.arange(run.n_soil)
    report = {"tag": args.tag, "scope": "Exact neighbour search only; no network, no GPU timing or quality evaluation",
              "timing_note": "Both paths materialize full ID/mask arrays for comparison; the old rollout normally retained only one inference batch. Tree construction is reported separately.",
              "platform": platform.platform(), "cpu_count": os.cpu_count(),
              "numpy": np.__version__, "scipy": scipy.__version__, "neighbors": args.neighbors,
              "soil_targets_per_frame": len(targets), "previous_batch": args.batch, "repeats": args.repeats,
              "result_arrays_mib": len(targets) * args.neighbors * 9 / 2**20, "frames": []}
    for frame in args.frames:
        pos = run.frame(frame)[:, POS]
        start = time.perf_counter()
        tree = cKDTree(pos)
        row = {"frame": frame, "tree_seconds": time.perf_counter() - start}
        reference = lambda: previous_frame(pos, targets, args.neighbors, tree, args.batch)
        expected = reference()
        row["previous"] = measure_query(reference, expected, args.repeats)
        row["optimized"] = {}
        for workers in args.workers:
            result = measure_query(lambda: nearest_neighbors(pos, targets, args.neighbors, tree, workers=workers),
                                   expected, args.repeats)
            result["search_speedup"] = row["previous"]["median_seconds"] / result["median_seconds"]
            row["optimized"][str(workers)] = result
            print(f"frame={frame} workers={workers} previous={row['previous']['median_seconds']:.4f}s "
                  f"optimized={result['median_seconds']:.4f}s search_speedup={result['search_speedup']:.2f}x "
                  "all IDs and masks identical", flush=True)
        report["frames"].append(row)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
