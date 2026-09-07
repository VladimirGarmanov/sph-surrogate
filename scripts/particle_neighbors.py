"""Measure radius populations and the physical reach of K nearest neighbours.

    python scripts/particle_neighbors.py --tag phi35_c1000 --neighbors 128 256 512
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surrogate.data import POS, Run  # noqa: E402
from surrogate.particle.neighbors import nearest_neighbors  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 50, 100, 150, 199])
    parser.add_argument("--radii", type=float, nargs="+", default=[.06, .08, .10])
    parser.add_argument("--neighbors", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--samples", type=int, default=2048, help="target particles per frame; 0 means all soil")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="optional JSON measurement report")
    args = parser.parse_args()
    run = Run(args.data_dir, args.tag)
    if args.samples < 0 or min(args.neighbors) < 1 or min(args.radii) <= 0:
        parser.error("samples must be nonnegative; radii and neighbour counts positive")
    if min(args.frames) < 0 or max(args.frames) >= run.n_frames:
        parser.error(f"frames must be in 0..{run.n_frames-1}")
    rng = np.random.default_rng(args.seed)
    counts = {radius: [] for radius in args.radii}
    reaches = {count: [] for count in args.neighbors}
    for t in args.frames:
        pos = run.frame(t)[:, POS]
        tree = cKDTree(pos)
        n = min(args.samples or run.n_soil, run.n_soil)
        target_ids = rng.choice(run.n_soil, n, replace=False)
        for radius in counts:
            counts[radius].append(tree.query_ball_point(pos[target_ids], radius, return_length=True) - 1)
        ids, valid = nearest_neighbors(pos, target_ids, max(reaches), tree)
        distances = np.linalg.norm(pos[ids] - pos[target_ids, None], axis=-1)
        for count in reaches:
            reaches[count].extend(distances[valid[:, count-1], count-1].tolist())
    report = {"tag": run.tag, "frames": args.frames, "sampled_targets": len(args.frames) * n,
              "radius_counts": [], "knn_reach": []}
    print(f"{run.tag}: frames={args.frames}; sampled targets={report['sampled_targets']}")
    print("radius[m]   neighbours p5/p50/p95     mean")
    for radius, samples in counts.items():
        values = np.concatenate(samples)
        row = {"radius_m": radius, "p5_p50_p95": np.percentile(values, [5, 50, 95]).tolist(),
               "mean": float(values.mean()),
               "below_k_fraction": {str(k): float((values < k).mean()) for k in reaches},
               "above_k_fraction": {str(k): float((values > k).mean()) for k in reaches}}
        report["radius_counts"].append(row)
        print(f"{radius:9.3f}   {row['p5_p50_p95']}   {row['mean']:.1f}")
    print("K       distance to Kth neighbour [m]: p5/p50/p95/max")
    for count, values in reaches.items():
        quantiles = np.percentile(values, [5, 50, 95, 100]).tolist() if values else None
        report["knn_reach"].append({"k": count, "reach_m_p5_p50_p95_max": quantiles,
                                    "targets_with_k_neighbors": len(values)})
        print(f"{count:4d}    {np.round(quantiles, 5).tolist() if quantiles else 'not enough particles in frame'}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
