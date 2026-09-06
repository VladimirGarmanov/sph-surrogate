"""Step 1 of NN_SPEC: mean neighbour count for a few radii.

    python scripts/neighbors.py --tag phi35_c1000 --frame 100 --radii 0.03 0.04 0.05 0.06

Target is 30-60 neighbours for a soil particle. The `all` column counts soil +
plate + wall markers (what the net will actually see), `soil` counts soil only.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surrogate.data import Run, POS, SOIL          # noqa: E402
from surrogate.graph import neighbor_counts       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--frame", type=int, default=100)
    ap.add_argument("--radii", type=float, nargs="+", default=[0.03, 0.04, 0.05, 0.06])
    args = ap.parse_args()

    run = Run(args.data_dir, args.tag)
    f = run.frame(args.frame)
    pos = f[:, POS]
    soil = run.types == SOIL
    print(f"{run}: frame {args.frame}, soil={run.n_soil} plate={run.n_plate} wall={run.n_wall}")
    print(f"{'radius':>7} {'r/spacing':>9} | {'all: mean':>9} {'p5':>5} {'p95':>5} | {'soil: mean':>10} {'p5':>5} {'p95':>5}")
    for r in args.radii:
        c_all = neighbor_counts(pos[soil], pos, r) - 1
        c_soil = neighbor_counts(pos[soil], pos[soil], r) - 1
        print(f"{r:7.3f} {r / 0.02:9.1f} | {c_all.mean():9.1f} {np.percentile(c_all, 5):5.0f} {np.percentile(c_all, 95):5.0f}"
              f" | {c_soil.mean():10.1f} {np.percentile(c_soil, 5):5.0f} {np.percentile(c_soil, 95):5.0f}")


if __name__ == "__main__":
    main()
