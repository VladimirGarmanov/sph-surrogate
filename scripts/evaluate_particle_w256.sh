#!/usr/bin/env bash
# Evaluate trained server weights: full-soil rollout, timing, then all 16 errors.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ckpt="checkpoints/particle_history_k256_h8_w256/best.pt"
out_dir="${1:-checkpoints/particle-eval-w256}"
if [[ ! -f "$ckpt" ]]; then
  echo "Trained checkpoint not found: $ckpt. Run this script on the training server." >&2
  exit 1
fi
if [[ ! -f data/phi30_c500.npy || ! -f data/phi30_c500_plate.npy || ! -f data/phi30_c500_boundary.npy ]]; then
  echo "The held-out phi30_c500 soil, plate and boundary arrays are required in data/." >&2
  exit 1
fi
# Refuse an old result directory before opening logs.
mkdir -p "$(dirname "$out_dir")"
mkdir "$out_dir"

python -u scripts/benchmark_particle_neighbors.py \
  --data_dir data --tag phi30_c500 --frames 8 --neighbors 256 \
  --workers 1 4 --repeats 3 --out "$out_dir/neighbors.json" \
  2>&1 | tee "$out_dir/neighbors.log"

python -u -m surrogate.particle.rollout \
  --ckpt "$ckpt" --data_dir data --tag phi30_c500 \
  --start_frame 8 --steps 10 --batch 32 --neighbor_workers 4 --device cuda \
  --save_particles --profile --plot \
  --out "$out_dir/rollout.npz" \
  2>&1 | tee "$out_dir/rollout.log"

python -u -m surrogate.particle.compare --result "$out_dir/rollout.npz" \
  2>&1 | tee "$out_dir/comparison.log"

echo "Read $out_dir/comparison.log for model errors and $out_dir/rollout.log for timings."
