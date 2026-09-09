#!/usr/bin/env bash
# Запускать на свободном GPU сервера: веса не обновляются.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ckpt="${1:-checkpoints/particle_history_k256_h8_w256_h5_noise_fullframes/model.pt}"
out_dir="checkpoints/training-speed-$(date +%Y%m%d-%H%M%S)"
mkdir -p checkpoints
mkdir "$out_dir"
python -u -m surrogate.particle.benchmark_training \
  --ckpt "$ckpt" --targets 512 --batches 32 64 128 256 --repeats 3 \
  --out "$out_dir/report.json" 2>&1 | tee "$out_dir/benchmark.log"
