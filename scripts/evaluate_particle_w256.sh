#!/usr/bin/env bash
# Проверка обученных на сервере весов: прогноз всего грунта, замеры времени и ошибок всех 16 величин.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ckpt="${2:-checkpoints/particle_history_k256_h8_w256_h5_noise_fullframes/best.pt}"
out_dir="${1:-checkpoints/particle-eval-fullframes-$(date +%Y%m%d-%H%M%S)}"
steps="${3:-10}"
if [[ ! -f "$ckpt" ]]; then
  echo "Trained checkpoint not found: $ckpt. Run this script on the training server." >&2
  exit 1
fi
if [[ ! -f data/phi30_c500.npy || ! -f data/phi30_c500_plate.npy || ! -f data/phi30_c500_boundary.npy ]]; then
  echo "The held-out phi30_c500 soil, plate and boundary arrays are required in data/." >&2
  exit 1
fi
# Отклоняем существующую папку результатов до открытия журналов.
mkdir -p "$(dirname "$out_dir")"
mkdir "$out_dir"
experiment_dir="$(dirname "$ckpt")"
mkdir "$out_dir/training"
for name in config.json metrics.jsonl train.log stats.npz; do
  if [[ -f "$experiment_dir/$name" ]]; then
    cp "$experiment_dir/$name" "$out_dir/training/$name"
  fi
done
printf '%s\n' "$ckpt" > "$out_dir/checkpoint_path.txt"

python -u scripts/benchmark_particle_neighbors.py \
  --data_dir data --tag phi30_c500 --frames 8 --neighbors 256 \
  --workers 1 4 --repeats 3 --out "$out_dir/neighbors.json" \
  2>&1 | tee "$out_dir/neighbors.log"

for mode in rollout teacher_forced; do
  extra=()
  if [[ "$mode" == teacher_forced ]]; then
    extra+=(--teacher_forced)
  fi
  python -u -m surrogate.particle.rollout \
    --ckpt "$ckpt" --data_dir data --tag phi30_c500 \
    --start_frame 8 --steps "$steps" --batch 32 --neighbor_workers 4 --device cuda \
    --save_particles --profile --plot "${extra[@]}" \
    --out "$out_dir/$mode.npz" \
    2>&1 | tee "$out_dir/$mode.log"

  python -u -m surrogate.particle.compare --result "$out_dir/$mode.npz" \
    2>&1 | tee "$out_dir/${mode}_comparison.log"
done

python -u -m surrogate.particle.report --directory "$out_dir"
tar -czf "$out_dir.tar.gz" -C "$(dirname "$out_dir")" "$(basename "$out_dir")"
echo "Report: $out_dir/REPORT_RU.md"
echo "Download to MacBook: $out_dir.tar.gz"
