#!/usr/bin/env bash
# K=32 и 3 кадра истории соседа вместо 256 и 9: в 24 раза меньше шагов истории соседей.
# Центральная частица сохраняет историю 8+1. Ускорение всего кадра ещё не измерено.
# Остальные настройки совпадают с train_particle_w256.sh для сравнения экспериментов.
# Запуск из существующего окружения Python на сервере с CUDA и всеми 25 запусками.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
out_dir="checkpoints/particle_k32_h3"
epochs="${1:-1}"

mkdir -p checkpoints
# Не допускаем повторный запуск в прежней папке, чтобы tee не перезаписал журнал обучения.
mkdir "$out_dir"

python -u -m surrogate.particle.train \
  --data_dir data \
  --out_dir "$out_dir" \
  --holdout phi30_c500,phi40_c2000,phi25_c5000,phi45_c0 \
  --neighbors 32 \
  --history_frames 8 \
  --neighbor_history_frames 3 \
  --prediction_horizon 5 \
  --frame_stride 1 \
  --dt 0.02 \
  --batch 32 \
  --hidden 256 \
  --training_mode full_frames \
  --epochs "$epochs" \
  --lr 1e-4 \
  --lr_decay_steps 50000 \
  --input_noise_std 0.05 \
  --stats_frames 40 \
  --val_samples 16 \
  --log_every 1 \
  --val_every 25 \
  --seed 0 \
  --device cuda \
  2>&1 | tee "$out_dir/train.log"
