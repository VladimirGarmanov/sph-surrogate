"""Поиск формулы давления под штампом, согласующейся с эталонной кривой решателя.
Используются ТОЛЬКО истинные данные симуляции, без нейросети. Так можно отдельно
оценить точность формулы и точность сети: одно число в rollout.py смешивало эти ошибки.

    python scripts/calibrate_pressure.py --tags phi30_c500,phi35_c1000,phi40_c1000

Выводит среднюю относительную ошибку по сравнению с pressure_sinkage.csv для
каждого сочетания толщины слоя, радиуса и взвешивания по всем указанным запускам
и всем их кадрам.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surrogate.data import Run, POS, P33  # noqa: E402

SPACING = 0.02   # одинаково для всех запусков согласно NN_SPEC.md


def estimate(soil_pos, soil_p33, plate_pos, spacing, radius, slab_k, weighted):
    center = plate_pos[:, :2].mean(0)
    z_bottom = plate_pos[:, 2].min()
    r = np.linalg.norm(soil_pos[:, :2] - center, axis=1)
    layer = (r <= radius) & (soil_pos[:, 2] <= z_bottom) & (soil_pos[:, 2] >= z_bottom - slab_k * spacing)
    if not layer.any():
        return np.nan
    if weighted:
        # sum(stress * particle_area) / plate_area отличается от обычного среднего, только если
        # частицы неравномерно покрывают заданный диск, например у его края реальных соседей меньше
        area = np.pi * radius ** 2
        return -float(soil_p33[layer].sum()) * spacing ** 2 / area
    return -float(soil_p33[layer].mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--tags", required=True, help="comma-separated run tags")
    ap.add_argument("--every", type=int, default=5, help="use every Nth frame (speed)")
    args = ap.parse_args()

    tags = args.tags.split(",")
    radii = [0.15, 0.15 + 0.02, 0.15 + 2 * 0.02]
    slabs = [1, 2, 3, 4, 6]
    weighted_opts = [False, True]

    results = {}
    for tag in tags:
        run = Run(args.data_dir, tag)
        if run.reference is None:
            print(f"  {tag}: no pressure_sinkage.csv, skipping")
            continue
        ref = run.reference
        ref_t = ref["time_s"].to_numpy() - ref["time_s"].iloc[0]
        ref_p = ref["pressure_Pa"].to_numpy()
        frames = range(1, run.n_frames, args.every)
        t = np.array([f * 0.02 for f in frames])
        p_ref = np.interp(t, ref_t, ref_p)
        for radius in radii:
            for slab_k in slabs:
                for weighted in weighted_opts:
                    key = (radius, slab_k, weighted)
                    errs = results.setdefault(key, [])
                    for f in frames:
                        gt = np.asarray(run.soil[f])
                        plate_pos = np.asarray(run.plate[f][:, POS])
                        est = estimate(gt[:, POS], gt[:, P33], plate_pos, SPACING, radius, slab_k, weighted)
                        i = list(frames).index(f)
                        if p_ref[i] > 1e3 and np.isfinite(est):   # пропускаем начало с близкими к нулю значениями, где преобладает шум
                            errs.append(abs(est - p_ref[i]) / abs(p_ref[i]))
        print(f"  {tag}: done ({len(list(frames))} frames)")

    print(f"\n{'radius':>7} {'slab(x spacing)':>16} {'weighted':>9}   mean rel. error")
    ranked = sorted(results.items(), key=lambda kv: np.mean(kv[1]) if kv[1] else 1e9)
    for (radius, slab_k, weighted), errs in ranked:
        if not errs:
            continue
        print(f"{radius:7.3f} {slab_k:16d} {str(weighted):>9}   {np.mean(errs):.1%}  (n={len(errs)})")


if __name__ == "__main__":
    main()
