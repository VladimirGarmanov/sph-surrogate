"""Проверка всех сохранённых значений и переходов грунта без нейросети и вычислений на GPU.

    python -m surrogate.particle.audit --data_dir data         --experiment checkpoints/particle_history_k256_h8_w256         --out_dir checkpoints/particle-data-audit
"""
import argparse
import csv
import heapq
import json
from pathlib import Path

import numpy as np

from ..data import parse_tag
from .config import Config
from .data import FEATURE_NAMES, FEATURE_UNITS

THRESHOLDS = (10, 100, 1000)
INVALID_LIMIT = 1000
EVENT_FIELDS = ("tag", "split", "from_frame", "to_frame", "particle_id", "quantity", "unit",
                "before", "after", "delta", "abs_delta_over_saved_std", "zero_delta_mse")
FRAME_FIELDS = ("tag", "split", "from_frame", "to_frame", "particle_pairs", "invalid_pairs",
                "zero_delta_mse", "pairs_above_10", "pairs_above_100", "pairs_above_1000")


def optional(value):
    return float(value) if np.isfinite(value) else None


class Moments:
    """Моменты по всем конечным значениям с явным подсчётом исключённых значений."""
    def __init__(self, size=16):
        self.count = np.zeros(size, np.int64)
        self.nan = np.zeros(size, np.int64)
        self.inf = np.zeros(size, np.int64)
        self.mean = np.zeros(size, np.float64)
        self.m2 = np.zeros(size, np.float64)
        self.minimum = np.full(size, np.inf)
        self.maximum = np.full(size, -np.inf)

    def add(self, values):
        values = np.asarray(values, np.float64)
        finite = np.isfinite(values)
        part = Moments(values.shape[1])
        part.count = finite.sum(axis=0)
        part.nan = np.isnan(values).sum(axis=0)
        part.inf = np.isinf(values).sum(axis=0)
        clean = np.where(finite, values, 0.)
        part.mean = clean.sum(axis=0) / np.maximum(part.count, 1)
        centered = np.where(finite, clean - part.mean, 0.)
        part.m2 = (centered * centered).sum(axis=0)
        part.minimum = np.where(finite, values, np.inf).min(axis=0)
        part.maximum = np.where(finite, values, -np.inf).max(axis=0)
        self.merge(part)

    def merge(self, other):
        total = self.count + other.count
        difference = other.mean - self.mean
        self.m2 += other.m2 + difference ** 2 * self.count * other.count / np.maximum(total, 1)
        self.mean += difference * other.count / np.maximum(total, 1)
        self.count = total
        self.nan += other.nan
        self.inf += other.inf
        self.minimum = np.minimum(self.minimum, other.minimum)
        self.maximum = np.maximum(self.maximum, other.maximum)

    def report(self, names=FEATURE_NAMES, units=FEATURE_UNITS):
        return {name: {"unit": unit, "finite_count": int(self.count[col]),
                       "nan_count": int(self.nan[col]), "inf_count": int(self.inf[col]),
                       "min": optional(self.minimum[col]), "max": optional(self.maximum[col]),
                       "mean": float(self.mean[col]) if self.count[col] else None,
                       "std": float(np.sqrt(self.m2[col] / self.count[col])) if self.count[col] else None}
                for col, (name, unit) in enumerate(zip(names, units))}


class Targets:
    """Те же допустимые целевые кадры и строки, что в ParticleDataset, но без соседей."""
    def __init__(self, scales):
        self.scales = scales
        self.moments = Moments()
        self.pairs = self.invalid = 0
        self.zero_error_sum = 0.
        self.exceed = np.zeros((len(THRESHOLDS), 16), np.int64)
        self.energy = np.zeros_like(self.exceed, np.float64)
        self.total_energy = np.zeros(16, np.float64)
        self.pair_exceed = np.zeros(len(THRESHOLDS), np.int64)

    def add(self, delta):
        self.moments.add(delta)
        finite = np.isfinite(delta)
        complete = finite.all(axis=1)
        self.pairs += len(delta)
        self.invalid += int((~complete).sum())
        if self.scales is None:
            return None, None, complete
        # Delta / sigma — ОШИБКА предиктора, оставляющего частицу без изменений.
        # У предсказания и цели одинаковое среднее, которое сокращается при вычитании.
        normalized = np.asarray(delta, np.float64) / self.scales
        squared = np.where(finite, normalized, 0.) ** 2
        mse = np.where(complete, squared.mean(axis=1), np.nan)
        self.zero_error_sum += float(mse[complete].sum())
        self.total_energy += squared.sum(axis=0)
        for row, threshold in enumerate(THRESHOLDS):
            large = finite & (np.abs(normalized) > threshold)
            self.exceed[row] += large.sum(axis=0)
            self.energy[row] += np.where(large, squared, 0.).sum(axis=0)
            self.pair_exceed[row] += (large.any(axis=1) & complete).sum()
        return normalized, mse, complete

    def merge(self, other):
        self.moments.merge(other.moments)
        self.pairs += other.pairs
        self.invalid += other.invalid
        self.zero_error_sum += other.zero_error_sum
        self.exceed += other.exceed
        self.energy += other.energy
        self.total_energy += other.total_energy
        self.pair_exceed += other.pair_exceed

    def report(self):
        count = self.pairs - self.invalid
        columns = self.moments.report()
        if self.scales is not None:
            for col, name in enumerate(FEATURE_NAMES):
                full_std = columns[name]["std"]
                columns[name].update(saved_std=float(self.scales[col]),
                    full_std_over_saved_std=full_std / self.scales[col] if full_std is not None else None,
                    thresholds={str(t): {"count": int(self.exceed[i, col]),
                        "fraction_of_finite_values": float(self.exceed[i, col] / self.moments.count[col])
                            if self.moments.count[col] else None,
                        "fraction_of_zero_delta_squared_error": float(self.energy[i, col] / self.total_energy[col])
                            if self.total_energy[col] else None} for i, t in enumerate(THRESHOLDS)})
        return {"particle_frame_pairs": self.pairs, "invalid_pairs": self.invalid,
                "zero_delta_mse": self.zero_error_sum / count if count and self.scales is not None else None,
                "pairs_above_saved_scale": {str(t): int(self.pair_exceed[i]) for i, t in enumerate(THRESHOLDS)}
                    if self.scales is not None else None, "features": columns}


class Worst:
    """Точное хранение K наибольших значений отдельно для каждой величины и MSE каждой пары."""
    def __init__(self, limit):
        self.limit = limit
        self.heaps = {}
        self.serial = 0

    def add(self, key, score, row):
        heap = self.heaps.setdefault(key, [])
        self.serial += 1
        item = (float(score), self.serial, row)
        if len(heap) < self.limit:
            heapq.heappush(heap, item)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, item)

    def rows(self):
        return [item[2] for key in sorted(self.heaps)
                for item in sorted(self.heaps[key], reverse=True)]


def top_indices(values, count):
    scores = np.where(np.isfinite(values), values, -np.inf)
    count = min(count, len(scores))
    selected = np.argpartition(scores, len(scores) - count)[-count:]
    return selected[np.isfinite(scores[selected])]


def load_array(path, kind):
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    ndim = 2 if kind == "boundary" else 3
    if array.ndim != ndim or array.shape[-1] != 16 or any(size == 0 for size in array.shape):
        raise ValueError(f"expected nonempty {'(N,16)' if ndim == 2 else '(T,N,16)'}, got {array.shape}")
    if array.dtype.kind != "f" or array.dtype.itemsize != 4:
        raise ValueError(f"expected float32 as in the dataset specification, got {array.dtype}")
    return array


def scan_values(array, kind, tag, invalid_writer):
    moments = Moments()
    examples = 0
    nonpositive_density = 0
    locations = [{} for _ in FEATURE_NAMES]
    for frame, values in enumerate(array if array.ndim == 3 else array[None]):
        old_min, old_max = moments.minimum.copy(), moments.maximum.copy()
        moments.add(values)
        invalid = ~np.isfinite(values)
        for col in np.flatnonzero(moments.minimum < old_min):
            particle = int(np.argmin(np.where(invalid[:, col], np.inf, values[:, col])))
            locations[col].update(min_frame=frame, min_particle_id=particle)
        for col in np.flatnonzero(moments.maximum > old_max):
            particle = int(np.argmax(np.where(invalid[:, col], -np.inf, values[:, col])))
            locations[col].update(max_frame=frame, max_particle_id=particle)
        inspect = invalid.copy()
        if kind == "soil":
            bad_density = np.isfinite(values[:, 6]) & (values[:, 6] <= 0)
            nonpositive_density += int(bad_density.sum())
            inspect[:, 6] |= bad_density
        if examples < INVALID_LIMIT and inspect.any():
            rows, cols = np.nonzero(inspect)
            for particle, col in zip(rows[:INVALID_LIMIT-examples], cols[:INVALID_LIMIT-examples]):
                invalid_writer.writerow({"tag": tag, "kind": kind, "frame": frame,
                    "particle_id": int(particle), "quantity": FEATURE_NAMES[col],
                    "value": str(values[particle, col]),
                    "reason": "nonfinite" if invalid[particle, col] else "nonpositive_soil_density"})
                examples += 1
    features = moments.report()
    for name, location in zip(FEATURE_NAMES, locations):
        features[name].update(location)
    return {"shape": list(array.shape), "dtype": str(array.dtype), "features": features,
            "nonfinite_values": int((moments.nan + moments.inf).sum()),
            "nonpositive_soil_density": nonpositive_density,
            "invalid_examples_exported": examples}


def scan_transitions(soil, plate, tag, split, cfg, scales, limit, frame_writer):
    adjacent = Moments()
    motion = Moments(3)
    motion_above = {"displacement_above_spacing": 0, "kinematic_residual_above_spacing": 0}
    targets = Targets(scales)
    worst = Worst(limit)
    first = cfg.history_frames * cfg.frame_stride
    stop = min(len(soil), len(plate)) - cfg.frame_stride if plate is not None else 0
    spacing = .02  # фиксированное пространственное разрешение из NN_SPEC.md, а не порог повреждения данных
    for frame in range(len(soil) - 1):
        current = soil[frame]
        following = soil[frame + 1]
        with np.errstate(invalid="ignore", over="ignore"):
            delta = following - current
        adjacent.add(delta)
        # Скорости на концах интервала лишь приближённо описывают движение внутри него.
        # Это диагностическая эвристика, а не доказательство ошибки в ID или физике.
        with np.errstate(invalid="ignore", over="ignore"):
            dx = following[:, :3].astype(np.float64) - current[:, :3]
            velocities = .5 * (following[:, 3:6].astype(np.float64) + current[:, 3:6])
            displacement = np.linalg.norm(dx, axis=1)
            residual = np.linalg.norm(dx - cfg.dt * velocities, axis=1)
            jump = np.linalg.norm(following[:, 3:6].astype(np.float64) - current[:, 3:6], axis=1)
        motion.add(np.column_stack((displacement, jump, residual)))
        motion_above["displacement_above_spacing"] += int((np.isfinite(displacement) & (displacement > spacing)).sum())
        motion_above["kinematic_residual_above_spacing"] += int((np.isfinite(residual) & (residual > spacing)).sum())
        if frame < first or frame >= stop:
            continue
        future = soil[frame + cfg.frame_stride]
        if cfg.frame_stride != 1:
            with np.errstate(invalid="ignore", over="ignore"):
                delta = future - current
        normalized, mse, complete = targets.add(delta)
        row = {"tag": tag, "split": split, "from_frame": frame, "to_frame": frame + cfg.frame_stride,
               "particle_pairs": len(delta), "invalid_pairs": int((~complete).sum()),
               "zero_delta_mse": optional(np.mean(mse[complete])) if mse is not None and complete.any() else None}
        if normalized is not None:
            for threshold in THRESHOLDS:
                row[f"pairs_above_{threshold}"] = int(((np.abs(normalized) > threshold).any(axis=1) & complete).sum())
        frame_writer.writerow(row)
        columns = list(enumerate(FEATURE_NAMES)) + ([(None, "all16_zero_delta_mse")] if mse is not None else [])
        for col, name in columns:
            scores = mse if col is None else np.abs(delta[:, col])
            for particle in top_indices(scores, limit):
                event = {"tag": tag, "split": split, "from_frame": frame,
                    "to_frame": frame + cfg.frame_stride, "particle_id": int(particle),
                    "quantity": name, "unit": "normalized_squared" if col is None else FEATURE_UNITS[col],
                    "before": optional(current[particle, col]) if col is not None else None,
                    "after": optional(future[particle, col]) if col is not None else None,
                    "delta": optional(delta[particle, col]) if col is not None else None,
                    "abs_delta_over_saved_std": optional(abs(normalized[particle, col]))
                        if col is not None and normalized is not None else None,
                    "zero_delta_mse": optional(mse[particle]) if mse is not None else None}
                worst.add(name, scores[particle], event)
    report = {"all_adjacent_soil_deltas": adjacent.report(), "eligible_targets": targets.report(),
              "motion": motion.report(("displacement", "velocity_jump", "endpoint_kinematic_residual"),
                                       ("m", "m/s", "m")),
              "motion_inspection_counts": {"spacing_m": spacing, **motion_above},
              "eligible_start_frame": first, "eligible_stop_frame_exclusive": max(first, stop)}
    return report, targets, worst.rows()


def csv_stream(path, fields):
    stream = path.open("w", newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    return stream, writer


def readable(value, unit=""):
    factor, display = {"m": (1000., "мм"), "m/s": (1000., "мм/с"),
                       "Pa": (.001, "кПа"), "kg/m^3": (1., "кг/м³"),
                       "1": (1., "")}.get(unit, (1., unit))
    return "—" if value is None else f"{value * factor:.5g} {display}".strip()


def percentage(value):
    return "—" if value is None else f"{100 * value:.5g}%"


def write_markdown(report, out_dir):
    totals = report["totals"]
    lines = ["# Проверка грунтовых данных", "",
        f"Проверено **{totals['raw_values_scanned']:,} чисел**. Прогонов: {totals['runs_found']}. "
        f"Полных читаемых троек массивов: {totals['complete_readable_triplets']}.", "",
        "## Целостность", "",
        f"- NaN/Inf в исходных значениях: **{totals['nonfinite_values']}**.",
        f"- Неположительная плотность грунта: **{totals['nonpositive_soil_density']}**.",
        f"- Недопустимые целевые пары из-за NaN/Inf в изменениях: **{totals['invalid_eligible_target_pairs']}**.",
        f"- Замечания к формату и комплектности файлов: **{totals['format_or_coverage_issues']}**.",
        f"- Отсутствующие отложенные прогоны: **{totals['missing_holdout_runs']}**.",
        f"- Нераспознанные NPY-файлы, исключённые из проверки: **{totals['unrecognized_npy_files']}**.", "",
        "Читаются все кадры, включая несовпадающие по длине окончания грунта и плиты. "
        "Некорректные по формату файлы перечислены ниже и не включены в численные статистики. "
        "Примеры недопустимых значений с номерами кадров и частиц — в `invalid_values.csv`.", ""]
    for run in report["runs"]:
        for issue in run["issues"]:
            lines.append(f"- `{run['tag']}`: {issue}")
    if report["missing_holdout_tags"]:
        lines.append("- Отсутствуют: " + ", ".join(f"`{tag}`" for tag in report["missing_holdout_tags"]))
    if report["unrecognized_npy_files"]:
        lines.append("- Нераспознанные файлы: " + ", ".join(report["unrecognized_npy_files"]))
    lines += ["", "## Масштабы выходов по всем доступным обучающим прогонам", "",
        "В этой таблице только TRAIN и только переходы, допустимые при текущей истории и шаге кадров. "
        "Пары «частица — кадр» имеют одинаковый вес. При разном числе частиц в прогонах это отличается "
        "от обучающей выборки, в которой кадры выбираются равновероятно. "
        "Отложенные прогоны записаны отдельно в `normalization.csv` и `summary.json`.", "",
        "Отношение больше 1 означает, что полный разброс выше сохранённого масштаба. "
        "Это повод проверить представительность исходной выборки, а не доказательство порчи данных. "
        "Нулевой разброс постоянной колонки допустим.", "",
        "| Величина | Полный разброс | Сохранённый масштаб | Отношение | Доля изменений >100× | Доля ошибки от них* |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, row in report["groups"]["train"]["features"].items():
        tail = row.get("thresholds", {}).get("100", {})
        fraction = tail.get("fraction_of_finite_values")
        energy = tail.get("fraction_of_zero_delta_squared_error")
        lines.append(f"| {name} | {readable(row['std'], row['unit'])} | "
                     f"{readable(row.get('saved_std'), row['unit'])} | {readable(row.get('full_std_over_saved_std'))} | "
                     f"{percentage(fraction)} | {percentage(energy)} |")
    lines += ["", "*Ошибка рассчитана для прогноза «оставить частицу неизменной»: "
        "каждое изменение делится на сохранённый масштаб и возводится в квадрат. "
        "Это не ошибка обученной сети и не реконструкция конкретного TRAIN-пика. "
        "По каждому порогу считаются все превышения; доли не складываются, поскольку пороги вложены.", "",
        "## Самые большие изменения относительно сохранённых масштабов", ""]
    with (out_dir / "worst_targets.csv").open() as stream:
        cases = [row for row in csv.DictReader(stream) if row["abs_delta_over_saved_std"]]
    cases.sort(key=lambda row: float(row["abs_delta_over_saved_std"]), reverse=True)
    if cases:
        lines += ["| Прогон | Группа | Кадры | Частица | Величина | Изменение | Во сколько раз больше масштаба |",
                  "|---|---|---|---:|---|---:|---:|"]
        for row in cases[:10]:
            lines.append(f"| {row['tag']} | {row['split']} | {row['from_frame']} → {row['to_frame']} | "
                f"{row['particle_id']} | {row['quantity']} | {readable(float(row['delta']), row['unit'])} | "
                f"{float(row['abs_delta_over_saved_std']):.5g} |")
    else:
        lines.append("Нет конечных случаев для сравнения или не переданы сохранённые масштабы. "
                     "Изменения в физических единицах доступны в `worst_targets.csv`.")
    lines += ["", "## Как принимать решение", "",
        "1. Разобрать NaN/Inf, неположительную плотность, пропущенные файлы и несовпадения длин. "
        "Для них есть конкретные счётчики, пути и примеры.",
        "2. Проверить самые резкие переходы в исходных CSV/визуализации солвера: соседние кадры, "
        "положение частицы, контакт с плитой, скорость и напряжения. "
        "Большой скачок может быть физическим событием или артефактом расчёта/экспорта.",
        "3. Если редкие переходы корректны, учитывать их при нормализации и проверке качества "
        "следующего эксперимента. Этот отчёт не переписывает `stats.npz` и не удаляет примеры.", "",
        "Малое число NaN/Inf не гарантирует правильную физику. Единого процента «плохих данных» "
        "без физических критериев нет: отчёт разделяет нарушения формата, недопустимые значения "
        "и редкие переходы, требующие просмотра.", "",
        "NPY хранит номера строк, но не исходные ID и временные метки. По нему нельзя доказать "
        "сохранение ID, правильность единиц и отсутствие пропущенных исходных кадров. "
        "Проверка перемещения по скоростям на концах интервала — приблизительная диагностика, "
        "а не тест физической корректности.", ""]
    (out_dir / "report.md").write_text("\n".join(lines))


def audit(data_dir, out_dir, cfg, scales=None, top=10, source_experiment=None):
    """Учитываем и неполные тройки файлов; проверяем все читаемые файлы без обрезания."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    if not data_dir.is_dir():
        raise ValueError(f"data directory does not exist: {data_dir}")
    if top < 1:
        raise ValueError("top must be positive")
    if scales is not None:
        scales = np.asarray(scales, np.float64)
        if scales.shape != (16,) or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("target_std must have 16 finite, positive scales")
    inventory, unrecognized = {}, []
    for path in sorted(data_dir.glob("*.npy")):
        kind = "boundary" if path.stem.endswith("_boundary") else "plate" if path.stem.endswith("_plate") else "soil"
        tag = path.stem if kind == "soil" else path.stem[:-(len(kind) + 1)]
        try:
            parse_tag(tag)
        except ValueError:
            unrecognized.append(path.name)
            continue
        inventory.setdefault(tag, {})[kind] = path
    if not inventory:
        raise ValueError("no phi*_c* arrays found")
    out_dir.mkdir(parents=True, exist_ok=False)
    groups = {name: Targets(scales) for name in ("train", "holdout")}
    report = {"data_dir": str(data_dir.resolve()), "config": cfg.to_dict(),
              "source_experiment": str(Path(source_experiment).resolve()) if source_experiment else None,
              "saved_target_std": scales.tolist() if scales is not None else None,
              "thresholds": list(THRESHOLDS), "top_per_quantity_per_run": top,
              "missing_holdout_tags": sorted(set(cfg.holdout_tags) - set(inventory)),
              "unrecognized_npy_files": unrecognized, "runs": [],
              "limitations": ["No random sampling: all readable values and adjacent soil transitions are scanned.",
                  "Malformed/missing files are reported; statistics exclude non-finite values with explicit counts.",
                  "Training/holdout target aggregates remain separate; no normalization or data files are changed.",
                  "Pooled statistics weight particle-frame pairs equally; the sampler weights frames equally.",
                  "zero_delta_mse measures an unchanged-particle predictor, NOT the trained network's loss.",
                  "Large transitions and kinematic residuals are inspection candidates, not proven corrupt data.",
                  "particle_id is the NPY row index. Original IDs, timestamps and units cannot be verified from NPY alone.",
                  "Boundary material states are scanned as stored, although the model masks them.",
                  f"Invalid value examples are capped at {INVALID_LIMIT} per file; all invalid values are counted."]}
    streams = []
    try:
        for filename, fields in (("invalid_values.csv", ("tag", "kind", "frame", "particle_id", "quantity", "value", "reason")),
                                 ("frames.csv", FRAME_FIELDS), ("worst_targets.csv", EVENT_FIELDS)):
            stream, writer = csv_stream(out_dir / filename, fields)
            streams.append((stream, writer))
        invalid_writer, frame_writer, event_writer = [item[1] for item in streams]
        for tag, paths in sorted(inventory.items()):
            split = "holdout" if tag in cfg.holdout_tags else "train"
            entry = {"tag": tag, "split": split, "files": {}, "issues": []}
            arrays = {}
            for kind in ("soil", "plate", "boundary"):
                if kind not in paths:
                    entry["issues"].append(f"missing {kind} file")
                    continue
                try:
                    array = load_array(paths[kind], kind)
                    entry["files"][kind] = scan_values(array, kind, tag, invalid_writer)
                    arrays[kind] = array
                except (ValueError, OSError, EOFError) as error:
                    entry["issues"].append(f"{kind}: {error}")
            soil, plate = arrays.get("soil"), arrays.get("plate")
            entry["included_in_split_aggregate"] = len(arrays) == 3
            if soil is not None and plate is not None and len(soil) != len(plate):
                entry["issues"].append(f"frame count mismatch: soil={len(soil)}, plate={len(plate)}; "
                                       "training uses their common prefix; raw scans include all frames")
            if soil is not None:
                transitions, targets, events = scan_transitions(soil, plate if len(arrays) == 3 else None,
                    tag, split, cfg, scales, top, frame_writer)
                entry.update(transitions)
                if len(arrays) == 3:
                    groups[split].merge(targets)
                else:
                    entry["issues"].append("incomplete/invalid triplet excluded from training/holdout aggregates")
                event_writer.writerows(events)
                if not targets.pairs:
                    entry["issues"].append("no eligible training targets with this history/stride")
            report["runs"].append(entry)
            bad = sum(value["nonfinite_values"] for value in entry["files"].values())
            print(f"{tag} [{split}]: nonfinite={bad}, issues={len(entry['issues'])}; scanned", flush=True)
    finally:
        for stream, _ in streams:
            stream.close()
    report["groups"] = {key: value.report() for key, value in groups.items()}
    report["totals"] = {"runs_found": len(report["runs"]),
        "complete_readable_triplets": sum(run["included_in_split_aggregate"] for run in report["runs"]),
        "missing_holdout_runs": len(report["missing_holdout_tags"]),
        "unrecognized_npy_files": len(unrecognized),
        "format_or_coverage_issues": sum(len(run["issues"]) for run in report["runs"]),
        "raw_values_scanned": sum(sum(x["finite_count"] + x["nan_count"] + x["inf_count"]
            for x in value["features"].values()) for run in report["runs"] for value in run["files"].values()),
        "nonfinite_values": sum(value["nonfinite_values"] for run in report["runs"] for value in run["files"].values()),
        "invalid_eligible_target_pairs": sum(group.invalid for group in groups.values()),
        "nonfinite_adjacent_deltas": sum(feature["nan_count"] + feature["inf_count"]
            for run in report["runs"] for feature in run.get("all_adjacent_soil_deltas", {}).values()),
        "nonpositive_soil_density": sum(value["nonpositive_soil_density"] for run in report["runs"] for value in run["files"].values())}
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    stream, writer = csv_stream(out_dir / "normalization.csv", ("split", "quantity", "unit", "finite_count",
        "full_mean", "full_std", "saved_std", "full_std_over_saved_std", "above_10", "above_100", "above_1000"))
    with stream:
        for split, group in report["groups"].items():
            for name, row in group["features"].items():
                writer.writerow({"split": split, "quantity": name, "unit": row["unit"], "finite_count": row["finite_count"],
                    "full_mean": row["mean"], "full_std": row["std"], "saved_std": row.get("saved_std"),
                    "full_std_over_saved_std": row.get("full_std_over_saved_std"),
                    **{f"above_{t}": row["thresholds"][str(t)]["count"] if scales is not None else None for t in THRESHOLDS}})
    write_markdown(report, out_dir)
    print(json.dumps(report["totals"], ensure_ascii=False), flush=True)
    if scales is not None:
        print("TRAIN full std / saved std (all eligible training particle-frame pairs):")
        for name, row in report["groups"]["train"]["features"].items():
            ratio = row.get("full_std_over_saved_std")
            if ratio is not None:
                print(f"  {name:8s} {ratio:10.4g}")
    print(f"read {out_dir / 'report.md'}; details: summary.json and CSV files", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--experiment", help="directory containing the training config.json and stats.npz")
    parser.add_argument("--out_dir", default="checkpoints/particle-data-audit", help="new report directory")
    parser.add_argument("--top", type=int, default=10, help="largest cases per quantity per run; counts always cover all values")
    args = parser.parse_args()
    cfg, scales = Config(), None
    if args.experiment:
        experiment = Path(args.experiment)
        cfg = Config.from_dict(json.loads((experiment / "config.json").read_text()))
        with np.load(experiment / "stats.npz", allow_pickle=False) as stored:
            scales = stored["target_std"]
    audit(args.data_dir, args.out_dir, cfg, scales, args.top, args.experiment)


if __name__ == "__main__":
    main()
