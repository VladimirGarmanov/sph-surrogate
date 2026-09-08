"""Сводка двух проверок одного checkpoint без повторного запуска сети."""
import argparse
import json
from pathlib import Path

import numpy as np

from .compare import load_comparison, summarize


def rms(values):
    return float(np.sqrt(np.mean(values))) if values.size else None


def mode_report(path):
    data = load_comparison(path)
    summary = summarize(data)
    names = list(data["feature_names"])
    pos = [names.index(name) for name in ("x", "y", "z")]
    vel = [names.index(name) for name in ("vx", "vy", "vz")]
    truth = data["particle_reference"].astype(np.float64)
    error = data["particle_predicted"].astype(np.float64) - truth
    pos_squared = (error[..., pos] ** 2).sum(-1)
    vel_squared = (error[..., vel] ** 2).sum(-1)
    active = np.linalg.norm(truth[..., vel], axis=-1) >= .001
    summary["active_fraction"] = float(active.mean())
    summary["active_position_rmse_m"] = rms(pos_squared[active])
    summary["active_velocity_rmse_m_s"] = rms(vel_squared[active])
    summary["per_frame"] = [
        {"frame": int(frame), "time_s": float(seconds),
         "position_rmse_m": rms(pos_squared[i]),
         "velocity_rmse_m_s": rms(vel_squared[i]),
         "active_count": int(active[i].sum()),
         "active_position_rmse_m": rms(pos_squared[i][active[i]]),
         "active_velocity_rmse_m_s": rms(vel_squared[i][active[i]])}
        for i, (frame, seconds) in enumerate(zip(data["particle_frames"], data["particle_t"]))
    ]
    with np.load(path, allow_pickle=False) as archive:
        summary["checkpoint_step"] = int(archive["checkpoint_step"])
        times = archive["step_wall"]
        summary["seconds_per_step"] = float((times[1:] if len(times) > 1 else times).mean())
        summary["profiled"] = "profile_seconds" in archive
        summary["pressure"] = {}
        for key in ("p_gt", "p_reference"):
            if key not in archive:
                continue
            predicted, reference = archive["p_pred"][1:], archive[key][1:]
            valid = np.isfinite(predicted) & np.isfinite(reference)
            relative = None
            if valid.any():
                relative = float(np.abs(predicted[valid] - reference[valid]).mean()
                                 / (np.abs(reference[valid]).mean() + 1e-9))
            summary["pressure"][key] = {"relative_error": relative,
                                       "valid_frames": int(valid.sum()), "frames": len(valid)}
    return summary, data


def fmt(value, scale=1):
    return "нет данных" if value is None else f"{value * scale:.5g}"


def build_report(directory):
    directory = Path(directory)
    reports, datasets = [], []
    for mode in ("rollout", "teacher_forced"):
        report, data = mode_report(directory / f"{mode}.npz")
        if data["mode"] != mode:
            raise ValueError(f"expected {mode}, got {data['mode']}")
        reports.append(report)
        datasets.append(data)
    for key in ("tag", "checkpoint_step"):
        if reports[0][key] != reports[1][key]:
            raise ValueError(f"cannot compare different {key}")
    for key in ("particle_frames", "particle_ids", "particle_t", "particle_reference", "feature_names"):
        if not np.array_equal(datasets[0][key], datasets[1][key]):
            raise ValueError(f"comparison requires matching {key}")
    lines = [
        "# Проверка модели грунта",
        "",
        f"Запуск: {reports[0]['tag']}. Шаг выбранного checkpoint: {reports[0]['checkpoint_step']}.",
        "Оба режима используют одни веса, кадры и ID частиц. Начальная истинная история исключена из метрик.",
        "Teacher forcing измеряет первый выход сети на истинной истории; rollout — накопление ошибок.",
        "Это проверка одного отложенного материала и короткого интервала, а не всех условий.",
        "",
        "| Показатель | Rollout | Teacher forcing |",
        "|---|---:|---:|",
    ]
    rows = [
        ("RMSE положения всех кадров, мм", lambda r: r["position"]["rmse"], 1000),
        ("RMSE положения последнего кадра, мм", lambda r: r["per_frame"][-1]["position_rmse_m"], 1000),
        ("Максимальная ошибка положения, мм", lambda r: r["position"]["abs_max"], 1000),
        ("Доля активных сравнений, %", lambda r: r["active_fraction"], 100),
        ("RMSE положения активных частиц, мм", lambda r: r["active_position_rmse_m"], 1000),
        ("RMSE скорости активных частиц, мм/с", lambda r: r["active_velocity_rmse_m_s"], 1000),
        ("Время кадра после прогрева, с", lambda r: r["seconds_per_step"], 1),
    ]
    for label, get, scale in rows:
        lines.append(f"| {label} | {fmt(get(reports[0]), scale)} | {fmt(get(reports[1]), scale)} |")
    lines += ["", "Активность определяется истинной скоростью ≥ 1 мм/с в каждом кадре.",
              "RMSE положения и скорости рассчитаны по норме трёхмерной ошибки.",
              "Время включает подготовку входов и сеть; профилирование синхронизирует GPU.",
              "", "## Давление", "",
              "Относительная ошибка = средний модуль ошибки / (средний модуль эталона + 1e-9).",
              "Начальный кадр исключён. Сравнение с p_gt отделяет ошибку сети от ошибки формулы давления.",
              "", "| Режим | Эталон | Ошибка, % | Валидные кадры |", "|---|---|---:|---:|"]
    for report in reports:
        for key, pressure in report["pressure"].items():
            lines.append(f"| {report['mode']} | {key} | {fmt(pressure['relative_error'], 100)} | "
                         f"{pressure['valid_frames']}/{pressure['frames']} |")
    for report in reports:
        lines += ["", f"## По кадрам: {report['mode']}", "",
                  "| Кадр | Время, с | RMSE положения, мм | RMSE скорости, мм/с | Активных | RMSE положения активных, мм |",
                  "|---:|---:|---:|---:|---:|---:|"]
        for row in report["per_frame"]:
            lines.append(f"| {row['frame']} | {row['time_s']:.3f} | "
                         f"{fmt(row['position_rmse_m'], 1000)} | {fmt(row['velocity_rmse_m_s'], 1000)} | "
                         f"{row['active_count']} | {fmt(row['active_position_rmse_m'], 1000)} |")
        lines += ["", f"![{report['mode']}]({report['mode']}.png)"]
    metrics = directory / "training" / "metrics.jsonl"
    if metrics.exists():
        records = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
        val = [r for r in records if r.get("split") == "VAL"]
        train = [r for r in records if r.get("split") == "TRAIN"]
        lines += ["", "## Обучение", "",
                  "TRAIN — среднее по очередному интервалу случайных пакетов с шумом; VAL — фиксированные пакеты без шума.",
                  "VAL при horizon=5 усредняет пять горизонтов. Она не равна ошибке авторегрессивной траектории."]
        if val:
            best = min(val, key=lambda r: r["mse"])
            lines += ["", f"Лучшая записанная VAL: {best['mse']:.6g}, шаг {best['step']}.",
                      "", "| Шаг | VAL MSE | Неизменная частица |", "|---:|---:|---:|"]
            lines += [f"| {r['step']} | {r['mse']:.6g} | {r['zero_delta_mse']:.6g} |" for r in val]
        if train:
            lines += ["", "Крупнейшие TRAIN-пики (средние по интервалам):", "",
                      "| Шаг | TRAIN MSE |", "|---:|---:|"]
            lines += [f"| {r['step']} | {r['mse']:.6g} |"
                      for r in sorted(train, key=lambda r: r["mse"], reverse=True)[:10]]
    lines += ["", "## Исходные файлы", "",
              "В training/ сохранены доступные config.json, metrics.jsonl, train.log и stats.npz.",
              "Путь весов записан в checkpoint_path.txt; веса и обучающий датасет в архив не включены.",
              "NPZ содержат прогнозы и эталон по всем 16 величинам; *_summary.json — сводки.",
              "REPORT.json содержит метрики этого отчёта для последующего сравнения.", ""]
    (directory / "REPORT_RU.md").write_text("\n".join(lines), encoding="utf-8")
    (directory / "REPORT.json").write_text(json.dumps(reports, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    build_report(args.directory)
    print(f"saved {Path(args.directory) / 'REPORT_RU.md'}")


if __name__ == "__main__":
    main()
