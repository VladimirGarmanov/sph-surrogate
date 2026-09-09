"""Reproduce the numerical audit using only the exported particle rollout NPZ."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def scalar(value):
    return float(value)


def ratio(numerator, denominator):
    return scalar(numerator / denominator) if denominator else None


def rms(values):
    return scalar(np.sqrt(np.mean(np.asarray(values) ** 2)))


def quantiles(values):
    points = [0, 50, 90, 95, 99, 99.9, 100]
    return {str(q): scalar(v) for q, v in zip(points, np.percentile(values, points))}


def vector_metrics(predicted, reference):
    error = predicted - reference
    error_norm = np.linalg.norm(error, axis=-1)
    reference_norm = np.linalg.norm(reference, axis=-1)
    predicted_norm = np.linalg.norm(predicted, axis=-1)
    error_squared = error_norm ** 2
    ordered = np.sort(error_squared.ravel())
    total_squared = ordered.sum()
    return {
        "count": int(error_norm.size),
        "error_vector_rmse": rms(error_norm),
        "reference_vector_rms": rms(reference_norm),
        "predicted_vector_rms": rms(predicted_norm),
        "error_bias_components": np.mean(error.reshape(-1, error.shape[-1]), axis=0).tolist(),
        "error_norm_quantiles": quantiles(error_norm),
        "reference_norm_quantiles": quantiles(reference_norm),
        "predicted_norm_quantiles": quantiles(predicted_norm),
        "top_1pct_error_squared_share": ratio(ordered[-int(np.ceil(len(ordered) * .01)):].sum(), total_squared),
        "top_0_1pct_error_squared_share": ratio(ordered[-int(np.ceil(len(ordered) * .001)):].sum(), total_squared),
    }


def skill_metrics(predicted, reference):
    """Positive skill means smaller vector MSE than predicting a zero vector."""
    error = predicted - reference
    error_mse = np.mean(np.sum(error ** 2, axis=-1))
    zero_mse = np.mean(np.sum(reference ** 2, axis=-1))
    return {
        "count": int(reference[..., 0].size),
        "model_rmse": scalar(np.sqrt(error_mse)),
        "zero_baseline_rmse": scalar(np.sqrt(zero_mse)),
        "predicted_vector_rms": rms(np.linalg.norm(predicted, axis=-1)),
        "model_to_baseline_rmse_ratio": ratio(np.sqrt(error_mse), np.sqrt(zero_mse)),
        "mse_skill_vs_zero": scalar(1 - error_mse / zero_mse) if zero_mse else None,
    }


def direction_metrics(predicted_velocity, reference_velocity):
    reference_speed = np.linalg.norm(reference_velocity, axis=-1)
    predicted_speed = np.linalg.norm(predicted_velocity, axis=-1)
    assert (reference_speed >= .001).all()
    denominator = predicted_speed * reference_speed
    defined = denominator > 1e-15
    cosine = np.sum(predicted_velocity[defined] * reference_velocity[defined], axis=-1) / denominator[defined]
    return {
        "speed_ratio_quantiles": quantiles(predicted_speed / reference_speed),
        "direction_cosine_quantiles": quantiles(cosine),
        "opposite_direction_fraction": scalar(np.mean(cosine < 0)),
        "predicted_speed_below_1mm_s_fraction": scalar(np.mean(predicted_speed < .001)),
        "undefined_direction_count": int((~defined).sum()),
    }


def scalar_metrics(predicted, reference):
    error = predicted - reference
    reference_std = scalar(reference.std())
    predicted_std = scalar(predicted.std())
    corr = scalar(np.corrcoef(predicted.ravel(), reference.ravel())[0, 1]) if reference_std and predicted_std else None
    return {"rmse": rms(error), "mae": scalar(np.mean(np.abs(error))),
            "signed_bias": scalar(error.mean()), "reference_rms": rms(reference),
            "reference_std": reference_std, "predicted_std": predicted_std,
            "rmse_over_reference_rms": ratio(rms(error), rms(reference)),
            "correlation": corr, "max_abs_error": scalar(np.max(np.abs(error)))}


def pressure_metrics(predicted, reference):
    assert np.all(np.abs(reference) > 0)
    error = predicted - reference
    return {**scalar_metrics(predicted, reference),
            "mape_fraction": scalar(np.mean(np.abs(error) / np.abs(reference))),
            "mean_signed_relative_error_fraction": scalar(np.mean(error / reference)),
            "final_signed_error_Pa": scalar(error[-1]),
            "final_signed_relative_error_fraction": scalar(error[-1] / reference[-1])}


def outliers(scores, predicted, reference, particle_ids, frames, count=10):
    flat = scores.ravel()
    indices = np.argsort(flat)[-count:][::-1]
    rows = []
    for index in indices:
        ti, pi = np.unravel_index(index, scores.shape)
        rows.append({"frame": int(frames[ti]), "particle_id": int(particle_ids[pi]),
                     "error_norm": scalar(scores[ti, pi]),
                     "reference_position_m": reference[ti, pi, :3].tolist(),
                     "position_error_m": (predicted[ti, pi, :3] - reference[ti, pi, :3]).tolist(),
                     "reference_speed_m_s": scalar(np.linalg.norm(reference[ti, pi, 3:6])),
                     "predicted_speed_m_s": scalar(np.linalg.norm(predicted[ti, pi, 3:6]))})
    return rows


def assert_finite_json(value):
    if isinstance(value, dict):
        for item in value.values():
            assert_finite_json(item)
    elif isinstance(value, list):
        for item in value:
            assert_finite_json(item)
    elif isinstance(value, float):
        assert np.isfinite(value), value


def audit(source):
    with np.load(source, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    numeric_keys = [key for key, value in data.items() if np.issubdtype(value.dtype, np.number)]
    for key in numeric_keys:
        assert np.isfinite(data[key]).all(), f"non-finite input: {key}"
    predicted = data["particle_predicted"].astype(np.float64)
    reference = data["particle_reference"].astype(np.float64)
    frames, times, particle_ids = data["particle_frames"], data["particle_t"], data["particle_ids"]
    assert predicted.shape == reference.shape == (len(frames), len(particle_ids), 16)
    assert len(np.unique(particle_ids)) == len(particle_ids)
    assert np.array_equal(frames, data["frames"][1:])
    assert np.array_equal(times, data["t"][1:])
    assert np.all(np.diff(frames) == 1)
    assert int(data["initial_frame"]) == int(data["frames"][0]) == 100
    assert int(frames[0]) == 101 and int(frames[-1]) == 110
    np.testing.assert_allclose(times, frames * data["dt"], atol=1e-15, rtol=0)
    error = predicted - reference
    position_error = np.linalg.norm(error[..., :3], axis=-1)
    velocity_error = np.linalg.norm(error[..., 3:6], axis=-1)
    feature_rmse = np.sqrt(np.mean(error ** 2, axis=1))
    np.testing.assert_allclose(feature_rmse, data["feature_rmse"][1:], rtol=1e-12, atol=1e-12)
    calculated_rmse = np.sqrt(np.mean(position_error ** 2, axis=1))
    np.testing.assert_allclose(calculated_rmse, data["rmse"][1:], rtol=1e-12, atol=1e-15)
    speed = np.linalg.norm(reference[..., 3:6], axis=-1)
    predicted_speed = np.linalg.norm(predicted[..., 3:6], axis=-1)
    active = speed >= .001
    groups = {"all": np.ones_like(active), "active_truth_speed_ge_1mm_s": active,
              "quiet_truth_speed_lt_1mm_s": ~active}
    position, velocity = {}, {}
    for name, mask in groups.items():
        position[name] = {"pooled": vector_metrics(predicted[..., :3][mask], reference[..., :3][mask]),
                          "final": vector_metrics(predicted[-1, :, :3][mask[-1]], reference[-1, :, :3][mask[-1]])}
        velocity[name] = {
            "pooled": {**vector_metrics(predicted[..., 3:6][mask], reference[..., 3:6][mask]),
                       **skill_metrics(predicted[..., 3:6][mask], reference[..., 3:6][mask])},
            "final": {**vector_metrics(predicted[-1, :, 3:6][mask[-1]], reference[-1, :, 3:6][mask[-1]]),
                      **skill_metrics(predicted[-1, :, 3:6][mask[-1]], reference[-1, :, 3:6][mask[-1]])}}
    active_name = "active_truth_speed_ge_1mm_s"
    velocity[active_name]["pooled"].update(direction_metrics(predicted[..., 3:6][active], reference[..., 3:6][active]))
    velocity[active_name]["final"].update(direction_metrics(predicted[-1, :, 3:6][active[-1]], reference[-1, :, 3:6][active[-1]]))
    delta_predicted, delta_reference = np.diff(predicted, axis=0), np.diff(reference, axis=0)
    interval_groups = {"all": np.ones_like(active[1:]), "active_at_destination": active[1:],
                       "active_at_either_endpoint": active[:-1] | active[1:]}
    increments = {"interpretation": "Differences between stored frames 101..110 (9 intervals); each series is differenced against itself. Zero increments are a local diagnostic, not a rollout from frame 100.",
                  "interval_start_frames": frames[:-1].tolist(), "interval_end_frames": frames[1:].tolist(),
                  "position_m": {}, "velocity_m_s": {}, "stress_6vector_Pa": {}}
    for name, mask in interval_groups.items():
        for key, columns in (("position_m", slice(0, 3)), ("velocity_m_s", slice(3, 6)), ("stress_6vector_Pa", slice(7, 13))):
            increments[key][name] = skill_metrics(delta_predicted[..., columns][mask], delta_reference[..., columns][mask])
    feature_metrics = {}
    for i, name in enumerate(data["feature_names"]):
        feature_metrics[str(name)] = {
            "unit": str(data["feature_units"][i]),
            "pooled": scalar_metrics(predicted[..., i], reference[..., i]),
            "final": scalar_metrics(predicted[-1, :, i], reference[-1, :, i]),
            "active_pooled": scalar_metrics(predicted[..., i][active], reference[..., i][active]),
            "active_final": scalar_metrics(predicted[-1, :, i][active[-1]], reference[-1, :, i][active[-1]]),
            "increment_skill_vs_zero": skill_metrics(delta_predicted[..., i:i + 1], delta_reference[..., i:i + 1])}
    per_frame = []
    for i, frame in enumerate(frames):
        per_frame.append({"frame": int(frame), "time_s": scalar(times[i]), "active_count": int(active[i].sum()),
                          "active_fraction": scalar(active[i].mean()),
                          "position_rmse_m": rms(position_error[i]),
                          "active_position_rmse_m": rms(position_error[i, active[i]]),
                          "quiet_position_rmse_m": rms(position_error[i, ~active[i]]),
                          "velocity_vs_zero": skill_metrics(predicted[i, :, 3:6], reference[i, :, 3:6]),
                          "active_velocity_vs_zero": skill_metrics(predicted[i, :, 3:6][active[i]], reference[i, :, 3:6][active[i]]),
                          "p_pred_Pa": scalar(data["p_pred"][i + 1]), "p_gt_Pa": scalar(data["p_gt"][i + 1]),
                          "p_reference_Pa": scalar(data["p_reference"][i + 1])})
    per_interval = []
    for i in range(len(frames) - 1):
        per_interval.append({"from_frame": int(frames[i]), "to_frame": int(frames[i + 1]),
                             "position_vs_zero_increment": skill_metrics(delta_predicted[i, :, :3], delta_reference[i, :, :3]),
                             "velocity_vs_zero_increment": skill_metrics(delta_predicted[i, :, 3:6], delta_reference[i, :, 3:6])})
    report = {
        "source": str(source.resolve()), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "metadata": {key: data[key].item() for key in ("tag", "mode", "initial_frame", "checkpoint_step", "hidden", "neighbors", "history_frames", "batch_size", "dt")},
        "scope": {"stored_particle_frames": frames.tolist(), "stored_particle_times_s": times.tolist(),
                  "warm_start_time_s": scalar(data["t"][0]), "rollout_duration_s": scalar(times[-1] - data["t"][0]),
                  "particles": int(len(particle_ids)), "particle_frame_pairs": int(active.size),
                  "active_threshold_m_s": .001,
                  "limitations": ["No particle states for frame 100 are stored, so no equal-start frozen-particle rollout baseline can be reconstructed.",
                                  "Only 10 predictions (0.2 seconds) and one run are available; training improvement and long-horizon stability cannot be established.",
                                  "p_gt and p_reference are separate exported pressure quantities, not interchangeable ground truths.",
                                  "All per-particle aggregate errors exclude the initial warm-start frame.",
                                  "Vector RMSE is sqrt(mean(sum(error_components**2))), not the mean component RMSE."]},
        "validation": {"all_numeric_arrays_finite": True, "numeric_array_count": len(numeric_keys),
                       "particle_ids_unique": True, "stored_rmse_matches_recalculation": True,
                       "stored_feature_rmse_matches_recalculation": True},
        "activity": {"active_pairs": int(active.sum()), "active_fraction": scalar(active.mean()),
                     "quiet_pairs": int((~active).sum()), "quiet_fraction": scalar((~active).mean()),
                     "ever_active_particles": int(active.any(axis=0).sum()),
                     "never_active_particles": int((~active.any(axis=0)).sum()),
                     "never_active_particle_fraction": scalar((~active.any(axis=0)).mean()),
                     "always_active_particles": int(active.all(axis=0).sum()),
                     "quiet_predicted_active_fraction": scalar(np.mean(predicted_speed[~active] >= .001)),
                     "active_position_error_squared_share": ratio(np.sum(position_error[active] ** 2), np.sum(position_error ** 2)),
                     "active_velocity_error_squared_share": ratio(np.sum(velocity_error[active] ** 2), np.sum(velocity_error ** 2))},
        "position_m": position, "velocity_m_s": velocity, "increments": increments,
        "features": feature_metrics, "per_frame": per_frame, "per_interval": per_interval,
        "pressure_Pa": {
            "aggregation_frames": frames.tolist(), "warm_start_excluded": True,
            "predicted_vs_particle_estimator_p_gt": pressure_metrics(data["p_pred"][1:], data["p_gt"][1:]),
            "predicted_vs_p_reference": pressure_metrics(data["p_pred"][1:], data["p_reference"][1:]),
            "particle_estimator_p_gt_vs_p_reference": pressure_metrics(data["p_gt"][1:], data["p_reference"][1:]),
            "warm_start_values_Pa": {key: scalar(data[key][0]) for key in ("p_pred", "p_gt", "p_reference")}},
        "outliers": {"pooled_position_error_m": outliers(position_error, predicted, reference, particle_ids, frames),
                     "pooled_velocity_error_m_s": outliers(velocity_error, predicted, reference, particle_ids, frames),
                     "final_position_error_m": outliers(position_error[-1:], predicted[-1:], reference[-1:], particle_ids, frames[-1:]),
                     "final_velocity_error_m_s": outliers(velocity_error[-1:], predicted[-1:], reference[-1:], particle_ids, frames[-1:])},
        "timing_s": {"mean_prediction_step": scalar(data["step_wall"].mean()),
                     "median_prediction_step": scalar(np.median(data["step_wall"])),
                     "min_prediction_step": scalar(data["step_wall"].min()),
                     "max_prediction_step": scalar(data["step_wall"].max()),
                     "total_prediction": scalar(data["step_wall"].sum()),
                     "mean_evaluation_step": scalar(data["evaluation_wall"].mean())}}
    assert_finite_json(report)
    return report


def notes(report):
    all_pos = report["position_m"]["all"]
    act = "active_truth_speed_ge_1mm_s"
    active_pos = report["position_m"][act]
    all_vel = report["velocity_m_s"]["all"]
    active_vel = report["velocity_m_s"][act]
    pressure = report["pressure_Pa"]
    pg = pressure["predicted_vs_particle_estimator_p_gt"]
    pr = pressure["predicted_vs_p_reference"]
    gr = pressure["particle_estimator_p_gt_vs_p_reference"]
    dx, dv, ds = (report["increments"][key]["all"] for key in ("position_m", "velocity_m_s", "stress_6vector_Pa"))
    pos_first = report["per_frame"][0]["position_rmse_m"]
    rho = report["features"]["rho"]
    return f"""# Численный аудит preview_k256.npz — step 191

Источник: `{report['source']}`. SHA-256: `{report['sha256']}`.
Проверка: все числовые массивы конечны; ID уникальны; пересчитанные позиционные и покомпонентные RMSE совпали с экспортом.

## Что именно проверено

Run `{report['metadata']['tag']}`, K={report['metadata']['neighbors']}, история центра 8+1; 46 464 частицы, 10 прогнозов: кадры 101–110, t=2.02–2.20 с. Прогноз стартовал от истинного кадра 100 (2.00 с). Его состояния частиц в файле нет. Все агрегаты ниже исключают warm start.
Активность определяется только истинной скоростью: ||v_true|| ≥ 1 мм/с. Векторный RMSE = sqrt(mean(sum(error_xyz²))).

## Движение есть, но малый общий RMSE скрывает ошибки активной области

- Доля малоподвижных particle-frame pairs: {report['activity']['quiet_fraction']:.4%}; ни разу не пересекли порог {report['activity']['never_active_particles']:,} из 46 464 частиц ({report['activity']['never_active_particle_fraction']:.3%}).
- RMSE положения: первый прогноз {pos_first * 1000:.4f} мм, pooled {all_pos['pooled']['error_vector_rmse'] * 1000:.4f} мм, финальный {all_pos['final']['error_vector_rmse'] * 1000:.4f} мм. Для активных: pooled {active_pos['pooled']['error_vector_rmse'] * 1000:.4f} мм, финальный {active_pos['final']['error_vector_rmse'] * 1000:.4f} мм.
- Активная область даёт {report['activity']['active_position_error_squared_share']:.3%} суммы квадратов позиционной ошибки и {report['activity']['active_velocity_error_squared_share']:.3%} скоростной ошибки.
- Среди активных медиана |v_pred|/|v_true| = {active_vel['pooled']['speed_ratio_quantiles']['50']:.4f}, медиана cos направления = {active_vel['pooled']['direction_cosine_quantiles']['50']:.4f}; противоположное полупространство направления у {active_vel['pooled']['opposite_direction_fraction']:.3%}. Это не модель с нулевой скоростью повсюду.
- RMS ошибки скорости pooled = {all_vel['pooled']['error_vector_rmse'] * 1000:.4f} мм/с против {all_vel['pooled']['zero_baseline_rmse'] * 1000:.4f} мм/с у v=0: выигрыш {all_vel['pooled']['mse_skill_vs_zero']:.3%} по MSE. На финальном кадре RMS ошибки {all_vel['final']['error_vector_rmse'] * 1000:.4f} мм/с против {all_vel['final']['zero_baseline_rmse'] * 1000:.4f} у v=0; ошибка MSE уже на {-all_vel['final']['mse_skill_vs_zero']:.3%} выше нулевого baseline. В активной области финальный RMSE {active_vel['final']['error_vector_rmse'] * 1000:.4f} мм/с против {active_vel['final']['zero_baseline_rmse'] * 1000:.4f} у v=0.
- Финальный median/P99/max ошибки положения: {all_pos['final']['error_norm_quantiles']['50'] * 1000:.4f}/{all_pos['final']['error_norm_quantiles']['99'] * 1000:.4f}/{all_pos['final']['error_norm_quantiles']['100'] * 1000:.4f} мм. Худшая частица: ID {report['outliers']['final_position_error_m'][0]['particle_id']}. Верхний 1% ошибок по pooled данным даёт {all_pos['pooled']['top_1pct_error_squared_share']:.3%} позиционной суммы квадратов; эти выбросы нельзя скрывать одной усреднённой кривой.

## Локальная динамика между доступными кадрами 101–110

Каждая серия вычитается сама из себя: Δpred = pred[t]−pred[t−1], Δtrue = true[t]−true[t−1]. Всего 9 интервалов. Это корректная диагностика приращений, но не замена равноправному rollout-baseline от кадра 100.

- Δпозиции: RMS ошибки {dx['model_rmse'] * 1000:.5f} мм/интервал против {dx['zero_baseline_rmse'] * 1000:.5f} у нулевого приращения; выигрыш {dx['mse_skill_vs_zero']:.3%} по MSE. Модель действительно сообщает полезную информацию о перемещении.
- Δскорости: RMS собственных обновлений модели {dv['predicted_vector_rms'] * 1000:.5f} мм/с против истинного RMS {dv['zero_baseline_rmse'] * 1000:.5f} мм/с; амплитуда обновлений {dv['predicted_vector_rms'] / dv['zero_baseline_rmse']:.3%} от истины. Ошибка обновления {dv['model_rmse'] * 1000:.5f} мм/с, MSE на {-dv['mse_skill_vs_zero']:.3%} хуже нулевого обновления.
- Шесть компонент напряжения p11,p22,p33,shear12,shear13,shear23: RMS ошибки приращения {ds['model_rmse']:.2f} Па против {ds['zero_baseline_rmse']:.2f} Па у нулевого приращения; MSE на {-ds['mse_skill_vs_zero']:.3%} хуже. Абсолютный пространственный профиль напряжений может выглядеть близким при слабом предсказании его изменения.

Вывод: здесь видно движение, но по одному раннему checkpoint нельзя приписать его обученной динамике. Модель слабо обновляет скорости на этом окне, и сохранение начального состояния/движения остаётся важным объяснением внешне хорошей картины. Нужен baseline с тем же истинным кадром 100 (файл его не содержит) и более длинный rollout.

## Напряжения и давление

| Компонента | pooled RMSE, кПа | финальный RMSE, кПа | pooled signed bias, кПа |
|---|---:|---:|---:|
""" + "\n".join(f"| {name} | {report['features'][name]['pooled']['rmse']/1000:.4f} | {report['features'][name]['final']['rmse']/1000:.4f} | {report['features'][name]['pooled']['signed_bias']/1000:+.4f} |" for name in ("p11", "p22", "p33", "shear12", "shear13", "shear23")) + f"""

- `rho` константна 1600 кг/м³, `pc`, `Ev`, `Sv` равны нулю и у модели, и у истины. Их нулевая ошибка не свидетельствует об обучении динамике.
- p_pred против p_gt: MAE {pg['mae']/1000:.4f} кПа, RMSE {pg['rmse']/1000:.4f} кПа, signed bias {pg['signed_bias']/1000:+.4f} кПа, MAPE {pg['mape_fraction']:.3%}. Финальная разность {pg['final_signed_error_Pa']/1000:+.4f} кПа.
- p_pred против p_reference: MAE {pr['mae']/1000:.4f} кПа, signed bias {pr['signed_bias']/1000:+.4f} кПа, MAPE {pr['mape_fraction']:.3%}; во всех прогнозируемых кадрах завышение. Финальная разность {pr['final_signed_error_Pa']/1000:+.4f} кПа.
- Уже p_gt против p_reference имеет signed bias {gr['signed_bias']/1000:+.4f} кПа и MAPE {gr['mape_fraction']:.3%}. Поэтому совпадение p_pred с p_gt на ≈1% нельзя выдавать за точность итогового давления относительно p_reference. p_reference не обозначено здесь как независимый лабораторный эксперимент.

Время сети с подготовкой входов: {report['timing_s']['mean_prediction_step']:.3f} с/кадр, всего {report['timing_s']['total_prediction']:.2f} с для 0.2 с моделирования. Полного solver wall-time в NPZ нет, ускорение относительно решателя этим файлом не проверено.

Полные значения, ошибки по кадрам, активным/малоподвижным группам и ID выбросов: `audit_metrics.json`.
Воспроизведение: `.venv/bin/python output/preview_k256_step191/audit_metrics.py`.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[2] / "checkpoints" / "preview_k256.npz")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    report = audit(args.source)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "audit_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (args.out_dir / "AUDIT_NOTES.md").write_text(notes(report))
    print(json.dumps({"validation": report["validation"], "quiet_fraction": report["activity"]["quiet_fraction"],
                      "final_position_rmse_mm": report["position_m"]["all"]["final"]["error_vector_rmse"] * 1000,
                      "position_increment_mse_skill": report["increments"]["position_m"]["all"]["mse_skill_vs_zero"],
                      "velocity_increment_mse_skill": report["increments"]["velocity_m_s"]["all"]["mse_skill_vs_zero"]}, indent=2))


if __name__ == "__main__":
    main()
