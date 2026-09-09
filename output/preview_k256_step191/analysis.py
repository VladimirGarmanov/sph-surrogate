"""Reproduce standalone figures for the downloaded step-191 rollout; no model execution."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / "checkpoints/preview_k256.npz"
MODEL, TRUTH, BASE, ACTIVE, INACTIVE = "#e16b32", "#207b9b", "#657180", "#8b4d9c", "#8c969f"
TEXT, GRID = "#24323f", "#dce3e8"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 14,
    "axes.titleweight": "bold", "axes.titlepad": 13, "axes.labelsize": 11,
    "axes.labelcolor": TEXT, "text.color": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "axes.edgecolor": "#9ba8b3", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": .7,
    "figure.facecolor": "#f8fafb", "axes.facecolor": "white", "lines.linewidth": 2.4,
    "legend.frameon": False, "legend.fontsize": 10, "savefig.dpi": 190,
    "pdf.fonttype": 42, "axes.formatter.useoffset": False,
})


def vrms(x, mask=None):
    squared = np.sum(x * x, axis=-1)
    if mask is not None:
        squared = squared[mask]
    return float(np.sqrt(np.mean(squared))) if squared.size else None


def ecdf(ax, values, color, label):
    x = np.sort(values)
    ax.plot(x, np.arange(1, len(x) + 1) / len(x) * 100, color=color, label=label)


def decorate(fig, title, subtitle, footer):
    fig.suptitle(title, x=.07, y=.978, ha="left", fontsize=24, fontweight="bold")
    fig.text(.07, .932, subtitle, fontsize=11, color=BASE)
    fig.text(.07, .018, footer, fontsize=10, color=BASE, va="bottom")


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", facecolor=fig.get_facecolor())
    fig.savefig(OUT / f"{stem}.pdf", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    with np.load(SOURCE, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    p, r = (data[key].astype(np.float64) for key in ("particle_predicted", "particle_reference"))
    assert p.shape == r.shape == (10, 46464, 16)
    assert np.isfinite(p).all() and np.isfinite(r).all()
    assert len(np.unique(data["particle_ids"])) == r.shape[1]
    err = p - r
    speed = np.linalg.norm(r[..., 3:6], axis=-1) * 1000
    predicted_speed = np.linalg.norm(p[..., 3:6], axis=-1) * 1000
    active = speed >= 1
    pos_error = np.linalg.norm(err[..., :3], axis=-1) * 1000
    per_frame = []
    for i, frame in enumerate(data["particle_frames"]):
        per_frame.append({
            "frame": int(frame), "time_s": float(data["particle_t"][i]),
            "position_rmse_mm": vrms(err[i, :, :3]) * 1000,
            "active_position_rmse_mm": vrms(err[i, :, :3], active[i]) * 1000,
            "inactive_position_rmse_mm": vrms(err[i, :, :3], ~active[i]) * 1000,
            "velocity_rmse_mm_s": vrms(err[i, :, 3:6]) * 1000,
            "active_velocity_rmse_mm_s": vrms(err[i, :, 3:6], active[i]) * 1000,
            "active_zero_velocity_rmse_mm_s": vrms(r[i, :, 3:6], active[i]) * 1000,
            "active_count": int(active[i].sum()),
            "position_p95_mm": float(np.percentile(pos_error[i], 95)),
            "position_max_mm": float(pos_error[i].max()),
            "position_above_10mm": int((pos_error[i] > 10).sum()),
            "wall_seconds": float(data["step_wall"][i]),
        })
    np.testing.assert_allclose([row["position_rmse_mm"] / 1000 for row in per_frame], data["rmse"][1:])
    np.testing.assert_allclose(np.sqrt(np.mean(err ** 2, axis=1)), data["feature_rmse"][1:])
    dt = np.diff(data["particle_t"])
    assert np.all(dt > 0)
    dr, dp = np.diff(r, axis=0), np.diff(p, axis=0)
    increments = []
    for i in range(len(dr)):
        increments.append({
            "from_frame": int(data["particle_frames"][i]),
            "to_frame": int(data["particle_frames"][i + 1]),
            "time_s": float(data["particle_t"][i + 1]),
            "position_increment_error_mm": vrms(dp[i, :, :3] - dr[i, :, :3]) * 1000,
            "zero_position_increment_error_mm": vrms(dr[i, :, :3]) * 1000,
            "true_velocity_increment_rms_mm_s": vrms(dr[i, :, 3:6]) * 1000,
            "predicted_velocity_increment_rms_mm_s": vrms(dp[i, :, 3:6]) * 1000,
        })
    pressure = {}
    for name, left, right in (("model_vs_proxy", "p_pred", "p_gt"),
                              ("model_vs_solver", "p_pred", "p_reference"),
                              ("proxy_vs_solver", "p_gt", "p_reference")):
        a, b = data[left][1:], data[right][1:]
        assert np.isfinite(a).all() and np.isfinite(b).all()
        pressure[name] = {"relative_mae_percent": float(np.abs(a-b).mean() / np.abs(b).mean() * 100),
                          "mae_kpa": float(np.abs(a-b).mean() / 1000),
                          "bias_kpa": float((a-b).mean() / 1000)}
    last_active = active[-1]
    summary = {
        "source": str(SOURCE.relative_to(ROOT)), "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "checkpoint_step": int(data["checkpoint_step"]), "neighbors": int(data["neighbors"]),
        "tag": str(data["tag"]), "frames": data["particle_frames"].tolist(),
        "initial_frame": int(data["initial_frame"]), "duration_s": float(data["t"][-1]-data["t"][0]),
        "particles_per_frame": r.shape[1], "active_threshold_mm_s": 1,
        "active_fraction_percent": float(active.mean()*100),
        "final_position_percentiles_mm": dict(zip(("p50", "p95", "p99", "max"),
                                                   map(float, np.percentile(pos_error[-1], [50,95,99,100])))),
        "final_position_above_10mm": int((pos_error[-1] > 10).sum()),
        "final_velocity_zero_baseline_skill_percent": float((1-np.sum(err[-1,:,3:6]**2)/np.sum(r[-1,:,3:6]**2))*100),
        "final_active_velocity_zero_baseline_skill_percent": float((1-np.sum(err[-1,last_active,3:6]**2)/np.sum(r[-1,last_active,3:6]**2))*100),
        "final_inactive_speed_median_mm_s": {
            "truth": float(np.median(speed[-1,~last_active])),
            "model": float(np.median(predicted_speed[-1,~last_active]))},
        "position_increment_zero_baseline_skill_percent": float((1-np.sum((dp[:,:,:3]-dr[:,:,:3])**2)/np.sum(dr[:,:,:3]**2))*100),
        "velocity_increment_zero_baseline_skill_percent": float((1-np.sum((dp[:,:,3:6]-dr[:,:,3:6])**2)/np.sum(dr[:,:,3:6]**2))*100),
        "velocity_increment_rms_mm_s": {"truth": vrms(dr[:,:,3:6])*1000, "model": vrms(dp[:,:,3:6])*1000},
        "mean_seconds_per_frame_excluding_first": float(data["step_wall"][1:].mean()),
        "compute_seconds_total": float(data["step_wall"].sum()), "profiled": "profile_seconds" in data,
        "solver_seconds_per_frame_context": 4.72,
        "solver_timing_source": "Rounded prior solver timing supplied in the conversation, not present in this NPZ",
        "pressure": pressure, "per_frame": per_frame, "increments": increments,
        "constant_exact_features": [str(data["feature_names"][k]) for k in range(16)
                                    if np.ptp(r[:,:,k]) == 0 and np.max(np.abs(err[:,:,k])) == 0],
        "limitations": ["true initial particle frame 100 is absent; no frozen-state rollout baseline from frame 100",
                        "increment baseline compares predicted and true frame-to-frame changes 101 to 110, not full rollout",
                        "no paired teacher-forced run at the same checkpoint and time window",
                        "one material, 10 frames, 0.2 seconds, warm start at 2 seconds"]}
    (OUT / "summary_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    with (OUT / "per_frame.csv").open("w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(per_frame[0])); writer.writeheader(); writer.writerows(per_frame)
    with (OUT / "features.csv").open("w", newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["feature","unit","pooled_rmse","final_rmse","final_bias","final_absolute_p95","final_absolute_max"])
        for k, name in enumerate(data["feature_names"]):
            writer.writerow([name,data["feature_units"][k],np.sqrt(np.mean(err[:,:,k]**2)),
                             np.sqrt(np.mean(err[-1,:,k]**2)),err[-1,:,k].mean(),
                             np.percentile(np.abs(err[-1,:,k]),95),np.abs(err[-1,:,k]).max()])

    t = data["particle_t"]
    # Overview: physical units, common masks and no initial zero-error frame in metrics.
    fig, axes = plt.subplots(2,2,figsize=(15,10))
    fig.subplots_adjust(left=.075,right=.96,bottom=.12,top=.84,wspace=.28,hspace=.45)
    decorate(fig, "Модель держит среднюю картину — локальные ошибки растут",
             "K256 · шаг обучения 191 · phi30_c500 · 46 464 частицы · прогноз 2,02–2,20 с",
             "Подвижные частицы: истинная скорость ≥ 1 мм/с в каждом кадре. Начальная история до 2,00 с дана солвером.\n"
             "Ошибка давления = средний модуль ошибки / средний модуль эталона. Один короткий rollout, без teacher forcing.")
    ax=axes[0,0]
    for key,label,color in (("active_position_rmse_mm","Подвижные",ACTIVE),
                            ("position_rmse_mm","Все частицы",MODEL),
                            ("inactive_position_rmse_mm","Фон < 1 мм/с",INACTIVE)):
        values=[row[key] for row in per_frame]
        ax.plot(t,values,"o-",ms=4,color=color,label=f"{label}: {values[-1]:.3f} мм")
    ax.set(title="01  Ошибка положения растёт",xlabel="Время солвера, с",ylabel="Векторная RMSE положения, мм",ylim=(0,None))
    ax.legend(loc="upper left")
    ax=axes[0,1]
    ax.plot(data["t"],data["p_reference"]/1000,"o-",color=TRUTH,ms=4,label="Эталон солвера (CSV)")
    ax.plot(data["t"],data["p_gt"]/1000,"--",color=BASE,label="Оценка по истинным частицам")
    ax.plot(data["t"],data["p_pred"]/1000,"o-",color=MODEL,ms=4,label="Оценка по прогнозу модели")
    ax.set(title="02  Давление: близко к оценке, далеко от CSV",xlabel="Время солвера, с",ylabel="Давление, кПа",ylim=(35,65))
    ax.legend(loc="lower left",fontsize=9)
    ax.text(.98,.96,"1,27% к оценке · 28,43% к CSV",ha="right",va="top",transform=ax.transAxes,fontsize=10,fontweight="bold")
    ax=axes[1,0]
    ax.plot(t,[row["active_zero_velocity_rmse_mm_s"] for row in per_frame],"--",color=BASE,label="Контроль: скорость = 0")
    ax.plot(t,[row["active_velocity_rmse_mm_s"] for row in per_frame],"o-",color=MODEL,ms=4,label="Модель")
    ax.set(title="03  Скорости подвижных частиц",xlabel="Время солвера, с",ylabel="Векторная RMSE скорости, мм/с",ylim=(0,None))
    ax.legend(loc="lower right")
    ax=axes[1,1]
    ecdf(ax,pos_error[-1],MODEL,"Все 46 464 частицы")
    ecdf(ax,pos_error[-1,last_active],ACTIVE,f"Подвижные: {last_active.sum():,}".replace(","," "))
    ax.set_xscale("log")
    ax.axvline(10,color=BASE,lw=1,ls="--")
    ax.set(title="04  Средняя ошибка скрывает редкие выбросы",xlabel="Ошибка положения на кадре 110, мм (логарифм)",ylabel="Доля частиц с ошибкой не выше X, %",ylim=(0,101))
    ax.legend(loc="lower right")
    ax.text(.96,.48,f"Максимум: {pos_error[-1].max():.1f} мм\nОшибка > 10 мм: {(pos_error[-1]>10).sum()} частиц",
            transform=ax.transAxes,va="top",ha="right",fontsize=10)
    for ax in axes.flat:
        if ax.get_xscale()=="linear": ax.xaxis.set_major_locator(MaxNLocator(5))
    save(fig,"overview")

    fig,axes=plt.subplots(2,2,figsize=(15,10))
    fig.subplots_adjust(left=.075,right=.96,bottom=.12,top=.84,wspace=.28,hspace=.45)
    decorate(fig,"Траектории движутся, но скорость почти не обновляется",
             "Контроли рассчитаны по сохранённым частицам · скорость запуска измерена без --profile",
             "Приращения: девять переходов 101→102, …, 109→110. Контроль Δx = 0 не является полным rollout от кадра 100.\n"
             "Время солвера ≈ 4,72 с/кадр взято из предыдущего отчёта; в этом NPZ оно не сохранено.")
    times=[row["time_s"] for row in increments]
    ax=axes[0,0]
    ax.plot(times,[row["zero_position_increment_error_mm"] for row in increments],"--",color=BASE,label="Контроль: Δx = 0")
    ax.plot(times,[row["position_increment_error_mm"] for row in increments],"o-",color=MODEL,ms=4,label="Ошибка приращения модели")
    ax.set(title="01  Приращения положения лучше нулевых",xlabel="Конец интервала, с",ylabel="Векторная RMSE приращения, мм",ylim=(0,None))
    ax.legend(loc="upper left")
    ax.text(.98,.06,f"MSE ниже на {summary['position_increment_zero_baseline_skill_percent']:.1f}%",ha="right",transform=ax.transAxes,fontweight="bold",fontsize=11)
    ax=axes[0,1]
    ax.plot(times,[row["true_velocity_increment_rms_mm_s"] for row in increments],"o-",color=TRUTH,ms=4,label="Солвер: изменение скорости")
    ax.plot(times,[row["predicted_velocity_increment_rms_mm_s"] for row in increments],"o-",color=MODEL,ms=4,label="Модель: изменение скорости")
    ax.set(title="02  Модель почти сохраняет прежнюю скорость",xlabel="Конец интервала, с",ylabel="Векторный RMS изменения скорости, мм/с",ylim=(0,None))
    ax.legend(loc="upper left")
    ax=axes[1,0]
    ecdf(ax,speed[-1,~last_active],TRUTH,"Солвер")
    ecdf(ax,predicted_speed[-1,~last_active],MODEL,"Модель")
    ax.set_xscale("log")
    ax.axvline(1,lw=1,ls="--",color=BASE)
    ax.set(title="03  На спокойном фоне появляется лишняя скорость",xlabel="Модуль скорости, мм/с (логарифм)",ylabel="Доля частиц со скоростью не выше X, %",ylim=(0,101))
    ax.legend(loc="upper left")
    med=summary["final_inactive_speed_median_mm_s"]
    ax.text(.04,.69,f"Медиана на кадре 110:\nсолвер {med['truth']:.3f} · модель {med['model']:.3f} мм/с",
            transform=ax.transAxes,fontsize=10,va="top")
    ax=axes[1,1]
    ax.bar(np.arange(1,11),data["step_wall"],color=MODEL,width=.66)
    ax.axhline(4.72,color=TRUTH,lw=2,label="Солвер ≈ 4,72 с/кадр")
    mean=summary["mean_seconds_per_frame_excluding_first"]
    ax.set(title="04  K256 остаётся медленнее солвера",xlabel="Шаг самостоятельного прогноза",ylabel="Время расчёта одного кадра, с",ylim=(0,70),xticks=np.arange(1,11))
    ax.text(.5,.94,f"Среднее: {mean:.2f} с/кадр · ≈ {mean/4.72:.1f}× медленнее",ha="center",va="top",transform=ax.transAxes,fontsize=11,fontweight="bold")
    ax.legend(loc="upper right",bbox_to_anchor=(1,.87))
    for ax in axes[0]: ax.xaxis.set_major_locator(MaxNLocator(5))
    save(fig,"dynamics")

    # Representative particles selected using a stated rule, not hand-picked model successes.
    active_rows=np.flatnonzero(last_active)
    ordered=active_rows[np.argsort(pos_error[-1,active_rows])]
    selected=[ordered[len(ordered)//2],ordered[int(.95*(len(ordered)-1))],int(np.argmax(pos_error[-1]))]
    labels=["Медиана ошибки среди подвижных", "95-й перцентиль среди подвижных", "Худшая ошибка среди всех частиц"]
    fig,axes=plt.subplots(2,3,figsize=(16,9.8))
    fig.subplots_adjust(left=.07,right=.96,bottom=.13,top=.82,wspace=.30,hspace=.34)
    decorate(fig,"Три частицы: где сохраняется траектория, а где теряется",
             "Солвер — синий · модель — оранжевый · ID выбраны по ошибке положения на кадре 110",
             "Верхний ряд: z относительно истинного z этой же частицы на первом сохранённом кадре 101.\n"
             "Оси каждой колонки имеют свой масштаб. Нижний ряд: вертикальная скорость; первый кадр уже предсказан моделью.")
    rows=[]
    for j,(row,label) in enumerate(zip(selected,labels)):
        particle_id=int(data["particle_ids"][row])
        base=r[0,row,2]
        axes[0,j].plot(t,(r[:,row,2]-base)*1000,"o-",color=TRUTH,ms=4,label="Солвер")
        axes[0,j].plot(t,(p[:,row,2]-base)*1000,"o-",color=MODEL,ms=4,label="Модель")
        axes[0,j].set(title=f"{label}\nID {particle_id} · ошибка {pos_error[-1,row]:.3f} мм",ylabel="z − z солвера на кадре 101, мм")
        axes[0,j].legend(loc="best")
        axes[1,j].plot(t,r[:,row,5]*1000,"o-",color=TRUTH,ms=4)
        axes[1,j].plot(t,p[:,row,5]*1000,"o-",color=MODEL,ms=4)
        axes[1,j].set(xlabel="Время солвера, с",ylabel="Вертикальная скорость vz, мм/с")
        for i in (0,1): axes[i,j].xaxis.set_major_locator(MaxNLocator(4))
        for i,frame in enumerate(data["particle_frames"]):
            rows.append({"selection":label,"particle_id":particle_id,"frame":int(frame),"time_s":float(t[i]),
                         "true_z_mm":float(r[i,row,2]*1000),"predicted_z_mm":float(p[i,row,2]*1000),
                         "true_vz_mm_s":float(r[i,row,5]*1000),"predicted_vz_mm_s":float(p[i,row,5]*1000),
                         "position_error_mm":float(pos_error[i,row])})
    with (OUT/"selected_particles.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    save(fig,"particle_tracks")
    print(json.dumps({key:summary[key] for key in (
        "active_fraction_percent","final_position_percentiles_mm","final_position_above_10mm",
        "final_active_velocity_zero_baseline_skill_percent","final_inactive_speed_median_mm_s",
        "position_increment_zero_baseline_skill_percent","velocity_increment_zero_baseline_skill_percent",
        "velocity_increment_rms_mm_s","mean_seconds_per_frame_excluding_first")},indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
