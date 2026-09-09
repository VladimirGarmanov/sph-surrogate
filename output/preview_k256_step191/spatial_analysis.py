"""Reproduce the spatial figures from the saved particle arrays, without solver geometry."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, colors, ticker
import numpy as np


BACKGROUND = "#ffffff"
INK = "#172534"
MUTED = "#536375"


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": "#c7d0d8", "axes.linewidth": .7,
        "figure.facecolor": BACKGROUND, "axes.facecolor": "#f8fafc",
        "savefig.facecolor": BACKGROUND, "pdf.fonttype": 42,
    })


def setup_axes(ax, title, limits):
    ax.set_title(title, loc="left", fontweight="semibold", fontsize=12, pad=13)
    ax.set(xlim=limits[0], ylim=limits[1], xlabel="x, м", ylabel="z, м")
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(.1))
    ax.grid(color="#dce3e9", alpha=.65, linewidth=.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=.6)


def velocity_colorbar(fig, artist, rectangle):
    cax = fig.add_axes(rectangle)
    bar = fig.colorbar(artist, cax=cax, orientation="horizontal")
    ticks = [-20, -5, -1, 0, 1, 5, 20]
    bar.set_ticks(ticks)
    bar.set_ticklabels(["−20", "−5", "−1", "0", "+1", "+5", "+20"])
    bar.ax.tick_params(labelsize=9, length=3)
    bar.outline.set_visible(False)
    bar.set_label("Вертикальная скорость v_z, мм/с  ·  − вниз / + вверх", fontsize=10, labelpad=6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--source", type=Path, default=here.parents[1] / "checkpoints/preview_k256.npz")
    parser.add_argument("--out", type=Path, default=here)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure_style()
    with np.load(args.source, allow_pickle=False) as archive:
        truth = archive["particle_reference"].astype(np.float64)
        predicted = archive["particle_predicted"].astype(np.float64)
        frames, times = archive["particle_frames"], archive["particle_t"]
        ids = archive["particle_ids"]
        tag, step, k = str(archive["tag"]), int(archive["checkpoint_step"]), int(archive["neighbors"])
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError("Particle arrays contain non-finite values")
    selection = (truth[0, :, 1] >= 0.) & (truth[0, :, 1] <= .02)
    reference, forecast = truth[:, selection], predicted[:, selection]
    count = int(selection.sum())
    if not count:
        raise ValueError("The selected central layer is empty")
    velocities = np.concatenate([reference[:, :, 5].ravel(), forecast[:, :, 5].ravel()]) * 1000
    velocity_limit = max(1., np.ceil(np.max(np.abs(velocities)) / 5.) * 5.)
    velocity_norm = colors.SymLogNorm(linthresh=.25, linscale=1., base=10,
                                     vmin=-velocity_limit, vmax=velocity_limit)
    error = np.linalg.norm(forecast[:, :, :3] - reference[:, :, :3], axis=-1) * 1000
    error_upper = max(1., np.ceil(error.max()))
    error_norm = colors.LogNorm(vmin=.001, vmax=error_upper)
    limits = ((-.46, .46), (0., .57))
    displacement = np.linalg.norm(reference[-1, :, :3] - reference[0, :, :3], axis=-1) * 1000
    metadata = f"{tag}  ·  checkpoint step {step}  ·  K = {k}"
    slice_caption = (f"Фиксированные ID по истинному кадру {frames[0]}: 0 ≤ y ≤ 20 мм  "
                     f"·  слой толщиной 20 мм, центр y = 10 мм  ·  {count:,} частиц".replace(",", " "))
    scale_caption = (f"Общая шкала v_z для всех кадров и обеих моделей: ±{velocity_limit:g} мм/с; "
                     "symlog, линейная область ±0.25 мм/с. Цвета не обрезаны.")

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 7.3))
    fig.subplots_adjust(left=.06, right=.975, bottom=.32, top=.79, wspace=.16)
    fig.text(.06, .948, "Как расходятся движения частиц", fontsize=23, fontweight="bold")
    fig.text(.06, .895, f"Конец короткого rollout: кадр {frames[-1]} · t = {times[-1]:.2f} с   |   {metadata}",
             color=MUTED, fontsize=11)
    titles = ["01  Решатель · вертикальная скорость", "02  Суррогат · вертикальная скорость",
              "03  Ошибка положения · те же ID"]
    for ax, title in zip(axes, titles):
        setup_axes(ax, title, limits)
    artists = []
    for ax, states in zip(axes[:2], (reference, forecast)):
        artists.append(ax.scatter(states[-1, :, 0], states[-1, :, 2], c=states[-1, :, 5] * 1000,
                                  s=12, cmap="RdBu_r", norm=velocity_norm, linewidths=0, rasterized=True))
    # Error is plotted at ground-truth coordinates, matching particle IDs exactly.
    error_artist = axes[2].scatter(reference[-1, :, 0], reference[-1, :, 2], c=error[-1],
                                   s=12, cmap="magma", norm=error_norm, linewidths=0, rasterized=True)
    for ax in axes[1:]:
        ax.set_ylabel("")
    velocity_colorbar(fig, artists[0], [.095, .239, .515, .018])
    error_bar = fig.colorbar(error_artist, cax=fig.add_axes([.716, .239, .225, .018]), orientation="horizontal")
    error_bar.set_ticks([.001, .01, .1, 1, error_upper])
    error_bar.set_ticklabels(["0.001", "0.01", "0.1", "1", f"{error_upper:g}"])
    error_bar.minorticks_off()
    error_bar.outline.set_visible(False)
    error_bar.set_label("‖r̂ − r‖, мм · логарифмическая шкала", fontsize=10, labelpad=6)
    fig.text(.06, .13, slice_caption, fontsize=10, color=MUTED)
    fig.text(.06, .093, scale_caption, fontsize=9.3, color=MUTED)
    fig.text(.06, .056, "Координаты показаны в реальном масштабе; границы и плита не дорисованы. "
             "Ошибка расположена в истинных координатах частиц.", fontsize=9.3, color=MUTED)
    fig.savefig(args.out / "spatial.png", dpi=180)
    fig.savefig(args.out / "spatial.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.3))
    fig.subplots_adjust(left=.075, right=.965, bottom=.32, top=.77, wspace=.17)
    fig.text(.075, .95, "Центральный слой · короткий rollout", fontsize=22, fontweight="bold")
    fig.text(.075, .898, metadata, fontsize=10.5, color=MUTED)
    timestamp = fig.text(.965, .898, "", ha="right", fontsize=10.5, color=INK, fontweight="semibold")
    dots = []
    for ax, states, title in zip(axes, (reference, forecast), ("Решатель", "Суррогат")):
        setup_axes(ax, title, limits)
        dots.append(ax.scatter(states[0, :, 0], states[0, :, 2], c=states[0, :, 5] * 1000,
                               s=14, cmap="RdBu_r", norm=velocity_norm, linewidths=0))
    axes[1].set_ylabel("")
    velocity_colorbar(fig, dots[0], [.20, .241, .60, .018])
    fig.text(.075, .133, slice_caption, fontsize=9.2, color=MUTED)
    fig.text(.075, .097, "Реальный масштаб, без усиления. За кадры "
             f"{frames[0]}→{frames[-1]} медиана истинного смещения {np.median(displacement):.3f} мм; "
             f"95-й процентиль {np.percentile(displacement, 95):.3f} мм.", fontsize=9.2, color=MUTED)
    fig.text(.075, .061, "Цвет: symlog, линейно ±0.25 мм/с; одна шкала на всё видео. "
             "Показ: 3 кадра/с, исходный шаг 20 мс. Геометрия плиты отсутствует.", fontsize=9.2, color=MUTED)

    def update(index):
        for scatter, states in zip(dots, (reference, forecast)):
            scatter.set_offsets(states[index][:, [0, 2]])
            scatter.set_array(states[index, :, 5] * 1000)
        timestamp.set_text(f"Кадр {frames[index]} / {frames[-1]}  ·  t = {times[index]:.2f} с")
        return *dots, timestamp

    update(0)
    fig.savefig(args.out / "motion_first_frame.png", dpi=140)
    movie = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / 3, blit=False)
    movie.save(args.out / "motion.gif", writer=animation.PillowWriter(fps=3), dpi=120)
    plt.close(fig)

    summary = {
        "source": str(args.source), "tag": tag, "checkpoint_step": step, "neighbors": k,
        "selection": {"coordinate_frame": int(frames[0]), "y_min_m": 0., "y_max_m": .02,
                      "count": count, "fixed_ids": True, "particle_ids": ids[selection].tolist()},
        "frames": frames.tolist(), "times_s": times.tolist(),
        "velocity_mm_s": {"minimum": float(velocities.min()), "maximum": float(velocities.max()),
                          "color_limit": float(velocity_limit), "linthresh": .25},
        "final_position_error_mm": {"median": float(np.median(error[-1])),
                                    "p95": float(np.percentile(error[-1], 95)), "max": float(error[-1].max())},
        "truth_displacement_from_first_saved_frame_mm": {"median": float(np.median(displacement)),
                                                       "p95": float(np.percentile(displacement, 95)),
                                                       "max": float(displacement.max())},
    }
    (args.out / "spatial_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "selection"}, indent=2))


if __name__ == "__main__":
    main()
