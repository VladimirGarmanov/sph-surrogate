"""Дообучение существующих весов на обновляемых последовательных прогнозах самой модели.

Каждый цикл собирает прогноз всего грунта без градиентов, обучает поправки
к следующему состоянию на зафиксированных историях и проверяет отдельный прогноз.
Это повторное обучение по собранным траекториям, а не обратное распространение
градиентов через все прошедшие шаги времени.
"""
import argparse
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import time

import numpy as np
import torch

from ..data import discover_runs
from ..model import pick_device
from .config import Config, migrate_saved_config
from .data import ParticleDataset, Stats, to_tensors
from .model import ParticleNet, predict
from .replay import RolloutReplay, rollout_windows
from .rollout import rollout
from .train import CHECKPOINT_KIND, evaluate, load_checkpoint, split_runs

TRAINING_MODE = "rollout-replay-v1"


@dataclass
class FinetuneConfig:
    data_dir: str = "data"
    out_dir: str = "checkpoints/particle-replay"
    cycles: int = 3
    rollout_steps: int = 10
    updates_per_cycle: int = 500
    batch: int = 32
    inference_batch: int = 32
    lr: float = 1e-5
    clean_fraction: float = .25
    val_tag: str = ""
    val_start_frame: int = -1
    val_steps: int = 10
    active_speed: float = .001       # м/с; используется только для отбора при валидации
    log_every: int = 50
    seed: int = 17
    device: str = "auto"

    def __post_init__(self):
        for name in ("cycles", "rollout_steps", "updates_per_cycle", "batch", "inference_batch",
                     "val_steps", "log_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.rollout_steps < 2:
            raise ValueError("rollout_steps must be at least 2 to train on a predicted current state")
        if not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not 0 <= self.clean_fraction < 1:
            raise ValueError("clean_fraction must be in [0, 1); some updates must use replay")
        if not np.isfinite(self.active_speed) or self.active_speed <= 0:
            raise ValueError("active_speed must be finite and positive")
        if self.val_start_frame < -1:
            raise ValueError("val_start_frame must be -1 (default) or a nonnegative frame")


def trajectory_metrics(result, active_speed):
    """Сравниваем одинаковые пары кадр/ID; маска активных частиц зависит только от истинных данных."""
    reference = np.asarray(result["particle_reference"], np.float64)
    error = np.asarray(result["particle_predicted"], np.float64) - reference
    speed = np.linalg.norm(reference[..., 3:6], axis=-1)
    active = speed >= active_speed
    velocity_squared = np.sum(error[..., 3:6] ** 2, axis=-1)
    selection = active if active.any() else np.ones_like(active)
    metric = "active_velocity_rmse" if active.any() else "all_velocity_rmse_no_active"
    score = float(np.sqrt(velocity_squared[selection].mean()))
    positions = np.linalg.norm(error[..., :3], axis=-1)
    predicted_speed = np.linalg.norm(np.asarray(result["particle_predicted"])[..., 3:6], axis=-1)
    report = {
        "selection_metric": metric, "score_m_s": score,
        "active_speed_min_m_s": active_speed, "active_particle_frame_pairs": int(active.sum()),
        "particle_frame_pairs": int(active.size),
        "velocity_rmse_m_s": float(np.sqrt(velocity_squared.mean())),
        "active_speed_ratio_median": float(np.median(predicted_speed[active] / speed[active])) if active.any() else None,
        "position_rmse_m": float(np.sqrt(np.mean(positions ** 2))),
        "position_p95_m": float(np.percentile(positions, 95)),
        "position_max_m": float(positions.max()),
        "last_position_rmse_m": float(np.sqrt(np.mean(positions[-1] ** 2))),
        "feature_rmse": np.sqrt(np.mean(error ** 2, axis=(0, 1))).tolist(),
    }
    if not np.isfinite(score) or not np.isfinite(error).all():
        raise FloatingPointError("non-finite validation trajectory")
    return report


def _save_npz(path, result):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **result)
    temporary.replace(path)


def run_finetuning(checkpoint, options, *, resume=False, source_path=""):
    """Сохраняем и возобновляем только между полными циклами: собранные траектории тогда можно отбросить."""
    previous = checkpoint.get("finetune") if resume else None
    if resume and (checkpoint.get("training_mode") != TRAINING_MODE or previous is None):
        raise ValueError("--resume requires a rollout-replay checkpoint; use --ckpt for initial weights")
    completed = int(previous["completed_cycles"]) if previous else 0
    if options.cycles <= completed:
        raise ValueError("--cycles is the total fine-tuning cycle count; increase it to continue")
    out = Path(options.out_dir)
    if not resume:
        if source_path and out.resolve() == Path(source_path).resolve().parent:
            raise ValueError("choose a new out_dir; keep the source checkpoint directory intact")
        reserved = ("model.pt", "best.pt", "finetune.json", "metrics.jsonl", "validation_before.npz")
        if any((out / name).exists() for name in reserved):
            raise ValueError("out_dir already contains an experiment; choose a new directory or --resume")
    cfg = Config.from_dict({**migrate_saved_config(checkpoint["config"]), "data_dir": options.data_dir,
                            "out_dir": str(out), "batch": options.batch,
                            "lr": options.lr, "device": options.device})
    stats = Stats(checkpoint["stats"])
    runs = discover_runs(options.data_dir)
    train_runs, val_runs = split_runs(runs, cfg)
    windows = rollout_windows(train_runs, cfg, options.rollout_steps)
    independent = bool(val_runs)
    candidates = val_runs if independent else train_runs
    if options.val_tag:
        matches = [run for run in candidates if run.tag == options.val_tag]
        if not matches:
            raise ValueError("val_tag must belong to the checkpoint holdout split (train only if holdout is empty)")
        val_run = matches[0]
    else:
        val_run = candidates[0]
    initial = cfg.history_frames * cfg.frame_stride
    val_start = initial if options.val_start_frame == -1 else options.val_start_frame
    if val_start < initial or val_start + options.val_steps * cfg.frame_stride >= val_run.n_frames:
        raise ValueError("validation window lacks the complete history or requested future frames")
    val_plan = {"tag": val_run.tag, "start_frame": val_start, "steps": options.val_steps}
    train_tags, val_tags = [run.tag for run in train_runs], [run.tag for run in val_runs]
    if previous:
        if previous["validation_plan"] != val_plan:
            raise ValueError("cannot change validation window on resume")
        if train_tags != checkpoint["train_tags"] or val_tags != checkpoint["val_tags"]:
            raise ValueError("training/holdout run lists changed on resume")
    torch.manual_seed(options.seed)
    device = pick_device() if options.device == "auto" else torch.device(options.device)
    model = ParticleNet(cfg.hidden).to(device)
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr)
    rng = np.random.default_rng(options.seed)
    clean_data = ParticleDataset(train_runs, cfg, seed=options.seed + 1)
    val_data = ParticleDataset(candidates, cfg, seed=options.seed + 2)
    clean_validation = [to_tensors(val_data.sample(), stats) for _ in range(cfg.val_samples)]
    source_step = int(previous["source_step"]) if previous else int(checkpoint["step"])
    updates = int(previous["updates"]) if previous else 0
    best_score = float(previous["best_score_m_s"]) if previous else float("inf")
    baseline = previous["baseline"] if previous else None
    if previous:
        optimizer.load_state_dict(checkpoint["optimizer"])
        rng.bit_generator.state = previous["selection_rng"]
        clean_data.rng.bit_generator.state = previous["clean_rng"]
        torch.set_rng_state(previous["torch_rng"])
        if device.type == "cuda" and previous.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(previous["cuda_rng"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "finetune.json").write_text(json.dumps({"training_mode": TRAINING_MODE,
        "config": asdict(options), "source_checkpoint": str(source_path),
        "source_step": source_step, "validation_plan": val_plan,
        "train_tags": train_tags, "val_tags": val_tags,
        "gradient_horizon": 1, "normalization": "unchanged checkpoint statistics"}, indent=2) + "\n")
    print(f"device={device}; source step={source_step}; fine-tune cycles={completed}->{options.cycles}; "
          f"lr={options.lr:g}; K={cfg.neighbors}; history={cfg.history_frames}+current", flush=True)
    print(f"train={train_tags}; holdout={val_tags}; validation={val_plan}", flush=True)
    print("Each cycle: full-soil rollout -> detached replay updates -> full-soil validation. "
          "Neighbour histories after warm start are predictions. Gradient horizon=1.", flush=True)
    if not independent:
        print("TRAIN_DIAGNOSTIC: checkpoint has no holdout; no best.pt will be selected.", flush=True)
    label = "VAL" if independent else "TRAIN_DIAGNOSTIC"

    def save(name, cycle):
        state = {"kind": CHECKPOINT_KIND, "training_mode": TRAINING_MODE,
            "config": cfg.to_dict(), "stats": stats.to_dict(), "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "step": source_step + updates,
            "train_tags": train_tags, "val_tags": val_tags,
            "finetune": {"config": asdict(options), "completed_cycles": cycle,
                "updates": updates, "source_step": source_step, "best_score_m_s": best_score,
                "baseline": baseline, "validation_plan": val_plan,
                "selection_rng": rng.bit_generator.state, "clean_rng": clean_data.rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}}
        path = out / name
        temporary = path.with_suffix(".pt.tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def validate(cycle):
        print(f"[cycle {cycle}] {label} rollout: {options.val_steps} frames, all {val_run.n_soil:,} soil particles", flush=True)
        result = rollout(model, cfg, stats, val_run, device, n_steps=options.val_steps,
                         batch_size=options.inference_batch, save_particles=True, start_frame=val_start)
        result.update(tag=np.asarray(val_run.tag), checkpoint_step=np.asarray(source_step + updates),
                      batch_size=np.asarray(options.inference_batch), neighbors=np.asarray(cfg.neighbors),
                      history_frames=np.asarray(cfg.history_frames), dt=np.asarray(cfg.dt * cfg.frame_stride))
        metrics = trajectory_metrics(result, options.active_speed)
        metrics["clean_one_step"] = evaluate(model, clean_validation, stats, device)
        print(f"[cycle {cycle}] {label} {metrics['selection_metric']}={metrics['score_m_s']*1000:.5g}mm/s "
              f"position_RMSE={metrics['position_rmse_m']*1000:.5g}mm "
              f"position_p95={metrics['position_p95_m']*1000:.5g}mm "
              f"clean_MSE={metrics['clean_one_step']['mse']:.5g}", flush=True)
        return result, metrics

    with (out / "metrics.jsonl").open("a" if resume else "w") as log:
        def record(values):
            log.write(json.dumps(values, allow_nan=False) + "\n")
            log.flush()

        if not previous:
            # Сбор исходного прогноза сам по себе может занять минуты. Заранее сохраняем
            # начальную контрольную точку для возобновления даже после прерывания на первом кадре.
            save("model.pt", 0)
        if baseline is None:
            result, baseline = validate(0)
            best_score = baseline["score_m_s"]
            _save_npz(out / "validation_before.npz", result)
            record({"cycle": 0, "updates": 0, "split": label, "baseline": True, **baseline})
            # Исходные веса первыми считаются лучшими: первый цикл не заменит их худшим результатом.
            if independent:
                _save_npz(out / "validation_best.npz", result)
                save("best.pt", 0)
            save("model.pt", 0)
            del result
        for cycle in range(completed + 1, options.cycles + 1):
            ri, start_frame = windows[int(rng.integers(len(windows)))]
            run = train_runs[ri]
            print(f"[cycle {cycle}/{options.cycles}] TRAIN collect {run.tag} start_frame={start_frame} "
                  f"steps={options.rollout_steps}; all {run.n_soil:,} soil particles", flush=True)
            collected = rollout(model, cfg, stats, run, device, n_steps=options.rollout_steps,
                                batch_size=options.inference_batch, save_particles=True, start_frame=start_frame)
            replay = RolloutReplay(run, cfg, collected, seed=int(rng.integers(2**32)))
            record({"cycle": cycle, "updates": updates, "split": "TRAIN_COLLECTION",
                    "tag": run.tag, "start_frame": start_frame,
                    **trajectory_metrics(collected, options.active_speed)})
            del collected
            clean_count = min(options.updates_per_cycle - 1,
                              round(options.updates_per_cycle * options.clean_fraction))
            clean_updates = np.arange(options.updates_per_cycle) < clean_count
            rng.shuffle(clean_updates)
            model.train()
            running = {"replay": [0., 0], "clean": [0., 0]}
            started = time.perf_counter()
            for update, clean in enumerate(clean_updates, start=1):
                sample = clean_data.sample() if clean else replay.sample()
                batch = {key: value.to(device) for key, value in to_tensors(sample, stats).items()}
                loss = ((predict(model, batch) - batch["y"]) ** 2).mean()
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"non-finite fine-tuning loss in cycle {cycle}, update {update}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                key = "clean" if clean else "replay"
                running[key][0] += loss.item()
                running[key][1] += 1
                if update % options.log_every == 0 or update == options.updates_per_cycle:
                    metrics = {key + "_mse": total / count if count else None
                               for key, (total, count) in running.items()}
                    count = sum(value[1] for value in running.values())
                    rate = count / (time.perf_counter() - started)
                    print(f"[cycle {cycle} update {update}/{options.updates_per_cycle}] "
                          f"replay_MSE={metrics['replay_mse']} clean_MSE={metrics['clean_mse']} "
                          f"{rate:.2f} updates/s", flush=True)
                    record({"cycle": cycle, "updates": updates, "split": "TRAIN", **metrics})
                    running = {"replay": [0., 0], "clean": [0., 0]}
                    started = time.perf_counter()
            del replay
            result, metrics = validate(cycle)
            improved = independent and metrics["score_m_s"] < best_score
            if improved:
                best_score = metrics["score_m_s"]
                _save_npz(out / "validation_best.npz", result)
            _save_npz(out / "validation_latest.npz", result)
            record({"cycle": cycle, "updates": updates, "split": label,
                    "improved": improved, "baseline_score_m_s": baseline["score_m_s"], **metrics})
            if improved:
                save("best.pt", cycle)
            save("model.pt", cycle)
            del result
            print(f"[cycle {cycle}] saved model.pt; "
                  f"{'new best.pt' if improved else 'best unchanged' if independent else 'TRAIN_DIAGNOSTIC'}; "
                  f"best_score={best_score*1000:.5g}mm/s", flush=True)
    return out / "model.pt"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ckpt", help="initial weights/stats; starts a new fine-tuning experiment")
    source.add_argument("--resume", help="fine-tuning model.pt saved at a complete cycle boundary")
    for field in fields(FinetuneConfig):
        parser.add_argument(f"--{field.name}", type=type(field.default), default=argparse.SUPPRESS,
                            help=f"default: {field.default}")
    args = vars(parser.parse_args(argv))
    resume_path, initial_path = args.pop("resume"), args.pop("ckpt")
    path = resume_path or initial_path
    checkpoint = load_checkpoint(path)
    if resume_path:
        if checkpoint.get("training_mode") != TRAINING_MODE:
            raise ValueError("--resume requires a rollout-replay checkpoint; use --ckpt for initial weights")
        settings = dict(checkpoint["finetune"]["config"])
        settings["out_dir"] = str(Path(path).parent)
        for key, value in args.items():
            if key not in ("cycles", "device", "data_dir") and value != settings[key]:
                raise ValueError(f"cannot change {key} on resume; use --ckpt and a new out_dir")
        settings.update(args)
    else:
        settings = args
    options = FinetuneConfig(**settings)
    run_finetuning(checkpoint, options, resume=bool(resume_path), source_path=path)


if __name__ == "__main__":
    main()
