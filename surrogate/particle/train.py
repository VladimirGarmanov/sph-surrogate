"""Обучение на всех частицах каждого кадра с накоплением градиентов.

    python -m surrogate.particle.train --data_dir data --neighbors 256 --epochs 1
"""
import argparse
import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from ..data import discover_runs
from ..model import pick_device
from .config import Config
from .data import FEATURE_NAMES, ParticleDataset, Stats, add_input_noise, to_tensors
from .model import ParticleNet, predict

CHECKPOINT_KIND = "particle-history-delta-v2"


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") != CHECKPOINT_KIND:
        raise ValueError("expected a particle-history-delta-v2 checkpoint; older architectures are incompatible")
    return checkpoint


def split_runs(runs, cfg):
    tags = {run.tag for run in runs}
    missing = set(cfg.holdout_tags) - tags
    if missing:
        raise ValueError(f"missing holdout runs: {sorted(missing)}; download them or set --holdout explicitly. "
                         "Use --holdout '' only for a training-data diagnostic.")
    train = [run for run in runs if run.tag not in cfg.holdout_tags]
    val = [run for run in runs if run.tag in cfg.holdout_tags]
    if not train:
        raise ValueError("no training runs remain after the holdout split")
    return train, val


@torch.no_grad()
def evaluate(model, samples, stats, device):
    was_training = model.training
    model.eval()
    squared = np.zeros(len(FEATURE_NAMES), np.float64)
    baseline_squared = np.zeros_like(squared)
    count = 0
    zero_delta = stats.norm("target", np.zeros(len(FEATURE_NAMES), np.float32))
    for sample in samples:
        batch = {key: value.to(device) for key, value in sample.items()}
        y = batch["y"].cpu().numpy()
        err = predict(model, batch).cpu().numpy() - y
        err = err.reshape(-1, len(FEATURE_NAMES))
        y = y.reshape(-1, len(FEATURE_NAMES))
        squared += (err.astype(np.float64) ** 2).sum(0)
        baseline_squared += ((y - zero_delta).astype(np.float64) ** 2).sum(0)
        count += len(y)
    model.train(was_training)
    feature_mse = squared / count
    return {"mse": float(feature_mse.mean()),
            "zero_delta_mse": float((baseline_squared / count).mean()),
            "feature_mse": feature_mse.tolist(),
            "feature_rmse": (np.sqrt(feature_mse) * stats.arrays["target_std"]).tolist(),
            "particles": count}


def train_frame(model, dataset, frame_index, stats, device, optimizer, progress=None):
    """Один шаг Adam по средней ошибке ВСЕГО кадра, независимо от размера порции GPU."""
    run_index, _ = dataset.index[frame_index]
    total = dataset.runs[run_index].n_soil
    if total < 1:
        raise ValueError("training frame must contain soil particles")
    optimizer.zero_grad(set_to_none=True)
    weighted_loss, visited = 0., 0
    for ids, sample in dataset.frame_batches(frame_index):
        batch = {key: value.to(device) for key, value in to_tensors(sample, stats).items()}
        batch = add_input_noise(batch, dataset.cfg.input_noise_std)
        prediction = predict(model, batch)
        if prediction.shape != batch["y"].shape:
            raise ValueError(f"prediction {prediction.shape} and target {batch['y'].shape} must match")
        loss = ((prediction - batch["y"]) ** 2).mean()
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"non-finite loss in frame index {frame_index}, particle {ids[0]}")
        # Последняя неполная порция весит пропорционально числу частиц.
        fraction = len(ids) / total
        (loss * fraction).backward()
        weighted_loss += loss.item() * fraction
        visited += len(ids)
        if progress is not None:
            progress(visited, total)
    if visited != total:
        raise ValueError(f"incomplete frame: visited {visited}/{total} particles")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return weighted_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in fields(Config):
        parser.add_argument(f"--{field.name}", type=type(field.default), default=argparse.SUPPRESS,
                            help=f"default: {field.default}")
    parser.add_argument("--resume", help="continue a particle checkpoint using its config and normalization")
    args = vars(parser.parse_args())
    resume = args.pop("resume")
    checkpoint = load_checkpoint(resume) if resume else None
    if checkpoint and checkpoint.get("training_mode") == "rollout-replay-v1":
        raise ValueError("resume rollout-replay training with surrogate.particle.finetune --resume")
    settings = dict(checkpoint["config"]) if checkpoint else {}
    if resume:
        # Старый checkpoint не должен молча сменить способ обучения.
        settings.setdefault("training_mode", "random_particles")
        settings["out_dir"] = str(Path(resume).parent)
    settings.update(args)
    cfg = Config.from_dict(settings)
    if checkpoint:
        for key in ("neighbors", "hidden", "dt", "frame_stride", "history_frames", "holdout", "batch",
                    "seed", "lr", "lr_decay_steps", "val_samples", "stats_frames",
                    "prediction_horizon", "input_noise_std", "training_mode"):
            default = "random_particles" if key == "training_mode" else Config().to_dict()[key]
            old_value = checkpoint["config"].get(key, default)
            if cfg.to_dict()[key] != old_value:
                raise ValueError(f"cannot change {key} on resume; start a new experiment instead")
    if cfg.training_mode == "full_frames" and "steps" in args:
        parser.error("full_frames uses --epochs, not --steps; one update processes a complete frame")
    out = Path(cfg.out_dir)
    if not resume and any((out / name).exists() for name in ("model.pt", "best.pt")):
        raise ValueError("out_dir already has a checkpoint; choose another --out_dir or use --resume")
    runs = discover_runs(cfg.data_dir)
    train_runs, val_runs = split_runs(runs, cfg)
    torch.manual_seed(cfg.seed)
    device = pick_device() if cfg.device == "auto" else torch.device(cfg.device)
    print(f"device={device} train={[r.tag for r in train_runs]} holdout={[r.tag for r in val_runs]}", flush=True)
    label = "VAL" if val_runs else "TRAIN_DIAGNOSTIC"
    if not val_runs:
        print("No independent validation: diagnostics use training runs; best.pt will not be selected.", flush=True)

    dataset = ParticleDataset(train_runs, cfg, seed=cfg.seed)
    full_frames = cfg.training_mode == "full_frames"
    total_steps = cfg.epochs * len(dataset.index) if full_frames else cfg.steps
    frame_layout = [{"tag": run.tag, "frames": run.n_frames, "soil": run.n_soil} for run in train_runs]
    if checkpoint:
        if full_frames and checkpoint.get("frame_layout") != frame_layout:
            raise ValueError("cannot resume full_frames with changed training runs or frame/particle counts")
        if total_steps <= checkpoint["step"]:
            raise ValueError("training already reached this limit; increase --epochs (full_frames) or --steps")
    if checkpoint:
        stats = Stats(checkpoint["stats"])
    else:
        stats_data = ParticleDataset(train_runs, cfg, seed=cfg.seed + 2)
        stats = Stats.compute(stats_data.sample() for _ in range(cfg.stats_frames))
    val_data = ParticleDataset(val_runs or train_runs, cfg, seed=cfg.seed + 1)
    val_samples = [to_tensors(val_data.sample(), stats) for _ in range(cfg.val_samples)]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    np.savez(out / "stats.npz", **stats.arrays)

    model = ParticleNet(cfg.hidden, cfg.prediction_horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.1 ** (step / cfg.lr_decay_steps))
    step, best_val = 0, float("inf")
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step, best_val = checkpoint["step"], checkpoint["best_val"]
        dataset.rng.bit_generator.state = checkpoint["data_rng"]
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)

    def save(name):
        payload = {"kind": CHECKPOINT_KIND, "config": cfg.to_dict(), "stats": stats.to_dict(),
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "step": step, "best_val": best_val,
                    "data_rng": dataset.rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                    "frame_layout": frame_layout,
                    "completed_epochs": step // len(dataset.index) if full_frames else None,
                    "next_frame_index": step % len(dataset.index) if full_frames else None,
                    "train_tags": [r.tag for r in train_runs], "val_tags": [r.tag for r in val_runs]}
        temporary = out / (name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(out / name)

    print(f"{sum(p.numel() for p in model.parameters()):,} parameters; "
          f"batch={cfg.batch} targets, K={cfg.neighbors}, history={cfg.history_frames}+current, "
          f"horizon={cfg.prediction_horizon}, input_noise_std={cfg.input_noise_std:g}, "
          f"dt={cfg.dt * cfg.frame_stride:g}s", flush=True)
    if full_frames:
        targets_per_epoch = sum(dataset.runs[ri].n_soil for ri, _ in dataset.index)
        print(f"FULL FRAMES: {len(dataset.index):,} frames/epoch; "
              f"{targets_per_epoch:,} particle-frame targets/epoch; {cfg.epochs} epoch(s). "
              f"One optimizer update per complete frame; GPU portion={cfg.batch}. "
              f"Resume at update {step}/{total_steps}.", flush=True)
        print("Normalization uses sampled training statistics; VAL uses fixed sampled batches. "
              "Every TRAIN frame includes all soil particles.", flush=True)
    model.train()
    start = time.perf_counter()
    running, seen = 0., 0
    with (out / "metrics.jsonl").open("a" if resume else "w") as log:
        while step < total_steps:
            frame_info = {}
            if full_frames:
                frame_index = step % len(dataset.index)
                ri, frame = dataset.index[frame_index]
                run = dataset.runs[ri]
                epoch = step // len(dataset.index) + 1
                frame_info = {"epoch": epoch, "frame_index": frame_index, "tag": run.tag,
                              "frame": frame, "targets": run.n_soil}
                print(f"epoch {epoch}/{cfg.epochs} frame {frame_index + 1}/{len(dataset.index)} "
                      f"{run.tag} input_frame={frame} (#{frame + 1}) "
                      f"targets=ALL {run.n_soil:,}; accumulating gradients", flush=True)
                last_progress = time.perf_counter()

                def progress(visited, total):
                    nonlocal last_progress
                    now = time.perf_counter()
                    if now - last_progress >= 30:
                        print(f"  particles {visited:,}/{total:,}; weights update after full frame", flush=True)
                        last_progress = now

                loss_value = train_frame(model, dataset, frame_index, stats, device, optimizer, progress)
            else:
                batch = {key: value.to(device) for key, value in to_tensors(dataset.sample(), stats).items()}
                batch = add_input_noise(batch, cfg.input_noise_std)
                loss = ((predict(model, batch) - batch["y"]) ** 2).mean()
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"non-finite training loss at step {step + 1}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                loss_value = loss.item()
            scheduler.step()
            step += 1
            running += loss_value
            seen += 1
            if full_frames or step % cfg.log_every == 0 or step == total_steps:
                elapsed = time.perf_counter() - start
                record = {"step": step, "split": "TRAIN", "mse": running / seen,
                          "lr": scheduler.get_last_lr()[0], "training_mode": cfg.training_mode,
                          **frame_info}
                if full_frames:
                    record["seconds_per_frame"] = elapsed
                timing = f"{elapsed:.1f}s/frame" if full_frames else f"{seen / elapsed:.2f} updates/s"
                print(f"[{step:6d}] MSE={record['mse']:.5g} lr={record['lr']:.2g} "
                      f"{timing}", flush=True)
                log.write(json.dumps(record) + "\n")
                running, seen, start = 0., 0, time.perf_counter()
            if step % cfg.val_every == 0 or step == total_steps:
                report = evaluate(model, val_samples, stats, device)
                if not np.isfinite(report["mse"]):
                    raise FloatingPointError("non-finite validation error")
                print(f"[{step:6d}] {label} MSE={report['mse']:.5g} "
                      f"unchanged-particle baseline={report['zero_delta_mse']:.5g}", flush=True)
                log.write(json.dumps({"step": step, "split": label, **report}) + "\n")
                log.flush()
                improved = bool(val_runs) and report["mse"] < best_val
                if improved:
                    best_val = report["mse"]
                save("model.pt")
                if improved:
                    save("best.pt")
            elif full_frames:
                # Кадр — минимальная сохраняемая единица; после сбоя неполный кадр повторяется.
                log.flush()
                save("model.pt")
        print(f"saved {out / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
