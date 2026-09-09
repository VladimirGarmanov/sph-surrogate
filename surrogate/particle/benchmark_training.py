"""Измерение подготовки и градиентов FP32 на фиксированных целях без обновления весов."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from ..data import discover_runs
from .config import Config
from .data import ParticleDataset, Stats, to_tensors
from .model import ParticleNet, predict
from .train import load_checkpoint, split_runs


def gradient_pass(model, dataset, frame_index, stats, device, count):
    """Одинаковые ID при любом разбиении; шум отключён для сравнения градиентов."""
    synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    model.zero_grad(set_to_none=True)
    synchronize()
    stages = dict(prepare=0., to_device=0., forward=0., backward=0.)
    seen, total_loss = 0, 0.
    iterator = iter(dataset.frame_batches(frame_index))
    while seen < count:
        start = time.perf_counter()
        ids, sample = next(iterator)
        size = min(len(ids), count - seen)
        sample = {key: value[:size] for key, value in sample.items()}
        batch = to_tensors(sample, stats)
        stages["prepare"] += time.perf_counter() - start
        start = time.perf_counter()
        batch = {key: value.to(device) for key, value in batch.items()}
        synchronize()
        stages["to_device"] += time.perf_counter() - start
        start = time.perf_counter()
        prediction = predict(model, batch)
        if prediction.shape != batch["y"].shape:
            raise ValueError("prediction and target shapes must match")
        loss = ((prediction - batch["y"]) ** 2).mean()
        synchronize()
        stages["forward"] += time.perf_counter() - start
        start = time.perf_counter()
        (loss * (size / count)).backward()
        synchronize()
        stages["backward"] += time.perf_counter() - start
        total_loss += loss.item() * (size / count)
        seen += size
    gradient = torch.cat([p.grad.detach().flatten().cpu() for p in model.parameters()])
    if not np.isfinite(total_loss) or not torch.isfinite(gradient).all():
        raise FloatingPointError("non-finite loss or gradient")
    return total_loss, gradient, stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--batches", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--targets", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if min(args.batches + [args.targets, args.repeats]) < 1:
        parser.error("batches, targets and repeats must be positive")
    out = Path(args.out)
    if out.exists():
        parser.error("output already exists; choose a new --out")
    if not torch.cuda.is_available():
        parser.error("run this benchmark on the CUDA training server")
    device = torch.device("cuda")
    checkpoint = load_checkpoint(args.ckpt)
    cfg = Config.from_dict(checkpoint["config"])
    cfg = replace(cfg, data_dir=args.data_dir, input_noise_std=0.)
    runs, _ = split_runs(discover_runs(args.data_dir), cfg)
    stats = Stats(checkpoint["stats"])
    model = ParticleNet(cfg.hidden, cfg.prediction_horizon).to(device).train()
    model.load_state_dict(checkpoint["model"])
    dataset = ParticleDataset(runs, cfg)
    frame_index = (checkpoint["step"] % len(dataset.index)) if cfg.training_mode == "full_frames" else 0
    ri, frame = dataset.index[frame_index]
    count = min(args.targets, runs[ri].n_soil)
    candidates = sorted(set(b for b in args.batches if b <= count))
    if not candidates:
        parser.error("no candidate batch fits the number of benchmark targets")
    report = {"checkpoint": args.ckpt, "checkpoint_step": checkpoint["step"],
              "device": torch.cuda.get_device_name(device), "torch": str(torch.__version__),
              "tag": runs[ri].tag, "frame": frame, "targets": count, "precision": "fp32",
              "input_noise_std": 0., "results": [],
              "note": "Fixed targets, no weight updates, no input noise. Stage timings synchronize CUDA. "
                      "Validate any speedup on a complete training frame with the actual noise setting."}
    baseline_gradient = None
    for portion in candidates:
        print(f"GPU portion={portion}; warmup + {args.repeats} measured passes over {count} targets", flush=True)
        dataset = ParticleDataset(runs, replace(cfg, gpu_batch=portion))
        try:
            gradient_pass(model, dataset, frame_index, stats, device, min(portion, count))
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            timings, differences = [], []
            for repeat in range(args.repeats):
                loss, gradient, stages = gradient_pass(model, dataset, frame_index, stats, device, count)
                if baseline_gradient is None:
                    baseline_gradient = gradient.clone()
                relative = float(torch.linalg.vector_norm(gradient - baseline_gradient)
                                 / torch.linalg.vector_norm(baseline_gradient).clamp_min(1e-12))
                differences.append(relative)
                timings.append(stages)
                print(f"  repeat {repeat + 1}: {sum(stages.values()):.3f}s, "
                      f"relative gradient difference={relative:.3g}", flush=True)
            means = {key: float(np.mean([t[key] for t in timings])) for key in timings[0]}
            seconds = sum(means.values())
            row = {"gpu_batch": portion, "status": "ok", "stages_seconds": means,
                   "targets_per_second": count / seconds, "loss": loss,
                   "max_relative_gradient_difference": max(differences),
                   "gradient_check_passed": max(differences) <= 1e-3,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30}
            report["results"].append(row)
            print(json.dumps(row), flush=True)
        except torch.cuda.OutOfMemoryError:
            report["results"].append({"gpu_batch": portion, "status": "out_of_memory"})
            print("  CUDA memory limit reached; weights and checkpoint are unchanged.", flush=True)
        finally:
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    usable = [r for r in report["results"] if r.get("gradient_check_passed")]
    if usable:
        best = max(usable, key=lambda r: r["targets_per_second"])
        report["suggested_gpu_batch"] = best["gpu_batch"]
        print(f"Fastest checked portion: {best['gpu_batch']}. "
              "Confirm with a full frame using train --resume ... --gpu_batch "
              f"{best['gpu_batch']}. This keeps all frame targets and one update per frame.", flush=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Report: {out}", flush=True)


if __name__ == "__main__":
    main()
