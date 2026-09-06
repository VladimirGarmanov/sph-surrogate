"""One-step training.

    python -m surrogate.train --data_dir data --steps 20000 --out_dir checkpoints/gns

Every Config field is a CLI flag. Writes config.json, stats.npz, model.pt to out_dir.
"""
import argparse
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import discover_runs
from .dataset import FrameDataset, collate, sample_frames
from .model import GNS, pick_device
from .normalize import Stats

GROUPS = {"acc": slice(0, 3), "rho": slice(3, 4), "stress": slice(4, 10), "plast": slice(10, 13)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(Config):
        typ = type(f.default)
        if typ is bool:
            p.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        else:
            p.add_argument(f"--{f.name}", type=typ, default=f.default)
    p.add_argument("--resume", default=None, help="model.pt to continue from")
    return p.parse_args()


def group_losses(pred, y):
    """Normalised MSE per output group, for logging."""
    return {k: ((pred[:, s] - y[:, s]) ** 2).mean().item() for k, s in GROUPS.items()}


def evaluate(model, samples, device):
    model.eval()
    tot, n = {k: 0.0 for k in GROUPS}, 0
    with torch.no_grad():
        for b in samples:
            b = {k: v.to(device) for k, v in b.items()}
            pred = model(b["x"], b["e"], b["senders"], b["receivers"])[b["mask"]]
            for k, v in group_losses(pred, b["y"][b["mask"]]).items():
                tot[k] += v
            n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in tot.items()}


def fmt(d):
    return " ".join(f"{k}={v:.4f}" for k, v in d.items())


def main():
    args = parse_args()
    cfg = Config.from_dict(vars(args))
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.json")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = pick_device()

    runs = discover_runs(cfg.data_dir)
    if not runs:
        raise SystemExit(f"no runs found in {cfg.data_dir}")
    train_runs = [r for r in runs if r.tag not in cfg.holdout_tags]
    val_runs = [r for r in runs if r.tag in cfg.holdout_tags]
    print(f"device={device}  train={[r.tag for r in train_runs]}  holdout={[r.tag for r in val_runs]}")

    # -- normalisation ------------------------------------------------------
    stats_path = out / "stats.npz"
    if stats_path.exists():
        stats = Stats.load(stats_path)
        print(f"loaded stats from {stats_path}")
    else:
        t0 = time.time()
        raw = FrameDataset(train_runs, cfg, stats=None, train=False)
        stats = Stats.compute(raw.build(run, t, noise=False, crop=cfg.crop_half > 0, rng=rng)
                              for run, t in sample_frames(train_runs, cfg.stats_frames, rng, cfg.frame_stride))
        stats.save(stats_path)
        print(f"stats from {cfg.stats_frames} frames in {time.time() - t0:.0f}s -> {stats_path}")

    # -- data ---------------------------------------------------------------
    train_ds = FrameDataset(train_runs, cfg, stats, train=True, seed=cfg.seed)
    loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, collate_fn=collate,
                        num_workers=cfg.workers, persistent_workers=cfg.workers > 0, drop_last=True)
    val_ds = FrameDataset(val_runs or train_runs, cfg, stats, train=False, seed=cfg.seed + 1)
    val_samples = [collate([val_ds[i]]) for i in rng.choice(len(val_ds), size=min(cfg.val_samples, len(val_ds)), replace=False)]

    # -- model --------------------------------------------------------------
    model = GNS(cfg.hidden, cfg.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.1 ** (s / cfg.lr_decay_steps))
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        sched.last_epoch = step - 1   # LambdaLR has no state_dict entry for this; must set manually
        sched.step()
        print(f"resumed from {args.resume} at step {step}, lr={sched.get_last_lr()[0]:.1e}")
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    def save(name="model.pt"):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                    "config": cfg.to_dict(), "stats": {k: v.tolist() for k, v in stats.as_dict().items()}}, out / name)

    # -- loop ---------------------------------------------------------------
    model.train()
    t0 = time.time()
    running = {k: 0.0 for k in GROUPS}
    best_val = float("inf")
    while step < cfg.steps:
        for b in loader:
            if step >= cfg.steps:
                break
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            pred = model(b["x"], b["e"], b["senders"], b["receivers"])[b["mask"]]
            y = b["y"][b["mask"]]
            loss = torch.nn.functional.huber_loss(pred, y, delta=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1

            for k, v in group_losses(pred.detach(), y).items():
                running[k] += v
            if step % cfg.log_every == 0:
                avg = {k: v / cfg.log_every for k, v in running.items()}
                rate = cfg.log_every / (time.time() - t0)
                print(f"[{step:6d}] loss={sum(avg.values()) / len(avg):.4f}  {fmt(avg)}  "
                      f"lr={sched.get_last_lr()[0]:.1e}  nodes={b['x'].shape[0]} edges={b['e'].shape[0]}  {rate:.1f} it/s")
                running = {k: 0.0 for k in GROUPS}
                t0 = time.time()
            if step % cfg.val_every == 0:
                val = evaluate(model, val_samples, device)
                val_mean = sum(val.values()) / len(val)
                print(f"[{step:6d}] VAL loss={val_mean:.4f}  {fmt(val)}")
                save()
                if val_mean < best_val:
                    best_val = val_mean
                    save("best.pt")
    save()
    print(f"done, saved to {out}")


if __name__ == "__main__":
    main()
