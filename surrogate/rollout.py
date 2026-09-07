"""Autoregressive rollout of a trained model against the solver.

    python -m surrogate.rollout --ckpt checkpoints/gns/best.pt --tag phi30_c500 --plot

Prints: per-step position RMSE, steps until divergence, plate pressure curve vs
reference, wall-clock per model-second for the net and (if runs_soil.csv is
present) for the solver.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import Run, POS, STATE, P33, SOIL, PLATE, solver_wall_seconds
from .graph import radius_edges, node_features, edge_features
from .model import GNS, pick_device
from .normalize import Stats


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device)
    cfg = Config.from_dict(ck["config"])
    stats = Stats.load(ck["stats"])
    model = GNS(cfg.hidden, cfg.layers).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, stats


def plate_pressure(soil_pos, soil_p33, plate_pos, spacing, plate_radius=0.15):
    """Mean normal stress of soil in a thin layer right under the plate face.

    Sign convention of p33 in Chrono CRM must be checked once against the
    reference: run with --calibrate, which prints this estimate on ground-truth
    frames next to pressure_sinkage.csv. If they are anti-correlated, flip the sign here.
    """
    center = plate_pos[:, :2].mean(0)
    z_bottom = plate_pos[:, 2].min()
    r = np.linalg.norm(soil_pos[:, :2] - center, axis=1)
    layer = (r <= plate_radius) & (soil_pos[:, 2] <= z_bottom) & (soil_pos[:, 2] >= z_bottom - 2 * spacing)
    if not layer.any():
        return np.nan
    return -float(soil_p33[layer].mean())


def rollout(model, cfg, stats, run, device, n_steps=None, verbose=True):
    """Start from frames k and k+1, predict soil for frames k+1..T-k.
    Plate and wall positions/states come from the data at every step (their motion is known)."""
    k = cfg.frame_stride
    dt = cfg.dt * k
    T = run.n_frames if n_steps is None else min(run.n_frames, n_steps + 2)
    types = run.types
    soil = types == SOIL
    recv = None if cfg.edges_to_static else soil

    f0, f1 = run.frame(0), run.frame(k)
    pos_prev, pos_t = f0[:, POS].copy(), f1[:, POS].copy()
    state = f1[:, STATE].copy()

    pred_pos = np.empty((T, run.n_soil, 3), np.float32)
    pred_state = np.empty((T, run.n_soil, state.shape[1]), np.float32)
    pred_pos[0], pred_pos[k] = pos_prev[soil], pos_t[soil]
    pred_state[0], pred_state[k] = f0[soil, STATE], state[soil]
    rmse = np.zeros(T, np.float32)
    step_wall = []

    with torch.inference_mode():
        for t in range(k, T - k, k):
            tic = time.perf_counter()
            s, r = radius_edges(pos_t, cfg.radius, receiver_mask=recv)
            vel = (pos_t - pos_prev) / dt
            x = stats.norm("node", node_features(pos_t, pos_prev, state, types, run.phi_deg, run.cohesion, dt))
            e = stats.norm("edge", edge_features(pos_t, vel, s, r))
            out = model(torch.from_numpy(x).to(device), torch.from_numpy(e).to(device),
                        torch.from_numpy(s).to(device), torch.from_numpy(r).to(device))
            y = stats.denorm("target", out.cpu().numpy())
            acc, rate = y[:, :3], y[:, 3:]

            pos_next = pos_t.copy()
            pos_next[soil] = 2 * pos_t[soil] - pos_prev[soil] + acc[soil] * dt ** 2
            state[soil] += rate[soil] * dt

            # static markers: take the next frame from the data
            f_next = run.frame(t + k)
            pos_next[~soil] = f_next[~soil, POS]
            state[~soil] = f_next[~soil, STATE]
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_wall.append(time.perf_counter() - tic)

            pred_pos[t + k], pred_state[t + k] = pos_next[soil], state[soil]
            rmse[t + k] = np.sqrt(((pos_next[soil] - f_next[soil, POS]) ** 2).sum(1).mean())
            if verbose and (t % 10 == 0 or t == T - 2):
                print(f"  step {t:3d}/{T - 2}  rmse={rmse[t + k] * 1e3:.3f} mm  {step_wall[-1] * 1e3:.0f} ms")
            pos_prev, pos_t = pos_t, pos_next

    frames = np.arange(0, T - k + 1, k)          # frames actually filled: 0, k, 2k, ...
    return {"pos": pred_pos, "state": pred_state, "rmse": rmse, "frames": frames,
            "step_wall": np.array(step_wall)}


STATE_NAMES = ["rho", "p11", "p22", "p33", "shear12", "shear13", "shear23", "pc", "Ev", "Sv"]


def zone_report(run, pred, cfg, plate_radius=0.15):
    """Where is the error: under the plate or in soil that should be at rest?
    Prints RMSE, the mean error vector (a drift shows up as a non-zero mean), and
    percentiles of the per-particle error -- a single mean can hide "almost every
    particle is fine, a handful are wildly off" behind a merely-large number."""
    T = int(pred["frames"][-1])
    gt = np.asarray(run.soil[T][:, POS])
    err = pred["pos"][T] - gt
    dist = np.linalg.norm(err, axis=1) * 1e3   # per-particle position error, mm
    plate0 = np.asarray(run.plate[0][:, POS])
    center, z_top = plate0[:, :2].mean(0), plate0[:, 2].min()
    lateral = np.linalg.norm(gt[:, :2] - center, axis=1)
    under = (lateral <= plate_radius + 2 * cfg.spacing) & (gt[:, 2] >= z_top - 0.15)
    moved = np.linalg.norm(gt - np.asarray(run.soil[0][:, POS]), axis=1) > cfg.spacing / 4
    print(f"\nposition error split at the last frame (n = {len(gt)}), per-particle |error| in mm:")
    print(f"  {'':28s} {'n':>6}  {'p50':>7} {'p90':>7} {'p99':>7} {'max':>8}   mean vector (x,y,z)")
    for name, m in (("under plate", under), ("rest of soil", ~under), ("moved > spacing/4 in truth", moved), ("static in truth", ~moved)):
        if not m.any():
            continue
        p50, p90, p99, pmax = np.percentile(dist[m], [50, 90, 99, 100])
        mean = err[m].mean(0) * 1e3
        print(f"  {name:28s} {m.sum():6d}  {p50:7.2f} {p90:7.2f} {p99:7.2f} {pmax:8.2f}   ({mean[0]:+.2f}, {mean[1]:+.2f}, {mean[2]:+.2f})")


def state_report(run, pred, cfg):
    """Per-particle error for each of the 10 predicted state features separately
    (density, the 6 stress components, pc/Ev/Sv), instead of the training loop's
    4 coarse groups (acc/rho/stress/plast) which hide which specific quantity is
    actually well or badly predicted."""
    T = int(pred["frames"][-1])
    gt_state = np.asarray(run.soil[T][:, STATE])
    pred_state = pred["state"][T]
    print(f"\nstate error at the last frame, per feature (n = {len(gt_state)}):")
    print(f"  {'feature':10s} {'true std':>10}  {'p50':>9} {'p90':>9} {'p99':>9}   (errors in the feature's own units)")
    for i, name in enumerate(STATE_NAMES):
        true_std = gt_state[:, i].std()
        abs_err = np.abs(pred_state[:, i] - gt_state[:, i])
        p50, p90, p99 = np.percentile(abs_err, [50, 90, 99])
        print(f"  {name:10s} {true_std:10.3g}  {p50:9.3g} {p90:9.3g} {p99:9.3g}")


def pressure_curves(run, pred, cfg):
    """(t, p_pred, p_gt_estimate, p_reference) per frame."""
    frames = pred["frames"]
    t = frames * cfg.dt
    p_pred, p_gt = np.full(len(frames), np.nan), np.full(len(frames), np.nan)
    for i, k in enumerate(frames):
        plate_pos = np.asarray(run.plate[k][:, POS])
        p_pred[i] = plate_pressure(pred["pos"][k], pred["state"][k][:, P33 - STATE.start], plate_pos, cfg.spacing)
        gt = np.asarray(run.soil[k])
        p_gt[i] = plate_pressure(gt[:, POS], gt[:, P33], plate_pos, cfg.spacing)
    p_ref = None
    if run.reference is not None and {"time_s", "pressure_Pa"} <= set(run.reference.columns):
        ref = run.reference
        p_ref = np.interp(t, ref["time_s"].to_numpy() - ref["time_s"].iloc[0], ref["pressure_Pa"].to_numpy())
    return t, p_pred, p_gt, p_ref


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data_dir", default=None, help="defaults to the one in the checkpoint config")
    ap.add_argument("--steps", type=int, default=None, help="limit rollout length")
    ap.add_argument("--threshold", type=float, default=None, help="RMSE [m] that counts as diverged (default spacing/2)")
    ap.add_argument("--calibrate", action="store_true", help="only check the plate-pressure estimator on ground truth")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default=None, help="npz to save predictions into")
    args = ap.parse_args()

    device = pick_device()
    model, cfg, stats = load_checkpoint(args.ckpt, device)
    run = Run(args.data_dir or cfg.data_dir, args.tag)
    thr = args.threshold or cfg.spacing / 2
    print(f"{run}  frames={run.n_frames}  device={device}")

    if args.calibrate:
        T = run.n_frames
        for k in range(0, T, max(1, T // 10)):
            gt = np.asarray(run.soil[k])
            est = plate_pressure(gt[:, POS], gt[:, P33], np.asarray(run.plate[k][:, POS]), cfg.spacing)
            ref = "n/a"
            if run.reference is not None:
                ref = np.interp(k * cfg.dt, run.reference["time_s"] - run.reference["time_s"].iloc[0], run.reference["pressure_Pa"])
                ref = f"{ref / 1e3:.1f} kPa"
            print(f"  frame {k:3d}: estimate {est / 1e3:8.1f} kPa   reference {ref}")
        return

    pred = rollout(model, cfg, stats, run, device, n_steps=args.steps)

    # -- divergence ---------------------------------------------------------
    frames = pred["frames"]
    rmse = pred["rmse"][frames]                # one value per network step
    over = np.flatnonzero(rmse > thr)
    stable = int(over[0]) - 1 if len(over) else len(rmse) - 1
    print(f"\nstable steps (rmse < {thr * 1e3:.1f} mm): {stable} of {len(rmse) - 1}  (stride {cfg.frame_stride}, {len(rmse) - 1} steps cover frames 0..{frames[-1]})")
    print(f"final rmse: {rmse[-1] * 1e3:.2f} mm")
    zone_report(run, pred, cfg)
    state_report(run, pred, cfg)

    # -- plate pressure -----------------------------------------------------
    t, p_pred, p_gt, p_ref = pressure_curves(run, pred, cfg)
    ok = ~np.isnan(p_pred) & ~np.isnan(p_gt)
    print(f"plate pressure, net vs particle estimate on ground truth: rel. error {np.abs(p_pred[ok] - p_gt[ok]).mean() / (np.abs(p_gt[ok]).mean() + 1e-9):.2%}")
    if p_ref is not None:
        print(f"plate pressure, net vs solver reference:                 rel. error {np.abs(p_pred[ok] - p_ref[ok]).mean() / (np.abs(p_ref[ok]).mean() + 1e-9):.2%}")

    # -- speed --------------------------------------------------------------
    wall = pred["step_wall"][1:].mean()        # skip warm-up step
    net_cost = wall / (cfg.dt * cfg.frame_stride)
    print(f"\nnet: {wall * 1e3:.0f} ms per step incl. graph build  ->  {net_cost:.1f} wall-s per model-s")
    solver = solver_wall_seconds(run.dir, run.tag)
    if solver is not None:
        model_seconds = (run.n_frames - 1) * cfg.dt
        solver_cost = solver / model_seconds
        print(f"solver: {solver_cost:.1f} wall-s per model-s  ->  speedup x{solver_cost / net_cost:.1f}")
    else:
        print("solver time unknown (no runs_soil.csv with wall_seconds in data dir)")

    if args.out:
        np.savez(args.out, **pred, t=t, p_pred=p_pred, p_gt=p_gt, p_ref=p_ref if p_ref is not None else np.array([]))
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(t[1:], rmse[1:] * 1e3)
        ax[0].axhline(thr * 1e3, ls="--", c="r")
        ax[0].set(xlabel="t [s]", ylabel="position RMSE [mm]", title=f"{run.tag}: divergence")
        ax[1].plot(t, p_gt / 1e3, label="particle estimate, solver frames")
        ax[1].plot(t, p_pred / 1e3, label="particle estimate, net rollout")
        if p_ref is not None:
            ax[1].plot(t, p_ref / 1e3, "k--", label="pressure_sinkage.csv")
        ax[1].set(xlabel="t [s]", ylabel="plate pressure [kPa]", title="pressure-sinkage")
        ax[1].legend()
        png = Path(args.ckpt).with_name(f"rollout_{run.tag}.png")
        fig.tight_layout()
        fig.savefig(png, dpi=120)
        print(f"plot -> {png}")


if __name__ == "__main__":
    main()
