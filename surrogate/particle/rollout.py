"""Synchronous particle-by-particle prediction of a whole trajectory.

    python -m surrogate.particle.rollout --ckpt checkpoints/particle/best.pt --tag phi30_c500 --plot
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from ..data import P33, POS, SOIL, STATE, Run, solver_wall_seconds
from ..model import pick_device
from ..rollout import plate_pressure
from .config import Config
from .data import FEATURE_NAMES, FEATURE_UNITS, Stats, build_inputs, input_features, to_tensors
from .model import ParticleNet, predict
from .timing import STAGES, StepTimer, measure
from .train import load_checkpoint


@torch.inference_mode()
def predict_next_frame(model, stats, history, types, phi_deg, cohesion, cfg, device,
                       boundary_kinematics, batch_size=None, timer=None):
    """All targets read the same history ending at t, regardless of batching.

    boundary_kinematics contains ONLY next prescribed x,y,z,vx,vy,vz for the
    non-soil markers in their existing order. No future soil or BCE stresses.
    """
    batch_size = cfg.batch if batch_size is None else batch_size
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if history.ndim != 3 or len(history) != cfg.history_frames + 1:
        raise ValueError("history must contain history_frames previous frames plus the current frame")
    current = history[-1]
    soil = types == SOIL
    if boundary_kinematics.shape != (int((~soil).sum()), 6):
        raise ValueError("boundary_kinematics must contain 6 known columns per boundary marker")
    targets = np.flatnonzero(soil)
    with measure(timer, "tree"):
        tree = cKDTree(current[:, POS])
    with measure(timer, "features"):
        features = np.stack([input_features(frame, types, phi_deg, cohesion) for frame in history])
    next_frame = current.copy()
    for start in range(0, len(targets), batch_size):
        ids = targets[start:start + batch_size]
        inputs = build_inputs(history, types, phi_deg, cohesion, ids, cfg.neighbors,
                              tree=tree, features=features, timer=timer)
        with measure(timer, "normalize"):
            tensors = to_tensors(inputs, stats)
        with measure(timer, "to_device"):
            batch = {key: value.to(device) for key, value in tensors.items()}
        with measure(timer, "network"):
            predicted = predict(model, batch)
        with measure(timer, "to_cpu"):
            normalized_delta = predicted.cpu().numpy()
        with measure(timer, "update"):
            delta = stats.denorm("target", normalized_delta)
            next_frame[ids] = current[ids] + delta
    with measure(timer, "update"):
        next_frame[~soil, :6] = boundary_kinematics
        next_frame[~soil, STATE] = 0
        if not np.isfinite(next_frame).all():
            raise FloatingPointError("non-finite particle prediction during rollout")
    return next_frame


def rollout(model, cfg, stats, run, device, n_steps=None, batch_size=None, verbose=True,
            save_particles=False, profile=False, start_frame=None):
    """Predict after a true history window ending at ``start_frame``.

    ``start_frame`` is an index in the original run, not a prediction count.
    The window spans ``history_frames * frame_stride`` original frames; its
    stride may start at any offset. By default, use the earliest full window.
    """
    history_span = cfg.history_frames * cfg.frame_stride
    initial_frame = history_span if start_frame is None else start_frame
    if isinstance(initial_frame, (bool, np.bool_)) or not isinstance(initial_frame, (int, np.integer)):
        raise ValueError("start_frame must be an integer frame index")
    if initial_frame < history_span:
        raise ValueError(f"start_frame must be at least {history_span} to provide the full history")
    frames = np.arange(initial_frame, run.n_frames, cfg.frame_stride)
    if n_steps is not None:
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        frames = frames[:n_steps + 1]
    if len(frames) < 2:
        raise ValueError("not enough frames for one prediction")
    model.eval()
    types, soil = run.types, run.types == SOIL
    history = np.stack([run.frame(t) for t in range(initial_frame - history_span,
                                                  initial_frame + 1, cfg.frame_stride)])
    history[:, ~soil, STATE] = 0
    current = history[-1]
    rmse = np.zeros(len(frames), np.float64)
    feature_rmse = np.zeros((len(frames), len(FEATURE_NAMES)), np.float64)
    p_pred = np.empty(len(frames), np.float64)
    p_gt = np.empty_like(p_pred)
    wall = []
    evaluation_wall, stage_seconds = [], []
    if save_particles:
        shape = (len(frames) - 1, int(soil.sum()), len(FEATURE_NAMES))
        particle_predicted = np.empty(shape, np.float32)
        particle_reference = np.empty_like(particle_predicted)
    spacing = 0.02                   # fixed resolution of this data set
    p_pred[0] = p_gt[0] = plate_pressure(current[soil, POS], current[soil, P33],
                                       run.plate[initial_frame, :, POS], spacing)
    for index, frame in enumerate(frames[1:], start=1):
        timer = StepTimer(device) if profile else None
        tic = time.perf_counter()
        # Read ONLY known boundary motion before prediction. Ground-truth soil
        # is read afterwards, exclusively to evaluate the result.
        boundary_motion = np.concatenate([run.plate[frame, :, :6], run.boundary[:, :6]])
        current = predict_next_frame(model, stats, history, types, run.phi_deg, run.cohesion,
                                     cfg, device, boundary_motion, batch_size,
                                     **({"timer": timer} if timer else {}))
        with measure(timer, "history"):
            history = np.concatenate([history[1:], current[None]], axis=0)
        wall.append(time.perf_counter() - tic)
        if timer:
            stage_seconds.append(timer.finish(wall[-1]))
        evaluation_start = time.perf_counter()
        truth = np.asarray(run.soil[frame], np.float32)
        if save_particles:
            # Predictions and truth share the original soil row IDs. The warm
            # start is deliberately excluded: every stored row is a prediction.
            particle_predicted[index - 1] = current[soil]
            particle_reference[index - 1] = truth
        error = current[soil].astype(np.float64) - truth
        feature_rmse[index] = np.sqrt((error ** 2).mean(0))
        rmse[index] = np.sqrt((error[:, POS] ** 2).sum(1).mean())
        plate_pos = np.asarray(run.plate[frame, :, POS])
        p_pred[index] = plate_pressure(current[soil, POS], current[soil, P33], plate_pos, spacing)
        p_gt[index] = plate_pressure(truth[:, POS], truth[:, P33], plate_pos, spacing)
        evaluation_wall.append(time.perf_counter() - evaluation_start)
        if verbose and (index == 1 or index % 10 == 0 or index == len(frames) - 1):
            print(f"step {index}/{len(frames)-1} frame={frame} position_RMSE={rmse[index]*1000:.3f}mm "
                  f"{wall[-1]:.3f}s/step", flush=True)
            if timer:
                print("  profile [s]: " + " ".join(
                    f"{key}={timer.seconds[key]:.3f}" for key in STAGES), flush=True)
    result = {"frames": frames, "t": frames * cfg.dt, "rmse": rmse, "feature_rmse": feature_rmse,
              "p_pred": p_pred, "p_gt": p_gt, "step_wall": np.asarray(wall),
              "evaluation_wall": np.asarray(evaluation_wall), "mode": np.asarray("rollout"),
              "initial_frame": np.asarray(initial_frame)}
    if save_particles:
        result.update(particle_predicted=particle_predicted, particle_reference=particle_reference,
                      particle_ids=np.flatnonzero(soil), particle_frames=frames[1:],
                      particle_t=frames[1:] * cfg.dt,
                      feature_names=np.asarray(FEATURE_NAMES), feature_units=np.asarray(FEATURE_UNITS))
    if profile:
        result.update(profile_stages=np.asarray(STAGES), profile_seconds=np.asarray(stage_seconds))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=None, help="predictions after the initial true history window")
    parser.add_argument("--start_frame", type=int, default=None,
                        help="original index of the last true history frame; first prediction is start_frame + "
                             "frame_stride (default: history_frames * frame_stride, the earliest full window)")
    parser.add_argument("--batch", type=int, default=None, help="targets per inference batch; does not change neighbours")
    parser.add_argument("--out", default=None, help="output npz; add --save_particles for full comparison arrays")
    parser.add_argument("--save_particles", action="store_true",
                        help="save all 16 predicted and reference values for every soil ID and predicted frame")
    parser.add_argument("--profile", action="store_true",
                        help="time search, input preparation, transfers and network; synchronizes GPU stages")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.ckpt)
    cfg, stats = Config.from_dict(checkpoint["config"]), Stats(checkpoint["stats"])
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    model = ParticleNet(cfg.hidden).to(device)
    model.load_state_dict(checkpoint["model"])
    run = Run(args.data_dir or cfg.data_dir, args.tag)
    batch_size = cfg.batch if args.batch is None else args.batch
    if batch_size < 1:
        parser.error("--batch must be positive")
    print(f"{run} device={device} K={cfg.neighbors} history={cfg.history_frames}+current "
          f"dt={cfg.dt * cfg.frame_stride:g}s", flush=True)
    print(f"checkpoint step={checkpoint['step']}; {run.n_soil:,} soil particles; "
          f"{(run.n_soil + batch_size - 1) // batch_size:,} network batches/frame; "
          f"{run.n_soil * cfg.neighbors:,} neighbour histories/frame", flush=True)
    if args.profile:
        print("Profile synchronizes GPU stages; compare throughput separately without --profile.", flush=True)
    result = rollout(model, cfg, stats, run, device, args.steps, args.batch,
                     save_particles=args.save_particles, profile=args.profile, start_frame=args.start_frame)
    result.update(tag=np.asarray(run.tag), checkpoint_step=np.asarray(checkpoint["step"]),
                  batch_size=np.asarray(batch_size), neighbors=np.asarray(cfg.neighbors),
                  history_frames=np.asarray(cfg.history_frames), dt=np.asarray(cfg.dt * cfg.frame_stride))
    over = np.flatnonzero(result["rmse"][1:] > .01)
    stable = int(over[0]) if len(over) else len(result["frames"]) - 1
    print(f"predicted steps before position RMSE exceeds 10mm: {stable}/{len(result['frames'])-1}")
    print("final per-feature RMSE in original units: " +
          " ".join(f"{name}={value:.4g}" for name, value in zip(FEATURE_NAMES, result["feature_rmse"][-1])))
    if run.reference is not None:
        ref = run.reference
        result["p_reference"] = np.interp(result["t"], ref["time_s"] - ref["time_s"].iloc[0], ref["pressure_Pa"])
    for key in ("p_gt", "p_reference"):
        if key not in result:
            continue
        valid = np.isfinite(result["p_pred"][1:]) & np.isfinite(result[key][1:])
        print(f"pressure frames with a valid estimate: {valid.sum()}/{len(valid)} ({key})")
        if valid.any():
            predicted, reference = result["p_pred"][1:][valid], result[key][1:][valid]
            error = np.abs(predicted - reference).mean() / (np.abs(reference).mean() + 1e-9)
            print(f"pressure relative error against {key}: {error:.2%}")
    times = result["step_wall"]
    step_seconds = float((times[1:] if len(times) > 1 else times).mean())
    net_cost = step_seconds / (cfg.dt * cfg.frame_stride)
    print(f"net: {step_seconds:.3f}s/step including neighbour search; {net_cost:.1f} wall-s/model-s")
    if args.profile:
        stages = result["profile_seconds"]
        means = (stages[1:] if len(stages) > 1 else stages).mean(0)
        print("mean profile per predicted frame (same frames as net timing):")
        for name, seconds in zip(STAGES, means):
            print(f"  {name:12s} {seconds:9.4f}s  {seconds / step_seconds:6.1%}")
    print(f"comparison/recording: {np.mean(result['evaluation_wall']):.3f}s/step (outside net timing)")
    solver = solver_wall_seconds(run.dir, run.tag)
    if solver is not None:
        print(f"speedup: {solver / ((run.n_frames-1)*cfg.dt) / net_cost:.2f}x")
    out = Path(args.out) if args.out else Path(args.ckpt).with_name(f"particle_rollout_{run.tag}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_start = time.perf_counter()
    np.savez_compressed(out, **result)
    print(f"saved {out}")
    print(f"result export: {time.perf_counter() - save_start:.2f}s (outside net timing)")
    if args.save_particles:
        print(f"particle comparison: {result['particle_predicted'].shape} = "
              "(predicted frames, soil IDs, 16 quantities); warm start excluded")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(result["t"], result["rmse"] * 1000)
        ax[0].axhline(10, ls="--", color="r")
        ax[0].set(xlabel="t [s]", ylabel="position RMSE [mm]", title=run.tag)
        for key, label in (("p_pred", "particle net"), ("p_gt", "solver particle estimate"),
                           ("p_reference", "solver reference")):
            if key in result:
                ax[1].plot(result["t"], result[key] / 1000, label=label)
        ax[1].set(xlabel="t [s]", ylabel="plate pressure [kPa]")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(out.with_suffix(".png"), dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    main()
