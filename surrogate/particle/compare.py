"""Read particle predictions and solver values from a result, without model weights.

    python -m surrogate.particle.compare --result checkpoints/comparison.npz
    python -m surrogate.particle.compare --result checkpoints/comparison.npz \
        --particle_ids 123 --csv checkpoints/particle_123.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_comparison(path, frame=None, particle_ids=None):
    """Select by original solver frame/particle ID, never by nearest position."""
    keys = ("particle_predicted", "particle_reference", "particle_frames", "particle_t",
            "particle_ids", "feature_names", "feature_units")
    with np.load(path, allow_pickle=False) as archive:
        missing = set(keys) - set(archive.files)
        if missing:
            raise ValueError("this result has no full particle comparison; rerun rollout with --save_particles")
        data = {key: archive[key] for key in keys}
        data["mode"] = str(archive["mode"]) if "mode" in archive else "unknown"
        data["tag"] = str(archive["tag"]) if "tag" in archive else "unknown"
    frames, ids = data["particle_frames"], data["particle_ids"]
    shape = (len(frames), len(ids), len(data["feature_names"]))
    if data["particle_predicted"].shape != shape or data["particle_reference"].shape != shape:
        raise ValueError("prediction/reference shapes do not match frame, particle and feature IDs")
    if len(data["particle_t"]) != len(frames) or len(data["feature_units"]) != shape[-1]:
        raise ValueError("time/unit metadata does not match the stored comparison")
    if len(np.unique(ids)) != len(ids) or len(np.unique(frames)) != len(frames):
        raise ValueError("duplicate particle or frame IDs in comparison")
    frame_rows = np.arange(len(frames)) if frame is None else np.flatnonzero(frames == frame)
    if not len(frame_rows):
        raise ValueError(f"frame {frame} is not a stored prediction frame")
    if particle_ids is None:
        particle_rows = np.arange(len(ids))
    else:
        requested = list(dict.fromkeys(particle_ids))
        positions = {int(value): i for i, value in enumerate(ids)}
        missing = set(requested) - set(positions)
        if missing:
            raise ValueError(f"unknown soil particle IDs: {sorted(missing)}")
        particle_rows = np.array([positions[value] for value in requested], np.int64)
    if not len(particle_rows):
        raise ValueError("select at least one particle")
    for key in ("particle_predicted", "particle_reference"):
        data[key] = data[key][frame_rows[:, None], particle_rows[None, :], :]
        if not np.isfinite(data[key]).all():
            raise ValueError(f"non-finite values in {key}")
    data["particle_ids"] = ids[particle_rows]
    data["particle_frames"] = frames[frame_rows]
    data["particle_t"] = data["particle_t"][frame_rows]
    return data


def error_statistics(error):
    """Physical errors; no division by near-zero reference values."""
    absolute = np.abs(error)
    p50, p95, p99 = np.percentile(absolute, [50, 95, 99])
    return {"rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(absolute.mean()), "bias": float(error.mean()),
            "abs_p50": float(p50), "abs_p95": float(p95), "abs_p99": float(p99),
            "abs_max": float(absolute.max())}


def summarize(data):
    predicted, reference = data["particle_predicted"], data["particle_reference"]
    features = {}
    for col, (name, unit) in enumerate(zip(data["feature_names"], data["feature_units"])):
        error = predicted[..., col].astype(np.float64) - reference[..., col]
        features[str(name)] = {"unit": str(unit), **error_statistics(error)}
    columns = [list(data["feature_names"]).index(name) for name in ("x", "y", "z")]
    displacement_error = predicted[..., columns].astype(np.float64) - reference[..., columns]
    position_error = np.linalg.norm(displacement_error, axis=-1)
    position = error_statistics(position_error)
    # Directional bias has no meaning for a vector norm.
    position.pop("bias")
    worst = np.argsort(position_error.ravel())[-min(10, position_error.size):][::-1]
    worst_rows = []
    for flat in worst:
        fi, pi = np.unravel_index(flat, position_error.shape)
        worst_rows.append({"frame": int(data["particle_frames"][fi]),
                           "particle_id": int(data["particle_ids"][pi]),
                           "position_error_m": float(position_error[fi, pi])})
    return {"tag": data["tag"], "mode": data["mode"],
            "frames": data["particle_frames"].tolist(),
            "particles_per_frame": len(data["particle_ids"]),
            "particle_frame_pairs": int(position_error.size),
            "features": features, "position": {"unit": "m", **position},
            "worst_position_errors": worst_rows}


def write_csv(path, data):
    """One row = one frame, particle ID and physical quantity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "time_s", "particle_id", "quantity", "unit",
                         "solver", "prediction", "error", "absolute_error"])
        for fi, (frame, seconds) in enumerate(zip(data["particle_frames"], data["particle_t"])):
            for pi, particle_id in enumerate(data["particle_ids"]):
                for col, (name, unit) in enumerate(zip(data["feature_names"], data["feature_units"])):
                    truth = float(data["particle_reference"][fi, pi, col])
                    prediction = float(data["particle_predicted"][fi, pi, col])
                    error = prediction - truth
                    writer.writerow([int(frame), float(seconds), int(particle_id), name, unit,
                                     truth, prediction, error, abs(error)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, help="rollout npz created with --save_particles")
    parser.add_argument("--frame", type=int, help="original solver frame ID; default: every predicted frame")
    parser.add_argument("--particle_ids", nargs="+", type=int, help="original soil IDs; default: every soil particle")
    parser.add_argument("--csv", help="optional detailed table for the selected frames and particles")
    parser.add_argument("--out", help="summary json path")
    args = parser.parse_args()
    data = load_comparison(args.result, args.frame, args.particle_ids)
    report = summarize(data)
    print(f"{report['tag']} mode={report['mode']}; {len(report['frames'])} predicted frames, "
          f"{report['particles_per_frame']:,} particles/frame; warm start excluded")
    print(f"{'quantity':10s} {'unit':8s} {'RMSE':>12s} {'MAE':>12s} {'abs p95':>12s} {'abs max':>12s}")
    for name, row in report["features"].items():
        print(f"{name:10s} {row['unit']:8s} {row['rmse']:12.5g} {row['mae']:12.5g} "
              f"{row['abs_p95']:12.5g} {row['abs_max']:12.5g}")
    position = report["position"]
    print(f"position error [mm]: RMSE={position['rmse']*1000:.4g} "
          f"p50={position['abs_p50']*1000:.4g} p95={position['abs_p95']*1000:.4g} "
          f"p99={position['abs_p99']*1000:.4g} max={position['abs_max']*1000:.4g}")
    worst = report["worst_position_errors"][0]
    print(f"largest position error: frame={worst['frame']} particle_id={worst['particle_id']}")
    suffix = "_selection_summary.json" if args.frame is not None or args.particle_ids is not None else "_summary.json"
    out = Path(args.out) if args.out else Path(args.result).with_name(Path(args.result).stem + suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"saved {out}")
    if args.csv:
        write_csv(args.csv, data)
        print(f"saved {args.csv}")


if __name__ == "__main__":
    main()
