"""Saved particle experiments retain their original neighbourhood and normalization."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from surrogate.particle import audit, finetune, rollout, train
from surrogate.particle.config import Config
from surrogate.particle.data import build_inputs
from test_particle_finetune import source_checkpoint, write_runs


LEGACY_MISSING = ("neighbors", "neighbor_history_frames", "training_mode")


def make_legacy(checkpoint):
    for key in LEGACY_MISSING:
        checkpoint["config"].pop(key, None)
    return checkpoint


class CheckpointCompatibilityTests(unittest.TestCase):
    def config(self, **overrides):
        settings = {"neighbors": 256, "history_frames": 2, "hidden": 8,
                    "batch": 2, "val_samples": 1, "stats_frames": 1,
                    "holdout": "phi30_c500", "device": "cpu"}
        return Config(**{**settings, **overrides})

    def test_disk_checkpoint_restores_legacy_defaults_without_changing_fresh_defaults(self):
        original = make_legacy(source_checkpoint(self.config(history_frames=8)))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.pt"
            torch.save(original, path)
            restored = train.load_checkpoint(path)
            cfg = Config.from_dict(restored["config"])
            self.assertEqual(cfg.neighbors, 256)
            self.assertEqual(cfg.neighbor_history, 9)
            self.assertEqual(cfg.training_mode, "random_particles")
            self.assertEqual(restored["stats"], original["stats"])
            self.assertEqual(torch.load(path, weights_only=True)["config"], original["config"])
        self.assertEqual(Config().neighbors, 32)
        self.assertEqual(Config.from_dict({}).neighbors, 32)

    def test_disk_checkpoint_preserves_explicit_new_experiment_settings(self):
        original = source_checkpoint(self.config(neighbors=32, history_frames=8,
                                                 neighbor_history_frames=3,
                                                 training_mode="full_frames"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "new.pt"
            torch.save(original, path)
            restored = train.load_checkpoint(path)
        self.assertEqual(restored["config"], original["config"])
        self.assertEqual(restored["stats"], original["stats"])
        cfg = Config.from_dict(restored["config"])
        self.assertEqual((cfg.neighbors, cfg.neighbor_history), (32, 3))

    def test_audit_restores_legacy_config_with_saved_target_scales(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = make_legacy(source_checkpoint(self.config()))["config"]
            (root / "config.json").write_text(json.dumps(settings))
            scales = np.arange(1, 17, dtype=np.float32)
            np.savez(root / "stats.npz", target_std=scales)
            with patch("sys.argv", ["audit", "--experiment", str(root)]), \
                    patch.object(audit, "audit") as run_audit:
                audit.main()
            run_audit.assert_called_once()
            cfg = run_audit.call_args.args[2]
            self.assertEqual((cfg.neighbors, cfg.neighbor_history), (256, 3))
            np.testing.assert_array_equal(run_audit.call_args.args[3], scales)

    def invoke_training(self, root, out, *extra):
        argv = ["train", "--data_dir", str(root / "data"), "--out_dir", str(out),
                "--holdout", "phi30_c500", "--history_frames", "2", "--hidden", "8",
                "--batch", "2", "--stats_frames", "1", "--val_samples", "1",
                "--val_every", "1", "--device", "cpu", *extra]
        with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
            train.main()

    def test_legacy_resume_matches_continuation_and_uses_embedded_stats(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_runs(root / "data")
            full, resumed = root / "full", root / "resumed"
            self.invoke_training(root, full, "--neighbors", "256",
                                 "--training_mode", "random_particles", "--steps", "2")
            self.invoke_training(root, resumed, "--neighbors", "256",
                                 "--training_mode", "random_particles", "--steps", "1")
            path = resumed / "model.pt"
            legacy = make_legacy(torch.load(path, weights_only=True))
            original_stats = legacy["stats"]
            torch.save(legacy, path)
            # An unrelated stale sidecar must never replace checkpoint normalization.
            np.savez(resumed / "stats.npz", **{key: np.full_like(value, 123.)
                                             for key, value in original_stats.items()})
            with patch.object(train.Stats, "compute", side_effect=AssertionError("recomputed stats")):
                self.invoke_training(root, resumed, "--resume", str(path), "--steps", "2")
            actual, expected = train.load_checkpoint(path), train.load_checkpoint(full / "model.pt")
            self.assertEqual(actual["step"], 2)
            cfg = Config.from_dict(actual["config"])
            self.assertEqual((cfg.neighbors, cfg.neighbor_history, cfg.training_mode),
                             (256, 3, "random_particles"))
            self.assertEqual(actual["stats"], original_stats)
            for key, value in expected["model"].items():
                torch.testing.assert_close(actual["model"][key], value, rtol=0, atol=0)
            self.assertEqual(actual["data_rng"], expected["data_rng"])
            torch.testing.assert_close(actual["torch_rng"], expected["torch_rng"], rtol=0, atol=0)
            with np.load(resumed / "stats.npz") as saved:
                for key, value in original_stats.items():
                    np.testing.assert_array_equal(saved[key], value)

    def test_resume_rejects_neighbour_changes_before_reading_data_or_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "legacy.pt"
            torch.save(make_legacy(source_checkpoint(self.config())), path)
            before = path.read_bytes()
            for key, value in (("neighbors", "32"), ("neighbor_history_frames", "1")):
                with self.subTest(key=key):
                    argv = ["train", "--resume", str(path), "--out_dir", str(root / "new"),
                            f"--{key}", value]
                    with patch("sys.argv", argv), patch.object(train, "discover_runs") as discover:
                        with self.assertRaisesRegex(ValueError, f"cannot change {key} on resume"):
                            train.main()
                    discover.assert_not_called()
                    self.assertEqual(list(root.iterdir()), [path])
                    self.assertEqual(path.read_bytes(), before)

    def test_rollout_cli_uses_legacy_neighbourhood(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            write_runs(root / "data")
            path, result = root / "legacy.pt", root / "rollout.npz"
            checkpoint = make_legacy(source_checkpoint(self.config(data_dir=str(root / "data"))))
            torch.save(checkpoint, path)
            argv = ["rollout", "--ckpt", str(path), "--tag", "phi30_c500", "--steps", "1",
                    "--device", "cpu", "--out", str(result), "--neighbor_workers", "1"]
            with patch("sys.argv", argv), patch.object(rollout, "build_inputs", wraps=build_inputs) as inputs:
                rollout.main()
            self.assertTrue(inputs.called)
            for call in inputs.call_args_list:
                self.assertEqual(call.args[5], 256)
                self.assertEqual(call.kwargs["neighbor_history"], 3)
            with np.load(result) as saved:
                self.assertEqual(int(saved["neighbors"]), 256)
                self.assertEqual(int(saved["history_frames"]), 2)

    def test_direct_finetuning_migrates_legacy_config_and_preserves_stats(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            write_runs(root / "data")
            checkpoint = make_legacy(source_checkpoint(self.config()))
            options = finetune.FinetuneConfig(
                data_dir=str(root / "data"), out_dir=str(root / "finetuned"), cycles=1,
                rollout_steps=2, val_steps=1, updates_per_cycle=1, batch=2,
                inference_batch=2, device="cpu")
            path = finetune.run_finetuning(checkpoint, options)
            saved = train.load_checkpoint(path)
            cfg = Config.from_dict(saved["config"])
            self.assertEqual((cfg.neighbors, cfg.neighbor_history), (256, 3))
            self.assertEqual(saved["stats"], checkpoint["stats"])
            self.assertEqual(saved["finetune"]["completed_cycles"], 1)
            self.assertTrue(all(key not in checkpoint["config"] for key in LEGACY_MISSING))


if __name__ == "__main__":
    unittest.main()
