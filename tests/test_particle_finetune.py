"""Recovery labels, causality, baseline protection and cycle-boundary resume."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from surrogate.data import PLATE, SOIL, WALL
from surrogate.particle.config import Config
from surrogate.particle.data import Stats
from surrogate.particle.finetune import FinetuneConfig, main, run_finetuning, trajectory_metrics
from surrogate.particle.model import ParticleNet
from surrogate.particle.replay import RolloutReplay, rollout_windows
from surrogate.particle.train import CHECKPOINT_KIND, load_checkpoint


def unit_stats():
    return Stats({f"{key}_{name}": np.full(size, value, np.float32)
                  for key, size in (("node", 18), ("edge", 7), ("target", 16))
                  for name, value in (("mean", 0), ("std", 1))})


def source_checkpoint(cfg):
    torch.manual_seed(7)
    model = ParticleNet(cfg.hidden)
    torch.nn.init.normal_(model.decoder[-1].weight, std=.005)
    return {"kind": CHECKPOINT_KIND, "config": cfg.to_dict(), "stats": unit_stats().to_dict(),
            "model": model.state_dict(), "step": 20}


def write_runs(folder):
    folder.mkdir()
    for tag, factor in (("phi35_c1000", 1.), ("phi30_c500", 1.5)):
        soil = np.zeros((9, 4, 16), np.float32)
        soil[..., 0] = np.array([0., .02, .04, .06])
        soil[..., 2] = -.01
        soil[..., 6] = 1600
        soil[..., 9] = -500
        for t in range(9):
            soil[t, :, 3] = .002 + t * .0001 * factor
            soil[t, :, 0] += .00004 * t * factor
        plate = np.zeros((9, 1, 16), np.float32)
        plate[:, :, 2] = -.00001 * np.arange(9)[:, None]
        plate[..., 5] = -.0005
        plate[..., 9] = -99999  # forbidden boundary response
        wall = np.zeros((1, 16), np.float32)
        wall[:, :3] = [-.02, 0., -.02]
        wall[:, 9] = -77777
        np.save(folder / f"{tag}.npy", soil)
        np.save(folder / f"{tag}_plate.npy", plate)
        np.save(folder / f"{tag}_boundary.npy", wall)


class ReplayTests(unittest.TestCase):
    def fixture(self, reference_offset=0.):
        cfg = Config(history_frames=1, neighbors=1, batch=1)
        source = np.zeros((4, 5, 16), np.float32)
        source[:, :, 0] = [2., 2.1, 3., 50., -50.]
        source[..., 6] = 1600
        source[:, 3:, 9] = -9999
        source[3, 0, 0] = 999
        source[2:, :3] += reference_offset

        class Run:
            types = np.array([SOIL, SOIL, SOIL, PLATE, WALL])
            phi_deg, cohesion, n_frames, n_soil = 35., 1000., 4, 3
            plate, boundary = source[:, 3:4].copy(), source[0, 4:].copy()

            def frame(self, t):
                if t > 1:
                    raise AssertionError("future solver frame used to build replay")
                return source[t].copy()

        predicted = np.zeros((2, 3, 16), np.float32)
        predicted[0, :, 0] = [7., 7.02, 100.]
        predicted[0, :, 3] = [11., 22., 33.]
        result = {"mode": np.asarray("rollout"), "frames": np.array([1, 2, 3]),
                  "particle_frames": np.array([2, 3]), "particle_ids": np.array([0, 1, 2]),
                  "particle_predicted": predicted, "particle_reference": source[2:, :3].copy()}
        return RolloutReplay(Run(), cfg, result, seed=1), result

    def test_recovery_target_subtracts_predicted_current_and_uses_predicted_neighbours(self):
        replay, result = self.fixture()
        sample = replay.build(1, [0])
        self.assertAlmostEqual(float(sample["y"][0, 0]), 992.)  # 999 - 7, not 999 - 2
        self.assertAlmostEqual(float(sample["y"][0, 0] + replay.history[2, 0, 0]), 999.)
        self.assertAlmostEqual(float(sample["x"][0, -1, 0]), 11.)
        self.assertAlmostEqual(float(sample["neighbors"][0, 0, -1, 0]), 22.)
        self.assertAlmostEqual(float(sample["e"][0, 0, -1, 0]), .02, places=5)
        np.testing.assert_array_equal(replay.history[:, 3:, 6:], 0.)
        frozen = replay.history.copy()
        result["particle_predicted"] += 900
        np.testing.assert_array_equal(replay.history, frozen)
        # Every random replay input contains a model-generated current state.
        for _ in range(8):
            self.assertIn(float(replay.sample()["x"][0, -1, 0]), [11., 22., 33.])

    def test_future_truth_changes_only_labels(self):
        first, _ = self.fixture()
        changed, _ = self.fixture(reference_offset=5000.)
        np.testing.assert_array_equal(first.history, changed.history)
        a, b = first.build(1, [0]), changed.build(1, [0])
        for key in ("x", "neighbors", "e", "valid"):
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_allclose(b["y"] - a["y"], 5000.)

    def test_windows_include_late_frames_and_obey_stride(self):
        class Run:
            n_frames = 12
        cfg = Config(history_frames=2, frame_stride=2)
        self.assertEqual(rollout_windows([Run()], cfg, 2), [(0, 4), (0, 5), (0, 6), (0, 7)])
        with self.assertRaisesRegex(ValueError, "enough frames"):
            rollout_windows([Run()], cfg, 4)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            FinetuneConfig(rollout_steps=1)

    def test_validation_active_mask_uses_truth_and_has_explicit_fallback(self):
        truth = np.zeros((1, 2, 16), np.float32)
        truth[0, 0, 3] = .002
        predicted = truth.copy()
        predicted[0, 0, 3] = .001
        predicted[0, 1, 3] = 100.
        report = trajectory_metrics({"particle_predicted": predicted, "particle_reference": truth}, .001)
        self.assertEqual(report["selection_metric"], "active_velocity_rmse")
        self.assertEqual(report["active_particle_frame_pairs"], 1)
        self.assertAlmostEqual(report["score_m_s"], .001)
        truth[:] = 0
        other = trajectory_metrics({"particle_predicted": predicted, "particle_reference": truth}, .001)
        self.assertEqual(other["selection_metric"], "all_velocity_rmse_no_active")
        self.assertIsNone(other["active_speed_ratio_median"])
        self.assertGreater(other["score_m_s"], 70.)


class FinetuneIntegrationTests(unittest.TestCase):
    def options(self, root, out, cycles):
        return FinetuneConfig(data_dir=str(root / "data"), out_dir=str(root / out), cycles=cycles,
                              rollout_steps=2, val_steps=2, updates_per_cycle=4, batch=2,
                              inference_batch=2, val_tag="phi30_c500", device="cpu", log_every=4)

    def checkpoint(self):
        return source_checkpoint(Config(history_frames=1, neighbors=3, batch=2, hidden=8,
                                        holdout="phi30_c500", val_samples=1))

    def test_cycle_resume_matches_uninterrupted_training_and_keeps_stats_and_holdout(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            write_runs(root / "data")
            original = self.checkpoint()
            full = run_finetuning(original, self.options(root, "full", 2))
            first = run_finetuning(original, self.options(root, "resumed", 1))
            main(["--resume", str(first), "--cycles", "2"])
            a, b = load_checkpoint(full), load_checkpoint(first)
            self.assertEqual(a["step"], 28)
            self.assertEqual(b["finetune"]["completed_cycles"], 2)
            self.assertEqual(a["stats"], original["stats"])
            self.assertEqual(b["stats"], original["stats"])
            self.assertEqual(a["train_tags"], ["phi35_c1000"])
            self.assertEqual(a["val_tags"], ["phi30_c500"])
            changed = False
            for key, value in a["model"].items():
                torch.testing.assert_close(value, b["model"][key], rtol=0, atol=0)
                changed |= not torch.equal(value, original["model"][key])
            self.assertTrue(changed)
            for key, state in a["optimizer"]["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, b["optimizer"]["state"][key][name], rtol=0, atol=0)
            self.assertEqual(a["finetune"]["selection_rng"], b["finetune"]["selection_rng"])
            self.assertEqual(a["finetune"]["clean_rng"], b["finetune"]["clean_rng"])
            for experiment in ("full", "resumed"):
                records = [json.loads(line) for line in (root / experiment / "metrics.jsonl").read_text().splitlines()]
                collections = [record for record in records if record["split"] == "TRAIN_COLLECTION"]
                self.assertTrue(all(record["tag"] == "phi35_c1000" for record in collections))
                with np.load(root / experiment / "validation_latest.npz", allow_pickle=False) as z:
                    self.assertEqual(str(z["tag"]), "phi30_c500")
                    self.assertEqual(z["particle_predicted"].shape, (2, 4, 16))

    def test_initial_weights_remain_best_if_first_cycle_worsens_selection_metric(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            write_runs(root / "data")
            original = self.checkpoint()
            scores = iter([.001, .1, .002])  # baseline, training collection, candidate validation

            def controlled(result, threshold):
                metrics = trajectory_metrics(result, threshold)
                metrics["score_m_s"] = next(scores)
                return metrics

            with patch("surrogate.particle.finetune.trajectory_metrics", side_effect=controlled):
                run_finetuning(original, self.options(root, "worse", 1))
            best = load_checkpoint(root / "worse/best.pt")
            latest = load_checkpoint(root / "worse/model.pt")
            self.assertEqual(best["finetune"]["completed_cycles"], 0)
            self.assertEqual(latest["finetune"]["completed_cycles"], 1)
            for key, value in original["model"].items():
                torch.testing.assert_close(best["model"][key], value, rtol=0, atol=0)

    def test_rejects_training_tag_for_validation_and_protects_source_directory(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            write_runs(root / "data")
            original = self.checkpoint()
            options = self.options(root, "out", 1)
            with self.assertRaisesRegex(ValueError, "val_tag"):
                run_finetuning(original, replace(options, val_tag="phi35_c1000"))
            with self.assertRaisesRegex(ValueError, "source checkpoint directory"):
                run_finetuning(original, options, source_path=root / "out/original.pt")

    def test_interrupted_initial_validation_can_resume_before_any_updates(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            write_runs(root / "data")
            options = self.options(root, "interrupted", 1)
            with patch("surrogate.particle.finetune.rollout", side_effect=InterruptedError("stop")):
                with self.assertRaises(InterruptedError):
                    run_finetuning(self.checkpoint(), options)
            path = root / "interrupted/model.pt"
            initial = load_checkpoint(path)
            self.assertIsNone(initial["finetune"]["baseline"])
            self.assertEqual(initial["finetune"]["updates"], 0)
            main(["--resume", str(path), "--cycles", "1"])
            completed = load_checkpoint(path)
            self.assertEqual(completed["finetune"]["completed_cycles"], 1)
            self.assertIsNotNone(completed["finetune"]["baseline"])


if __name__ == "__main__":
    unittest.main()
