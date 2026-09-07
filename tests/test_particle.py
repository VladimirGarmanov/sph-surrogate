"""Behavioral checks for particle sampling, information flow and synchronous rollout."""
import csv
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from surrogate.data import PLATE, SOIL, WALL
from surrogate.particle.config import Config
from surrogate.particle.compare import load_comparison, summarize, write_csv
from surrogate.particle.data import FEATURE_NAMES, FEATURE_UNITS, ParticleDataset, Stats, build_inputs, to_tensors
from surrogate.particle.model import ParticleNet, predict
from surrogate.particle.neighbors import nearest_neighbors
from surrogate.particle.rollout import predict_next_frame, rollout
from surrogate.particle.train import split_runs
from surrogate.particle.timing import STAGES, StepTimer


def fixture():
    frame = np.zeros((6, 16), np.float32)
    frame[:, 0] = [0, .02, .04, .08, .11, .20]
    frame[:, 3:6] = np.arange(18, dtype=np.float32).reshape(6, 3) / 100
    frame[:, 6] = 1600
    frame[:, 7:13] = np.arange(36, dtype=np.float32).reshape(6, 6)
    return frame, np.array([SOIL, SOIL, SOIL, SOIL, PLATE, WALL])


def unit_stats():
    arrays = {}
    for name, dim in (("node", 18), ("edge", 7), ("target", 16)):
        arrays[f"{name}_mean"] = np.zeros(dim, np.float32)
        arrays[f"{name}_std"] = np.ones(dim, np.float32)
    return Stats(arrays)


class ParticleTests(unittest.TestCase):
    def test_nearest_excludes_identity_including_coincident_particles(self):
        pos = np.array([[0., 0, 0], [0., 0, 0], [.1, 0, 0], [4., 0, 0]])
        ids, valid = nearest_neighbors(pos, [0, 1], 6)
        for row, target in enumerate([0, 1]):
            selected = ids[row, valid[row]]
            self.assertEqual(set(selected), set(range(4)) - {target})
            self.assertEqual(len(selected), 3)
            self.assertEqual(selected[-1], 3)  # fill from outside any small radius
        trimmed, mask = nearest_neighbors(pos, [0], 2)
        self.assertEqual(set(trimmed[0, mask[0]]), {1, 2})

    def test_future_only_changes_target_and_particle_ids_are_preserved(self):
        current, types = fixture()

        class Run:
            tag, phi_deg, cohesion = "phi35_c1000", 35., 1000.
            n_frames, n_soil = 2, 4

            def frame(self, t):
                if t != 0:
                    raise AssertionError("future frame used to construct neighbours")
                return current.copy()

        run = Run()
        run.types = types
        run.soil = np.stack([current[:4], current[:4] + np.arange(64).reshape(4, 16)])
        ds = ParticleDataset([run], Config(neighbors=3, batch=2, history_frames=0))
        first = ds.build(run, 0, np.array([2, 0]))
        np.testing.assert_allclose(first["y"], np.arange(64).reshape(4, 16)[[2, 0]])
        run.soil[1] += 9000
        second = ds.build(run, 0, np.array([2, 0]))
        for key in ("x", "neighbors", "e", "valid"):
            np.testing.assert_array_equal(first[key], second[key])

    def test_boundary_response_is_not_an_input_but_boundary_velocity_is(self):
        frame, types = fixture()
        first = build_inputs(frame[None], types, 35, 1000, [0, 1], 5)
        changed = frame.copy()
        changed[types != SOIL, 6:] += 1e6
        second = build_inputs(changed[None], types, 35, 1000, [0, 1], 5)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        changed[types == PLATE, 5] += 1
        third = build_inputs(changed[None], types, 35, 1000, [0, 1], 5)
        self.assertFalse(np.array_equal(first["neighbors"], third["neighbors"]))
        self.assertFalse(np.array_equal(first["e"], third["e"]))

    def test_neighbor_order_and_padding_do_not_change_prediction(self):
        torch.manual_seed(0)
        model = ParticleNet(16)
        # Randomize the zero-initialized head so this checks real dependence.
        torch.nn.init.normal_(model.decoder[-1].weight, std=.1)
        x = torch.randn(2, 9, 18)
        neighbors, edges = torch.randn(2, 7, 9, 18), torch.randn(2, 7, 9, 7)
        valid = torch.tensor([[True] * 5 + [False] * 2, [False] * 7])
        expected = model(x, neighbors, edges, valid)
        order = torch.tensor([6, 1, 4, 0, 5, 2, 3])
        torch.testing.assert_close(model(x, neighbors[:, order], edges[:, order], valid[:, order]), expected)
        neighbors[~valid], edges[~valid] = 1e6, -1e6
        actual = model(x, neighbors, edges, valid)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_rollout_reads_same_snapshot_for_every_target(self):
        frame, types = fixture()

        class NeighborVelocity(torch.nn.Module):
            def forward(self, x, neighbors, e, valid):
                delta = torch.zeros((len(x), 16))
                delta[:, 3] = neighbors[:, 0, -1, 0]  # neighbour's CURRENT velocity
                return delta

        cfg = Config(neighbors=2, batch=2, history_frames=0)
        known = frame[types != SOIL, :6].copy()
        known[:, 2] -= .001
        model = NeighborVelocity()
        seen = []

        def observe(current, *args, **kwargs):
            seen.append(current.copy())
            return build_inputs(current, *args, **kwargs)

        with patch("surrogate.particle.rollout.build_inputs", side_effect=observe):
            actual = predict_next_frame(model, unit_stats(), frame[None], types, 35, 1000,
                                        cfg, torch.device("cpu"), known, batch_size=1)
        for snapshot in seen:
            np.testing.assert_array_equal(snapshot, frame[None])
        all_at_once = predict_next_frame(model, unit_stats(), frame[None], types, 35, 1000,
                                         cfg, torch.device("cpu"), known, batch_size=4)
        np.testing.assert_array_equal(actual, all_at_once)
        self.assertFalse(np.array_equal(actual[types == SOIL, 3], frame[types == SOIL, 3]))
        np.testing.assert_array_equal(actual[types != SOIL, :6], known)
        np.testing.assert_array_equal(actual[types != SOIL, 6:], 0)

    def test_training_can_fit_a_small_particle_batch(self):
        torch.manual_seed(4)
        frame, types = fixture()
        sample = build_inputs(frame[None], types, 35, 1000, [0, 1, 2, 3], 5)
        sample["y"] = np.broadcast_to(np.linspace(-.1, .1, 16, dtype=np.float32), (4, 16)).copy()
        stats = Stats.compute([sample])
        batch = to_tensors(sample, stats)
        # Nonconstant normalized targets exercise the optimizer, not the mean baseline.
        batch["y"] = torch.randn(4, 16) * .1
        model = ParticleNet(16)
        opt = torch.optim.Adam(model.parameters(), lr=.01)
        initial = ((predict(model, batch) - batch["y"]) ** 2).mean().item()
        for _ in range(60):
            loss = ((predict(model, batch) - batch["y"]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        final = ((predict(model, batch) - batch["y"]) ** 2).mean().item()
        self.assertLess(final, initial * .5)

    def test_missing_holdout_cannot_silently_validate_on_train(self):
        class Run:
            tag = "phi35_c1000"
        with self.assertRaisesRegex(ValueError, "missing holdout"):
            split_runs([Run()], Config())
        train, val = split_runs([Run()], Config(holdout=""))
        self.assertEqual(len(train), 1)
        self.assertEqual(val, [])

    def test_history_tracks_current_neighbor_identity_not_past_nearest_rank(self):
        frame, types = fixture()
        history = np.repeat(frame[None], 9, axis=0)
        history[0, 1, 0] = 10.0       # current neighbour was far away
        history[0, 2, 0] = .001      # a different particle used to be closer
        history[:, 1, 3] = np.arange(9) + 100
        sample = build_inputs(history, types, 35, 1000, [0], 1)
        self.assertEqual(sample["neighbors"].shape, (1, 1, 9, 18))
        np.testing.assert_array_equal(sample["neighbors"][0, 0, :, 0], np.arange(9) + 100)
        self.assertAlmostEqual(sample["e"][0, 0, 0, 0], 10.)
        self.assertEqual(sample["x"].shape, (1, 9, 18))

    def test_earlier_frames_affect_prediction_with_identical_current_states(self):
        torch.manual_seed(2)
        model = ParticleNet(16)
        torch.nn.init.normal_(model.decoder[-1].weight, std=.2)
        x, near, edges = torch.randn(2, 9, 18), torch.randn(2, 3, 9, 18), torch.randn(2, 3, 9, 7)
        valid = torch.ones(2, 3, dtype=torch.bool)
        first = model(x, near, edges, valid)
        changed = near.clone()
        changed[:, :, :-1, 0] += 10
        second = model(x, changed, edges, valid)
        self.assertGreater((first - second).abs().max().item(), 1e-6)
        changed_x = x.clone()
        changed_x[:, :-1, 1] += 10
        third = model(changed_x, near, edges, valid)
        self.assertGreater((first - third).abs().max().item(), 1e-6)

    def test_rollout_shifts_predictions_into_history_without_future_truth(self):
        frame, types = fixture()
        source = np.repeat(frame[None], 5, axis=0)
        source[:, :4, 0] += np.array([0., 1., 2., 999., 9999.])[:, None]

        class Run:
            n_frames, phi_deg, cohesion = 5, 35., 1000.

            def frame(self, t):
                if t > 2:
                    raise AssertionError("future truth requested as model input")
                return source[t].copy()

        run = Run()
        run.types, run.soil, run.plate, run.boundary = types, source[:, :4], source[:, 4:5], source[0, 5:]
        histories = []

        def step(model, stats, history, *args):
            histories.append(history.copy())
            result = history[-1].copy()
            result[:4, 0] += 5
            return result

        with patch("surrogate.particle.rollout.predict_next_frame", side_effect=step):
            report = rollout(torch.nn.Identity(), Config(history_frames=2), unit_stats(), run,
                             torch.device("cpu"), n_steps=2, verbose=False, save_particles=True)
        np.testing.assert_array_equal(histories[0][:, 0, 0], [0, 1, 2])
        np.testing.assert_array_equal(histories[1][:, 0, 0], [1, 2, 7])
        np.testing.assert_array_equal(report["frames"], [2, 3, 4])
        np.testing.assert_array_equal(report["particle_frames"], [3, 4])
        np.testing.assert_array_equal(report["particle_ids"], [0, 1, 2, 3])
        self.assertEqual(report["particle_predicted"].shape, (2, 4, 16))
        np.testing.assert_array_equal(report["particle_reference"], source[3:, :4])
        np.testing.assert_array_equal(report["particle_predicted"][:, 0, 0], [7, 12])
        error = report["particle_predicted"].astype(np.float64) - report["particle_reference"]
        np.testing.assert_allclose(np.sqrt((error ** 2).mean(1)), report["feature_rmse"][1:])

    def test_profile_and_batch_size_preserve_actual_network_predictions(self):
        torch.manual_seed(12)
        model = ParticleNet(16).eval()
        torch.nn.init.normal_(model.decoder[-1].weight, std=.1)
        frame, types = fixture()
        history = np.repeat(frame[None], 3, axis=0)
        history[0, :, 3] -= .03
        cfg = Config(history_frames=2, neighbors=3, batch=2)
        args = (model, unit_stats(), history, types, 35., 1000., cfg,
                torch.device("cpu"), frame[types != SOIL, :6])
        ordinary = predict_next_frame(*args)
        timer = StepTimer("cpu")
        start = time.perf_counter()
        profiled = predict_next_frame(*args, timer=timer)
        seconds = timer.finish(time.perf_counter() - start)
        np.testing.assert_array_equal(ordinary, profiled)
        np.testing.assert_allclose(predict_next_frame(*args, batch_size=4), ordinary, rtol=1e-5, atol=1e-6)
        self.assertEqual(len(seconds), len(STAGES))
        self.assertTrue(np.isfinite(seconds).all())
        self.assertTrue(np.all(np.asarray(seconds) >= 0))
        self.assertGreater(timer.seconds["neighbors"], 0)
        self.assertGreater(timer.seconds["network"], 0)

    def test_comparison_roundtrip_selects_original_ids_and_exports_physical_errors(self):
        truth = np.zeros((2, 2, 16), np.float32)
        truth[..., 6] = 1600
        truth[1, 1, 9] = -500
        prediction = truth.copy()
        prediction[1, 1, :3] = [.003, .004, 0]
        prediction[1, 1, 9] = -400
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "comparison.npz"
            np.savez_compressed(path, particle_predicted=prediction, particle_reference=truth,
                                particle_ids=[10, 42], particle_frames=[24, 26], particle_t=[.48, .52],
                                feature_names=FEATURE_NAMES, feature_units=FEATURE_UNITS,
                                mode="rollout", tag="test")
            all_data = load_comparison(path)
            report = summarize(all_data)
            self.assertAlmostEqual(report["features"]["p33"]["rmse"], 50.)
            self.assertAlmostEqual(report["position"]["rmse"], .0025)
            self.assertEqual(report["worst_position_errors"][0]["particle_id"], 42)
            self.assertEqual(report["worst_position_errors"][0]["frame"], 26)
            selected = load_comparison(path, frame=26, particle_ids=[42])
            self.assertEqual(selected["particle_predicted"].shape, (1, 1, 16))
            table = Path(folder) / "particle.csv"
            write_csv(table, selected)
            with table.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 16)
            stress = rows[9]
            self.assertEqual((stress["particle_id"], stress["frame"], stress["quantity"], stress["unit"]),
                             ("42", "26", "p33", "Pa"))
            self.assertEqual(float(stress["solver"]), -500.)
            self.assertEqual(float(stress["prediction"]), -400.)
            self.assertEqual(float(stress["error"]), 100.)
            with self.assertRaisesRegex(ValueError, "unknown soil particle"):
                load_comparison(path, particle_ids=[0])
            with self.assertRaisesRegex(ValueError, "not a stored prediction"):
                load_comparison(path, frame=25)
            summary_only = Path(folder) / "old.npz"
            np.savez(summary_only, rmse=[0, .1])
            with self.assertRaisesRegex(ValueError, "save_particles"):
                load_comparison(summary_only)


if __name__ == "__main__":
    unittest.main()
