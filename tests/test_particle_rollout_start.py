"""Warm starts keep original frame indices and use future soil only for scoring."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from surrogate.data import PLATE, SOIL, WALL
from surrogate.particle.config import Config
from surrogate.particle.data import Stats
from surrogate.particle.rollout import predict_next_frame, rollout


class SyntheticRun:
    n_frames, n_soil, phi_deg, cohesion = 16, 2, 35., 1000.

    def __init__(self, last_true_frame, future_velocity_offset=0.):
        self.last_true_frame = last_true_frame
        self.types = np.array([SOIL, SOIL, PLATE, WALL])
        self.source = np.zeros((self.n_frames, len(self.types), 16), np.float32)
        self.source[:, :, 0] = [0., .02, 0., -.02]
        self.source[:, :, 2] = [-.01, -.02, 0., -.03]
        self.source[:, :, 6] = 1600.
        self.source[:, :, 9] = -500.
        self.source[:, :2, 3] = np.arange(1, self.n_frames + 1)[:, None] + [0., 10.]
        self.source[:, 2, 2] -= np.arange(self.n_frames) * .0001
        self.source[self.last_true_frame + 1:, :2, 3] += future_velocity_offset
        self.plate = self.source[:, 2:3]
        self.boundary = self.source[0, 3:]
        self.frame_reads = []
        self.completed_predictions = set()

        class EvaluationSoil:
            def __getitem__(_, frame):
                if frame not in self.completed_predictions:
                    raise AssertionError("future soil read before its prediction completed")
                return self.source[frame, :2]

        self.soil = EvaluationSoil()

    def frame(self, frame):
        if frame > self.last_true_frame:
            raise AssertionError("future full frame used as a model input")
        self.frame_reads.append(frame)
        return self.source[frame].copy()


class DoubleVelocity(torch.nn.Module):
    def forward(self, x, neighbors, e, valid):
        delta = x.new_zeros((len(x), 16))
        delta[:, 3] = x[:, -1, 0]
        return delta


def unit_stats():
    arrays = {}
    for key, size in (("node", 18), ("edge", 7), ("target", 16)):
        arrays[f"{key}_mean"] = np.zeros(size, np.float32)
        arrays[f"{key}_std"] = np.ones(size, np.float32)
    return Stats(arrays)


class RolloutStartTests(unittest.TestCase):
    def run_rollout(self, cfg, start_frame=None, steps=2, future_velocity_offset=0.):
        initial_frame = cfg.history_frames * cfg.frame_stride if start_frame is None else start_frame
        run = SyntheticRun(initial_frame, future_velocity_offset)
        histories = []

        def observe(model, stats, history, *args, **kwargs):
            histories.append(history.copy())
            next_frame = predict_next_frame(model, stats, history, *args, **kwargs)
            run.completed_predictions.add(initial_frame + len(histories) * cfg.frame_stride)
            return next_frame

        with patch("surrogate.particle.rollout.predict_next_frame", side_effect=observe):
            result = rollout(DoubleVelocity(), cfg, unit_stats(), run, torch.device("cpu"),
                             n_steps=steps, verbose=False, save_particles=True, start_frame=start_frame)
        return run, histories, result

    def test_default_still_uses_earliest_full_history(self):
        cfg = Config(history_frames=2, frame_stride=2, neighbors=3)
        run, histories, result = self.run_rollout(cfg)
        self.assertEqual(run.frame_reads, [0, 2, 4])
        self.assertEqual(int(result["initial_frame"]), 4)
        np.testing.assert_array_equal(result["frames"], [4, 6, 8])
        np.testing.assert_allclose(result["t"], [.08, .12, .16])
        np.testing.assert_array_equal(histories[0][:, 0, 3], [1, 3, 5])
        np.testing.assert_array_equal(result["particle_predicted"][:, 0, 3], [10, 20])

    def test_start_can_have_different_phase_from_stride(self):
        cfg = Config(history_frames=2, frame_stride=2, neighbors=3)
        run, histories, result = self.run_rollout(cfg, start_frame=7)
        self.assertEqual(run.frame_reads, [3, 5, 7])
        self.assertEqual(int(result["initial_frame"]), 7)
        np.testing.assert_array_equal(result["frames"], [7, 9, 11])
        np.testing.assert_array_equal(result["particle_frames"], [9, 11])
        np.testing.assert_allclose(result["particle_t"], [.18, .22])
        np.testing.assert_array_equal(histories[0][:, 0, 3], [4, 6, 8])
        np.testing.assert_array_equal(histories[1][:, 0, 3], [6, 8, 16])
        np.testing.assert_array_equal(histories[1][-1, 2:, :6], run.source[9, 2:, :6])
        for history in histories:
            np.testing.assert_array_equal(history[:, 2:, 6:], 0.)
        np.testing.assert_array_equal(result["particle_predicted"][:, 0, 3], [16, 32])
        np.testing.assert_array_equal(result["particle_reference"], run.source[[9, 11], :2])

    def test_future_soil_changes_scores_but_never_inputs_or_predictions(self):
        cfg = Config(history_frames=2, neighbors=3)
        first_run, first_histories, first = self.run_rollout(cfg, start_frame=5, steps=3)
        other_run, other_histories, other = self.run_rollout(
            cfg, start_frame=5, steps=3, future_velocity_offset=10000.)
        self.assertEqual(first_run.frame_reads, [3, 4, 5])
        self.assertEqual(other_run.frame_reads, first_run.frame_reads)
        for first_history, other_history in zip(first_histories, other_histories):
            np.testing.assert_array_equal(first_history, other_history)
        np.testing.assert_array_equal(first["particle_predicted"], other["particle_predicted"])
        np.testing.assert_array_equal(first["particle_predicted"][:, 0, 3], [12, 24, 48])
        self.assertFalse(np.array_equal(first["feature_rmse"], other["feature_rmse"]))

    def test_zero_history_and_truncated_tail(self):
        cfg = Config(history_frames=0, frame_stride=2, neighbors=3)
        run, histories, result = self.run_rollout(cfg, start_frame=13, steps=5)
        self.assertEqual(run.frame_reads, [13])
        self.assertEqual(histories[0].shape[0], 1)
        np.testing.assert_array_equal(result["frames"], [13, 15])
        np.testing.assert_array_equal(result["particle_predicted"][:, 0, 3], [28])

    def test_invalid_start_is_rejected_before_reading_inputs(self):
        cfg = Config(history_frames=2, frame_stride=2)
        run = SyntheticRun(15)
        for start_frame, message in ((3, "at least 4"), (-1, "at least 4"),
                                     (7.5, "integer"), (True, "integer"),
                                     (14, "not enough frames"), (16, "not enough frames")):
            with self.subTest(start_frame=start_frame):
                with self.assertRaisesRegex(ValueError, message):
                    rollout(DoubleVelocity(), cfg, unit_stats(), run, torch.device("cpu"),
                            start_frame=start_frame, verbose=False)
                self.assertEqual(run.frame_reads, [])


if __name__ == "__main__":
    unittest.main()
