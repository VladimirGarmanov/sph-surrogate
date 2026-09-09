"""Сокращённая история проходит через обучение, нормализацию, rollout и replay."""
from dataclasses import replace
import unittest

import numpy as np
import torch

from surrogate.data import PLATE, SOIL, WALL
from surrogate.particle.config import Config
from surrogate.particle.data import ParticleDataset, Stats, to_tensors
from surrogate.particle.model import ParticleNet, predict
from surrogate.particle.replay import RolloutReplay
from surrogate.particle.rollout import rollout


class TinyRun:
    def __init__(self):
        self.tag, self.phi_deg, self.cohesion = "phi35_c1000", 35., 1000.
        self.n_frames, self.n_soil = 12, 5
        self.types = np.array([SOIL] * self.n_soil + [PLATE, WALL])
        source = np.random.default_rng(12).normal(size=(12, 7, 16)).astype(np.float32) * .01
        source[:, :, 0] = np.arange(7) * .02
        source[:, :, 1:3] = 0
        source[:, :5, 2] = -.02
        source[:, 5, 2] = .01 - np.arange(12) * .001
        source[:, :, 3] = np.arange(12)[:, None] * .01 + np.arange(7) * .001
        self.soil, self.plate, self.boundary = source[:, :5], source[:, 5:6], source[0, 6:]

    def frame(self, t):
        return np.concatenate([self.soil[t], self.plate[t], self.boundary])


def unit_stats():
    return Stats({f"{key}_{name}": np.full(size, value, np.float32)
                  for key, size in (("node", 18), ("edge", 7), ("target", 16))
                  for name, value in (("mean", 0), ("std", 1))})


class NeighborHistoryIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.run = TinyRun()
        # K exceeds the available particles so every path also exercises padding.
        self.cfg = Config(history_frames=8, neighbor_history_frames=3,
                          neighbors=9, batch=2, gpu_batch=4, hidden=8)
        self.dataset = ParticleDataset([self.run], self.cfg)

    def test_full_frame_batches_match_samples_and_cover_every_particle(self):
        for index, (_, frame) in enumerate(self.dataset.index):
            batches = list(self.dataset.frame_batches(index))
            self.assertEqual([len(ids) for ids, _ in batches], [4, 1])
            np.testing.assert_array_equal(np.concatenate([ids for ids, _ in batches]), np.arange(5))
            for ids, sample in batches:
                self.assertEqual(sample["x"].shape, (len(ids), 9, 18))
                self.assertEqual(sample["neighbors"].shape, (len(ids), 9, 3, 18))
                self.assertEqual(sample["e"].shape, (len(ids), 9, 3, 7))
                ordinary = self.dataset.build(self.run, frame, ids)
                for key in sample:
                    np.testing.assert_array_equal(sample[key], ordinary[key])
                np.testing.assert_array_equal(sample["y"],
                                              self.run.soil[frame + 1, ids] - self.run.soil[frame, ids])

    def test_statistics_match_manual_moments_and_ignore_padded_history(self):
        samples = [sample for index in range(len(self.dataset.index))
                   for _, sample in self.dataset.frame_batches(index)]
        self.assertTrue(all((~sample["valid"]).any() for sample in samples))
        # Padding must not affect moments, even if its stored values are nonzero.
        for sample in samples:
            sample["neighbors"][~sample["valid"]] = 1e6
            sample["e"][~sample["valid"]] = -1e6
        actual = Stats.compute(samples)
        node = [rows for sample in samples for rows in
                (sample["x"].reshape(-1, 18), sample["neighbors"][sample["valid"]].reshape(-1, 18))]
        edge = [sample["e"][sample["valid"]].reshape(-1, 7) for sample in samples]
        target = [sample["y"].reshape(-1, 16) for sample in samples]
        for key, chunks in (("node", node), ("edge", edge), ("target", target)):
            rows = np.concatenate(chunks).astype(np.float64)
            std = rows.std(axis=0)
            np.testing.assert_allclose(actual.arrays[f"{key}_mean"], rows.mean(axis=0), rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(actual.arrays[f"{key}_std"], np.where(std < 1e-8, 1., std),
                                       rtol=1e-6, atol=1e-7)
        for sample in samples:
            normalized = to_tensors(sample, actual)
            for key in ("neighbors", "e"):
                self.assertTrue(torch.all(normalized[key][~normalized["valid"]] == 0))

    def test_rollout_and_replay_use_the_same_recent_neighbor_frames(self):
        class HistoryProbe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, x, neighbors, e, valid):
                self.calls.append({key: value.detach().clone() for key, value in
                                   (("x", x), ("neighbors", neighbors), ("e", e), ("valid", valid))})
                delta = torch.zeros((len(x), 16))
                # Both ends of the neighbor window influence the next predicted history.
                delta[:, 0] = neighbors[:, 0, 0, 0] * .01
                delta[:, 3] = neighbors[:, 0, -1, 0] + x[:, -1, 0]
                return delta

        model, stats = HistoryProbe(), unit_stats()
        result = rollout(model, self.cfg, stats, self.run, torch.device("cpu"),
                         n_steps=2, verbose=False, save_particles=True)
        self.assertEqual(len(model.calls), 6)  # three batches for each synchronous step
        replay = RolloutReplay(self.run, self.cfg, result)
        for transition in range(2):
            sample = replay.build(transition, np.arange(self.run.n_soil))
            self.assertEqual(sample["x"].shape[1], 9)
            self.assertEqual(sample["neighbors"].shape[2], 3)
            expected = to_tensors(sample, stats)
            for key in ("x", "neighbors", "e", "valid"):
                seen = torch.cat([call[key] for call in model.calls[transition * 3:(transition + 1) * 3]])
                torch.testing.assert_close(seen, expected[key], rtol=0, atol=0)
        first = self.dataset.build(self.run, 8, np.arange(self.run.n_soil))
        for key in ("x", "neighbors", "e", "valid"):
            np.testing.assert_array_equal(replay.build(0, np.arange(5))[key], first[key])
        np.testing.assert_array_equal(replay.build(1, np.arange(5))["y"],
                                      self.run.soil[10] - result["particle_predicted"][0])
        self.assertFalse(np.array_equal(result["particle_predicted"][0], self.run.soil[9]))
        # These are the tail frames 7, 8, predicted 9; frames 1, 2, 3 would differ.
        full = RolloutReplay(self.run, replace(self.cfg, neighbor_history_frames=0), result).build(1, [0])
        short = replay.build(1, [0])
        for key in ("neighbors", "e"):
            np.testing.assert_array_equal(short[key], full[key][:, :, -3:])

    def test_asymmetric_history_backpropagates_through_both_encoders_and_shared_gru(self):
        torch.manual_seed(41)
        sample = self.dataset.build(self.run, 8, np.arange(self.run.n_soil))
        batch = to_tensors(sample, Stats.compute([sample]))
        model = ParticleNet(hidden=8)
        # The usual zero output initialization blocks upstream gradients on the first step.
        torch.nn.init.normal_(model.decoder[-1].weight, std=.02)
        loss = ((predict(model, batch) - batch["y"]) ** 2).mean()
        loss.backward()
        for module in (model.center_encoder, model.neighbor_encoder, model.history):
            gradients = [parameter.grad for parameter in module.parameters()]
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all()
                                for gradient in gradients))
            self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0.)


if __name__ == "__main__":
    unittest.main()
