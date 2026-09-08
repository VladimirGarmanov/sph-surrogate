import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from surrogate.data import SOIL, WALL
from surrogate.particle.config import Config
from surrogate.particle.data import ParticleDataset, Stats, to_tensors
from surrogate.particle.model import ParticleNet, predict
from surrogate.particle.train import load_checkpoint, main, train_frame


class TinyRun:
    def __init__(self, tag="phi35_c1000", frames=12, particles=5):
        self.tag, self.phi_deg, self.cohesion = tag, 35., 1000.
        self.n_frames, self.n_soil = frames, particles
        self.types = np.array([SOIL] * particles + [WALL])
        self.soil = np.zeros((frames, particles, 16), np.float32)
        for t in range(frames):
            self.soil[t, :, 0] = np.arange(particles) * .02
            self.soil[t, :, 3] = np.arange(particles) * .01 + t * .001
            self.soil[t, :, 7] = t * np.arange(1, particles + 1)
        self.boundary = np.zeros((1, 16), np.float32)
        self.boundary[:, 0] = -.03

    def frame(self, t):
        return np.concatenate([self.soil[t], self.boundary])


def unit_stats():
    arrays = {}
    for key, dim in (("node", 18), ("edge", 7), ("target", 16)):
        arrays[key + "_mean"] = np.zeros(dim, np.float32)
        arrays[key + "_std"] = np.ones(dim, np.float32)
    return Stats(arrays)


class FullFrameTests(unittest.TestCase):
    def test_every_id_every_frame_every_run_and_correct_horizon(self):
        runs = [TinyRun(), TinyRun("phi40_c1000", frames=13, particles=3)]
        cfg = Config(history_frames=8, prediction_horizon=2, batch=2, neighbors=3)
        dataset = ParticleDataset(runs, cfg)
        self.assertEqual(dataset.index, [(0, 8), (0, 9), (1, 8), (1, 9), (1, 10)])
        for index, (ri, t) in enumerate(dataset.index):
            visited = []
            for ids, sample in dataset.frame_batches(index):
                visited.extend(ids.tolist())
                ordinary = dataset.build(runs[ri], t, ids)
                for key in sample:
                    np.testing.assert_array_equal(sample[key], ordinary[key])
                self.assertEqual(sample["y"].shape, (len(ids), 2, 16))
                for row, particle in enumerate(ids):
                    for k in (1, 2):
                        np.testing.assert_array_equal(sample["y"][row, k - 1],
                                                      runs[ri].soil[t + k, particle] - runs[ri].soil[t, particle])
            self.assertEqual(visited, list(range(runs[ri].n_soil)))

    def test_stride_and_single_horizon_preserve_full_coverage(self):
        run = TinyRun(frames=8)
        cfg = Config(history_frames=2, frame_stride=2, prediction_horizon=1, batch=3)
        dataset = ParticleDataset([run], cfg)
        self.assertEqual(dataset.index, [(0, 4), (0, 5)])
        for index, (_, t) in enumerate(dataset.index):
            for ids, sample in dataset.frame_batches(index):
                self.assertEqual(sample["y"].shape, (len(ids), 16))
                np.testing.assert_array_equal(sample["y"], run.soil[t + 2, ids] - run.soil[t, ids])

    def test_accumulation_matches_entire_frame_and_updates_once(self):
        torch.manual_seed(31)
        run = TinyRun()
        cfg = Config(history_frames=2, prediction_horizon=5, batch=2, neighbors=3)
        dataset = ParticleDataset([run], cfg)
        model = ParticleNet(hidden=8, prediction_horizon=5)
        torch.nn.init.normal_(model.decoder[-1].weight, std=.02)
        whole = copy.deepcopy(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=.03)
        whole_optimizer = torch.optim.SGD(whole.parameters(), lr=.03)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        visits = []

        def progress(visited, total):
            visits.append((visited, total))
            # Во время всех порций параметры ещё не изменены.
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)

        with patch.object(optimizer, "step", wraps=optimizer.step) as update:
            actual = train_frame(model, dataset, 0, unit_stats(), "cpu", optimizer, progress)
            self.assertEqual(update.call_count, 1)
        self.assertEqual(visits, [(2, 5), (4, 5), (5, 5)])
        batch = to_tensors(dataset.build(run, 2, np.arange(5)), unit_stats())
        expected = ((predict(whole, batch) - batch["y"]) ** 2).mean()
        expected.backward()
        torch.nn.utils.clip_grad_norm_(whole.parameters(), 1.)
        whole_optimizer.step()
        self.assertAlmostEqual(actual, expected.item(), places=5)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, whole.state_dict()[key], rtol=1e-5, atol=1e-7)

    def invoke(self, out, extra=()):
        argv = ["train", "--out_dir", str(out), "--history_frames", "1",
                "--prediction_horizon", "2", "--hidden", "8", "--batch", "2", "--neighbors", "2",
                "--stats_frames", "1", "--val_samples", "1", "--val_every", "2",
                "--input_noise_std", ".05", "--holdout", "phi30_c500", "--device", "cpu", *extra]
        with patch("sys.argv", argv), patch("surrogate.particle.train.discover_runs",
                                            return_value=[TinyRun(frames=5),
                                                          TinyRun("phi30_c500", frames=5)]):
            with contextlib.redirect_stdout(io.StringIO()):
                main()

    def test_mid_epoch_resume_matches_uninterrupted_and_saves_frame_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            full, resumed = Path(tmp) / "full", Path(tmp) / "resumed"
            self.invoke(full)
            calls = 0

            def interrupt(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated interruption")
                return train_frame(*args, **kwargs)

            with patch("surrogate.particle.train.train_frame", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    self.invoke(resumed)
            partial = load_checkpoint(resumed / "model.pt")
            self.assertEqual(partial["step"], 1)
            self.assertEqual(partial["next_frame_index"], 1)
            self.invoke(resumed, ["--resume", str(resumed / "model.pt")])
            a, b = load_checkpoint(full / "model.pt"), load_checkpoint(resumed / "model.pt")
            self.assertEqual(a["step"], 2)
            self.assertEqual(b["completed_epochs"], 1)
            self.assertEqual(b["next_frame_index"], 0)
            for key, value in a["model"].items():
                torch.testing.assert_close(value, b["model"][key], rtol=0, atol=0)
            records = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
            rows = [r for r in records if r["split"] == "TRAIN"]
            self.assertEqual([r["frame"] for r in rows], [1, 2])
            self.assertEqual([r["targets"] for r in rows], [5, 5])

    def test_old_checkpoint_keeps_random_mode_and_cannot_silently_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.invoke(out, ["--training_mode", "random_particles", "--steps", "2"])
            ckpt = load_checkpoint(out / "model.pt")
            ckpt["config"].pop("training_mode")
            ckpt["config"].pop("epochs")
            torch.save(ckpt, out / "model.pt")
            self.invoke(out, ["--resume", str(out / "model.pt"), "--steps", "3"])
            self.assertEqual(load_checkpoint(out / "model.pt")["config"]["training_mode"], "random_particles")
            with self.assertRaisesRegex(ValueError, "training_mode"):
                self.invoke(out, ["--resume", str(out / "model.pt"), "--training_mode", "full_frames"])

    def test_full_frames_rejects_step_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.invoke(Path(tmp), ["--steps", "1"])


if __name__ == "__main__":
    unittest.main()
