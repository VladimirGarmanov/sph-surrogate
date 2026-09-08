"""Whole-array audit: exact tails, corrupt inputs, target windows and holdout isolation."""
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from surrogate.particle.audit import audit
from surrogate.particle.config import Config


def states(frames=5, particles=3):
    data = np.zeros((frames, particles, 16), np.float32)
    data[..., 6] = 1600
    return data


def write_run(root, tag, soil, plate=None):
    np.save(root / f"{tag}.npy", soil)
    np.save(root / f"{tag}_plate.npy", states(len(soil), 1) if plate is None else plate)
    np.save(root / f"{tag}_boundary.npy", states(1, 1)[0])


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.out = self.root / "report"
        self.cfg = Config(holdout="", history_frames=1)

    def run_audit(self, cfg=None, scales=None, **kwargs):
        with redirect_stdout(io.StringIO()):
            return audit(self.data, self.out, cfg or self.cfg, scales, **kwargs)

    def test_all_values_and_last_rare_target_are_counted_without_sampling(self):
        soil = states()
        soil[-1, -1, 0] = 2.
        write_run(self.data, "phi35_c1000", soil)
        before = {path.name: path.read_bytes() for path in self.data.iterdir()}
        scales = np.ones(16)
        scales[0] = .01
        report = self.run_audit(scales=scales, top=2)
        targets = report["groups"]["train"]
        delta = soil[2:] - soil[1:-1]
        self.assertEqual(report["totals"]["raw_values_scanned"], (5*3 + 5 + 1)*16)
        self.assertEqual(targets["particle_frame_pairs"], 9)
        self.assertAlmostEqual(targets["features"]["x"]["std"], delta[..., 0].astype(np.float64).std())
        self.assertAlmostEqual(targets["zero_delta_mse"], np.mean((delta / scales) ** 2))
        threshold = targets["features"]["x"]["thresholds"]["100"]
        self.assertEqual(threshold["count"], 1)
        self.assertAlmostEqual(threshold["fraction_of_finite_values"], 1/9)
        self.assertEqual(threshold["fraction_of_zero_delta_squared_error"], 1.)
        self.assertEqual(targets["pairs_above_saved_scale"], {"10": 1, "100": 1, "1000": 0})
        raw = report["runs"][0]["files"]["soil"]["features"]["x"]
        self.assertEqual((raw["max_frame"], raw["max_particle_id"]), (4, 2))
        with (self.out / "worst_targets.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        extreme = max((row for row in rows if row["quantity"] == "x"), key=lambda row: float(row["delta"]))
        self.assertEqual((extreme["from_frame"], extreme["to_frame"], extreme["particle_id"]), ("3", "4", "2"))
        self.assertEqual(float(extreme["before"]), 0.)
        self.assertEqual(float(extreme["after"]), 2.)
        for path in self.data.iterdir():
            self.assertEqual(path.read_bytes(), before[path.name])
        self.assertEqual(json.loads((self.out / "summary.json").read_text())["totals"], report["totals"])
        readable = (self.out / "report.md").read_text()
        self.assertIn("phi35_c1000", readable)
        self.assertIn("0.01", (self.out / "normalization.csv").read_text())
        self.assertIn("не ошибка обученной сети", readable)

    def test_nonfinite_values_and_nonpositive_density_are_reported_with_locations(self):
        soil = states(4, 2)
        soil[1, 0, 4] = np.nan
        soil[3, 1, 7] = np.inf
        soil[0, 1, 6] = -10
        write_run(self.data, "phi35_c1000", soil)
        with np.errstate(invalid="ignore"):
            report = self.run_audit(scales=np.ones(16))
        self.assertEqual(report["totals"]["nonfinite_values"], 2)
        self.assertEqual(report["totals"]["nonpositive_soil_density"], 1)
        self.assertEqual(report["groups"]["train"]["invalid_pairs"], 2)
        features = report["runs"][0]["files"]["soil"]["features"]
        self.assertEqual(features["vy"]["nan_count"], 1)
        self.assertEqual(features["p11"]["inf_count"], 1)
        self.assertEqual(features["p11"]["finite_count"], 7)
        with (self.out / "invalid_values.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3)
        self.assertEqual({row["reason"] for row in rows}, {"nonfinite", "nonpositive_soil_density"})
        self.assertTrue(any(row["frame"] == "1" and row["particle_id"] == "0" and row["quantity"] == "vy" for row in rows))
        json.dumps(report, allow_nan=False)

    def test_stride_warm_start_and_holdout_are_not_mixed_into_training_scales(self):
        soil = states(6, 2)
        soil[..., 0] = np.array([100, 200, 0, 1, 4, 9], np.float32)[:, None]
        write_run(self.data, "phi35_c1000", soil)
        holdout = soil.copy()
        holdout[..., 0] *= 100
        write_run(self.data, "phi30_c500", holdout)
        cfg = Config(history_frames=1, frame_stride=2, holdout="phi30_c500")
        report = self.run_audit(cfg, np.ones(16))
        train = report["groups"]["train"]
        self.assertEqual(train["particle_frame_pairs"], 4)
        self.assertEqual(train["features"]["x"]["mean"], 6.)
        self.assertEqual(train["features"]["x"]["std"], 2.)
        self.assertEqual(report["groups"]["holdout"]["features"]["x"]["std"], 200.)
        with (self.out / "frames.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual({(row["from_frame"], row["to_frame"]) for row in rows}, {("2", "4"), ("3", "5")})

    def test_mismatched_frames_and_incomplete_triplets_do_not_hide_values(self):
        soil = states(5, 2)
        soil[-1, :, 0] = 100
        write_run(self.data, "phi35_c1000", soil, plate=states(3, 1))
        np.save(self.data / "phi30_c500_plate.npy", states(2, 1))
        report = self.run_audit()
        self.assertEqual(report["totals"]["runs_found"], 2)
        self.assertEqual(report["totals"]["complete_readable_triplets"], 1)
        run = next(row for row in report["runs"] if row["tag"] == "phi35_c1000")
        self.assertEqual(run["files"]["soil"]["features"]["x"]["finite_count"], 10)
        self.assertEqual(run["all_adjacent_soil_deltas"]["x"]["max"], 100.)
        self.assertEqual(run["eligible_targets"]["features"]["x"]["max"], 0.)
        self.assertEqual(report["groups"]["train"]["particle_frame_pairs"], 2)
        self.assertTrue(any("frame count mismatch" in issue for issue in run["issues"]))
        missing = next(row for row in report["runs"] if row["tag"] == "phi30_c500")
        self.assertIn("missing soil file", missing["issues"])
        self.assertIn("plate", missing["files"])

    def test_bad_format_is_explicit_and_valid_sibling_files_are_still_scanned(self):
        write_run(self.data, "phi35_c1000", np.zeros((4, 2, 8), np.float32))
        np.save(self.data / "unrelated.npy", np.zeros(1))
        report = self.run_audit()
        self.assertEqual(report["totals"]["complete_readable_triplets"], 0)
        self.assertEqual(report["unrecognized_npy_files"], ["unrelated.npy"])
        self.assertIn("plate", report["runs"][0]["files"])
        self.assertNotIn("soil", report["runs"][0]["files"])
        self.assertTrue(any("expected nonempty" in issue for issue in report["runs"][0]["issues"]))
        self.assertEqual(report["groups"]["train"]["particle_frame_pairs"], 0)

    def test_missing_boundary_keeps_raw_scan_but_excludes_training_targets(self):
        np.save(self.data / "phi35_c1000.npy", states())
        np.save(self.data / "phi35_c1000_plate.npy", states(5, 1))
        report = self.run_audit(scales=np.ones(16))
        run = report["runs"][0]
        self.assertIn("missing boundary file", run["issues"])
        self.assertEqual(run["all_adjacent_soil_deltas"]["x"]["finite_count"], 12)
        self.assertEqual(run["eligible_targets"]["particle_frame_pairs"], 0)
        self.assertEqual(report["groups"]["train"]["particle_frame_pairs"], 0)

    def test_constant_columns_stay_zero_variance_and_outputs_cannot_overwrite(self):
        write_run(self.data, "phi35_c1000", states())
        report = self.run_audit(scales=np.ones(16))
        self.assertEqual(report["groups"]["train"]["features"]["rho"]["std"], 0.)
        self.assertEqual(report["groups"]["train"]["zero_delta_mse"], 0.)
        summary = (self.out / "summary.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_audit()
        self.assertEqual((self.out / "summary.json").read_bytes(), summary)

    def test_invalid_saved_scale_is_rejected_before_creating_output(self):
        write_run(self.data, "phi35_c1000", states())
        scales = np.ones(16)
        scales[3] = 0
        with self.assertRaisesRegex(ValueError, "finite, positive"):
            self.run_audit(scales=scales)
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
