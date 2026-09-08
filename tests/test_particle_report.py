import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from surrogate.particle.data import FEATURE_NAMES, FEATURE_UNITS
from surrogate.particle.report import build_report, mode_report


class ReportTests(unittest.TestCase):
    def write_result(self, directory, mode, active=True, step=46000):
        reference = np.zeros((2, 2, 16), np.float32)
        if active:
            reference[:, 0, 3] = .002
        predicted = reference.copy()
        predicted[:, 0, 0] += .003
        predicted[:, 0, 1] += .004
        predicted[:, 0, 3] += .006
        path = directory / f"{mode}.npz"
        np.savez(path, particle_predicted=predicted, particle_reference=reference,
                 particle_frames=[9, 10], particle_t=[.18, .20], particle_ids=[4, 7],
                 feature_names=FEATURE_NAMES, feature_units=FEATURE_UNITS,
                 mode=mode, tag="phi30_c500", checkpoint_step=step,
                 step_wall=[100., 2.], p_pred=[999., 12., np.nan],
                 p_gt=[1., 10., 10.])
        return path

    def test_vector_errors_active_selection_and_pressure_exclude_warm_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_result(Path(tmp), "rollout")
            report, _ = mode_report(path)
            self.assertAlmostEqual(report["active_position_rmse_m"], .005)
            self.assertAlmostEqual(report["position"]["rmse"], .005 / np.sqrt(2))
            self.assertAlmostEqual(report["active_velocity_rmse_m_s"], .006)
            self.assertEqual(report["active_fraction"], .5)
            self.assertEqual(report["seconds_per_step"], 2.)
            self.assertAlmostEqual(report["pressure"]["p_gt"]["relative_error"], .2)
            self.assertEqual(report["pressure"]["p_gt"]["valid_frames"], 1)

    def test_report_handles_no_active_particles_and_records_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for mode in ("rollout", "teacher_forced"):
                self.write_result(directory, mode, active=False)
            (directory / "training").mkdir()
            records = [{"step": 46000, "split": "VAL", "mse": .12798, "zero_delta_mse": .59665},
                       {"step": 48800, "split": "TRAIN", "mse": 481.59}]
            (directory / "training/metrics.jsonl").write_text("\n".join(map(json.dumps, records)))
            build_report(directory)
            result = json.loads((directory / "REPORT.json").read_text())
            self.assertIsNone(result[0]["active_position_rmse_m"])
            document = (directory / "REPORT_RU.md").read_text()
            self.assertIn("нет данных", document)
            self.assertIn("0.12798", document)
            self.assertIn("481.59", document)

    def test_reject_different_checkpoint_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.write_result(directory, "rollout")
            self.write_result(directory, "teacher_forced", step=50000)
            with self.assertRaisesRegex(ValueError, "checkpoint_step"):
                build_report(directory)


if __name__ == "__main__":
    unittest.main()
