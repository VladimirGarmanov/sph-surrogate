"""Проверка серверного launcher без CUDA и полного набора данных."""
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from surrogate.particle.train import load_checkpoint, main
from test_particle_full_frames import TinyRun


class LauncherTests(unittest.TestCase):
    def test_k32_shell_command_trains_full_frames_with_baseline_horizon_and_noise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts, binaries = root / "scripts", root / "bin"
            scripts.mkdir()
            binaries.mkdir()
            source = Path(__file__).resolve().parents[1] / "scripts" / "train_particle_k32.sh"
            launcher = scripts / source.name
            shutil.copyfile(source, launcher)
            captured = root / "argv.json"
            # Исполняем настоящий shell launcher, но сначала только записываем его аргументы.
            capture = "import json,os,sys; open(os.environ['PARTICLE_LAUNCH_ARGS'],'w').write(json.dumps(sys.argv[1:]))"
            python = binaries / "python"
            python.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} -c {shlex.quote(capture)} \"$@\"\n")
            python.chmod(0o755)
            env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ.get("PATH", ""),
                       PARTICLE_LAUNCH_ARGS=str(captured))
            subprocess.run(["bash", str(launcher)], env=env, check=True, capture_output=True, text=True)
            args = json.loads(captured.read_text())
            self.assertEqual(args[:3], ["-u", "-m", "surrogate.particle.train"])
            out = root / "checkpoints" / "particle_k32_h3"
            # Те же аргументы проходят реальный parser и один полный учебный кадр на CPU.
            argv = ["train", *args[3:], "--out_dir", str(out), "--hidden", "8", "--device", "cpu",
                    "--stats_frames", "1", "--val_samples", "1", "--holdout", "phi30_c500"]
            runs = [TinyRun(frames=14), TinyRun("phi30_c500", frames=14)]
            with patch("sys.argv", argv), patch("surrogate.particle.train.discover_runs", return_value=runs):
                with contextlib.redirect_stdout(io.StringIO()):
                    main()
            checkpoint = load_checkpoint(out / "model.pt")
            cfg = checkpoint["config"]
            self.assertEqual((cfg["training_mode"], cfg["epochs"], checkpoint["step"]), ("full_frames", 1, 1))
            self.assertEqual((cfg["neighbors"], cfg["history_frames"], cfg["neighbor_history_frames"]), (32, 8, 3))
            self.assertEqual((cfg["prediction_horizon"], cfg["input_noise_std"]), (5, .05))
            self.assertEqual(checkpoint["frame_layout"][0]["soil"], runs[0].n_soil)


if __name__ == "__main__":
    unittest.main()
