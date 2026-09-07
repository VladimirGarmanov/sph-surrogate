"""Optional wall-time measurements, with completed CUDA work at stage boundaries."""
from contextlib import contextmanager, nullcontext
import time

import torch


STAGES = ("tree", "features", "neighbors", "gather", "normalize", "to_device",
          "network", "to_cpu", "update", "history", "other")
GPU_STAGES = {"to_device", "network", "to_cpu"}


class StepTimer:
    def __init__(self, device):
        self.device = torch.device(device)
        self.seconds = dict.fromkeys(STAGES, 0.)

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    @contextmanager
    def measure(self, stage):
        if stage in GPU_STAGES:
            self.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if stage in GPU_STAGES:
                self.synchronize()
            self.seconds[stage] += time.perf_counter() - start

    def finish(self, total_seconds):
        self.seconds["other"] = max(0., total_seconds - sum(
            value for key, value in self.seconds.items() if key != "other"))
        return [self.seconds[key] for key in STAGES]


def measure(timer, stage):
    return timer.measure(stage) if timer is not None else nullcontext()
