import functools
import json
import os
import sys
import time
from pathlib import Path

import msgspec

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.utils.device_timer import DeviceTimer


original_run_batch = Scheduler.run_batch
original_wrap = DeviceTimer.wrap


def record_interval(*, t, aisim_snapshot=None, **metadata):
    if aisim_snapshot is None:
        return
    row = {
        **aisim_snapshot,
        "reported_unix": time.time(),
        "elapsed_ms": 1000 * t,
        "category": metadata.get("category"),
    }
    destination = Path("/campaign") / f"paired-forward-rank{row['dp_rank']}.jsonl"
    with destination.open("a") as stream:
        stream.write(json.dumps(row, separators=(",", ":")) + "\n")


@functools.wraps(original_wrap)
def paired_wrap(self, metadata):
    snapshot = getattr(self, "_aisim_snapshot", None)
    return original_wrap(self, {**metadata, "aisim_snapshot": snapshot})


@functools.wraps(original_run_batch)
def paired_run_batch(self, batch, *args, **kwargs):
    timer = self.metrics_reporter.forward_pass_device_timer
    enabled = self.enable_fpm and timer is not None
    if enabled:
        if not getattr(timer, "_aisim_reporter_installed", False):
            timer.add_reporter(record_interval)
            timer._aisim_reporter_installed = True
        timer._aisim_snapshot = {
            "pid": os.getpid(),
            "dp_rank": self._fpm_dp_rank,
            "forward_iteration": self.forward_ct + 1,
            "launch_unix": time.time(),
            "scheduled_requests": msgspec.to_builtins(
                self.metrics_reporter._build_scheduled_request_metrics(batch)
            ),
            "forward_mode": batch.forward_mode.name,
        }
    try:
        return original_run_batch(self, batch, *args, **kwargs)
    finally:
        if enabled:
            timer._aisim_snapshot = None


DeviceTimer.wrap = paired_wrap
Scheduler.run_batch = paired_run_batch
if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args

    load_plugins()
    run_server(prepare_server_args(sys.argv[1:]))
