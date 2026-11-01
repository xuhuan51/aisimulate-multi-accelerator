import functools
import json
import os
import sys
import time
from pathlib import Path

import msgspec

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.device_timer import DeviceTimer


original_run_batch = Scheduler.run_batch
original_wrap = DeviceTimer.wrap
original_req_init = Req.__init__
original_send_one_request = TokenizerManager._send_one_request


@functools.wraps(original_send_one_request)
async def paired_send_one_request(self, tokenized_obj):
    stats = tokenized_obj.time_stats
    started = time.perf_counter()
    result = await original_send_one_request(self, tokenized_obj)
    row = {
        "event": "tokenizer_dispatched",
        "rid": tokenized_obj.rid,
        "bootstrap_room": getattr(tokenized_obj, "bootstrap_room", None),
        "prompt_tokens": len(tokenized_obj.input_ids),
        "reported_unix": time.time(),
        "host_monotonic": time.perf_counter(),
        "send_call_start_monotonic": started,
        "times": {key: value for key, value in vars(stats).items() if key.endswith("time") and isinstance(value, (int, float))},
    }
    with (Path("/campaign") / f"tokenizer-ingress-pid{os.getpid()}.jsonl").open("a") as stream:
        stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    return result


def record_request(stats, event):
    if not getattr(stats, "_aisim_record_request", False):
        return
    row = {
        "event": event,
        "reported_unix": time.time(),
        "host_monotonic": time.perf_counter(),
        "dp_rank": stats._aisim_dp_rank,
        "rid": stats._aisim_rid,
        "bootstrap_room": stats._aisim_bootstrap_room,
        "role": stats.disagg_mode_str(),
        "times": {key: value for key, value in vars(stats).items() if key.endswith("time") and isinstance(value, (int, float))},
    }
    destination = Path("/campaign") / f"request-stages-rank{row['dp_rank']}.jsonl"
    with destination.open("a") as stream:
        stream.write(json.dumps(row, separators=(",", ":")) + "\n")


@functools.wraps(original_req_init)
def paired_req_init(self, *args, **kwargs):
    original_req_init(self, *args, **kwargs)
    parallel = get_parallel()
    stats = self.time_stats
    stats._aisim_record_request = parallel.attn_tp_rank == 0
    stats._aisim_dp_rank = parallel.dp_rank or 0
    stats._aisim_rid = self.rid
    stats._aisim_bootstrap_room = self.bootstrap_room
    record_request(stats, "scheduler_received")


def request_setter(original, name):
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        record_request(self, name)
        return result
    return wrapped


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
    if snapshot is not None:
        iteration = getattr(self, "_aisim_timed_forward_iteration", 0) + 1
        self._aisim_timed_forward_iteration = iteration
        snapshot = {**snapshot, "timed_forward_iteration": iteration}
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
            "requests": [{"rid": request.rid, "bootstrap_room": request.bootstrap_room,
                          "prompt_tokens": len(request.origin_input_ids), "output_tokens": len(request.output_ids)} for request in batch.reqs],
        }
    try:
        return original_run_batch(self, batch, *args, **kwargs)
    finally:
        if enabled:
            timer._aisim_snapshot = None


DeviceTimer.wrap = paired_wrap
Scheduler.run_batch = paired_run_batch
Req.__init__ = paired_req_init
TokenizerManager._send_one_request = paired_send_one_request
for setter_name in (
    "set_prefill_bootstrap_queue_entry_time", "set_bootstrap_done_time", "set_wait_queue_entry_time",
    "set_prefill_transfer_queue_entry_time", "set_prefill_kv_transfer_finish_time",
    "set_decode_prealloc_queue_entry_time", "set_decode_transfer_queue_entry_time",
    "set_decode_prebuilt_finish_time", "set_completion_time",
):
    original_setter = getattr(SchedulerReqTimeStats, setter_name)
    setattr(SchedulerReqTimeStats, setter_name, request_setter(original_setter, setter_name))
if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args

    load_plugins()
    run_server(prepare_server_args(sys.argv[1:]))
