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
    host_started = time.perf_counter()
    succeeded = False
    try:
        result = original_run_batch(self, batch, *args, **kwargs)
        succeeded = True
        return result
    finally:
        if enabled:
            row = {
                **timer._aisim_snapshot,
                "host_run_batch_ms": 1000 * (time.perf_counter() - host_started),
                "host_return_unix": time.time(),
                "succeeded": succeeded,
            }
            with (Path("/campaign") / f"host-forward-rank{row['dp_rank']}.jsonl").open("a") as stream:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            timer._aisim_snapshot = None


DeviceTimer.wrap = paired_wrap
Scheduler.run_batch = paired_run_batch
Req.__init__ = paired_req_init
TokenizerManager._send_one_request = paired_send_one_request


def phase_timer(original, phase):
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        parallel = get_parallel()
        if parallel.attn_tp_rank != 0:
            return original(self, *args, **kwargs)
        requests = {}
        for value in (*args, *kwargs.values(), getattr(self, "last_batch", None)):
            candidates = getattr(value, "reqs", None)
            if candidates is None:
                candidates = [getattr(value, "req", value)]
            for request in candidates:
                if hasattr(request, "rid"):
                    requests[request.rid] = {
                        "rid": request.rid,
                        "bootstrap_room": getattr(request, "bootstrap_room", None),
                    }
        started = time.perf_counter()
        start_unix = time.time()
        succeeded = False
        try:
            result = original(self, *args, **kwargs)
            succeeded = True
            return result
        finally:
            elapsed_ms = 1000 * (time.perf_counter() - started)
            if requests or elapsed_ms >= 1:
                row = {
                    "phase": phase,
                    "start_monotonic": started,
                    "start_unix": start_unix,
                    "elapsed_ms": elapsed_ms,
                    "dp_rank": parallel.dp_rank or 0,
                    "requests": list(requests.values()),
                    "succeeded": succeeded,
                }
                destination = Path("/campaign") / f"host-phases-rank{row['dp_rank']}.jsonl"
                with destination.open("a") as stream:
                    stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    return wrapped


if os.environ.get("AISIM_PAIRED_PHASES") == "1":
    import inspect
    import triton.compiler as triton_compiler

    from sglang.srt.disaggregation.decode import DecodePreallocQueue
    from sglang.srt.hardware_backend.npu.allocator_npu import NPUPagedTokenToKVPoolAllocator
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    for owner, names in (
        (Scheduler, ("process_prefill_chunk", "send_kv_chunk", "process_batch_result_disagg_prefill")),
        (DecodePreallocQueue, ("_pre_alloc", "_send_kv_metadata", "_transfer_page_indices")),
        (ScheduleBatch, ("prepare_for_extend",)),
    ):
        for name in names:
            original_phase = getattr(owner, name)
            setattr(owner, name, phase_timer(original_phase, f"{owner.__name__}.{name}"))

    original_alloc_extend = NPUPagedTokenToKVPoolAllocator.alloc_extend
    alloc_signature = inspect.signature(original_alloc_extend)
    original_compile = triton_compiler.compile

    def write_allocator_record(row):
        parallel = get_parallel()
        row.update({"pid": os.getpid(), "dp_rank": parallel.dp_rank or 0,
                    "attn_tp_rank": parallel.attn_tp_rank})
        with (Path("/campaign") / f"allocator-events-pid{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")

    @functools.wraps(original_alloc_extend)
    def measured_alloc_extend(self, *args, **kwargs):
        parallel = get_parallel()
        if parallel.attn_tp_rank != 0:
            return original_alloc_extend(self, *args, **kwargs)
        arguments = alloc_signature.bind(self, *args, **kwargs).arguments
        row = {"event": "alloc_extend", "page_size": self.page_size,
               "free_pages_before": len(self.free_pages),
               "free_pages_alignment_16": self.free_pages.data_ptr() % 16,
               "free_pages_alignment_512": self.free_pages.data_ptr() % 512,
               "extend_num_tokens": arguments["extend_num_tokens"],
               "num_new_pages": arguments.get("num_new_pages"),
               "cpu_lengths": {name: arguments[name].tolist() for name in ("prefix_lens_cpu", "seq_lens_cpu")
                               if arguments[name].device.type == "cpu"}}
        started = time.perf_counter()
        row["start_unix"] = time.time()
        succeeded = False
        try:
            result = original_alloc_extend(self, *args, **kwargs)
            succeeded = result is not None
            return result
        finally:
            row.update({"elapsed_ms": 1000 * (time.perf_counter() - started),
                        "free_pages_after": len(self.free_pages), "succeeded": succeeded})
            write_allocator_record(row)

    @functools.wraps(original_compile)
    def measured_compile(*args, **kwargs):
        source = args[0] if args else kwargs.get("src")
        row = {"event": "triton_compile", "kernel": getattr(getattr(source, "fn", None), "__name__", None),
               "signature": str(getattr(source, "signature", None)),
               "constants": str(getattr(source, "constants", getattr(source, "constexprs", None))),
               "attributes": str(getattr(source, "attrs", None)), "start_unix": time.time()}
        started = time.perf_counter()
        succeeded = False
        try:
            result = original_compile(*args, **kwargs)
            succeeded = True
            return result
        finally:
            row.update({"elapsed_ms": 1000 * (time.perf_counter() - started), "succeeded": succeeded})
            write_allocator_record(row)

    NPUPagedTokenToKVPoolAllocator.alloc_extend = measured_alloc_extend
    triton_compiler.compile = measured_compile

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
