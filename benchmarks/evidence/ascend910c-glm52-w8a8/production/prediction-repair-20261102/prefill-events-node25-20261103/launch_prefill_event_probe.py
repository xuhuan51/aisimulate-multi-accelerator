import functools
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

import launch_paired_telemetry
import torch

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA, DeepseekV2DecoderLayer, DeepseekV2ForCausalLM, DeepseekV2MoE
from sglang.srt.runtime_context import get_parallel


ROOT = Path("/campaign")
CAPTURE = None
PENDING = []
LAST_POLL = None
CALLS = 0
SOURCES = {}
for owner, expected in (
    (Scheduler, "ab23556739fc4a1a089da5d33f8cb925c14b7c68ae564294b5ed13d96b427c36"),
    (ForwardBatch, "167b747b28ee6030bc30e612c6da5c0a3dc0a5611a7324799427606655b6efcc"),
    (DeepseekV2ForCausalLM, "1edf877e54ca6acf37e2d7dccb1704054175e2b78b8737bb9b617ffc97a2134e"),
):
    source = Path(inspect.getfile(owner))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError(f"Pinned source mismatch: {source}")
    SOURCES[str(source)] = digest


def append(name, row):
    with (ROOT / name).open("a") as stream:
        stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def boundary(owner):
    original = owner.forward

    @functools.wraps(original)
    def measured(self, *args, **kwargs):
        if CAPTURE is None:
            return original(self, *args, **kwargs)
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        record = {"module": owner.__name__, "layer": self.layer_id, "begin": begin, "end": end}
        CAPTURE["boundaries"].append(record)
        started = time.perf_counter()
        begin.record()
        result = original(self, *args, **kwargs)
        end.record()
        record["host_ms"] = 1000 * (time.perf_counter() - started)
        return result

    owner.forward = measured


for owner in (DeepseekV2DecoderLayer, DeepseekV2AttentionMLA, DeepseekV2MoE):
    boundary(owner)

original_model = DeepseekV2ForCausalLM.forward
original_result = Scheduler.process_batch_result
original_ingest = Scheduler.ingest_requests
original_run = Scheduler.run_batch


@functools.wraps(original_model)
def model_forward(self, *args, **kwargs):
    global CAPTURE, CALLS
    CALLS += 1
    batch = next(value for value in (*args, *kwargs.values()) if isinstance(value, ForwardBatch))
    control_path = ROOT / "prefill-event-control.json"
    control = json.loads(control_path.read_text()) if batch.is_extend_in_batch and control_path.exists() else None
    if control is not None and control.get("events", False):
        parallel = get_parallel()
        CAPTURE = {
            "case": control["id"], "model_call": CALLS, "start_unix": time.time(),
            "pid": os.getpid(), "tp_rank": parallel.tp_rank, "dp_rank": parallel.dp_rank or 0,
            "forward_mode": batch.forward_mode.name, "is_extend_in_batch": batch.is_extend_in_batch,
            "batch_size": batch.batch_size, "global_num_tokens_cpu": batch.global_num_tokens_cpu,
            "sequence_lengths": batch.seq_lens_cpu.tolist(),
            "extend_lengths": batch.extend_seq_lens_cpu,
            "prefix_lengths": batch.extend_prefix_lens_cpu,
            "sources": SOURCES, "boundaries": [],
        }
    try:
        return original_model(self, *args, **kwargs)
    finally:
        if CAPTURE is not None:
            PENDING.append(CAPTURE)
            CAPTURE = None


@functools.wraps(original_result)
def process_result(self, *args, **kwargs):
    result = original_result(self, *args, **kwargs)
    while PENDING:
        row = PENDING.pop(0)
        for record in row["boundaries"]:
            begin, end = record.pop("begin"), record.pop("end")
            end.synchronize()
            record["device_ms"] = begin.elapsed_time(end)
        append(f"prefill-events-rank{row['tp_rank']}.jsonl", row)
    return result


@functools.wraps(original_ingest)
def ingest(self, *args, **kwargs):
    global LAST_POLL
    started, started_unix = time.perf_counter(), time.time()
    result = original_ingest(self, *args, **kwargs)
    LAST_POLL = {
        "poll_start_monotonic": started, "poll_start_unix": started_unix,
        "poll_end_monotonic": time.perf_counter(),
        "received_ids": [getattr(request, "rid", None) for request in result or []],
    }
    return result


@functools.wraps(original_run)
def run_batch(self, batch, *args, **kwargs):
    parallel = get_parallel()
    if parallel.attn_tp_rank == 0 and LAST_POLL is not None:
        append(f"input-polls-rank{parallel.dp_rank or 0}.jsonl", {
            **LAST_POLL, "launch_monotonic": time.perf_counter(), "launch_unix": time.time(),
            "forward_iteration": self.forward_ct + 1, "forward_mode": batch.forward_mode.name,
            "request_ids": [request.rid for request in batch.reqs],
        })
    return original_run(self, batch, *args, **kwargs)


DeepseekV2ForCausalLM.forward = model_forward
Scheduler.process_batch_result = process_result
Scheduler.ingest_requests = ingest
Scheduler.run_batch = run_batch


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args

    load_plugins()
    run_server(prepare_server_args(sys.argv[1:]))
