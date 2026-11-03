import functools
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

import launch_layer_replay_probe as replay
import torch
import torch.distributed as distributed
from sglang.srt.layers.layer_boundary.exit import FfnExit
from sglang.srt.layers.layer_boundary.stage import StageBoundary
from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimLinearMethod


TRACE = None
STACK = []
SOURCES = {}


def instrument(owner, method, instance=True):
    original = getattr(owner, method)
    source = Path(inspect.getfile(original))
    SOURCES[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        if TRACE is None:
            return original(*args, **kwargs)
        device = torch.get_device_module()
        events = TRACE["mode"] == "events"
        begin = device.Event(enable_timing=True) if events else None
        end = device.Event(enable_timing=True) if events else None
        record = {"id": len(TRACE["operations"]), "parent": STACK[-1] if STACK else None,
                  "operation": owner.__name__ + "." + method, "begin": begin, "end": end}
        TRACE["operations"].append(record)
        STACK.append(record["id"])
        if events:
            begin.record()
        started = time.perf_counter_ns()
        cpu_started = time.thread_time_ns()
        try:
            result = original(*args, **kwargs)
        finally:
            record["host_cpu_ms"] = (time.thread_time_ns() - cpu_started) / 1e6
            record["host_ms"] = (time.perf_counter_ns() - started) / 1e6
            record["host_start_ns"] = started
            if events:
                end.record()
            STACK.pop()
        record["inputs"] = [replay.shapes(value) for value in (*args[1 if instance else 0:], *kwargs.values())
                            if isinstance(value, torch.Tensor)]
        if instance and isinstance(args[0], StageBoundary):
            record["stage"] = str(args[0].kind)
        return result

    setattr(owner, method, wrapped)


for owner, method, instance in (
    (StageBoundary, "prepare", True), (StageBoundary, "finish", True), (FfnExit, "finish", True),
    (replay.DeepseekV2AttentionMLA, "forward", True), (replay.DeepseekV2MoE, "forward", True),
    (ModelSlimLinearMethod, "apply", True),
    (distributed, "all_gather_into_tensor", False), (distributed, "reduce_scatter_tensor", False),
    (distributed, "all_reduce", False),
):
    instrument(owner, method, instance)


def probe():
    global TRACE
    control = replay.get_parallel().tp_group.cpu_group
    rank = distributed.get_rank(group=control)
    device = torch.get_device_module()
    if distributed.get_world_size(group=control) != 16:
        raise ValueError("This diagnostic requires the recorded sixteen-rank topology")
    signatures = [None] * 16
    keys = sorted(replay.CAPTURED)
    distributed.all_gather_object(signatures, keys, group=control)
    if any(value != keys for value in signatures) or len(keys) != len(replay.TARGET_LAYERS):
        raise ValueError("Incomplete or unsynchronized captured layer boundaries")
    records = []
    for key in keys:
        module, saved, forward_batch, expected = replay.CAPTURED[key]
        rank_lengths = [None] * 16
        distributed.all_gather_object(rank_lengths, forward_batch.seq_lens_cpu.tolist(), group=control)
        global_lengths = rank_lengths[::4]
        final_residual = forward_batch.residual_stream
        populations = []
        for mode in ("control_before", "host", "events", "control_after"):
            samples = []
            for repeat in range(12):
                args, kwargs, residual_stream = replay.copy_inputs(saved)
                forward_batch.residual_stream = residual_stream
                device.synchronize()
                distributed.barrier(group=control)
                begin, end = device.Event(enable_timing=True), device.Event(enable_timing=True)
                trace = {"mode": mode, "operations": []}
                TRACE = trace if mode in ("host", "events") else None
                begin.record()
                started = time.perf_counter_ns()
                cpu_started = time.thread_time_ns()
                try:
                    with replay.get_attn_tp_context().maybe_input_scattered(forward_batch):
                        output = module(*args, **kwargs)
                finally:
                    host_cpu_ms = (time.thread_time_ns() - cpu_started) / 1e6
                    host_ms = (time.perf_counter_ns() - started) / 1e6
                    end.record()
                    TRACE = None
                end.synchronize()
                validation = replay.check_boundary_outputs(key, output, expected, forward_batch, global_lengths)
                for operation in trace["operations"]:
                    op_begin, op_end = operation.pop("begin"), operation.pop("end")
                    operation["host_start_ms"] = (operation.pop("host_start_ns") - started) / 1e6
                    if op_begin is not None:
                        operation["device_start_ms"] = begin.elapsed_time(op_begin)
                        operation["device_ms"] = op_begin.elapsed_time(op_end)
                samples.append({"repeat": repeat, "warmup": repeat < 3, "device_ms": begin.elapsed_time(end),
                                "host_ms": host_ms, "host_cpu_ms": host_cpu_ms, "validation": validation,
                                "operations": trace["operations"]})
                del output, args, kwargs
            populations.append({"mode": mode, "samples": samples})
        forward_batch.residual_stream = final_residual
        records.append({"layer": key[1], "module": key[0], "sequence_lengths": forward_batch.seq_lens_cpu.tolist(),
                        "global_sequence_lengths": global_lengths, "forward_mode": str(forward_batch.forward_mode),
                        "batch_size": forward_batch.batch_size, "populations": populations})
    result = {"schema": "aisimulate-component-diagnosis/v1", "status": "completed", "rank": rank,
              "control": replay.CONTROL, "model_call": replay.MODEL_CALLS,
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "source_sha256": {**replay.SOURCES, **SOURCES},
              "scope": "Paired control/host/event isolated layer invocation; nested operations overlap and must not be added. Diagnostic only, not serving ground truth or a fitted residual.",
              "layers": records}
    destination = Path("/campaign") / replay.CONTROL["id"] / f"components-rank{rank}.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    replay.CAPTURED.clear()


replay.probe = probe

if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args

    load_plugins()
    run_server(prepare_server_args(sys.argv[1:]))
