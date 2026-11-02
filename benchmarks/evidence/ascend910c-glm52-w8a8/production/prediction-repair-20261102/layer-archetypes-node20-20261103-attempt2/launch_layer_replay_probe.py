import cProfile
import functools
import hashlib
import inspect
import json
import os
import pstats
import sys
import time
from pathlib import Path

import launch_paired_telemetry
import msgspec
import torch
import torch.distributed as distributed

from sglang.srt.layers.layer_boundary import get_attn_tp_context
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2AttentionMLA,
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
    DeepseekV2MoE,
)


MODEL_CALLS = 0
CAPTURE = False
CAPTURED = {}
SOURCES = {}
PROBE_CALL = int(os.environ.get("AISIM_PROBE_CALL", "128"))
TARGET_LAYERS = {int(value) for value in os.environ.get("AISIM_PROBE_LAYERS", "3,4").split(",")}
LAYER_ONLY = os.environ.get("AISIM_PROBE_BOUNDARY") == "layer"
REPEATS = 12


def copy_inputs(value, memo=None):
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.Tensor):
        result = value.clone()
        memo[id(value)] = result
        return result
    if isinstance(value, ResidualStream):
        result = ResidualStream()
        memo[id(value)] = result
        result.residual = copy_inputs(value.residual, memo)
        result.pending = copy_inputs(value.pending, memo)
        return result
    if isinstance(value, msgspec.Struct):
        result = type(value)(**{field.name: copy_inputs(getattr(value, field.name), memo) for field in msgspec.structs.fields(type(value))})
        memo[id(value)] = result
        return result
    if isinstance(value, tuple):
        return tuple(copy_inputs(item, memo) for item in value)
    if isinstance(value, list):
        return [copy_inputs(item, memo) for item in value]
    if isinstance(value, dict):
        return {key: copy_inputs(item, memo) for key, item in value.items()}
    return value


def check_outputs(observed, expected):
    if isinstance(expected, torch.Tensor):
        tolerance = 0.02 if expected.is_floating_point() else 0.0
        torch.testing.assert_close(observed, expected, rtol=tolerance, atol=tolerance)
    elif isinstance(expected, (tuple, list)):
        for output, reference in zip(observed, expected, strict=True):
            check_outputs(output, reference)
    elif isinstance(expected, msgspec.Struct):
        for field in msgspec.structs.fields(type(expected)):
            check_outputs(getattr(observed, field.name), getattr(expected, field.name))


def shapes(value):
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device), "stride": list(value.stride())}
    if isinstance(value, (tuple, list)):
        return [shapes(item) for item in value]
    if isinstance(value, dict):
        return {key: shapes(item) for key, item in value.items()}
    return type(value).__name__


def install(owner):
    original = owner.forward
    path = inspect.getfile(original)
    SOURCES[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @functools.wraps(original)
    def captured(self, *args, **kwargs):
        layer = getattr(self, "layer_id", None)
        if CAPTURE and layer in TARGET_LAYERS:
            forward_batch = next(value for value in (*args, *kwargs.values()) if isinstance(value, ForwardBatch))
            saved = copy_inputs((args, kwargs, forward_batch.residual_stream))
            output = original(self, *args, **kwargs)
            CAPTURED[(owner.__name__, layer)] = (self, saved, forward_batch, copy_inputs(output))
            return output
        return original(self, *args, **kwargs)

    owner.forward = captured


for owner in ((DeepseekV2DecoderLayer,) if LAYER_ONLY else (DeepseekV2AttentionMLA, DeepseekV2MoE, DeepseekV2DecoderLayer)):
    install(owner)


ORIGINAL_FORWARD = DeepseekV2ForCausalLM.forward


def probe():
    control = get_parallel().tp_group.cpu_group
    rank = distributed.get_rank(group=control)
    device = torch.get_device_module()
    records = []
    keys = sorted(CAPTURED)
    signatures = [None] * distributed.get_world_size(group=control)
    distributed.all_gather_object(signatures, keys, group=control)
    expected_count = len(TARGET_LAYERS) if LAYER_ONLY else 3 * len(TARGET_LAYERS)
    if any(signature != keys for signature in signatures) or len(keys) != expected_count:
        raise RuntimeError(f"Layer replay requires identical {expected_count} captured boundaries: {signatures}")
    for key in keys:
        module, saved, forward_batch, expected = CAPTURED[key]
        original_args, original_kwargs, _ = saved
        final_residual_stream = forward_batch.residual_stream
        samples = []
        for repeat in range(REPEATS):
            args, kwargs, residual_stream = copy_inputs(saved)
            forward_batch.residual_stream = residual_stream
            device.synchronize()
            distributed.barrier(group=control)
            begin, end = device.Event(enable_timing=True), device.Event(enable_timing=True)
            cpu_start = time.thread_time_ns()
            started = time.perf_counter_ns()
            begin.record()
            with get_attn_tp_context().maybe_input_scattered(forward_batch):
                output = module(*args, **kwargs)
            end.record()
            submitted = time.perf_counter_ns()
            cpu_ns = time.thread_time_ns() - cpu_start
            end.synchronize()
            check_outputs(output, expected)
            samples.append({"warmup": repeat < 3, "repeat": repeat,
                            "device_ms": begin.elapsed_time(end),
                            "host_submit_ms": (submitted - started) / 1e6,
                            "host_thread_cpu_ms": cpu_ns / 1e6,
                            "output": shapes(output)})
            del output, args, kwargs
        function_profile = None
        if os.environ.get("AISIM_CAPTURE_CPROFILE") == "1":
            args, kwargs, residual_stream = copy_inputs(saved)
            forward_batch.residual_stream = residual_stream
            device.synchronize()
            distributed.barrier(group=control)
            profiler = cProfile.Profile() if rank == 0 else None
            if profiler is not None:
                profiler.enable()
            with get_attn_tp_context().maybe_input_scattered(forward_batch):
                output = module(*args, **kwargs)
            if profiler is not None:
                profiler.disable()
                function_profile = [{"file": identity[0], "line": identity[1], "function": identity[2],
                                     "primitive_calls": values[0], "calls": values[1],
                                     "self_seconds": values[2], "cumulative_seconds": values[3]}
                                    for identity, values in pstats.Stats(profiler).stats.items()]
            device.synchronize()
            check_outputs(output, expected)
            del output, args, kwargs
        forward_batch.residual_stream = final_residual_stream
        records.append({"module": key[0], "layer": key[1], "inputs": shapes(original_args),
                        "kwargs": shapes(original_kwargs), "samples": samples,
                        "sequence_lengths": forward_batch.seq_lens_cpu.tolist(),
                        "forward_mode": str(forward_batch.forward_mode),
                        "batch_size": forward_batch.batch_size,
                        "global_num_tokens_cpu": forward_batch.global_num_tokens_cpu,
                        "function_profile": function_profile})
    result = {"scope": "Isolated replay of captured real layers; cloned input values and actual resident weights. Additional device events/barriers perturb timing; not serving latency ground truth or an additive residual.",
              "model_call": MODEL_CALLS, "rank": rank, "pid": os.getpid(),
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "source_sha256": SOURCES,
              "boundaries": records, "status": "completed"}
    path = Path("/campaign") / f"layer-replay-rank{rank}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2) + "\n")
    CAPTURED.clear()


@functools.wraps(ORIGINAL_FORWARD)
def model_forward(self, *args, **kwargs):
    global MODEL_CALLS, CAPTURE
    MODEL_CALLS += 1
    CAPTURE = MODEL_CALLS == PROBE_CALL
    result = ORIGINAL_FORWARD(self, *args, **kwargs)
    if CAPTURE:
        CAPTURE = False
        probe()
    return result


DeepseekV2ForCausalLM.forward = model_forward


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args

    load_plugins()
    run_server(prepare_server_args(sys.argv[1:]))
