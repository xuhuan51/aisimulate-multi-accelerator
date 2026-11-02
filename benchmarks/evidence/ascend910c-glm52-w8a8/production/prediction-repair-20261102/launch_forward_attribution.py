import functools
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

import launch_paired_telemetry

from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.layers.layer_boundary.stage import StageBoundary
from sglang.srt.layers.layer_boundary.exit import FfnExit
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import NPUW8A8Int8LinearMethod, NPUW8A8Int8DynamicLinearMethod
from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUW8A8Int8MoEMethod
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2AttentionMLA,
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
    DeepseekV2MLP,
    DeepseekV2Model,
    DeepseekV2MoE,
)


CONTEXT = None
SOURCES = {}
ORIGINAL_RUN_BATCH = Scheduler.run_batch


def install(owner, method):
    original = getattr(owner, method)
    name = f"{owner.__name__}.{method}"
    source = inspect.getfile(original)
    SOURCES[source] = hashlib.sha256(Path(source).read_bytes()).hexdigest()

    @functools.wraps(original)
    def timed(self, *args, **kwargs):
        context = CONTEXT
        if context is None:
            return original(self, *args, **kwargs)
        parent = context["stack"][-1] if context["stack"] else None
        layer_id = getattr(self, "layer_id", parent["layer"] if parent else None)
        label = name + (":" + str(self.kind) if isinstance(self, StageBoundary) else "")
        frame = {"layer": layer_id, "children_ns": 0, "children_cpu_ns": 0}
        if isinstance(self, AscendAttnBackend) and "length_metadata" not in context:
            lengths = self.forward_metadata.seq_lens_cpu_int
            context["length_metadata"] = None if lengths is None else {
                "device": str(lengths.device), "dtype": str(lengths.dtype),
                "shape": list(lengths.shape), "pinned": lengths.is_pinned(),
            }
        context["stack"].append(frame)
        cpu_started = time.thread_time_ns()
        started = time.perf_counter_ns()
        try:
            return original(self, *args, **kwargs)
        finally:
            elapsed = time.perf_counter_ns() - started
            cpu_elapsed = time.thread_time_ns() - cpu_started
            context["stack"].pop()
            if parent is not None:
                parent["children_ns"] += elapsed
                parent["children_cpu_ns"] += cpu_elapsed
            key = (label, layer_id)
            record = context["spans"].setdefault(key, {"name": label, "layer": layer_id, "calls": 0, "inclusive_ns": 0, "exclusive_ns": 0, "cpu_inclusive_ns": 0, "cpu_exclusive_ns": 0})
            record["calls"] += 1
            record["inclusive_ns"] += elapsed
            record["exclusive_ns"] += elapsed - frame["children_ns"]
            record["cpu_inclusive_ns"] += cpu_elapsed
            record["cpu_exclusive_ns"] += cpu_elapsed - frame["children_cpu_ns"]

    setattr(owner, method, timed)


for owner, methods in (
    (DeepseekV2ForCausalLM, ("forward",)),
    (DeepseekV2Model, ("forward",)),
    (DeepseekV2DecoderLayer, ("forward",)),
    (DeepseekV2AttentionMLA, ("forward", "forward_prepare", "forward_core")),
    (DeepseekV2MoE, ("forward", "forward_deepep", "_forward_shared_experts")),
    (DeepseekV2MLP, ("forward",)),
    (Indexer, ("forward_npu",)),
    (StageBoundary, ("prepare", "finish", "complete_now")),
    (FfnExit, ("finish",)),
    (LogitsProcessor, ("forward",)),
    (GroupCoordinator, ("all_reduce", "reduce_scatter_tensor", "reduce_scatter", "reduce_scatterv", "all_gather_into_tensor", "all_gather", "all_gatherv")),
    (AscendAttnBackend, ("forward_sparse",)),
    (NPUW8A8Int8LinearMethod, ("apply",)),
    (NPUW8A8Int8DynamicLinearMethod, ("apply",)),
    (NPUW8A8Int8MoEMethod, ("apply",)),
    (FusedMoE, ("forward", "forward_impl", "run_moe_core")),
    (DeepEPDispatcher, ("dispatch", "combine", "dispatch_a", "dispatch_b", "combine_a", "combine_b")),
):
    for method in methods:
        install(owner, method)


@functools.wraps(ORIGINAL_RUN_BATCH)
def run_batch(self, batch, *args, **kwargs):
    global CONTEXT
    control_path = Path("/campaign/attribution-control.json")
    control = json.loads(control_path.read_text()) if control_path.exists() else {"enabled": False, "case": "startup"}
    enabled = control["enabled"] and getattr(self, "enable_fpm", False)
    context = {"stack": [], "spans": {}} if enabled else None
    CONTEXT = context
    cpu_started = time.thread_time_ns()
    started = time.perf_counter_ns()
    metadata = {"pid": os.getpid(), "rank": getattr(self, "_fpm_dp_rank", None),
                "tp_rank": getattr(self, "tp_rank", None), "case": control["case"],
                "forward_mode": batch.forward_mode.name, "forward_iteration": self.forward_ct + 1,
                "requests": [{"rid": request.rid, "input": len(request.origin_input_ids),
                              "output": len(request.output_ids)} for request in batch.reqs],
                "launch_unix": time.time(), "enabled": enabled}
    try:
        return ORIGINAL_RUN_BATCH(self, batch, *args, **kwargs)
    finally:
        elapsed = time.perf_counter_ns() - started
        cpu_elapsed = time.thread_time_ns() - cpu_started
        CONTEXT = None
        if getattr(self, "enable_fpm", False):
            row = {**metadata, "host_ns": elapsed, "thread_cpu_ns": cpu_elapsed,
                   "length_metadata": context.get("length_metadata") if context else None,
                   "spans": list(context["spans"].values()) if context else []}
            destination = Path("/campaign") / f"attribution-rank{metadata['rank']}.jsonl"
            with destination.open("a") as stream:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            source_path = Path("/campaign") / f"attribution-source-pid{os.getpid()}.json"
            if not source_path.exists():
                source_path.write_text(json.dumps(SOURCES, indent=2) + "\n")


Scheduler.run_batch = run_batch


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args

    load_plugins()
    run_server(prepare_server_args(sys.argv[1:]))
