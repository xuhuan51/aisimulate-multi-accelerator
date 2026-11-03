import functools
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

import launch_paired_telemetry

from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV2MoE
from sglang.srt.runtime_context import get_parallel


ROOT = Path("/campaign")
CAPTURE = None
CURRENT_LAYER = None
CALLS = 0
SOURCES = {}
for owner, expected in (
    (ForwardBatch, "167b747b28ee6030bc30e612c6da5c0a3dc0a5611a7324799427606655b6efcc"),
    (DeepseekV2ForCausalLM, "1edf877e54ca6acf37e2d7dccb1704054175e2b78b8737bb9b617ffc97a2134e"),
):
    path = Path(inspect.getfile(owner))
    signature = hashlib.sha256(path.read_bytes()).hexdigest()
    if signature != expected:
        raise RuntimeError(f"Pinned routing source mismatch: {path}")
    SOURCES[str(path)] = signature
path = Path(inspect.getfile(DeepEPDispatcher))
SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
ORIGINAL_MODEL = DeepseekV2ForCausalLM.forward
ORIGINAL_MOE = DeepseekV2MoE.forward
ORIGINAL_DISPATCH = DeepEPDispatcher.dispatch


@functools.wraps(ORIGINAL_MOE)
def moe_forward(self, *args, **kwargs):
    global CURRENT_LAYER
    previous, CURRENT_LAYER = CURRENT_LAYER, self.layer_id
    try:
        return ORIGINAL_MOE(self, *args, **kwargs)
    finally:
        CURRENT_LAYER = previous


@functools.wraps(ORIGINAL_DISPATCH)
def dispatch(self, *args, **kwargs):
    result = ORIGINAL_DISPATCH(self, *args, **kwargs)
    if CAPTURE is not None:
        counts = getattr(result, "num_recv_tokens_per_expert", None)
        if not isinstance(counts, (list, tuple)) or len(counts) != 8:
            raise TypeError("TP32 normal dispatch must expose eight existing host expert counts")
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("Invalid runtime expert assignment count")
        if CURRENT_LAYER is None or str(CURRENT_LAYER) in CAPTURE["layers"]:
            raise ValueError("Missing or repeated MoE layer identity")
        CAPTURE["layers"][str(CURRENT_LAYER)] = {
            "expert_counts": list(counts), "received_shape": list(result.hidden_states.shape),
            "received_dtype": str(result.hidden_states.dtype),
        }
    return result


@functools.wraps(ORIGINAL_MODEL)
def model_forward(self, *args, **kwargs):
    global CAPTURE, CALLS
    CALLS += 1
    batch = next(value for value in (*args, *kwargs.values()) if isinstance(value, ForwardBatch))
    path = ROOT / "routing-profile-control.json"
    control = json.loads(path.read_text()) if batch.is_extend_in_batch and path.exists() else None
    if control is not None:
        parallel = get_parallel()
        if (parallel.tp_size, parallel.attn_tp_size) != (32, 1):
            raise ValueError("Unexpected routing collection topology")
        CAPTURE = {
            "case": control["id"], "control": control, "model_call": CALLS, "launch_unix": time.time(),
            "pid": os.getpid(), "tp_rank": parallel.tp_rank, "dp_rank": parallel.dp_rank or 0,
            "forward_mode": batch.forward_mode.name, "global_num_tokens_cpu": batch.global_num_tokens_cpu,
            "sequence_lengths": batch.seq_lens_cpu.tolist(), "extend_lengths": batch.extend_seq_lens_cpu,
            "prefix_lengths": batch.extend_prefix_lens_cpu, "sources": SOURCES, "layers": {},
            "scope": "Existing host-side per-expert assignment counts only; no device copy, extra synchronization, layer replay, or timing fit",
        }
    try:
        output = ORIGINAL_MODEL(self, *args, **kwargs)
        if CAPTURE is not None:
            if sorted(map(int, CAPTURE["layers"])) != list(range(3, 78)):
                raise ValueError("All75 MoE layers must contribute observed expert counts")
            with (ROOT / f"routing-profile-rank{CAPTURE['tp_rank']}.jsonl").open("a") as stream:
                stream.write(json.dumps(CAPTURE, separators=(",", ":")) + "\n")
        return output
    finally:
        CAPTURE = None


DeepseekV2MoE.forward = moe_forward
DeepEPDispatcher.dispatch = dispatch
DeepseekV2ForCausalLM.forward = model_forward


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args

    load_plugins()
    run_server(prepare_server_args(sys.argv[1:]))
