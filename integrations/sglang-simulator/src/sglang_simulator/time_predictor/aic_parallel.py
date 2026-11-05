# SPDX-License-Identifier: Apache-2.0
# Modified for AISimulate multi-accelerator workload replay.
# Upstream attribution is retained in LICENSE and source-manifest.json.
"""AIConfigurator pricing for ordinary dense/MoE attention-DP forwards.

Native SGLang supplies the actual per-group batch and padding decision. AIC
supplies operator and collective costs. Layers execute serially (no overlap);
the simulator explicitly orders the local attention and shared MLP phases.
"""

import math
from dataclasses import replace

from sglang_simulator.time_predictor.base import ScheduleBatch, ScheduleRequest

_DENSE_MLP = {"gate_ffn1_gemm", "act_gate", "ffn2_gemm"}
_SHARED = _DENSE_MLP | {"router_gemm", "moe"}
_COMM = {"moe_pre_dispatch", "moe_post_dispatch", "ar_1", "ar_2", "p2p"}
_KNOWN = (
    _SHARED
    | _COMM
    | {
        "embedding",
        "embedding_ar",
        "add_norm_1",
        "add_norm_2",
        "qkv_gemm",
        "attention",
        "proj_gemm",
        "logits_gemm",
    }
)


def op_name(op):
    return op._name.split("_", 1)[1]


class AICParallelPredictor:
    def __init__(self, predictor):
        from sglang_simulator.time_predictor.aiconfigurator import get_perf_model

        self.predictor = predictor
        self.model = predictor._session._model
        self.database = predictor._session._database
        self.config = predictor.config
        if type(self.model).__name__ not in {"LLAMAModel", "MOEModel"}:
            raise ValueError(
                "Attention-DP operator adapter currently supports AIC LLAMA and MOE "
                f"models; received {type(self.model).__name__}"
            )
        self.global_model = get_perf_model(
            replace(self.config, dp_size=1), predictor.model
        )
        self.cache = {}
        for op in [*self.model.context_ops, *self.model.generation_ops]:
            if op_name(op) not in _KNOWN:
                raise ValueError(f"Unmapped AIConfigurator operation: {op._name}")

    def query(self, op, batch, tokens):
        # MoE.query multiplies x by attention_dp_size. Supplying the actual
        # global work divided by that factor cancels its equal-load assumption.
        name = op_name(op)
        x = tokens / self.config.dp_size if name == "moe" else tokens
        bs = batch.batch_size
        if name == "attention" and bs == 0:
            return 0.0
        if tokens == 0:
            return 0.0
        if name == "logits_gemm":
            x = bs
            if not x:
                return 0.0
        if bs:
            mean_new = sum(r.extend_length for r in batch.reqs) / bs
            mean_past = sum(r.past_kv_length for r in batch.reqs) / bs
        else:
            mean_new, mean_past = 1, 0
        kwargs = {
            "x": x,
            "batch_size": bs,
            "beam_width": 1,
            "s": max(1, math.ceil(mean_past + 1))
            if batch.is_decode()
            else max(1, math.ceil(mean_new)),
            "prefix": math.floor(mean_past),
        }
        if bs and not batch.is_decode():
            kwargs["seq_imbalance_correction_scale"] = (
                self.predictor.ctx_attn_flops_ratio_with_avg(batch.reqs)
            )
        key = (id(op), tuple(sorted(kwargs.items())))
        if key not in self.cache:
            if len(self.cache) >= 4096:
                self.cache.clear()
            self.cache[key] = float(op.query(self.database, **kwargs)) / 1000
        return self.cache[key]

    def communication(self, kind, size, tokens):
        from aiconfigurator.sdk.common import CommQuantMode

        if size == 1 or tokens == 0:
            return 0.0
        # Native non-DeepEP DP gather transports the model activation dtype.
        volume = int(tokens * self.predictor.model.hidden_size)
        return (
            float(self.database.query_nccl(CommQuantMode.half, size, kind, volume))
            / 1000
        )

    def predict(self, batches, padded_tokens, padding_mode):
        dp = self.config.dp_size
        if padded_tokens is None or len(padded_tokens) != dp:
            raise ValueError(
                "Attention-DP prediction requires native per-group padding metadata"
            )
        context = any(b.reqs and not b.is_decode() for b in batches)
        ops = self.model.context_ops if context else self.model.generation_ops
        global_ops = (
            self.global_model.context_ops
            if context
            else self.global_model.generation_ops
        )
        global_by_name = {op_name(op): op for op in global_ops}
        local_ops = [
            {
                op_name(op): op
                for op in (
                    self.model.generation_ops
                    if batch.is_decode()
                    else self.model.context_ops
                )
            }
            for batch in batches
        ]
        layers = self.predictor.model.num_hidden_layers
        local_before, local_attention, local_after = ([0.0] * dp for _ in range(3))
        global_tokens = sum(padded_tokens)
        global_batch = ScheduleBatch(
            [ScheduleRequest(global_tokens, 0)], "EXTEND" if context else "DECODE"
        )
        shared = 0.0
        details = {}
        for op in ops:
            name = op_name(op)
            if name in _COMM:
                continue
            if name in _SHARED:
                target_op = global_by_name[name] if name in _DENSE_MLP else op
                cost = self.query(target_op, global_batch, global_tokens)
                shared += cost
                details[op._name] = {"scope": "global_mlp", "seconds": cost}
                continue
            # Different DP groups can prefill and decode in the same iteration.
            # In particular, context and generation attention query distinct
            # performance tables and interpret sequence length differently.
            # Other operators use token counts from the global padding plan;
            # their context/generation lists need not have matching entries.
            selected_ops = [
                group_ops[name] if name == "attention" else op
                for group_ops in local_ops
            ]
            values = [
                self.query(local_op, b, n)
                for local_op, b, n in zip(
                    selected_ops, batches, padded_tokens, strict=True
                )
            ]
            details[op._name] = {
                "scope": "attention_dp_group",
                "operations": [local_op._name for local_op in selected_ops],
                "seconds": values,
            }
            target = (
                local_before
                if name in {"embedding", "embedding_ar"}
                else local_after
                if name == "logits_gemm"
                else local_attention
            )
            for i, value in enumerate(values):
                target[i] += value

        # SGLang non-DeepEP: attention-TP reduce-scatter, TP-world gather;
        # reverse after the MLP. SUM_LEN gathers are represented by one
        # all-reduce on the disjoint global buffer, matching native dp_gather.
        tp, attn_tp = self.config.tp_size, self.config.attn_tp_size
        if padding_mode == "MAX_LEN":
            pre_local = [
                self.communication("reduce_scatter", attn_tp, n) for n in padded_tokens
            ]
            post_local = [
                self.communication("all_gather", attn_tp, n) for n in padded_tokens
            ]
            pre_global = self.communication("all_gather", tp, global_tokens)
            post_global = self.communication("reduce_scatter", tp, global_tokens)
        else:
            pre_local = post_local = [0.0] * dp
            pre_global = self.communication("all_reduce", tp, global_tokens)
            post_global = self.communication("all_reduce", tp, global_tokens)

        ready = list(local_before)
        events = []
        for layer in range(layers):
            arrivals = [
                ready[i] + local_attention[i] / layers + pre_local[i] for i in range(dp)
            ]
            gather_start = max(arrivals)
            mlp_end = gather_start + pre_global + shared / layers
            end = mlp_end + post_global
            ready = [end + post_local[i] for i in range(dp)]
            events.append(
                {
                    "layer": layer,
                    "gather_start": gather_start,
                    "mlp_end": mlp_end,
                    "combine_end": end,
                    "arrivals": arrivals,
                }
            )
        durations = [ready[i] + local_after[i] for i in range(dp)]
        scale = (
            self.predictor.prefill_scale_factor
            if context
            else self.predictor.decode_scale_factor
        )
        durations = [value * scale for value in durations]
        if context:
            durations = [
                max(value, self.predictor.prefill_min_latency / 1000)
                for value in durations
            ]
        return {
            "durations": durations,
            "timing_model": "aic_serial_layers",
            "expert_workload": "balanced"
            if not hasattr(self.model.config, "workload_distribution")
            else self.model.config.workload_distribution,
            "attention_shape": "per_group_mean_with_flops_correction",
            "operator_costs": details,
            "layer_events": events,
            "communication": {
                "pre_local": pre_local,
                "post_local": post_local,
                "pre_global": pre_global,
                "post_global": post_global,
            },
        }
