"""Qwen397B physical composition for audited Ascend ModelSlim execution paths.

Graph construction only: native operators own timing and weight arithmetic.
Boundary contracts distinguish historical Eager data and the TP4 Graph pilot.
"""

import json

import aisimulate_core._native as core
from aisimulate_core.sdk import common
from aisimulate_core.sdk.errors import InvalidEngineConfigurationError
from aisimulate_core.sdk.fpm_config import OpLevelCompileConfig


def validate_checkpoint(raw):
    text = raw.get("text_config", {})
    expected = {
        "hidden_size": 4096,
        "num_hidden_layers": 60,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "vocab_size": 248320,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 64,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "rms_norm_eps": 1e-6,
    }
    if any(text.get(k) != v for k, v in expected.items()):
        raise InvalidEngineConfigurationError("Qwen Ascend requires the audited Qwen3.5-397B-A17B geometry")
    meta = raw.get("modelslim_metadata", {})
    kinds = text.get("layer_types", [])
    if (
        meta.get("schema") != "aisimulate-qwen35-modelslim-metadata/v1"
        or len(kinds) != 60
        or kinds.count("linear_attention") != 45
        or kinds.count("full_attention") != 15
        or len(meta.get("layers", [])) != 60
        or meta.get("embedding") != "FLOAT"
        or meta.get("lm_head") != "FLOAT"
    ):
        raise InvalidEngineConfigurationError("Incomplete Qwen ModelSlim layer/precision metadata")
    for index, (kind, layer) in enumerate(zip(kinds, meta["layers"], strict=True)):
        attn = (
            dict.fromkeys(("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "conv1d", "norm"), "FLOAT")
            if kind == "linear_attention"
            else dict(
                q_proj="W8A8_DYNAMIC",
                k_proj="W8A8_DYNAMIC",
                v_proj="W8A8_DYNAMIC",
                o_proj="FLOAT",
                q_norm="FLOAT",
                k_norm="FLOAT",
            )
        )
        expected_layer = {
            "layer": index,
            "kind": kind,
            "attention": attn,
            "norms": dict(input_layernorm="FLOAT", post_attention_layernorm="FLOAT"),
            "mlp": {
                "experts": 512,
                "routed": dict.fromkeys(("gate_proj", "up_proj", "down_proj"), "W8A8_DYNAMIC"),
                "router": "FLOAT",
                "shared": dict.fromkeys(("gate_proj", "up_proj", "down_proj"), "FLOAT"),
                "shared_gate": "FLOAT",
            },
        }
        if layer != expected_layer:
            raise InvalidEngineConfigurationError(f"Unqualified Qwen precision partition at layer {index}")
    return kinds


def physical_specs(raw, tp, phase, attention_dp=1, decode_execution_mode="eager"):
    """Return serializable native ops. Prefill token count is before TP scatter.

    The Qwen-specific routing/transport rows use global attention token counts
    and include TP-local balanced placement; shared prefill GEMMs use the local
    scattered count. Graph Decode also scatters the TP-padded attention input
    before MoE. Its measured custom-role axes retain real requests per DP.
    """
    kinds = validate_checkpoint(raw)
    if (
        tp not in (1, 2, 4, 8, 16)
        or attention_dp < 1
        or (tp * attention_dp) not in (2, 4, 8, 16, 32, 64, 128, 256, 512)
        or phase not in ("prefill", "decode")
    ):
        raise InvalidEngineConfigurationError(
            "Qwen Ascend requires integral GDN/GQA head partition and a supported DeepEP group"
        )
    context = phase == "prefill"
    graph = not context and decode_execution_mode == "graph"
    execution = "graph" if graph else "eager"
    contract = (
        "qwen35_397b_modelslim_fp32_state_graph_tp4_dp4_ep16/v1"
        if graph
        else "qwen35_397b_modelslim_bf16_state_eager/v1"
    )
    if graph and (tp, attention_dp) != (4, 4):
        raise InvalidEngineConfigurationError("Qwen Graph calibration currently requires TP4/DP4/EP16")
    ep = tp * attention_dp
    expert_roles = {"routed_experts", "expert_dispatch", "expert_combine"}
    graph_roles = expert_roles | {
        "gdn_conv",
        "gdn_scan",
        "gdn_norm_gate",
        "rms_norm",
        "add_rms_norm",
        "router",
        "shared_swiglu",
        "shared_gate_merge",
    }

    def qwen(name, role, count=1):
        operation = {
            "AscendQwen": {
                "name": phase + "_" + name,
                "scale_factor": float(count),
                "boundary_contract": (
                    contract if not graph or role in graph_roles else "qwen35_397b_modelslim_bf16_state_eager/v1"
                ),
                "key": {
                    "role": role,
                    "tp": (1 if role == "embedding" and attention_dp > 1 else ep if role in expert_roles else tp),
                    "phase": phase,
                },
            }
        }

        if context and role in expert_roles and attention_dp != 1:
            return {"TokenScale": {"op": operation, "numerator": attention_dp, "denominator": 1}}
        return operation

    def linear(name, n, k, count=1, quant="bfloat16", split=1, requests=False):
        return {
            "AscendLinear": {
                "name": phase + "_" + name,
                "scale_factor": float(count),
                "key": {
                    "quant_mode": quant,
                    "n": n,
                    "k": k,
                    "weight_format": 2 if quant == "bfloat16" else 29,
                    "execution_mode": execution,
                },
                "bias": "never",
                "token_axis": "requests" if requests else "tokens",
                "token_split": split,
                "token_multiple": tp if graph and split == 1 else 1,
            }
        }

    def collective(name, count=1, width=4096, operation="all_reduce", split=1, requests=False):
        if tp == 1:
            return None
        return {
            "Hccl": {
                "name": phase + "_" + name,
                "scale_factor": float(count),
                "key": {
                    "dtype": "bfloat16",
                    "operation": operation,
                    "num_ranks": tp,
                    "ranks_per_node": tp,
                    "rank_layout": "contiguous",
                    "execution_mode": execution,
                },
                "elements_per_token": width,
                "token_split": split,
                "token_axis": "requests" if requests else "tokens",
                "dp_padding_mode": None,
            }
        }

    entries = [qwen("embedding", "embedding")]
    if attention_dp == 1:
        entries.append(collective("embedding_sum"))
    # Preserve actual layer order, including the first non-residual norm.
    for index, kind in enumerate(kinds):
        name = f"layer{index}"
        entries.append(qwen(name + "_input_norm", "rms_norm" if index == 0 else "add_rms_norm"))
        if kind == "linear_attention":
            entries += [
                linear(name + "_qkvz", 20480 // tp, 4096),
                linear(name + "_ba", 128 // tp, 4096),
                qwen(name + "_conv", "gdn_conv"),
                *([qwen(name + "_state_io", "gdn_state_io")] if context else []),
                qwen(name + "_scan", "gdn_scan"),
                qwen(name + "_norm_gate", "gdn_norm_gate"),
                linear(name + "_out", 4096, 8192 // tp),
            ]
        else:
            entries += [
                linear(name + "_qkv", 16384 // tp + 512 * max(1, 2 // tp), 4096, quant="w8a8_dynamic"),
                qwen(name + "_prepare", "gqa_prepare"),
                qwen(name + "_cache_write", "gqa_cache_write"),
                qwen(name + "_attention", "gqa_attention"),
                qwen(name + "_gate", "gqa_gate"),
                linear(name + "_out", 4096, 8192 // tp),
            ]
        entries += [
            collective(name + "_attention_sum", operation="reduce_scatter", width=4096 * tp, split=tp)
            if graph
            else collective(name + "_attention_sum"),
            qwen(name + "_ffn_norm", "add_rms_norm"),
            linear(name + "_router_gemm", 512, 4096, split=tp if context or graph else 1),
            qwen(name + "_router", "router"),
            qwen(name + "_dispatch", "expert_dispatch"),
            qwen(name + "_experts", "routed_experts"),
            qwen(name + "_combine", "expert_combine"),
            linear(name + "_shared_gate", 1, 4096, split=tp if context or graph else 1),
            linear(name + "_shared_up", 2048, 4096, split=tp if context or graph else 1),
            qwen(name + "_shared_swiglu", "shared_swiglu"),
            linear(name + "_shared_down", 4096, 1024, split=tp if context or graph else 1),
            qwen(name + "_merge", "shared_gate_merge"),
        ]
        if context or graph:
            entries.append(collective(name + "_ffn_gather", operation="all_gather", split=tp))
    entries += [
        qwen("final_norm", "add_rms_norm"),
        linear("logits", 248320 // tp, 4096, requests=True),
        collective(
            "logits_gather",
            width=248320 if graph else 248320 // tp,
            operation="all_gather",
            requests=True,
            split=tp if graph else 1,
        ),
        qwen("logits_finalize", "logits_finalize"),
    ]
    return [entry for entry in entries if entry is not None]


def configure_ascend_qwen35(model, raw):
    c = model.config
    validate_checkpoint(raw)
    if (
        model._backend_name != "sglang"
        or c.tp_size not in (1, 2, 4, 8, 16)
        or c.attention_dp_size < 1
        or c.moe_ep_size != c.tp_size * c.attention_dp_size
        or c.moe_ep_size not in (2, 4, 8, 16, 32, 64, 128, 256, 512)
        or c.moe_tp_size != 1
        or c.pp_size != 1
        or c.cp_size != 1
        or (c.dcp_size or 1) != 1
        or c.nextn != 0
        or c.enable_eplb
        or c.moe_backend not in (None, common.MoEBackend.deepep_moe)
        or c.attention_backend is not None
        or c.gemm_quant_mode != common.GEMMQuantMode.bfloat16
        or c.moe_quant_mode != common.MoEQuantMode.w8a8_dynamic
        or c.kvcache_quant_mode != common.KVCacheQuantMode.bfloat16
    ):
        raise InvalidEngineConfigurationError(
            "Qwen Ascend DeepEP requires BF16/W8A8_DYNAMIC, local TP1/2/4/8/16, "
            "TP*ADP=EP, MoE-TP1, PP1/CP1/DCP1, no MTP/EPLB"
        )
    if c.op_level_config is None:
        c.op_level_config = OpLevelCompileConfig(None, system=c.system, backend=model._backend_name)
    settings = c.op_level_config.options["ascend"]
    graph = settings["decode_execution_mode"] == "graph"
    if graph and (c.tp_size, c.attention_dp_size, c.moe_ep_size) != (4, 4, 16):
        raise InvalidEngineConfigurationError("Qwen Graph calibration currently requires TP4/DP4/EP16")
    if settings["shared_expert_overlap"] or settings.get("shape_policy", "measured") != "measured":
        raise InvalidEngineConfigurationError(
            "Qwen calibration requires explicit shared_expert_overlap=false and shape_policy=measured"
        )
    model.raw_config = raw
    model._mamba_ssm_dtype = "float32" if graph else "bfloat16"
    model._mtp_scale_factor = 1.0
    model._gpus_per_node = 16
    model.execution_assumptions = {
        "runtime": "q397 FP32-state Graph Decode pilot"
        if graph
        else "q397 historical ModelSlim eager with explicit GDN KV/VK state boundary adapter",
        "mamba_ssm_dtype": model._mamba_ssm_dtype,
        "decode_execution_mode": settings["decode_execution_mode"],
        "prefill_calibration": "historical BF16-state eager; FP32 Prefill qualification pending"
        if graph
        else "historical BF16-state eager",
        "decode_proxy_roles": [
            "embedding",
            "gqa_prepare",
            "gqa_cache_write",
            "gqa_attention",
            "gqa_gate",
            "logits_finalize",
        ]
        if graph
        else [],
        "max_dispatch_tokens_per_rank": 16 if graph else 128,
        "graph_calibrated_requests_per_dp": [1, 2, 3, 4] if graph else None,
        "routing": "balanced",
        "moe_backend": "deepep normal Prefill; low_latency Decode",
        "prefill_batch": 1,
        "batch_scope": "per_attention_dp_replica",
        "attention_dp_load": "uniform",
        "embedding_layout": "replicated" if c.attention_dp_size > 1 else "vocab TP shard",
        "enable_dp_lm_head": c.attention_dp_size > 1,
        "attention_tp": c.tp_size,
        "expert_parallel": c.moe_ep_size,
        "shared_expert_overlap": False,
        "speculative_decoding": False,
        "vision": False,
        "missing_data": "error; no GLM substitution or analytical fallback",
        "memory": "per-sequence GDN state; fixed serving pool reservations require explicit resolved capacity",
    }
    model.context_ops = [
        core.op_from_spec_json(json.dumps(s)) for s in physical_specs(raw, c.tp_size, "prefill", c.attention_dp_size)
    ]
    model.generation_ops = [
        core.op_from_spec_json(json.dumps(s))
        for s in physical_specs(raw, c.tp_size, "decode", c.attention_dp_size, settings["decode_execution_mode"])
    ]
