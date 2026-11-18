"""Qwen397B physical composition for audited Ascend ModelSlim execution paths.

Graph construction only: native operators own timing and weight arithmetic.
Boundary contracts distinguish historical Eager data and local Graph geometries.
"""

import json
import math

import aisimulate_core._native as core
from aisimulate_core.sdk import common
from aisimulate_core.sdk.errors import InvalidEngineConfigurationError
from aisimulate_core.sdk.fpm_config import OpLevelCompileConfig


def _graph_contract(tp, attention_dp, execution_mode="graph"):
    if (tp, attention_dp) not in ((2, 8), (4, 4), (8, 2)):
        raise InvalidEngineConfigurationError(
            f"Qwen {execution_mode.title()} requires TP2/DP8, TP4/DP4 or TP8/DP2 with EP16"
        )
    return f"qwen35_397b_modelslim_fp32_state_{execution_mode}_tp{tp}_dp{attention_dp}_ep16/v1"


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


def physical_specs(
    raw,
    tp,
    phase,
    attention_dp=1,
    decode_execution_mode="eager",
    qwen_graph_small_ops=False,
    qwen_runtime_contract=None,
    decode_expert_workload=None,
    prefill_expert_workload=None,
    qwen_prefill_current_roles=(),
    qwen_eager_ep_pair=False,
    qwen_eager_host_cpu_bound=False,
    qwen_eager_host_cpu_transfer=False,
):
    """Return serializable native ops. Prefill token count is before TP scatter.

    The Qwen-specific routing/transport rows use global attention token counts
    and include TP-local balanced placement; shared prefill GEMMs use the local
    scattered count. Graph and the explicit FP32 runtime contract scatter the
    TP-padded Decode attention input before MoE. Custom-role axes retain real
    requests per DP, while expert transport rows use sent tokens per rank.
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
    if qwen_runtime_contract not in (None, "historical", "modelslim_fp32_tp_scatter"):
        raise InvalidEngineConfigurationError("Unknown Qwen runtime contract")
    current_runtime = qwen_runtime_contract == "modelslim_fp32_tp_scatter"
    if qwen_eager_host_cpu_transfer and not qwen_eager_host_cpu_bound:
        raise InvalidEngineConfigurationError("Host CPU transfer requires an explicitly selected CPU bound")
    if qwen_eager_host_cpu_bound and (
        context
        or graph
        or not current_runtime
        or (tp, attention_dp) not in ((2, 8), (4, 4), (8, 2))
        or (tp != 4 and not qwen_eager_host_cpu_transfer)
    ):
        raise InvalidEngineConfigurationError(
            "Measured host CPU bound requires TP4/DP4 FP32 eager Decode or explicit TP4 reference transfer"
        )
    if qwen_eager_ep_pair and (not current_runtime or graph):
        raise InvalidEngineConfigurationError("Paired eager EP timing requires the FP32 TP-scatter eager runtime")
    paired_transport = qwen_eager_ep_pair and not context
    if decode_expert_workload is not None and (not graph or (tp, attention_dp) != (4, 4)):
        raise InvalidEngineConfigurationError("Observed Qwen expert curves require TP4/DP4 Graph Decode")
    scatter_decode = not context and (graph or current_runtime)
    execution = "graph" if graph else "eager"
    eager_contract = (
        _graph_contract(tp, attention_dp, "eager") if current_runtime else "qwen35_397b_modelslim_bf16_state_eager/v1"
    )
    contract = _graph_contract(tp, attention_dp) if graph else eager_contract
    prefill_component_contract = (
        _graph_contract(tp, attention_dp, "eager") if context and qwen_prefill_current_roles else None
    )
    ep = tp * attention_dp
    if prefill_expert_workload is not None and (
        not context
        or ep != 16
        or (tp, attention_dp) not in ((2, 8), (4, 4), (8, 2))
        or len(prefill_expert_workload["layer_expert_counts"]) != 60
        or any(len(counts) != 512 for counts in prefill_expert_workload["layer_expert_counts"])
    ):
        raise InvalidEngineConfigurationError(
            "Observed Qwen Prefill workload requires60layers,512experts and TP2/DP8, TP4/DP4 or TP8/DP2 with EP16"
        )
    if context and "normal_transport" in qwen_prefill_current_roles and prefill_expert_workload is None:
        raise InvalidEngineConfigurationError("Qwen Normal transport requires its complete Prefill expert workload")
    expert_roles = {"routed_experts", "expert_dispatch", "expert_combine", "expert_transport_pair"}
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
    if qwen_graph_small_ops:
        if not graph:
            raise InvalidEngineConfigurationError("qwen_graph_small_ops requires Graph Decode")
        graph_roles |= {"embedding", "gqa_prepare", "gqa_cache_write", "gqa_attention", "gqa_gate", "logits_finalize"}

    def qwen(name, role, count=1, expert_layer=None):
        if role == "routed_experts" and decode_expert_workload is not None:
            return {
                "AscendMoeChain": {
                    "name": phase + "_" + name,
                    "scale_factor": float(count),
                    "key": {
                        "phase": "decode",
                        "local_experts": 32,
                        "distribution": "balanced",
                        "execution_mode": "graph",
                    },
                    "token_numerator": 10 * attention_dp,
                    "token_denominator": ep,
                    "input_token_multiple": tp,
                    "decode_workload": decode_expert_workload,
                    "profile_request_counts": None,
                }
            }
        boundary_contract = contract if not graph or role in graph_roles else eager_contract
        if context and qwen_prefill_current_roles:
            boundary_contract = (
                prefill_component_contract
                if role in qwen_prefill_current_roles
                else "qwen35_397b_modelslim_bf16_state_eager/v1"
            )
            if role == "routed_experts" and role in qwen_prefill_current_roles:
                boundary_contract = "qwen35_397b_modelslim_prefill_int8_expert_chain/v1"
        operation = {
            "AscendQwen": {
                "name": phase + "_" + name,
                "scale_factor": float(count),
                "boundary_contract": boundary_contract,
                "key": {
                    "role": role,
                    "tp": (1 if role == "embedding" and attention_dp > 1 else ep if role in expert_roles else tp),
                    "phase": phase,
                },
            }
        }

        if (
            context
            and "normal_transport" in qwen_prefill_current_roles
            and role in {"expert_dispatch", "expert_combine"}
        ):
            operation = {
                "AscendNormalTransport": {
                    "name": phase + "_" + name,
                    "scale_factor": float(count),
                    "key": {
                        "role": "normal_dispatch_ready" if role == "expert_dispatch" else "normal_combine",
                        "tp": ep,
                        "phase": phase,
                    },
                    "reference_tokens": prefill_expert_workload["input_tokens"],
                    "expert_counts": prefill_expert_workload["layer_expert_counts"][expert_layer],
                }
            }

        if context and role == "routed_experts" and prefill_expert_workload is not None:
            counts = prefill_expert_workload["layer_expert_counts"][expert_layer]
            local_experts = 512 // ep
            received = max(sum(counts[start : start + local_experts]) for start in range(0, 512, local_experts))
            # The historical table's global-token axis represents
            # ceil(global_tokens / EP) * top10 rank-local assignments.
            numerator = received * ep * attention_dp
            denominator = prefill_expert_workload["input_tokens"] * 10
            divisor = math.gcd(numerator, denominator)
            numerator //= divisor
            denominator //= divisor
            if max(numerator, denominator) >= 2**32:
                raise InvalidEngineConfigurationError("Observed Qwen expert count ratio exceeds native shape limits")
            return {"TokenScale": {"op": operation, "numerator": numerator, "denominator": denominator}}
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
                "token_multiple": tp if scatter_decode and split == 1 else 1,
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
            if scatter_decode or (context and current_runtime)
            else collective(name + "_attention_sum"),
            qwen(name + "_ffn_norm", "add_rms_norm"),
            linear(name + "_router_gemm", 512, 4096, split=tp if context or scatter_decode else 1),
            qwen(name + "_router", "router"),
            *([] if paired_transport else [qwen(name + "_dispatch", "expert_dispatch", expert_layer=index)]),
            qwen(name + "_experts", "routed_experts", expert_layer=index),
            *(
                [qwen(name + "_transport_pair", "expert_transport_pair")]
                if paired_transport
                else [qwen(name + "_combine", "expert_combine", expert_layer=index)]
            ),
            linear(name + "_shared_gate", 1, 4096, split=tp if context or scatter_decode else 1),
            linear(name + "_shared_up", 2048, 4096, split=tp if context or scatter_decode else 1),
            qwen(name + "_shared_swiglu", "shared_swiglu"),
            linear(name + "_shared_down", 4096, 1024, split=tp if context or scatter_decode else 1),
            qwen(name + "_merge", "shared_gate_merge"),
        ]
        if context or scatter_decode:
            entries.append(collective(name + "_ffn_gather", operation="all_gather", split=tp))
    entries += [
        qwen("final_norm", "add_rms_norm"),
        linear("logits", 248320 // tp, 4096, requests=True),
        collective(
            "logits_gather",
            width=248320 if scatter_decode else 248320 // tp,
            operation="all_gather",
            requests=True,
            split=tp if scatter_decode else 1,
        ),
        qwen("logits_finalize", "logits_finalize"),
    ]
    entries = [entry for entry in entries if entry is not None]
    if qwen_eager_host_cpu_bound:
        # Exclusive CPU intervals are measured independently of device timing.
        # Decoder counts follow the checkpoint's actual layer architecture.
        components = {
            "linear_attention": kinds.count("linear_attention"),
            "linear_layer_host": kinds.count("linear_attention"),
            "full_attention_layer_host": kinds.count("full_attention"),
            "moe": len(kinds),
            "layer_norms": len(kinds),
            "outer_forward": 1,
        }
        rank_paths = []
        for rank in range(16):
            path = [
                {
                    "AscendQwen": {
                        "name": f"cpu_{component}_rank{rank}",
                        "scale_factor": float(count),
                        "key": {"role": f"host_cpu_{component}_rank{rank}", "tp": 4, "phase": phase},
                        "boundary_contract": "qwen35_a6b5428_node20_fp32_eager_tp4_dp4_ep16_exclusive_cpu_mean/v1",
                    }
                }
                for component, count in components.items()
            ]
            rank_paths = (
                [
                    {
                        "Overlap": {
                            "name": f"cpu_critical_rank{rank}",
                            "group_a": rank_paths,
                            "group_b": path,
                        }
                    }
                ]
                if rank_paths
                else path
            )
        return [
            {
                "Overlap": {
                    "name": "eager_device_and_host_cpu_bound",
                    "group_a": entries,
                    "group_b": rank_paths,
                }
            }
        ]
    return entries


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
    runtime_contract = settings["qwen_runtime_contract"]
    current_runtime = runtime_contract == "modelslim_fp32_tp_scatter"
    if (
        settings["qwen_eager_host_cpu_bound"]
        and not settings["qwen_eager_host_cpu_transfer"]
        and (c.tp_size, c.attention_dp_size, c.moe_ep_size) != (4, 4, 16)
    ):
        raise InvalidEngineConfigurationError("Measured host CPU bound requires TP4/DP4/EP16")
    decode_workload = settings.get("decode_expert_workload")
    prefill_workload = settings.get("prefill_expert_workload")
    if prefill_workload is not None and (
        (c.tp_size, c.attention_dp_size, c.moe_ep_size) not in ((2, 8, 16), (4, 4, 16), (8, 2, 16))
        or prefill_workload["quantization_sha256"] != raw["modelslim_metadata"].get("source_sha256")
        or len(prefill_workload["layer_expert_counts"]) != 60
        or any(len(counts) != 512 for counts in prefill_workload["layer_expert_counts"])
    ):
        raise InvalidEngineConfigurationError(
            "Observed Qwen Prefill workload requires its matching ModelSlim descriptor, "
            "60-layer512-expert geometry and TP2/DP8, TP4/DP4 or TP8/DP2 with EP16"
        )
    if decode_workload is not None and (
        not graph
        or (c.tp_size, c.attention_dp_size, c.moe_ep_size) != (4, 4, 16)
        or decode_workload["graph_bucket"] != 4
        or decode_workload["quantization_sha256"] != raw["modelslim_metadata"].get("source_sha256")
    ):
        raise InvalidEngineConfigurationError(
            "Observed Qwen expert workload requires its matching ModelSlim descriptor, TP4/DP4/EP16 Graph Decode"
        )
    if graph:
        _graph_contract(c.tp_size, c.attention_dp_size)
    if current_runtime:
        _graph_contract(c.tp_size, c.attention_dp_size, "eager")
    if settings["shared_expert_overlap"] or settings.get("shape_policy", "measured") != "measured":
        raise InvalidEngineConfigurationError(
            "Qwen calibration requires explicit shared_expert_overlap=false and shape_policy=measured"
        )
    model.raw_config = raw
    model._mamba_ssm_dtype = "float32" if graph or current_runtime else "bfloat16"
    model._mtp_scale_factor = 1.0
    model._gpus_per_node = 16
    model.execution_assumptions = {
        "runtime": "q397 FP32-state Graph Decode pilot"
        if graph
        else "q397 historical ModelSlim eager with explicit GDN KV/VK state boundary adapter",
        "qwen_runtime_contract": runtime_contract,
        "mamba_ssm_dtype": model._mamba_ssm_dtype,
        "decode_execution_mode": settings["decode_execution_mode"],
        "qwen_graph_small_ops": settings["qwen_graph_small_ops"],
        "qwen_eager_ep_pair": settings["qwen_eager_ep_pair"],
        "qwen_eager_host_cpu_bound": settings["qwen_eager_host_cpu_bound"],
        "qwen_eager_host_cpu_transfer": settings["qwen_eager_host_cpu_transfer"],
        "host_cpu_profile_source": "Node20 TP4/DP4/EP16 source-defined CPU component means"
        if settings["qwen_eager_host_cpu_bound"]
        else None,
        "host_cpu_cross_topology_assumption": (
            "CPU API costs retained from TP4. Device paths and physical DP ownership use target topology"
        )
        if settings["qwen_eager_host_cpu_bound"] and c.tp_size != 4
        else None,
        "host_cpu_bound_scope": (
            "Node20 pinned runtime, logical requests1..3 per DP, exclusive CPU component means, "
            "max(host, device) lower bound"
        )
        if settings["qwen_eager_host_cpu_bound"]
        else None,
        "eager_ep_timing_boundary": "one unprofiled dispatch+combine pair per serial MoE layer"
        if settings["qwen_eager_ep_pair"]
        else "separate dispatch and combine measurements",
        "qwen_prefill_current_roles": settings["qwen_prefill_current_roles"],
        "prefill_transport_workload": (
            "SUM_LEN global input and expected48-AIV critical receive-core work from the retained expert distribution"
        )
        if "normal_transport" in settings["qwen_prefill_current_roles"]
        else "balanced global-token transport curves",
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
        if graph and not settings["qwen_graph_small_ops"]
        else [],
        "max_dispatch_tokens_per_rank": 16 if graph or current_runtime else 128,
        "graph_calibrated_requests_per_dp": [1, 2, 3, 4] if graph and c.tp_size == 4 else None,
        "graph_topology_validation": "TP4 pilot" if graph and c.tp_size == 4 else "pending" if graph else None,
        "routing": "balanced",
        "prefill_expert_workload": None
        if prefill_workload is None
        else {key: prefill_workload[key] for key in ("observation_id", "quantization_sha256", "input_tokens")},
        "prefill_expert_cost_boundary": (
            "current INT8-input BF16-output expert chain; Normal BF16 input quantization remains unpriced"
            if "routed_experts" in settings["qwen_prefill_current_roles"]
            else "historical INT8 input and scales, Normal BF16 input quantization remains unpriced"
        ),
        "decode_expert_workload": None
        if decode_workload is None
        else {
            key: decode_workload[key]
            for key in (
                "observation_id",
                "samples_sha256",
                "framework_commit",
                "history_policy",
                "input_tokens",
                "graph_bucket",
            )
        },
        "decode_workload_unmatched_shapes": "balanced receive-volume curve",
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
    if current_runtime:
        model.execution_assumptions["runtime"] = "ModelSlim FP32 state with TP-scattered MoE"
        model.execution_assumptions["prefill_calibration"] = (
            "requires current recurrent/state-boundary costs; identical BF16 Prefill "
            "embedding, Gemma norm and shared SwiGLU signatures may reuse historical eager measurements"
        )
    if settings["qwen_prefill_current_roles"]:
        model.execution_assumptions["prefill_calibration"] = (
            "selected current Prefill components with historical remaining boundaries; "
            "full current-runtime cost coverage remains separately qualified"
        )
    model.context_ops = [
        core.op_from_spec_json(json.dumps(s))
        for s in physical_specs(
            raw,
            c.tp_size,
            "prefill",
            c.attention_dp_size,
            qwen_runtime_contract=runtime_contract,
            prefill_expert_workload=prefill_workload,
            qwen_prefill_current_roles=settings["qwen_prefill_current_roles"],
        )
    ]
    model.generation_ops = [
        core.op_from_spec_json(json.dumps(s))
        for s in physical_specs(
            raw,
            c.tp_size,
            "decode",
            c.attention_dp_size,
            settings["decode_execution_mode"],
            settings["qwen_graph_small_ops"],
            runtime_contract,
            decode_workload,
            qwen_eager_ep_pair=settings["qwen_eager_ep_pair"],
            qwen_eager_host_cpu_bound=settings["qwen_eager_host_cpu_bound"],
            qwen_eager_host_cpu_transfer=settings["qwen_eager_host_cpu_transfer"],
        )
    ]
