"""Compact checkpoint precision metadata, without loading weights or estimating time."""

import hashlib
import json
from pathlib import Path


def read_glm_modelslim(model_path: str, config: dict) -> dict | None:
    path = Path(model_path) / "quant_model_description.json"
    if not path.is_file():
        return None
    if config.get("architectures") != ["GlmMoeDsaForCausalLM"]:
        raise ValueError("ModelSlim structural metadata currently supports GlmMoeDsaForCausalLM")
    payload = path.read_bytes()
    description = json.loads(payload)
    if not isinstance(description, dict) or description.get("model_quant_type") not in {"W8A8", "W8A8_DYNAMIC"}:
        raise ValueError("Unsupported ModelSlim checkpoint format")
    layers = int(config["num_hidden_layers"])
    dense_layers = int(config["first_k_dense_replace"])
    experts = int(config["n_routed_experts"])
    if layers < 1 or not 0 <= dense_layers <= layers or experts < 1 or config.get("moe_layer_freq", 1) != 1:
        raise ValueError("Unsupported ModelSlim layer/expert structure")

    def precision(name):
        value = description.get(name)
        if value not in {"FLOAT", "W8A8", "W8A8_DYNAMIC"}:
            raise ValueError(f"Missing or unsupported ModelSlim declaration: {name}={value!r}")
        return value

    def projections(prefix):
        values = {name: precision(prefix + name + ".weight") for name in ("gate_proj", "up_proj", "down_proj")}
        if values["gate_proj"] != values["up_proj"]:
            raise ValueError(f"Fused gate/up projections have different ModelSlim modes: {prefix}")
        return values

    plan = []
    attention_modes = set()
    routed_modes = set()
    for layer in range(layers):
        prefix = f"model.layers.{layer}."
        attention = {
            name: precision(prefix + "self_attn." + name + ".weight")
            for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")
        }
        if attention["q_a_proj"] != attention["kv_a_proj_with_mqa"]:
            raise ValueError(f"Fused Q/KV-A projections have different ModelSlim modes: {prefix}")
        attention_modes.update(attention[name] for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"))
        indexer = None
        if prefix + "self_attn.indexer.wq_b.weight" in description:
            indexer = {
                name: precision(prefix + "self_attn.indexer." + name + ".weight")
                for name in ("wq_b", "wk", "weights_proj")
            }
        if layer < dense_layers:
            mlp = {"kind": "dense", "projections": projections(prefix + "mlp.")}
        else:
            representative = None
            for expert in range(experts):
                current = projections(prefix + f"mlp.experts.{expert}.")
                if representative is not None and representative != current:
                    raise ValueError(f"Per-expert mixed ModelSlim quantization is not represented: {prefix}")
                representative = current
                routed_modes.update(current.values())
            mlp = {
                "kind": "moe",
                "experts": experts,
                "routed": representative,
                "router": precision(prefix + "mlp.gate.weight"),
                "shared": projections(prefix + "mlp.shared_experts.") if config.get("n_shared_experts", 0) else None,
            }
        plan.append({"layer": layer, "attention": attention, "indexer": indexer, "mlp": mlp})
    if attention_modes == {"W8A8"}:
        primary_gemm = "w8a8"
    elif attention_modes == {"W8A8_DYNAMIC"}:
        primary_gemm = "w8a8_dynamic"
    else:
        raise ValueError(f"A single primary attention quant mode cannot represent {sorted(attention_modes)}")
    if routed_modes and routed_modes != {"W8A8_DYNAMIC"}:
        raise ValueError("Current ModelSlim MoE identity requires dynamic INT8 routed projections")
    return {
        "schema": "aisimulate-modelslim-metadata/v1",
        "source": str(path),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "declared_model_quant_type": description["model_quant_type"],
        "primary_gemm_mode": primary_gemm,
        "moe_mode": "w8a8_dynamic" if routed_modes else None,
        "embedding": precision("model.embed_tokens.weight"),
        "lm_head": precision("lm_head.weight"),
        "rotation_declared": bool(description.get("is_rot_used", False)),
        "rotation_metadata": description.get("optional", {}).get("quarot"),
        "floating_dtype_scope": "FLOAT means unquantized; storage/runtime dtype must be resolved separately.",
        "layers": plan,
    }


def read_modelslim(model_path: str, config: dict) -> dict | None:
    """Dispatch structural metadata; estimator support is validated separately."""
    if config.get("architectures") == ["Qwen3_5MoeForConditionalGeneration"]:
        from aisimulate_core.qwen_modelslim import read_qwen35_modelslim

        return read_qwen35_modelslim(model_path, config)
    return read_glm_modelslim(model_path, config)
