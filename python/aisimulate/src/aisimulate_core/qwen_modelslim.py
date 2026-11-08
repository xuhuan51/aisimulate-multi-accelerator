"""Read Qwen3.5 ModelSlim precision without loading tensors or predicting time.

The metadata describes the text decoder only. A successful read does not qualify
an Ascend execution graph, physical memory capacity or operator timing coverage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def read_qwen35_modelslim(model_path: str, config: dict) -> dict | None:
    path = Path(model_path) / "quant_model_description.json"
    if not path.is_file():
        return None
    if config.get("architectures") != ["Qwen3_5MoeForConditionalGeneration"]:
        raise ValueError("Qwen ModelSlim metadata requires Qwen3_5MoeForConditionalGeneration")
    payload = path.read_bytes()
    description = json.loads(payload)
    if not isinstance(description, dict) or description.get("model_quant_type") != "W8A8_DYNAMIC":
        raise ValueError("Qwen ModelSlim metadata requires W8A8_DYNAMIC")
    text = config.get("text_config", {})
    layers = text.get("layer_types", [])
    experts = text.get("num_experts", 0)
    if (
        not isinstance(layers, list)
        or not layers
        or len(layers) != text.get("num_hidden_layers")
        or any(kind not in ("linear_attention", "full_attention") for kind in layers)
        or type(experts) is not int
        or experts < 1
        or text.get("mlp_only_layers")
        or not text.get("shared_expert_intermediate_size", 0)
    ):
        raise ValueError("Unsupported Qwen ModelSlim layer/expert structure")
    dtype = text.get("dtype", text.get("torch_dtype", config.get("torch_dtype")))
    if dtype != "bfloat16":
        raise ValueError("Qwen ModelSlim FLOAT precision requires an explicit bfloat16 checkpoint dtype")
    if description.get("is_rot_used", False) not in (False, 0, "0"):
        raise ValueError("Qwen ModelSlim rotation is not represented")

    def require(name, expected):
        actual = description.get(name)
        if actual != expected:
            raise ValueError(f"Unsupported Qwen ModelSlim precision: {name}={actual!r}; expected {expected}")
        return actual

    plan = []
    for index, kind in enumerate(layers):
        prefix = f"model.language_model.layers.{index}."
        if kind == "linear_attention":
            attention_prefix = "linear_attn."
            projections = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "conv1d", "norm")
        else:
            attention_prefix = "self_attn."
            projections = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
        attention = {
            name: require(
                prefix + attention_prefix + name + ".weight",
                "W8A8_DYNAMIC" if kind == "full_attention" and name in ("q_proj", "k_proj", "v_proj") else "FLOAT",
            )
            for name in projections
        }
        norms = {
            name: require(prefix + name + ".weight", "FLOAT")
            for name in ("input_layernorm", "post_attention_layernorm")
        }
        # Validate every expert, including the final one; representative metadata
        # is only emitted after the whole layer is known to be homogeneous.
        for expert in range(experts):
            for name in ("gate_proj", "up_proj", "down_proj"):
                require(prefix + f"mlp.experts.{expert}.{name}.weight", "W8A8_DYNAMIC")
        shared = {
            name: require(prefix + "mlp.shared_expert." + name + ".weight", "FLOAT")
            for name in ("gate_proj", "up_proj", "down_proj")
        }
        plan.append(
            {
                "layer": index,
                "kind": kind,
                "attention": attention,
                "norms": norms,
                "mlp": {
                    "experts": experts,
                    "routed": dict.fromkeys(("gate_proj", "up_proj", "down_proj"), "W8A8_DYNAMIC"),
                    "router": require(prefix + "mlp.gate.weight", "FLOAT"),
                    "shared": shared,
                    "shared_gate": require(prefix + "mlp.shared_expert_gate.weight", "FLOAT"),
                },
            }
        )
    return {
        "schema": "aisimulate-qwen35-modelslim-metadata/v1",
        "source": str(path),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "declared_model_quant_type": description["model_quant_type"],
        "primary_gemm_mode": "bfloat16",
        "full_attention_qkv_mode": "w8a8_dynamic",
        "precision_policy": "per_operation; primary_gemm_mode is only the unquantized default",
        "moe_mode": "w8a8_dynamic",
        "embedding": require("model.language_model.embed_tokens.weight", "FLOAT"),
        "lm_head": require("lm_head.weight", "FLOAT"),
        "checkpoint_float_dtype": dtype,
        "declared_mamba_ssm_dtype": text.get("mamba_ssm_dtype", config.get("mamba_ssm_dtype", "float32")),
        "floating_dtype_scope": (
            "Checkpoint dtype only; serving overrides and tensor storage need independent validation."
        ),
        "execution_status": "ascend_operator_graph_pending",
        "scope": "text_decoder_only; vision and MTP are not qualified",
        "layers": plan,
    }
