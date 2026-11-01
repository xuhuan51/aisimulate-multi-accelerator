"""Expand declared Ascend linear shapes for a verified local ModelSlim artifact.

Geometry follows SGLang baa9befa381511cd4d56f83980951e5a45fffc6d:
models/deepseek_v2.py:2020 attention shards, :800 shared expert TP1;
hardware_backend/npu/quantization/linear_method_npu.py:64-156 distinguishes
static quant-bias on RowParallel rank0 from dynamic quantization.
The checkpoint's own config and sidecar must agree with the declared profile.
This explicit Ascend plan does not add NPU work to CUDA model plans.
"""

import hashlib
import json
from pathlib import Path

import yaml

PHYSICAL_FIELDS = ("quant_mode", "m", "n", "k", "quant_bias", "execution_mode")


def get_ascend_linear_cases(model_path, case_root=None):
    model_path = Path(model_path)
    case_root = Path(case_root) if case_root else Path(__file__).resolve().parents[1] / "cases"
    config_path = model_path / "config.json"
    quant_path = model_path / "quant_model_description.json"
    config = json.loads(config_path.read_text())
    description = json.loads(quant_path.read_text())
    architecture = config["architectures"][0]
    model_file = case_root / "models" / f"{architecture}_cases.yaml"
    model_cases = yaml.safe_load(model_file.read_text())
    profiles = model_cases["model_case_values"]["ascend_linear"]
    matching = [profile for profile in profiles if profile["checkpoint_quantization"] == "modelslim_w8a8"]
    if len(matching) != 1:
        raise ValueError("Need one declared Ascend ModelSlim geometry for this architecture")
    profile = matching[0]
    geometry_fields = (
        "hidden_size",
        "num_attention_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "index_n_heads",
        "index_head_dim",
        "intermediate_size",
        "moe_intermediate_size",
        "n_routed_experts",
        "n_shared_experts",
        "vocab_size",
    )
    for field in geometry_fields:
        if type(config.get(field)) is not int or config[field] != profile[field]:
            raise ValueError(f"Checkpoint does not match declared {field}")
    dense_layers = config["first_k_dense_replace"]
    if type(dense_layers) is not int or not 0 < dense_layers < config["num_hidden_layers"]:
        raise ValueError("This plan requires dense-prefix and MoE layers")
    expected = {
        "model.layers.0.self_attn.q_a_proj.weight": "W8A8",
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": "W8A8",
        "model.layers.0.self_attn.q_b_proj.weight": "W8A8",
        "model.layers.0.self_attn.o_proj.weight": "W8A8",
        "model.layers.0.self_attn.indexer.wq_b.weight": "W8A8",
        "model.layers.0.self_attn.indexer.wk.weight": "FLOAT",
        "model.layers.0.self_attn.indexer.weights_proj.weight": "FLOAT",
        "lm_head.weight": "FLOAT",
    }
    for projection in ("gate_proj", "up_proj", "down_proj"):
        expected[f"model.layers.0.mlp.{projection}.weight"] = "W8A8_DYNAMIC"
        expected[f"model.layers.{dense_layers}.mlp.shared_experts.{projection}.weight"] = "W8A8_DYNAMIC"
    expected[f"model.layers.{dense_layers}.mlp.gate.weight"] = "FLOAT"
    for name, quantization in expected.items():
        if description.get(name) != quantization:
            raise ValueError(f"Unqualified ModelSlim precision for {name}: {description.get(name)}")
    base_file = case_root / "base_ops/ascend_linear.yaml"
    base = yaml.safe_load(base_file.read_text())["common_case_values"]["ascend_linear"]
    matrices = []

    def add(owner, quant_mode, output_size, input_size, quant_bias=False, requests=False):
        matrices.append(
            {
                "owner": owner,
                "quant_mode": quant_mode,
                "n": output_size,
                "k": input_size,
                "quant_bias": quant_bias,
                "requests": requests,
            }
        )

    hidden = config["hidden_size"]
    query_rank = config["q_lora_rank"]
    expert_intermediate = config["moe_intermediate_size"]
    for parallel in profile["tensor_parallel_sizes"]:
        if config["num_attention_heads"] % parallel or config["vocab_size"] % parallel:
            raise ValueError("Declared attention TP does not divide the model")
        heads = config["num_attention_heads"] // parallel
        add(
            f"qkv_a.tp{parallel}",
            "w8a8",
            query_rank + config["kv_lora_rank"] + config["qk_rope_head_dim"],
            hidden,
            True,
        )
        add(
            f"q_b.tp{parallel}",
            "w8a8",
            heads * (config["qk_nope_head_dim"] + config["qk_rope_head_dim"]),
            query_rank,
            True,
        )
        for bias in (False, True):
            add(f"o.tp{parallel}.bias{int(bias)}", "w8a8", hidden, heads * config["v_head_dim"], bias)
        add(f"lm_head.tp{parallel}", "bfloat16", config["vocab_size"] // parallel, hidden, requests=True)
    for parallel in profile["global_tensor_parallel_sizes"]:
        add(f"lm_head_global.tp{parallel}", "bfloat16", config["vocab_size"] // parallel, hidden, requests=True)
        if config["intermediate_size"] % parallel:
            raise ValueError("Declared dense TP does not divide intermediate size")
        intermediate = config["intermediate_size"] // parallel
        add(f"dense_up.tp{parallel}", "w8a8_dynamic", 2 * intermediate, hidden)
        add(f"dense_down.tp{parallel}", "w8a8_dynamic", hidden, intermediate)
    add("shared_up", "w8a8_dynamic", 2 * expert_intermediate, hidden)
    add("shared_down", "w8a8_dynamic", hidden, expert_intermediate)
    for parallel in profile["shared_tensor_parallel_sizes"]:
        if expert_intermediate % parallel:
            raise ValueError("Declared shared expert TP does not divide intermediate size")
        intermediate = expert_intermediate // parallel
        add(f"shared_tensor_up.tp{parallel}", "w8a8_dynamic", 2 * intermediate, hidden)
        add(f"shared_tensor_down.tp{parallel}", "w8a8_dynamic", hidden, intermediate)
    add("indexer_q", "w8a8", config["index_n_heads"] * config["index_head_dim"], query_rank, True)
    add("indexer_k", "bfloat16", config["index_head_dim"], hidden)
    add("indexer_weights", "float32_bfloat16", config["index_n_heads"], hidden)
    add("router", "bfloat16", config["n_routed_experts"], hidden)
    cases = {}
    requested = 0
    for matrix in matrices:
        counts = base["request_counts"] if matrix["requests"] else base["token_counts"]
        for count in counts:
            for mode in base["execution_modes"]:
                requested += 1
                case = {field: matrix[field] for field in ("quant_mode", "n", "k", "quant_bias")}
                case.update(m=count, execution_mode=mode)
                key = tuple(case[field] for field in PHYSICAL_FIELDS)
                stored = cases.setdefault(key, {**case, "owners": []})
                stored["owners"].append(matrix["owner"])
    return {
        "schema": "aisimulate-ascend-linear-plan/v1",
        "status": "experimental_collection_plan",
        "model_path": str(model_path),
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (config_path, quant_path, base_file, model_file)
        },
        "recipe_invocations": requested,
        "unique_invocations": len(cases),
        "protocol": {key: base[key] for key in ("weight_copies", "iterations", "warmup", "repeats")},
        "cases": list(cases.values()),
    }
