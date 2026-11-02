"""Declare vector identities from the pinned runtime and current model geometry."""

import hashlib
import json
from pathlib import Path

import yaml


def get_ascend_vector_cases(model_path, case_root=None):
    model_path = Path(model_path)
    case_root = Path(case_root) if case_root else Path(__file__).resolve().parents[1] / "cases"
    config_file = model_path / "config.json"
    config = json.loads(config_file.read_text())
    quant_file = model_path / "quant_model_description.json"
    quantization = json.loads(quant_file.read_text())
    if quantization.get("model.layers.0.self_attn.q_a_proj.weight") != "W8A8":
        raise ValueError("Expected the source-audited ModelSlim W8A8 artifact")
    model_file = case_root / "models" / f"{config['architectures'][0]}_cases.yaml"
    profiles = yaml.safe_load(model_file.read_text())["model_case_values"]["ascend_vector"]
    selected = [value for value in profiles if value["checkpoint_quantization"] == "modelslim_w8a8"]
    if len(selected) != 1:
        raise ValueError("Expected one declared ModelSlim vector geometry")
    geometry = selected[0]
    for name, value in geometry.items():
        if (
            name
            not in {
                "checkpoint_quantization",
                "tensor_parallel_sizes",
                "global_tensor_parallel_sizes",
                "shared_tensor_parallel_sizes",
                "indexer_norm_eps",
            }
            and config.get(name) != value
        ):
            raise ValueError(f"Model differs from declared {name}")
    base_file = case_root / "base_ops/ascend_vector.yaml"
    base = yaml.safe_load(base_file.read_text())["common_case_values"]["ascend_vector"]
    hidden, rope = geometry["hidden_size"], geometry["qk_rope_head_dim"]
    sites = [
        ("rms_bias", [hidden]),
        ("add_rms_bias", [hidden]),
        ("split_qk_norm", [geometry["q_lora_rank"], geometry["kv_lora_rank"], rope]),
        ("indexer_norm", [geometry["index_head_dim"]]),
        ("rope_indexer", [geometry["index_n_heads"], rope, geometry["index_head_dim"], geometry["index_head_dim"]]),
        ("concat_indexer_q", [geometry["index_n_heads"], rope, geometry["index_head_dim"]]),
        ("concat_indexer_k", [1, rope, geometry["index_head_dim"]]),
        ("swiglu", [geometry["moe_intermediate_size"]]),
        ("router_topk", [geometry["n_routed_experts"], geometry["num_experts_per_tok"]]),
        ("cast_hidden_fp32", [hidden]),
        ("cast_weights_bf16", [geometry["index_n_heads"]]),
        ("routed_scale", [hidden]),
        ("shared_add", [hidden]),
        ("rope_cache_gather", [geometry["max_position_embeddings"], rope]),
    ]
    for parallel in geometry["tensor_parallel_sizes"]:
        if geometry["num_attention_heads"] % parallel or geometry["vocab_size"] % parallel:
            raise ValueError("Declared attention/vocabulary shards are not integral")
        sites.append(
            (
                "rope_attention",
                [geometry["num_attention_heads"] // parallel, rope, geometry["qk_nope_head_dim"] + rope, rope],
            )
        )
        sites.append(("embedding", [geometry["vocab_size"] // parallel, hidden]))
        sites.append(("logits_finalize", [parallel, geometry["vocab_size"]]))
    for parallel in geometry["global_tensor_parallel_sizes"]:
        if geometry["intermediate_size"] % parallel:
            raise ValueError("Dense MLP shard is not integral")
        sites.append(("swiglu", [geometry["intermediate_size"] // parallel]))
    for parallel in geometry["shared_tensor_parallel_sizes"]:
        if geometry["moe_intermediate_size"] % parallel:
            raise ValueError("Shared expert shard is not integral")
        sites.append(("swiglu", [geometry["moe_intermediate_size"] // parallel]))
    for width in (geometry["kv_lora_rank"], rope, geometry["index_head_dim"]):
        sites.append(("cache_scatter", [base["reference_cache_tokens"], width]))
    cases = [
        {"role": role, "dimensions": dimensions, "tokens": tokens, "execution_mode": mode}
        for role, dimensions in sites
        for mode in base["execution_modes"]
        for tokens in (base["request_counts"] if role == "logits_finalize" else base["token_counts"])
    ]
    cases.extend(
        {"role": "seq_lens_h2d", "dimensions": [4], "tokens": requests, "execution_mode": "eager"}
        for requests in base["request_counts"]
    )
    return {
        "schema": "aisimulate-ascend-vector-plan/v1",
        "status": "experimental_collection_plan",
        "model_path": str(model_path),
        "cases": cases,
        "protocol": {name: base[name] for name in ("iterations", "warmup", "repeats")},
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (config_file, quant_file, model_file, base_file)
        },
        "source_contract": (
            "SGLang baa9befa381511cd4d56f83980951e5a45fffc6d; vector-state-report.json and current startup memory "
            "confirm BF16 norm bias, FP32 indexer LayerNorm, BF16 RoPE cache and eager INT64/graph INT32 scatter "
            "indices. ModelSlim RMSNorm and MoE scale/add boundaries are source-audited."
        ),
    }
