"""Declare the observed NPU DSA cache, layout and graph-padding contracts."""

import hashlib
import json
from pathlib import Path

import yaml


def get_ascend_attention_cases(model_path, case_root=None):
    model_path = Path(model_path)
    case_root = Path(case_root) if case_root else Path(__file__).resolve().parents[1] / "cases"
    config_file = model_path / "config.json"
    config = json.loads(config_file.read_text())
    model_file = case_root / "models" / f"{config['architectures'][0]}_cases.yaml"
    profiles = yaml.safe_load(model_file.read_text())["model_case_values"]["ascend_attention"]
    matching = [value for value in profiles if value["checkpoint_quantization"] == "modelslim_w8a8"]
    if len(matching) != 1:
        raise ValueError("Expected one declared NPU DSA geometry")
    geometry = matching[0]
    for name in ("num_attention_heads", "kv_lora_rank", "qk_rope_head_dim", "index_n_heads", "index_head_dim"):
        if config.get(name) != geometry[name]:
            raise ValueError(f"Model differs from declared {name}")
    base_file = case_root / "base_ops/ascend_attention.yaml"
    base = yaml.safe_load(base_file.read_text())["common_case_values"]["ascend_attention"]
    bucket_sets = base["graph_bucket_sets"]
    if not bucket_sets or any(
        not buckets or buckets != sorted(set(buckets)) or min(buckets) < 1 or max(buckets) < max(base["decode_batches"])
        for buckets in bucket_sets
    ):
        raise ValueError("Graph bucket sets must cover the declared active batch range")
    cases = []

    def add(kind, heads, phase, tokens, active, padded, length, prefix, mode):
        if phase == "prefill":
            pages_per_request = (length + geometry["page_size"] - 1) // geometry["page_size"]
            table_width = pages_per_request
        else:
            pages_per_request = (length + geometry["page_size"] - 1) // geometry["page_size"]
            table_width = (geometry["context_limit"] + geometry["page_size"] - 1) // geometry["page_size"]
        cases.append(
            {
                "operation": kind,
                "phase": phase,
                "local_heads": heads,
                "tokens": tokens,
                "active_batch": active,
                "padded_batch": padded,
                "sequence_length": length,
                "prefix": prefix,
                "execution_mode": mode,
                "page_size": geometry["page_size"],
                "sparse_topk": geometry["sparse_topk"],
                "kv_lora_rank": geometry["kv_lora_rank"],
                "rope_dim": geometry["qk_rope_head_dim"],
                "index_heads": geometry["index_n_heads"],
                "index_dim": geometry["index_head_dim"],
                "pool_pages": max(geometry["minimum_pool_pages"], 1 + active * pages_per_request),
                "block_table_width": table_width,
            }
        )

    for operation in ("lightning_indexer", "sparse_attention"):
        head_counts = (
            [geometry["index_n_heads"]]
            if operation == "lightning_indexer"
            else [geometry["num_attention_heads"] // parallel for parallel in geometry["tensor_parallel_sizes"]]
        )
        for heads in head_counts:
            eager_alignments = (
                geometry["tensor_parallel_sizes"]
                if operation == "lightning_indexer"
                else [geometry["num_attention_heads"] // heads]
            )
            for mode in base["execution_modes"]:
                for tokens in base["prefill_tokens"]:
                    for active in base["prefill_batches"]:
                        if active * tokens > base["max_prefill_tokens"]:
                            continue
                        for prefix in base["prefix_tokens"]:
                            add(
                                operation,
                                heads,
                                "prefill",
                                active * tokens,
                                active,
                                active,
                                tokens + prefix,
                                prefix,
                                mode,
                            )
                layouts = set()
                for active in base["decode_batches"]:
                    padded_batches = (
                        sorted({next(value for value in buckets if value >= active) for buckets in bucket_sets})
                        if mode == "graph"
                        else sorted({active, *((active + size - 1) // size * size for size in eager_alignments)})
                    )
                    layouts.update((active, padded) for padded in padded_batches)
                if mode == "eager" and base.get("eager_dp_max_len", False):
                    maximum = max(base["decode_batches"])
                    for alignment in eager_alignments:
                        limit = (maximum + alignment - 1) // alignment * alignment
                        for padded in range(alignment, limit + 1, alignment):
                            layouts.update(
                                (active, padded)
                                for active in (1, (padded + 1) // 2, padded - 1)
                                if 0 < active < padded and active <= maximum
                            )
                for active, padded in sorted(layouts):
                    for length in base["context_lengths"]:
                        add(operation, heads, "decode", padded, active, padded, length, 0, mode)
    return {
        "schema": "aisimulate-ascend-attention-plan/v1",
        "status": "experimental_collection_plan",
        "model_path": str(model_path),
        "cases": cases,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (config_file, model_file, base_file)
        },
        "protocol": {key: base[key] for key in ("iterations", "warmup", "repeats")},
        "source_contract": (
            "SGLang baa9befa381511cd4d56f83980951e5a45fffc6d; "
            "operator-input-state-node31-20261022-attempt2 confirms TND/PA_BSND, "
            "128page_size,2048topk,zero-length padding and decode blocktable width1407; "
            "operator-input-state-node32-20261022-tp16-adp1-20261024 also confirms decode buckets16/32; "
            "model_executor/forward_batch_info.py:1663-1908 pads eager MLP-sync decode tokens to attention TP; "
            "hardware_backend/npu/attention/ascend_backend.py:1074-1075 uses zero KV length for padded rows."
        ),
    }
