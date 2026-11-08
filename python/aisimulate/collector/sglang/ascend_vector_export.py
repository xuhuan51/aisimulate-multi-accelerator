"""Validate complete vector campaigns and publish their qualified native rows."""

import argparse
import hashlib
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path

CONTRACT = "glm52_modelslim_vectors/v1"


def expected_invocation(case):
    role, dimensions, tokens = case["role"], case["dimensions"], case["tokens"]

    def tensor(shape, dtype="bfloat16", stride=None, offset=0, storage_format=2):
        contiguous, width = [], 1
        for size in reversed(shape):
            contiguous.insert(0, width)
            width *= size
        return {
            "shape": shape,
            "stride": stride or contiguous,
            "storage_offset": offset,
            "dtype": "torch." + dtype,
            "format": storage_format,
        }

    if role == "dp_buffer_zero" and dimensions == [6144]:
        values = tensor([tokens, 6144])
        return {"buffer": values}, values, "torch.Tensor.fill_"
    if role in {"dp_gather_copy", "dp_scatter_copy"} and dimensions == [6144]:
        scatter = role == "dp_scatter_copy"
        source = tensor([tokens * (2 if scatter else 1), 6144])
        target = tensor([tokens * (1 if scatter else 2), 6144])
        return (
            {"source": source, "target": target, "offset": tensor([], "int32"), "size": tensor([], "int32")},
            target,
            "sglang.srt.layers.dp_attention.memcpy",
        )
    if role == "seq_lens_h2d" and dimensions == [4] and case["execution_mode"] == "eager":
        return (
            {"lengths": {**tensor([tokens], "int32", storage_format=None), "device": "cpu", "pinned": False}},
            tensor([tokens], "int32"),
            "torch.Tensor.to",
        )
    if role in {"rms_bias", "add_rms_bias"} and dimensions == [6144]:
        values, parameter = tensor([tokens, 6144]), tensor([6144])
        return (
            {
                "input": values,
                "weight": parameter,
                "bias": parameter,
                "residual": values if role == "add_rms_bias" else None,
            },
            [values, values] if role == "add_rms_bias" else values,
            "sglang.srt.layers.quantization.modelslim.modelslim.npu_wrapper_rmsnorm_forward",
        )
    if role == "split_qk_norm" and dimensions == [2048, 512, 64]:
        return (
            {
                "input": tensor([tokens, 2624]),
                "query_weight": tensor([2048]),
                "query_bias": tensor([2048]),
                "key_weight": tensor([512]),
                "key_bias": tensor([512]),
            },
            [tensor([tokens, 2048]), tensor([tokens, 1, 512]), tensor([tokens, 1, 64])],
            "sgl_kernel_npu.norm.fused_split_qk_norm.fused_split_qk_norm",
        )
    if role == "indexer_norm" and dimensions == [128]:
        return (
            {"input": tensor([tokens, 128]), "weight": tensor([128], "float32"), "bias": tensor([128], "float32")},
            tensor([tokens, 128]),
            "sglang.srt.layers.layernorm.LayerNorm.forward_npu",
        )
    if (role == "rope_attention" and dimensions in ([heads, 64, 256, 64] for heads in (4, 8, 16, 32, 64))) or (
        role == "rope_indexer" and dimensions == [32, 64, 128, 128]
    ):
        heads, width, query_width, key_width = dimensions
        query = tensor(
            [tokens, heads, width],
            stride=[heads * query_width, query_width, 1],
            offset=192 if role == "rope_attention" else 0,
        )
        key = tensor([tokens, 1, width], stride=[key_width, width, 1])
        return (
            {"query": query, "key": key, "cos_sin": tensor([tokens, width])},
            [tensor([tokens, heads, width]), tensor([tokens, 1, width])],
            "sgl_kernel_npu.norm.fused_rope_qk_mqa.fused_rope_qk_mqa",
        )
    if role == "concat_indexer_q" and dimensions == [32, 64, 128]:
        return (
            {
                "leading": tensor([tokens, 32, 64]),
                "trailing": tensor([tokens, 32, 64], stride=[4096, 128, 1], offset=64),
            },
            tensor([tokens, 32, 128]),
            "torch.cat",
        )
    if role == "concat_indexer_k" and dimensions == [1, 64, 128]:
        return (
            {"leading": tensor([tokens, 64]), "trailing": tensor([tokens, 64], stride=[128, 1], offset=64)},
            tensor([tokens, 128]),
            "torch.cat",
        )
    if role == "cache_scatter" and dimensions in ([53376, 64], [53376, 128], [53376, 512]):
        capacity, width = dimensions
        cache = tensor([capacity, 1, width], storage_format=30)
        return (
            {
                "cache": cache,
                "indices": tensor([tokens, 1], "int32" if case["execution_mode"] == "graph" else "int64"),
                "updates": tensor([tokens, 1, width]),
            },
            cache,
            "torch_npu.npu_scatter_nd_update_",
        )
    if role == "swiglu" and dimensions in ([64], [128], [256], [384], [512], [768], [1536], [2048], [3072]):
        return {"input": tensor([tokens, dimensions[0] * 2])}, tensor([tokens, dimensions[0]]), "torch_npu.npu_swiglu"
    if role == "router_topk" and dimensions == [256, 8]:
        return (
            {"logits": tensor([tokens, 256]), "bias": tensor([256], "float32")},
            [tensor([tokens, 8], "float32"), tensor([tokens, 8], "int32"), tensor([tokens, 256], "float32")],
            "torch.ops.npu.npu_moe_gating_top_k",
        )
    if role == "cast_hidden_fp32" and dimensions == [6144]:
        return {"input": tensor([tokens, 6144])}, tensor([tokens, 6144], "float32"), "torch.Tensor.to"
    if role == "cast_weights_bf16" and dimensions == [32]:
        return {"input": tensor([tokens, 32], "float32")}, tensor([tokens, 32]), "torch.Tensor.to"
    if role == "logits_finalize" and dimensions in ([parallel, 154880] for parallel in (1, 2, 4, 8, 16)):
        parallel, vocabulary = dimensions
        output = tensor([tokens, vocabulary], "float32")
        return (
            {
                "gathered_logits": tensor([parallel, tokens, vocabulary // parallel]),
                "logits_buffer": output if case["execution_mode"] == "graph" else None,
            },
            output,
            "sglang.srt.layers.logits_processor.LogitsProcessor._copy_logits_to_buffer",
        )
    if (
        role == "embedding" and dimensions in ([vocabulary, 6144] for vocabulary in (9680, 19360, 38720, 77440, 154880))
    ) or (role == "rope_cache_gather" and dimensions == [1048576, 64]):
        return (
            {"indices": tensor([tokens], "int64"), "weight": tensor(dimensions)},
            tensor([tokens, dimensions[1]]),
            "torch.nn.functional.embedding" if role == "embedding" else "torch.Tensor.index_select",
        )
    if role in {"routed_scale", "shared_add"} and dimensions == [6144]:
        values = tensor([tokens, 6144])
        return (
            {"input": values, "shared": values if role == "shared_add" else None},
            values,
            "torch.Tensor.mul_" if role == "routed_scale" else "torch.Tensor.add_",
        )
    raise ValueError("Unsupported vector role/geometry")


def matches_signature(actual, expected):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            return False
        if "shape" in expected:
            if any(type(value) is not int for value in actual["shape"] + actual["stride"] + [actual["storage_offset"]]):
                return False
            if expected["format"] is not None and type(actual["format"]) is not int:
                return False
            if "pinned" in expected and type(actual["pinned"]) is not bool:
                return False
            return (
                all(actual[field] == expected[field] for field in expected if field != "stride")
                and len(actual["stride"]) == len(expected["stride"])
                and all(
                    size == 1 or value == wanted
                    for size, value, wanted in zip(expected["shape"], actual["stride"], expected["stride"], strict=True)
                )
            )
        return all(matches_signature(actual[name], value) for name, value in expected.items())
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(matches_signature(value, wanted) for value, wanted in zip(actual, expected, strict=True))
        )
    return actual == expected


def validated_rows(plan_path, observations, runtime_identity):
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    plan_hash = hashlib.sha256(plan_bytes).hexdigest()
    required = (
        "image_digest",
        "framework_commit",
        "driver_version_info",
        "hardware",
        "collector_source_sha256",
        "torch",
        "torch_npu",
    )
    if any(not runtime_identity.get(field) for field in required):
        raise ValueError("Incomplete runtime identity")
    if plan.get("schema") != "aisimulate-ascend-vector-plan/v1":
        raise ValueError("Unsupported vector plan")
    seen, rows, rejected = set(), [], []
    protocol, environment = None, None
    for path in observations:
        source_bytes = path.read_bytes()
        record = json.loads(source_bytes)
        if record.get("schema") != "aisimulate-ascend-vector-observations/v1" or record.get("status") != "completed":
            raise ValueError("Incomplete/unsupported vector observation")
        if (
            record["plan_sha256"] != plan_hash
            or record["collector_source_sha256"] != runtime_identity["collector_source_sha256"]
        ):
            raise ValueError("Plan/collector source mismatch")
        for field in ("framework_commit", "driver_version_info", "torch", "torch_npu"):
            if record["runtime"][field] != runtime_identity[field]:
                raise ValueError(f"Runtime identity mismatch: {field}")
        if record["runtime"]["device"] != runtime_identity["hardware"]:
            raise ValueError("Runtime hardware mismatch")
        current = record["protocol"]
        if current != plan["protocol"] or min(current.values()) < 1 or current["repeats"] < 3:
            raise ValueError("Invalid measurement protocol")
        setup = record["runtime"]["serving_setup"]
        if setup["worker_cpu_affinity"][str(record["runtime"]["device_index"])] != record["runtime"]["cpu_affinity"]:
            raise ValueError("Serving CPU affinity mismatch")
        if protocol is not None and (protocol != current or environment != setup):
            raise ValueError("Cannot merge different protocols or serving setups")
        protocol, environment = current, setup
        if [row["plan_index"] for row in record["cases"]] != record["selected_plan_indices"]:
            raise ValueError("Missing planned shard results")
        for row in record["cases"]:
            index = row["plan_index"]
            if type(index) is not int or not 0 <= index < len(plan["cases"]) or index in seen:
                raise ValueError("Duplicate or unknown case identity")
            seen.add(index)
            case = plan["cases"][index]
            if any(row.get(field) != value for field, value in case.items()):
                raise ValueError("Case differs from the frozen plan")
            inputs, outputs, kernel = expected_invocation(case)
            if (
                row["kernel_source"] != kernel
                or not matches_signature(row["input_signature"], inputs)
                or not matches_signature(row["output_signature"], outputs)
            ):
                raise ValueError(f"Vector invocation violates the native consumer contract: {index}/{case['role']}")
            verification = "captured_graph_replay" if case["execution_mode"] == "graph" else "eager_invocation"
            if row["execution_verification"] != verification or len(row["checks"]) != 2:
                raise ValueError("Actual execution must be checked before and after timing")
            if any(
                check["sampled_elements"] <= 0
                or not math.isfinite(check["max_abs_error"])
                or check["max_abs_error"] < 0
                for check in row["checks"]
            ):
                raise ValueError("Invalid numerical check")
            values = row["samples_ms"]
            if len(values) != current["repeats"] or any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError("Invalid timing samples")
            variation = statistics.pstdev(values) / statistics.mean(values)
            if variation > 0.1:
                rejected.append({"plan_index": index, "cv": variation, "reason": "CV exceeds10%"})
                continue
            signature = json.dumps(
                [row["input_signature"], row["output_signature"]], sort_keys=True, separators=(",", ":")
            )
            rows.append(
                {
                    **case,
                    "dimensions": json.dumps(case["dimensions"], separators=(",", ":")),
                    "kernel_source": kernel,
                    "boundary_contract": CONTRACT,
                    "signature_sha256": hashlib.sha256(signature.encode()).hexdigest(),
                    "execution_verification": verification,
                    "latency_ms": statistics.median(values),
                    "cv": variation,
                    "sample_count": len(values),
                    "numerical_checks": sum(check["sampled_elements"] for check in row["checks"]),
                    "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                }
            )
    if seen != set(range(len(plan["cases"]))):
        raise ValueError("Publication requires the complete frozen plan")
    if not rows:
        raise ValueError("No qualified measurements")
    identity = {
        "runtime": runtime_identity,
        "protocol": protocol,
        "serving_setup": environment,
        "plan_sha256": plan_hash,
    }
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for row in rows:
        row["runtime_identity_sha256"] = identity_hash
    return rows, {"identity": identity, "attempted": len(seen), "qualified": len(rows), "unqualified": rejected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path, nargs="+")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--collector-ref", required=True)
    args = parser.parse_args()
    rows, report = validated_rows(args.plan, args.observations, json.loads(args.runtime_identity.read_text()))
    import pyarrow as arrow
    import pyarrow.parquet as parquet
    from collector.provenance import case_plan_hash, collector_hash, load_closures, write_collection_meta
    from collector.sglang.ascend_vector import runtime_version, validate_runtime_case

    package = Path(__file__).resolve().parents[2]
    identity = report["identity"]["runtime"]
    plan = json.loads(args.plan.read_text())
    version = runtime_version(identity["framework_commit"], plan.get("framework_commit"))
    for case in plan["cases"]:
        validate_runtime_case(identity["framework_commit"], case)
    sources = {"collector": "ascend_vector.py", "kernels": "ascend_vector_kernels.py"}
    if any(
        hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()
        != identity["collector_source_sha256"][name]
        for name, filename in sources.items()
    ):
        raise ValueError("Publish from the exact measured collector sources")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    parquet.write_table(
        arrow.Table.from_pylist(rows), args.output_dir / "ascend_vector_perf.parquet", compression="zstd"
    )
    (args.output_dir / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    write_collection_meta(
        args.output_dir,
        {
            "framework": "sglang",
            "version": version,
            "image_digest": identity["image_digest"],
            "source_commit": identity["framework_commit"],
            "abi": {name: identity[name] for name in ("torch", "torch_npu", "driver_version_info")},
        },
        {
            "ascend_vector_perf": {
                "collector_ref": args.collector_ref,
                "collector_hash": collector_hash(
                    "collector.sglang.ascend_vector", package, load_closures(package / "collector/hash_closures.yaml")
                ),
                "case_plan_hash": case_plan_hash(
                    [json.dumps(case, sort_keys=True, separators=(",", ":")) for case in plan["cases"]]
                ),
                "collected_at": datetime.fromtimestamp(
                    max(path.stat().st_mtime for path in args.observations), UTC
                ).isoformat(),
                "rows": len(rows),
                "classified_failures": len(report["unqualified"]),
                "status": "complete",
            }
        },
        provenance_tier="collected",
    )
    print(json.dumps({"qualified": len(rows), "unqualified": len(report["unqualified"])}))


if __name__ == "__main__":
    main()
