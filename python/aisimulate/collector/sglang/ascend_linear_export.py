"""Publish complete Ascend linear campaigns after numerical and stability checks."""

import argparse
import hashlib
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path


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
    seen, rows, rejected = set(), [], []
    protocol, environment = None, None
    for path in observations:
        source_bytes = path.read_bytes()
        record = json.loads(source_bytes)
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        if record.get("schema") != "aisimulate-ascend-linear-observations/v1" or record.get("status") != "completed":
            raise ValueError("Incomplete/unsupported linear observation")
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
        if current["repeats"] < 3 or current["weight_copies"] < 1 or current["warmup"] < 1:
            raise ValueError("Insufficient measurement protocol")
        if current["iterations"] < current["weight_copies"] or current["iterations"] % current["weight_copies"]:
            raise ValueError("Invalid call/weight rotation ratio")
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
            for field, value in plan["cases"][index].items():
                if row.get(field) != value:
                    raise ValueError(f"Case differs from the frozen plan: {index}/{field}")
            verification = "captured_graph_replay" if row["execution_mode"] == "graph" else "eager_invocation"
            if row["execution_verification"] != verification or len(row["checks"]) != 2:
                raise ValueError("Actual execution must be numerically checked before and after timing")
            if row["weight_copies"] != protocol["weight_copies"]:
                raise ValueError("Weight rotation differs from protocol")
            kernels = {
                "w8a8": "sglang.srt.hardware_backend.npu.quantization.linear_method_npu.NPUW8A8Int8LinearMethod",
                "w8a8_dynamic": (
                    "sglang.srt.hardware_backend.npu.quantization.linear_method_npu.NPUW8A8Int8DynamicLinearMethod"
                ),
                "bfloat16": "torch.nn.functional.linear",
                "float32_bfloat16": "torch.nn.functional.linear",
            }
            boundary_dtype = "float32" if row["quant_mode"] == "float32_bfloat16" else "bfloat16"
            if (
                row["input_dtype"] != boundary_dtype
                or row["output_dtype"] != boundary_dtype
                or row["weight_format"] not in (2, 29)
                or row["kernel_source"] != kernels.get(row["quant_mode"])
                or (row["quant_bias"] and row["quant_mode"] != "w8a8")
            ):
                raise ValueError("Linear invocation does not match the native consumer contract")
            for check in row["checks"]:
                if (
                    check["sampled_elements"] <= 0
                    or not math.isfinite(check["max_abs_error"])
                    or not 0 <= check["quant_max_bin_error"] <= 1
                    or not math.isfinite(check["quant_max_scale_error"])
                    or check["quant_max_scale_error"] < 0
                ):
                    raise ValueError("Invalid numerical check")
            values = row["samples_ms"]
            if len(values) != protocol["repeats"] or any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError("Invalid timing samples")
            variation = statistics.pstdev(values) / statistics.mean(values)
            if variation > 0.1:
                rejected.append({"plan_index": index, "cv": variation, "reason": "CV exceeds10%"})
                continue
            fields = (
                "quant_mode",
                "m",
                "n",
                "k",
                "quant_bias",
                "weight_format",
                "execution_mode",
                "input_dtype",
                "output_dtype",
                "kernel_source",
                "execution_verification",
            )
            rows.append(
                {
                    **{field: row[field] for field in fields},
                    "latency_ms": statistics.median(values),
                    "cv": variation,
                    "sample_count": len(values),
                    "numerical_checks": sum(check["sampled_elements"] for check in row["checks"]),
                    "source_sha256": source_hash,
                }
            )
    if seen != set(range(len(plan["cases"]))):
        raise ValueError("Publication requires the complete frozen plan; smoke/subsets remain observations")
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

    package = Path(__file__).resolve().parents[2]
    source_hash = hashlib.sha256(Path(__file__).with_name("ascend_linear.py").read_bytes()).hexdigest()
    identity = report["identity"]["runtime"]
    from collector.sglang.ascend_vector import runtime_version

    version = runtime_version(identity["framework_commit"])
    if source_hash != identity["collector_source_sha256"]:
        raise ValueError("Publish from the exact measured collector source checkout")
    plan = json.loads(args.plan.read_text())
    case_ids = [
        json.dumps(
            {field: case[field] for field in ("quant_mode", "m", "n", "k", "quant_bias", "execution_mode")},
            sort_keys=True,
            separators=(",", ":"),
        )
        for case in plan["cases"]
    ]
    closure_hash = collector_hash(
        "collector.sglang.ascend_linear", package, load_closures(package / "collector/hash_closures.yaml")
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    parquet.write_table(
        arrow.Table.from_pylist(rows), args.output_dir / "ascend_linear_perf.parquet", compression="zstd"
    )
    (args.output_dir / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    write_collection_meta(
        args.output_dir,
        {
            "framework": "sglang",
            "version": version,
            "image_digest": identity["image_digest"],
            "source_commit": identity["framework_commit"],
            "abi": {key: identity[key] for key in ("torch", "torch_npu", "driver_version_info")},
        },
        {
            "ascend_linear_perf": {
                "collector_ref": args.collector_ref,
                "collector_hash": closure_hash,
                "case_plan_hash": case_plan_hash(case_ids),
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
