"""Publish complete MLA BMM observations with checked strides and runtime identity."""

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
        "dispatch_source_sha256",
    )
    if any(not runtime_identity.get(field) for field in required):
        raise ValueError("Incomplete runtime identity")
    if plan.get("schema") != "aisimulate-ascend-bmm-plan/v1":
        raise ValueError("Unsupported BMM plan")
    seen, rows, rejected = set(), [], []
    protocol, environment = None, None
    for path in observations:
        source_bytes = path.read_bytes()
        record = json.loads(source_bytes)
        if record.get("schema") != "aisimulate-ascend-bmm-observations/v1" or record.get("status") != "completed":
            raise ValueError("Incomplete/unsupported BMM observation")
        if (
            record["plan_sha256"] != plan_hash
            or record["collector_source_sha256"] != runtime_identity["collector_source_sha256"]
        ):
            raise ValueError("Plan/collector source mismatch")
        for field in ("framework_commit", "driver_version_info", "torch", "torch_npu", "dispatch_source_sha256"):
            if record["runtime"][field] != runtime_identity[field]:
                raise ValueError(f"Runtime identity mismatch: {field}")
        if record["runtime"]["device"] != runtime_identity["hardware"]:
            raise ValueError("Runtime hardware mismatch")
        current = record["protocol"]
        if (
            current != plan["protocol"]
            or min(current.values()) < 1
            or current["repeats"] < 3
            or current["iterations"] % current["weight_copies"]
        ):
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
            query = row["role"] == "query_nope"
            heads = row["local_heads"]
            expected_input = [heads * 256, 256, 1] if query else [heads * 512, 512, 1]
            expected_weight = [98304, 1, 192] if query else [131072, 256, 1]
            expected_kernel = (
                "torch.ops.npu.batch_matmul_transpose"
                if row["role"] == "output_decode"
                else "torch_npu.npu_transpose_batchmatmul"
            )
            if (
                row["role"] not in {"query_nope", "output_prefill", "output_decode"}
                or heads not in (4, 8, 16, 32, 64)
                or row["input_width"] != (192 if query else 512)
                or row["output_width"] != (512 if query else 256)
                or row["input_stride"] != expected_input
                or row["weight_stride"] != expected_weight
                or row["kernel_source"] != expected_kernel
                or row["dtype"] != "bfloat16"
                or row["format"] != 2
            ):
                raise ValueError("BMM invocation does not match the native consumer contract")
            verification = "captured_graph_replay" if row["execution_mode"] == "graph" else "eager_invocation"
            if (
                row["execution_verification"] != verification
                or len(row["checks"]) != 2
                or row["weight_copies"] != current["weight_copies"]
            ):
                raise ValueError("Actual execution and weight rotation must be checked before and after timing")
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
            result = {key: value for key, value in case.items() if key not in ("input_stride", "weight_stride")}
            for name, value in zip(
                (
                    "input_token_stride",
                    "input_head_stride",
                    "input_inner_stride",
                    "weight_head_stride",
                    "weight_inner_stride",
                    "weight_output_stride",
                ),
                expected_input + expected_weight,
                strict=True,
            ):
                result[name] = value
            result.update(
                execution_verification=verification,
                latency_ms=statistics.median(values),
                cv=variation,
                sample_count=len(values),
                numerical_checks=sum(check["sampled_elements"] for check in row["checks"]),
                source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            )
            rows.append(result)
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

    package = Path(__file__).resolve().parents[2]
    identity = report["identity"]["runtime"]
    from collector.sglang.ascend_vector import runtime_version

    version = runtime_version(identity["framework_commit"])
    if (
        hashlib.sha256(Path(__file__).with_name("ascend_bmm.py").read_bytes()).hexdigest()
        != identity["collector_source_sha256"]
    ):
        raise ValueError("Publish from the exact measured collector source checkout")
    plan = json.loads(args.plan.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    parquet.write_table(arrow.Table.from_pylist(rows), args.output_dir / "ascend_bmm_perf.parquet", compression="zstd")
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
            "ascend_bmm_perf": {
                "collector_ref": args.collector_ref,
                "collector_hash": collector_hash(
                    "collector.sglang.ascend_bmm", package, load_closures(package / "collector/hash_closures.yaml")
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
