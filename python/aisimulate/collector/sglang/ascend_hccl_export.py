"""Validate HCCL observations and publish only qualified physical-key rows."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

KEY_FIELDS = (
    "dtype",
    "operation",
    "num_ranks",
    "ranks_per_node",
    "rank_layout",
    "execution_mode",
    "input_bytes_per_rank",
)


def validated_rows(campaigns, runtime_identity):
    required = {
        "image_digest",
        "framework_commit",
        "cann_hccl_build",
        "driver_versions",
        "hardware",
        "collector_source_sha256",
    }
    if not required <= runtime_identity.keys() or any(not runtime_identity[field] for field in required):
        raise ValueError("Runtime identity must include immutable image/source and hardware/software provenance")
    rows, rejected, seen = [], [], set()
    protocol = None
    for path in campaigns:
        data = json.loads(path.read_text())
        if data.get("schema") != "aisimulate-hccl-observations/v3":
            raise ValueError("HCCL publication requires v3 verification of the actual timed execution path")
        if (
            data.get("collector_source_sha256") != runtime_identity["collector_source_sha256"]
            or data.get("framework_commit") != runtime_identity["framework_commit"]
        ):
            raise ValueError("Observed source differs from the selected runtime identity")
        if data.get("status") != "completed":
            raise ValueError(f"Incomplete collection: {path}")
        current_protocol = {field: data[field] for field in ("warmup", "repeats", "iterations")}
        if (
            data.get("warmup_scope") != "batches_of_iterations"
            or data.get("all_reduce_buffer_policy") != "one_buffer_per_invocation"
        ):
            raise ValueError("Missing independent AllReduce buffers or batch warmup protocol")
        current_protocol.update(
            warmup_scope=data["warmup_scope"], all_reduce_buffer_policy=data["all_reduce_buffer_policy"]
        )
        current_protocol["gc_policy"] = data.get("gc_policy", "enabled")
        current_protocol["runtime_environment"] = data.get("runtime_environment")
        current_protocol["hccl_group_options"] = data.get("hccl_group_options")
        current_protocol["serving_setup"] = data.get("serving_setup")
        setup = current_protocol["serving_setup"]
        environment = current_protocol["runtime_environment"]
        options = current_protocol["hccl_group_options"]
        # None records an unset option. Its default belongs to the immutable
        # runtime identity, and must stay distinct from explicitly setting0/1.
        if (
            not isinstance(environment, dict)
            or "TASK_QUEUE_ENABLE" not in environment
            or environment["TASK_QUEUE_ENABLE"] not in {None, "0", "1"}
        ):
            raise ValueError("Missing actual serving task-queue policy")
        if not isinstance(options, dict) or not isinstance(options.get("hccl_buffer_size"), int):
            raise ValueError("Missing resolved SGLang HCCL group options")
        configured_buffer = environment.get("DEEPEP_HCCL_BUFFSIZE") or environment.get("HCCL_BUFFSIZE") or "200"
        if options["hccl_buffer_size"] <= 0 or options["hccl_buffer_size"] != int(configured_buffer):
            raise ValueError("Resolved HCCL buffer differs from the serving environment")
        if not isinstance(setup, dict) or setup.get("schema") != "sglang-runtime-setup/v1":
            raise ValueError("Publication requires a complete recorded serving setup")
        if setup.get("environment") != environment:
            raise ValueError("Observed runtime environment differs from the selected setup")
        if protocol is not None and protocol != current_protocol:
            raise ValueError("Do not merge different measurement protocols")
        protocol = current_protocol
        world, local = data["num_ranks"], data["ranks_per_node"]
        if world < 2 or local < 1 or world % local or data["rank_layout"] != "contiguous":
            raise ValueError("Invalid collective placement")
        if sorted(device["rank"] for device in data["devices"]) != list(range(world)):
            raise ValueError("Incomplete rank identities")
        for device in data["devices"]:
            expected_mask = setup["worker_cpu_affinity"].get(str(device["local_rank"]))
            if device.get("cpu_affinity") != expected_mask:
                raise ValueError("Observed CPU affinity differs from the serving setup")
        source_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for case in data["cases"]:
            row = {field: case[field] for field in ("dtype", "operation", "execution_mode", "input_bytes_per_rank")}
            row.update(num_ranks=world, ranks_per_node=local, rank_layout=data["rank_layout"])
            key = tuple(row[field] for field in KEY_FIELDS)
            if key in seen:
                raise ValueError(f"Duplicate HCCL physical key: {key}")
            seen.add(key)
            rank_samples = case["rank_samples"]
            repeats = data["repeats"]
            if len(rank_samples) != world or repeats < 3:
                raise ValueError(f"Incomplete timing samples: {key}")
            if case["full_payload_correctness_checks"] != 2 * world:
                raise ValueError(f"Missing pre/post correctness on every rank: {key}")
            verification = "captured_graph_replay" if row["execution_mode"] == "graph" else "eager_invocation"
            buffers = data["iterations"] if row["operation"] == "all_reduce" else 1
            if (
                case.get("execution_verification") != verification
                or case.get("verification_buffers_per_rank") != buffers
            ):
                raise ValueError(f"Correctness was not checked on the actual timed buffers/execution path: {key}")
            for samples in rank_samples:
                values = samples["device_ms"]
                if len(values) != repeats or any(not math.isfinite(value) or value <= 0 for value in values):
                    raise ValueError(f"Invalid device timings: {key}")
            critical = [max(samples["device_ms"][index] for samples in rank_samples) for index in range(repeats)]
            variation = statistics.pstdev(critical) / statistics.mean(critical)
            if variation > 0.10:
                rejected.append({**row, "cv": variation, "source": str(path), "reason": "CV exceeds10%"})
                continue
            row.update(
                latency_ms=statistics.median(critical),
                cv=variation,
                sample_count=repeats,
                correctness_checks=2 * world,
                execution_verification=verification,
                kernel_source="torch.distributed.hccl",
                source_sha256=source_digest,
            )
            rows.append(row)
    identity = {"runtime": runtime_identity, "measurement_protocol": protocol}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for row in rows:
        row["runtime_identity_sha256"] = digest
    if not rows:
        raise ValueError("No qualified HCCL rows")
    return rows, {
        "identity": identity,
        "runtime_identity_sha256": digest,
        "qualified": len(rows),
        "unqualified": rejected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaigns", nargs="+", type=Path)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, report = validated_rows(args.campaigns, json.loads(args.runtime_identity.read_text()))
    import pyarrow as arrow
    import pyarrow.parquet as parquet

    args.output_dir.mkdir(parents=True, exist_ok=False)
    parquet.write_table(arrow.Table.from_pylist(rows), args.output_dir / "hccl_perf.parquet", compression="zstd")
    (args.output_dir / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"qualified": len(rows), "unqualified": len(report["unqualified"])}))


if __name__ == "__main__":
    main()
