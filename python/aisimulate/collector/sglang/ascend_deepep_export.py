"""Publish verified transport samples with explicit rank placement and routing."""

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
    runtime_fields = ("torch", "torch_npu", "deep_ep", "driver_version_info")
    required = (*runtime_fields, "image_digest", "framework_commit", "hardware", "collector_source_sha256")
    if any(not runtime_identity.get(field) for field in required):
        raise ValueError("Incomplete runtime identity")
    protocol = plan["protocol"]
    if (
        plan.get("schema") != "aisimulate-ascend-deepep-plan/v1"
        or min(protocol.values()) < 1
        or protocol["repeats"] < 3
    ):
        raise ValueError("Invalid transport plan/protocol")
    seen, rows, rejected, placements = set(), [], [], []
    environment = None
    for path in observations:
        source = path.read_bytes()
        record = json.loads(source)
        if record.get("schema") != "aisimulate-ascend-deepep-observations/v1" or record.get("status") not in (
            "completed",
            "failed",
        ):
            raise ValueError("Incomplete transport observations")
        if record.get("timing_method") != "per_call_outer_events_barrier_between_calls":
            raise ValueError("Unqualified transport timing method")
        if record.get("negative_control"):
            raise ValueError("Cannot publish negative controls")
        if (
            record["plan_sha256"] != plan_hash
            or record["collector_source_sha256"] != runtime_identity["collector_source_sha256"]
        ):
            raise ValueError("Plan/collector source mismatch")
        if record["framework_commit"] != runtime_identity["framework_commit"] or record["protocol"] != protocol:
            raise ValueError("Framework or protocol mismatch")
        setup, devices = record["serving_setup"], record["devices"]
        if environment is not None and environment != setup:
            raise ValueError("Cannot merge different serving setups")
        environment = setup
        world = len(devices)
        if world not in (16, 32) or [device["rank"] for device in devices] != list(range(world)):
            raise ValueError("Incomplete rank identity")
        for device in devices:
            if any(device[field] != runtime_identity[field] for field in runtime_fields):
                raise ValueError("Rank runtime mismatch")
            if device["device"] != runtime_identity["hardware"] or device["local_rank"] != device["rank"] % 16:
                raise ValueError("Rank hardware/layout mismatch")
            if device["cpu_affinity"] != setup["worker_cpu_affinity"][str(device["local_rank"])]:
                raise ValueError("Serving affinity mismatch")
        hosts = [devices[start]["host"] for start in range(0, world, 16)]
        if len(set(hosts)) != world // 16 or any(device["host"] != hosts[device["rank"] // 16] for device in devices):
            raise ValueError("Rank placement is not contiguous per node")
        placements.append({"hosts": hosts, "ranks": devices})
        completed = [case["plan_index"] for case in record["cases"]]
        selected = record["selected_plan_indices"]
        if record["status"] == "failed":
            failed = record.get("failed_plan_index")
            if (
                completed != selected[: len(completed)]
                or not record.get("error")
                or len(completed) >= len(selected)
                or selected[len(completed)] != failed
                or failed in seen
            ):
                raise ValueError("Invalid failed case record")
            seen.add(failed)
            rejected.append(
                {
                    "plan_index": failed,
                    "classification": "unexpected",
                    "error": record["error"],
                    "operations": ["dispatch", "combine"],
                }
            )
        elif completed != selected:
            raise ValueError("Missing selected cases")
        for observed in record["cases"]:
            index = observed["plan_index"]
            if type(index) is not int or not 0 <= index < len(plan["cases"]) or index in seen:
                raise ValueError("Duplicate or unknown case")
            seen.add(index)
            case = plan["cases"][index]
            if any(observed.get(name) != value for name, value in case.items()):
                raise ValueError("Case differs from frozen plan")
            if (
                case["num_ranks"] != world
                or case["ranks_per_node"] != 16
                or case["rank_layout"] != "contiguous"
                or case["hidden_size"] != 6144
                or case["num_experts"] != 256
                or case["topk"] != 8
                or case["max_dispatch_tokens_per_rank"] != 32
                or case["tokens_per_rank"] < 1
                or case["phase"] not in ("normal", "low_latency")
                or case["distribution"] not in ("balanced", "single_expert_group")
                or case["execution_mode"] not in ("eager", "graph")
                or (case["phase"] == "normal" and case["execution_mode"] != "eager")
                or (case["phase"] == "low_latency" and case["tokens_per_rank"] > 32)
            ):
                raise ValueError("Unsupported transport geometry")
            implementation = (
                "_DeepEPDispatcherImplNormal" if case["phase"] == "normal" else "_DeepEPDispatcherImplLowLatency"
            )
            if observed["kernel_source"] != "sglang.srt.layers.moe.token_dispatcher.deepep." + implementation:
                raise ValueError("Transport kernel mismatch")
            measurements = observed["measurements"]
            population = world * case["tokens_per_rank"]
            if case["distribution"] == "balanced":
                expert_counts = [population // 256 * 8] * 256
                for token in range(population % 256):
                    for choice in range(8):
                        expert_counts[(token * 17 + choice * 31) % 256] += 1
            else:
                expert_counts = [population] * 8 + [0] * 248
            if [measurement["operation"] for measurement in measurements] != ["dispatch", "combine"]:
                raise ValueError("Both transport boundaries are required")
            for measurement in measurements:
                verification = "captured_graph_replay" if case["execution_mode"] == "graph" else "eager_invocation"
                rank_results = measurement["rank_results"]
                if measurement["execution_verification"] != verification or [
                    rank["rank"] for rank in rank_results
                ] != list(range(world)):
                    raise ValueError("Missing actual execution/rank results")
                checked = 0
                for rank in rank_results:
                    if "combine_dtype" in rank and rank["combine_dtype"] != "bfloat16":
                        raise ValueError("Observed combine precision differs from the BF16 performance-table contract")
                    # Historical collectors checked INT8/FP32 before publishing
                    # but omitted these fields. New records also persist them.
                    precision_fields = {"dispatch_dtype", "dispatch_scale_dtype"}
                    if precision_fields.intersection(rank) and (
                        rank.get("dispatch_dtype") != "int8" or rank.get("dispatch_scale_dtype") != "float32"
                    ):
                        raise ValueError("Observed dispatch precision differs from the INT8 performance-table contract")
                    for name in ("samples_ms", "wall_samples_ms"):
                        samples = rank[name]
                        if len(samples) != protocol["repeats"] or any(
                            not math.isfinite(value) or value <= 0 for value in samples
                        ):
                            raise ValueError("Invalid timing samples")
                    counts = rank["expected_counts"]
                    if len(counts) != 256 // world or any(type(count) is not int or count < 0 for count in counts):
                        raise ValueError("Invalid expert counts")
                    start = rank["rank"] * (256 // world)
                    if counts != expert_counts[start : start + 256 // world]:
                        raise ValueError("Expert counts differ from the declared routing profile")
                    elements = (
                        sum(counts) * 6144 if measurement["operation"] == "dispatch" else case["tokens_per_rank"] * 6144
                    )
                    if len(rank["checks"]) != 2 or any(
                        check["calls"] != 1 or check["checked_elements"] != elements for check in rank["checks"]
                    ):
                        raise ValueError("Incomplete full-payload numerical checks")
                    checked += sum(check["checked_elements"] for check in rank["checks"])
                if sum(sum(rank["expected_counts"]) for rank in rank_results) != world * case["tokens_per_rank"] * 8:
                    raise ValueError("Incomplete routed token population")
                critical = [
                    max(rank["samples_ms"][sample] for rank in rank_results) for sample in range(protocol["repeats"])
                ]
                latency = statistics.median(critical)
                variation = statistics.pstdev(critical) / statistics.mean(critical)
                if measurement["latency_ms"] != latency or measurement["cv"] != variation:
                    raise ValueError("Timing aggregate differs from rank samples")
                if variation > 0.1:
                    rejected.append({"plan_index": index, "operation": measurement["operation"], "cv": variation})
                    continue
                rows.append(
                    {
                        **case,
                        "operation": measurement["operation"],
                        "dispatch_dtype": "int8",
                        "combine_dtype": "bfloat16",
                        "kernel_source": observed["kernel_source"],
                        "execution_verification": verification,
                        "latency_ms": latency,
                        "cv": variation,
                        "sample_count": protocol["repeats"],
                        "numerical_checks": checked,
                        "source_sha256": hashlib.sha256(source).hexdigest(),
                    }
                )
    if seen != set(range(len(plan["cases"]))):
        raise ValueError("Publication requires the complete frozen plan")
    if not rows:
        raise ValueError("No qualified transport samples")
    identity = {
        "runtime": runtime_identity,
        "protocol": protocol,
        "serving_setup": environment,
        "plan_sha256": plan_hash,
    }
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for row in rows:
        row["runtime_identity_sha256"] = identity_hash
    return rows, {
        "identity": identity,
        "placements": placements,
        "attempted": len(seen) * 2,
        "qualified": len(rows),
        "unqualified": rejected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path, nargs="+")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--collector-ref", required=True)
    args = parser.parse_args()
    identity = json.loads(args.runtime_identity.read_text())
    rows, report = validated_rows(args.plan, args.observations, identity)
    if (
        hashlib.sha256(Path(__file__).with_name("ascend_deepep.py").read_bytes()).hexdigest()
        != identity["collector_source_sha256"]
    ):
        raise ValueError("Publish from the exact measured collector source")
    import pyarrow as arrow
    import pyarrow.parquet as parquet
    from collector.provenance import case_plan_hash, collector_hash, load_closures, write_collection_meta

    package = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    parquet.write_table(
        arrow.Table.from_pylist(rows), args.output_dir / "ascend_deepep_perf.parquet", compression="zstd"
    )
    (args.output_dir / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    plan = json.loads(args.plan.read_text())
    write_collection_meta(
        args.output_dir,
        {
            "framework": "sglang",
            "version": "0.5.22",
            "image_digest": identity["image_digest"],
            "source_commit": identity["framework_commit"],
            "abi": {name: identity[name] for name in ("torch", "torch_npu", "deep_ep", "driver_version_info")},
        },
        {
            "ascend_deepep_perf": {
                "collector_ref": args.collector_ref,
                "collector_hash": collector_hash(
                    "collector.sglang.ascend_deepep", package, load_closures(package / "collector/hash_closures.yaml")
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
