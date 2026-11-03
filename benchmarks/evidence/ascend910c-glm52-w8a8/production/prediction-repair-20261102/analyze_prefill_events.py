import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path

from compare_runtime_layer_prediction import configuration
from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def geometry(group):
    return tuple(sorted(tuple(row["scheduled_requests"][key] for key in (
        "num_prefill_requests", "sum_prefill_tokens", "sum_prefill_kv_tokens",
        "num_decode_requests", "sum_decode_kv_tokens")) for row in group))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.campaign
    manifest = json.loads((root / "prefill-event-campaign.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Campaign incomplete")
    sources = json.loads((root / "snapshot-sha256.json").read_text())
    for name, expected in sources.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen measurement changed: {name}")
    paired, events = collections.defaultdict(dict), collections.defaultdict(dict)
    for path in root.glob("paired-forward-rank*.jsonl"):
        for row in rows(path):
            group = paired[row["forward_iteration"]]
            if row["dp_rank"] in group:
                raise ValueError("Duplicate paired rank")
            group[row["dp_rank"]] = row
    for path in root.glob("prefill-events-rank*.jsonl"):
        for row in rows(path):
            group = events[row["case"], row["model_call"]]
            if row["tp_rank"] in group:
                raise ValueError("Duplicate event rank")
            group[row["tp_rank"]] = row
    controls = collections.defaultdict(list)
    for case in manifest["cases"]:
        if "-r2-control" not in case["id"]:
            continue
        for group in paired.values():
            if len(group) != 4 or not all(case["start_unix"] <= row["launch_unix"] <= case["end_unix"] for row in group.values()):
                continue
            controls[geometry(group.values())].append(max(row["elapsed_ms"] for row in group.values()))
    configs = {scope: {**configuration(scope), "worker_type": "aggregated"} for scope in ("isolated_kernels", "runtime_layers")}
    models = {scope: RustForwardPassPerfModel.best_available(config) for scope, config in configs.items()}
    output = []
    for (case, call), group in sorted(events.items()):
        if "-r3-events" not in case:
            continue
        if set(group) != set(range(16)):
            raise ValueError("Incomplete event rank set")
        matching = []
        for row in group.values():
            candidates = [paired_row for ranks in paired.values() for rank, paired_row in ranks.items()
                          if rank == row["dp_rank"] and paired_row["launch_unix"] <= row["start_unix"]]
            matched = max(candidates, key=lambda value: value["launch_unix"])
            if not 0 <= row["start_unix"] - matched["launch_unix"] < 1:
                raise ValueError("Event/forward association is not identifiable")
            matching.append(matched["forward_iteration"])
        if len(set(matching)) != 1:
            raise ValueError("Events span multiple global forwards")
        forward = paired[matching[0]]
        if len(forward) != 4:
            raise ValueError("Incomplete paired forward")
        metrics = [{"scheduled_requests": forward[rank]["scheduled_requests"]} for rank in range(4)]
        rank_parts = []
        for rank, row in sorted(group.items()):
            parts = {}
            for module, expected_count in (("DeepseekV2DecoderLayer", 78), ("DeepseekV2AttentionMLA", 78), ("DeepseekV2MoE", 75)):
                observed = [part for part in row["boundaries"] if part["module"] == module]
                if len(observed) != expected_count:
                    raise ValueError(f"Incomplete boundary: {module}")
                parts[module] = sum(part["device_ms"] for part in observed)
            rank_parts.append({"rank": rank, "phase": row["forward_mode"], "device_ms": parts})
        active = [row for row in rank_parts if row["phase"] == "EXTEND"]
        critical = max(active, key=lambda row: row["device_ms"]["DeepseekV2DecoderLayer"])
        predictions = {scope: model.estimate_forward_pass_time_ms(metrics) for scope, model in models.items()}
        operations = models["runtime_layers"].estimate_ascend_dp_operations(metrics)
        if abs(sum(entry[1] for entry in operations) - predictions["runtime_layers"]) > 1e-7:
            raise ValueError("Canonical group timing and decomposition diverge")
        if abs(predictions["runtime_layers"] - predictions["isolated_kernels"]) > 1e-7:
            raise ValueError("Pure-decode runtime profiles leaked into a prefill-containing group")
        measured = max(row["elapsed_ms"] for row in forward.values())
        comparison = controls.get(geometry(forward.values()), [])
        routing = []
        for layer in (3, 4, 5, 6, 77):
            dispatches = [{"rank": rank, **part} for rank, row in sorted(group.items()) for part in row["boundaries"]
                          if part["module"] == "DeepEPDispatcher.dispatch" and part["layer"] == layer]
            if dispatches:
                if len(dispatches) != 16:
                    raise ValueError("Incomplete expert-routing observation")
                counts = [sum(part["num_recv_tokens_per_expert"]) for part in dispatches]
                routing.append({"layer": layer, "received_by_rank": counts, "mean_received": statistics.mean(counts),
                                "max_received": max(counts), "max_to_mean": max(counts) / statistics.mean(counts),
                                "rank_details": dispatches})
        output.append({"case": case, "model_call": call, "forward_iteration": matching[0], "metrics": metrics,
                       "global_physical_tokens": group[0]["global_num_tokens_cpu"], "measured_forward_ms": measured,
                       "prediction_ms": predictions, "prediction_error_percent": 100 * (predictions["runtime_layers"] / measured - 1),
                       "native_operations": operations, "critical_active_prefill_rank": critical, "all_rank_boundaries": rank_parts,
                       "modeled_moe_boundary_ms": sum(entry[1] for entry in operations if entry[0] in (
                           "context_router", "context_moe_overlap", "context_shared_add")),
                       "matched_control_ms": comparison,
                       "event_vs_control_percent": None if not comparison else 100 * (measured / statistics.median(comparison) - 1),
                       "routing": routing,
                       "sampled_suboperations": {str(rank): [part for part in row["boundaries"] if "." in part["module"]]
                                                 for rank, row in sorted(group.items())}})
    report = {"scope": "Observed-batch attribution, not an independent request forecast or a calibration dataset",
              "campaign": str(root), "source_hashes": sources, "configurations": configs, "groups": output,
              "selection": "Third repeat event forwards; second repeat controls of identical geometry. Earlier cold/first-touch samples retained in raw evidence, not used as warmed accuracy claims.",
              "interpretation": "Outer/inner spans overlap. Idle-rank MoE includes remote attention wait. Never add maxima from different ranks or multiply these inclusive spans into a serving residual."}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"groups": len(output), "output": str(args.output)}))


if __name__ == "__main__":
    main()
