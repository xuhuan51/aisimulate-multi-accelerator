import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path

from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel
from validate_context_transfer import configuration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.campaign / "context-holdouts.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("All predeclared holdouts must finish")
    groups = collections.defaultdict(dict)
    hashes = {}
    for rank in range(4):
        path = args.campaign / f"paired-forward-rank{rank}.jsonl"
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            row = json.loads(line)
            group = groups[row["timed_forward_iteration"]]
            if rank in group:
                raise ValueError("Duplicate timed forward")
            group[rank] = row
    results = []
    for scope in ("isolated_kernels", "runtime_layers"):
        config = configuration(args.model, scope, "anchored")
        config["worker_type"] = "aggregated"
        model = RustForwardPassPerfModel.best_available(config)
        for case in manifest["cases"]:
            samples = []
            for iteration, rows in sorted(groups.items()):
                if len(rows) != 4:
                    continue
                values = [rows[rank] for rank in range(4)]
                if any(not case["start_unix"] <= row["launch_unix"] <= case["end_unix"] for row in values):
                    continue
                if any(row["forward_mode"] != "DECODE" for row in values):
                    continue
                requests = [request for row in values for request in row["requests"]]
                if (len(requests) != case["concurrency"]
                        or any(request["prompt_tokens"] != case["input_length"] or request["output_tokens"] < 16
                               for request in requests)):
                    continue
                if any(row["scheduled_requests"]["num_decode_requests"] != case["concurrency"] // 4 for row in values):
                    continue
                measured = max(row["elapsed_ms"] for row in values)
                batch = [{"scheduled_requests": row["scheduled_requests"]} for row in values]
                prediction = model.estimate_forward_pass_time_ms(batch)
                samples.append({"iteration": iteration, "measured_ms": measured, "predicted_ms": prediction,
                                "error_pct": 100 * (prediction / measured - 1),
                                "scheduled_requests": [row["scheduled_requests"] for row in values]})
            if not samples:
                raise ValueError(f"No complete steady forward groups: {case['id']}")
            results.append({"case": case["id"], "timing_scope": scope, "config": config,
                            "forward_groups": len(samples),
                            "measured_mean_ms": statistics.mean(row["measured_ms"] for row in samples),
                            "predicted_mean_ms": statistics.mean(row["predicted_ms"] for row in samples),
                            "mean_error_pct": statistics.mean(row["error_pct"] for row in samples),
                            "mape_pct": statistics.mean(abs(row["error_pct"]) for row in samples),
                            "samples": samples})
    args.output.write_text(json.dumps({"scope": "Paired steady decode forward validation; actual schedules are conditional inputs, not predicted request latency",
                                       "source_sha256": hashes, "results": results}, indent=2) + "\n")
    for row in results:
        print(json.dumps({key: value for key, value in row.items() if key not in ("samples", "config")}))


if __name__ == "__main__":
    main()
