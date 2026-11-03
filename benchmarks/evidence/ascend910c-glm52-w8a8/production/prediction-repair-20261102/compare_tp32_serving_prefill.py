import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path

from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

from compare_runtime_layer_prediction import configuration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serving", type=Path, required=True)
    parser.add_argument("--telemetry-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert-workload", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    content = (args.serving / "campaign.json").read_bytes()
    cases = [case for case in json.loads(content)["cases"] if case["status"] == "completed"]
    groups, sources = collections.defaultdict(dict), {str(args.serving / "campaign.json"): hashlib.sha256(content).hexdigest()}
    for folder in args.telemetry_dir:
        for path in folder.glob("paired-forward-rank*.jsonl"):
            content = path.read_bytes()
            sources[str(path)] = hashlib.sha256(content).hexdigest()
            for line in content.splitlines():
                row = json.loads(line)
                group = groups[row["timed_forward_iteration"]]
                if row["dp_rank"] in group:
                    raise ValueError("Duplicate paired rank/iteration")
                group[row["dp_rank"]] = row
    config = configuration("runtime_layers", 1, 32, 32, True)
    config["worker_type"] = "aggregated"
    if args.expert_workload:
        config["estimator_config"]["op_level"]["ascend"]["prefill_expert_workload"] = json.loads(args.expert_workload.read_text())
        sources[str(args.expert_workload)] = hashlib.sha256(args.expert_workload.read_bytes()).hexdigest()
    model = RustForwardPassPerfModel.best_available(config)
    results = []
    for case in cases:
        selected = [(iteration, [ranks[rank] for rank in range(32)]) for iteration, ranks in sorted(groups.items())
                    if len(ranks) == 32 and all(case["start_unix"] <= row["launch_unix"] <= case["end_unix"] for row in ranks.values())]
        formal = {request["rid"] for _, rows in selected for row in rows for request in row["requests"]
                  if request["output_tokens"] > 64}
        if len(formal) != case["prompts"]:
            raise ValueError("Cannot identify all formal requests independently of warmup")
        forwards = []
        for iteration, rows in selected:
            if not any(row["scheduled_requests"]["sum_prefill_tokens"] for row in rows):
                continue
            requests = [request for row in rows for request in row["requests"]]
            if any(request["rid"] not in formal for request in requests):
                continue
            metrics = [{"scheduled_requests": row["scheduled_requests"]} for row in rows]
            measured = max(row["elapsed_ms"] for row in rows)
            forward = {"iteration": iteration, "metrics": metrics, "measured_ms": measured,
                       "mixed": any(row["scheduled_requests"]["num_decode_requests"] for row in rows),
                       "launch_unix": min(row["launch_unix"] for row in rows),
                       "requests_by_rank": [row["requests"] for row in rows]}
            try:
                predicted = model.estimate_forward_pass_time_ms(metrics)
                forward.update(predicted_ms=predicted, error_percent=(predicted / measured - 1) * 100,
                               operations=model.estimate_ascend_dp_operations(metrics))
            except ValueError as error:
                forward["coverage_gap"] = str(error)
            forwards.append(forward)
        summaries = []
        for mixed in (False, True):
            subset = [row for row in forwards if row["mixed"] == mixed and "predicted_ms" in row]
            if subset:
                summaries.append({"mixed": mixed, "rounds": len(subset),
                    "measured_ms": statistics.mean(row["measured_ms"] for row in subset),
                    "predicted_ms": statistics.mean(row["predicted_ms"] for row in subset),
                    "mape_percent": statistics.mean(abs(row["error_percent"]) for row in subset)})
        result = {"case": case["id"], "formal_requests": len(formal), "summaries": summaries, "forwards": forwards}
        results.append(result)
        print(json.dumps({key: result[key] for key in ("case", "formal_requests", "summaries")}), flush=True)
    args.output.write_text(json.dumps({"scope": "Observed-batch operator attribution, not independent scheduler prediction or fitted correction; warmups excluded by completed formal request IDs", "configuration": config, "source_sha256": sources, "cases": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
