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
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = configuration("isolated_kernels")
    config["worker_type"] = "prefill"
    model = RustForwardPassPerfModel.best_available(config)
    groups, sources = collections.defaultdict(dict), {}
    cases = []
    for path in args.campaign.glob("chain-*.json"):
        manifest = json.loads(path.read_text())
        if manifest.get("status") == "completed":
            cases.extend(manifest["cases"])
            sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in args.campaign.glob("paired-forward-rank*.jsonl"):
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            row = json.loads(line)
            group = groups[row["timed_forward_iteration"]]
            if row["dp_rank"] in group:
                raise ValueError("Duplicate rank/iteration")
            group[row["dp_rank"]] = row
    results, skipped = [], collections.Counter()
    for iteration, ranks in sorted(groups.items()):
        if len(ranks) != 4:
            continue
        rows = [ranks[rank] for rank in range(4)]
        matching = [case for case in cases if all(case["start_unix"] <= row["launch_unix"] <= case["end_unix"] for row in rows)]
        if len(matching) != 1:
            skipped["outside_completed_benchmark_windows"] += 1
            continue
        metrics = [{"scheduled_requests": row["scheduled_requests"]} for row in rows]
        if not any(row["scheduled_requests"]["sum_prefill_tokens"] for row in rows):
            continue
        if any(row["scheduled_requests"]["num_decode_requests"] for row in rows):
            skipped["mixed_prefill_decode_dp_group"] += 1
            continue
        elapsed = max(row["elapsed_ms"] for row in rows)
        prediction = model.estimate_forward_pass_time_ms(metrics)
        results.append({"iteration": iteration, "metrics": metrics, "measured_ms": elapsed,
                        "case": matching[0]["id"],
                        "prediction_ms": prediction, "error_percent": (prediction / elapsed - 1) * 100,
                        "prompt_lengths": sorted({request["prompt_tokens"] for row in rows for request in row["requests"]})})
    grouped = collections.defaultdict(list)
    for result in results:
        active = sum(row["scheduled_requests"]["num_prefill_requests"] > 0 for row in result["metrics"])
        tokens = max(row["scheduled_requests"]["sum_prefill_tokens"] for row in result["metrics"])
        grouped[(active, tokens)].append(result)
    summaries = [{"active_dp": active, "max_tokens_per_dp": tokens, "count": len(rows),
                  "measured_ms": statistics.median(row["measured_ms"] for row in rows),
                  "prediction_ms": statistics.median(row["prediction_ms"] for row in rows),
                  "mape_percent": statistics.mean(abs(row["error_percent"]) for row in rows)}
                 for (active, tokens), rows in sorted(grouped.items())]
    args.output.write_text(json.dumps({"scope": "Pure prefill calls within completed benchmark windows; includes benchmark warmup, excludes service initialization; not TTFT",
                                      "configuration": config, "source_sha256": sources, "skipped": dict(skipped),
                                      "summary": summaries, "forwards": results}, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
