import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path

from compare_runtime_layer_prediction import configuration
from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.campaign
    manifest = json.loads((root / "full-request-validation/campaign.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Full requests must finish before attribution")
    groups, sources = collections.defaultdict(dict), {}
    ingress, received = {}, {}
    for path in sorted(root.glob("paired-forward-rank*.jsonl")):
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["dp_rank"] in groups[row["timed_forward_iteration"]]:
                raise ValueError("Duplicate paired forward rank")
            groups[row["timed_forward_iteration"]][row["dp_rank"]] = row
    for path in sorted(root.glob("tokenizer-ingress-pid*.jsonl")):
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            row = json.loads(line)
            ingress[row["rid"]] = row
    for path in sorted(root.glob("request-stages-rank*.jsonl")):
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["event"] == "scheduler_received":
                received[row["rid"]] = row
    configurations = {scope: configuration(scope) for scope in ("isolated_kernels", "runtime_layers")}
    for config in configurations.values():
        config["worker_type"] = "aggregated"
    models = {scope: RustForwardPassPerfModel.best_available(config) for scope, config in configurations.items()}
    results = []
    for case in manifest["cases"]:
        selected = {iteration: rows for iteration, rows in sorted(groups.items())
                    if len(rows) == 4 and all(case["start_unix"] <= row["launch_unix"] <= case["end_unix"]
                                             for row in rows.values())}
        main_rids = {request["rid"] for rows in selected.values() for row in rows.values()
                     for request in row["requests"] if request["output_tokens"] > 64}
        if len(main_rids) != case["concurrency"] or not main_rids <= ingress.keys():
            raise ValueError("Cannot identify all completed main requests")
        origin = min(ingress[rid]["reported_unix"] for rid in main_rids)
        request_arrivals = [{"rid": rid, "tokenizer_dispatch_ms": (ingress[rid]["reported_unix"] - origin) * 1000,
                             "scheduler_received_ms": (received[rid]["reported_unix"] - origin) * 1000,
                             "dp_rank": received[rid]["dp_rank"]}
                            for rid in sorted(main_rids, key=lambda rid: ingress[rid]["reported_unix"])]
        timelines, steady = [], []
        for iteration, ranks in selected.items():
            rows = [ranks[rank] for rank in range(4)]
            requests = [request for row in rows for request in row["requests"]]
            if not requests or any(request["rid"] not in main_rids for request in requests):
                continue
            metrics = [{"scheduled_requests": row["scheduled_requests"]} for row in rows]
            has_prefill = any(row["scheduled_requests"]["sum_prefill_tokens"] > 0 for row in rows)
            has_decode = any(row["scheduled_requests"]["num_decode_requests"] > 0 for row in rows)
            if has_prefill or any(request["output_tokens"] <= 1 for request in requests):
                entry = {"iteration": iteration,
                         "launch_ms": (min(row["launch_unix"] for row in rows) - origin) * 1000,
                         "measured_ms": max(row["elapsed_ms"] for row in rows),
                         "mixed_prefill_decode": has_prefill and has_decode,
                         "metrics": metrics, "requests_by_dp": [row["requests"] for row in rows]}
                if has_prefill and has_decode:
                    entry["prediction_unavailable"] = "Mixed-rank prefill/decode boundary remains unsupported; no timing substituted"
                else:
                    entry["prediction_ms"] = models["runtime_layers"].estimate_forward_pass_time_ms(metrics)
                timelines.append(entry)
            if (has_prefill or len(requests) != case["concurrency"]
                    or any(not 64 <= request["output_tokens"] <= 1008 for request in requests)):
                continue
            entry = {"iteration": iteration, "measured_ms": max(row["elapsed_ms"] for row in rows)}
            for scope, model in models.items():
                entry[scope + "_ms"] = model.estimate_forward_pass_time_ms(metrics)
            steady.append(entry)
        if len(steady) < 800:
            raise ValueError("Incomplete full-length steady decode coverage")
        summary = {"paired_steady_groups": len(steady),
                   "measured_forward_median_ms": statistics.median(row["measured_ms"] for row in steady)}
        for scope in models:
            summary[scope + "_median_ms"] = statistics.median(row[scope + "_ms"] for row in steady)
            summary[scope + "_mape_percent"] = statistics.mean(
                abs(row[scope + "_ms"] / row["measured_ms"] - 1) * 100 for row in steady)
        measured_path = root / "full-request-validation" / case["id"] / "requests.jsonl"
        sources[str(measured_path.relative_to(root))] = hashlib.sha256(measured_path.read_bytes()).hexdigest()
        measured = json.loads(measured_path.read_text().splitlines()[-1])
        results.append({"case": case["id"], "concurrency": case["concurrency"], "request_arrivals": request_arrivals,
                        "client_ttfts_ms": [value * 1000 for value in measured["ttfts"]],
                        "prefill_timeline": timelines, "steady_summary": summary, "paired_decode": steady})
        print(json.dumps({"case": case["id"], **summary}), flush=True)
    args.output.write_text(json.dumps({"scope": "Causal diagnosis using observed batches; separate from independent scheduler replay",
                                      "configurations": configurations, "source_sha256": sources, "cases": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
