import argparse
import collections
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
        raise ValueError("Complete held-out measurements are required")
    groups = collections.defaultdict(dict)
    for path in args.campaign.glob("paired-forward-rank*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            groups[row["timed_forward_iteration"]][row["dp_rank"]] = row
    arrivals = {row["rid"]: row for path in args.campaign.glob("tokenizer-ingress-pid*.jsonl")
                for row in (json.loads(line) for line in path.read_text().splitlines())}
    completions = {row["rid"]: row for path in args.campaign.glob("request-stages-rank*.jsonl")
                   for row in (json.loads(line) for line in path.read_text().splitlines())
                   if row["event"] == "set_completion_time"}
    config = configuration(args.model, "runtime_layers", "anchored")
    config["worker_type"] = "aggregated"
    model = RustForwardPassPerfModel.best_available(config)
    results = []
    for case in manifest["cases"]:
        selected = {request["rid"] for rows in groups.values() for row in rows.values()
                    if case["start_unix"] <= row["launch_unix"] <= case["end_unix"]
                    for request in row["requests"] if request["output_tokens"] >= case["output_length"] - 1
                    and request["prompt_tokens"] == case["input_length"]}
        if len(selected) != case["prompts"]:
            raise ValueError("Formal request identity mismatch")
        records = []
        for iteration, rows in sorted(groups.items()):
            request_ids = {request["rid"] for row in rows.values() for request in row["requests"]}
            if not request_ids.intersection(selected):
                continue
            if not any(row["scheduled_requests"]["num_prefill_requests"] for row in rows.values()):
                continue
            if len(rows) != 4 or request_ids - selected:
                raise ValueError("Incomplete or mixed-background formal forward group")
            batch = [{"scheduled_requests": rows[rank]["scheduled_requests"]} for rank in range(4)]
            prefill = sum(row["scheduled_requests"]["num_prefill_requests"] for row in batch)
            decode = sum(row["scheduled_requests"]["num_decode_requests"] for row in batch)
            if not prefill:
                continue
            measured = max(row["elapsed_ms"] for row in rows.values())
            predicted = model.estimate_forward_pass_time_ms(batch)
            records.append({"iteration": iteration, "phase": "mixed" if decode else "prefill",
                            "measured_ms": measured, "predicted_ms": predicted,
                            "error_pct": 100 * (predicted / measured - 1), "metrics": batch})
        timings = []
        origin = min(arrivals[rid]["send_call_start_monotonic"] for rid in selected)
        for rid in sorted(selected, key=lambda rid: arrivals[rid]["send_call_start_monotonic"]):
            sent = arrivals[rid]
            finish = completions[rid]
            times = finish["times"]
            timings.append({"rid": rid, "dp_rank": finish["dp_rank"],
                            "external_send_offset_ms": 1000 * (sent["send_call_start_monotonic"] - origin),
                            "scheduler_recv_offset_ms": 1000 * (times["scheduler_recv_time"] - origin),
                            "send_to_receive_ms": 1000 * (times["scheduler_recv_time"] - sent["send_call_start_monotonic"]),
                            "receive_to_prefill_done_ms": 1000 * (times["prefill_finished_time"] - times["scheduler_recv_time"]),
                            "api_creation_to_prefill_done_ms": 1000 * (times["prefill_finished_time"] - sent["times"]["created_time"])})
        results.append({"case": case["id"], "requests": timings, "prefill_and_mixed_forwards": records,
                        "prefill_and_mixed_mape_pct": statistics.mean(abs(row["error_pct"]) for row in records)})
    args.output.write_text(json.dumps({"scope": "Diagnostic decomposition using measured request identities and real batch shapes; does not inject observations into the predictor", "config": config, "cases": results}, indent=2) + "\n")
    for case in results:
        print(json.dumps({"case": case["case"], "prefill_and_mixed_mape_pct": case["prefill_and_mixed_mape_pct"],
                          "mean_send_to_receive_ms": statistics.mean(row["send_to_receive_ms"] for row in case["requests"])}))


if __name__ == "__main__":
    main()
