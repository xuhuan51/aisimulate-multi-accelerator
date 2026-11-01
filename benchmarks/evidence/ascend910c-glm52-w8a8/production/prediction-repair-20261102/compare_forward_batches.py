import argparse
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path


def records(folder, pattern):
    for path in sorted(folder.glob(pattern)):
        if path.suffix != ".gz" and path.with_suffix(path.suffix + ".gz").exists():
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            for line in stream:
                yield json.loads(line)


def record_batch(rows, role, rank, prefill, decode, duration):
    if prefill + decode == 0:
        return
    phase = "mixed" if prefill and decode else "prefill" if prefill else "decode"
    row = rows.setdefault((role, rank, phase), {
        "stage": role, "worker_id": 0, "dp_rank": rank, "phase": phase,
        "passes": 0, "request_instances": 0, "max_batch_size": 0,
        "duration_ms": 0.0, "request_duration_ms": 0.0, "batch_histogram": collections.Counter(),
    })
    batch = prefill + decode
    row["passes"] += 1
    row["request_instances"] += batch
    row["max_batch_size"] = max(row["max_batch_size"], batch)
    row["duration_ms"] += duration
    row["request_duration_ms"] += duration * batch
    row["batch_histogram"][batch] += 1


def measured_batches(grouped, identities, expected_ranks):
    summaries = {}
    formal_summaries = {}
    background = []
    for role, groups in grouped.items():
        for iteration, ranks in groups.items():
            request_ids = {request["rid"] for row in ranks.values() for request in row["requests"]}
            if not request_ids.intersection(identities):
                continue
            unknown = request_ids - identities
            if any(not request_id.startswith("HEALTH_CHECK_") for request_id in unknown):
                raise ValueError(f"Mixed formal/warmup traffic: {role}, {iteration}")
            if unknown:
                background.append({"stage": role, "timed_forward_iteration": iteration,
                                   "request_ids": sorted(unknown)})
            if set(ranks) != set(range(expected_ranks[role])):
                raise ValueError(f"Incomplete timed forward group: {role}, {iteration}")
            prefill_total = decode_total = 0
            formal_prefill = formal_decode = 0
            for rank, row in ranks.items():
                scheduled = row["scheduled_requests"]
                prefill, decode = scheduled["num_prefill_requests"], scheduled["num_decode_requests"]
                if prefill + decode != len(row["requests"]):
                    raise ValueError(f"Request identity/shape mismatch: {role}, {iteration}, {rank}")
                prefill_total += prefill
                decode_total += decode
                record_batch(summaries, role, rank, prefill, decode, row["elapsed_ms"])
                formal_count = sum(request["rid"] in identities for request in row["requests"])
                if prefill and decode and unknown:
                    raise ValueError("Mixed-phase health traffic cannot be split by phase from this collector")
                kept_prefill = prefill if not unknown else formal_count if prefill else 0
                kept_decode = decode if not unknown else formal_count if decode else 0
                formal_prefill += kept_prefill
                formal_decode += kept_decode
                record_batch(formal_summaries, role, rank, kept_prefill, kept_decode, row["elapsed_ms"])
            record_batch(summaries, role, None, prefill_total, decode_total,
                         max(row["elapsed_ms"] for row in ranks.values()))
            record_batch(formal_summaries, role, None, formal_prefill, formal_decode,
                         max(row["elapsed_ms"] for row in ranks.values()))
    return list(summaries.values()), list(formal_summaries.values()), background


def metrics(row):
    return {"mean_batch": row["request_instances"] / row["passes"],
            "duration_weighted_mean_batch": row["request_duration_ms"] / row["duration_ms"],
            "peak_batch": row["max_batch_size"], "forward_passes": row["passes"],
            "request_instances": row["request_instances"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    campaign, predictions = args.campaign.resolve(), args.predictions.resolve()
    progress = json.loads((campaign / "progress.json").read_text())
    clients = list(records(campaign / "prefill", "client-requests-*.jsonl*"))
    grouped = {}
    for role in ("prefill", "decode"):
        grouped[role] = collections.defaultdict(dict)
        for row in records(campaign / role, "paired-forward-rank*.jsonl*"):
            group = grouped[role][row["timed_forward_iteration"]]
            if row["dp_rank"] in group:
                raise ValueError("Duplicate rank record")
            group[row["dp_rank"]] = row
    evidence, comparison, unavailable = [], [], []
    for case in progress["cases"]:
        prediction_path = predictions / case["name"] / "result/prediction.json"
        if case["status"] != "completed" or not prediction_path.exists():
            continue
        formal = [row for row in clients if case["start_unix"] <= row["wrapper_start_unix"] <= case["end_unix"]
                  and row["requested_output_tokens"] == case["output_length"]]
        identities = {row["rid"] for row in formal}
        if len(identities) != case["requests"] or any(not row["success"] or row["input_tokens"] != case["input_length"]
                                                    or row["output_tokens"] != case["output_length"] for row in formal):
            raise ValueError(f"Invalid formal client trace: {case['name']}")
        native = json.loads(prediction_path.read_text())
        predicted = native["forward_pass_batches"]
        runtime_path = campaign / case["name"] / "runtime-config.json"
        runtime = json.loads(runtime_path.read_text())
        ranks = {node["effective"]["disaggregation_mode"]: node["effective"]["attn_dp_size"]
                 for node in runtime["nodes"]}
        if len(runtime["nodes"]) != 2 or set(ranks) != {"prefill", "decode"}:
            raise ValueError("This paired collector requires one worker per role")
        if any(row["worker_id"] != 0 for row in predicted):
            raise ValueError("Prediction is not a one-worker-per-role deployment")
        try:
            measured, measured_formal, health_groups = measured_batches(grouped, identities, ranks)
        except ValueError as error:
            if not args.allow_partial:
                raise
            unavailable.append({"case": case["name"], "reason": str(error),
                                "qualification": "Serving latency/throughput remains valid; complete grouped-forward capture is unavailable."})
            continue
        key = lambda row: (row["stage"], row["worker_id"], row["dp_rank"], row["phase"])
        measured_rows, predicted_rows = ({key(row): row for row in source} for source in (measured, predicted))
        expected_decode_instances = case["requests"] * (case["output_length"] - 1)
        for source in (measured_formal, predicted):
            instances = sum(row["request_instances"] for row in source if row["stage"] == "decode" and row["dp_rank"] is None)
            if instances != expected_decode_instances:
                raise ValueError(f"Lost or duplicate decode observations: {case['name']}, {instances}")
        for identity in sorted(measured_rows.keys() | predicted_rows.keys(), key=str):
            actual, estimate = measured_rows.get(identity), predicted_rows.get(identity)
            row = {"case": case["name"], "concurrency": case["concurrency"], "stage": identity[0],
                   "worker_id": identity[1], "dp_rank": identity[2], "phase": identity[3],
                   "status": "paired" if actual and estimate else "unmatched_rank"}
            actual_metrics = metrics(actual) if actual else {}
            predicted_metrics = metrics(estimate) if estimate else {}
            for name in sorted(actual_metrics.keys() | predicted_metrics.keys()):
                row[f"measured_{name}"] = actual_metrics.get(name)
                row[f"predicted_{name}"] = predicted_metrics.get(name)
                row[f"error_percent_{name}"] = (100 * (predicted_metrics[name] / actual_metrics[name] - 1)
                                               if name in actual_metrics and name in predicted_metrics else None)
            comparison.append(row)
        evidence.append({"case": case["name"], "formal_request_ids": sorted(identities),
                         "measured": measured, "measured_formal_only": measured_formal,
                         "overlapping_health_checks": health_groups, "predicted": predicted,
                         "prediction_path": str(prediction_path),
                         "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest()})
    if not evidence:
        raise ValueError("No completed matched cases")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps({
        "campaign": str(campaign), "method": "committed forward batches; exact RID and timed-forward/rank joins",
        "qualification": "Measured elapsed time is the paired device timer; model duration includes its explicit timing profile. "
                         "Unmatched rank rows are retained: warmup rotates user placement, and health checks use an independent rank counter. "
                         "Overlapping health checks stay in the measured running batch; formal-only summaries and health identities are also retained.",
        "cases": evidence, "comparison": comparison, "unavailable_batch_comparisons": unavailable,
    }, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparison)
    for row in comparison:
        if row["dp_rank"] is None:
            print(json.dumps(row))


if __name__ == "__main__":
    main()
