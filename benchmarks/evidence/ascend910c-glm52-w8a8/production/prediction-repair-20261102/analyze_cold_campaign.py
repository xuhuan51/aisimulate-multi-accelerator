import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path

from analyze_ingress_campaign import records


ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT / "pd-cold-1p1d"


def mean(rows, key):
    return statistics.mean(row[key] for row in rows) if rows else None


def main():
    global CAMPAIGN
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN)
    parser.add_argument("--output", type=Path, default=ROOT / "pd-cold-timeline-analysis.json")
    args = parser.parse_args()
    CAMPAIGN = args.campaign.resolve()
    progress = json.loads((CAMPAIGN / "progress.json").read_text())
    if progress["status"] != "completed" and not args.allow_partial:
        raise ValueError("Complete the cold-start controls before final analysis")
    client = sorted(records(CAMPAIGN / "prefill", "client-requests-*.jsonl*"), key=lambda row: row["request_start_perf"])
    decode_dir = CAMPAIGN / "decode"
    if not decode_dir.exists():
        decode_dir = CAMPAIGN / "decode-live"
    stage = {role: {row["rid"]: row for row in records(folder, "request-stages-*.jsonl*") if row["event"] == "set_completion_time"} for role, folder in (("prefill", CAMPAIGN / "prefill"), ("decode", decode_dir))}
    host = records(CAMPAIGN / "prefill", "host-forward-*.jsonl*")
    npu = {(row["dp_rank"], row["forward_iteration"]): row for row in records(CAMPAIGN / "prefill", "paired-forward-*.jsonl*")}
    history = {role: collections.Counter() for role in stage}
    counts = {}
    for request in client:
        if any(request["rid"] not in stage[role] for role in stage):
            continue
        counts[request["rid"]] = {role: history[role][stage[role][request["rid"]]["dp_rank"]] for role in stage}
        for role in stage:
            history[role][stage[role][request["rid"]]["dp_rank"]] += 1
    cases = []
    for case in progress["cases"]:
        if case["status"] != "completed":
            continue
        formal = [row for row in client if case["start_unix"] <= row["wrapper_start_unix"] <= case["end_unix"] and row["requested_output_tokens"] == case["output_length"]]
        if len(formal) != case["requests"] or any(not row["success"] or row["input_tokens"] != case["input_length"] or row["output_tokens"] != case["output_length"] for row in formal):
            raise ValueError(f"Invalid completed client trace: {case['name']}")
        if any(row["rid"] not in counts for row in formal):
            if args.allow_partial:
                continue
            raise ValueError("Incomplete remote final stages")
        rows = []
        for request in formal:
            rid = request["rid"]
            source, destination = stage["prefill"][rid], stage["decode"][rid]
            prefill, decode = source["times"], destination["times"]
            if source["bootstrap_room"] != destination["bootstrap_room"]:
                raise ValueError("PD request identity mismatch")
            forwards = [row for row in host if row["dp_rank"] == source["dp_rank"] and any(item["rid"] == rid for item in row["requests"])]
            measured = [npu.get((row["dp_rank"], row["forward_iteration"])) for row in forwards]
            if not forwards or any(not row["succeeded"] for row in forwards):
                raise ValueError("Missing or failed host forward")
            elapsed = 1000 * (prefill["prefill_transfer_queue_entry_time"] - prefill["forward_entry_time"])
            host_elapsed = sum(row["host_run_batch_ms"] for row in forwards)
            rows.append({"rid": rid, "prefill_dp_rank": source["dp_rank"], "decode_dp_rank": destination["dp_rank"], "previous_client_requests_by_rank": counts[rid], "ttft_ms": request["ttft_seconds"] * 1000, "first_visible_text_token": request["output_tokens"] - len(request["itl_seconds"]), "prefill_interval_ms": elapsed, "host_forward_total_ms": host_elapsed, "prefill_outside_run_batch_ms": elapsed - host_elapsed, "npu_forward_total_ms": sum(row["elapsed_ms"] for row in measured) if all(measured) else None, "prefill_bootstrap_ms": 1000 * (prefill["bootstrap_done_time"] - prefill["scheduler_recv_time"]), "decode_allocation_metadata_ms": 1000 * (decode["decode_transfer_queue_entry_time"] - decode["bootstrap_done_time"]), "source_rows": {"client": request, "prefill": source, "decode": destination, "host_forwards": forwards, "npu_forwards": measured}})
        fresh = [row for row in rows if 0 in row["previous_client_requests_by_rank"].values()]
        reused = [row for row in rows if row not in fresh]
        summary = {"case": case["name"], "input_length": case["input_length"], "requests": len(rows), "mean_ttft_ms": mean(rows, "ttft_ms"), "first_seen_rank_requests": len(fresh), "first_seen_rank_mean_ttft_ms": mean(fresh, "ttft_ms"), "repeated_rank_requests": len(reused), "repeated_rank_mean_ttft_ms": mean(reused, "ttft_ms"), "max_prefill_outside_run_batch_ms": max(row["prefill_outside_run_batch_ms"] for row in rows), "rows": rows}
        cases.append(summary)
        print(json.dumps({key: value for key, value in summary.items() if key != "rows"}), flush=True)
    tokenizer = {row["rid"]: row for row in records(CAMPAIGN / "prefill", "tokenizer-ingress-*.jsonl*")}
    long_case = next((case for case in progress["cases"] if case["input_length"] == 32768), None)
    if long_case and long_case["status"] == "completed":
        warmups = [row for row in client if row["wrapper_start_unix"] >= long_case["start_unix"] and row["requested_output_tokens"] == 32 and row["requested_input_tokens"] == 32768]
        if len(warmups) == 2 and all(row["success"] and row["input_tokens"] == 32768 and row["output_tokens"] == 32 for row in warmups):
            sources = [tokenizer[row["rid"]] for row in warmups]
            durations = [1000 * (row["times"]["api_server_dispatch_finish_time"] - row["times"]["created_time"]) for row in sources]
            identity = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
            profile = {"calibration_id": f"pd-prefill-tokenizer-32k-warmup-{identity[:16]}", "input_tokens": 32768, "payload": "pretokenized input_ids", "config": {"workers": 1, "base_ms": statistics.median(durations), "per_input_token_us": 0.0}, "scope": "Two32K warmup API created-to-dispatch intervals only; no formal request or full TTFT used for fitting.", "source_rows": sources}
            target = ROOT / "prefill-tokenizer-32k-profile.json"
            encoded = json.dumps(profile, indent=2) + "\n"
            if target.exists() and target.read_text() != encoded:
                raise ValueError("Do not change a saved calibration")
            target.write_text(encoded)
    args.output.write_text(json.dumps({"status": progress["status"], "scope": "Client RID distinguishes first observed benchmark visits from repeated visits; includes benchmark warmups in rank history, not arbitrary initialization/health internals. Host run_batch is timed without added synchronization; gaps outside it are observations, not fitted cold penalties.", "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    main()
