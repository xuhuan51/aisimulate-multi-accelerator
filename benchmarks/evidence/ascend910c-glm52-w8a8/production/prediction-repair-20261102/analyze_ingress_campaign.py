import argparse
import gzip
import hashlib
import json
import statistics
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT / "pd-ingress-1p1d"


def records(folder, pattern):
    rows = []
    for path in sorted(folder.glob(pattern)):
        raw = path.read_bytes()
        if path.suffix == ".gz":
            if path.with_suffix("").exists():
                continue
            raw = gzip.decompress(raw)
        rows.extend(json.loads(line) for line in raw.splitlines())
    return rows


def summarize(values):
    return {"count": len(values), "min_ms": min(values), "mean_ms": statistics.mean(values), "median_ms": statistics.median(values), "max_ms": max(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    progress = json.loads((CAMPAIGN / "progress.json").read_text())
    if progress["status"] != "completed" and not args.allow_partial:
        raise ValueError("Use --allow-partial for incomplete diagnostic snapshots")
    client_rows = records(CAMPAIGN / "prefill", "client-requests-*.jsonl*")
    tokenizer = {row["rid"]: row for row in records(CAMPAIGN / "prefill", "tokenizer-ingress-*.jsonl*")}
    decoder = CAMPAIGN / "decode"
    if not decoder.exists():
        decoder = CAMPAIGN / "decode-live"
    stages = {role: {row["rid"]: row for row in records(folder, "request-stages-*.jsonl*") if row["event"] == "set_completion_time"} for role, folder in (("prefill", CAMPAIGN / "prefill"), ("decode", decoder))}
    cases = []
    for case in progress["cases"]:
        if case["status"] != "completed":
            continue
        clients = sorted((row for row in client_rows if case["start_unix"] <= row["wrapper_start_unix"] <= case["end_unix"] and row["requested_output_tokens"] == case["output_length"]), key=lambda row: row["request_start_perf"])
        if len(clients) != case["requests"] or any(not row["success"] or row["input_tokens"] != case["input_length"] or row["output_tokens"] != case["output_length"] for row in clients):
            raise ValueError(f"Incomplete client trace: {case['name']}")
        if any(row["rid"] not in stages[role] for row in clients for role in stages):
            if args.allow_partial:
                continue
            raise ValueError(f"Incomplete remote snapshot: {case['name']}")
        details = []
        trace = []
        reference = clients[0]["request_start_perf"]
        for client in clients:
            request_id = client["rid"]
            prefill, decode = stages["prefill"][request_id], stages["decode"][request_id]
            api = tokenizer[request_id]
            source, destination, front = prefill["times"], decode["times"], api["times"]
            if prefill["bootstrap_room"] != decode["bootstrap_room"] or prefill["bootstrap_room"] != api["bootstrap_room"]:
                raise ValueError(f"Cross-role identity mismatch: {request_id}")
            start = client["request_start_perf"]
            visible = start + client["ttft_seconds"]
            native_id = str(uuid.UUID(request_id.removeprefix("aisim-")))
            record = {
                "rid": request_id, "native_uuid": native_id, "bootstrap_room": prefill["bootstrap_room"], "prefill_dp_rank": prefill["dp_rank"], "decode_dp_rank": decode["dp_rank"],
                "arrival_ms": 1000 * (start - reference), "client_ttft_ms": 1000 * client["ttft_seconds"],
                "client_to_tokenizer_ms": 1000 * (front["created_time"] - start),
                "tokenizer_and_dispatch_ms": 1000 * (front["api_server_dispatch_finish_time"] - front["created_time"]),
                "dispatch_to_scheduler_ms": 1000 * (source["scheduler_recv_time"] - front["api_server_dispatch_finish_time"]),
                "prefill_bootstrap_ms": 1000 * (source["bootstrap_done_time"] - source["scheduler_recv_time"]),
                "prefill_admission_ms": 1000 * (source["forward_entry_time"] - source["bootstrap_done_time"]),
                "prefill_compute_interval_ms": 1000 * (source["prefill_transfer_queue_entry_time"] - source["forward_entry_time"]),
                "prefill_source_hold_ms": 1000 * (source["completion_time"] - source["prefill_transfer_queue_entry_time"]),
                "prefill_response_to_text_ms": 1000 * (visible - source["completion_time"]),
                "decode_bootstrap_ms": 1000 * (destination["bootstrap_done_time"] - destination["scheduler_recv_time"]),
                "decode_allocation_metadata_ms": 1000 * (destination["decode_transfer_queue_entry_time"] - destination["bootstrap_done_time"]),
                "decode_transfer_wait_ms": 1000 * (destination["wait_queue_entry_time"] - destination["decode_transfer_queue_entry_time"]),
                "decode_ready_to_prebuilt_ms": 1000 * (destination["decode_prebuilt_finish_time"] - destination["wait_queue_entry_time"]),
                "first_nonempty_text_token_index": client["output_tokens"] - len(client["itl_seconds"]),
                "sources": {"client": client, "tokenizer": api, "prefill": prefill, "decode": decode},
            }
            if record["prefill_response_to_text_ms"] < -2:
                raise ValueError(f"Client text precedes P response completion: {request_id}")
            if not 1 <= record["first_nonempty_text_token_index"] <= client["output_tokens"]:
                raise ValueError("Invalid first-visible-text token count")
            components = ("client_to_tokenizer_ms", "tokenizer_and_dispatch_ms", "dispatch_to_scheduler_ms", "prefill_bootstrap_ms", "prefill_admission_ms", "prefill_compute_interval_ms", "prefill_source_hold_ms", "prefill_response_to_text_ms")
            if abs(sum(record[key] for key in components) - record["client_ttft_ms"]) > 0.001:
                raise ValueError("TTFT decomposition does not close")
            details.append(record)
            blocks = (client["input_tokens"] + 127) // 128
            trace.append({"request_id": native_id, "timestamp": record["arrival_ms"], "input_length": client["input_tokens"], "output_length": client["output_tokens"], "hash_ids": [len(trace) * blocks + block + 1 for block in range(blocks)]})
        measured = json.loads((CAMPAIGN / "prefill" / f"{case['name']}.jsonl").read_text().splitlines()[-1])
        if abs(statistics.mean(row["client_ttft_ms"] for row in details) - measured["mean_ttft_ms"]) > 0.001:
            raise ValueError("Client trace does not match the saved benchmark")
        trace_path = CAMPAIGN / case["name"] / "client-arrivals.jsonl"
        trace_path.write_text("".join(json.dumps(row) + "\n" for row in trace))
        summary = {key: summarize([row[key] for row in details]) for key in details[0] if key.endswith("_ms") and key != "arrival_ms"}
        result = {"case": case["name"], "requests": case["requests"], "stages": summary, "details": details, "client_arrival_trace": str(trace_path.relative_to(ROOT)), "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest()}
        cases.append(result)
        print(json.dumps({"case": case["name"], "mean_ms": {key: row["mean_ms"] for key, row in summary.items()}}), flush=True)
    calibration = next(case for case in cases if case["case"] == "diagnostic-c1-8k128")
    info = json.loads((CAMPAIGN / "prefill-server-info.json").read_text())
    config = {"workers": info["tokenizer_worker_num"], "base_ms": calibration["stages"]["tokenizer_and_dispatch_ms"]["median_ms"], "per_input_token_us": 0.0}
    identity = hashlib.sha256(json.dumps([row["sources"]["tokenizer"] for row in calibration["details"]], sort_keys=True).encode()).hexdigest()
    profile = {"calibration_id": f"pd-prefill-tokenizer-c1-{identity[:16]}", "input_tokens": 8192, "payload": "pretokenized input_ids; pinned benchmark --tokenize-prompt", "config": config, "scope": "Median of two formal C1 API created-to-dispatch intervals; no per-token slope fitted, no HTTP/cold bootstrap delay included; other cases not fitted."}
    target = ROOT / "prefill-tokenizer-8k-profile.json"
    encoded = json.dumps(profile, indent=2) + "\n"
    if target.exists() and target.read_text() != encoded:
        raise ValueError("Do not overwrite an existing tokenizer calibration")
    target.write_text(encoded)
    (ROOT / "pd-ingress-timeline-analysis.json").write_text(json.dumps({"status": progress["status"], "scope": "Explicit RID links client/P/D. TTFT decomposition uses node20 monotonic clock only; Decode intervals use node25 local monotonic clock. Measured arrival traces are diagnostic inputs, not independent closed-loop search validation.", "tokenizer_profile": profile, "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    main()
