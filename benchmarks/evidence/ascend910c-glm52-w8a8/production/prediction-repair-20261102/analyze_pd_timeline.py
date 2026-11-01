import collections
import gzip
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT / "pd-paired-1p1d"


def read_rows(path):
    if not path.exists():
        path = path.with_suffix(path.suffix + ".gz")
    content = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return [json.loads(line) for line in content.splitlines()]


def interval(times, begin, end):
    if not times.get(begin) or not times.get(end):
        return None
    value = 1000 * (times[end] - times[begin])
    if value < -0.001:
        raise ValueError(f"Noncausal request interval {begin} -> {end}: {value}")
    return value


def summarize(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return {"count": len(values), "min_ms": min(values), "median_ms": statistics.median(values), "mean_ms": statistics.mean(values), "max_ms": max(values)}


def main():
    progress = json.loads((CAMPAIGN / "progress.json").read_text())
    folders = {"prefill": CAMPAIGN / "prefill", "decode": CAMPAIGN / "decode"}
    if not list(folders["decode"].glob("paired-forward-rank*.jsonl*")):
        folders["decode"] = CAMPAIGN / "decode-live"
    stages = {role: [row for rank in range(4) for row in read_rows(folder / f"request-stages-rank{rank}.jsonl")] for role, folder in folders.items()}
    forwards = [row for rank in range(4) for row in read_rows(folders["decode"] / f"paired-forward-rank{rank}.jsonl")]
    cases = []
    for case in progress["cases"]:
        if case["status"] != "completed":
            continue
        rooms = {str(request["bootstrap_room"]) for row in forwards if case["start_unix"] <= row["launch_unix"] <= case["end_unix"]
                 for request in row["requests"] if request["output_tokens"] >= 40}
        if len(rooms) != case["requests"]:
            print(json.dumps({"case": case["name"], "status": "incomplete_remote_snapshot", "rooms": len(rooms)}))
            continue
        latest = {role: {} for role in stages}
        for role, records in stages.items():
            for record in records:
                room = str(record.get("bootstrap_room"))
                if room in rooms and record["reported_unix"] <= case["end_unix"]:
                    if record["reported_unix"] > latest[role].get(room, {}).get("reported_unix", 0):
                        latest[role][room] = record
        rows = []
        for room in sorted(rooms):
            prefill = latest["prefill"][room]
            decode = latest["decode"][room]
            if prefill["event"] != "set_completion_time" or decode["event"] != "set_completion_time":
                raise ValueError(f"Incomplete terminal stages for {case['name']} room {room}")
            prefill_times, decode_times = prefill["times"], decode["times"]
            rows.append({"bootstrap_room": room, "prefill_rank": prefill["dp_rank"], "decode_rank": decode["dp_rank"],
                         "prefill_bootstrap_ms": interval(prefill_times, "scheduler_recv_time", "bootstrap_done_time"),
                         "prefill_admission_wait_ms": interval(prefill_times, "bootstrap_done_time", "forward_entry_time"),
                         "prefill_chunks_elapsed_ms": interval(prefill_times, "forward_entry_time", "prefill_transfer_queue_entry_time"),
                         "prefill_source_hold_after_compute_ms": interval(prefill_times, "prefill_transfer_queue_entry_time", "prefill_kv_transfer_finish_time"),
                         "decode_bootstrap_ms": interval(decode_times, "scheduler_recv_time", "bootstrap_done_time"),
                         "decode_allocation_and_metadata_ms": interval(decode_times, "bootstrap_done_time", "decode_transfer_queue_entry_time"),
                         "decode_wait_for_prefill_and_transfer_ms": interval(decode_times, "decode_transfer_queue_entry_time", "wait_queue_entry_time"),
                         "decode_ready_to_first_token_ms": interval(decode_times, "wait_queue_entry_time", "decode_prebuilt_finish_time"),
                         "decode_scheduler_to_first_token_ms": interval(decode_times, "scheduler_recv_time", "decode_prebuilt_finish_time"),
                         "source_rows": {"prefill": prefill, "decode": decode}})
        raw_path = CAMPAIGN / "prefill" / f"{case['name']}.jsonl"
        measured = read_rows(raw_path)[-1]
        first_visible = [length - len(intervals) for length, intervals in zip(measured["output_lens"], measured["itls"])]
        if len(first_visible) != case["requests"] or any(value < 1 for value in first_visible):
            raise ValueError(f"Invalid first-visible-token reconstruction: {case['name']}")
        summary = {key: summarize(row[key] for row in rows) for key in rows[0] if key.endswith("_ms")}
        result = {"case": case["name"], "requests": case["requests"], "client_ttft_ms": measured["mean_ttft_ms"],
                  "first_nonempty_text_token_indices": first_visible, "first_nonempty_text_distribution": dict(collections.Counter(first_visible)),
                  "mean_additional_tokens_before_text": statistics.mean(value - 1 for value in first_visible), "stages": summary,
                  "client_minus_decode_scheduler_first_token_ms": measured["mean_ttft_ms"] - summary["decode_scheduler_to_first_token_ms"]["mean_ms"],
                  "client_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(), "requests_detail": rows}
        cases.append(result)
        print(json.dumps({key: value for key, value in result.items() if key not in ("requests_detail", "stages")}, ensure_ascii=False))
    output = {"scope": "Measured request-stage intervals from identical pinned SGLang source; per-role intervals avoid cross-node clock assumptions. Source hold time is not necessarily on the client critical path. First nonempty text index is reconstructed from raw output counts and ITL lengths, not assumed to be the first generated token.", "campaign_status": progress["status"], "cases": cases}
    (ROOT / "pd-request-timeline-analysis.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
