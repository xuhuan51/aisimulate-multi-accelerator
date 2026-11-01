import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path

from analyze_ingress_campaign import records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    progress = json.loads((campaign / "progress.json").read_text())
    if progress["status"] != "completed":
        raise ValueError("Complete formal requests before attributing compiler calls")
    launches = [json.loads((campaign / role / "launch.json").read_text()) for role in ("prefill", "decode")]
    if launches[0]["image_id"] != launches[1]["image_id"] or launches[0]["model_metadata_sha256"] != launches[1]["model_metadata_sha256"]:
        raise ValueError("P/D baseline mismatch")
    events = {role: records(campaign / role, "allocator-events-*.jsonl*") for role in ("prefill", "decode")}
    observations, cache_groups = [], collections.defaultdict(list)
    for role, rows in events.items():
        for event in rows:
            if event["event"] != "triton_compile" or event["attn_tp_rank"] != 0:
                continue
            identity = {name: event.get(name) for name in ("kernel", "signature", "constants", "attributes")}
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            cache_groups[(role, digest)].append(event)
    timeline = json.loads(args.timeline.read_text())
    for case in timeline["cases"]:
        for request in case["rows"]:
            for role in ("prefill", "decode"):
                source = request["source_rows"][role]
                local_clock_offset = source["reported_unix"] - source["host_monotonic"]
                start_key = "forward_entry_time" if role == "prefill" else "bootstrap_done_time"
                end_key = "prefill_transfer_queue_entry_time" if role == "prefill" else "decode_transfer_queue_entry_time"
                start = source["times"][start_key] + local_clock_offset
                end = source["times"][end_key] + local_clock_offset
                allocations = [event for event in events[role] if event["event"] == "alloc_extend"
                               and event["dp_rank"] == source["dp_rank"]
                               and start - 0.003 <= event["start_unix"] < end]
                for allocation in allocations:
                    if allocation["start_unix"] + allocation["elapsed_ms"] / 1000 > end + 0.003:
                        raise ValueError("Allocator call crosses the selected request-stage boundary")
                    if not allocation["succeeded"]:
                        raise ValueError("Failed allocation in a completed request")
                    nested = [event for event in events[role] if event["event"] == "triton_compile"
                              and event["pid"] == allocation["pid"]
                              and allocation["start_unix"] <= event["start_unix"]
                              and event["start_unix"] + event["elapsed_ms"] / 1000
                              <= allocation["start_unix"] + allocation["elapsed_ms"] / 1000 + 0.001]
                    observations.append({"case": case["case"], "rid": request["rid"], "role": role,
                                         "dp_rank": source["dp_rank"], "ttft_ms": request["ttft_ms"],
                                         "first_visible_text_token": request["first_visible_text_token"],
                                         "allocation": allocation, "nested_compile_calls": nested})
    slow = [row for row in observations if row["allocation"]["elapsed_ms"] >= 100]
    summaries = []
    for (role, identity), rows in sorted(cache_groups.items()):
        rows.sort(key=lambda row: row["start_unix"])
        if rows[0]["kernel"] != "alloc_extend_kernel":
            continue
        summaries.append({"role": role, "observed_specialization_sha256": identity,
                          "kernel": rows[0]["kernel"], "signature": rows[0]["signature"],
                          "constants": rows[0]["constants"], "attributes": rows[0]["attributes"],
                          "calls": len(rows), "max_compile_api_ms": max(row["elapsed_ms"] for row in rows),
                          "min_compile_api_ms": min(row["elapsed_ms"] for row in rows),
                          "dp_ranks": sorted({row["dp_rank"] for row in rows}), "source_rows": rows})
    warm_allocations = [row["allocation"]["elapsed_ms"] for row in observations if not row["nested_compile_calls"]]
    result = {
        "image_id": launches[0]["image_id"], "model_metadata_sha256": launches[0]["model_metadata_sha256"],
        "source_scope": "Same pinned runtime, P/D TP16 ADP4 EP16; no weight-shard hash claim. No calibration fitted from these requests.",
        "method": "Join request-stage windows with allocator start/end on the same node and rank; join compiler calls by PID and enclosure. "
                  "Use each node's local monotonic/unix offset; do not subtract clocks across nodes. Multi-request batch intervals can share an allocator observation; do not sum across requests.",
        "cache_qualification": "Observed specialization hashes include signature/constants/attrs, not the full Triton cache key (which also includes backend options and environment). "
                               "Compile-API calls can load cached code; a call is not automatically a full compilation. Health checks have an independent DP counter.",
        "first_text_qualification": "The pinned cumulative-text benchmark expands each later chunk's ITL by completion-token delta. For successful complete streams, output_tokens minus ITL count identifies tokens already emitted at first nonempty text; it does not identify why text was initially withheld.",
        "formal_requests": sum(case["requests"] for case in timeline["cases"]),
        "slow_formal_allocations": slow, "all_formal_allocation_observations": observations,
        "no_nested_compile_allocator_median_ms": statistics.median(warm_allocations),
        "allocator_compile_specializations": summaries,
        "remaining": "Cold cache state, health-generated allocator alignment changes and first-visible-text latency are not yet simulated. "
                     "Neither a fixed penalty per rank nor adding all observed delays to final TTFT is valid prediction.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"formal_requests": result["formal_requests"], "slow_formal_allocations": len(slow),
                      "no_compile_median_ms": result["no_nested_compile_allocator_median_ms"]}))
    for row in slow:
        print(json.dumps({"case": row["case"], "role": row["role"], "dp_rank": row["dp_rank"],
                          "allocation_ms": row["allocation"]["elapsed_ms"],
                          "compile_api_ms": sum(call["elapsed_ms"] for call in row["nested_compile_calls"]),
                          "ttft_ms": row["ttft_ms"]}))


if __name__ == "__main__":
    main()
