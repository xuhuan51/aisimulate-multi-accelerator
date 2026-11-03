import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.campaign
    manifest = json.loads((root / "prefill-event-campaign.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Incomplete input-poll campaign")
    arrivals, received, polls, sources = {}, {}, {}, {}
    for pattern in ("tokenizer-ingress-pid*.jsonl", "request-stages-rank*.jsonl", "input-polls-rank*.jsonl"):
        for path in root.glob(pattern):
            sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if "ingress" in path.name:
                    arrivals[row["rid"]] = row
                elif "stages" in path.name and row["event"] == "scheduler_received":
                    received[row["rid"]] = row
                elif "polls" in path.name:
                    rank = int(path.stem.split("rank")[1])
                    polls.setdefault(row["forward_iteration"], {})[rank] = row
    cases = []
    for case in manifest["cases"]:
        if "-r3-events" not in case["id"]:
            continue
        selected = {rid: row for rid, row in arrivals.items()
                    if case["start_unix"] <= row["reported_unix"] <= case["end_unix"] and row["prompt_tokens"] == 8192}
        if len(selected) != case["concurrency"] or not selected.keys() <= received.keys():
            raise ValueError("Request identity is incomplete")
        origin = min(row["send_call_start_monotonic"] for row in selected.values())
        iterations = sorted(iteration for iteration, group in polls.items()
                            if any(rid in selected for row in group.values() for rid in row["request_ids"]))
        first = polls[iterations[0]]
        if set(first) != set(range(4)):
            raise ValueError("First synchronized round lacks a poll snapshot")
        cases.append({"case": case["id"], "concurrency": case["concurrency"], "first_iteration": iterations[0],
                      "requests": [{"rid": rid, "dp_rank": received[rid]["dp_rank"],
                                    "send_start_ms": 1000 * (row["send_call_start_monotonic"] - origin),
                                    "scheduler_received_ms": 1000 * (received[rid]["host_monotonic"] - origin)}
                                   for rid, row in sorted(selected.items(), key=lambda item: item[1]["send_call_start_monotonic"])],
                      "first_round_polls": [{"dp_rank": rank,
                                             "poll_start_ms": 1000 * (row["poll_start_monotonic"] - origin),
                                             "poll_end_ms": 1000 * (row["poll_end_monotonic"] - origin),
                                             "launch_ms": 1000 * (row["launch_monotonic"] - origin),
                                             "forward_mode": row["forward_mode"],
                                             "scheduled_ids": [rid for rid in row["request_ids"] if rid in selected],
                                             "received_ids": [rid for rid in row["received_ids"] if rid in selected]}
                                            for rank, row in sorted(first.items())]})
    args.output.write_text(json.dumps({"scope": "Observed request input snapshots; these scheduler times are never replay inputs",
                                      "source_sha256": sources, "cases": cases}, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "output": str(args.output)}))


if __name__ == "__main__":
    main()
