import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, action="append", required=True)
    parser.add_argument("--snapshot-name", action="append", required=True)
    parser.add_argument("--serving-name", action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.campaign) != len(args.snapshot_name):
        parser.error("Each campaign needs one snapshot name")
    serving_names = args.serving_name or ["serving-validation"] * len(args.campaign)
    if len(serving_names) != len(args.campaign):
        parser.error("Each campaign needs one serving directory name")
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = {}
    results = []
    for campaign, snapshot, serving_name in zip(args.campaign, args.snapshot_name, serving_names):
        serving = campaign / serving_name
        manifest_path = serving / "campaign.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] != "completed":
            raise ValueError("Only completed serving campaigns may be compared")
        sources[str(manifest_path)] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        groups = collections.defaultdict(dict)
        for node in ("node0", "node1"):
            for path in (campaign / node / snapshot).glob("paired-forward-rank*.jsonl"):
                content = path.read_bytes()
                sources[str(path)] = hashlib.sha256(content).hexdigest()
                for line in content.splitlines():
                    row = json.loads(line)
                    group = groups[row["timed_forward_iteration"]]
                    if row["dp_rank"] in group:
                        raise ValueError("Duplicate rank/iteration")
                    group[row["dp_rank"]] = row
        if set(rank for group in groups.values() for rank in group) != set(range(32)):
            raise ValueError("Missing rank telemetry")
        for case in manifest["cases"]:
            paired = [(iteration, [ranks[rank] for rank in range(32)]) for iteration, ranks in sorted(groups.items())
                      if len(ranks) == 32 and all(case["start_unix"] <= row["launch_unix"] <= case["end_unix"] for row in ranks.values())]
            formal = {request["rid"] for _, rows in paired for row in rows for request in row["requests"]
                      if request["prompt_tokens"] == case["input_length"] and request["output_tokens"] > 64}
            if len(formal) != case["prompts"]:
                raise ValueError("Could not distinguish all formal requests from warmups")
            steady = []
            populations = collections.Counter()
            for iteration, rows in paired:
                requests = [request for row in rows for request in row["requests"]]
                if not requests or any(row["scheduled_requests"]["sum_prefill_tokens"] for row in rows):
                    continue
                if any(request["rid"] not in formal or not 40 <= request["output_tokens"] <= case["output_length"] - 13 for request in requests):
                    continue
                counts = tuple(row["scheduled_requests"]["num_decode_requests"] for row in rows)
                populations[counts] += 1
                steady.append({"iteration": iteration, "requests_by_rank": counts,
                               "measured_max_forward_ms": max(row["elapsed_ms"] for row in rows),
                               "rank0_forward_ms": rows[0]["elapsed_ms"],
                               "rank0_launch_unix": rows[0]["launch_unix"]})
            if len(steady) < 20:
                raise ValueError("Insufficient complete unprofiled steady Decode groups")
            cadences = [(right["rank0_launch_unix"] - left["rank0_launch_unix"]) * 1000
                        for left, right in zip(steady, steady[1:]) if right["iteration"] == left["iteration"] + 1]
            requests_path = serving / case["id"] / "requests.jsonl"
            sources[str(requests_path)] = hashlib.sha256(requests_path.read_bytes()).hexdigest()
            client = json.loads(requests_path.read_text().splitlines()[-1])
            if (client["completed"] != case["prompts"] or any(client["errors"])
                    or client["input_lens"] != [case["input_length"]] * case["prompts"]
                    or client["output_lens"] != [case["output_length"]] * case["prompts"]):
                raise ValueError("Request/token conservation failed")
            result = {"campaign": str(campaign), "serving_source": str(serving), "case": case, "forwards": steady,
                      "population_histogram": [{"counts": counts, "rounds": rounds} for counts, rounds in populations.items()],
                      "summary": {"rounds": len(steady), "mean_max_forward_ms": statistics.mean(row["measured_max_forward_ms"] for row in steady),
                                  "mean_rank0_forward_ms": statistics.mean(row["rank0_forward_ms"] for row in steady),
                                  "mean_rank0_cadence_ms": statistics.mean(cadences) if cadences else None,
                                  "ttft_ms": client["mean_ttft_ms"], "tpot_ms": client["mean_tpot_ms"]}}
            results.append(result)
            print(json.dumps({"case": case["id"], **result["summary"]}))
    args.output.write_text(json.dumps({"scope": "Complete32-rank unprofiled serving windows; output positions40 through output_length-13; warmup excluded by formal request IDs. No timing fit or prediction input.",
                                      "source_sha256": sources, "cases": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
