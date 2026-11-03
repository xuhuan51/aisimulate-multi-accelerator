import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    cases = json.loads((args.campaign / "context-holdouts.json").read_text())["cases"]
    case = next(row for row in cases if row["id"] == "holdout-12288-4096-c4")
    if case["status"] != "completed" or case["prompts"] != case["concurrency"]:
        raise ValueError("This diagnostic requires one complete request wave")
    maximum_output = {}
    for path in args.campaign.glob("paired-forward-rank*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if not case["start_unix"] <= row["launch_unix"] <= case["end_unix"]:
                continue
            for request in row["requests"]:
                maximum_output[request["rid"]] = max(maximum_output.get(request["rid"], 0), request["output_tokens"])
    selected = {rid for rid, count in maximum_output.items() if count >= case["output_length"] - 1}
    arrivals = []
    for path in args.campaign.glob("tokenizer-ingress-pid*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["rid"] in selected:
                arrivals.append(row)
    arrivals.sort(key=lambda row: row["send_call_start_monotonic"])
    if len(arrivals) != case["concurrency"] or any(row["prompt_tokens"] != case["input_length"] for row in arrivals):
        raise ValueError("External arrival/request conservation failed")
    origin = arrivals[0]["send_call_start_monotonic"]
    blocks = (case["input_length"] + 511) // 512
    trace_rows = [{"input_length": case["input_length"], "output_length": case["output_length"],
                   "hash_ids": list(range(index * blocks, (index + 1) * blocks)),
                   "timestamp": 1000 * (row["send_call_start_monotonic"] - origin)}
                  for index, row in enumerate(arrivals)]
    trace = args.output / "external-arrivals.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in trace_rows))
    source = args.predictions / (case["id"] + "-runtime_layers") / "config.yaml"
    config = yaml.safe_load(source.read_text())
    config["traffic"] = {"source": {"type": "trace", "format": "mooncake", "block_size": 512,
                                     "paths": [str(trace.resolve())]},
                         "load": {"type": "trace_timestamps"}, "stop": {}}
    config_path = args.output / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    command = ["aisimulate", "predict", "--stack", "engine", "--config", str(config_path),
               "--capture-per-request", "--detail", "source", "--output-dir", str(args.output / "prediction")]
    metadata = {"scope": "Conditional external-arrival replay, not an unconditional concurrency forecast. No measured scheduler receive times, routing, forward durations or fitted residuals supplied.",
                "input_records": arrivals, "trace_rows": trace_rows,
                "source_config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "command": command}
    with (args.output / "prediction.log").open("w") as stream:
        outcome = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
    metadata["returncode"] = outcome.returncode
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    outcome.check_returncode()


if __name__ == "__main__":
    main()
