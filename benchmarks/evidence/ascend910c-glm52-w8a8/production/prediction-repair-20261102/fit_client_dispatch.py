import argparse
import hashlib
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.profile.read_text())
    if data["schema"] != "aisimulate-client-dispatch-profile/v1" or data["status"] != "completed":
        raise ValueError("Completed CPU-only dispatch profile required")
    training_lengths = {512, 2048, 8192, 32768}
    rows = [row for row in data["summaries"] if row["input_tokens"] in training_lengths]
    if {row["input_tokens"] for row in rows} != training_lengths:
        raise ValueError("Missing preselected training lengths")
    lengths = [row["input_tokens"] for row in rows]
    costs = [row["median_dispatch_interval_ms"] for row in rows]
    center_length, center_cost = statistics.mean(lengths), statistics.mean(costs)
    slope = sum((length - center_length) * (cost - center_cost) for length, cost in zip(lengths, costs)) / sum(
        (length - center_length) ** 2 for length in lengths)
    base = center_cost - slope * center_length
    if min(base, slope) < 0:
        raise ValueError("CPU dispatch fit has negative costs")
    validation = [{**row, "prediction_ms": base + slope * row["input_tokens"],
                   "error_pct": 100 * ((base + slope * row["input_tokens"]) / row["median_dispatch_interval_ms"] - 1)}
                  for row in data["summaries"] if row["input_tokens"] not in training_lengths]
    result = {"schema": "aisimulate-client-dispatch-profile-fit/v1",
              "method": "OLS of CPU-only median request preparation intervals at preselected 512,2048,8192,32768 token lengths; no inference timings or serving arrivals used",
              "source_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
              "train_input_tokens": sorted(training_lengths),
              "client_dispatch": {"base_ms": base, "per_input_token_us": slope * 1000},
              "heldout_cpu_lengths": validation}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
