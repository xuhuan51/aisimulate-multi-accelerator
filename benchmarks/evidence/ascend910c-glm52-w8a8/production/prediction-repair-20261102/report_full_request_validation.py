import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}

    def read(path):
        content = path.read_bytes()
        sources[str(path)] = hashlib.sha256(content).hexdigest()
        return json.loads(content.splitlines()[-1] if path.suffix == ".jsonl" else content)

    campaign = read(args.campaign / "full-request-validation/campaign.json")
    predicted_campaign = read(args.predictions / "campaign.json")
    baseline_campaign = read(args.baseline / "campaign.json")
    if any(item["status"] != "completed" for item in [campaign, predicted_campaign, baseline_campaign]):
        raise ValueError("Incomplete serving/prediction campaign")
    if predicted_campaign["image_id"] != baseline_campaign["image_id"]:
        raise ValueError("Baseline and new prediction must use the same native build")
    runtime = read(args.campaign / "full-request-validation/server-info.json")
    if (runtime["tp_size"], runtime["attn_dp_size"], runtime["ep_size"], runtime["max_total_num_tokens"]) != (16, 4, 16, 32384):
        raise ValueError("Serving identity differs from frozen prediction configuration")
    pairs = {"ttft_ms": ("mean_ttft_ms", "mean_ttft_ms"), "tpot_ms": ("mean_tpot_ms", "mean_tpot_ms"),
             "input_tok_s": ("input_throughput", "input_throughput_tok_s"),
             "output_tok_s": ("output_throughput", "output_throughput_tok_s")}
    rows = []
    for case in campaign["cases"]:
        concurrency = case["concurrency"]
        actual = read(args.campaign / "full-request-validation" / case["id"] / "requests.jsonl")
        predicted = read(args.predictions / f"c{concurrency}/prediction.json")
        baseline = read(args.baseline / f"c{concurrency}/prediction.json")
        if (actual["completed"] != concurrency or actual["input_lens"] != [8192] * concurrency
                or actual["output_lens"] != [1024] * concurrency or any(actual["errors"])):
            raise ValueError("Serving request/token conservation mismatch")
        for result in (predicted, baseline):
            if (result["completed_requests"] != concurrency or result["total_input_tokens"] != concurrency * 8192
                    or result["total_output_tokens"] != concurrency * 1024):
                raise ValueError("Prediction request/token conservation mismatch")
        row = {"concurrency": concurrency, "requests": concurrency, "global_tp": 16, "attention_dp": 4,
               "attention_tp": 4, "moe_ep": 16, "mode": "aggregated", "graph": False, "overlap": False}
        for name, (actual_key, predicted_key) in pairs.items():
            row["measured_" + name] = actual[actual_key]
            row["isolated_" + name] = baseline[predicted_key]
            row["predicted_" + name] = predicted[predicted_key]
            row["error_percent_" + name] = (predicted[predicted_key] / actual[actual_key] - 1) * 100
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {"scope": "Independent native scheduler predictions vs full8192/1024 held-out requests, N=C and two32-token warmups; single-node aggregate, not PD serving acceptance",
              "arrival_limit": "Prediction synthetic burst arrives simultaneously; actual asynchronous HTTP requests arrive over milliseconds and can enter different prefill rounds. No per-request measured timings injected.",
              "error_formula": "100 * (predicted / measured - 1)", "image_id": predicted_campaign["image_id"],
              "source_sha256": sources, "rows": rows}
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
