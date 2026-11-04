import json
import statistics

from calibrate_eager_overhead import ROOT, configuration
from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel


def main():
    campaign = ROOT / "forward-metrics-node20"
    records = [json.loads(line) for line in (campaign / "forward-metrics.jsonl").read_text().splitlines()]
    identity = configuration()
    identity["worker_type"] = "prefill"
    model = RustForwardPassPerfModel.best_available(identity)
    results = []
    for case in json.loads((campaign / "progress.json").read_text())["cases"]:
        selected = [row for row in records if case["start_unix"] <= row["received_unix"] <= case["end_unix"] and row["metrics"]["wall_time"] > 0]
        groups = []
        for row in sorted(selected, key=lambda item: item["received_unix"]):
            if not groups or row["received_unix"] - groups[-1][-1]["received_unix"] > 0.2:
                groups.append([])
            groups[-1].append(row)
        rows = []
        for group in groups:
            if len(group) != 4 or {row["metrics"]["dp_rank"] for row in group} != set(range(4)):
                continue
            ordered = sorted(group, key=lambda row: row["metrics"]["dp_rank"])
            metrics = [{"scheduled_requests": row["metrics"]["scheduled_requests"]} for row in ordered]
            if not any(rank["scheduled_requests"]["sum_prefill_tokens"] > 0 for rank in metrics):
                continue
            if any(rank["scheduled_requests"]["num_decode_requests"] > 0 for rank in metrics):
                continue
            predicted = model.estimate_forward_pass_time_ms(metrics)
            measured = max(row["metrics"]["wall_time"] for row in ordered) * 1000
            rows.append({"metrics": metrics, "measured_ms": measured, "predicted_ms": predicted,
                         "residual_ms": measured - predicted, "source_rows": ordered})
        results.append({"case": case["name"], "rows": rows})
        print(case["name"], "complete_groups", len(rows))
        for row in rows:
            print([(rank["scheduled_requests"]["sum_prefill_tokens"], rank["scheduled_requests"]["sum_prefill_kv_tokens"], rank["scheduled_requests"]["num_decode_requests"]) for rank in row["metrics"]],
                  "actual/predicted/residual", *[round(row[key], 2) for key in ("measured_ms", "predicted_ms", "residual_ms")])
        if rows:
            print("median_residual", statistics.median(row["residual_ms"] for row in rows))
    (ROOT / "prefill-telemetry-diagnostic.json").write_text(json.dumps({"scope": "Diagnostic only; mixed-mode and startup/publisher effects require qualification before calibration", "configuration": identity, "cases": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
