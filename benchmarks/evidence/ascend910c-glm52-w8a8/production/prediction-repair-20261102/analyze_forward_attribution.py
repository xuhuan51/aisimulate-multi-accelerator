import argparse
import collections
import gzip
import json
import statistics
from pathlib import Path

from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

from calibrate_eager_overhead import configuration


ROOT = Path(__file__).resolve().parent


def records(folder, pattern):
    paths = {path.name: path for path in folder.glob(pattern)}
    for path in folder.glob(pattern + ".gz"):
        paths.setdefault(path.name.removesuffix(".gz"), path)
    for name in sorted(paths):
        path = paths[name]
        content = gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
        for line in content.splitlines():
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operator-only", action="store_true")
    args = parser.parse_args()
    state = json.loads((args.campaign / "attribution-progress.json").read_text())
    if state["status"] != "completed":
        raise ValueError("Analyze a completed campaign")
    actual = collections.defaultdict(dict)
    host = {}
    for row in records(args.campaign, "paired-forward-rank*.jsonl"):
        group = actual[row["timed_forward_iteration"]]
        if row["dp_rank"] in group:
            raise ValueError("Duplicate timed group rank")
        group[row["dp_rank"]] = row
    for row in records(args.campaign, "attribution-rank*.jsonl"):
        identity = (row["rank"], row["forward_iteration"], row["pid"])
        if identity in host:
            raise ValueError("Duplicate host forward")
        host[identity] = row
    raw_model = RustForwardPassPerfModel.best_available(configuration())
    profile = json.loads((ROOT / "eager-overhead-active-tp16-adp4.json").read_text())
    calibrated = None if args.operator_only else RustForwardPassPerfModel.best_available(configuration(profile))
    outputs = []
    for case in state["cases"]:
        selected = []
        excluded = collections.Counter()
        for iteration, ranks in actual.items():
            if len(ranks) != 4:
                excluded["incomplete_group"] += 1
                continue
            ordered = [ranks[rank] for rank in range(4)]
            mapped = [host.get((row["dp_rank"], row["forward_iteration"], row["pid"])) for row in ordered]
            if any(row is None or row["case"] != case["name"] for row in mapped):
                continue
            if any(row["scheduled_requests"]["sum_prefill_tokens"] for row in ordered):
                excluded["prefill"] += 1
                continue
            requests = [request for row in ordered for request in row["requests"]]
            if not requests or any(request["prompt_tokens"] != case["input_length"] or not 40 <= request["output_tokens"] <= case["output_length"] - 8 for request in requests):
                excluded["warmup_or_unsteady_decode"] += 1
                continue
            metrics = [{"scheduled_requests": row["scheduled_requests"]} for row in ordered]
            largest = max(mapped, key=lambda row: row["host_ns"])
            inclusive, exclusive, counts = collections.Counter(), collections.Counter(), collections.Counter()
            cpu_inclusive, cpu_exclusive = collections.Counter(), collections.Counter()
            for span in largest["spans"]:
                inclusive[span["name"]] += span["inclusive_ns"] / 1e6
                exclusive[span["name"]] += span["exclusive_ns"] / 1e6
                counts[span["name"]] += span["calls"]
                cpu_inclusive[span["name"]] += span.get("cpu_inclusive_ns", 0) / 1e6
                cpu_exclusive[span["name"]] += span.get("cpu_exclusive_ns", 0) / 1e6
            raw_ms = raw_model.estimate_forward_pass_time_ms(metrics)
            measured_ms = max(row["elapsed_ms"] for row in ordered)
            selected.append({"iteration": iteration, "metrics": metrics, "measured_ms": measured_ms,
                             "raw_prediction_ms": raw_ms, "unattributed_ms": measured_ms - raw_ms,
                             **({"calibrated_prediction_ms": calibrated.estimate_forward_pass_time_ms(metrics)} if calibrated is not None else {}),
                             "host_run_batch_ms": largest["host_ns"] / 1e6,
                             "largest_host_rank": largest["rank"], "host_inclusive_ms": dict(inclusive),
                             "host_exclusive_ms": dict(exclusive), "calls": dict(counts),
                             "thread_cpu_ms": largest.get("thread_cpu_ns", 0) / 1e6,
                             "cpu_inclusive_ms": dict(cpu_inclusive), "cpu_exclusive_ms": dict(cpu_exclusive),
                             "length_metadata": largest.get("length_metadata")})
        if not selected:
            raise ValueError(f"No complete steady forward groups: {case['name']}")
        scalar_keys = ["measured_ms", "raw_prediction_ms", "unattributed_ms", "host_run_batch_ms", "thread_cpu_ms"]
        if calibrated is not None:
            scalar_keys.append("calibrated_prediction_ms")
        summary = {key: statistics.median(row[key] for row in selected) for key in scalar_keys}
        for key in ("host_inclusive_ms", "host_exclusive_ms", "calls", "cpu_inclusive_ms", "cpu_exclusive_ms"):
            names = set().union(*(row[key] for row in selected))
            summary[key] = {name: statistics.median(row[key].get(name, 0) for row in selected) for name in sorted(names)}
        outputs.append({"case": case, "complete_steady_groups": len(selected), "summary": summary,
                        "exclusions": dict(excluded), "forwards": selected})
    summary_by_name = {item["case"]["name"]: item["summary"] for item in outputs}
    controls = [summary_by_name[name]["measured_ms"] for name in ("control-before-8k-c4", "control-after-8k-c4")]
    host_control = summary_by_name["host-8k-c4"]["measured_ms"]
    result = {"scope": "Measured host API elapsed spans include blocking/device wait; thread CPU time excludes blocked/descheduled intervals but includes driver polling. Neither is additive to existing eager device-event profiles. No residual coefficients fitted.",
              "operator_only": args.operator_only,
              "selection": "All four rank records, exact forward identity joins; pure decode; output positions40..O-8, no warmup or tail; largest host rank per group. Nested exclusive spans remove only instrumented child intervals.",
              "instrumentation_control": {"baseline_device_ms": controls, "instrumented_device_ms": host_control,
                                          "change_from_control_mean_percent": 100*(host_control/statistics.mean(controls)-1)},
              "cases": outputs}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for case in outputs:
        print(json.dumps({"case": case["case"]["name"], "groups": case["complete_steady_groups"], **case["summary"]}), flush=True)


if __name__ == "__main__":
    main()
