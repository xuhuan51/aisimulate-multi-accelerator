import collections
import gzip
import hashlib
import json
import statistics
from pathlib import Path

from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

from calibrate_eager_overhead import configuration


ROOT = Path(__file__).resolve().parent


def read_rows(path):
    raw = path.read_bytes() if path.exists() else gzip.decompress(path.with_suffix(path.suffix + ".gz").read_bytes())
    return [json.loads(line) for line in raw.splitlines()]


def model_config(profile=None):
    result = configuration(profile)
    result["worker_type"] = "prefill"
    options = result["estimator_config"]["op_level"]["ascend"]
    options.update(enable_dp_lm_head=True, enable_tp_lm_head_all_to_all=False)
    return result


def extract_cases(campaign, completed):
    groups = collections.defaultdict(list)
    for rank in range(4):
        for row in read_rows(campaign / "prefill" / f"paired-forward-rank{rank}.jsonl"):
            groups[row["timed_forward_iteration"]].append(row)
    decoder = campaign / "decode"
    if not decoder.exists():
        decoder = campaign / "decode-live"
    forwards = [row for rank in range(4) for row in read_rows(decoder / f"paired-forward-rank{rank}.jsonl")]
    native = RustForwardPassPerfModel.best_available(model_config())
    cases = {}
    for case in completed:
        rooms = {str(request["bootstrap_room"]) for row in forwards if case["start_unix"] <= row["launch_unix"] <= case["end_unix"] for request in row["requests"] if request["output_tokens"] >= 40}
        if len(rooms) != case["requests"]:
            raise ValueError(f"Incomplete decode request identity: {case['name']}")
        selected = []
        exclusions = collections.Counter()
        for iteration, group in sorted(groups.items()):
            if not case["start_unix"] <= min(row["launch_unix"] for row in group) <= case["end_unix"]:
                continue
            if len(group) != 4 or {row["dp_rank"] for row in group} != set(range(4)):
                exclusions["incomplete_group"] += 1
                continue
            ordered = sorted(group, key=lambda row: row["dp_rank"])
            if max(row["launch_unix"] for row in ordered) - min(row["launch_unix"] for row in ordered) > 0.2:
                exclusions["inconsistent_launch_times"] += 1
                continue
            active_rooms = {str(request["bootstrap_room"]) for row in ordered for request in row["requests"]}
            if not active_rooms or not active_rooms.issubset(rooms):
                exclusions["warmup_or_unrelated"] += 1
                continue
            metrics = [{"scheduled_requests": row["scheduled_requests"]} for row in ordered]
            if any(row["scheduled_requests"]["num_decode_requests"] for row in ordered):
                exclusions["mixed"] += 1
                continue
            actual = max(row["elapsed_ms"] for row in ordered)
            predicted = native.estimate_forward_pass_time_ms(metrics)
            selected.append({"timed_forward_iteration": iteration, "metrics": metrics, "measured_ms": actual, "predicted_ms": predicted, "residual_ms": actual - predicted, "source_rows": ordered})
        cases[case["name"]] = {"rows": selected, "exclusions": dict(exclusions)}
    return cases


def main():
    campaign = ROOT / "pd-paired-1p1d"
    progress = json.loads((campaign / "progress.json").read_text())
    if progress["status"] != "completed":
        raise ValueError("Fit only a frozen completed campaign")
    cases = extract_cases(campaign, progress["cases"])
    calibration_case = "diagnostic-c4-8k128"
    calibration = cases[calibration_case]["rows"]
    if len(calibration) < 4:
        raise ValueError("Too few complete formal prefill forwards")
    identity = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()
    profile = {"calibration_id": f"node20-pd-prefill-c4-{identity[:16]}", "attention_tp": 4, "attention_dp": 4, "moe_ep_size": 16, "prefill_ms": statistics.median(row["residual_ms"] for row in calibration), "decode_ms": 0.0}
    if profile["prefill_ms"] < 0:
        raise ValueError("Nonnegative additive model is not supported by this observation")
    calibrated = RustForwardPassPerfModel.best_available(model_config(profile))
    for name, case in cases.items():
        rows = case["rows"]
        for row in rows:
            row["corrected_ms"] = calibrated.estimate_forward_pass_time_ms(row["metrics"])
        case["summary"] = {"used_for_fit": name == calibration_case, "groups": len(rows), "raw_mape": statistics.mean(abs(row["predicted_ms"] / row["measured_ms"] - 1) * 100 for row in rows), "corrected_mape": statistics.mean(abs(row["corrected_ms"] / row["measured_ms"] - 1) * 100 for row in rows), "median_measured_ms": statistics.median(row["measured_ms"] for row in rows)}
        print(json.dumps({"case": name, **case["summary"]}), flush=True)
    report = {"scope": "Prefill TP16/ADP4/EP16, paired same-runtime NPU-event residual; fitted only on eight formal requests of previous 1P1D short C4 case. Warmup excluded by bootstrap_room. Cold formal requests remain visible in validation errors. No original four-node data used for fitting.", "configuration": model_config(), "profile": profile, "calibration_case": calibration_case, "cases": cases}
    (ROOT / "pd-prefill-calibration-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    target = ROOT / "eager-overhead-prefill-tp16-adp4.json"
    encoded = json.dumps(profile, indent=2) + "\n"
    if target.exists() and target.read_text() != encoded:
        raise ValueError("Preserve the original profile rather than overwriting it")
    target.write_text(encoded)


if __name__ == "__main__":
    main()
