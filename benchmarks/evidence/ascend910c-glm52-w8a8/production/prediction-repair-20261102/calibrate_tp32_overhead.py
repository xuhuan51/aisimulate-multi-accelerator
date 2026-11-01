import argparse
import collections
import gzip
import hashlib
import json
import statistics
from pathlib import Path

from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

from calibrate_eager_overhead import configuration


ROOT = Path(__file__).resolve().parent


def read_bytes(path):
    return path.read_bytes() if path.exists() else gzip.decompress(path.with_suffix(path.suffix + ".gz").read_bytes())


def model_config(profile=None):
    result = configuration()
    result.update({"tp": 1, "attention_dp": 32, "moe_ep_size": 32})
    options = result["estimator_config"]["op_level"]["ascend"]
    options["enable_tp_lm_head_all_to_all"] = True
    if profile is not None:
        options["eager_runtime_overhead"] = profile
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    campaign = ROOT / "tp32-paired-node20-25"
    progress = json.loads((campaign / "node0/progress.json").read_text())
    if progress["status"] != "completed" and not args.allow_partial:
        raise ValueError("Freeze all cases or explicitly request partial diagnostics")
    cases = {case["name"]: case for case in progress["cases"] if case["status"] == "completed"}
    if "calibration-c1" not in cases:
        raise ValueError("C1 calibration must complete before fitting")
    groups = collections.defaultdict(list)
    hashes = {}
    for rank in range(32):
        path = campaign / f"node{rank // 16}" / f"paired-forward-rank{rank}.jsonl"
        raw = read_bytes(path)
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        for line in raw.splitlines():
            row = json.loads(line)
            if row["dp_rank"] != rank:
                raise ValueError(f"Wrong global DP rank in {path}")
            groups[row["timed_forward_iteration"]].append(row)
    native = RustForwardPassPerfModel.best_available(model_config())
    selected = {}
    exclusions = {}
    for name, case in cases.items():
        rows = []
        counts = collections.Counter()
        for iteration, group in sorted(groups.items()):
            if not case["start_unix"] <= min(row["launch_unix"] for row in group) <= case["end_unix"]:
                continue
            if len(group) != 32 or {row["dp_rank"] for row in group} != set(range(32)):
                counts["incomplete_group"] += 1
                continue
            ordered = sorted(group, key=lambda row: row["dp_rank"])
            launch_spread = max(row["launch_unix"] for row in ordered) - min(row["launch_unix"] for row in ordered)
            if launch_spread > min(0.2, min(row["elapsed_ms"] for row in ordered) / 2000):
                counts["inconsistent_launch_times"] += 1
                continue
            scheduled = [row["scheduled_requests"] for row in ordered]
            if any(row["sum_prefill_tokens"] for row in scheduled):
                counts["prefill_or_mixed"] += 1
                continue
            active = [row for row in scheduled if row["num_decode_requests"]]
            if not active:
                counts["idle"] += 1
                continue
            positions = [row["sum_decode_kv_tokens"] / row["num_decode_requests"] - case["input_length"] for row in active]
            if not all(40 <= position <= case["output_length"] - 8 for position in positions):
                counts["warmup_or_boundary"] += 1
                continue
            metrics = [{"scheduled_requests": row} for row in scheduled]
            actual = max(row["elapsed_ms"] for row in ordered)
            predicted = native.estimate_forward_pass_time_ms(metrics)
            rows.append({"timed_forward_iteration": iteration, "metrics": metrics, "measured_ms": actual, "raw_prediction_ms": predicted,
                         "residual_ms": actual - predicted, "source_rows": ordered})
        selected[name] = rows
        exclusions[name] = dict(counts)
    calibration = selected["calibration-c1"]
    if len(calibration) < 50 or any(sum(rank["scheduled_requests"]["num_decode_requests"] for rank in row["metrics"]) != 1 for row in calibration):
        raise ValueError("Require at least50 complete C1 steady decode groups")
    residual = statistics.median(row["residual_ms"] for row in calibration)
    if residual < 0:
        raise ValueError("Negative residual requires an operator-model investigation")
    calibration_hash = hashlib.sha256(json.dumps(calibration, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    profile = {"calibration_id": f"node20-25-tp32-c1-paired-{calibration_hash[:16]}", "attention_tp": 1, "attention_dp": 32,
               "moe_ep_size": 32, "prefill_ms": 0.0, "decode_ms": residual}
    corrected = RustForwardPassPerfModel.best_available(model_config(profile))
    summaries = []
    for name, rows in selected.items():
        if not rows:
            raise ValueError(f"No steady decode groups for {name}")
        for row in rows:
            row["corrected_ms"] = corrected.estimate_forward_pass_time_ms(row["metrics"])
        summary = {"case": name, "used_for_fit": name == "calibration-c1", "groups": len(rows), "excluded": exclusions[name],
                   "median_measured_ms": statistics.median(row["measured_ms"] for row in rows),
                   "raw_mape": statistics.mean(abs(row["raw_prediction_ms"] / row["measured_ms"] - 1) * 100 for row in rows),
                   "corrected_mape": statistics.mean(abs(row["corrected_ms"] / row["measured_ms"] - 1) * 100 for row in rows)}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    (ROOT / "eager-overhead-tp32-adp32.json").write_text(json.dumps(profile, indent=2) + "\n")
    analysis = {"campaign_status": progress["status"], "scope": "TP32/ADP32/EP32 paired steady decode only, fitted on C1; other cases are not used for fitting. Colocated prefill and endpoint latency are not PD serving validation.",
                "profile": profile, "calibration_sha256": calibration_hash, "source_sha256": hashes, "summaries": summaries, "forwards": selected}
    (ROOT / "tp32-paired-forward-analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")


if __name__ == "__main__":
    main()
