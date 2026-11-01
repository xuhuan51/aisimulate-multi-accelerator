import collections
import gzip
import hashlib
import json
import statistics

from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

from calibrate_eager_overhead import ROOT, configuration


def summarize(rows, key):
    if not rows:
        return None
    return {
        "forwards": len(rows),
        "median_measured_ms": statistics.median(row["measured_ms"] for row in rows),
        "median_residual_ms": statistics.median(row["residual_ms"] for row in rows),
        "raw_mape": statistics.mean(abs(row["predicted_ms"] / row["measured_ms"] - 1) * 100 for row in rows),
        "corrected_mape": statistics.mean(abs(row[key] / row["measured_ms"] - 1) * 100 for row in rows),
    }


def main():
    campaign = ROOT / "forward-paired-node20"
    progress = json.loads((campaign / "progress.json").read_text())
    groups = collections.defaultdict(list)
    hashes = {}
    for rank in range(4):
        source = campaign / f"paired-forward-rank{rank}.jsonl"
        raw = source.read_bytes() if source.exists() else gzip.decompress(source.with_suffix(".jsonl.gz").read_bytes())
        hashes[source.name] = hashlib.sha256(raw).hexdigest()
        for line in raw.splitlines():
            record = json.loads(line)
            groups[record["forward_iteration"]].append(record)
    profile = json.loads((ROOT / "eager-overhead-tp16-adp4.json").read_text())
    native = RustForwardPassPerfModel.best_available(configuration())
    calibrated = RustForwardPassPerfModel.best_available(configuration(profile))
    cases = []
    for case in progress["cases"]:
        if case["status"] != "completed":
            continue
        decode = []
        prefill = []
        excluded = collections.Counter()
        for iteration, group in sorted(groups.items()):
            if not case["start_unix"] <= min(row["launch_unix"] for row in group) <= case["end_unix"]:
                continue
            if len(group) != 4 or {row["dp_rank"] for row in group} != set(range(4)):
                excluded["incomplete_group"] += 1
                continue
            ordered = sorted(group, key=lambda row: row["dp_rank"])
            metrics = [{"scheduled_requests": row["scheduled_requests"]} for row in ordered]
            active_prefill = [row["scheduled_requests"] for row in ordered if row["scheduled_requests"]["sum_prefill_tokens"]]
            active_decode = [row["scheduled_requests"] for row in ordered if row["scheduled_requests"]["num_decode_requests"]]
            if active_prefill and active_decode:
                excluded["mixed_mode"] += 1
                continue
            if active_prefill and max(row["sum_prefill_tokens"] for row in active_prefill) < 4096:
                excluded["small_probe"] += 1
                continue
            if active_decode:
                positions = [row["sum_decode_kv_tokens"] / row["num_decode_requests"] - case["input_length"] for row in active_decode]
                if not all(40 <= position <= case["output_length"] - 8 for position in positions):
                    excluded["warmup_or_boundary"] += 1
                    continue
            if not active_decode and not active_prefill:
                excluded["empty"] += 1
                continue
            actual = max(row["elapsed_ms"] for row in ordered)
            predicted = native.estimate_forward_pass_time_ms(metrics)
            result = {"forward_iteration": iteration, "metrics": metrics, "measured_ms": actual,
                      "predicted_ms": predicted, "residual_ms": actual - predicted,
                      "corrected_ms": calibrated.estimate_forward_pass_time_ms(metrics), "source_rows": ordered}
            (prefill if active_prefill else decode).append(result)
        summary = {"case": case["name"], "excluded": dict(excluded), "decode": summarize(decode, "corrected_ms"),
                   "prefill_including_warmup": summarize(prefill, "predicted_ms"), "decode_rows": decode, "prefill_rows": prefill}
        cases.append(summary)
        print(json.dumps({key: value for key, value in summary.items() if not key.endswith("_rows")}))
    (ROOT / "paired-forward-analysis.json").write_text(json.dumps({
        "scope": "Unchanged NPU events paired to the scheduler forward ID and batch snapshot at launch; no extra device synchronization. Prefill includes warmup and must be qualified separately.",
        "status": progress["status"], "source_sha256": hashes, "decode_profile": profile, "cases": cases,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
