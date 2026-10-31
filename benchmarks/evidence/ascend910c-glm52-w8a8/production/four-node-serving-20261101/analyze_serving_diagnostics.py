import argparse
import collections
import gzip
import hashlib
import json
import pathlib
import statistics


GAUGES = {
    "sglang:num_queue_reqs", "sglang:num_decode_prealloc_queue_reqs",
    "sglang:num_decode_transfer_queue_reqs", "sglang:num_decode_host_receive_queue_reqs",
    "sglang:num_prefill_bootstrap_queue_reqs", "sglang:num_prefill_inflight_queue_reqs",
    "sglang:token_usage", "sglang:pending_prealloc_token_usage",
}
HISTOGRAMS = {
    "sglang:kv_transfer_latency_ms", "sglang:kv_transfer_speed_gb_s",
    "sglang:kv_transfer_alloc_ms", "sglang:kv_transfer_bootstrap_ms",
    "sglang:dp_attention_sync_wait_seconds", "sglang:queue_time_seconds",
    "sglang:per_stage_req_latency_seconds",
}
STAGE = "sglang:scheduler_stage_seconds_total"
COUNTERS = {STAGE} | {name + suffix for name in HISTOGRAMS for suffix in ("_sum", "_count")}


def series_key(replica, row):
    return replica, row["name"], tuple(sorted(row["labels"].items()))


def analyze_records(records):
    first = last = previous = None
    first_time = last_time = None
    resets = set()
    gauge_samples = collections.defaultdict(list)
    measured_samples = excluded_samples = 0
    for record in records:
        if not record["measured_approx"]:
            continue
        if any("error" in endpoint for endpoint in record["endpoints"]):
            excluded_samples += 1
            continue
        counters = {}
        gauges = collections.defaultdict(dict)
        for endpoint in record["endpoints"]:
            replica = endpoint["replica"]
            role = "decode" if replica.startswith("D") else "prefill"
            for row in endpoint.get("metrics", []):
                name = row["name"]
                if name in COUNTERS:
                    key = series_key(replica, row)
                    counters[key] = max(counters.get(key, 0), row["value"])
                elif name in GAUGES:
                    rank = replica, row["labels"].get("dp_rank", "0")
                    values = gauges[(role, name)]
                    values[rank] = max(values.get(rank, 0), row["value"])
        for identity, ranks in gauges.items():
            values = list(ranks.values())
            ratio = identity[1].endswith("token_usage")
            gauge_samples[identity].append({
                "value": statistics.mean(values) if ratio else sum(values),
                "rank_peak": max(values),
                "ranks": len(values),
            })
        if first is None:
            first, first_time = counters, record["time"]
        if previous is not None:
            resets.update(key for key in counters.keys() & previous.keys() if counters[key] < previous[key])
        last, last_time, previous = counters, record["time"], counters
        measured_samples += 1
    first, last = first or {}, last or {}
    valid = (first.keys() & last.keys()) - resets
    deltas = {key: last[key] - first[key] for key in valid if last[key] >= first[key]}
    stages = collections.defaultdict(lambda: collections.defaultdict(dict))
    invalid_stage_ranks = {
        (replica, tuple((name, value) for name, value in labels if name not in ("category", "forward_overlap")))
        for replica, metric, labels in (first.keys() ^ last.keys()) | resets if metric == STAGE
    }
    for (replica, metric, labels), delta in deltas.items():
        if metric != STAGE:
            continue
        rank = replica, tuple((name, value) for name, value in labels if name not in ("category", "forward_overlap"))
        if rank in invalid_stage_ranks:
            continue
        role = "decode" if replica.startswith("D") else "prefill"
        category = dict(labels)["category"]
        stages[role][rank][category] = stages[role][rank].get(category, 0) + delta
    stage_summary = {}
    for role, ranks in stages.items():
        categories = sorted({category for values in ranks.values() for category in values})
        means = {category: statistics.mean(values.get(category, 0) for values in ranks.values()) for category in categories}
        total = sum(means.values())
        stage_summary[role] = {
            "valid_physical_ranks": len(ranks), "mean_seconds_per_rank": means,
            "run_batch_percent": 100 * means["run_batch"] / total if "run_batch" in means and total else None,
        }
    histograms = collections.defaultdict(lambda: {"sum": 0, "rank_observation_count": 0, "valid_rank_series": 0})
    for (replica, metric, labels), delta in deltas.items():
        if not metric.endswith("_sum") or metric[:-4] not in HISTOGRAMS:
            continue
        count = deltas.get((replica, metric[:-4] + "_count", labels))
        if count is None or count <= 0:
            continue
        role = "decode" if replica.startswith("D") else "prefill"
        semantic = tuple((name, value) for name, value in labels if name in ("stage", "mode"))
        summary = histograms[(role, metric[:-4], semantic)]
        summary["sum"] += delta
        summary["rank_observation_count"] += count
        summary["valid_rank_series"] += 1
    return {
        "window": {"start": first_time, "end": last_time, "complete_samples": measured_samples, "excluded_endpoint_error_samples": excluded_samples},
        "counter_quality": {"reset_series": len(resets), "missing_boundary_series": len(first.keys() ^ last.keys()), "invalid_stage_ranks": len(invalid_stage_ranks)},
        "scheduler": stage_summary,
        "gauges": [{
            "role": role, "metric": name, "samples": len(samples),
            "mean": statistics.mean(sample["value"] for sample in samples),
            "peak": max(sample["value"] for sample in samples),
            "rank_peak": max(sample["rank_peak"] for sample in samples),
            "dp_rank_counts": sorted({sample["ranks"] for sample in samples}),
        } for (role, name), samples in sorted(gauge_samples.items())],
        "histograms": [{
            "role": role, "metric": name, "labels": dict(labels), **values,
            "mean": values["sum"] / values["rank_observation_count"],
        } for (role, name, labels), values in sorted(histograms.items())],
    }


def analyze_point(point):
    if not (point / ".complete").exists():
        raise ValueError(f"Incomplete measurement: {point}")
    path = point / "metrics.jsonl.gz"
    with gzip.open(path, "rt") as stream:
        result = analyze_records(json.loads(line) for line in stream)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    result.update({"source": str(path), "source_sha256": digest.hexdigest(), "source_bytes": path.stat().st_size})
    return result


def analyze_matrix(measurement_root, output):
    plan = json.loads((measurement_root / "comparison-plan.json").read_text())
    rows = []
    for configuration in plan["configurations"]:
        for concurrency in plan["workload"]["concurrencies"]:
            requests = concurrency * plan["workload"]["prompt_multiplier"]
            label = configuration["label"]
            point = measurement_root / (label + "-20261101") / f"c{concurrency}-n{requests}"
            row = {"configuration": label, "concurrency": concurrency, **analyze_point(point)}
            rows.append(row)
            print(f"Diagnosed {label} concurrency={concurrency}", flush=True)
    result = {
        "scope": "One-second approximate formal window after warmup log; end boundary precedes final client exit. Queue gauges deduplicate TP by max per replica/DP rank; their means average sampled totals. Ratio gauge mean averages per-rank maxima; rank_peak is the largest reported rank ratio. Scheduler deltas use stable physical-rank series; resets/missing boundaries excluded. Scheduler run_batch is wall time, not device kernel time. Histogram means weight reported rank observations, not unique requests; missing/zero-count series do not imply zero latency. v1 collection lacks stage/PD-queue metrics.",
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    analyze_matrix(args.measurement_root, args.output)


if __name__ == "__main__":
    main()
