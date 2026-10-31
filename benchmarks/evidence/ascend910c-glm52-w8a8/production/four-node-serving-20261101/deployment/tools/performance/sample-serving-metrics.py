#!/usr/bin/env python3

import argparse
import concurrent.futures
import gzip
import json
import math
import pathlib
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request


METRICS = {
    "sglang:num_running_reqs", "sglang:num_queue_reqs", "sglang:num_used_tokens",
    "sglang:token_usage", "sglang:kv_available_tokens", "sglang:kv_used_tokens",
    "sglang:max_total_num_tokens", "sglang:num_retracted_requests_total",
    "sglang:forward_execution_seconds_total", "sglang:realtime_tokens_total",
    "sglang:dp_attention_tokens_total", "sglang:kv_transfer_latency_ms",
    "sglang:kv_transfer_speed_gb_s", "sglang:pending_prealloc_token_usage",
    "sglang:num_prealloc_queue_reqs", "sglang:num_transfer_queue_reqs",
    "sglang:num_inflight_reqs", "sglang:prefill_effective_tokens_total",
    "sglang:num_aborted_requests_total", "sglang:prompt_tokens_total",
    "sglang:generation_tokens_total", "sglang:num_requests_total",
}


def parse_metrics(text):
    rows = []
    for line in text.splitlines():
        if not line.startswith("sglang:"):
            continue
        match = re.fullmatch(r'([^\s{]+)(?:\{(.*?)\})?\s+([-+\w.eE]+)(?:\s+\d+)?', line)
        if not match or match[1] not in METRICS:
            continue
        labels = dict(re.findall(r'(\w+)="((?:\\.|[^"\\])*)"', match[2] or ""))
        value = float(match[3])
        if math.isfinite(value):
            rows.append({"name": match[1], "labels": labels, "value": value})
    return rows


def fetch(endpoint):
    replica, url = endpoint.split("=", 1)
    started = time.time()
    try:
        with urllib.request.urlopen(url + "/metrics", timeout=5) as response:
            rows = parse_metrics(response.read().decode())
        return {"replica": replica, "url": url, "time": started, "metrics": rows}
    except Exception as error:
        return {"replica": replica, "url": url, "time": started, "error": str(error)}


def running_by_rank(samples):
    ranks = {}
    for sample in samples:
        for row in sample.get("metrics", []):
            if row["name"] != "sglang:num_running_reqs":
                continue
            labels = row["labels"]
            key = (sample["replica"], labels.get("dp_rank", "0"))
            ranks[key] = max(ranks.get(key, 0), row["value"])
    return ranks


def decode_tokens_by_rank(samples):
    ranks = {}
    for sample in samples:
        if not sample["replica"].startswith("D"):
            continue
        for row in sample.get("metrics", []):
            if row["name"] != "sglang:realtime_tokens_total" or row["labels"].get("mode") != "decode":
                continue
            key = (sample["replica"], row["labels"].get("dp_rank", "0"))
            ranks[key] = max(ranks.get(key, 0), row["value"])
    return ranks


def aggregate(record, previous_tokens):
    samples = record["endpoints"]
    ranks = running_by_rank(samples)
    decode = {key: value for key, value in ranks.items() if key[0].startswith("D")}
    counters = decode_tokens_by_rank(samples)
    active = {key: value for key, value in decode.items() if counters.get(key, 0) > previous_tokens.get(key, 0)}
    row = {
        "time": record["time"], "measured_approx": record["measured_approx"],
        "decode_running": sum(active.values()), "raw_decode_running": sum(decode.values()),
        "decode_rank_max": max(active.values(), default=0), "decode_rank_count": len(decode),
        "complete": all("error" not in sample for sample in samples),
    }
    previous_tokens.update(counters)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.interval <= 0:
        parser.error("需要执行命令和正数采样间隔")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "metrics.jsonl.gz").exists():
        parser.error("结果目录已有 metrics.jsonl.gz，拒绝覆盖")
    timing = {"started": time.time(), "measurement_start_approx": None}
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def copy_output():
        with (args.output / "benchmark.log").open("w") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
                if "Warmup completed with" in line:
                    timing["measurement_start_approx"] = time.time() + 1.0

    reader = threading.Thread(target=copy_output, daemon=True)
    reader.start()
    aggregate_samples = []
    previous_tokens = {}
    sample_errors = 0
    try:
        with gzip.open(args.output / "metrics.jsonl.gz", "wt") as raw:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.endpoint)) as executor:
                while process.poll() is None:
                    started = time.time()
                    samples = list(executor.map(fetch, args.endpoint))
                    sample_errors += sum("error" in sample for sample in samples)
                    measured = timing["measurement_start_approx"] is not None and started >= timing["measurement_start_approx"]
                    record = {"time": started, "measured_approx": measured, "endpoints": samples}
                    raw.write(json.dumps(record) + "\n")
                    raw.flush()
                    aggregate_samples.append(aggregate(record, previous_tokens))
                    time.sleep(max(0.0, args.interval - (time.time() - started)))
        exit_code = process.wait()
        reader.join(timeout=10)
    except BaseException:
        process.terminate()
        raise
    timing["finished"] = time.time()
    measured = [row for row in aggregate_samples if row["measured_approx"] and row["complete"]]
    active = [row["decode_running"] for row in measured if row["decode_running"] > 0]
    summary = {
        "exit_code": exit_code, "timing": timing,
        "metric_sampling_interval_seconds": args.interval,
        "metric_endpoint_errors": sample_errors,
        "decode_running_peak": max(active, default=0),
        "decode_running_mean_when_nonzero": statistics.mean(active) if active else 0,
        "decode_rank_running_peak": max((row["decode_rank_max"] for row in measured), default=0),
        "decode_rank_counts_seen": sorted({row["decode_rank_count"] for row in measured}),
        "raw_decode_running_peak": max((row["raw_decode_running"] for row in measured), default=0),
        "samples": aggregate_samples,
        "aggregation_version": 2,
        "scope": "1秒采样；TP副本去重，仅累计本采样间隔decode token counter增长的DP rank，过滤协同forward期间空闲rank的旧gauge；非精确逐轮batch。measured窗口由预热结束日志加1秒估计。",
    }
    (args.output / "metrics-summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
