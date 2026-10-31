#!/usr/bin/env python3

import gzip
import importlib.util
import json
import pathlib
import shutil
import statistics
import sys


def main():
    module_path = pathlib.Path(__file__).with_name("sample-serving-metrics.py")
    spec = importlib.util.spec_from_file_location("serving_metrics", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    count = 0
    for root in map(pathlib.Path, sys.argv[1:]):
        for complete in root.glob("c*-n*/.complete"):
            point = complete.parent
            summary_path = point / "metrics-summary.json"
            summary = json.loads(summary_path.read_text())
            if summary.get("aggregation_version") == 2:
                continue
            shutil.copyfile(summary_path, point / "metrics-summary-raw-gauge.json")
            previous = {}
            with gzip.open(point / "metrics.jsonl.gz", "rt") as source:
                samples = [module.aggregate(json.loads(line), previous) for line in source]
            measured = [row for row in samples if row["measured_approx"] and row["complete"]]
            active = [row["decode_running"] for row in measured if row["decode_running"] > 0]
            summary.update({
                "aggregation_version": 2, "samples": samples,
                "raw_decode_running_peak": max((row["raw_decode_running"] for row in measured), default=0),
                "decode_running_peak": max(active, default=0),
                "decode_running_mean_when_nonzero": statistics.mean(active) if active else 0,
                "decode_rank_running_peak": max((row["decode_rank_max"] for row in measured), default=0),
                "scope": "TP副本去重后，仅累计本采样间隔decode token counter增长的DP rank；过滤空闲rank旧gauge；逐秒近似统计。原始gauge汇总单独保留。",
            })
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
            count += 1
            print(point.name, "raw_peak=", summary["raw_decode_running_peak"], "active_peak=", summary["decode_running_peak"])
    print(f"Recomputed {count} completed points")


if __name__ == "__main__":
    main()
