import collections
import csv
import gzip
import hashlib
import io
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT / "native-attribution-node20-20261103"


def read_bytes(path):
    return path.read_bytes() if path.exists() else gzip.decompress(path.with_name(path.name + ".gz").read_bytes())


def union_us(intervals):
    total = 0.0
    previous_start = previous_end = None
    for start, end in sorted(intervals):
        if previous_end is None or start > previous_end:
            if previous_end is not None:
                total += previous_end - previous_start
            previous_start, previous_end = start, end
        else:
            previous_end = max(previous_end, end)
    return total if previous_end is None else total + previous_end - previous_start


def main():
    outputs = []
    for folder in sorted((CAMPAIGN / "operator-trace").glob("*_ascend_pt")):
        info = next(folder.glob("profiler_info_*.json"))
        rank = int(info.stem.rsplit("_", 1)[1])
        source = folder / "ASCEND_PROFILER_OUTPUT"
        kernel_path = source / "kernel_details.csv"
        operator_path = source / "operator_details.csv"
        trace_path = source / "trace_view.json"
        events = json.loads(read_bytes(trace_path))
        selected = [event for event in events if event.get("name", "").startswith("step[DECODE")]
        origin = int(min(Decimal(str(event["ts"])) for event in selected))
        steps = [{"name": event["name"], "start_us": float(Decimal(str(event["ts"])) - origin),
                  "duration_us": float(event["dur"])} for event in selected]
        if len(steps) != 2:
            raise ValueError(f"Expected two decode steps, got {len(steps)} for rank{rank}")
        kernels = list(csv.DictReader(io.StringIO(read_bytes(kernel_path).decode())))
        device = collections.defaultdict(lambda: {"count": 0, "total_us": 0.0})
        for step in steps:
            start, end = step["start_us"], step["start_us"] + step["duration_us"]
            all_intervals, compute_intervals, communication_intervals = [], [], []
            for row in kernels:
                kernel_start = float(Decimal(row["Start Time(us)"].strip()) - origin)
                kernel_end = kernel_start + float(row["Duration(us)"])
                clipped = (max(start, kernel_start), min(end, kernel_end))
                if clipped[1] <= clipped[0]:
                    continue
                all_intervals.append(clipped)
                operation = row["Type"]
                communication = operation.startswith(("MoeLowLatencyDispatch", "MoeLowLatencyCombine", "hcom_")) or "aicpukernel" in operation.lower()
                (communication_intervals if communication else compute_intervals).append(clipped)
                device[operation]["count"] += 1
                device[operation]["total_us"] += clipped[1] - clipped[0]
            step.update(kernel_coverage_us=union_us(all_intervals), compute_kernel_coverage_us=union_us(compute_intervals),
                        communication_kernel_coverage_us=union_us(communication_intervals),
                        uncovered_us=step["duration_us"] - union_us(all_intervals))
        host = collections.defaultdict(lambda: {"calls": 0, "self_us": 0.0})
        copies = collections.defaultdict(lambda: {"calls": 0, "self_us": 0.0})
        for row in csv.DictReader(io.StringIO(read_bytes(operator_path).decode())):
            name = "step[DECODE]" if row["Name"].startswith("step[DECODE") else row["Name"]
            host[name]["calls"] += 1
            host[name]["self_us"] += float(row["Host Self Duration(us)"])
            if name == "aten::copy_":
                copies[row["Input Shapes"]]["calls"] += 1
                copies[row["Input Shapes"]]["self_us"] += float(row["Host Self Duration(us)"])
        outputs.append({"rank": rank, "timestamp_origin_us": str(origin), "steps": steps, "kernels": dict(device), "host_scopes": dict(host), "copy_shapes": dict(copies),
                        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(read_bytes(path)).hexdigest()
                                          for path in (info, kernel_path, operator_path, trace_path)}})
    if sorted(row["rank"] for row in outputs) != list(range(16)):
        raise ValueError("Need all16ranks")
    result = {"scope": "Instrumented CPU/NPU trace attribution, not a calibrated latency table",
              "limits": ["Profiled forwards take roughly580-596ms, versus roughly400-420ms unprofiled; do not use profiler wall times as serving ground truth.",
                         "DeepEP dispatch kernel duration includes inter-rank waiting; it is not solely transport or matrix compute.",
                         "Kernel category unions can overlap and must not be added. Uncovered time also includes DMA/event activity absent from kernel_details.csv.",
                         "Host operator_details includes some scheduling outside the two main steps; kernel intervals are clipped to the recorded model steps."],
              "ranks": sorted(outputs, key=lambda row: row["rank"])}
    (ROOT / "operator-trace-analysis-20261103.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in result["ranks"]:
        means = {key: sum(step[key] for step in row["steps"]) / 2000 for key in ("duration_us", "compute_kernel_coverage_us", "communication_kernel_coverage_us", "uncovered_us")}
        print(json.dumps({"rank": row["rank"], "mean_ms": means}))


if __name__ == "__main__":
    main()
