import argparse
import collections
import csv
import hashlib
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node0", type=Path, required=True)
    parser.add_argument("--node1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = {}
    ranks = {}
    totals = {}
    for node, root in enumerate((args.node0, args.node1)):
        for path in sorted(root.rglob("kernel_details.csv")):
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            devices = {int(row["Device_id"]) for row in rows}
            if len(devices) != 1:
                raise ValueError("Expected one device per trace")
            rank = node * 16 + devices.pop()
            if rank in ranks:
                raise ValueError("Duplicate trace rank")
            events = sorted(({"start": float(row["Start Time(us)"]),
                              "end": float(row["Start Time(us)"]) + float(row["Duration(us)"]),
                              "duration": float(row["Duration(us)"]), "kind": row["Type"],
                              "inputs": row["Input Shapes"]} for row in rows), key=lambda row: row["start"])
            by_type = collections.defaultdict(list)
            for event in events:
                by_type[event["kind"]].append(event["duration"])
            totals[rank] = {kind: {"calls": len(values), "per_forward_ms": sum(values) / 2000,
                                   "median_call_ms": statistics.median(values) / 1000}
                            for kind, values in by_type.items()}
            dispatches = [row for row in events if row["kind"] == "MoeLowLatencyDispatchV2"]
            combines = [row for row in events if row["kind"] == "MoeLowLatencyCombineV2"]
            if len(dispatches) != 150 or len(combines) != 150:
                raise ValueError("Expected two complete low-latency Decode forwards")
            layers = []
            for index, (dispatch, combine) in enumerate(zip(dispatches, combines)):
                if combine["start"] < dispatch["end"]:
                    raise ValueError("Invalid dispatch/combine ordering")
                between = [row for row in events if dispatch["end"] <= row["start"] < combine["start"]]
                grouped = [row for row in between if row["kind"] == "GroupedMatmul"]
                if len(grouped) != 2:
                    raise ValueError("Expected two grouped matrix multiplications")
                layers.append({"step": index // 75, "layer": index % 75 + 3,
                               "dispatch_ms": dispatch["duration"] / 1000,
                               "grouped_gemm_ms": sum(row["duration"] for row in grouped) / 1000,
                               "dispatch_end_to_combine_start_ms": (combine["start"] - dispatch["end"]) / 1000,
                               "combine_ms": combine["duration"] / 1000,
                               "dispatch_start_us": dispatch["start"], "dispatch_end_us": dispatch["end"],
                               "combine_start_us": combine["start"], "combine_end_us": combine["end"],
                               "dispatch_inputs": dispatch["inputs"], "combine_inputs": combine["inputs"]})
            ranks[rank] = layers
    if set(ranks) != set(range(32)):
        raise ValueError("Expected complete32-rank capture")
    groups = []
    components = ("dispatch_ms", "grouped_gemm_ms", "dispatch_end_to_combine_start_ms", "combine_ms")
    latest = [collections.Counter(), collections.Counter()]
    for index in range(150):
        group = {"step": index // 75, "layer": index % 75 + 3}
        for key in components:
            values = [ranks[rank][index][key] for rank in range(32)]
            group[key] = {"min": min(values), "mean": statistics.mean(values),
                          "median": statistics.median(values), "max": max(values)}
        group["same_host_spreads_ms"] = []
        for node in range(2):
            local_ranks = range(node * 16, (node + 1) * 16)
            latest[node][max(local_ranks, key=lambda rank: ranks[rank][index]["dispatch_start_us"])] += 1
            group["same_host_spreads_ms"].append({
                key: (max(ranks[rank][index][key] for rank in local_ranks)
                      - min(ranks[rank][index][key] for rank in local_ranks)) / 1000
                for key in ("dispatch_start_us", "dispatch_end_us", "combine_start_us", "combine_end_us")})
        groups.append(group)
    result = {"scope": "Profiled diagnosis only; not serving accuracy or training data. Kernel durations include peer readiness waits. Cross-host absolute clocks are not compared; per-layer min/max sums need not describe a realizable critical path.",
              "source_sha256": sources, "ranks": ranks, "kernel_totals": totals, "layers": groups,
              "same_host_latest_dispatch_rank": latest,
              "per_forward_layer_sum": {key: {stat: sum(group[key][stat] for group in groups) / 2
                                               for stat in ("min", "mean", "median", "max")}
                                         for key in components},
              "same_host_median_spreads_ms": [{key: statistics.median(group["same_host_spreads_ms"][node][key] for group in groups)
                                                 for key in ("dispatch_start_us", "dispatch_end_us", "combine_start_us", "combine_end_us")}
                                                for node in range(2)]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("per_forward_layer_sum", "same_host_median_spreads_ms", "same_host_latest_dispatch_rank")}, indent=2))


if __name__ == "__main__":
    main()
