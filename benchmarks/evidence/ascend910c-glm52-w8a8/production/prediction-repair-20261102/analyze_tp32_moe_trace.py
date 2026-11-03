import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def covered_us(events, start, end):
    intervals = sorted((max(start, row["start"]), min(end, row["end"])) for row in events
                       if row["start"] < end and row["end"] > start)
    covered = 0.0
    cursor = start
    for begin, finish in intervals:
        covered += max(0.0, finish - max(begin, cursor))
        cursor = max(cursor, finish)
    return covered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node0", type=Path, required=True)
    parser.add_argument("--node1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    ranks = {}
    sources = {}
    for node, root in enumerate((args.node0, args.node1)):
        for path in root.rglob("kernel_details.csv"):
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            devices = {int(row["Device_id"]) for row in rows}
            if len(devices) != 1:
                raise ValueError("Trace must contain exactly one device")
            rank = node * 16 + devices.pop()
            if rank in ranks:
                raise ValueError("Duplicate rank trace")
            events = sorted(({"start": float(row["Start Time(us)"]),
                              "end": float(row["Start Time(us)"]) + float(row["Duration(us)"]),
                              "duration": float(row["Duration(us)"]), "kind": row["Type"],
                              "inputs": row["Input Shapes"]} for row in rows), key=lambda row: row["start"])
            dispatches = [row for row in events if row["kind"] == "CamMoeDispatchNormal"]
            combines = [row for row in events if row["kind"] == "CamMoeCombineNormal"]
            if len(dispatches) != 150 or len(combines) != 150:
                raise ValueError("Expected two complete75-MoE-layer forwards per rank")
            layers = []
            for index, (dispatch, combine) in enumerate(zip(dispatches, combines)):
                if combine["start"] <= dispatch["end"] or (index < 149 and combine["end"] > dispatches[index + 1]["start"]):
                    raise ValueError("MoE ordering failed")
                between = [row for row in events if dispatch["end"] <= row["start"] < combine["start"]]
                grouped = [row for row in between if row["kind"] == "GroupedMatmul"]
                if len(grouped) != 2:
                    raise ValueError("Expected gate-up and down grouped GEMM")
                layers.append({"step": index // 75, "layer": index % 75 + 3,
                               "received_tokens": int(combine["inputs"].lstrip('"').split(",")[0]),
                               "dispatch_ms": dispatch["duration"] / 1000,
                               "grouped_gemm_ms": sum(row["duration"] for row in grouped) / 1000,
                               "dispatch_end_to_combine_start_ms": (combine["start"] - dispatch["end"]) / 1000,
                               "uncovered_between_ms": ((combine["start"] - dispatch["end"]) - covered_us(events, dispatch["end"], combine["start"])) / 1000,
                               "combine_ms": combine["duration"] / 1000,
                               "dispatch_end_us": dispatch["end"], "combine_start_us": combine["start"],
                               "combine_end_us": combine["end"]})
            ranks[rank] = layers
    if set(ranks) != set(range(32)):
        raise ValueError("Expected32 ranks")
    groups = []
    for index in range(150):
        layers = [ranks[rank][index] for rank in range(32)]
        group = {"step": index // 75, "layer": index % 75 + 3}
        for key in ("received_tokens", "dispatch_ms", "grouped_gemm_ms", "dispatch_end_to_combine_start_ms", "uncovered_between_ms", "combine_ms"):
            values = [row[key] for row in layers]
            group[key] = {"min": min(values), "mean": statistics.mean(values), "median": statistics.median(values), "max": max(values)}
        group["same_host_spreads_ms"] = []
        for node in range(2):
            local = layers[node * 16:(node + 1) * 16]
            group["same_host_spreads_ms"].append({key: (max(row[key] for row in local) - min(row[key] for row in local)) / 1000
                                                   for key in ("dispatch_end_us", "combine_start_us", "combine_end_us")})
        groups.append(group)
    result = {"scope": "Instrumented causal diagnostic only; no fitted latency. Compare spreads only within a host. An uncovered device interval does not identify a specific host operation or independent additive delay.",
              "source_sha256": sources, "ranks": ranks, "layers": groups,
              "per_forward_layer_sum": {key: {stat: sum(group[key][stat] for group in groups) / 2
                                              for stat in ("min", "mean", "median", "max")}
                                        for key in ("dispatch_ms", "grouped_gemm_ms", "dispatch_end_to_combine_start_ms", "uncovered_between_ms", "combine_ms")}}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["per_forward_layer_sum"], indent=2))


if __name__ == "__main__":
    main()
