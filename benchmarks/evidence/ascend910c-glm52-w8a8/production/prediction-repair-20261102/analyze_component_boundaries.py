import argparse
import collections
import gzip
import hashlib
import json
import statistics
from pathlib import Path


COUNTS = {0: 1, 1: 1, 2: 1, 3: 1, 4: 37, 5: 18, 6: 18, 77: 1}


def read(path):
    payload = path.read_bytes() if path.exists() else gzip.decompress(path.with_suffix(path.suffix + ".gz").read_bytes())
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign
    manifest, manifest_sha = read(root / "component-campaign-verified.json")
    if manifest["status"] != "completed":
        raise ValueError("A complete component campaign is required")
    result = {"schema": "aisimulate-component-diagnosis-summary/v1", "cases": [],
              "source_sha256": {"component-campaign-verified.json": manifest_sha},
              "limits": ["Diagnostic invocations are not client request latency or whole-forward validation.",
                         "Nested spans overlap. Only root spans are aggregated per component.",
                         "NPU event instrumentation perturbs host submission. These spans are not imported into performance tables.",
                         "The 78-layer combination is a diagnostic composition, not a measured whole forward."]}
    for case in manifest["cases"]:
        if case["status"] != "completed":
            raise ValueError("Incomplete component case")
        rank_results = []
        for rank in range(16):
            relative = f"{case['id']}/components-rank{rank}.json"
            observed, digest = read(root / relative)
            result["source_sha256"][relative] = digest
            layers = []
            totals = collections.defaultdict(float)
            for layer in observed["layers"]:
                medians = {}
                components = {}
                for population in layer["populations"]:
                    samples = [sample for sample in population["samples"] if not sample["warmup"]]
                    medians[population["mode"]] = {
                        key: statistics.median(sample[key] for sample in samples)
                        for key in ("device_ms", "host_ms", "host_cpu_ms")}
                    if population["mode"] != "host":
                        continue
                    groups = collections.defaultdict(list)
                    for sample in samples:
                        values = collections.defaultdict(float)
                        for operation in sample["operations"]:
                            if operation["parent"] is None:
                                key = operation["operation"] + ":" + operation.get("stage", "")
                                values[key] += operation["host_ms"]
                        for key, value in values.items():
                            groups[key].append(value)
                    components = {key: statistics.median(values) for key, values in groups.items()}
                baseline = (medians["control_before"]["device_ms"] + medians["control_after"]["device_ms"]) / 2
                layers.append({"layer": layer["layer"], "sequence_lengths": layer["sequence_lengths"],
                               "modes": medians, "root_host_components_ms": components,
                               "event_perturbation_pct": 100 * (medians["events"]["device_ms"] / baseline - 1),
                               "host_probe_perturbation_pct": 100 * (medians["host"]["device_ms"] / baseline - 1)})
                for mode, measurements in medians.items():
                    for key, value in measurements.items():
                        totals[mode + ":" + key] += COUNTS[layer["layer"]] * value
            rank_results.append({"rank": rank, "layers": layers, "combined_78_layers_ms": dict(totals)})
        case_result = {"id": case["id"], "input_length": case["input_length"],
                       "global_requests": case["global_requests"], "ranks": rank_results}
        result["cases"].append(case_result)
    destination = root / "component-summary.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    for case in result["cases"]:
        rank = case["ranks"][0]
        totals = rank["combined_78_layers_ms"]
        baseline = (totals["control_before:device_ms"] + totals["control_after:device_ms"]) / 2
        host_cpu = (totals["control_before:host_cpu_ms"] + totals["control_after:host_cpu_ms"]) / 2
        print(json.dumps({"id": case["id"], "rank0_combined_layers_ms": baseline,
                          "rank0_host_cpu_ms": host_cpu,
                          "events_perturbation_pct": 100 * (totals["events:device_ms"] / baseline - 1)}))


if __name__ == "__main__":
    main()
