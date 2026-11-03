import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import pyarrow as arrow
import pyarrow.parquet as parquet
import yaml


FILES = ("ascend_layer_perf.parquet", "collection_meta.yaml", "metadata/provenance.json")
TOPOLOGY = ("attention_tp", "attention_dp", "moe_ep")
COORDINATE = (*TOPOLOGY, "layer", "active_attention_dp", "requests_per_dp", "kv_tokens")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coordinate(row):
    return tuple(row[key] for key in COORDINATE)


def merge(inputs, output):
    if output.exists():
        raise FileExistsError(output)
    rows, accepted, rejected, holdouts, identities, declarations = [], [], [], [], [], []
    owners, source_hashes, sources = set(), {}, []
    plan_statuses, excluded_populations = [], []
    common_metadata, schema = None, None
    for root in inputs:
        metadata = yaml.safe_load((root / FILES[1]).read_text())
        report = json.loads((root / FILES[2]).read_text())
        table = parquet.read_table(root / FILES[0])
        current = table.to_pylist()
        identity = report["runtime_identity"]
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        topology = {tuple(row[key] for key in TOPOLOGY) for row in current}
        if len(topology) != 1 or not topology.isdisjoint(owners):
            raise ValueError("Inputs must contain disjoint, independently qualified topologies")
        actual_topology = next(iter(topology))
        plan_statuses.append(metadata.get("planned_measurements_complete", True))
        excluded_populations.extend({"topology": actual_topology, **item}
                                    for item in metadata.get("excluded_unstable_populations", []))
        if actual_topology not in ((4, 4, 16), (1, 32, 32)):
            raise ValueError("Unaudited runtime-layer topology")
        if (
            metadata["schema_version"] != 2
            or metadata["provenance"] != "collected"
            or metadata["boundary_contract"] != "glm52_eager_decoder_layer_valid_rows/v2"
            or metadata["serving_accuracy_acceptance"] is not False
            or metadata["tables"]["ascend_layer_perf"]["status"] != "complete"
            or metadata["tables"]["ascend_layer_perf"]["rows"] != len(current)
        ):
            raise ValueError("Incomplete or unsupported collection declaration")
        if common_metadata is None:
            common_metadata, schema = metadata, table.schema
        elif metadata["runtime"] != common_metadata["runtime"] or table.schema != schema:
            raise ValueError("Cannot merge different runtimes or row schemas")
        observed = [entry["row"] for entry in report["accepted"]]
        if sorted(current, key=coordinate) != sorted(observed, key=coordinate):
            raise ValueError("Published rows differ from accepted numerical observations")
        if len({coordinate(row) for row in current}) != len(current):
            raise ValueError("Duplicate runtime-layer coordinates")
        for row in current:
            if (
                row["runtime_identity_sha256"] != signature
                or row["quantization_sha256"] != identity["quantization_sha256"]
                or row["sample_count"] < 9
                or row["numerical_checks"] < actual_topology[0] * actual_topology[1] * 9
                or not math.isfinite(row["latency_ms"])
                or row["latency_ms"] <= 0
                or not math.isfinite(row["cv"])
                or not 0 <= row["cv"] <= 0.1
            ):
                raise ValueError("Unqualified runtime-layer observation")
        for path, signature in report["source_sha256"].items():
            if path in source_hashes and source_hashes[path] != signature:
                raise ValueError("Conflicting source identity")
            source_hashes[path] = signature
        sources.append({"root": str(root), "rows": len(current), "topology": actual_topology,
                        "sha256": {name: digest(root / name) for name in FILES}})
        identities.append({"topology": actual_topology, "runtime_identity": identity})
        owners.update(topology)
        rows.extend(current)
        accepted.extend(report["accepted"])
        rejected.extend(report["rejected"])
        holdouts.extend(report["holdouts_excluded"])
        declarations.extend(metadata["tables"]["ascend_layer_perf"]["collections"])
    if not rows:
        raise ValueError("No qualified observations")
    metadata = copy.deepcopy(common_metadata)
    metadata["tables"]["ascend_layer_perf"].update(rows=len(rows), collections=declarations)
    metadata["scope"] = "Independent exact-topology layer invocations; disjoint union, no timing averaging or residual fit"
    metadata["planned_measurements_complete"] = all(plan_statuses)
    metadata["excluded_unstable_populations"] = excluded_populations
    output.mkdir(parents=True)
    (output / "metadata").mkdir()
    parquet.write_table(arrow.Table.from_pylist(sorted(rows, key=coordinate), schema=schema), output / FILES[0])
    (output / FILES[1]).write_text(yaml.safe_dump(metadata, sort_keys=False))
    quality = [{"input": str(root), **json.loads((root / FILES[2]).read_text())["population_quality_selection"]}
               for root in inputs if "population_quality_selection" in json.loads((root / FILES[2]).read_text())]
    report = {"runtime_identities_by_topology": identities, "source_sha256": source_hashes,
              "population_quality_selections": quality,
              "accepted": accepted, "rejected": rejected, "holdouts_excluded": holdouts,
              "inputs": sources, "merger_source_sha256": digest(Path(__file__)),
              "row_preservation": "Every input row retained exactly, including timing and numerical identity",
              "serving_accuracy_acceptance": False}
    (output / FILES[2]).write_text(json.dumps(report, indent=2) + "\n")
    reread = parquet.read_table(output / FILES[0]).to_pylist()
    if reread != sorted(rows, key=coordinate):
        raise ValueError("Publication changed input observations")
    return {"rows": len(rows), "topologies": sorted(owners), "all_input_rows_preserved": True,
            "serving_accuracy_acceptance": False,
            "output_sha256": {name: digest(output / name) for name in FILES}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
