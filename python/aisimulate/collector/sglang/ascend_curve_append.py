"""Append disjoint measured operator grids without replacing existing points."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def physical_curve(operation, row):
    if operation == "linear":
        return tuple(row[field] for field in ("quant_mode", "n", "k", "weight_format", "quant_bias", "execution_mode"))
    if operation == "vector":
        dimensions = row["dimensions"]
        return (
            row["role"],
            tuple(json.loads(dimensions) if isinstance(dimensions, str) else dimensions),
            row["execution_mode"],
        )
    if operation == "bmm":
        return tuple(row[field] for field in ("role", "local_heads", "execution_mode"))
    if operation == "attention":
        return tuple(row[field] for field in ("operation", "phase", "local_heads", "execution_mode"))
    if operation == "pd":
        return tuple(row[field] for field in ("source_tp", "destination_tp", "page_layout"))
    if operation == "dcp_attention":
        return tuple(row[field] for field in ("phase", "local_heads", "dcp_size", "remap_topk", "execution_mode"))
    if operation == "deepep":
        return tuple(
            row[field]
            for field in ("phase", "operation", "num_ranks", "ranks_per_node", "distribution", "execution_mode")
        )
    raise ValueError("Unsupported Ascend operator grid")


def physical_coordinate(operation, row):
    if operation in ("attention", "dcp_attention"):
        return tuple(row[field] for field in ("active_batch", "padded_batch", "sequence_length", "prefix"))
    if operation == "deepep":
        return row["tokens_per_rank"]
    return row["m" if operation == "linear" else "tokens"]


def collection_identities(report):
    identities = dict(report.get("merged_identities", {}))
    if "identity" in report:
        identity = report["identity"]
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        identities[digest] = identity
        return identities
    for source in report["campaigns"]:
        identities.update(collection_identities(source["report"]))
    return identities


def append_curves(operation, campaigns):
    if not campaigns:
        raise ValueError("At least one verified campaign is required")
    runtime = campaigns[0]["metadata"]["runtime"]
    rows, decisions, owners, counts = [], [], set(), Counter()
    setup = None
    hardware = None
    for index, campaign in enumerate(campaigns):
        if campaign["metadata"]["runtime"] != runtime:
            raise ValueError("Different runtime conditions cannot share a table")
        identities = collection_identities(campaign["report"])
        for identity in identities.values():
            current_hardware = identity["runtime"].get("hardware")
            if current_hardware is None or (hardware is not None and hardware != current_hardware):
                raise ValueError("Different or missing hardware identities cannot share a table")
            hardware = current_hardware
            current_setup = identity["serving_setup"]
            if setup is not None and setup != current_setup:
                raise ValueError("Different serving setups cannot share a table")
            setup = current_setup
        groups = defaultdict(list)
        for row in campaign["rows"]:
            if row["runtime_identity_sha256"] not in identities:
                raise ValueError("Table row has no verified collection identity")
            groups[physical_curve(operation, row)].append(row)
        for key, values in sorted(groups.items()):
            if key in owners:
                raise ValueError(
                    "Overlapping curves require a separate coverage decision; point replacement is forbidden"
                )
            owners.add(key)
            samples = [physical_coordinate(operation, row) for row in values]
            identities_in_curve = {row["runtime_identity_sha256"] for row in values}
            if len(samples) != len(set(samples)) or len(identities_in_curve) != 1:
                raise ValueError("A curve cannot contain duplicate points or mixed collection protocols")
            rows.extend(sorted(values, key=lambda row: physical_coordinate(operation, row)))
            counts[index] += len(values)
            decisions.append(
                {
                    "key": key,
                    "campaign": index,
                    "points": sorted(samples),
                    "collection_identity_sha256": next(iter(identities_in_curve)),
                }
            )
    return rows, decisions, counts


def extend_attention_points(campaigns):
    return extend_points("attention", campaigns)


def extend_points(operation, campaigns):
    if not campaigns:
        raise ValueError("At least one verified campaign is required")
    rows, decisions, counts = [], [], Counter()
    identities, groups = {}, defaultdict(list)
    runtime = campaigns[0]["metadata"]["runtime"]
    setup, hardware = None, None
    for index, campaign in enumerate(campaigns):
        if campaign["metadata"]["runtime"] != runtime:
            raise ValueError("Different runtime conditions cannot share a table")
        verified = collection_identities(campaign["report"])
        for digest, identity in verified.items():
            actual = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if actual != digest:
                raise ValueError("Collection identity digest mismatch")
        identities.update(verified)
        for row in campaign["rows"]:
            digest = row["runtime_identity_sha256"]
            if digest not in verified:
                raise ValueError("Table row has no verified collection identity")
            identity = verified[digest]
            current_hardware = identity["runtime"].get("hardware")
            current_setup = identity["serving_setup"]
            if current_hardware is None or (hardware is not None and current_hardware != hardware):
                raise ValueError("Different or missing hardware identities cannot share a table")
            if setup is not None and current_setup != setup:
                raise ValueError("Different serving setups cannot share a table")
            hardware, setup = current_hardware, current_setup
            groups[physical_curve(operation, row)].append((index, row))
            counts[index] += 1
    merged_identities = {}
    table_sources = (
        sorted({row["runtime_identity_sha256"] for values in groups.values() for _, row in values})
        if operation == "deepep"
        else None
    )
    for key, values in sorted(groups.items()):
        coordinates = [physical_coordinate(operation, row) for _, row in values]
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("Overlapping physical points cannot replace existing measurements")
        source_ids = table_sources or sorted({row["runtime_identity_sha256"] for _, row in values})
        first = identities[source_ids[0]]
        conditions = {name: first[name] for name in ("runtime", "protocol", "serving_setup")}
        if any({name: identities[digest][name] for name in conditions} != conditions for digest in source_ids):
            raise ValueError("Extended curves require identical runtime, protocol and serving setup")
        if len(source_ids) == 1:
            merged_digest = source_ids[0]
        else:
            merged = {**conditions, "coverage_sources": source_ids}
            merged_bytes = json.dumps(merged, sort_keys=True, separators=(",", ":")).encode()
            merged_digest = hashlib.sha256(merged_bytes).hexdigest()
            merged_identities[merged_digest] = merged
        for index, row in sorted(values, key=lambda item: physical_coordinate(operation, item[1])):
            rows.append({**row, "runtime_identity_sha256": merged_digest})
        decisions.append(
            {
                "key": key,
                "points": sorted(coordinates),
                "collection_identity_sha256": merged_digest,
                "sources": [
                    {
                        "campaign": index,
                        "collection_identity_sha256": digest,
                        "points": sorted(
                            physical_coordinate(operation, row)
                            for owner, row in values
                            if owner == index and row["runtime_identity_sha256"] == digest
                        ),
                    }
                    for index, digest in sorted({(owner, row["runtime_identity_sha256"]) for owner, row in values})
                ],
            }
        )
    return rows, decisions, counts, merged_identities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operation", choices=("linear", "vector", "bmm", "attention", "pd", "dcp_attention", "deepep"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extend-attention-points", action="store_true")
    parser.add_argument("--extend-points", action="store_true")
    parser.add_argument("campaign", type=Path, nargs="+")
    args = parser.parse_args()
    if args.operation == "deepep" and not args.extend_points:
        parser.error("DeepEP requires --extend-points to retain one verified table-wide runtime identity")
    import pyarrow as arrow
    import pyarrow.parquet as parquet
    import yaml
    from collector.provenance import append_collection_event, write_collection_meta

    name = "ascend_" + args.operation + "_perf"
    campaigns = []
    for directory in args.campaign:
        provenance = directory / "provenance.json"
        if not provenance.is_file():
            provenance = directory / "metadata/provenance.json"
        table = directory / (name + ".parquet")
        campaigns.append(
            {
                "report": json.loads(provenance.read_text()),
                "metadata": yaml.safe_load((directory / "collection_meta.yaml").read_text()),
                "rows": parquet.read_table(table).to_pylist(),
                "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
            }
        )
    merged_identities = {}
    if args.extend_attention_points or args.extend_points:
        if args.extend_attention_points and args.operation != "attention":
            parser.error("Point extension is qualified only for attention")
        rows, decisions, counts, merged_identities = extend_points(args.operation, campaigns)
    else:
        rows, decisions, counts = append_curves(args.operation, campaigns)
    entry, total = None, 0
    for index, contributed in sorted(counts.items()):
        metadata = campaigns[index]["metadata"]
        source = metadata["tables"][name]
        events = source.get("collections", [source])
        if source["rows"] != contributed or sum(event["rows"] for event in events) != contributed:
            raise ValueError("Disjoint grid history must account for every retained row")
        for original in events:
            event = dict(original)
            event.setdefault("source_campaign_rows", event["rows"])
            event.setdefault("source_campaign_status", event["status"])
            event.setdefault("runtime", metadata["runtime"])
            total += event["rows"]
            entry = event if entry is None else append_collection_event(entry, event, table=name, merged_rows=total)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    parquet.write_table(arrow.Table.from_pylist(rows), args.output_dir / (name + ".parquet"), compression="zstd")
    write_collection_meta(
        args.output_dir, campaigns[0]["metadata"]["runtime"], {name: entry}, provenance_tier="collected"
    )
    report = {
        "selection": (
            "Extend disjoint physical coordinates under identical runtime, protocol and serving setup; "
            "preserve every timing and source hash, join collection identities with explicit coverage_sources."
            if args.extend_attention_points or args.extend_points
            else "Append disjoint whole curves; existing curves are retained unchanged. Overlaps are rejected."
        ),
        "curves": decisions,
        "campaigns": [
            {"table_sha256": campaign["table_sha256"], "report": campaign["report"]} for campaign in campaigns
        ],
        "rows": len(rows),
    }
    if merged_identities:
        report["merged_identities"] = merged_identities
    (args.output_dir / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "curves": len(decisions)}))


if __name__ == "__main__":
    main()
