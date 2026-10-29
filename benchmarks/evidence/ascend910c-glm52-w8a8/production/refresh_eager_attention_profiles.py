"""Refresh whole eager decode curves while preserving other attention observations."""

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as parquet
import yaml
from collector.provenance import write_collection_meta
from collector.sglang.ascend_curve_append import physical_coordinate, physical_curve


def leaf_paths(report, path=()):
    if "identity" in report:
        return [path]
    return [
        leaf
        for index, campaign in enumerate(report["campaigns"])
        for leaf in leaf_paths(campaign["report"], (*path, index))
    ]


def owner(report, curve, point, path=()):
    if "identity" in report:
        return path
    matching = [entry for entry in report["curves"] if tuple(entry["key"]) == curve]
    if len(matching) != 1:
        raise ValueError("Missing or ambiguous historical curve")
    entry = matching[0]
    sources = entry.get("sources", [entry])
    candidates = [source["campaign"] for source in sources if list(point) in source["points"]]
    if len(candidates) != 1:
        raise ValueError("Missing or ambiguous historical coordinate owner")
    index = candidates[0]
    return owner(report["campaigns"][index]["report"], curve, point, (*path, index))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--refresh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[4]
    revision = subprocess.check_output(
        ["git", "-c", f"safe.directory={repository}", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    name = "ascend_attention_perf"
    filename = name + ".parquet"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    before = args.output_dir / "before"
    before.mkdir()
    locations = {
        filename: filename,
        "collection_meta.yaml": "collection_meta.yaml",
        "provenance.json": "metadata/provenance.json",
    }
    for destination, source in locations.items():
        shutil.copyfile(args.live / source, before / destination)
    original = parquet.read_table(before / filename).to_pylist()
    fresh = parquet.read_table(args.refresh / filename).to_pylist()
    if not fresh or any(row["phase"] != "decode" or row["execution_mode"] != "eager" for row in fresh):
        raise ValueError("Refresh must contain only qualified eager decode curves")
    keys = {physical_curve("attention", row) for row in fresh}
    expected = {
        physical_curve("attention", row)
        for row in original
        if row["phase"] == "decode" and row["execution_mode"] == "eager"
    }
    if keys != expected:
        raise ValueError("Refresh must cover every existing eager decode curve")
    retained = [row for row in original if physical_curve("attention", row) not in keys]
    report = json.loads((before / "provenance.json").read_text())
    metadata = yaml.safe_load((before / "collection_meta.yaml").read_text())
    paths = leaf_paths(report)
    events = metadata["tables"][name]["collections"]
    counts = Counter(
        owner(report, physical_curve("attention", row), physical_coordinate("attention", row)) for row in original
    )
    if len(paths) != len(events) or [counts[path] for path in paths] != [event["rows"] for event in events]:
        raise ValueError("Historical collection events do not match coordinate ownership")
    selected_counts = Counter(
        owner(report, physical_curve("attention", row), physical_coordinate("attention", row)) for row in retained
    )
    selected_events = copy.deepcopy(events)
    for path, event in zip(paths, selected_events, strict=True):
        event.setdefault("source_campaign_rows", event["rows"])
        event.setdefault("source_campaign_status", event["status"])
        event["rows"] = selected_counts[path]
    selected = args.output_dir / "retained"
    selected.mkdir()
    import pyarrow as arrow

    parquet.write_table(arrow.Table.from_pylist(retained), selected / filename, compression="zstd")
    write_collection_meta(
        selected,
        metadata["runtime"],
        {name: {"rows": len(retained), "status": "complete", "collections": selected_events}},
        provenance_tier="collected",
    )
    report.update(
        selection="Retain whole non-eager-decode curves after an explicitly audited collector protocol refresh.",
        rows=len(retained),
        removed_curves=sorted(keys),
    )
    report["curves"] = [entry for entry in report["curves"] if tuple(entry["key"]) not in keys]
    (selected / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    merged = args.output_dir / "merged"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "collector.sglang.ascend_curve_append",
            "--operation",
            "attention",
            "--output-dir",
            str(merged),
            str(selected),
            str(args.refresh),
        ],
        check=True,
    )
    combined = parquet.read_table(merged / filename).to_pylist()
    identity = lambda row: (physical_curve("attention", row), physical_coordinate("attention", row))
    indexed = {identity(row): row for row in combined}
    if len(indexed) != len(combined) or any(indexed.get(identity(row)) != row for row in retained + fresh):
        raise ValueError("Publication changed a retained or refreshed observation")
    updated = yaml.safe_load((merged / "collection_meta.yaml").read_text())
    tables = {**metadata["tables"], **updated["tables"]}
    write_collection_meta(merged, metadata["runtime"], tables, provenance_tier="collected")
    try:
        for source, destination in locations.items():
            staged = args.live / (destination + ".new")
            shutil.copyfile(merged / source, staged)
            staged.replace(args.live / destination)
    except Exception:
        for source, destination in locations.items():
            shutil.copyfile(before / source, args.live / destination)
        raise
    manifest = {
        "old_rows": len(original),
        "retained_rows_unchanged": len(retained),
        "replaced_eager_decode_rows": len(original) - len(retained),
        "fresh_eager_decode_rows": len(fresh),
        "rows": len(combined),
        "whole_curves_replaced": sorted(keys),
        "source_revision": revision,
        "table_sha256": hashlib.sha256((args.live / filename).read_bytes()).hexdigest(),
    }
    (args.output_dir / "publication.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
