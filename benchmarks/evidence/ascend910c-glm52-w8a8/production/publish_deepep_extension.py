"""Extend qualified DeepEP coordinates while retaining all historical timings."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow as arrow
import pyarrow.parquet as parquet
import yaml
from collector.sglang.ascend_curve_append import physical_coordinate, physical_curve


def coordinate(row):
    return physical_curve("deepep", row), physical_coordinate("deepep", row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[4]
    live = repository / "python/aisimulate/src/aisimulate_core/systems/data/ascend910c/moe/sglang/0.5.22"
    filename = "ascend_deepep_perf.parquet"
    table_name = "ascend_deepep_perf"
    source = live / "metadata/ascend_deepep_provenance.json"
    observations = [args.campaign / "fresh-cases/node20" / f"case-{index}/observations.json" for index in range(6)]
    observations.append(args.campaign / "isolated-32/node20/observations.json")
    for observation in observations:
        peer_parts = ["node19" if part == "node20" else part for part in observation.relative_to(args.campaign).parts]
        peer = args.campaign.joinpath(*peer_parts)
        for participant in (observation, peer):
            manifest = json.loads((participant.parent / "campaign.json").read_text())
            if manifest["status"] != "completed" or manifest["returncode"] != 0 or manifest["contamination"]:
                raise ValueError("Publish only completed uncontaminated per-case campaigns on both nodes")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    before = args.output_dir / "before"
    before.mkdir()
    for path, name in (
        (live / filename, filename),
        (live / "collection_meta.yaml", "collection_meta.yaml"),
        (source, "provenance.json"),
    ):
        shutil.copyfile(path, before / name)
    prior_report = json.loads(source.read_text())
    identity = prior_report["identity"]["runtime"]
    identity_path = args.output_dir / "runtime-identity.json"
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    revision = subprocess.check_output(
        ["git", "-c", f"safe.directory={repository}", "rev-parse", "HEAD"], text=True
    ).strip()
    exported = args.output_dir / "qualified"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "collector.sglang.ascend_deepep_export",
            *map(str, observations),
            "--plan",
            str(observations[0].parent / "plan.json"),
            "--runtime-identity",
            str(identity_path),
            "--output-dir",
            str(exported),
            "--collector-ref",
            revision,
        ],
        check=True,
    )
    original = parquet.read_table(before / filename).to_pylist()
    incoming = parquet.read_table(exported / filename).to_pylist()
    old_by_key = {coordinate(row): row for row in original}
    extension = [row for row in incoming if coordinate(row) not in old_by_key]
    if not extension:
        raise ValueError("No new qualified coordinates")
    comparisons = [
        {
            "coordinate": coordinate(row),
            "historical_ms": old_by_key[coordinate(row)]["latency_ms"],
            "current_ms": row["latency_ms"],
            "relative_change": row["latency_ms"] / old_by_key[coordinate(row)]["latency_ms"] - 1,
        }
        for row in incoming
        if coordinate(row) in old_by_key
    ]
    disjoint = args.output_dir / "disjoint"
    disjoint.mkdir()
    parquet.write_table(arrow.Table.from_pylist(extension), disjoint / filename, compression="zstd")
    report = json.loads((exported / "provenance.json").read_text())
    report["coordinate_selection"] = {
        "rule": "Retain historical physical coordinates regardless of which measured latency is faster",
        "source_qualified_rows": len(incoming),
        "retained_new_coordinates": len(extension),
        "overlap_comparisons": comparisons,
    }
    (disjoint / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    metadata = yaml.safe_load((exported / "collection_meta.yaml").read_text())
    entry = metadata["tables"][table_name]
    entry.update(
        source_campaign_rows=entry["rows"],
        rows=len(extension),
    )
    (disjoint / "collection_meta.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
    merged = args.output_dir / "merged"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "collector.sglang.ascend_curve_append",
            "--operation",
            "deepep",
            "--extend-points",
            "--output-dir",
            str(merged),
            str(before),
            str(disjoint),
        ],
        check=True,
    )
    combined = parquet.read_table(merged / filename).to_pylist()
    new_by_key = {coordinate(row): row for row in combined}
    if len(new_by_key) != len(combined) or len({row["runtime_identity_sha256"] for row in combined}) != 1:
        raise ValueError("Duplicate coordinates or mixed DeepEP runtime identities")
    for row in original:
        if any(
            value != new_by_key[coordinate(row)][field]
            for field, value in row.items()
            if field != "runtime_identity_sha256"
        ):
            raise ValueError("A historical observation changed")
    old_metadata = yaml.safe_load((before / "collection_meta.yaml").read_text())
    new_metadata = yaml.safe_load((merged / "collection_meta.yaml").read_text())
    old_metadata["tables"][table_name] = new_metadata["tables"][table_name]
    staged = live / (filename + ".new")
    shutil.copyfile(merged / filename, staged)
    staged.replace(live / filename)
    shutil.copyfile(merged / "provenance.json", source)
    (live / "collection_meta.yaml").write_text(yaml.safe_dump(old_metadata, sort_keys=False))
    publication = {
        "old_rows_preserved_except_composite_identity": len(original),
        "qualified_new_campaign_rows": len(incoming),
        "added_disjoint_coordinates": len(extension),
        "total_rows": len(combined),
        "unqualified": report["unqualified"],
        "overlap_comparisons": comparisons,
        "table_sha256": hashlib.sha256((live / filename).read_bytes()).hexdigest(),
        "measurement_scope": "Identical pinned runtime and256/10/7 protocol on node19/20, fresh process per case",
        "limitations": [
            "Original multi-case native failure preserved and unresolved",
            "Uniform synthetic balanced transport; not measured nonuniform routing or end-to-end accuracy",
        ],
    }
    (args.output_dir / "publication.json").write_text(json.dumps(publication, indent=2) + "\n")
    print(json.dumps(publication, indent=2))


if __name__ == "__main__":
    main()
