import argparse
import hashlib
import json
from pathlib import Path

import yaml

from collector.sglang.ascend_layer_population_export import export_population


LAYERS = {0, 1, 2, 3, 4, 5, 6, 77}


def complete_populations(quality):
    populations = {}
    for cell in quality["selected_cases"]:
        layer, active, requests, context = cell["coordinate"]
        populations.setdefault((active, requests, context), set()).add(layer)
    complete = {key for key, layers in populations.items() if layers == LAYERS}
    missing = {tuple(cell[1:]) for cell in quality["missing_cells"]}
    if not missing or complete & missing:
        raise ValueError("An explicit unresolved quality gap is required")
    anchors = {}
    for active, requests, context in missing:
        candidates = sorted(length for candidate_active, candidate_requests, length in complete
                            if (candidate_active, candidate_requests) == (active, requests) and length >= 2048)
        if context < 2048 or not candidates:
            raise ValueError("A missing population has no independently qualified sparse-context anchor")
        anchors[(active, requests, context)] = candidates
    return complete, anchors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--campaign-set", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collector-ref", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    quality = json.loads(args.quality.read_text())
    complete, anchors = complete_populations(quality)
    selected = {(item["campaign"], item["case"]) for item in quality["selected_cases"]
                if tuple(item["coordinate"][1:]) in complete}
    retained, originals = [], []
    for item in json.loads(args.campaign_set.read_text()):
        campaign = Path(item["campaign"])
        names = []
        for name in item["manifests"]:
            source = campaign / name
            manifest = json.loads(source.read_text())
            if manifest["status"] != "completed":
                raise ValueError("Raw collection must complete, including recorded timing-quality failures")
            cases = [case for case in manifest["cases"]
                     if not case["training"] or (str(campaign), case["id"]) in selected]
            originals.append({"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
            if not cases:
                continue
            output = source.with_name(source.stem + "-complete-population-subset.json")
            if output.exists():
                raise FileExistsError(output)
            manifest.update(cases=cases, selection={"source": str(source), "quality": str(args.quality),
                "policy": "Publish only complete qualified layer populations; preserve excluded variance failures and require another measured KV anchor for the same request count and topology"})
            output.write_text(json.dumps(manifest, indent=2) + "\n")
            names.append(output.name)
        if names:
            retained.append({"campaign": str(campaign), "manifests": names})
    campaign_set = args.quality.with_name("complete-population-subset-campaigns.json")
    if campaign_set.exists():
        raise FileExistsError(campaign_set)
    campaign_set.write_text(json.dumps(retained, indent=2) + "\n")
    result = export_population(campaign_set, args.model_config, args.output, args.collector_ref)
    if result["rows"] != len(complete) * 8:
        raise ValueError("Published coverage differs from the qualified complete populations")
    audit = {"planned_measurements_complete": False, "published_cells": result["rows"],
             "qualified_cells_before_population_selection": quality["accepted_cells"],
             "excluded_populations": [{"coordinate": key, "qualified_contexts": contexts} for key, contexts in sorted(anchors.items())],
             "quality_report": str(args.quality), "quality_report_sha256": hashlib.sha256(args.quality.read_bytes()).hexdigest(),
             "original_manifests": originals, "rejected": quality["rejected"],
             "policy": "First qualified cell, CV<=0.1 and all-rank numerical gates unchanged. One unstable population is explicitly excluded; its prediction requires the existing Rust context-transfer model and separate validation. No serving-error-based selection."}
    path = args.output / "metadata/provenance.json"
    provenance = json.loads(path.read_text())
    provenance["population_quality_selection"] = audit
    path.write_text(json.dumps(provenance, indent=2) + "\n")
    path = args.output / "collection_meta.yaml"
    metadata = yaml.safe_load(path.read_text())
    metadata["planned_measurements_complete"] = False
    metadata["excluded_unstable_populations"] = audit["excluded_populations"]
    path.write_text(yaml.safe_dump(metadata, sort_keys=False))
    print(json.dumps({**result, "qualification": audit}, indent=2))


if __name__ == "__main__":
    main()
