import argparse
import hashlib
import json
from pathlib import Path

from collector.sglang.ascend_layer_population_export import qualified_population


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--manifest", action="append")
    parser.add_argument("--campaign-set", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.campaign_set:
        if args.campaign or args.manifest:
            raise ValueError("Use either a campaign set or a single campaign")
        campaigns = json.loads(args.campaign_set.read_text())
    elif args.campaign and args.manifest:
        campaigns = [{"campaign": str(args.campaign), "manifests": args.manifest}]
    else:
        raise ValueError("A campaign set or a campaign with manifests is required")
    required, accepted, rejected, cases = {}, {}, [], []
    for campaign, name in ((Path(item["campaign"]), name) for item in campaigns for name in item["manifests"]):
        manifest = json.loads((campaign / name).read_text())
        if manifest["status"] != "completed":
            raise ValueError("Quality gate requires a completed campaign")
        for case in manifest["cases"]:
            records = [json.loads((campaign / case["id"] / f"layer-replay-rank{rank}.json").read_text())
                       for rank in range(32)]
            control = {key: case[key] for key in ("id", "forward_call", "training", "input_length", "global_requests")}
            rows = [qualified_population(records, layer, control, attention_tp=1, attention_dp=32, moe_ep=32)[0]
                    for layer in (0, 1, 2, 3, 4, 5, 6, 77)]
            contexts = {row["kv_tokens"] for row in rows}
            if len(contexts) != 1:
                raise ValueError("A captured layer population must share one actual KV coordinate")
            maximum_cv = max(row["cv"] for row in rows)
            cases.append({"campaign": str(campaign), "case": case["id"], "training": case["training"],
                          "max_cv": maximum_cv, "kv_tokens": next(iter(contexts))})
            if not case["training"]:
                continue
            for row in rows:
                coordinate = (row["layer"], row["active_attention_dp"], row["requests_per_dp"], case["input_length"])
                required.setdefault(coordinate, case)
                if row["cv"] > 0.1:
                    rejected.append({"campaign": str(campaign), "case": case["id"],
                                     "coordinate": coordinate, "cv": row["cv"]})
                else:
                    accepted.setdefault(coordinate, {"campaign": str(campaign), "case": case["id"],
                        "coordinate": coordinate, "kv_tokens": row["kv_tokens"]})
    missing = sorted(required.keys() - accepted.keys())
    retry = {required[key]["id"]: required[key] for key in missing}
    result = {"qualified": not missing, "cases": cases, "accepted_cells": len(accepted),
              "selected_cases": list(accepted.values()), "missing_populations": sorted({key[1:] for key in missing}),
              "missing_cells": missing,
              "rejected": rejected,
              "retry_cases": [{key: case[key] for key in ("id", "input_length", "global_requests", "training")}
                              for case in retry.values()],
              "scope": "First numerically valid layer cell with CV<=0.1; preserve each layer's actual KV coordinate for independent Rust attention/indexer anchoring; no serving-error-based selection",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("qualified", "accepted_cells", "missing_cells")}), flush=True)


if __name__ == "__main__":
    main()
