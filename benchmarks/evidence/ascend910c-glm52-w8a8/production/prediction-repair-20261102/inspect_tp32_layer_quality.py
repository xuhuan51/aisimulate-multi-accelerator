import argparse
import hashlib
import json
from pathlib import Path

from collector.sglang.ascend_layer_population_export import qualified_population


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    required, accepted, rejected, cases = {}, {}, [], []
    for name in args.manifest:
        manifest = json.loads((args.campaign / name).read_text())
        if manifest["status"] != "completed":
            raise ValueError("Quality gate requires a completed campaign")
        for case in manifest["cases"]:
            records = [json.loads((args.campaign / case["id"] / f"layer-replay-rank{rank}.json").read_text())
                       for rank in range(32)]
            control = {key: case[key] for key in ("id", "forward_call", "training", "input_length", "global_requests")}
            rows = [qualified_population(records, layer, control, attention_tp=1, attention_dp=32, moe_ep=32)[0]
                    for layer in (0, 1, 2, 3, 4, 5, 6, 77)]
            cases.append({"case": case["id"], "training": case["training"], "max_cv": max(row["cv"] for row in rows)})
            if not case["training"]:
                continue
            for row in rows:
                key = (row["layer"], row["active_attention_dp"], row["requests_per_dp"], case["input_length"])
                required.setdefault(key, case)
                if row["cv"] > 0.1:
                    rejected.append({"case": case["id"], "coordinate": key, "cv": row["cv"]})
                else:
                    accepted.setdefault(key, {"case": case["id"], "row": row})
    missing = sorted(required.keys() - accepted.keys())
    retry = {required[key]["id"]: required[key] for key in missing}
    result = {"qualified": not missing, "cases": cases, "accepted_cells": len(accepted),
              "missing_cells": missing, "rejected": rejected,
              "retry_cases": [{key: case[key] for key in ("id", "input_length", "global_requests", "training")}
                              for case in retry.values()],
              "scope": "First numerically valid observation per cell with CV<=0.1; no serving-error-based selection",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("qualified", "accepted_cells", "missing_cells")}), flush=True)


if __name__ == "__main__":
    main()
