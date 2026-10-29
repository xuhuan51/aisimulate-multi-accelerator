"""Declare missing eager MAX_LEN active/padded planes for an existing campaign."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as parquet

COORDINATES = (
    "operation",
    "phase",
    "local_heads",
    "execution_mode",
    "active_batch",
    "padded_batch",
    "sequence_length",
    "prefix",
)


def coordinate(case):
    return tuple(case[name] for name in COORDINATES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--measured", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batch", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists() or args.max_batch < 2:
        parser.error("Use a fresh output path and a positive batch range")
    base = json.loads(args.source_plan.read_text())
    eligible = [case for case in base["cases"] if case["phase"] == "decode" and case["execution_mode"] == "eager"]
    minimum_pool = min(case["pool_pages"] for case in eligible)
    total_heads = max(case["local_heads"] for case in eligible if case["operation"] == "sparse_attention")
    templates = {(case["operation"], case["local_heads"], case["sequence_length"]): case for case in eligible}
    measured = {coordinate(row) for row in parquet.read_table(args.measured, columns=list(COORDINATES)).to_pylist()}
    cases = []
    for (operation, heads, length), template in sorted(templates.items()):
        alignment = 1 if operation == "lightning_indexer" else total_heads // heads
        maximum = (args.max_batch + alignment - 1) // alignment * alignment
        for padded in range(alignment, maximum + 1, alignment):
            for active in sorted({1, (padded + 1) // 2, padded - 1}):
                if not 0 < active < padded or active > args.max_batch:
                    continue
                case = {
                    **template,
                    "active_batch": active,
                    "padded_batch": padded,
                    "tokens": padded,
                    "pool_pages": max(
                        minimum_pool, 1 + active * ((length + template["page_size"] - 1) // template["page_size"])
                    ),
                }
                if coordinate(case) not in measured:
                    cases.append(case)
    if len({coordinate(case) for case in cases}) != len(cases):
        raise ValueError("Duplicate extension coordinate")
    plan = {
        **base,
        "status": "experimental_collection_plan",
        "cases": cases,
        "extension": {
            "scope": "Eager MAX_LEN; minimum, middle and upper active-batch anchors per padding plane",
            "source_plan_sha256": hashlib.sha256(args.source_plan.read_bytes()).hexdigest(),
            "measured_sha256": hashlib.sha256(args.measured.read_bytes()).hexdigest(),
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "max_batch": args.max_batch,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "output": str(args.output), "extension": plan["extension"]}))


if __name__ == "__main__":
    main()
