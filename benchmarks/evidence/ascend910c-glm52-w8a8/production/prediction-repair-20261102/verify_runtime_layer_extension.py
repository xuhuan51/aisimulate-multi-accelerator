import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as parquet


def key(row):
    return tuple(row[name] for name in ("attention_tp", "attention_dp", "moe_ep", "active_attention_dp",
                                      "requests_per_dp", "kv_tokens", "layer", "quantization_sha256"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    before = parquet.read_table(args.prior).to_pylist()
    after = parquet.read_table(args.candidate).to_pylist()
    old = {key(row): row for row in before}
    new = {key(row): row for row in after}
    if len(old) != len(before) or len(new) != len(after):
        raise ValueError("Duplicate runtime layer coordinate")
    if len(before) != 560 or len(after) != 576:
        raise ValueError("Unexpected extension cardinality")
    if any(new.get(identity) != row for identity, row in old.items()):
        raise ValueError("A previously published timing row changed")
    additions = [row for identity, row in new.items() if identity not in old]
    expected = {(active, layer) for active in (4, 31) for layer in (0, 1, 2, 3, 4, 5, 6, 77)}
    if {(row["active_attention_dp"], row["layer"]) for row in additions} != expected:
        raise ValueError("Unexpected new population")
    for row in additions:
        if ((row["attention_tp"], row["attention_dp"], row["moe_ep"], row["requests_per_dp"], row["kv_tokens"]) != (1, 32, 32, 2, 8224)
                or not 0 <= row["cv"] <= 0.1 or row["sample_count"] != 9 or row["numerical_checks"] != 384):
            raise ValueError("New observation does not satisfy the frozen shape/quality contract")
    result = {"status": "qualified_extension_verified", "preserved_rows": len(before), "new_rows": additions,
              "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.prior, args.candidate)},
              "scope": "Table preservation, coordinate and per-cell numeric/repeatability gates only; serving accuracy is separate."}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"preserved_rows": len(before), "added_rows": len(additions), "max_cv": max(row["cv"] for row in additions)}))


if __name__ == "__main__":
    main()
