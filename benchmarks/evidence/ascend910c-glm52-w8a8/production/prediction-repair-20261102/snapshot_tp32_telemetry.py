import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def snapshot(source, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    files = []
    patterns = ("paired-forward-rank*.jsonl", "request-stages-rank*.jsonl", "tokenizer-ingress-pid*.jsonl")
    for path in sorted({path for pattern in patterns for path in source.glob(pattern)}):
        with path.open("rb") as stream:
            observed_bytes = os.fstat(stream.fileno()).st_size
            captured = stream.read(observed_bytes)
        if len(captured) != observed_bytes:
            raise ValueError("The source was truncated during capture")
        complete_bytes = captured.rfind(b"\n") + 1
        complete = captured[:complete_bytes]
        records = complete.splitlines()
        for line in records:
            json.loads(line)
        (output / path.name).write_bytes(complete)
        files.append({"source": str(path), "name": path.name, "observed_bytes": observed_bytes,
                      "complete_bytes": complete_bytes, "incomplete_tail_bytes": observed_bytes - complete_bytes,
                      "records": len(records), "sha256": hashlib.sha256(complete).hexdigest()})
    if not any(item["name"].startswith("paired-forward-rank") for item in files):
        raise ValueError("No paired forward telemetry found")
    manifest = {"status": "completed", "created_unix": time.time(), "files": files,
                "scope": "Immutable complete-line prefixes of append-only telemetry; source files unchanged; cross-rank analysis must require complete matched iterations"}
    (output / "snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"files": len(files), "records": sum(item["records"] for item in files),
            "incomplete_tail_bytes": sum(item["incomplete_tail_bytes"] for item in files)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.source, args.output)))


if __name__ == "__main__":
    main()
