import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    files = set()
    selection = json.loads(args.selection.read_text())
    if selection["status"] != "completed_and_immutable":
        raise ValueError("Only completed evidence may be archived")
    for name in selection["paths"]:
        path = root / name
        candidates = [path] if path.is_file() else path.rglob("*")
        if not path.exists():
            raise FileNotFoundError(path)
        for candidate in candidates:
            if candidate.is_symlink():
                raise ValueError("Select resolved evidence explicitly, not a symlink")
            candidate.resolve().relative_to(root)
            if candidate.is_file():
                files.add(candidate)
    if not files:
        raise ValueError("Empty evidence selection")
    args.output.mkdir(exist_ok=False)
    archive = args.output / "evidence.tar.gz"
    records = []
    with tarfile.open(archive, "w:gz", compresslevel=6) as stream:
        for path in sorted(files):
            before = path.stat()
            content = path.read_bytes()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Evidence changed during capture: " + str(path))
            member = tarfile.TarInfo(str(path.relative_to(root)))
            member.size, member.mtime, member.mode = len(content), int(before.st_mtime), 0o644
            stream.addfile(member, io.BytesIO(content))
            records.append({"member": member.name, "bytes": member.size, "sha256": hashlib.sha256(content).hexdigest()})
    expected = {row["member"]: row for row in records}
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            content = stream.extractfile(member).read()
            record = expected.pop(member.name)
            if len(content) != record["bytes"] or hashlib.sha256(content).hexdigest() != record["sha256"]:
                raise ValueError("Archive verification failed")
    if expected:
        raise ValueError("Archive lost selected files")
    receipt = {"archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "verified_members": len(records),
               "members": records, "selection": selection, "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
               "restore_relative_to": "repository root", "archive_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"archive": str(archive), "verified_members": len(records), "bytes": archive.stat().st_size}))


if __name__ == "__main__":
    main()
