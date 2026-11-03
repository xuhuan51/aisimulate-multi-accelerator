import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path


def selected_files(campaign, closed_telemetry):
    files = {path for path in campaign.iterdir() if path.is_file()}
    for directory in campaign.iterdir():
        if directory.is_dir() and directory.name.startswith(("train-", "holdout-", "qualification-")):
            files.update(path for path in directory.rglob("*") if path.is_file())
    for node in ("node0", "node1"):
        directory = campaign / node
        for path in directory.iterdir():
            if path.is_file() and path.suffix in (".py", ".json", ".txt"):
                if path.name != "layer-probe-control.json":
                    files.add(path)
            elif closed_telemetry and path.is_file() and path.name.startswith(("paired-forward-rank", "request-stages-rank", "tokenizer-ingress-pid")) and path.suffix == ".jsonl":
                files.add(path)
            elif path.is_dir() and path.name.startswith("telemetry-"):
                if not (path / "snapshot.json").exists():
                    raise ValueError(f"Telemetry snapshot has no immutable receipt: {path}")
                files.update(child for child in path.rglob("*") if child.is_file())
    return sorted(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--closed-telemetry", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    archive = args.output / "raw-collection.tar.gz"
    records = []
    with tarfile.open(archive, "w:gz", compresslevel=6) as stream:
        for path in selected_files(args.campaign, args.closed_telemetry):
            before = path.stat()
            content = path.read_bytes()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"Collection evidence changed during archiving: {path}")
            name = str(path.relative_to(args.campaign.parent))
            member = tarfile.TarInfo(name)
            member.size = len(content)
            member.mtime = int(before.st_mtime)
            member.mode = 0o644
            stream.addfile(member, io.BytesIO(content))
            records.append({"member": name, "source": str(path), "bytes": len(content),
                            "sha256": hashlib.sha256(content).hexdigest()})
    by_name = {record["member"]: record for record in records}
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            content = stream.extractfile(member).read()
            record = by_name.pop(member.name)
            if len(content) != record["bytes"] or hashlib.sha256(content).hexdigest() != record["sha256"]:
                raise ValueError(f"Archive verification failed: {member.name}")
    if by_name:
        raise ValueError("Archive lost evidence members")
    receipt = {"schema": "aisimulate-tp32-raw-collection-archive/v1", "campaign": str(args.campaign),
               "archive": str(archive), "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
               "verified_members": len(records), "members": records,
               "restore_base": str(args.campaign.parent),
               "closed_telemetry": args.closed_telemetry,
               "selection": "Canonical train/holdout cases, frozen root/node metadata and immutable telemetry snapshots; original telemetry included only for explicitly closed workers. Live telemetry and the separately recorded serving campaign are excluded. Original files remain unchanged."}
    (args.output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("archive", "archive_sha256", "verified_members")}))


if __name__ == "__main__":
    main()
