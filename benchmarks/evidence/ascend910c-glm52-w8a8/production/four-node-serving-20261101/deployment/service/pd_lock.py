import argparse
import hashlib
import os
import pathlib


def lock_path(root, name):
    if name not in {"start", "benchmark", "matrix", "comparison"}:
        raise ValueError(f"Unknown controller lock: {name}")
    root = pathlib.Path(root).resolve()
    owner = root.stat()
    identity = hashlib.sha256(str(root).encode()).hexdigest()[:20]
    directory = pathlib.Path("/tmp") / f"aisim-pd-locks-{identity}"
    directory.mkdir(mode=0o700, exist_ok=True)
    if os.geteuid() == 0:
        os.chown(directory, owner.st_uid, owner.st_gid)
        directory.chmod(0o700)
    path = directory / f"{name}.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        if os.geteuid() == 0:
            os.fchown(descriptor, owner.st_uid, owner.st_gid)
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    print(lock_path(args.root, args.name))


if __name__ == "__main__":
    main()
