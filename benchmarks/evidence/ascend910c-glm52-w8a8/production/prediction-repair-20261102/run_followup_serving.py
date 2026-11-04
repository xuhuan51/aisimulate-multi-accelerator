import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--wait-manifest", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--ssh-socket", required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name or args.output_name in (".", ".."):
        parser.error("Output name must be a single directory name")
    path = args.campaign / (args.output_name + "-controller.json")
    if path.exists():
        raise FileExistsError(path)
    state = {"status": "waiting", "started_unix": time.time(),
             "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
             "scope": "Diagnostic repeat after the declared preceding workload sequence; no intervening restart, layer replay, profiler, timing fit or error-based selection."}

    def save():
        path.write_text(json.dumps(state, indent=2) + "\n")

    save()
    try:
        deadline = time.monotonic() + 5400
        while time.monotonic() < deadline:
            previous = json.loads(args.wait_manifest.read_text())
            if previous["status"] == "completed":
                break
            if previous["status"] == "failed":
                raise RuntimeError("Preceding campaign failed")
            time.sleep(5)
        else:
            raise TimeoutError("Preceding campaign did not finish")
        command = ["python3", "-u", str(Path(__file__).parent / "run_tp32_serving_validation.py"),
                   "--campaign", str(args.campaign), "--ssh-socket", args.ssh_socket,
                   "--output-name", args.output_name, "--plan", str(args.plan), "--qualification", str(args.qualification)]
        state.update(status="running", command=command)
        save()
        subprocess.run(command, check=True, timeout=3600)
        state.update(status="completed", completed_unix=time.time())
    except Exception as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
