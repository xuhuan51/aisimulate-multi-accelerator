import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--ssh-socket", required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign
    source = Path(__file__).parent
    path = root / "layer-replay-effect-controller.json"
    if path.exists():
        raise FileExistsError(path)
    state = {"status": "waiting", "created_unix": time.time(),
             "scope": "Predeclared diagnostic only; compare same16K/128/C32 workload after ordinary shape history and after one isolated layer-replay case. No rows from this control will be published as performance data.",
             "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def save():
        path.write_text(json.dumps(state, indent=2) + "\n")

    save()
    try:
        deadline = time.monotonic() + 5400
        while time.monotonic() < deadline:
            prior = json.loads((root / "serving-after-shape-history-controller.json").read_text())
            if prior["status"] == "completed":
                break
            if prior["status"] == "failed":
                raise RuntimeError("Prior shape-history control failed")
            time.sleep(5)
        else:
            raise TimeoutError("Prior shape-history control did not finish")
        for name in ("run_tp32_layer_campaign.py", "run_tp32_serving_validation.py"):
            shutil.copy2(source / name, root / ("effect-control-" + name))
        state["status"] = "isolated_replay_diagnostic"
        save()
        subprocess.run(["python3", "-u", str(source / "run_tp32_layer_campaign.py"), "--campaign", str(root),
                        "--ssh-socket", args.ssh_socket, "--manifest", "diagnostic-layer-replay-campaign.json",
                        "--plan", str(source / "tp32-layer-replay-effect-plan-20261104.json"),
                        "--wait-serving", str(root / "serving-after-shape-history/campaign.json")], check=True, timeout=2400)
        state["status"] = "unprofiled_post_replay_serving"
        save()
        subprocess.run(["python3", "-u", str(source / "run_tp32_serving_validation.py"), "--campaign", str(root),
                        "--ssh-socket", args.ssh_socket, "--output-name", "serving-after-layer-replay",
                        "--plan", str(source / "tp32-after-layer-replay-plan-20261104.json"),
                        "--qualification", str(args.qualification)], check=True, timeout=2400)
        state.update(status="completed", completed_unix=time.time())
    except Exception as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
