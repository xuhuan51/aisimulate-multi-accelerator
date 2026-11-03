import argparse
import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--ssh-socket", required=True)
    parser.add_argument("--cpu-validation-container", required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).parent
    plan = source / "tp32-fresh-steady-plan-20261104.json"
    path = args.campaign / "fresh-validation-controller.json"
    if path.exists():
        raise FileExistsError(path)
    state = {"status": "waiting_for_own_cpu_validation_and_health", "started_unix": time.time(),
             "scope": "Same deployment restarted without invoking layer replay or operator profiling before these serving measurements. All observations retained independently of prediction errors.",
             "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def save():
        path.write_text(json.dumps(state, indent=2) + "\n")

    save()
    try:
        deadline = time.monotonic() + 5400
        while time.monotonic() < deadline:
            container = json.loads(subprocess.check_output([
                "docker", "inspect", "--format", "{{json .State}}", args.cpu_validation_container], text=True))
            if not container["Running"]:
                state["cpu_validation"] = container
                if container["ExitCode"]:
                    raise RuntimeError("CPU validation failed; serving run not started")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:19180/health", timeout=5) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    pass
            time.sleep(5)
        else:
            raise TimeoutError("CPU validation or fresh worker readiness timed out")
        state.update(status="running", serving_start_unix=time.time())
        save()
        command = ["python3", "-u", str(source / "run_tp32_serving_validation.py"),
                   "--campaign", str(args.campaign), "--ssh-socket", args.ssh_socket,
                   "--plan", str(plan), "--qualification", str(args.qualification)]
        state["command"] = command
        save()
        subprocess.run(command, check=True, timeout=5400)
        state.update(status="completed", completed_unix=time.time())
    except Exception as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
