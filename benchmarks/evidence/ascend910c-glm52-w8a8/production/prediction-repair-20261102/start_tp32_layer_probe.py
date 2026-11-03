import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--remote", default="youxingye@npu1-25")
    parser.add_argument("--ssh-socket", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--max-running-requests", type=int, default=128)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    args.campaign.mkdir(exist_ok=False)
    campaign = args.campaign.resolve()
    ssh = ["ssh", "-S", args.ssh_socket, "-o", "BatchMode=yes", args.remote]
    subprocess.run(ssh + [shlex.join(["mkdir", "-p", args.remote_root])], check=True)
    for node in range(2):
        local = campaign / f"node{node}"
        local.mkdir()
        for name in ("launch_layer_replay_probe.py", "launch_paired_telemetry.py", "layer_output_validation.py"):
            shutil.copy2(source / name, local / name)
        root = str(local) if node == 0 else args.remote_root
        if node:
            for path in local.iterdir():
                with path.open("rb") as stream:
                    subprocess.run(ssh + ["cat > " + shlex.quote(root + "/" + path.name)], stdin=stream, check=True)
        template = json.loads((source / f"tp32-paired-node20-25/node{node}/launch.json").read_text())
        container = f"aisim-layer-tp32-node{node}-20261104"
        command = template["command"][:]
        command[1] = "/campaign/launch_layer_replay_probe.py"
        command[command.index("--forward-pass-metrics-worker-id") + 1] = container
        if args.max_running_requests <= 0 or args.max_running_requests % 32:
            raise ValueError("Global running-request limit must be a positive multiple of32")
        command[command.index("--max-running-requests") + 1] = str(args.max_running_requests)
        command += ["--max-total-tokens", "123904"]
        docker = ["docker"] if node == 0 else ["docker", "-H", "unix:///run/aisim-docker-20261102.sock"]
        launch = docker + ["run", "-d", "--name", container, "--init", "--network", "host",
                           "--privileged", "--shm-size", "16g",
                           "--tmpfs", "/tmp:rw,exec,nosuid,nodev,size=2g,mode=1777"]
        original = template["docker_command"]
        for index, value in enumerate(original[:-1]):
            if value == "-v":
                mount = original[index + 1]
                if mount.endswith(":/campaign"):
                    mount = root + ":/campaign"
                launch += ["-v", mount]
        environment = {key: value for key, value in template["environment"].items() if value is not None}
        environment.update(AISIM_PROBE_BOUNDARY="layer", AISIM_PROBE_CONTROL="1", AISIM_CAPTURE_CPROFILE="0",
                           AISIM_PROBE_LAYERS="0,1,2,3,4,5,6,77")
        for key, value in environment.items():
            launch += ["-e", key + "=" + value]
        shell = "set -e\nsource /usr/local/Ascend/ascend-toolkit/set_env.sh\nsource /usr/local/Ascend/nnal/atb/set_env.sh\nexec " + shlex.join(command)
        launch += ["--entrypoint", "bash", template["image"], "-lc", shell]
        manifest = {"node_rank": node, "container": container, "image": template["image"],
                    "command": command, "docker_command": launch, "environment": environment,
                    "started_unix": time.time(), "remote_root": root,
                    "scope": "TP32/ADP32/EP32 decode layer invocation; aggregated prefill is diagnostic only, no PD TTFT acceptance",
                    "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in local.iterdir()}}
        (local / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
        execution = launch if node == 0 else ssh + [shlex.join(launch)]
        result = subprocess.run(execution, capture_output=True, text=True, timeout=60)
        (local / "launch-result.json").write_text(json.dumps({"returncode": result.returncode,
                                                           "stdout": result.stdout, "stderr": result.stderr}, indent=2) + "\n")
        result.check_returncode()
        print(container, result.stdout.strip(), flush=True)


if __name__ == "__main__":
    main()
