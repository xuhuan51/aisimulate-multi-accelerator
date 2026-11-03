import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    template = json.loads(subprocess.check_output(["docker", "inspect", args.template]))[0]
    if template["State"]["Running"]:
        raise ValueError("Template must be stopped before starting a replacement")
    if "@sha256:" not in template["Config"]["Image"]:
        raise ValueError("An immutable serving image is required")
    args.campaign.mkdir(exist_ok=False)
    campaign = args.campaign.resolve()
    original = next(mount["Source"] for mount in template["Mounts"] if mount["Destination"] == "/campaign")
    script = campaign / "launch_paired_telemetry.py"
    shutil.copy2(Path(original) / script.name, script)
    command = ["docker", "run", "-d", "--name", args.container, "--network", "host",
               "--privileged", "--security-opt", "label=disable", "--shm-size", "16g",
               "--entrypoint", "python3"]
    for mount in template["Mounts"]:
        source = str(campaign) if mount["Destination"] == "/campaign" else mount["Source"]
        command.extend(["--mount", f"type=bind,source={source},destination={mount['Destination']}" +
                        (",readonly" if not mount["RW"] else "")])
    for value in template["Config"]["Env"]:
        if not value.startswith("AISIM_"):
            command.extend(["-e", value])
    launch = list(template["Config"]["Cmd"])
    launch[0] = "/campaign/launch_paired_telemetry.py"
    launch[launch.index("--forward-pass-metrics-worker-id") + 1] = args.container
    command.extend([template["Config"]["Image"], *launch])
    manifest = {"schema": "aisimulate-ingress-diagnostic/v1", "started_unix": time.time(),
                "template": args.template, "container": args.container,
                "image": template["Config"]["Image"], "command": command,
                "telemetry_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                "scope": "Independent serving holdouts; no component replay or measured timing injection"}
    (campaign / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run(command, check=True, timeout=60)


if __name__ == "__main__":
    main()
