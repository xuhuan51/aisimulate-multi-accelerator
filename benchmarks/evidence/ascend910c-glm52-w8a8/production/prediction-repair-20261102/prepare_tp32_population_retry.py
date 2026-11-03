import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--source-campaign", type=Path, required=True)
    parser.add_argument("--ssh-socket", required=True)
    args = parser.parse_args()
    root = args.campaign
    source = Path(__file__).parent
    ssh = ["ssh", "-S", args.ssh_socket, "-o", "BatchMode=yes", "youxingye@npu1-25"]
    prior_host = json.loads((args.source_campaign / "host-identity.json").read_text())
    for node, paths in prior_host.items():
        for name, expected in paths.items():
            actual = Path(name).read_text() if node == "node0" else subprocess.check_output(ssh + [shlex.join(["cat", name])], text=True)
            if actual != expected:
                raise ValueError("Host identity changed: " + node + ":" + name)
    prior_model = json.loads((args.source_campaign / "model-identity.json").read_text())
    for name, expected in prior_model.items():
        path = Path("/data_lib/data/models/glm-q") / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Local model metadata changed")
        peer = subprocess.check_output(ssh + [shlex.join(["sha256sum", str(path)])], text=True).split()[0]
        if peer != expected:
            raise ValueError("Peer model metadata changed")
    for name in ("host-identity.json", "model-identity.json"):
        shutil.copy2(args.source_campaign / name, root / name)
    launch = json.loads((root / "node0/launch.json").read_text())
    (root / "launch.json").write_text(json.dumps({"command": launch["docker_command"]}, indent=2) + "\n")
    for name in ("launch_layer_replay_probe.py", "layer_output_validation.py", "run_tp32_layer_campaign.py", "run_routed_layer_benchmark.py"):
        shutil.copy2(source / name, root / name)
    plan = [
        ["routed-uniform-a4-b2-k8192-retry1", 8192, 8, True, [2] * 4 + [0] * 28],
        ["routed-uniform-a31-b2-k8192-retry1", 8192, 62, True, [2] * 31 + [0]],
    ]
    (root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (root / "preparation.json").write_text(json.dumps({"source_campaign": str(args.source_campaign),
        "host_and_model_metadata_reverified_on_both_nodes": True,
        "selection": "Repeat only the two predeclared uniform training populations with missing CV-qualified cells; keep prior failures and unchanged0.1 CV and numerical gates",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
