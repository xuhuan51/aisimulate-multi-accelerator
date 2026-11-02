import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
PACKAGE = REPO / "python/aisimulate"
IMAGE = "quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--serving-setup", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    prepare = ["docker", "run", "--rm", "--network", "none", "--cpuset-cpus", "280-281",
               "-v", f"{REPO}:{REPO}:ro", "-v", "/data_lib/data/models/glm-q:/data_lib/data/models/glm-q:ro",
               "-w", str(PACKAGE), "--entrypoint", "python3", "aisimulate-910c:forward-batch-repair-20261102", "-c",
               "import json; from collector.sglang.ascend_vector_cases import get_ascend_vector_cases; "
               "print(json.dumps(get_ascend_vector_cases('/data_lib/data/models/glm-q')))"]
    source = json.loads(subprocess.check_output(prepare, text=True))
    plan = {
        **source,
        "cases": [case for case in source["cases"] if case["role"] == "seq_lens_h2d"],
        "protocol": {name: getattr(args, name) for name in ("iterations", "warmup", "repeats")},
        "runtime_selection": {
            "role": "seq_lens_h2d",
            "full_plan_count": len(source["cases"]),
            "full_plan_sha256": hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        },
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.serving_setup:
        setup = json.loads(args.serving_setup.read_text())
    else:
        setup = json.loads((ROOT.parent / "serving-setup.json").read_text())
        launch = json.loads((ROOT / "forward-attribution-deep-node20/launch.json").read_text())
        setup["environment"].update(launch["environment"])
        setup["environment_scope"] = "Exact eager attribution launcher environment; same node20 worker affinity mapping"
    (output / "serving-setup.json").write_text(json.dumps(setup, indent=2) + "\n")
    command = ["docker", "run", "--rm", "--name", f"aisim-seq-lens-device{args.device}-20261102",
               "--network", "none", "--privileged", "--shm-size", "2g",
               "-v", f"{REPO}:/workspace/aisimulate-910c",
               "-v", f"{output}:/campaign"]
    for source_path in ("/usr/local/Ascend/driver", "/usr/local/Ascend/firmware", "/usr/local/sbin", "/etc/ascend_install.info"):
        command.extend(["-v", f"{source_path}:{source_path}:ro"])
    command.extend(["--entrypoint", "python3", IMAGE,
                    "/workspace/aisimulate-910c/python/aisimulate/collector/sglang/ascend_vector.py",
                    "--plan", "/campaign/plan.json", "--serving-setup", "/campaign/serving-setup.json",
                    "--output", "/campaign/observations.json", "--device", str(args.device)])
    (output / "launch.json").write_text(json.dumps({"command": command, "device": args.device}, indent=2) + "\n")
    with (output / "collector.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Collector exited {result.returncode}; see {output / 'collector.log'}")
    observed = json.loads((output / "observations.json").read_text())
    identity = {key: observed["runtime"][key] for key in ("framework_commit", "driver_version_info", "torch", "torch_npu")}
    identity.update(image_digest=IMAGE, hardware=observed["runtime"]["device"],
                    collector_source_sha256=observed["collector_source_sha256"])
    (output / "runtime-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps({"status": observed["status"], "cases": len(observed["cases"]), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
