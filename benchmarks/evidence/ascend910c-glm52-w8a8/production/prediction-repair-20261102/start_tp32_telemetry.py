import argparse
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
IMAGE = "quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-rank", type=int, choices=[0, 1], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    model = Path("/data_lib/data/models/glm-q")
    inventory = subprocess.check_output(["npu-smi", "info"], text=True)
    if inventory.count("No running processes found in NPU") != 8:
        raise RuntimeError("Preserve existing NPU workloads")
    for port in ([19180, 19500] if args.node_rank == 0 else [19180]):
        with socket.socket() as connection:
            connection.bind(("0.0.0.0", port))
    image_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE], text=True).strip()
    metadata = {name: hashlib.sha256((model / name).read_bytes()).hexdigest() for name in ("config.json", "quant_model_description.json")}
    environment = json.loads((ROOT / "runtime-environment.json").read_text())
    environment.update({"HCCL_IF_BASE_PORT": "45000", "SGLANG_NPU_PROFILING": "0", "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    environment.pop("ASCEND_MF_STORE_URL", None)
    container = f"aisim-paired-tp32-node{args.node_rank}-20261102"
    command = ["python3", "/campaign/launch_paired_telemetry.py", "--model-path", "/models/glm-q", "--host", "0.0.0.0", "--port", "19180",
               "--nnodes", "2", "--node-rank", str(args.node_rank), "--dist-init-addr", "10.120.72.20:19500", "--tp-size", "32", "--attn-dp-size", "32", "--ep-size", "32",
               "--mem-fraction-static", "0.90", "--max-running-requests", "128", "--attention-backend", "ascend", "--device", "npu", "--quantization", "modelslim",
               "--served-model-name", "glm-5", "--watchdog-timeout", "9000", "--context-length", "180000", "--tokenizer-worker-num", "4",
               "--disable-shared-experts-fusion", "--dtype", "bfloat16", "--load-balance-method", "round_robin", "--page-size", "128",
               "--enable-cache-report", "--enable-metrics", "--enable-metrics-for-all-schedulers", "--trust-remote-code", "--moe-a2a-backend", "deepep", "--deepep-mode", "auto",
               "--cuda-graph-backend-prefill", "disabled", "--cuda-graph-backend-decode", "disabled", "--disable-overlap-schedule", "--disable-radix-cache",
               "--chunked-prefill-size", "32768", "--max-prefill-tokens", "32768", "--enable-tp-lm-head-all-to-all",
               "--enable-forward-pass-metrics", "--forward-pass-metrics-worker-id", container, "--forward-pass-metrics-ipc-name", "ipc:///campaign/forward-metrics"]
    destination.mkdir(parents=True)
    shutil.copyfile(ROOT / "launch_paired_telemetry.py", destination / "launch_paired_telemetry.py")
    (destination / "npu-before.txt").write_text(inventory)
    docker = ["docker", "run", "-d", "--name", container, "--init", "--network", "host", "--privileged", "--shm-size", "16g",
              "-v", f"{model}:/models/glm-q:ro", "-v", f"{destination}:/campaign"]
    for path in ("/usr/local/Ascend/driver", "/usr/local/Ascend/firmware", "/usr/local/sbin", "/etc/ascend_install.info", "/var/queue_schedule"):
        if Path(path).exists():
            docker.extend(["-v", f"{path}:{path}"])
    for key, value in environment.items():
        if value is not None:
            docker.extend(["-e", f"{key}={value}"])
    shell = "set -e\nsource /usr/local/Ascend/ascend-toolkit/set_env.sh\nsource /usr/local/Ascend/nnal/atb/set_env.sh\nexec " + shlex.join(command)
    docker.extend(["--entrypoint", "bash", IMAGE, "-lc", shell])
    record = {"node_rank": args.node_rank, "host": socket.gethostname(), "image": IMAGE, "image_id": image_id,
              "model_metadata_sha256": metadata, "environment": environment, "command": command, "docker_command": docker,
              "docker_host": os.environ.get("DOCKER_HOST", "unix:///var/run/docker.sock"),
              "scope": "Colocated workload used only for phase-separated TP32 decode forward calibration. Explicit TP LM-head all-to-all matches pure-DP PD Decode. Prefill and full-serving latency are not a PD reference.",
              "telemetry_script_sha256": hashlib.sha256((destination / "launch_paired_telemetry.py").read_bytes()).hexdigest()}
    (destination / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    result = subprocess.run(docker, capture_output=True, text=True)
    record.update({"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    (destination / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    result.check_returncode()
    print(json.dumps({"container": container, "image_id": image_id, "evidence": str(destination)}))


if __name__ == "__main__":
    main()
