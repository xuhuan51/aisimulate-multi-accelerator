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
    parser.add_argument("--role", choices=["prefill", "decode"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--container-prefix", default="aisim-paired-pd")
    args = parser.parse_args()
    role = args.role
    node = 20 if role == "prefill" else 25
    worker_ip = f"10.120.72.{node}"
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    model = Path("/data_lib/data/models/glm-q")
    for name in ("config.json", "quant_model_description.json"):
        if not (model / name).is_file():
            raise FileNotFoundError(model / name)
    inventory = subprocess.check_output(["npu-smi", "info"], text=True)
    if inventory.count("No running processes found in NPU") != 8:
        raise RuntimeError("Preserve the existing NPU processes")
    ports = (19000, 19998, 19707) if role == "prefill" else (19003,)
    for port in ports:
        with socket.socket() as connection:
            connection.bind((worker_ip, port))
    image_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE], text=True).strip()
    variables = {
        "DRY_RUN": "1", "WORKER_IP": worker_ip, "WORKER_PORT": str(ports[0]),
        "WORKER_BOOTSTRAP_PORT": "19998", "STORE_IP": "10.120.72.20", "STORE_PORT": "19707",
        "PREFILL_TP_SIZE": "16", "PREFILL_DP_SIZE": "4", "PREFILL_EP_SIZE": "16",
        "DECODE_TP_SIZE": "16", "DECODE_DP_SIZE": "4", "DECODE_EP_SIZE": "16",
        "DECODE_NNODES": "1", "WORKER_NNODES": "1", "WORKER_NODE_RANK": "0",
        "DECODE_MAX_RUNNING_REQUESTS": "128", "PREFILL_MAX_RUNNING_REQUESTS": "128",
        "DISABLE_GRAPH": "1", "DISABLE_OVERLAP": "1", "DISABLE_PREFIX_CACHE": "1",
        "ENABLE_SPECULATIVE_DECODING": "0", "HCCL_IF_BASE_PORT": "41000",
    }
    source = ROOT / "serving-source" / f"run-{role}.sh"
    rendered = subprocess.check_output(["bash", str(source)], env={**os.environ, **variables}, text=True)
    line = next(line for line in rendered.splitlines() if line.startswith("执行 "))
    original = shlex.split(line.split("：", 1)[1])
    if original[:3] != ["python3", "-m", "sglang.launch_server"]:
        raise ValueError("Unexpected serving entrypoint")
    container = f"{args.container_prefix}-{role}-20261102"
    command = ["python3", "/campaign/launch_paired_telemetry.py", *original[3:],
               "--enable-forward-pass-metrics", "--forward-pass-metrics-worker-id", container,
               "--forward-pass-metrics-ipc-name", "ipc:///campaign/forward-metrics"]
    environment = json.loads((ROOT / "runtime-environment.json").read_text())
    environment.update({"ASCEND_MF_STORE_URL": "tcp://10.120.72.20:19707", "HCCL_IF_BASE_PORT": "41000",
                        "SGLANG_NPU_PROFILING": "0", "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
                        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600", "TRANSFORMERS_VERBOSITY": "error",
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    destination.mkdir(parents=True)
    shutil.copyfile(ROOT / "launch_paired_telemetry.py", destination / "launch_paired_telemetry.py")
    (destination / "npu-before.txt").write_text(inventory)
    docker = ["docker", "run", "-d", "--name", container, "--init", "--network", "host", "--privileged",
              "--shm-size", "16g", "-v", f"{model}:/models/glm-q:ro", "-v", f"{destination}:/campaign"]
    for path in ("/usr/local/Ascend/driver", "/usr/local/Ascend/firmware", "/usr/local/sbin", "/etc/ascend_install.info", "/var/queue_schedule"):
        if Path(path).exists():
            docker.extend(["-v", f"{path}:{path}"])
    for key, value in environment.items():
        if value is not None:
            docker.extend(["-e", f"{key}={value}"])
    shell = "set -e\nsource /usr/local/Ascend/ascend-toolkit/set_env.sh\nsource /usr/local/Ascend/nnal/atb/set_env.sh\nexec " + shlex.join(command)
    docker.extend(["--entrypoint", "bash", IMAGE, "-lc", shell])
    record = {"role": role, "host": socket.gethostname(), "model": str(model), "image": IMAGE, "image_id": image_id,
              "docker_host": os.environ.get("DOCKER_HOST", "unix:///var/run/docker.sock"),
              "source_command": original, "command": command, "environment": environment,
              "docker_command": docker, "model_metadata_sha256": {name: hashlib.sha256((model / name).read_bytes()).hexdigest() for name in ("config.json", "quant_model_description.json")},
              "source_script_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "telemetry_script_sha256": hashlib.sha256((destination / "launch_paired_telemetry.py").read_bytes()).hexdigest()}
    (destination / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    result = subprocess.run(docker, capture_output=True, text=True)
    record.update({"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    (destination / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    result.check_returncode()
    print(json.dumps({"container": container, "image_id": image_id, "evidence": str(destination)}))


if __name__ == "__main__":
    main()
