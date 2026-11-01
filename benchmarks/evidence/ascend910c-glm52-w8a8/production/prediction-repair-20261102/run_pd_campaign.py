import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT / "pd-paired-1p1d"
TOOLS = Path("/home/youxingye/youxingye_data/910c_script/PD disaggregation/tools/performance")
ENDPOINTS = ["P0=http://10.120.72.20:19000", "D0=http://10.120.72.25:19003"]
IMAGE = "quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443"
CASES = [
    ("diagnostic-c1-8k128", 1, 128),
    ("diagnostic-c4-8k128", 4, 128),
    ("diagnostic-c8-8k128", 8, 128),
    ("validation-c4-8k1k", 4, 1024),
    ("validation-repeat-c4-8k128", 4, 128),
]


def save(state):
    temporary = CAMPAIGN / "progress.json.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(CAMPAIGN / "progress.json")


def wait_health(url, deadline):
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url + "/health", timeout=5).close()
            return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(5)
    raise TimeoutError(url)


def main():
    global CAMPAIGN
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN)
    parser.add_argument("--container-prefix", default="aisim-paired-pd")
    parser.add_argument("--client-trace", action="store_true")
    parser.add_argument("--cases-file", type=Path)
    args = parser.parse_args()
    CAMPAIGN = args.campaign.resolve()
    if (CAMPAIGN / "progress.json").exists():
        raise FileExistsError("Inspect the existing PD campaign before starting another")
    state = {"status": "waiting_for_services", "nodes": [20, 25], "layout": "1P1D, bothTP16/ADP4/EP16", "cases": []}
    save(state)
    deadline = time.monotonic() + 2400
    for endpoint in ENDPOINTS:
        wait_health(endpoint.split("=", 1)[1], deadline)
    with socket.socket() as connection:
        connection.bind(("10.120.72.20", 19688))
    router = ["docker", "run", "-d", "--name", f"{args.container_prefix}-router-20261102", "--init", "--network", "host",
              "-e", "TORCH_DEVICE_BACKEND_AUTOLOAD=0", "-v", "/data_lib/data/models/glm-q:/models/glm-q:ro",
              "--entrypoint", "python3", IMAGE, "-m", "sglang_router.launch_router", "--pd-disaggregation",
              "--policy", "round_robin", "--request-timeout-secs", "1800", "--prefill", "http://10.120.72.20:19000",
              "19998", "--decode", "http://10.120.72.25:19003", "--host", "10.120.72.20", "--port", "19688"]
    (CAMPAIGN / "router-command.json").write_text(json.dumps(router, indent=2) + "\n")
    subprocess.run(router, check=True)
    wait_health("http://10.120.72.20:19688", time.monotonic() + 300)
    cases = json.loads(args.cases_file.read_text()) if args.cases_file else [
        {"name": name, "concurrency": concurrency, "input_length": 8192, "output_length": output_length, "requests": 2 * concurrency}
        for name, concurrency, output_length in CASES
    ]
    for specification in cases:
        name = specification["name"]
        concurrency, input_length, output_length, requests = (specification[key] for key in ("concurrency", "input_length", "output_length", "requests"))
        if Path(name).name != name or any(type(value) is not int or value <= 0 for value in (concurrency, input_length, output_length, requests)):
            raise ValueError("Invalid case name or positive integer workload")
        destination = CAMPAIGN / name
        destination.mkdir()
        snapshot = [sys.executable, str(TOOLS / "snapshot-serving-config.py")]
        for endpoint in ENDPOINTS:
            snapshot.extend(["--endpoint", endpoint])
        subprocess.run(snapshot, env={**os.environ, "SNAPSHOT_OUTPUT": str(destination)}, check=True)
        entrypoint = ["/campaign/trace_serving_client.py"] if args.client_trace else ["-m", "sglang.benchmark.serving"]
        command = ["docker", "exec", "-e", "PYTHONUNBUFFERED=1", f"{args.container_prefix}-prefill-20261102",
                   "python3", *entrypoint, "--backend", "sglang", "--host", "10.120.72.20",
                   "--port", "19688", "--model", "/models/glm-q", "--served-model-name", "glm-5",
                   "--dataset-name", "random-ids", "--num-prompts", str(requests), "--max-concurrency", str(concurrency),
                   "--random-input-len", str(input_length), "--random-output-len", str(output_length), "--random-range-ratio", "1.0",
                   "--request-rate", "inf", "--warmup-requests", "2", "--seed", "910", "--tokenize-prompt",
                   "--output-details", "--output-file", f"/campaign/{name}.jsonl"]
        monitored = [sys.executable, str(TOOLS / "sample-serving-metrics.py"), "--output", str(destination)]
        for endpoint in ENDPOINTS:
            monitored.extend(["--endpoint", endpoint])
        monitored.extend(["--", *command])
        case = {"name": name, "input_length": input_length, "output_length": output_length, "concurrency": concurrency,
                "requests": requests, "command": command, "start_unix": time.time(), "status": "running"}
        state["cases"].append(case)
        state["status"] = "running"
        save(state)
        with (destination / "collection.log").open("w") as stream:
            result = subprocess.run(monitored, stdout=stream, stderr=subprocess.STDOUT, timeout=2400)
        case.update({"end_unix": time.time(), "returncode": result.returncode})
        if result.returncode:
            case["status"] = state["status"] = "failed"
            save(state)
            raise RuntimeError(f"Benchmark failed: {name}")
        raw = CAMPAIGN / "prefill" / f"{name}.jsonl"
        measured = json.loads(raw.read_text().splitlines()[-1])
        if measured["completed"] != requests or measured["input_lens"] != [input_length] * requests or measured["output_lens"] != [output_length] * requests or any(measured["errors"]):
            case["status"] = state["status"] = "invalid_workload"
            save(state)
            raise ValueError(name)
        case["status"] = "completed"
        case["metrics"] = {key: measured[key] for key in ["mean_ttft_ms", "mean_tpot_ms", "input_throughput", "output_throughput"]}
        (destination / "complete.json").write_text(json.dumps(case, indent=2) + "\n")
        save(state)
        print(json.dumps({"case": name, **case["metrics"]}), flush=True)
    state["status"] = "completed"
    save(state)


if __name__ == "__main__":
    main()
