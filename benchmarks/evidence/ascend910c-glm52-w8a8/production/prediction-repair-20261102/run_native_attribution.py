import json
import subprocess
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "native-attribution-node20-20261103"
CONTAINER = "aisim-native-attribution-20261103"


def save(state):
    (ROOT / "attribution-progress.json").write_text(json.dumps(state, indent=2) + "\n")


def main():
    if (ROOT / "attribution-progress.json").exists():
        raise FileExistsError(ROOT / "attribution-progress.json")
    with urllib.request.urlopen("http://127.0.0.1:18080/get_server_info", timeout=15) as response:
        info = json.load(response)
    if (info["tp_size"], info["attn_dp_size"], info["ep_size"]) != (16, 4, 16):
        raise ValueError("Unexpected topology")
    if not info["disable_overlap_schedule"] or info["speculative_algorithm"] is not None:
        raise ValueError("Unexpected execution mode")
    if any(info["cuda_graph_config"][phase]["backend"] != "disabled" for phase in ("prefill", "decode")):
        raise ValueError("Unexpected graph mode")
    (ROOT / "server-info-before.json").write_text(json.dumps(info, indent=2) + "\n")
    state = {"status": "running", "purpose": "Native stack attribution, no residual fitting", "cases": []}
    for name, outputs, sample in [("control-before-8k-c4", 96, False), ("native-8k-c4", 192, True), ("control-after-8k-c4", 96, False)]:
        control = ROOT / "attribution-control.next.json"
        control.write_text(json.dumps({"enabled": False, "case": name}) + "\n")
        control.replace(ROOT / "attribution-control.json")
        command = ["docker", "exec", "-e", "PYTHONUNBUFFERED=1", CONTAINER,
                   "python3", "-m", "sglang.benchmark.serving", "--backend", "sglang",
                   "--host", "127.0.0.1", "--port", "18080", "--model", "/models/glm-q",
                   "--served-model-name", "glm-5", "--dataset-name", "random-ids",
                   "--num-prompts", "4", "--max-concurrency", "4", "--random-input-len", "8192",
                   "--random-output-len", str(outputs), "--random-range-ratio", "1.0",
                   "--request-rate", "inf", "--warmup-requests", "2", "--seed", "910",
                   "--tokenize-prompt", "--output-details", "--output-file", f"/campaign/{name}.jsonl"]
        case = {"name": name, "enabled": False, "native_sampling": sample, "input_length": 8192,
                "output_length": outputs, "concurrency": 4, "requests": 4, "start_unix": time.time(),
                "command": command, "status": "running"}
        state["cases"].append(case)
        save(state)
        with (ROOT / f"{name}.log").open("w") as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            profiler = None
            while process.poll() is None:
                if sample and profiler is None:
                    telemetry = ROOT / "paired-forward-rank0.jsonl"
                    lines = telemetry.read_text().splitlines()
                    row = json.loads(lines[-1]) if lines else {}
                    requests = row.get("requests", [])
                    if requests and any(request["prompt_tokens"] == 8192 and 50 <= request["output_tokens"] <= 120 for request in requests) and row["launch_unix"] >= case["start_unix"]:
                        profile_command = ["docker", "exec", CONTAINER, "py-spy", "record", "--pid", str(row["pid"]),
                                           "--native", "--threads", "--rate", "75", "--duration", "20", "--format", "raw",
                                           "--output", f"/campaign/{name}.pyspy.txt"]
                        case["profile_command"] = profile_command
                        case["profile_start_unix"] = time.time()
                        profile_log = (ROOT / f"{name}.pyspy.log").open("w")
                        profiler = subprocess.Popen(profile_command, stdout=profile_log, stderr=subprocess.STDOUT)
                        save(state)
                if time.time() - case["start_unix"] > 900:
                    process.terminate()
                    raise TimeoutError(name)
                time.sleep(1)
        case.update(returncode=process.returncode, end_unix=time.time())
        if profiler is not None:
            case["profile_returncode"] = profiler.wait(timeout=30)
            profile_log.close()
        if process.returncode or (sample and profiler is None):
            case["status"] = state["status"] = "failed"
            save(state)
            raise RuntimeError(name)
        measured = json.loads((ROOT / f"{name}.jsonl").read_text().splitlines()[-1])
        if measured["completed"] != 4 or any(measured["errors"]) or measured["input_lens"] != [8192] * 4 or measured["output_lens"] != [outputs] * 4:
            raise ValueError("Workload mismatch")
        case["status"] = "completed"
        case["metrics"] = {key: measured[key] for key in ("mean_ttft_ms", "mean_tpot_ms", "input_throughput", "output_throughput")}
        save(state)
        print(json.dumps({"case": name, **case["metrics"]}), flush=True)
    state["status"] = "completed"
    save(state)


if __name__ == "__main__":
    main()
