import argparse
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--container", required=True)
    args = parser.parse_args()
    root = args.campaign
    destination = root / "context-holdouts.json"
    if destination.exists():
        raise FileExistsError(destination)
    with urllib.request.urlopen("http://127.0.0.1:18080/get_server_info", timeout=10) as response:
        info = json.load(response)
    if (info["tp_size"], info["attn_dp_size"], info["ep_size"]) != (16, 4, 16):
        raise ValueError("Unexpected topology")
    if not info["disable_overlap_schedule"] or info["speculative_algorithm"] is not None:
        raise ValueError("Unexpected overlap/speculation mode")
    if any(info["cuda_graph_config"][phase]["backend"] != "disabled" for phase in ("prefill", "decode")):
        raise ValueError("Unexpected graph mode")
    (root / "holdout-server-info.json").write_text(json.dumps(info, indent=2) + "\n")
    previous = root / "layer-probe-control.json"
    previous.rename(root / "layer-probe-control-before-holdouts.json")
    manifest = {"status": "running", "cases": [], "instrumentation": "No replay, cProfile or per-component events; existing forward telemetry retained",
                "scope": "Independent serving validation, never used to fit the context extrapolation rule",
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "plan": [("control-8192-128-c4", 8192, 128, 4, 8),
                         ("holdout-12288-4096-c4", 12288, 4096, 4, 4),
                         ("holdout-16384-128-c4", 16384, 128, 4, 8)]}

    def save():
        destination.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    try:
        for name, input_length, output_length, concurrency, prompts in manifest["plan"]:
            case_root = root / name
            case_root.mkdir(exist_ok=False)
            command = ["docker", "exec", args.container, "python3", "-m", "sglang.benchmark.serving",
                       "--backend", "sglang", "--host", "127.0.0.1", "--port", "18080",
                       "--model", "/models/glm-q", "--served-model-name", "glm-5", "--dataset-name", "random-ids",
                       "--num-prompts", str(prompts), "--max-concurrency", str(concurrency),
                       "--random-input-len", str(input_length), "--random-output-len", str(output_length),
                       "--random-range-ratio", "1.0", "--request-rate", "inf", "--warmup-requests", "2",
                       "--seed", "974", "--tokenize-prompt", "--output-details",
                       "--output-file", f"/campaign/{name}/requests.jsonl"]
            case = {"id": name, "command": command, "start_unix": time.time(), "status": "running",
                    "input_length": input_length, "output_length": output_length,
                    "concurrency": concurrency, "prompts": prompts}
            manifest["cases"].append(case)
            save()
            with (case_root / "benchmark.log").open("w") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=3600)
            case.update(end_unix=time.time(), returncode=result.returncode)
            result.check_returncode()
            observed = json.loads((case_root / "requests.jsonl").read_text().splitlines()[-1])
            if (observed["completed"] != prompts or any(observed["errors"])
                    or observed["input_lens"] != [input_length] * prompts
                    or observed["output_lens"] != [output_length] * prompts):
                raise ValueError(f"Request completion/length mismatch: {name}")
            case.update(status="completed", result_sha256=hashlib.sha256((case_root / "requests.jsonl").read_bytes()).hexdigest())
            save()
            print(json.dumps({key: case[key] for key in ("id", "status", "end_unix")}), flush=True)
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
