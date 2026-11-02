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
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    args = parser.parse_args()
    root = args.campaign / "full-request-validation"
    root.mkdir(exist_ok=False)
    with urllib.request.urlopen("http://127.0.0.1:18080/get_server_info", timeout=10) as response:
        info = json.load(response)
    if (info["tp_size"], info["attn_dp_size"], info["ep_size"]) != (16, 4, 16):
        raise ValueError("Unexpected deployment topology")
    if not info["disable_overlap_schedule"] or info["speculative_algorithm"] is not None:
        raise ValueError("Overlap or speculation differs from prediction identity")
    if any(info["cuda_graph_config"][phase]["backend"] != "disabled" for phase in ("prefill", "decode")):
        raise ValueError("Graph execution differs from prediction identity")
    (root / "server-info.json").write_text(json.dumps(info, indent=2) + "\n")
    control = args.campaign / "layer-probe-control.json"
    if control.exists():
        control.rename(root / "disabled-layer-probe-control.json")
    manifest = {"status": "running", "cases": [], "input_length": 8192, "output_length": 1024,
                "warmup_requests": 2, "scope": "Single-node aggregate held-out full requests, no layer replay or residual fit",
                "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    destination = root / "campaign.json"

    def save():
        destination.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    try:
        for concurrency in args.concurrency:
            name = f"8192-1024-c{concurrency}"
            case_root = root / name
            case_root.mkdir(exist_ok=False)
            command = ["docker", "exec", args.container, "python3", "-m", "sglang.benchmark.serving",
                       "--backend", "sglang", "--host", "127.0.0.1", "--port", "18080",
                       "--model", "/models/glm-q", "--served-model-name", "glm-5", "--dataset-name", "random-ids",
                       "--num-prompts", str(concurrency), "--max-concurrency", str(concurrency),
                       "--random-input-len", "8192", "--random-output-len", "1024", "--random-range-ratio", "1.0",
                       "--request-rate", "inf", "--warmup-requests", "2", "--seed", "920", "--tokenize-prompt",
                       "--output-details", "--output-file", f"/campaign/full-request-validation/{name}/requests.jsonl"]
            case = {"id": name, "concurrency": concurrency, "command": command,
                    "status": "running", "start_unix": time.time()}
            manifest["cases"].append(case)
            save()
            with (case_root / "benchmark.log").open("w") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
            case.update(end_unix=time.time(), returncode=result.returncode)
            result.check_returncode()
            benchmark = json.loads((case_root / "requests.jsonl").read_text().splitlines()[-1])
            if (benchmark["completed"] != concurrency or any(benchmark["errors"])
                    or benchmark["input_lens"] != [8192] * concurrency
                    or benchmark["output_lens"] != [1024] * concurrency):
                raise ValueError(f"Incomplete or truncated requests: {name}")
            case["status"] = "completed"
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
