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
    destination = root / "chain-campaign.json"
    if destination.exists():
        raise FileExistsError(destination)
    plan = [
        ("train-512-c4", 512, 4, True),
        ("train-8192-c4", 8192, 4, True),
        ("train-16384-c4", 16384, 4, True),
        ("holdout-4096-c4", 4096, 4, False),
        ("holdout-8192-c8", 8192, 8, False),
        ("holdout-8192-c12", 8192, 12, False),
    ]
    manifest = {"scope": "Real-layer invocation profiles and independent workload holdouts; no forward residual fit",
                "status": "waiting", "cases": [], "plan": plan,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def save():
        destination.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    for attempt in range(600):
        state = subprocess.check_output(["docker", "inspect", "-f", "{{.State.Status}}", args.container], text=True).strip()
        if state != "running":
            raise RuntimeError(f"Diagnostic service stopped before readiness: {state}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:18080/get_server_info", timeout=2) as response:
                info = json.load(response)
            break
        except (OSError, ValueError):
            time.sleep(1)
    else:
        raise TimeoutError("Layer-chain service did not become ready")
    if (info["tp_size"], info["attn_dp_size"], info["ep_size"]) != (16, 4, 16):
        raise ValueError("Unexpected topology")
    if not info["disable_overlap_schedule"] or info["speculative_algorithm"] is not None:
        raise ValueError("Unexpected execution mode")
    if any(info["cuda_graph_config"][phase]["backend"] != "disabled" for phase in ("prefill", "decode")):
        raise ValueError("Unexpected graph mode")
    (root / "server-info.json").write_text(json.dumps(info, indent=2) + "\n")
    manifest["status"] = "running"
    save()
    try:
        for name, input_length, concurrency, training in plan:
            case_root = root / name
            case_root.mkdir(exist_ok=False)
            control = {"id": name, "forward_call": 80, "training": training,
                       "input_length": input_length, "global_requests": concurrency}
            temporary = root / "layer-probe-control.tmp"
            temporary.write_text(json.dumps(control) + "\n")
            temporary.replace(root / "layer-probe-control.json")
            command = ["docker", "exec", args.container, "python3", "-m", "sglang.benchmark.serving",
                       "--backend", "sglang", "--host", "127.0.0.1", "--port", "18080",
                       "--model", "/models/glm-q", "--served-model-name", "glm-5", "--dataset-name", "random-ids",
                       "--num-prompts", str(concurrency), "--max-concurrency", str(concurrency),
                       "--random-input-len", str(input_length), "--random-output-len", "192",
                       "--random-range-ratio", "1.0", "--request-rate", "inf", "--warmup-requests", "2",
                       "--seed", "910", "--tokenize-prompt", "--output-details",
                       "--output-file", f"/campaign/{name}/requests.jsonl"]
            case = {**control, "command": command, "start_unix": time.time(), "status": "running"}
            manifest["cases"].append(case)
            save()
            with (case_root / "benchmark.log").open("w") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=1200)
            case.update(end_unix=time.time(), returncode=result.returncode)
            result.check_returncode()
            benchmark = json.loads((case_root / "requests.jsonl").read_text().splitlines()[-1])
            if (benchmark["completed"] != concurrency or any(benchmark["errors"])
                    or benchmark["input_lens"] != [input_length] * concurrency
                    or benchmark["output_lens"] != [192] * concurrency):
                raise ValueError(f"Request completion/length mismatch: {name}")
            probes = sorted(case_root.glob("layer-replay-rank*.json"))
            if len(probes) != 16:
                raise ValueError(f"Incomplete rank replay: {name}, {len(probes)}")
            for path in probes:
                observed = json.loads(path.read_text())
                if (observed["status"] != "completed" or len(observed["boundaries"]) != 8
                        or observed["control"] != control):
                    raise ValueError(f"Invalid rank profile: {path}")
                for boundary in observed["boundaries"]:
                    lengths = [value for value in boundary["sequence_lengths"] if value > 0]
                    if len(lengths) != concurrency // 4 or not all(input_length <= value < input_length + 192 for value in lengths):
                        raise ValueError(f"Captured batch differs from requested steady workload: {path}")
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
