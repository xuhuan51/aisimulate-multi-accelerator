import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "forward-attribution-node20"
CONTAINER = "aisim-forward-attribution-20261102"
CASES = [
    ("control-before-8k-c4", False, 8192, 64, 4, 4),
    ("host-8k-c4", True, 8192, 64, 4, 4),
    ("control-after-8k-c4", False, 8192, 64, 4, 4),
    ("host-512-c4", True, 512, 96, 4, 4),
    ("host-16k-c4", True, 16384, 96, 4, 4),
    ("host-8k-c1", True, 8192, 96, 1, 2),
    ("host-8k-c8", True, 8192, 64, 8, 8),
]


def main():
    global ROOT, CONTAINER, CASES
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.deep or args.cpu:
        suffix = "cpu" if args.cpu else "deep"
        ROOT = ROOT.with_name(f"forward-attribution-{suffix}-node20")
        CONTAINER = f"aisim-forward-attribution-{suffix}-20261102"
        CASES = [
            ("control-before-8k-c4", False, 8192, 64, 4, 4),
            ("host-8k-c4", True, 8192, 64, 4, 4),
            ("control-after-8k-c4", False, 8192, 64, 4, 4),
            ("host-512-c4", True, 512, 64, 4, 4),
            ("host-16k-c4", True, 16384, 64, 4, 4),
        ]
    progress = ROOT / "attribution-progress.json"
    if progress.exists():
        raise FileExistsError(progress)
    with urllib.request.urlopen("http://127.0.0.1:18080/get_server_info", timeout=10) as response:
        info = json.load(response)
    if (info["tp_size"], info["attn_dp_size"], info["ep_size"]) != (16, 4, 16):
        raise ValueError("Unexpected live topology")
    if not info["disable_overlap_schedule"] or info["speculative_algorithm"] is not None:
        raise ValueError("Unexpected execution mode")
    if any(info["cuda_graph_config"][phase]["backend"] != "disabled" for phase in ("prefill", "decode")):
        raise ValueError("Graph must be disabled")
    (ROOT / "server-info-before.json").write_text(json.dumps(info, indent=2) + "\n")
    state = {"status": "running", "purpose": "Attribute actual host preparation/submission paths; no whole-forward residual fitting", "cases": []}
    for name, enabled, inputs, outputs, concurrency, count in CASES:
        control = ROOT / "attribution-control.next.json"
        control.write_text(json.dumps({"enabled": enabled, "case": name}) + "\n")
        control.replace(ROOT / "attribution-control.json")
        command = ["docker", "exec", "-e", "PYTHONUNBUFFERED=1", CONTAINER,
                   "python3", "-m", "sglang.benchmark.serving", "--backend", "sglang",
                   "--host", "127.0.0.1", "--port", "18080", "--model", "/models/glm-q",
                   "--served-model-name", "glm-5", "--dataset-name", "random-ids",
                   "--num-prompts", str(count), "--max-concurrency", str(concurrency),
                   "--random-input-len", str(inputs), "--random-output-len", str(outputs),
                   "--random-range-ratio", "1.0", "--request-rate", "inf", "--warmup-requests", "2",
                   "--seed", "910", "--tokenize-prompt", "--output-details", "--output-file", f"/campaign/{name}.jsonl"]
        point = {"name": name, "enabled": enabled, "input_length": inputs, "output_length": outputs,
                 "concurrency": concurrency, "requests": count, "start_unix": time.time(), "command": command,
                 "status": "running"}
        state["cases"].append(point)
        progress.write_text(json.dumps(state, indent=2) + "\n")
        with (ROOT / f"{name}.log").open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
        point.update({"returncode": result.returncode, "end_unix": time.time()})
        if result.returncode:
            point["status"] = state["status"] = "failed"
            progress.write_text(json.dumps(state, indent=2) + "\n")
            raise RuntimeError(f"Serving failed: {name}")
        measured = json.loads((ROOT / f"{name}.jsonl").read_text().splitlines()[-1])
        if measured["completed"] != count or any(measured["errors"]) or measured["input_lens"] != [inputs] * count or measured["output_lens"] != [outputs] * count:
            raise ValueError(f"Formal workload mismatch: {name}")
        point["status"] = "completed"
        point["metrics"] = {key: measured[key] for key in ("mean_ttft_ms", "mean_tpot_ms", "input_throughput", "output_throughput")}
        progress.write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps({"case": name, **point["metrics"]}), flush=True)
    state["status"] = "completed"
    progress.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
