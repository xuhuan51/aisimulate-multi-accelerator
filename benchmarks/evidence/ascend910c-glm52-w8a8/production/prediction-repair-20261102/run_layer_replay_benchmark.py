import argparse
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
    destination = root / "benchmark-launch.json"
    if destination.exists():
        raise FileExistsError(destination)
    for attempt in range(600):
        try:
            with urllib.request.urlopen("http://127.0.0.1:18080/get_server_info", timeout=2) as response:
                info = json.load(response)
            break
        except (OSError, ValueError):
            time.sleep(1)
    else:
        raise TimeoutError("Layer replay service is not ready")
    if (info["tp_size"], info["attn_dp_size"], info["ep_size"]) != (16, 4, 16):
        raise ValueError("Unexpected topology")
    if not info["disable_overlap_schedule"] or info["speculative_algorithm"] is not None:
        raise ValueError("Unexpected execution mode")
    if any(info["cuda_graph_config"][phase]["backend"] != "disabled" for phase in ("prefill", "decode")):
        raise ValueError("Unexpected graph mode")
    (root / "server-info.json").write_text(json.dumps(info, indent=2) + "\n")
    command = ["docker", "exec", args.container, "python3", "-m", "sglang.benchmark.serving",
               "--backend", "sglang", "--host", "127.0.0.1", "--port", "18080",
               "--model", "/models/glm-q", "--served-model-name", "glm-5", "--dataset-name", "random-ids",
               "--num-prompts", "4", "--max-concurrency", "4", "--random-input-len", "8192",
               "--random-output-len", "192", "--random-range-ratio", "1.0", "--request-rate", "inf",
               "--warmup-requests", "2", "--seed", "910", "--tokenize-prompt", "--output-details",
               "--output-file", "/campaign/layer-replay-8k-c4.jsonl"]
    manifest = {"scope": "Diagnostic request metrics contain the isolation replay pause; excluded from serving accuracy acceptance",
                "command": command, "start_unix": time.time(), "status": "running"}
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    with (root / "benchmark.log").open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
    manifest.update(end_unix=time.time(), returncode=result.returncode,
                    probe_rank_files=len(list(root.glob("layer-replay-rank*.json"))))
    path = root / "layer-replay-8k-c4.jsonl"
    if path.exists():
        measured = json.loads(path.read_text().splitlines()[-1])
        manifest.update(completed_requests=measured["completed"], request_errors=measured["errors"],
                        input_lengths=measured["input_lens"], output_lengths=measured["output_lens"])
    succeeded = (result.returncode == 0 and manifest.get("completed_requests") == 4
                 and not any(manifest.get("request_errors", [])) and manifest["probe_rank_files"] == 16
                 and manifest.get("input_lengths") == [8192] * 4 and manifest.get("output_lengths") == [192] * 4)
    manifest["status"] = "completed" if succeeded else "failed"
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest.get(key) for key in ("status", "returncode", "probe_rank_files", "completed_requests")}))
    if not succeeded:
        raise RuntimeError("Layer replay benchmark did not complete")


if __name__ == "__main__":
    main()
