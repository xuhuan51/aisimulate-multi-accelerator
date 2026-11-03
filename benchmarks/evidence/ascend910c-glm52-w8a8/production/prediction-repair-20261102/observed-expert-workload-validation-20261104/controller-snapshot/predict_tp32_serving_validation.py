import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-info", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert-workload", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    source = Path(__file__).parent
    base = yaml.safe_load((source / "full-request-aggregate-8k1k-20261103.yaml").read_text())
    info = json.loads(args.server_info.read_text())
    profile = json.loads(args.profile.read_text())
    if (info["tp_size"], info["attn_dp_size"], info["ep_size"]) != (32, 32, 32):
        raise ValueError("Unexpected serving topology")
    if info["max_total_num_tokens"] % info["page_size"] or info["max_running_requests"] % info["attn_dp_size"]:
        raise ValueError("Unaligned capacity or running-request limit")
    if not info["disable_overlap_schedule"] or info["speculative_algorithm"] is not None:
        raise ValueError("Unexpected overlap or speculation")
    if any(info["cuda_graph_config"][phase]["backend"] != "disabled" for phase in ("prefill", "decode")):
        raise ValueError("Unexpected graph execution")
    if profile["schema"] != "aisimulate-client-dispatch-profile-fit/v1":
        raise ValueError("Independent CPU client profile required")
    worker = base["engine"]["workers"]["aggregated"]
    worker["parallelism"].update(tensor=1, attention_data=32, moe_expert=32)
    worker["kv_cache"].update(block_size=info["page_size"], prefix_caching=not info["disable_radix_cache"])
    worker["kv_cache"]["capacity"]["blocks"] = info["max_total_num_tokens"] // info["page_size"]
    worker["scheduler"].update(chunked_prefill_size=info["chunked_prefill_size"] * info["attn_dp_size"],
                                max_prefill_tokens=info["max_prefill_tokens"],
                                max_batched_tokens=info["max_prefill_tokens"],
                                max_sequences=info["max_running_requests"] // info["attn_dp_size"])
    base["engine"]["estimator_config"]["op_level"]["ascend"].update(
        decode_timing_scope="runtime_layers", enable_dp_lm_head=info["enable_dp_lm_head"],
        enable_tp_lm_head_all_to_all=info["enable_tp_lm_head_all_to_all"])
    base["traffic"]["load"]["client_dispatch"] = profile["client_dispatch"]
    if args.expert_workload:
        base["engine"]["estimator_config"]["op_level"]["ascend"]["prefill_expert_workload"] = json.loads(args.expert_workload.read_text())
    manifest = {"status": "running", "cases": [],
                "scope": "Unconditional full-request predictions from predeclared workload plan, actual launch settings and fixed resolved KV capacity only; no measured arrival, batch, forward or serving residual input",
                "normalization": "get_server_info chunked_prefill_size is already per-DP; CLI/replay takes the global value. max_sequences is per-DP.",
                "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (args.server_info, args.plan, args.profile, Path(__file__))}}
    destination = args.output / "manifest.json"
    if args.expert_workload:
        manifest["source_sha256"][str(args.expert_workload)] = hashlib.sha256(args.expert_workload.read_bytes()).hexdigest()

    def save():
        destination.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    for name, input_length, output_length, concurrency, prompts in json.loads(args.plan.read_text()):
        config = copy.deepcopy(base)
        config["traffic"]["source"].update(input_tokens=input_length, output_tokens=output_length)
        config["traffic"]["load"]["concurrency"] = concurrency
        config["traffic"]["stop"]["requests"] = prompts
        folder = args.output / name
        folder.mkdir(exist_ok=False)
        config_path = folder / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        command = ["aisimulate", "predict", "--stack", "engine", "--config", str(config_path),
                   "--capture-per-request", "--detail", "source", "--output-dir", str(folder / "prediction")]
        started = time.time()
        with (folder / "prediction.log").open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
        case = {"id": name, "input_length": input_length, "output_length": output_length,
                "concurrency": concurrency, "prompts": prompts, "command": command, "returncode": result.returncode,
                "config": str(config_path), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "folder": str(folder), "wall_seconds": time.time() - started,
                "status": "completed" if result.returncode == 0 else "failed"}
        if result.returncode == 0:
            prediction = json.loads((folder / "prediction/prediction.json").read_text())
            expected = {"completed_requests": prompts, "total_input_tokens": input_length * prompts,
                        "total_output_tokens": output_length * prompts, "failed_requests": 0,
                        "rejected_requests": 0, "canceled_requests": 0}
            if any(prediction[key] != value for key, value in expected.items()):
                case.update(status="conservation_failure")
        manifest["cases"].append(case)
        save()
        print(json.dumps({key: case[key] for key in ("id", "status", "wall_seconds")}), flush=True)
    manifest["status"] = "completed" if all(case["status"] == "completed" for case in manifest["cases"]) else "coverage_or_execution_failures"
    save()
    if manifest["status"] != "completed":
        raise RuntimeError("All prediction attempts retained; no unsupported shape fallback")


if __name__ == "__main__":
    main()
