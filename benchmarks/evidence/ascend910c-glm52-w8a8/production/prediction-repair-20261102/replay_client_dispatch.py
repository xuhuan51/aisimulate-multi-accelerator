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
    parser.add_argument("--campaign", type=Path, action="append", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    profile = json.loads(args.profile.read_text())
    if profile["schema"] != "aisimulate-client-dispatch-profile-fit/v1":
        raise ValueError("Independent CPU client profile required")
    root = Path(__file__).resolve().parent
    base = yaml.safe_load((root / "full-request-aggregate-8k1k-20261103.yaml").read_text())
    base["engine"]["estimator_config"]["op_level"]["ascend"]["decode_timing_scope"] = "runtime_layers"
    manifest = {"scope": "Unconditional concurrency forecast with independently measured CPU dispatch costs; no serving arrivals, routing, batches or forward latencies supplied",
                "profile_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
                "client_dispatch": profile["client_dispatch"], "cases": []}
    destination = args.output / "manifest.json"
    for campaign in args.campaign:
        measured = json.loads((campaign / "context-holdouts.json").read_text())
        if measured["status"] != "completed":
            raise ValueError(f"Incomplete serving campaign: {campaign}")
        info = json.loads((campaign / "holdout-server-info.json").read_text())
        if (info["tp_size"], info["attn_dp_size"], info["ep_size"]) != (16, 4, 16):
            raise ValueError("Unexpected serving topology")
        if info["max_total_num_tokens"] % info["page_size"]:
            raise ValueError("KV token capacity is not page aligned")
        for case in measured["cases"]:
            for mode in ("instantaneous", "profiled_client"):
                config = copy.deepcopy(base)
                config["engine"]["workers"]["aggregated"]["kv_cache"]["capacity"]["blocks"] = (
                    info["max_total_num_tokens"] // info["page_size"])
                config["traffic"]["source"].update(input_tokens=case["input_length"], output_tokens=case["output_length"])
                config["traffic"]["load"]["concurrency"] = case["concurrency"]
                config["traffic"]["stop"]["requests"] = case["prompts"]
                if mode == "profiled_client":
                    config["traffic"]["load"]["client_dispatch"] = profile["client_dispatch"]
                folder = args.output / campaign.name / case["id"] / mode
                folder.mkdir(parents=True, exist_ok=False)
                config_path = folder / "config.yaml"
                config_path.write_text(yaml.safe_dump(config, sort_keys=False))
                command = ["aisimulate", "predict", "--stack", "engine", "--config", str(config_path),
                           "--capture-per-request", "--detail", "source", "--output-dir", str(folder / "prediction")]
                started = time.time()
                with (folder / "prediction.log").open("w") as stream:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
                record = {"campaign": str(campaign), "case": case["id"], "mode": mode,
                          "folder": str(folder), "command": command, "returncode": result.returncode,
                          "wall_seconds": time.time() - started,
                          "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}
                manifest["cases"].append(record)
                destination.write_text(json.dumps(manifest, indent=2) + "\n")
                print(json.dumps(record), flush=True)
    if any(case["returncode"] != 0 for case in manifest["cases"]):
        raise RuntimeError("Some predictions failed; all failures are preserved in the manifest")


if __name__ == "__main__":
    main()
