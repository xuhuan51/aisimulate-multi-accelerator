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
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    root = Path(__file__).resolve().parent
    base = yaml.safe_load((root / "full-request-aggregate-8k1k-20261103.yaml").read_text())
    info = json.loads((args.campaign / "holdout-server-info.json").read_text())
    if info["max_total_num_tokens"] % info["page_size"]:
        raise ValueError("Runtime KV capacity is not page aligned")
    base["engine"]["workers"]["aggregated"]["kv_cache"]["capacity"]["blocks"] = info["max_total_num_tokens"] // info["page_size"]
    plan = json.loads((args.campaign / "context-holdouts.json").read_text())["plan"]
    results = []
    for name, input_length, output_length, concurrency, prompts in plan:
        for scope in ("isolated_kernels", "runtime_layers"):
            config = copy.deepcopy(base)
            config["traffic"]["source"].update(input_tokens=input_length, output_tokens=output_length)
            config["traffic"]["load"]["concurrency"] = concurrency
            config["traffic"]["stop"]["requests"] = prompts
            config["engine"]["estimator_config"]["op_level"]["ascend"]["decode_timing_scope"] = scope
            case_root = args.output / f"{name}-{scope}"
            case_root.mkdir()
            config_path = case_root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            command = ["aisimulate", "predict", "--stack", "engine", "--config", str(config_path),
                       "--capture-per-request", "--detail", "source", "--output-dir", str(case_root / "prediction")]
            started = time.time()
            with (case_root / "prediction.log").open("w") as stream:
                outcome = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
            row = {"case": name, "timing_scope": scope, "command": command,
                   "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                   "wall_seconds": time.time() - started, "returncode": outcome.returncode}
            results.append(row)
            (args.output / "manifest.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    if any(row["returncode"] != 0 for row in results):
        raise RuntimeError("A replay failed; inspect preserved logs")


if __name__ == "__main__":
    main()
