import argparse
import concurrent.futures
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

METRICS = {
    "output_tokens_per_s": ("output_throughput_tok_s", 1.0),
    "mean_ttft_s": ("mean_ttft_ms", 0.001),
    "p99_ttft_s": ("p99_ttft_ms", 0.001),
    "mean_tpot_ms": ("mean_tpot_ms", 1.0),
    "p99_tpot_ms": ("p99_tpot_ms", 1.0),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--decode-reserved-tokens", type=int, default=512)
    parser.add_argument("--scheduler-alignment", choices=("runtime", "legacy"), default="runtime")
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[4]
    template = yaml.safe_load((repository / "examples/cli/ascend910c-glm52-pd-8k1k.yaml").read_text())
    baseline = json.loads(args.baseline.read_text())
    selected = [
        row
        for row in baseline["records"]
        if row["deployment"] != "colocated_two_nodes" and row["workload"] in ("w1", "w2")
    ]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    configs = args.output_dir / "configs"
    configs.mkdir()
    tables = repository / "python/aisimulate/src/aisimulate_core/systems/data/ascend910c"
    manifest = {
        "decode_reserved_tokens": args.decode_reserved_tokens,
        "scheduler_alignment": args.scheduler_alignment,
        "ignore_eos": args.ignore_eos,
        "scheduler_limits": (
            {"chunked_prefill_size": 16384, "max_prefill_tokens": 32768, "global_max_running_requests": 64}
            if args.scheduler_alignment == "runtime"
            else {"chunked_prefill_size": 8192, "max_prefill_tokens": 16384, "max_sequences_per_rank": 64}
        ),
        "source_baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "runtime_source": subprocess.check_output(
            ["git", "-c", f"safe.directory={repository}", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip(),
        "native_sha256": hashlib.sha256(
            (repository / "build/ascend910c-runtime/_runtime.abi3.so").read_bytes()
        ).hexdigest(),
        "tables": {
            str(path.relative_to(tables)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(tables.rglob("*.parquet"))
        },
        "limitations": [
            "Declared-config alignment, not verified historical-runtime parity: "
            "baseline image and library versions are absent.",
            "Use each row's completed request count, fixed label-derived lengths, "
            "closed-loop concurrency and no cache reuse.",
            "P/D each use attention TP8, ADP2, DeepEP EP16; P memory fraction0.88 and D0.90; each replica is one node.",
            "Use context180000; runtime mode aligns the declared global chunk16384, max-prefill32768 "
            "and global max-running64 (ADP2 ->32/rank). Legacy mode retains the previous mismatched limits.",
            "Native round-robin routing does not reproduce the document's cache-aware prefill router "
            "or missing arrival trace.",
            "P TASK_QUEUE_ENABLE1 and historical eager/overlap details are not a separately calibrated table identity.",
            "No empirical adjustment multiplier, queue cap fitted from throughput, "
            "or reference measurements are fed into prediction.",
        ],
        "records": [],
    }

    def execute(row):
        name = f"{row['deployment']}-{row['workload']}-c{row['concurrency']}"
        config = copy.deepcopy(template)
        config["traffic"]["source"]["input_tokens"] = row["input_tokens_assumed_from_label"]
        config["traffic"]["source"]["output_tokens"] = row["output_tokens_assumed_from_label"]
        config["traffic"]["load"]["concurrency"] = row["concurrency"]
        config["traffic"]["stop"]["requests"] = row["completed_requests"]
        engine = config["engine"]
        engine["model"] = str(
            repository
            / "benchmarks/evidence/ascend910c-glm52-w8a8/production/deepep-contract-node19-20261024/model-metadata"
        )
        engine["context_length"] = 180000
        prefill_count, decode_count = row["deployment"].replace("d", "").split("p")
        for role, count in (("prefill", int(prefill_count)), ("decode", int(decode_count))):
            worker = engine["workers"][role]
            worker["parallelism"]["replicas"] = count
            if args.ignore_eos is not None:
                worker["scheduler"]["ignore_eos"] = args.ignore_eos
            if args.scheduler_alignment == "runtime":
                worker["scheduler"].update(
                    chunked_prefill_size=16384,
                    max_prefill_tokens=32768,
                    max_sequences=64 // worker["parallelism"]["attention_data"],
                )
            else:
                worker["scheduler"].update(
                    chunked_prefill_size=8192,
                    max_prefill_tokens=16384,
                    max_sequences=64,
                )
        engine["workers"]["prefill"]["scheduler"]["max_batched_tokens"] = 16384
        engine["workers"]["prefill"]["kv_cache"]["capacity"]["memory_fraction"] = 0.88
        engine["workers"]["decode"]["scheduler"]["decode_reserved_tokens"] = args.decode_reserved_tokens
        config_path = configs / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        output = args.output_dir / name
        command = [
            sys.executable,
            "-m",
            "aisimulate",
            "predict",
            "--stack",
            "engine",
            "--config",
            str(config_path),
            "--capture-per-request",
            "--detail",
            "memory,time,source",
            "--output-dir",
            str(output),
        ]
        started = time.monotonic()
        timed_out = False
        with (args.output_dir / f"{name}.log").open("w") as stream:
            try:
                process = subprocess.run(command, cwd=repository, stdout=stream, stderr=subprocess.STDOUT, timeout=150)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                timed_out, returncode = True, None
        result = {
            "name": name,
            "measured": row,
            "config": str(config_path),
            "command": command,
            "returncode": returncode,
            "timeout": timed_out,
            "wall_seconds": time.monotonic() - started,
        }
        path = output / "prediction.json"
        if returncode == 0 and path.exists():
            prediction = json.loads(path.read_text())
            result["predicted"] = {name: prediction[key] * factor for name, (key, factor) in METRICS.items()}
            result["relative_error"] = {name: result["predicted"][name] / row[name] - 1 for name in METRICS}
            result["completed_requests"] = prediction["completed_requests"]
            result["memory_diagnostics"] = prediction.get("memory_diagnostics")
            result["performance_diagnostics"] = prediction.get("performance_diagnostics")
            result["prediction_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            if prediction["completed_requests"] != row["completed_requests"]:
                raise ValueError(f"Incomplete request replay: {name}")
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallelism) as pool:
        futures = [pool.submit(execute, row) for row in selected]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            manifest["records"].append(result)
            (args.output_dir / "summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "case": result["name"],
                        "returncode": result["returncode"],
                        "completed": len(manifest["records"]),
                        "total": len(selected),
                    }
                ),
                flush=True,
            )
    manifest["status"] = (
        "completed" if all(row["returncode"] == 0 for row in manifest["records"]) else "completed_with_failures"
    )
    (args.output_dir / "summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
