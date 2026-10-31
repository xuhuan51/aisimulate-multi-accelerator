#!/usr/bin/env python3

import argparse
import csv
import fcntl
import gzip
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import time


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]


def environment(configuration, workload, directory):
    result = os.environ.copy()
    result.update({key: str(value) for key, value in configuration.items() if key != "label"})
    result.update({
        "CONFIG_LABEL": configuration["label"], "RUN_DIR": str(directory),
        "CONCURRENCIES": " ".join(map(str, workload["concurrencies"])),
        "PROMPT_MULTIPLIER": str(workload["prompt_multiplier"]),
        "WARMUP_REQUESTS": str(workload["warmup_requests"]),
        "BENCHMARK_SEED": str(workload["seed"]),
        "PREFILL_HCCL_BUFFSIZE": "650", "DECODE_HCCL_BUFFSIZE": "650",
        "DISABLE_GRAPH": "1", "DISABLE_OVERLAP": "1", "DISABLE_PREFIX_CACHE": "1",
        "ENABLE_SPECULATIVE_DECODING": "0", "PREFILL_MAX_RUNNING_REQUESTS": "128",
        "PREFILL_MEM_FRACTION_STATIC": "0.88", "DECODE_MEM_FRACTION_STATIC": "0.90",
        "PREFILL_CHUNK_SIZE": "16384", "PREFILL_MAX_TOKENS": "32768",
    })
    return result


def execute(command, env, log_path, timeout):
    print(f"Running {command[0]} -> {log_path}", flush=True)
    with log_path.open("a") as log:
        result = subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}), inspect {log_path}")


def archive_logs(env, directory):
    destination = directory / "worker-logs"
    destination.mkdir(exist_ok=True)
    nodes = []
    for role, key in (("prefill", "PREFILL_NODES"), ("decode", "DECODE_NODES")):
        for index, spec in enumerate(env[key].split()):
            nodes.append((spec.split(":")[0], f"glm52-aisim-eager-{role}-{index}"))
    nodes.append(("10.120.72.20", "glm52-aisim-eager-router"))
    for node, container in nodes:
        text = f'source {shlex.quote(str(ROOT / "service/pd-common.sh"))}; node_exec {shlex.quote(node)} {shlex.quote("docker logs " + shlex.quote(container))}'
        with gzip.open(destination / f"{node}-{container}.log.gz", "wb") as compressed:
            process = subprocess.Popen(["bash", "-c", text], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            shutil.copyfileobj(process.stdout, compressed)
            if process.wait(timeout=60):
                raise RuntimeError(f"Could not archive {container} on {node}")


def summarize(directories, target):
    rows = []
    for directory in directories:
        for point in sorted(directory.glob("c*-n*")):
            if not (point / ".complete").exists():
                continue
            data = json.loads(sorted(point.glob("bench_*.jsonl"))[-1].read_text().splitlines()[-1])
            metrics = json.loads((point / "metrics-summary.json").read_text())
            row = {"configuration": directory.name, "point": point.name}
            for key in ("completed", "duration", "input_throughput", "output_throughput", "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms", "concurrency"):
                row[key] = data.get(key)
            for key in ("decode_running_peak", "decode_running_mean_when_nonzero", "decode_rank_running_peak", "metric_endpoint_errors"):
                row[key] = metrics.get(key)
            rows.append(row)
    if rows:
        with target.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=pathlib.Path, default=SCRIPT_DIR / "comparison-configs.json")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--suffix", default="20261101")
    parser.add_argument("--adopt-existing-first", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / ".aisim-matrix.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads(args.plan.read_text())
    (args.output / "comparison-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    workload = plan["workload"]
    configurations = plan["configurations"]
    directories = [args.output / f'{configuration["label"]}-{args.suffix}' for configuration in configurations]
    state_path = args.output / "matrix-progress.json"
    state = {"status": "running", "started": time.time(), "finished_configurations": [], "warmup_requests": workload["warmup_requests"]}

    def save():
        state["updated"] = time.time()
        state["completed_points"] = summarize(directories, args.output / "matrix-summary.csv")
        state_path.write_text(json.dumps(state, indent=2) + "\n")

    current_env = None
    try:
        for index, (configuration, directory) in enumerate(zip(configurations, directories)):
            directory.mkdir(parents=True, exist_ok=True)
            env = environment(configuration, workload, directory)
            state["configuration"] = configuration["label"]
            state["phase"] = "waiting_for_existing" if index == 0 and args.adopt_existing_first else "starting"
            save()
            if index == 0 and args.adopt_existing_first:
                print("Adopting existing recommended-config sweep; waiting for its benchmark lock", flush=True)
                with (ROOT / ".aisim-benchmark.lock").open("a") as benchmark_lock:
                    while True:
                        try:
                            fcntl.flock(benchmark_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            fcntl.flock(benchmark_lock, fcntl.LOCK_UN)
                            break
                        except BlockingIOError:
                            save()
                            time.sleep(15)
            else:
                if current_env is not None:
                    execute([str(ROOT / "service/stop-pd.sh")], current_env, directory / "stop-previous.log", 180)
                execute([str(ROOT / "service/start-pd.sh")], env, directory / "startup.log", 3600)
                state["phase"] = "benchmarking"
                save()
                execute([str(SCRIPT_DIR / "run-aisim-comparison.sh")], env, directory / "campaign.log", 21600)
            current_env = env
            missing = [concurrency for concurrency in workload["concurrencies"] if not (directory / f'c{concurrency}-n{concurrency * workload["prompt_multiplier"]}' / ".complete").exists()]
            if missing:
                raise RuntimeError(f'{configuration["label"]}: incomplete concurrency points {missing}; retaining failed evidence')
            archive_logs(env, directory)
            state["finished_configurations"].append(configuration["label"])
            save()
        state["phase"] = "restoring_recommended"
        save()
        execute([str(ROOT / "service/stop-pd.sh")], current_env, args.output / "stop-final-control.log", 180)
        recommended_env = environment(configurations[0], workload, directories[0])
        execute([str(ROOT / "service/start-pd.sh")], recommended_env, args.output / "restore-recommended.log", 3600)
        state["status"] = "measured"
        state["phase"] = "awaiting_offline_comparison"
        save()
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = str(error)
        save()
        raise


if __name__ == "__main__":
    main()
