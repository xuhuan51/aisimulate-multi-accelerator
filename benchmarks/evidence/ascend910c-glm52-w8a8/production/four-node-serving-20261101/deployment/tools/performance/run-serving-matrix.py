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
import sys
import time


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(ROOT / "service"))
from pd_lock import lock_path


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


def process_is_same(pid, start_ticks):
    try:
        fields = pathlib.Path(f"/proc/{pid}/stat").read_text().split()
    except FileNotFoundError:
        return False
    return fields[2] != "Z" and fields[21] == str(start_ticks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=pathlib.Path, default=SCRIPT_DIR / "comparison-configs.json")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--suffix", default="20261101")
    parser.add_argument("--adopt-existing-first", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--adopt-running-configuration")
    parser.add_argument("--adopt-pid", type=int)
    parser.add_argument("--adopt-process-start", type=int)
    parser.add_argument("--campaign-timeout", type=int, default=43200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = lock_path(ROOT, "matrix").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads(args.plan.read_text())
    saved_plan = args.output / "comparison-plan.json"
    if args.resume and saved_plan.exists() and json.loads(saved_plan.read_text()) != plan:
        raise ValueError("Cannot resume a changed comparison plan")
    (args.output / "comparison-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    workload = plan["workload"]
    configurations = plan["configurations"]
    if args.adopt_running_configuration and (not args.resume or args.adopt_pid is None or args.adopt_process_start is None):
        raise ValueError("Adopting an active sweep requires --resume, its PID and process start ticks")
    if args.adopt_running_configuration and args.adopt_running_configuration not in {configuration["label"] for configuration in configurations}:
        raise ValueError("Unknown active configuration")
    directories = [args.output / f'{configuration["label"]}-{args.suffix}' for configuration in configurations]
    state_path = args.output / "matrix-progress.json"
    state = {"status": "running", "started": time.time(), "finished_configurations": [], "warmup_requests": workload["warmup_requests"]}
    if args.resume:
        previous_state = json.loads(state_path.read_text())
        history_path = args.output / f"matrix-state-before-resume-{time.time_ns()}.json"
        history_path.write_text(json.dumps(previous_state, indent=2) + "\n")
        state["started"] = previous_state["started"]
        state["finished_configurations"] = previous_state["finished_configurations"]
        state["resumed_at"] = time.time()

    def save():
        state["updated"] = time.time()
        state["completed_points"] = summarize(directories, args.output / "matrix-summary.csv")
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)

    current_env = None
    try:
        for index, (configuration, directory) in enumerate(zip(configurations, directories)):
            directory.mkdir(parents=True, exist_ok=True)
            env = environment(configuration, workload, directory)
            if configuration["label"] in state["finished_configurations"]:
                missing = [concurrency for concurrency in workload["concurrencies"] if not (directory / f'c{concurrency}-n{concurrency * workload["prompt_multiplier"]}' / ".complete").exists()]
                if missing or not (directory / "worker-logs").is_dir():
                    raise ValueError(f"Completed configuration has missing evidence: {configuration['label']}")
                print(f"Resume: retaining completed configuration {configuration['label']}", flush=True)
                current_env = env
                continue
            if args.resume and configuration["label"] != args.adopt_running_configuration:
                for point in directory.glob("c*-n*"):
                    if point.is_dir() and not (point / ".complete").exists():
                        attempts = directory / "failed-attempts"
                        attempts.mkdir(exist_ok=True)
                        point.rename(attempts / f"{point.name}-{time.time_ns()}")
            state["configuration"] = configuration["label"]
            state["phase"] = "waiting_for_existing" if index == 0 and args.adopt_existing_first else "starting"
            save()
            if configuration["label"] == args.adopt_running_configuration:
                state["phase"] = "adopting_running_sweep"
                state["adopted_pid"] = args.adopt_pid
                state["adopted_process_start"] = args.adopt_process_start
                save()
                deadline = time.monotonic() + args.campaign_timeout
                print(f"Adopting live sweep PID {args.adopt_pid}; serving and benchmark remain running", flush=True)
                while process_is_same(args.adopt_pid, args.adopt_process_start):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Adopted sweep exceeded campaign timeout")
                    save()
                    time.sleep(15)
            elif index == 0 and args.adopt_existing_first:
                print("Adopting existing recommended-config sweep; waiting for its benchmark lock", flush=True)
                with lock_path(ROOT, "benchmark").open("a") as benchmark_lock:
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
                execute([str(SCRIPT_DIR / "run-aisim-comparison.sh")], env, directory / "campaign.log", args.campaign_timeout)
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
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
