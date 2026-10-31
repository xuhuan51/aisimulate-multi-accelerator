import argparse
import concurrent.futures
import json
import os
import queue
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
OUTPUT = ROOT / "aisimulate-output/single-node-8k-eager-20261031"
CONCURRENCIES = (1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128)
CPU_SETS = queue.Queue()
for start in range(280, 296, 4):
    CPU_SETS.put(",".join(map(str, range(start, start + 4))))


def run_case(concurrency, requests, output):
    case = f"isl8192-osl1024-c{concurrency}-n{requests}"
    destination = output / case
    destination.mkdir(parents=True, exist_ok=False)
    os.chown(destination, int(os.environ.get("SUDO_UID", os.getuid())), int(os.environ.get("SUDO_GID", os.getgid())))
    command = [
        "bash", str(ROOT / "scripts/run-ascend910c.sh"),
        "recommend", "--stack", "engine", "--config", str(Path(__file__).with_name("search.yaml")),
        "--set", f"traffic.load.concurrency={concurrency}",
        "--set", f"traffic.stop.requests={requests}",
        "--output-dir", str(destination / "result"),
    ]
    cpuset = CPU_SETS.get()
    environment = {
        **os.environ,
        "AISIMULATE_910C_MODEL": "/data_lib/data/models/glm-q",
        "AISIMULATE_910C_CPUS": "4",
        "AISIMULATE_910C_CPUSET": cpuset,
        "AISIMULATE_910C_MEMORY": "8g",
    }
    started = time.monotonic()
    try:
        with (destination / "command.log").open("w") as log:
            completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    finally:
        CPU_SETS.put(cpuset)
    record = {
        "case": case,
        "concurrency": concurrency,
        "requests": requests,
        "command": command,
        "runtime_environment": {key: value for key, value in environment.items() if key.startswith("AISIMULATE_910C_")},
        "returncode": completed.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "finite_search_exhausted": "finite search space exhausted" in (destination / "command.log").read_text(),
    }
    report_path = destination / "result/recommendation.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        record["counts"] = report["counts"]
        record["best"] = report["views"]["top_n"][:1]
    (destination / "execution.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrencies", type=int, nargs="+", default=CONCURRENCIES)
    parser.add_argument("--requests", type=int)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    assert all(concurrency > 0 for concurrency in args.concurrencies)
    assert len(set(args.concurrencies)) == len(args.concurrencies)
    assert args.requests is None or args.requests >= max(args.concurrencies)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_case, concurrency, args.requests or 2 * concurrency, output) for concurrency in args.concurrencies]
        records = [future.result() for future in futures]
    (output / "execution-summary.json").write_text(json.dumps(records, indent=2) + "\n")
    assert all(record["returncode"] == 0 and record["finite_search_exhausted"] for record in records)


if __name__ == "__main__":
    main()
