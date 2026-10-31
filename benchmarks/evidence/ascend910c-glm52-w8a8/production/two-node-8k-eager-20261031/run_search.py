import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
OUTPUT = ROOT / "aisimulate-output/two-node-8k-eager-20261031"
CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64, 128)
CPU_IDS = tuple(range(280, 296))


def run_case(index, concurrency, mode):
    case = f"{mode}-isl8192-osl1024-c{concurrency}-n{2 * concurrency}"
    destination = OUTPUT / case
    destination.mkdir(parents=True, exist_ok=False)
    runtime_uid = int(os.environ.get("SUDO_UID", os.getuid()))
    runtime_gid = int(os.environ.get("SUDO_GID", os.getgid()))
    os.chown(destination, runtime_uid, runtime_gid)
    config = Path(__file__).with_name(f"search-{mode}.yaml")
    command = [
        "bash", str(ROOT / "scripts/run-ascend910c.sh"),
        "recommend", "--stack", "engine", "--config", str(config),
        "--set", f"traffic.load.concurrency={concurrency}",
        "--set", f"traffic.stop.requests={2 * concurrency}",
        "--output-dir", str(destination / "result"),
    ]
    environment = {
        **os.environ,
        "AISIMULATE_910C_MODEL": "/data_lib/data/models/glm-q",
        "AISIMULATE_910C_CPUS": "2",
        "AISIMULATE_910C_CPUSET": ",".join(map(str, CPU_IDS[2 * (index % 8):2 * (index % 8) + 2])),
        "AISIMULATE_910C_MEMORY": "8g",
    }
    started = time.monotonic()
    with (destination / "command.log").open("w") as log:
        completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    record = {
        "case": case,
        "mode": mode,
        "concurrency": concurrency,
        "requests": 2 * concurrency,
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
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(CONCURRENCIES)) as pool:
        cases = [(concurrency, mode) for mode in ("pd", "agg") for concurrency in CONCURRENCIES]
        futures = [pool.submit(run_case, index, concurrency, mode) for index, (concurrency, mode) in enumerate(cases)]
        records = [future.result() for future in futures]
    (OUTPUT / "execution-summary.json").write_text(json.dumps(records, indent=2) + "\n")
    assert all(record["returncode"] == 0 and record["finite_search_exhausted"] for record in records)


if __name__ == "__main__":
    main()
