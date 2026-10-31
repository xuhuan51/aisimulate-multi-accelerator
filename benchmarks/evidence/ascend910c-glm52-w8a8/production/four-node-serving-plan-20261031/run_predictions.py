import concurrent.futures
import csv
import hashlib
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path


DESTINATION = Path(__file__).resolve().parent
ROOT = DESTINATION.parents[4]
OUTPUT = ROOT / "aisimulate-output/four-node-serving-plan-20261031"
CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64, 128)
METRICS = ("num_requests", "completed_requests", "total_input_tokens", "total_output_tokens", "output_throughput_tok_s", "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms", "duration_ms")


def write(path, text):
    assert not path.exists(), path
    patch = "*** Begin Patch\n*** Add File: " + str(path) + "\n" + "".join("+" + line + "\n" for line in text.splitlines()) + "*** End Patch\n"
    subprocess.run(["apply_patch"], input=patch, text=True, check=True, stdout=subprocess.DEVNULL)


def run_case(configuration, concurrency):
    name = configuration.stem + f"-c{concurrency}"
    destination = OUTPUT / name
    destination.mkdir(parents=True, exist_ok=False)
    os.chown(destination, int(os.environ.get("SUDO_UID", os.getuid())), int(os.environ.get("SUDO_GID", os.getgid())))
    command = ["bash", str(ROOT / "scripts/run-ascend910c.sh"), "predict", "--stack", "engine", "--config", str(configuration), "--set", f"traffic.load.concurrency={concurrency}", "--capture-per-request", "--output-dir", str(destination / "result")]
    environment = {**os.environ, "AISIMULATE_910C_MODEL": "/data_lib/data/models/glm-q", "AISIMULATE_910C_CPUSET": "280-295", "AISIMULATE_910C_CPUS": "4", "AISIMULATE_910C_MEMORY": "8g"}
    with (destination / "command.log").open("w") as log:
        completed = subprocess.run(command, env=environment, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=300)
    assert completed.returncode == 0, name
    prediction = json.loads((destination / "result/prediction.json").read_text())
    assert prediction["completed_requests"] == prediction["num_requests"] == 256
    assert prediction["total_input_tokens"] == prediction["committed_prefill_tokens"] == 256 * 8192
    assert prediction["total_output_tokens"] == 256 * 1024
    assert all(prediction[field] == 0 for field in ("rejected_requests", "failed_requests", "canceled_requests"))
    requests = [json.loads(line) for line in (destination / "result/requests.jsonl").read_text().splitlines()]
    assert len(requests) == 256 and all(request["input_length"] == 8192 and request["output_length"] == 1024 and request["terminal_status"] == "completed" for request in requests)
    record = {"configuration": configuration.name, "concurrency": concurrency, **{field: prediction[field] for field in METRICS}}
    (destination / "execution.json").write_text(json.dumps({"command": command, "result": record, "requests_checked": len(requests)}, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    return record


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_case, configuration, concurrency) for configuration in sorted(DESTINATION.glob("*.yaml")) for concurrency in CONCURRENCIES]
        records = [future.result() for future in futures]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    write(DESTINATION / "predictions.csv", stream.getvalue())
    archive = DESTINATION / "prediction-evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in sorted(OUTPUT.glob("*/*")):
            bundle.add(path, arcname=str(path.relative_to(ROOT)))
    report = {"date": "2026-10-31", "runtime_source": "db5f2e721670d3eb2a10785fe55ad2d28af81349", "scope": "Four fixed64-device PD topologies;N256,C1..128 powers of2;not a full search;CPU-only", "serving_measured": False, "records": records, "archive": {"path": str(archive.relative_to(ROOT)), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}}
    write(DESTINATION / "prediction-report.json", json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
