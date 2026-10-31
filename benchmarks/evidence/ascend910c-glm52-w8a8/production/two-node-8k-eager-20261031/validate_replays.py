import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import tarfile
from pathlib import Path

from summarize import DESTINATION, METRICS, OUTPUT, ROOT, write


def validate(index, winner):
    name = f"{winner['layout']}-c{winner['concurrency']}"
    destination = OUTPUT / "replays" / name
    destination.mkdir(parents=True, exist_ok=False)
    os.chown(destination, int(os.environ.get("SUDO_UID", os.getuid())), int(os.environ.get("SUDO_GID", os.getgid())))
    config = DESTINATION / winner["prediction_config"]
    command = [
        "bash", str(ROOT / "scripts/run-ascend910c.sh"), "predict", "--stack", "engine",
        "--config", str(config), "--capture-per-request", "--output-dir", str(destination / "result"),
    ]
    environment = {
        **os.environ,
        "AISIMULATE_910C_MODEL": "/data_lib/data/models/glm-q",
        "AISIMULATE_910C_CPUS": "4",
        "AISIMULATE_910C_CPUSET": ",".join(map(str, range(280 + 4 * (index % 4), 284 + 4 * (index % 4)))),
        "AISIMULATE_910C_MEMORY": "8g",
    }
    with (destination / "command.log").open("w") as log:
        completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=300)
    assert completed.returncode == 0, name
    prediction = json.loads((destination / "result/prediction.json").read_text())
    differences = {field: prediction[field] - winner["metrics"][field] for field in METRICS}
    assert all(math.isclose(prediction[field], winner["metrics"][field], rel_tol=1e-12, abs_tol=1e-9) for field in METRICS), (name, differences)
    requests = [json.loads(line) for line in (destination / "result/requests.jsonl").read_text().splitlines()]
    assert len(requests) == 2 * winner["concurrency"]
    assert len({request["uuid"] for request in requests}) == len(requests)
    for request in requests:
        assert request["input_length"] == 8192
        assert request["output_length"] == request["requested_output_length"] == 1024
        assert request["terminal_status"] == "completed"
        assert request["arrival_time_ms"] <= request["first_admit_ms"] <= request["first_token_ms"] <= request["last_token_ms"]
        assert math.isclose(request["ttft_ms"], request["first_token_ms"] - request["arrival_time_ms"], rel_tol=1e-12, abs_tol=1e-9)
    waiting = [request["first_admit_ms"] - request["arrival_time_ms"] for request in requests]
    record = {
        "layout": winner["layout"], "concurrency": winner["concurrency"], "prediction_config": winner["prediction_config"],
        "requests_checked": len(requests), "returncode": completed.returncode, "command": command,
        "search_differences": differences, "metrics": {field: prediction[field] for field in METRICS},
        "mean_first_admission_wait_ms": sum(waiting) / len(waiting),
        "max_first_admission_wait_ms": max(waiting),
        "total_readmissions": sum(request["readmission_count"] for request in requests),
        "maximum_admissions_per_request": max(request["admission_count"] for request in requests),
    }
    (destination / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"layout": record["layout"], "concurrency": record["concurrency"], "requests_checked": record["requests_checked"], "matches_search": True}), flush=True)
    return record


def main():
    report = json.loads((DESTINATION / "report.json").read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(validate, index, winner) for index, winner in enumerate(report["winners_by_layout"])]
        records = [future.result() for future in futures]
    assert len(records) == 24
    archive = DESTINATION / "replay-evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(Path(__file__), arcname=str(Path(__file__).relative_to(ROOT)))
        for path in sorted((OUTPUT / "replays").glob("*/*")):
            bundle.add(path, arcname=str(path.relative_to(ROOT)))
    validation = {
        "date": "2026-10-31", "runtime_source": report["runtime_source"], "image_id": report["image_id"],
        "cpu_only": True, "serving_measured": False, "records": records,
        "archive": {"path": str(archive.relative_to(ROOT)), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "bytes": archive.stat().st_size},
    }
    write(DESTINATION / "replay-validation.json", json.dumps(validation, indent=2) + "\n")
    print(json.dumps({"replays": len(records), "requests_checked": sum(record["requests_checked"] for record in records), "archive_bytes": archive.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
