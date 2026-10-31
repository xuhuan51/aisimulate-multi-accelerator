import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import tarfile

from summarize import DESTINATION, METRICS, ROOT, write


OUTPUT = ROOT / "aisimulate-output/single-node-8k-replays-20261031"


def validate(winner):
    name = winner["prediction_config"].removesuffix(".yaml")
    destination = OUTPUT / name
    destination.mkdir(parents=True, exist_ok=False)
    os.chown(destination, int(os.environ.get("SUDO_UID", os.getuid())), int(os.environ.get("SUDO_GID", os.getgid())))
    command = ["bash", str(ROOT / "scripts/run-ascend910c.sh"), "predict", "--stack", "engine", "--config", str(DESTINATION / winner["prediction_config"]), "--capture-per-request", "--output-dir", str(destination / "result")]
    environment = {**os.environ, "AISIMULATE_910C_MODEL": "/data_lib/data/models/glm-q", "AISIMULATE_910C_CPUSET": "280-295", "AISIMULATE_910C_CPUS": "4", "AISIMULATE_910C_MEMORY": "8g"}
    with (destination / "command.log").open("w") as log:
        completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=300)
    assert completed.returncode == 0, name
    prediction = json.loads((destination / "result/prediction.json").read_text())
    differences = {field: prediction[field] - winner["metrics"][field] for field in METRICS}
    assert all(math.isclose(prediction[field], winner["metrics"][field], rel_tol=1e-12, abs_tol=1e-8) for field in METRICS), differences
    requests = [json.loads(line) for line in (destination / "result/requests.jsonl").read_text().splitlines()]
    assert len(requests) == int(prediction["num_requests"])
    assert all(request["input_length"] == 8192 and request["output_length"] == 1024 and request["terminal_status"] == "completed" for request in requests)
    assert all(request["arrival_time_ms"] <= request["first_admit_ms"] <= request["first_token_ms"] <= request["last_token_ms"] for request in requests)
    waiting = [request["first_admit_ms"] - request["arrival_time_ms"] for request in requests]
    record = {"configuration": winner["prediction_config"], "requests_checked": len(requests), "metrics": {field: prediction[field] for field in METRICS}, "search_differences": differences, "mean_first_admission_wait_ms": sum(waiting) / len(waiting), "total_readmissions": sum(request["readmission_count"] for request in requests), "command": command}
    (destination / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    report = json.loads((DESTINATION / "report.json").read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(validate, report["selected_replays"]))
    archive = DESTINATION / "replay-evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in sorted(OUTPUT.glob("*/*")):
            bundle.add(path, arcname=str(path.relative_to(ROOT)))
    write(DESTINATION / "replay-validation.json", json.dumps({"records": records, "serving_measured": False, "archive": {"path": str(archive.relative_to(ROOT)), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}}, indent=2) + "\n")
    print(json.dumps({"replays": len(records), "requests_checked": sum(record["requests_checked"] for record in records)}), flush=True)


if __name__ == "__main__":
    main()
