import argparse
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
SOURCE = ROOT.parent / "four-node-serving-20261101/compare_serving_predictions.py"
SPEC = importlib.util.spec_from_file_location("serving_comparison", SOURCE)
COMPARISON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARISON)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_job(job):
    destination = Path(job["destination"])
    destination.mkdir(parents=True, exist_ok=True)
    if os.geteuid() == 0:
        os.chown(destination, int(os.environ.get("SUDO_UID", os.getuid())), int(os.environ.get("SUDO_GID", os.getgid())))
    execution = destination / "execution.json"
    if execution.exists():
        previous = json.loads(execution.read_text())
        if any(previous.get(key) != job[key] for key in ("command", "measured_sha256", "runtime_sha256", "template_sha256", "image_id")):
            raise ValueError(f"Stale result: {destination}")
        return previous["comparison"]
    if (destination / "result").exists():
        raise FileExistsError(destination / "result")
    if (destination / "command.log").exists():
        (destination / "command.log").rename(destination / f"failed-command-{time.time_ns()}.log")
    environment = {**os.environ, "AISIMULATE_910C_IMAGE": job["image_id"], "AISIMULATE_910C_MODEL": "/data_lib/data/models/glm-q",
                   "AISIMULATE_910C_CPUSET": "280-295", "AISIMULATE_910C_CPUS": "4", "AISIMULATE_910C_MEMORY": "8g"}
    with (destination / "command.log").open("w") as stream:
        subprocess.run(job["command"], cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=600, check=True)
    predicted = json.loads((destination / "result/prediction.json").read_text())
    requests = [json.loads(line) for line in (destination / "result/requests.jsonl").read_text().splitlines()]
    expected = {"completed_requests": job["requests"], "total_input_tokens": job["requests"] * job["input_length"],
                "total_output_tokens": job["requests"] * job["output_length"], "failed_requests": 0, "canceled_requests": 0, "rejected_requests": 0}
    if any(predicted[key] != value for key, value in expected.items()):
        raise ValueError(f"Predicted workload mismatch: {destination}")
    if len(requests) != job["requests"] or any(row["input_length"] != job["input_length"] or row["output_length"] != job["output_length"] or row["terminal_status"] != "completed" for row in requests):
        raise ValueError(f"Predicted per-request mismatch: {destination}")
    row = {key: job[key] for key in ("case", "concurrency", "requests", "input_length", "output_length", "mode")}
    for metric, (prediction_key, measurement_key) in COMPARISON.METRIC_PAIRS.items():
        row[f"predicted_{metric}"] = predicted[prediction_key]
        row[f"measured_{metric}"] = job["measured"][measurement_key]
        row[f"error_percent_{metric}"] = 100 * (predicted[prediction_key] / job["measured"][measurement_key] - 1)
    execution.write_text(json.dumps({**job, "comparison": row}, indent=2) + "\n")
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default="aisimulate-910c:eager-overhead-20261102")
    parser.add_argument("--ingress-base-ms", type=float)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    campaign = ROOT / "pd-paired-1p1d"
    state = json.loads((campaign / "progress.json").read_text())
    template = ROOT.parent / "four-node-serving-plan-20261031/control-1p3d-tp16.yaml"
    image_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", args.image], text=True).strip()
    profile = json.loads((ROOT / "eager-overhead-tp16-adp4.json").read_text())
    configuration = {"label": "paired-1p1d-tp16", "PREFILL_TP_SIZE": 16, "PREFILL_DP_SIZE": 4, "PREFILL_EP_SIZE": 16,
                     "DECODE_TP_SIZE": 16, "DECODE_DP_SIZE": 4, "DECODE_EP_SIZE": 16, "DECODE_NNODES": 1,
                     "DECODE_MAX_RUNNING_REQUESTS": 128, "PREFILL_NODES": "10.120.72.20:19000:19998", "DECODE_NODES": "10.120.72.25:19003"}
    jobs = []
    for case in state["cases"]:
        if case["status"] != "completed":
            continue
        runtime_path = campaign / case["name"] / "runtime-config.json"
        runtime = json.loads(runtime_path.read_text())
        COMPARISON.validate_runtime(configuration, runtime)
        layout = COMPARISON.audit_logits_layout(runtime, runtime_path)
        raw_path = campaign / "prefill" / f"{case['name']}.jsonl"
        measured = json.loads(raw_path.read_text().splitlines()[-1])
        if measured["completed"] != case["requests"] or measured["input_lens"] != [case["input_length"]] * case["requests"] or measured["output_lens"] != [case["output_length"]] * case["requests"] or any(measured["errors"]):
            raise ValueError(f"Invalid measurement: {case['name']}")
        destination = args.output.resolve() / case["name"]
        command = ["bash", str(REPO / "scripts/run-ascend910c.sh"), "predict", "--stack", "engine", "--config", str(template),
                   "--capture-per-request", "--output-dir", str(destination / "result")]
        settings = {"engine.workers.decode.parallelism.replicas": 1, "traffic.source.input_tokens": case["input_length"],
                    "traffic.source.output_tokens": case["output_length"], "traffic.stop.requests": case["requests"],
                    "traffic.load.concurrency": case["concurrency"],
                    "engine.workers.decode.timing.estimator_config.op_level.ascend.eager_runtime_overhead": profile}
        for role in ("prefill", "decode"):
            rows = [row for row in layout if row["role"] == role]
            if len(rows) != 1:
                raise ValueError(f"Expected one {role} replica")
            for key in ("enable_dp_lm_head", "enable_tp_lm_head_all_to_all"):
                settings[f"engine.workers.{role}.timing.estimator_config.op_level.ascend.{key}"] = rows[0][f"actual_{key}"]
        if args.ingress_base_ms is not None:
            settings.update({"engine.tokenization.workers": 1, "engine.tokenization.base_ms": args.ingress_base_ms})
        for key, value in settings.items():
            command.extend(["--set", f"{key}={json.dumps(value, separators=(',', ':'))}"])
        jobs.append({"case": case["name"], "concurrency": case["concurrency"], "requests": case["requests"],
                     "input_length": case["input_length"], "output_length": case["output_length"],
                     "mode": "native" if args.ingress_base_ms is None else f"ingress-sensitivity-{args.ingress_base_ms}ms",
                     "command": command, "destination": str(destination), "measured": measured,
                     "measured_path": str(raw_path), "measured_sha256": digest(raw_path), "runtime_path": str(runtime_path),
                     "runtime_sha256": digest(runtime_path), "template_sha256": digest(template), "image_id": image_id})
    args.output.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_job, jobs))
    (args.output / "comparison.json").write_text(json.dumps({"scope": "Independent node20+25 paired PD collection; unchanged node20 C1 eager decode calibration", "rows": rows}, indent=2) + "\n")
    with (args.output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
