import argparse
import concurrent.futures
import copy
import datetime
import gzip
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import replay_pd_campaign as replay
import report_joint_comparison as history

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
EVIDENCE = ROOT / "joint-latest-replay-20261103/attempt2"
OUTPUT = REPO / "aisimulate-output/joint-latest-replay-20261103/attempt2"
IMAGE = "aisimulate-910c:mixed-round-repair-20261103-r2"
IMAGE_ID = "sha256:9f8b7b90c438b10daa203fadb3beebc537e61c722019915265aa002d7738c6c9"
SCOPES = ("isolated_kernels", "runtime_layers")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text() != payload:
        raise ValueError(f"Refusing to overwrite different evidence: {path}")
    path.write_text(payload)
    path.chmod(0o644)


def clean_estimator(config):
    if isinstance(config, dict):
        config.pop("eager_runtime_overhead", None)
        for value in config.values():
            clean_estimator(value)
        if "estimator_config" in config:
            config["estimator_config"].setdefault("correction", {})["enabled"] = False
    elif isinstance(config, list):
        for value in config:
            clean_estimator(value)


def prepare():
    if (EVIDENCE / "manifest.json").exists():
        return json.loads((EVIDENCE / "manifest.json").read_text())
    actual_image = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE], text=True
    ).strip()
    if actual_image != IMAGE_ID:
        raise ValueError("The latest verified image tag changed")
    source = ROOT / "joint-serving-comparison.json"
    records = json.loads(source.read_text())["rows"]
    if len(records) != 39 or sum(row["requests"] for row in records) != 1900:
        raise ValueError("Expected the original 39 rows and 1900 successful requests")
    archive_manifest = ROOT / "joint-serving-comparison-manifest.json"
    for artifact in json.loads(archive_manifest.read_text())["artifacts"]:
        archive = REPO / artifact["archive"]
        if digest(archive) != artifact["archive_sha256"]:
            raise ValueError(f"Archived input changed: {archive}")
        if hashlib.sha256(gzip.decompress(archive.read_bytes())).hexdigest() != artifact["source_sha256"]:
            raise ValueError(f"Archived payload changed: {archive}")
    prepared = []
    for number, row in enumerate(records, 1):
        execution_path = Path(row["execution"])
        execution = json.loads(execution_path.read_text())
        command = execution["command"]
        template = Path(command[command.index("--config") + 1])
        if digest(template) != execution["template_sha256"]:
            raise ValueError(f"Template changed: {template}")
        for key in ("runtime", "measured"):
            if digest(Path(execution[f"{key}_path"])) != execution[f"{key}_sha256"]:
                raise ValueError(f"Measurement source changed: row {number}, {key}")
        runtime = json.loads(Path(execution["runtime_path"]).read_text())
        replay.COMPARISON.validate_runtime(row["configuration_spec"], runtime)
        logits_audit = replay.COMPARISON.audit_logits_layout(runtime, Path(execution["runtime_path"]))
        measured = json.loads(Path(execution["measured_path"]).read_text().splitlines()[-1])
        if measured != execution["measured"]:
            raise ValueError(f"Measurement snapshot differs: row {number}")
        count = row["requests"]
        if (measured["completed"] != count or any(measured["errors"])
                or measured["input_lens"] != [row["input_length"]] * count
                or measured["output_lens"] != [row["output_length"]] * count):
            raise ValueError(f"Invalid measured workload: row {number}")
        settings = [command[position + 1] for position, token in enumerate(command) if token == "--set"]
        prepared.append({"number": number, "historical": row, "template_text": template.read_text(),
                         "settings": settings, "logits_audit": logits_audit,
                         "source_execution_sha256": digest(execution_path),
                         "source_template_sha256": digest(template),
                         "source_measurement_sha256": execution["measured_sha256"],
                         "source_runtime_sha256": execution["runtime_sha256"]})
    resolver = """
import json,sys,yaml
resolved=[]
for record in json.load(sys.stdin):
    config=yaml.safe_load(record['template_text'])
    for setting in record['settings']:
        key,value=setting.split('=',1)
        parent=config
        parts=key.split('.')
        for part in parts[:-1]:
            parent=parent.setdefault(part,{})
        parent[parts[-1]]=yaml.safe_load(value)
    resolved.append(config)
json.dump(resolved,sys.stdout)
"""
    configs = json.loads(subprocess.check_output(
        ["docker", "run", "--rm", "-i", "--network", "none", "--entrypoint", "python3", IMAGE_ID, "-c", resolver],
        input=json.dumps(prepared), text=True,
    ))
    jobs = {}
    for record, config in zip(prepared, configs):
        row = record["historical"]
        clean_estimator(config)
        record["removed_residuals"] = [setting for setting in record.pop("settings") if "eager_runtime_overhead=" in setting]
        record.pop("template_text")
        if config["traffic"]["source"]["type"] != "synthetic" or config["traffic"]["load"]["type"] != "concurrency":
            raise ValueError("These comparisons must use native closed-loop arrivals")
        if config["engine"]["nextn"] != 0 or config["engine"]["decoder_replay"]:
            raise ValueError("Speculative decode or graph was enabled")
        record["jobs"] = {}
        for scope in SCOPES:
            candidate = copy.deepcopy(config)
            candidate["engine"]["workers"]["decode"]["timing"]["estimator_config"]["op_level"]["ascend"]["decode_timing_scope"] = scope
            candidate["engine"]["workers"]["prefill"]["timing"]["estimator_config"]["op_level"]["ascend"]["decode_timing_scope"] = "isolated_kernels"
            payload = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
            config_id = hashlib.sha256(payload.encode()).hexdigest()
            record["jobs"][scope] = config_id
            if config_id not in jobs:
                config_path = EVIDENCE / "configs" / f"{config_id}.yaml"
                jobs[config_id] = {"id": config_id, "scope": scope, "config": str(config_path),
                                   "candidate": candidate, "row_numbers": [],
                                   "requests": row["requests"], "input_length": row["input_length"],
                                   "output_length": row["output_length"],
                                   "destination": str(OUTPUT / scope / config_id)}
            jobs[config_id]["row_numbers"].append(record["number"])
    serialized = json.loads(subprocess.check_output(
        ["docker", "run", "--rm", "-i", "--network", "none", "--entrypoint", "python3", IMAGE_ID,
         "-c", "import json,sys,yaml;json.dump([yaml.safe_dump(item,sort_keys=False) for item in json.load(sys.stdin)],sys.stdout)"],
        input=json.dumps([job.pop("candidate") for job in jobs.values()]), text=True,
    ))
    for job, text in zip(jobs.values(), serialized):
        config_path = Path(job["config"])
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(text)
        config_path.chmod(0o644)
        job["config_sha256"] = digest(config_path)
    for directory in (OUTPUT, EVIDENCE):
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o755)
    manifest = {"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "image": IMAGE, "image_id": IMAGE_ID,
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "historical_json_sha256": digest(source), "historical_manifest_sha256": digest(archive_manifest),
                "scope_policy": "Both scopes are attempted independently for every unique configuration. No fallback, fitted forward residual, or measured arrival/batch injection.",
                "records": prepared, "jobs": list(jobs.values())}
    save(EVIDENCE / "manifest.json", manifest)
    return manifest


def run_job(job):
    destination = Path(job["destination"])
    evidence = EVIDENCE / "jobs" / job["id"]
    completed = evidence / "execution.json"
    if completed.exists():
        previous = json.loads(completed.read_text())
        if previous["config_sha256"] != digest(Path(job["config"])) or previous["image_id"] != IMAGE_ID:
            raise ValueError(f"Changed rerun identity: {job['id']}")
        return previous
    destination.mkdir(parents=True, exist_ok=False)
    evidence.mkdir(parents=True, exist_ok=True)
    command = ["bash", str(REPO / "scripts/run-ascend910c.sh"), "predict", "--stack", "engine",
               "--config", job["config"], "--capture-per-request", "--detail", "source",
               "--output-dir", str(destination / "result")]
    environment = {**os.environ, "AISIMULATE_910C_IMAGE": IMAGE_ID,
                   "AISIMULATE_910C_MODEL": "/data_lib/data/models/glm-q", "AISIMULATE_910C_CPUSET": "280-295",
                   "AISIMULATE_910C_CPUS": "4", "AISIMULATE_910C_MEMORY": "8g",
                   "SUDO_UID": str(os.getuid()), "SUDO_GID": str(os.getgid())}
    started = time.monotonic()
    with (evidence / "command.log").open("w") as stream:
        result = subprocess.run(command, cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=1200)
    execution = {**job, "image_id": IMAGE_ID, "command": command, "exit_code": result.returncode,
                 "wall_seconds": time.monotonic() - started,
                 "status": "completed" if result.returncode == 0 else "failed", "artifacts": []}
    if result.returncode == 0:
        prediction_path = destination / "result/prediction.json"
        requests_path = destination / "result/requests.jsonl"
        prediction = json.loads(prediction_path.read_text())
        requests = [json.loads(line) for line in requests_path.read_text().splitlines()]
        expected = {"completed_requests": job["requests"], "total_input_tokens": job["requests"] * job["input_length"],
                    "total_output_tokens": job["requests"] * job["output_length"],
                    "failed_requests": 0, "canceled_requests": 0, "rejected_requests": 0}
        if any(prediction[key] != value for key, value in expected.items()):
            raise ValueError(f"Prediction token/request conservation failed: {job['id']}")
        if len(requests) != job["requests"] or any(request["input_length"] != job["input_length"]
                or request["output_length"] != job["output_length"] or request["terminal_status"] != "completed" for request in requests):
            raise ValueError(f"Invalid individual prediction records: {job['id']}")
        execution["metrics"] = {metric: prediction[key] for metric, (key, _) in replay.COMPARISON.METRIC_PAIRS.items()}
        execution["batches"] = {role: history.batch_summary(prediction["forward_pass_batches"], role) for role in ("prefill", "decode")}
        execution["validation"] = expected
        for path in (prediction_path, requests_path, destination / "result/resource-runtime.json"):
            archive = evidence / (path.name + ".gz")
            archive.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
            execution["artifacts"].append({"source": str(path), "source_sha256": digest(path),
                                           "archive": str(archive.relative_to(REPO)), "archive_sha256": digest(archive)})
    else:
        execution["failure_log_tail"] = (evidence / "command.log").read_text()[-7000:]
    save(completed, execution)
    print(json.dumps({"rows": job["row_numbers"], "scope": job["scope"], "status": execution["status"],
                      "seconds": round(execution["wall_seconds"], 2), "metrics": execution.get("metrics")}), flush=True)
    return execution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rows", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    manifest = prepare()
    jobs = [job for job in manifest["jobs"] if args.rows is None or set(args.rows).intersection(job["row_numbers"])]
    print(json.dumps({"rows": len(manifest["records"]), "unique_jobs": len(manifest["jobs"]), "selected_jobs": len(jobs)}), flush=True)
    if not args.prepare_only:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(run_job, jobs))


if __name__ == "__main__":
    main()
