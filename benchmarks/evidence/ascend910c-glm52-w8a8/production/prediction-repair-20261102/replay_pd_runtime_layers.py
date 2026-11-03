import argparse
import gzip
import hashlib
import json
import subprocess
import time
from pathlib import Path

import yaml

from replay_pd_client_dispatch import REPO, ROOT, digest, local_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numbers", type=int, nargs="+")
    parser.add_argument("--expert-workload", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    profile = json.loads(args.profile.read_text())
    if profile["schema"] != "aisimulate-client-dispatch-profile-fit/v1":
        raise ValueError("Independent CPU client preparation profile required")
    source = ROOT / "joint-latest-replay-20261103/attempt2/manifest.json"
    history = json.loads(source.read_text())
    archive = ROOT / "joint-serving-comparison-manifest.json"
    for artifact in json.loads(archive.read_text())["artifacts"]:
        path = REPO / artifact["archive"]
        if digest(path) != artifact["archive_sha256"]:
            raise ValueError(f"Archived evidence changed: {path}")
        if hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() != artifact["source_sha256"]:
            raise ValueError(f"Archived payload changed: {path}")
    originals = {job["id"]: job for job in history["jobs"]}
    manifest = {
        "status": "pending",
        "scope": "Same-condition unconditional historical PD replay; no observed arrival, routing, batch or forward residual input",
        "scope_policy": "Decode runtime_layers selected in advance for both audited topologies; missing coverage remains a failed prediction, without fallback",
        "source_manifest_sha256": digest(source), "source_archives_sha256": digest(archive),
        "cpu_profile_sha256": digest(args.profile), "records": [], "jobs": {},
        "source_sha256": digest(Path(__file__)),
    }
    if args.expert_workload:
        manifest["expert_workload_sha256"] = digest(args.expert_workload)
        manifest["scope"] += "; independent expert count observations supplied for prefill compute shape only"
    for record in history["records"]:
        if args.numbers and record["number"] not in args.numbers:
            continue
        original = originals[record["jobs"]["runtime_layers"]]
        config_path = local_path(original["config"])
        if digest(config_path) != original["config_sha256"]:
            raise ValueError("Historical configuration changed")
        config = yaml.safe_load(config_path.read_text())
        if "eager_runtime_overhead" in json.dumps(config):
            raise ValueError("A forward residual is not permitted")
        ascend = config["engine"]["workers"]["decode"]["timing"]["estimator_config"]["op_level"]["ascend"]
        if ascend["decode_timing_scope"] != "runtime_layers" or ascend["decode_execution_mode"] != "eager":
            raise ValueError("Unexpected historical timing scope")
        config["traffic"]["load"]["client_dispatch"] = profile["client_dispatch"]
        if args.expert_workload:
            config["engine"]["workers"]["prefill"]["timing"]["estimator_config"]["op_level"]["ascend"]["prefill_expert_workload"] = json.loads(args.expert_workload.read_text())
        key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        manifest["records"].append({"number": record["number"], "historical": record["historical"], "job": key})
        if key not in manifest["jobs"]:
            folder = args.output / "jobs" / key
            folder.mkdir(parents=True, exist_ok=False)
            destination = folder / "config.yaml"
            destination.write_text(yaml.safe_dump(config, sort_keys=False))
            manifest["jobs"][key] = {"config": str(destination), "config_sha256": digest(destination),
                                     "folder": str(folder), "row_numbers": [], "status": "pending"}
        manifest["jobs"][key]["row_numbers"].append(record["number"])
    destination = args.output / "manifest.json"

    def save():
        destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    manifest["status"] = "running"
    save()
    for job in manifest["jobs"].values():
        folder = Path(job["folder"])
        command = ["aisimulate", "predict", "--stack", "engine", "--config", job["config"],
                   "--capture-per-request", "--detail", "source", "--output-dir", str(folder / "prediction")]
        started = time.time()
        try:
            with (folder / "prediction.log").open("w") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
            job.update(returncode=result.returncode, status="completed" if result.returncode == 0 else "failed")
        except subprocess.TimeoutExpired:
            job.update(returncode=None, status="timeout")
        job.update(command=command, wall_seconds=time.time() - started)
        save()
        print(json.dumps({name: job[name] for name in ("row_numbers", "status", "wall_seconds")}), flush=True)
    manifest["status"] = "completed" if all(job["status"] == "completed" for job in manifest["jobs"].values()) else "coverage_or_execution_failures"
    save()
    if manifest["status"] != "completed":
        raise RuntimeError("All replay attempts recorded; unsupported configurations were not substituted")


if __name__ == "__main__":
    main()
