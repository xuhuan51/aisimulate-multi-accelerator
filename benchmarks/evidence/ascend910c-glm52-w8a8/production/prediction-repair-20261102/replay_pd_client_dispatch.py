import argparse
import copy
import gzip
import hashlib
import json
import subprocess
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_path(value):
    path = Path(value)
    for prefix in ("/tgm_rent_data/data/youxingye/aisimulate-910c", "/home/youxingye/youxingye_data/aisimulate-910c"):
        if path.is_relative_to(prefix):
            return REPO / path.relative_to(prefix)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    jobs_by_id = {job["id"]: job for job in history["jobs"]}
    manifest = {"scope": "Unconditional closed-loop replay of the original 39 PD configurations; independently profiled client dispatch, no measured request arrivals, batches or forward residuals supplied",
                "scope_policy": "Predeclared original main-table policy: rows 1-6 (TP32 decode) use isolated_kernels, all other rows use runtime_layers. No selection by prediction error and no failure fallback.",
                "source_manifest_sha256": digest(source), "source_archives_sha256": digest(archive),
                "cpu_profile_sha256": digest(args.profile), "records": [], "jobs": {}}
    for record in history["records"]:
        scope = "isolated_kernels" if record["number"] <= 6 else "runtime_layers"
        original = jobs_by_id[record["jobs"][scope]]
        config_path = local_path(original["config"])
        if digest(config_path) != original["config_sha256"]:
            raise ValueError("Historical configuration changed")
        config = yaml.safe_load(config_path.read_text())
        if "eager_runtime_overhead" in json.dumps(config):
            raise ValueError("A forward residual is not permitted")
        summary = {"number": record["number"], "scope": scope,
                   "historical": record["historical"], "jobs": {}}
        for mode in ("instantaneous", "profiled_client"):
            candidate = copy.deepcopy(config)
            if mode == "profiled_client":
                candidate["traffic"]["load"]["client_dispatch"] = profile["client_dispatch"]
            key = hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest()
            summary["jobs"][mode] = key
            if key not in manifest["jobs"]:
                folder = args.output / "jobs" / key
                folder.mkdir(parents=True, exist_ok=False)
                destination = folder / "config.yaml"
                destination.write_text(yaml.safe_dump(candidate, sort_keys=False))
                manifest["jobs"][key] = {"mode": mode, "scope": scope, "config": str(destination),
                                        "config_sha256": digest(destination), "folder": str(folder),
                                        "row_numbers": [], "status": "pending"}
            manifest["jobs"][key]["row_numbers"].append(record["number"])
        manifest["records"].append(summary)
    destination = args.output / "manifest.json"

    def save():
        destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    save()
    for job in manifest["jobs"].values():
        folder = Path(job["folder"])
        command = ["aisimulate", "predict", "--stack", "engine", "--config", job["config"],
                   "--capture-per-request", "--detail", "source", "--output-dir", str(folder / "prediction")]
        started = time.time()
        with (folder / "prediction.log").open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
        job.update(command=command, returncode=result.returncode, wall_seconds=time.time() - started,
                   status="completed" if result.returncode == 0 else "failed")
        save()
        print(json.dumps({key: job[key] for key in ("row_numbers", "mode", "scope", "status", "wall_seconds")}), flush=True)
    if any(job["status"] != "completed" for job in manifest["jobs"].values()):
        raise RuntimeError("PD replay failures are retained; no fallback was attempted")


if __name__ == "__main__":
    main()
