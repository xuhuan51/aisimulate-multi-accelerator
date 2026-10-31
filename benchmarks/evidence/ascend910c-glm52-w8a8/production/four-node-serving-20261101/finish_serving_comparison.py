import argparse
import fcntl
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import tarfile
import time

from report_serving_comparison import archive_measurement


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE / "deployment/service"))
from pd_lock import lock_path

matrix_spec = importlib.util.spec_from_file_location("pd_matrix_summary", HERE / "deployment/tools/performance/run-serving-matrix.py")
matrix_summary = importlib.util.module_from_spec(matrix_spec)
matrix_spec.loader.exec_module(matrix_summary)


def commit(paths, message):
    relative = [str(path.relative_to(ROOT)) for path in paths]
    subprocess.run(["git", "add", "--", *relative], cwd=ROOT, check=True)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *relative], cwd=ROOT)
    if changed.returncode == 0:
        return
    if changed.returncode != 1:
        raise RuntimeError("Could not inspect staged measurement evidence")
    subprocess.run(["git", "-c", "user.name=youxingye", "-c", "user.email=youxingye@npu1-20", "commit", "--only", "-m", message, "--", *relative], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-root", type=pathlib.Path, required=True)
    parser.add_argument("--prediction-root", type=pathlib.Path, default=ROOT / "aisimulate-output/four-node-serving-matched-20261101")
    parser.add_argument("--commit-measurements", action="store_true")
    args = parser.parse_args()
    lock = lock_path(args.measurement_root, "comparison").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = args.measurement_root / "comparison-progress.json"
    state = {"status": "waiting_for_measurements", "archived_configurations": [], "started": time.time()}

    def save():
        state["updated"] = time.time()
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)

    try:
        while True:
            try:
                matrix = json.loads((args.measurement_root / "matrix-progress.json").read_text())
            except json.JSONDecodeError:
                time.sleep(1)
                continue
            plan = json.loads((args.measurement_root / "comparison-plan.json").read_text())
            directories = [args.measurement_root / f"{configuration['label']}-20261101" for configuration in plan["configurations"]]
            state["completed_points"] = matrix_summary.summarize(directories, args.measurement_root / "live-summary.csv")
            state["configuration"] = matrix.get("configuration")
            state["measurement_phase"] = matrix.get("phase")
            for label in matrix["finished_configurations"]:
                if label in state["archived_configurations"]:
                    continue
                source = args.measurement_root / f"{label}-20261101"
                archive_measurement(label, source, HERE / "measurements")
                paths = [HERE / "measurements" / f"{label}{suffix}" for suffix in (".tar.gz", "-manifest.json")]
                if args.commit_measurements:
                    commit(paths, f"test: record {label} 8k serving concurrency sweep")
                state["archived_configurations"].append(label)
                print(f"Archived completed configuration: {label}", flush=True)
            save()
            if matrix["status"] == "failed":
                raise RuntimeError(f"Serving matrix failed: {matrix.get('error')}")
            if matrix["status"] == "measured":
                break
            time.sleep(30)
        state["status"] = "predicting_matched_workloads"
        save()
        subprocess.run(["python3", str(HERE / "compare_serving_predictions.py"), "--measurement-root", str(args.measurement_root), "--output", str(args.prediction_root)], cwd=ROOT, check=True)
        state["status"] = "writing_comparison"
        save()
        subprocess.run(["python3", str(HERE / "report_serving_comparison.py"), "--measurement-root", str(args.measurement_root), "--prediction-root", str(args.prediction_root), "--output", str(HERE / "comparison-report.md")], cwd=ROOT, check=True)
        archive = HERE / "predictions.tar.gz"
        with tarfile.open(archive, "w:gz") as output:
            output.add(args.prediction_root, arcname="predictions")
        manifest = HERE / "prediction-manifest.json"
        manifest.write_text(json.dumps({"source": str(args.prediction_root), "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "bytes": archive.stat().st_size, "comparison_jobs": 48, "accuracy_accepted": False}, indent=2) + "\n")
        paths = [archive, manifest, HERE / "comparison-report.md", HERE / "ranking.csv"]
        for name in ("comparison-plan.json", "matrix-summary.csv", "matrix-progress.json", "host-runtime-identities.json", "matrix-handoff-recovery.json", "matrix-local-lock-migration.json", "live-summary.csv"):
            source = args.measurement_root / name
            if name.startswith("matrix-") and name.endswith(("recovery.json", "migration.json")) and not source.exists():
                continue
            destination = HERE / name
            destination.write_bytes(source.read_bytes())
            paths.append(destination)
        for name in ("comparison.csv", "comparison.json"):
            destination = HERE / name
            destination.write_bytes((args.prediction_root / name).read_bytes())
            paths.append(destination)
        if args.commit_measurements:
            commit(paths, "test: compare four-node serving measurements with matched offline predictions")
        state["status"] = "compared"
        state["report"] = str(HERE / "comparison-report.md")
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
