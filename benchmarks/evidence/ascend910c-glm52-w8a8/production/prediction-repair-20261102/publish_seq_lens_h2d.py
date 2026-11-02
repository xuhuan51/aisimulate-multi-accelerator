import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
PACKAGE = REPO / "python/aisimulate"


def run(arguments):
    command = ["docker", "run", "--rm", "--network", "none", "--cpuset-cpus", "304-307", "--memory", "8g",
               "-v", f"{REPO}:{REPO}", "-w", str(PACKAGE), "--entrypoint", "python3",
               "aisimulate-910c:forward-batch-repair-20261102", *map(str, arguments)]
    subprocess.run(command, check=True)


def main():
    campaign = ROOT / "seq-lens-h2d-database-node20"
    observations = json.loads((campaign / "observations.json").read_text())
    if observations["status"] != "completed" or len(observations["cases"]) != 8:
        raise ValueError("Require the complete transfer curve")
    if any(row["cv"] > 0.1 for row in observations["cases"]):
        raise ValueError("Retain the rejected campaign and recollect unstable measurements")
    snapshot = campaign / "collector-snapshot"
    snapshot.mkdir(exist_ok=False)
    for filename in ("ascend_vector.py", "ascend_vector_kernels.py", "ascend_vector_cases.py", "ascend_vector_export.py"):
        shutil.copy2(PACKAGE / "collector/sglang" / filename, snapshot / filename)
    exported = ROOT / "seq-lens-h2d-published"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPO).strip()
    run(["-m", "collector.sglang.ascend_vector_export", campaign / "observations.json", "--plan", campaign / "plan.json",
         "--runtime-identity", campaign / "runtime-identity.json", "--output-dir", exported,
         "--collector-ref", revision + "+worktree-source-sha256-in-provenance"])
    target = PACKAGE / "src/aisimulate_core/systems/data/ascend910c/vector/sglang/0.5.22"
    before = ROOT / "vector-before-metadata"
    before.mkdir(exist_ok=False)
    for filename in ("ascend_vector_perf.parquet", "collection_meta.yaml"):
        shutil.copy2(target / filename, before / filename)
    shutil.copy2(target / "metadata/provenance.json", before / "provenance.json")
    merged = ROOT / "vector-with-metadata"
    run(["-m", "collector.sglang.ascend_curve_append", "--operation", "vector", "--output-dir", merged, before, exported])
    manifest = {"operation": "Append a new disjoint measured eager CPU INT32 transfer curve; preserve all historical timings",
                "before_sha256": {}, "after_sha256": {}, "new_curve_rows": len(observations["cases"])}
    for filename in ("ascend_vector_perf.parquet", "collection_meta.yaml", "provenance.json"):
        manifest["before_sha256"][filename] = hashlib.sha256((before / filename).read_bytes()).hexdigest()
        manifest["after_sha256"][filename] = hashlib.sha256((merged / filename).read_bytes()).hexdigest()
        destination = target / ("metadata/provenance.json" if filename == "provenance.json" else filename)
        shutil.copy2(merged / filename, destination)
    (ROOT / "seq-lens-h2d-publication.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
