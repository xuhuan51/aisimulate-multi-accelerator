import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    source = Path(__file__).parent
    profile = source / "ingress-validation-node20-20261104/client-dispatch-config.json"
    workload = source / "observed-expert-workload-validation-20261104/prefill-expert-workload.json"
    cohorts = [
        ("original-serving", "layer-tp32-node20-25-20261104-attempt4/serving-validation", "tp32-serving-validation-plan-20261104.json"),
        ("slow-repeats", "layer-tp32-routed-retry-20261104/serving-validation", "tp32-steady-repeat-plan-20261104.json"),
        ("fresh-serving", "tp32-fresh-steady-validation-20261104/serving-validation", "tp32-fresh-steady-plan-20261104.json"),
        ("after-shape-history", "tp32-fresh-steady-validation-20261104/serving-after-shape-history", "tp32-after-shape-history-plan-20261104.json"),
        ("after-layer-replay", "tp32-fresh-steady-validation-20261104/serving-after-layer-replay", "tp32-after-layer-replay-plan-20261104.json"),
    ]
    state = {"status": "running", "cohorts": [],
             "scope": "All predeclared serving cohorts, including unfavorable repeats. Same launch/workload configurations; no observed arrival/batch/latency input or estimator fallback.",
             "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    destination = args.output / "campaign.json"

    def save():
        destination.write_text(json.dumps(state, indent=2) + "\n")

    save()
    for name, actual, plan in cohorts:
        actual = source / actual
        if json.loads((actual / "campaign.json").read_text())["status"] != "completed":
            raise ValueError("Only completed measured cohorts may be replayed")
        output = args.output / name
        command = ["python3", "-u", str(source / "predict_tp32_serving_validation.py"),
                   "--server-info", str(actual / "server-info.json"), "--plan", str(source / plan),
                   "--profile", str(profile), "--expert-workload", str(workload), "--output", str(output)]
        with (args.output / (name + ".log")).open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=5400)
        state["cohorts"].append({"name": name, "actual": str(actual), "predicted": str(output),
                                 "command": command, "returncode": result.returncode})
        save()
        print(json.dumps(state["cohorts"][-1]), flush=True)
    state["status"] = "completed" if all(row["returncode"] == 0 for row in state["cohorts"]) else "coverage_or_execution_failures"
    save()
    if state["status"] != "completed":
        raise RuntimeError("All declared cohorts attempted; failures retained")


if __name__ == "__main__":
    main()
