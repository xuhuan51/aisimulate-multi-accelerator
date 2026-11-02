import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--allow-asymmetric", action="store_true")
    parser.add_argument("--manifest", default="chain-campaign.json")
    parser.add_argument("--prefix", default="quality")
    parser.add_argument("--controller", default="run_layer_chain_quality_campaign.py")
    args = parser.parse_args()
    root = args.campaign
    for attempt in range(7200):
        path = root / args.manifest
        if path.exists():
            try:
                manifest = json.loads(path.read_text())
            except json.JSONDecodeError:
                manifest = {}
            if manifest.get("status") == "completed":
                break
            if manifest.get("status") == "failed":
                raise RuntimeError("Initial campaign failed; quality retries cannot hide that failure")
        time.sleep(1)
    else:
        raise TimeoutError("Initial campaign did not finish")
    plan, rejected = [], []
    for case in manifest["cases"]:
        if not case["training"]:
            continue
        records = [json.loads((root / case["id"] / f"layer-replay-rank{rank}.json").read_text()) for rank in range(16)]
        bad = []
        for boundary_index, boundary in enumerate(records[0]["boundaries"]):
            maxima = [max(record["boundaries"][boundary_index]["samples"][sample]["device_ms"] for record in records)
                      for sample in range(3, 12)]
            coefficient = statistics.pstdev(maxima) / statistics.mean(maxima)
            if coefficient > 0.1:
                bad.append({"layer": boundary["layer"], "cv": coefficient, "samples_ms": maxima})
        if bad:
            rejected.append({"case": case["id"], "rejected": bad})
            retry = [args.prefix + "-retry-" + case["id"], case["input_length"], case["global_requests"], True]
            if case.get("routed_dp_counts") is not None:
                retry.append(case["routed_dp_counts"])
            plan.append(retry)
    (root / (args.prefix + "-rejected.json")).write_text(json.dumps(rejected, indent=2) + "\n")
    if not plan:
        print("No quality retry required", flush=True)
        return
    plan_path = root / (args.prefix + "-retry-plan.json")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    command = ["python3", "-u", str(root / args.controller), "--campaign", str(root),
               "--container", args.container, "--plan", str(plan_path), "--manifest", "chain-" + args.prefix + "-retry.json",
               "--output-length", "64"]
    if args.allow_asymmetric:
        command.append("--allow-asymmetric")
    print(json.dumps({"recollect_cases": len(plan), "command": command}), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
