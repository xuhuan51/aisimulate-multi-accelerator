import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--container", required=True)
    args = parser.parse_args()
    root = args.campaign
    for attempt in range(3600):
        path = root / "chain-quality-retry.json"
        try:
            status = json.loads(path.read_text())["status"] if path.exists() else "waiting"
        except json.JSONDecodeError:
            status = "waiting"
        if status == "completed":
            break
        if status == "failed":
            raise RuntimeError("Previous quality campaign failed")
        time.sleep(1)
    else:
        raise TimeoutError("Previous quality campaign is still running")
    training, heldout = [], []
    for active in (1, 2, 3):
        for requests in (2, 3):
            counts = [requests] * active + [0] * (4 - active)
            for length in (512, 10240):
                training.append([f"routed-a{active}-b{requests}-k{length}", length, sum(counts), True, counts])
            heldout.append([f"holdout-routed-a{active}-b{requests}-8192", 8192, sum(counts), False, counts])
    for counts in ([2, 1, 1, 1], [2, 2, 1, 1], [3, 2, 2, 2], [3, 3, 2, 2], [3, 2, 1, 0], [2, 1, 0, 0]):
        heldout.append(["holdout-skew-" + "-".join(map(str, counts)), 8192, sum(counts), False, counts])
    for name, plan in (("routed-plan.json", training), ("routed-holdout-plan.json", heldout)):
        (root / name).write_text(json.dumps(plan, indent=2) + "\n")
    common = ["python3", "-u", str(root / "run_layer_routed_campaign.py"), "--campaign", str(root),
              "--container", args.container, "--allow-asymmetric"]
    training_command = [*common, "--plan", str(root / "routed-plan.json"), "--manifest", "chain-routed-campaign.json",
                        "--output-length", "64"]
    print(json.dumps({"training_command": training_command}), flush=True)
    subprocess.run(training_command, check=True)
    quality_command = ["python3", "-u", str(root / "queue_layer_routed_quality_retry.py"), "--campaign", str(root),
                       "--container", args.container, "--allow-asymmetric", "--manifest", "chain-routed-campaign.json",
                       "--prefix", "routed-quality", "--controller", "run_layer_routed_campaign.py"]
    subprocess.run(quality_command, check=True)
    subprocess.run([*common, "--plan", str(root / "routed-holdout-plan.json"), "--manifest", "chain-routed-holdout.json",
                    "--output-length", "128"], check=True)


if __name__ == "__main__":
    main()
