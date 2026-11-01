import argparse
import json
from pathlib import Path

import yaml

from aisimulate.cli_args import _apply_overrides, _load_mapping
from aisimulate.config import CorePredictionConfig


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]


def export(command, destination):
    config_path = command[command.index("--config") + 1]
    overrides = [command[index + 1] for index, item in enumerate(command) if item == "--set"]
    raw = _load_mapping(config_path)
    _apply_overrides(raw, overrides, command="predict")
    validated = CorePredictionConfig.model_validate(raw)
    encoded = yaml.safe_dump(validated.model_dump(mode="json", exclude_none=True), sort_keys=False, allow_unicode=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(encoded)
    restored = CorePredictionConfig.model_validate(yaml.safe_load(destination.read_text()))
    if restored != validated:
        raise ValueError(f"Saved config changed: {destination}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, default=REPO / "aisimulate-output/four-node-request-timing-20261102/comparison-jobs.json")
    parser.add_argument("--paired-execution", type=Path, default=REPO / "aisimulate-output/pd-ingress-closed-loop-20261102/validation-c4-8k1k/execution.json")
    parser.add_argument("--output", type=Path, default=ROOT / "request-timing-configs")
    args = parser.parse_args()
    jobs = json.loads(args.jobs.read_text())
    for job in jobs:
        if job["mode"] == "stock" and job["concurrency"] == 8:
            export(job["command"], args.output / f"{job['configuration']}-8k1k-c8.yaml")
    execution = json.loads(args.paired_execution.read_text())
    export(execution["command"], args.output / "paired-1p1d-8k1k-c4.yaml")


if __name__ == "__main__":
    main()
