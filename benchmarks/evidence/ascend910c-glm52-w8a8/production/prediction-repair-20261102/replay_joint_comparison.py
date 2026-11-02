import concurrent.futures
import hashlib
import json
import os
import subprocess
from pathlib import Path

import replay_pd_campaign as replay


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
OUTPUT = REPO / "aisimulate-output/joint-serving-table-20261102"
IMAGE = "aisimulate-910c:forward-batch-repair-20261102"
CAMPAIGNS = {
    "pd-paired-1p1d": "pd-paired-latest-table-20261102",
    "pd-ingress-1p1d": "pd-ingress-forward-batch-20261102",
    "pd-cold-1p1d": "pd-cold-forward-batch-20261102",
    "pd-phases-1p1d": "pd-phase-forward-batch-20261102",
    "pd-allocator-1p1d": "pd-allocator-forward-batch-20261102",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE], text=True
    ).strip()
    configurations = json.loads(
        (ROOT.parent / "four-node-serving-20261101/deployment/tools/performance/comparison-configs.json").read_text()
    )["configurations"]
    configurations = {row["label"]: row for row in configurations}
    inputs = [
        ("four-node", path)
        for path in sorted((REPO / "aisimulate-output/four-node-active-rank-20261102").glob("*/c*-n*/observed-kv/execution.json"))
    ]
    for campaign, directory in CAMPAIGNS.items():
        inputs.extend((campaign, path) for path in sorted((REPO / "aisimulate-output" / directory).glob("*/execution.json")))
    if len(inputs) != 39:
        raise ValueError(f"Expected 23 four-node and 16 two-node completed cases, got {len(inputs)}")
    jobs = []
    for campaign, source in inputs:
        previous = json.loads(source.read_text())
        runtime_path = Path(previous["runtime_path"])
        measured_path = Path(previous["measured_path"])
        for path, key in [(runtime_path, "runtime_sha256"), (measured_path, "measured_sha256")]:
            if sha256(path) != previous[key]:
                raise ValueError(f"Changed measurement input: {path}")
        runtime = json.loads(runtime_path.read_text())
        if campaign == "four-node":
            configuration = configurations[previous["configuration"]]
            case = f"{previous['configuration']}-c{previous['concurrency']}-n{previous['requests']}"
        else:
            configuration = {
                "label": "paired-1p1d-tp16", "PREFILL_TP_SIZE": 16, "PREFILL_DP_SIZE": 4,
                "PREFILL_EP_SIZE": 16, "DECODE_TP_SIZE": 16, "DECODE_DP_SIZE": 4,
                "DECODE_EP_SIZE": 16, "DECODE_NNODES": 1, "DECODE_MAX_RUNNING_REQUESTS": 128,
                "PREFILL_NODES": "10.120.72.20:19000:19998", "DECODE_NODES": "10.120.72.25:19003",
            }
            case = previous["case"]
        replay.COMPARISON.validate_runtime(configuration, runtime)
        logits = replay.COMPARISON.audit_logits_layout(runtime, runtime_path)
        overrides, capacities = replay.COMPARISON.capacity_overrides(runtime)
        command = list(previous["command"])
        destination = OUTPUT / campaign / case
        command[command.index("--output-dir") + 1] = str(destination / "result")
        if "--detail" not in command:
            command.extend(["--detail", "time"])
        filtered = []
        position = 0
        while position < len(command):
            if command[position] == "--set" and ".kv_cache.capacity." in command[position + 1]:
                position += 2
            else:
                filtered.append(command[position])
                position += 1
        command = filtered
        for setting in overrides:
            command.extend(["--set", setting])
        template = Path(command[command.index("--config") + 1])
        if sha256(template) != previous["template_sha256"]:
            raise ValueError(f"Changed prediction template: {template}")
        measured = previous["measured"]
        count = previous["requests"]
        input_length = previous.get("input_length", 8192)
        output_length = previous.get("output_length", 1024)
        if measured["completed"] != count or any(measured["errors"]):
            raise ValueError(f"Invalid measured workload: {source}")
        if measured["input_lens"] != [input_length] * count or measured["output_lens"] != [output_length] * count:
            raise ValueError(f"Measured token lengths differ: {source}")
        jobs.append({
            **previous, "case": case, "campaign": campaign, "mode": "observed-kv",
            "configuration": configuration["label"], "configuration_spec": configuration,
            "input_length": input_length, "output_length": output_length,
            "command": command, "destination": str(destination), "image_id": image_id,
            "observed_blocks_per_rank": capacities, "logits_layout_audit": logits,
            "prior_execution": str(source), "prior_execution_sha256": sha256(source),
        })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(replay.run_job, jobs))
    rows = [
        {**row, "campaign": job["campaign"], "configuration": job["configuration"],
         "execution": str(Path(job["destination"]) / "execution.json")}
        for job, row in zip(jobs, results)
    ]
    (OUTPUT / "comparison.json").write_text(json.dumps({"image": IMAGE, "image_id": image_id, "rows": rows}, indent=2) + "\n")
    print(json.dumps({"completed": len(rows), "output": str(OUTPUT)}), flush=True)


if __name__ == "__main__":
    main()
