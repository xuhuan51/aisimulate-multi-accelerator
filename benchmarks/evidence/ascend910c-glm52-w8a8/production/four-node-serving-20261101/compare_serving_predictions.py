import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import pathlib
import subprocess
import time


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[4]
TEMPLATES = HERE.parent / "four-node-serving-plan-20261031"
METRIC_PAIRS = {
    "ttft_ms": ("mean_ttft_ms", "mean_ttft_ms"),
    "tpot_ms": ("mean_tpot_ms", "mean_tpot_ms"),
    "input_tok_s": ("input_throughput_tok_s", "input_throughput"),
    "output_tok_s": ("output_throughput_tok_s", "output_throughput"),
}


def validate_runtime(configuration, runtime):
    observed = {"prefill": set(), "decode": set()}
    for node in runtime["nodes"]:
        effective = node.get("effective")
        if effective is None:
            if not node.get("metrics_only_node"):
                raise ValueError(f"Unexplained missing server configuration: {node}")
            continue
        role = effective["disaggregation_mode"]
        prefix = role.upper()
        expected = {
            "status": "ready", "version": "0.5.22", "model_path": "/models/glm-q",
            "tp_size": configuration[f"{prefix}_TP_SIZE"],
            "attn_dp_size": configuration[f"{prefix}_DP_SIZE"],
            "ep_size": configuration[f"{prefix}_EP_SIZE"],
            "nnodes": 1 if role == "prefill" else configuration["DECODE_NNODES"],
            "pp_size": 1, "attn_cp_size": 1, "page_size": 128,
            "context_length": 180000, "disable_overlap_schedule": True,
            "disable_radix_cache": True, "speculative_algorithm": None,
            "mem_fraction_static": 0.88 if role == "prefill" else 0.90,
            "max_running_requests": 128 if role == "prefill" else configuration["DECODE_MAX_RUNNING_REQUESTS"],
            "quantization": "modelslim", "kv_cache_dtype": "auto",
            "disaggregation_transfer_backend": "ascend", "moe_a2a_backend": "deepep",
            "deepep_mode": "normal" if role == "prefill" else "low_latency", "disable_shared_experts_fusion": True,
            "load_balance_method": "round_robin", "schedule_policy": "fcfs",
        }
        if role == "prefill":
            expected.update({"chunked_prefill_size": 16384 // configuration["PREFILL_DP_SIZE"], "max_prefill_tokens": 32768})
        else:
            expected["num_reserved_decode_tokens"] = 512
        mismatches = {key: {"expected": value, "actual": effective.get(key)} for key, value in expected.items() if effective.get(key) != value}
        if mismatches:
            raise ValueError(f"Runtime differs from candidate {configuration['label']} at {node['endpoint']}: {mismatches}")
        for phase in ("prefill", "decode"):
            if effective["cuda_graph_config"][phase]["backend"] != "disabled":
                raise ValueError(f"Graph enabled at {node['endpoint']}")
        observed[role].add(node["replica"])
    expected_counts = {
        "prefill": len(configuration["PREFILL_NODES"].split()),
        "decode": len(configuration["DECODE_NODES"].split()) // configuration["DECODE_NNODES"],
    }
    if {role: len(replicas) for role, replicas in observed.items()} != expected_counts:
        raise ValueError(f"Replica count mismatch: {observed} versus {expected_counts}")


def capacity_overrides(runtime):
    capacities = {"prefill": [], "decode": []}
    for node in runtime["nodes"]:
        effective = node.get("effective")
        if effective is None:
            continue
        role = effective["disaggregation_mode"]
        page = effective["page_size"]
        tokens = effective["max_total_num_tokens"]
        if page != 128 or tokens < page:
            raise ValueError("Observed KV pool does not match the frozen comparison")
        capacities[role].append(tokens // page)
    overrides = []
    for role, counts in capacities.items():
        if not counts:
            raise ValueError(f"No observed capacity for {role}")
        key = f"engine.workers.{role}.kv_cache.capacity"
        overrides.extend([f"{key}.type=fixed", f"{key}.memory_fraction=null", f"{key}.blocks={min(counts)}"])
    return overrides, capacities


def make_jobs(data_root, output_root):
    plan = json.loads((data_root / "comparison-plan.json").read_text())
    image = os.environ.get("AISIMULATE_910C_IMAGE", "aisimulate-910c:glm52-w8a8")
    image_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True).strip()
    jobs = []
    for configuration in plan["configurations"]:
        label = configuration["label"]
        directory = data_root / (label + "-20261101")
        template = TEMPLATES / (("best-2p1d-dtp32" if label.startswith("recommended") else label) + ".yaml")
        for concurrency in plan["workload"]["concurrencies"]:
            requests = concurrency * plan["workload"]["prompt_multiplier"]
            point = directory / f"c{concurrency}-n{requests}"
            if not (point / ".complete").exists():
                continue
            results = sorted(point.glob("bench_*.jsonl"))
            actual = json.loads(results[-1].read_text().splitlines()[-1])
            if actual["completed"] != requests or actual["input_lens"] != [8192] * requests or actual["output_lens"] != [1024] * requests:
                raise ValueError(f"Measured workload mismatch: {point}")
            if actual["max_concurrency"] != concurrency or any(actual["errors"]):
                raise ValueError(f"Measured concurrency or request errors mismatch: {point}")
            runtime_path = point / "runtime-config.json"
            if not runtime_path.exists():
                runtime_path = directory / "runtime-config.json"
            runtime = json.loads(runtime_path.read_text())
            validate_runtime(configuration, runtime)
            fixed, capacities = capacity_overrides(runtime)
            for mode in ("stock", "observed-kv"):
                destination = output_root / label / point.name / mode
                command = [
                    "bash", str(ROOT / "scripts/run-ascend910c.sh"), "predict", "--stack", "engine",
                    "--config", str(template), "--capture-per-request", "--output-dir", str(destination / "result"),
                    "--set", f"traffic.stop.requests={requests}", "--set", f"traffic.load.concurrency={concurrency}",
                ]
                if mode == "observed-kv":
                    for setting in fixed:
                        command.extend(["--set", setting])
                jobs.append({
                    "configuration": label, "concurrency": concurrency, "requests": requests,
                    "mode": mode, "command": command, "destination": str(destination),
                    "measured": actual, "measured_path": str(results[-1]),
                    "measured_sha256": hashlib.sha256(results[-1].read_bytes()).hexdigest(),
                    "runtime_path": str(runtime_path), "observed_blocks_per_rank": capacities,
                    "runtime_sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
                    "template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
                    "simulator_image_id": image_id,
                    "capacity_policy": "minimum across replicas of each role; diagnostic only",
                })
    return jobs


def run_job(job):
    destination = pathlib.Path(job["destination"])
    destination.mkdir(parents=True, exist_ok=True)
    if os.geteuid() == 0:
        os.chown(destination, int(os.environ.get("SUDO_UID") or os.getuid()), int(os.environ.get("SUDO_GID") or os.getgid()))
    execution = destination / "execution.json"
    if execution.exists():
        saved = json.loads(execution.read_text())
        identity = ("command", "measured_sha256", "runtime_sha256", "template_sha256", "simulator_image_id")
        if any(saved.get(key) != job[key] for key in identity):
            raise ValueError(f"Stale comparison result at {destination}")
        return saved["comparison"]
    previous = [destination / name for name in ("command.log", "result") if (destination / name).exists()]
    if previous:
        attempt = destination / "failed-attempts" / str(time.time_ns())
        attempt.mkdir(parents=True)
        for path in previous:
            path.rename(attempt / path.name)
    environment = os.environ.copy()
    environment.update({"AISIMULATE_910C_MODEL": "/data_lib/data/models/glm-q", "AISIMULATE_910C_CPUSET": "280-295", "AISIMULATE_910C_CPUS": "4", "AISIMULATE_910C_MEMORY": "8g"})
    environment["AISIMULATE_910C_IMAGE"] = job["simulator_image_id"]
    with (destination / "command.log").open("w") as log:
        result = subprocess.run(job["command"], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    if result.returncode:
        raise RuntimeError(f"Prediction failed: {destination / 'command.log'}")
    prediction = json.loads((destination / "result/prediction.json").read_text())
    if prediction["completed_requests"] != job["requests"] or prediction["total_input_tokens"] != job["requests"] * 8192 or prediction["total_output_tokens"] != job["requests"] * 1024:
        raise ValueError(f"Prediction workload mismatch: {destination}")
    if any(prediction[field] for field in ("failed_requests", "rejected_requests", "canceled_requests")):
        raise ValueError(f"Prediction contains failed requests: {destination}")
    per_request = [json.loads(line) for line in (destination / "result/requests.jsonl").read_text().splitlines()]
    if len(per_request) != job["requests"] or any(row["input_length"] != 8192 or row["output_length"] != 1024 or row["terminal_status"] != "completed" for row in per_request):
        raise ValueError(f"Prediction per-request validation failed: {destination}")
    comparison = {key: job[key] for key in ("configuration", "concurrency", "requests", "mode")}
    for name, (predicted_key, measured_key) in METRIC_PAIRS.items():
        predicted, measured = prediction[predicted_key], job["measured"][measured_key]
        comparison[f"predicted_{name}"] = predicted
        comparison[f"measured_{name}"] = measured
        comparison[f"error_percent_{name}"] = (predicted / measured - 1) * 100
    execution.write_text(json.dumps({**job, "comparison": comparison}, indent=2) + "\n")
    print(json.dumps(comparison), flush=True)
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "aisimulate-output/four-node-serving-matched-20261101")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    state = json.loads((args.measurement_root / "matrix-progress.json").read_text())
    if not args.prepare_only and state["status"] != "measured":
        raise RuntimeError("Wait until the serving matrix finishes; avoid CPU contention during measured runs")
    jobs = make_jobs(args.measurement_root, args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = [{key: value for key, value in job.items() if key != "measured"} for job in jobs]
    (args.output / "comparison-jobs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        print(f"Prepared {len(jobs)} comparison jobs; no predictions executed")
        return
    if len(jobs) != 48:
        raise ValueError(f"Expected 24 measured points and 48 paired diagnostics; got {len(jobs)}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(run_job, jobs))
    with (args.output / "comparison.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "comparison.json").write_text(json.dumps({"error_formula": "100 * (predicted / measured - 1)", "scope": "fixed candidate matrix, not a new global optimality search; no latency fitting; no serving accuracy acceptance", "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
