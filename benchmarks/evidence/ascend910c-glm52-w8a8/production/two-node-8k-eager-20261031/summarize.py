import collections
import csv
import hashlib
import io
import json
import math
import subprocess
import tarfile
from pathlib import Path


DESTINATION = Path(__file__).resolve().parent
ROOT = DESTINATION.parents[4]
OUTPUT = ROOT / "aisimulate-output/two-node-8k-eager-20261031"
PARALLEL_FIELDS = ("tp", "attention_dp", "moe_tp", "moe_ep", "pp", "replicas")
METRICS = (
    "num_requests", "completed_requests", "rejected_requests", "failed_requests", "canceled_requests",
    "total_input_tokens", "total_output_tokens", "committed_prefill_tokens",
    "output_throughput_tok_s", "request_throughput_rps", "duration_ms",
    "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms", "mean_e2e_latency_ms",
)


def write(path, text):
    assert not path.exists(), path
    patch = "*** Begin Patch\n*** Add File: " + str(path) + "\n"
    patch += "".join("+" + line + "\n" for line in text.splitlines()) + "*** End Patch\n"
    subprocess.run(["apply_patch"], input=patch, text=True, check=True, stdout=subprocess.DEVNULL)


def layout(row):
    if row["config"]["deployment_mode"] == "disagg":
        return "pd-1p1d"
    return "agg-2x1node" if row["config"]["replicas"] == 2 else "agg-1x2node"


def parallelism(row):
    config = row["config"]
    roles = ("prefill", "decode") if config["deployment_mode"] == "disagg" else ("aggregated",)
    result = {}
    for role in roles:
        prefix = "" if role == "aggregated" else role + "_"
        values = {field: config[prefix + field] for field in PARALLEL_FIELDS}
        values["global_tp"] = values["tp"] * values["attention_dp"]
        values["nodes_per_replica"] = values["global_tp"] * values["pp"] // 16
        result[role] = values
    return result


def brief(row, case):
    return {
        "case": case,
        "candidate_id": row["candidate_id"],
        "layout": layout(row),
        "concurrency": row["config"]["concurrency"],
        "parallelism": parallelism(row),
        "metrics": {field: row["metrics"][field] for field in METRICS},
    }


def pd_coordinates(row):
    return tuple(row["config"][role + "_" + field] for role in ("prefill", "decode") for field in PARALLEL_FIELDS)


def portable_configs(configurations):
    script = """import json,sys,yaml
configs=json.load(sys.stdin)
result={}
for name,config in configs.items():
    config['engine']['model']='/data_lib/data/models/glm-q'
    config['engine'].pop('systems_paths',None)
    for worker in config['engine']['workers'].values():
        worker.get('timing',{}).pop('systems_paths',None)
    result[name]=yaml.safe_dump(config,sort_keys=False)
json.dump(result,sys.stdout)
"""
    serialized = subprocess.check_output(
        ["docker", "exec", "-i", "aisim-node20-validation-20261027-v2", "python3", "-c", script],
        input=json.dumps(configurations), text=True,
    )
    return json.loads(serialized)


def main():
    executions = json.loads((OUTPUT / "execution-summary.json").read_text())
    assert len(executions) == 16
    cases = []
    winners = []
    reference = []
    csv_rows = []
    failures = []
    counts = collections.Counter()
    configurations = {}
    files = [OUTPUT / "execution-summary.json", Path(__file__), DESTINATION / "run_search.py", DESTINATION / "search-pd.yaml", DESTINATION / "search-agg.yaml"]
    cross_checked = 0
    for execution in executions:
        name = execution["case"]
        concurrency = execution["concurrency"]
        requests = execution["requests"]
        assert execution["returncode"] == 0 and execution["finite_search_exhausted"], execution
        directory = OUTPUT / name
        report = json.loads((directory / "result/recommendation.json").read_text())
        assert report["counts"]["evaluated"] == len(report["candidates"])
        counts.update(report["counts"])
        feasible = []
        for row in report["candidates"]:
            assert row["used_gpus"] == 32
            topology = parallelism(row)
            normalized = {role + "_" + field: values[field] for role, values in topology.items() for field in PARALLEL_FIELDS}
            record = {"case": name, "concurrency": concurrency, "requests": requests, "candidate_id": row["candidate_id"], "layout": layout(row), "status": row["status"], "reason": row["reason"], "used_devices": row["used_gpus"], **normalized}
            if row["status"] != "feasible":
                failures.append(record)
            else:
                metrics = row["metrics"]
                assert metrics["num_requests"] == metrics["completed_requests"] == requests
                assert metrics["total_input_tokens"] == requests * 8192
                assert metrics["committed_prefill_tokens"] >= requests * 8192
                assert metrics["total_output_tokens"] == requests * 1024
                assert all(metrics[field] == 0 for field in ("rejected_requests", "failed_requests", "canceled_requests"))
                assert math.isclose(metrics["output_throughput_tok_s"], metrics["total_output_tokens"] * 1000 / metrics["duration_ms"], rel_tol=1e-12)
                record.update({field: metrics[field] for field in METRICS})
                feasible.append(row)
            csv_rows.append(record)
        selected = []
        for category in sorted({layout(row) for row in feasible}):
            group = [row for row in feasible if layout(row) == category]
            winner = max(group, key=lambda row: row["metrics"]["output_throughput_tok_s"])
            entry = brief(winner, name)
            entry["feasible_configurations"] = len(group)
            filename = f"best-{category}-c{concurrency}.yaml"
            entry["prediction_config"] = filename
            configurations[filename] = winner["prediction_config"]
            winners.append(entry)
            selected.append(entry)
        if execution["mode"] == "pd":
            baseline = [row for row in feasible if all(row["config"][role + "_" + field] == expected for role in ("prefill", "decode") for field, expected in (("tp", 8), ("attention_dp", 2), ("moe_tp", 1), ("moe_ep", 16), ("replicas", 1)))]
            assert len(baseline) == 1
            reference.append(brief(baseline[0], name))
            if concurrency in (4, 8, 16, 32):
                previous = ROOT / f"aisimulate-output/pd-restore-four-node-20261031/isl8192-osl1024-c{concurrency}-n{requests}/result/recommendation.json"
                original = json.loads(previous.read_text())
                by_coordinates = {pd_coordinates(row): row for row in original["candidates"] if row["used_gpus"] == 32}
                assert len(by_coordinates) == len(feasible)
                for row in feasible:
                    earlier = by_coordinates[pd_coordinates(row)]
                    assert all(math.isclose(row["metrics"][field], earlier["metrics"][field], rel_tol=1e-12, abs_tol=1e-9) for field in METRICS)
                    cross_checked += 1
        cases.append({"case": name, "concurrency": concurrency, "requests": requests, "mode": execution["mode"], "counts": report["counts"], "winners": selected, "elapsed_seconds": execution["elapsed_seconds"], "finite_search_exhausted": execution["finite_search_exhausted"]})
        files.extend((directory / "command.log", directory / "execution.json", directory / "result/recommendation.json"))
    for name, text in portable_configs(configurations).items():
        write(DESTINATION / name, text)
    columns = ["case", "concurrency", "requests", "candidate_id", "layout", "status", "reason", "used_devices"]
    columns.extend(role + "_" + field for role in ("aggregated", "prefill", "decode") for field in PARALLEL_FIELDS)
    columns.extend(METRICS)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(csv_rows)
    write(DESTINATION / "candidates.csv", stream.getvalue())
    archive = DESTINATION / "search-evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in sorted(files):
            bundle.add(path, arcname=str(path.relative_to(ROOT)))
    identity = json.loads((ROOT / "build/ascend910c-runtime/native-identity.json").read_text())
    image_id = subprocess.check_output(["docker", "image", "inspect", "aisimulate-910c:glm52-w8a8", "--format", "{{.Id}}"], text=True).strip()
    report = {
        "date": "2026-10-31",
        "runtime_source": identity["source_revision"],
        "native_sha256": identity["sha256"],
        "image_id": image_id,
        "model": "/data_lib/data/models/glm-q",
        "model_metadata_sha256": {name: hashlib.sha256((Path('/data_lib/data/models/glm-q') / name).read_bytes()).hexdigest() for name in ("config.json", "quant_model_description.json")},
        "scope": "Exactly32 logical devices,16 per node;8192 input/1024 output;C1..128 powers of2,N=2C;no warmup or SLA;CPU-only,eager,synchronous scheduler.",
        "counts": dict(counts),
        "cases": cases,
        "winners_by_layout": winners,
        "reference_pd_tp16_adp2_ep16": reference,
        "failures": failures,
        "matches_existing_four_node_search_subset": cross_checked,
        "archive": {"path": str(archive.relative_to(ROOT)), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "bytes": archive.stat().st_size},
        "limitations": [
            "Predictions have not been compared with matching eager two-node serving measurements.",
            "FiniteN=2C cold runs are not steady-state throughput or a latency-SLA search.",
            "PP1/CP1;no eager DCP;fixed scheduler and memory controls.",
            "Host KV backup/restore time, CPU overhead and cross-request sender contention remain uncalibrated.",
            "balanced expert routing and anchored shape extrapolation remain modeling assumptions.",
        ],
    }
    write(DESTINATION / "report.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps({"counts": dict(counts), "winners": len(winners), "cross_checked": cross_checked, "failures": len(failures), "archive_bytes": archive.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
