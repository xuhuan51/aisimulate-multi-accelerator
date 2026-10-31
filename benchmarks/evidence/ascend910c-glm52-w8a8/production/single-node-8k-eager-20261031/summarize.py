import csv
import hashlib
import io
import json
import math
import subprocess
import tarfile
from collections import Counter
from pathlib import Path


DESTINATION = Path(__file__).resolve().parent
ROOT = DESTINATION.parents[4]
METRICS = (
    "num_requests", "completed_requests", "rejected_requests", "failed_requests", "canceled_requests",
    "total_input_tokens", "total_output_tokens", "committed_prefill_tokens", "duration_ms",
    "output_throughput_tok_s", "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms", "mean_e2e_latency_ms",
)
PARALLEL_FIELDS = ("tp", "attention_dp", "moe_tp", "moe_ep", "pp", "replicas")


def write(path, text):
    assert not path.exists(), path
    patch = "*** Begin Patch\n*** Add File: " + str(path) + "\n"
    patch += "".join("+" + line + "\n" for line in text.splitlines()) + "*** End Patch\n"
    subprocess.run(["apply_patch"], input=patch, text=True, check=True, stdout=subprocess.DEVNULL)


def brief(row, campaign, case):
    return {
        "campaign": campaign, "case": case, "candidate_id": row["candidate_id"],
        "concurrency": row["config"]["concurrency"],
        "parallelism": {field: row["config"][field] for field in PARALLEL_FIELDS},
        "metrics": {field: row["metrics"][field] for field in METRICS},
    }


def main():
    cases = []
    rows = []
    counts = Counter()
    selected = {}
    files = [Path(__file__), DESTINATION / "run_search.py", DESTINATION / "search.yaml"]
    winners = {}
    for campaign, directory in (
        ("requests_2c", "single-node-8k-eager-20261031"),
        ("requests_256", "single-node-8k-fixed256-eager-20261031"),
    ):
        output = ROOT / "aisimulate-output" / directory
        executions = json.loads((output / "execution-summary.json").read_text())
        assert len(executions) == (13 if campaign == "requests_2c" else 36)
        results = []
        for execution in executions:
            assert execution["returncode"] == 0 and execution["finite_search_exhausted"]
            case = execution["case"]
            report = json.loads((output / case / "result/recommendation.json").read_text())
            assert report["counts"]["evaluated"] == report["counts"]["feasible"] == len(report["candidates"]) == 6
            counts.update(report["counts"])
            for row in report["candidates"]:
                assert row["status"] == "feasible" and row["used_gpus"] == 16
                assert row["config"]["replicas"] == row["config"]["pp"] == 1
                assert row["config"]["tp"] * row["config"]["attention_dp"] == 16
                metrics = row["metrics"]
                assert metrics["num_requests"] == metrics["completed_requests"] == execution["requests"]
                assert metrics["total_input_tokens"] == metrics["committed_prefill_tokens"] == execution["requests"] * 8192
                assert metrics["total_output_tokens"] == execution["requests"] * 1024
                assert all(metrics[field] == 0 for field in ("rejected_requests", "failed_requests", "canceled_requests"))
                assert math.isclose(metrics["output_throughput_tok_s"], metrics["total_output_tokens"] * 1000 / metrics["duration_ms"], rel_tol=1e-12)
                rows.append({"campaign": campaign, "case": case, "candidate_id": row["candidate_id"], "concurrency": row["config"]["concurrency"], **{field: row["config"][field] for field in PARALLEL_FIELDS}, **{field: metrics[field] for field in METRICS}})
            winner = max(report["candidates"], key=lambda row: row["metrics"]["output_throughput_tok_s"])
            result = brief(winner, campaign, case)
            cases.append({**result, "counts": report["counts"], "finite_search_exhausted": True})
            results.append((result, winner))
            if (campaign == "requests_256" and execution["concurrency"] in (1, 8, 16, 20, 24, 128)) or (campaign == "requests_2c" and execution["concurrency"] == 20):
                name = f"best-{campaign}-c{execution['concurrency']}.yaml"
                selected[name] = (result, winner["prediction_config"])
            files.extend((output / case / "command.log", output / case / "execution.json", output / case / "result/recommendation.json"))
        maximum = max(result["metrics"]["output_throughput_tok_s"] for result, _ in results)
        tied = [result for result, _ in results if math.isclose(result["metrics"]["output_throughput_tok_s"], maximum, rel_tol=1e-10)]
        winner = min(tied, key=lambda result: (result["metrics"]["mean_ttft_ms"], result["concurrency"]))
        winners[campaign] = {**winner, "tied_throughput_concurrencies": [result["concurrency"] for result in tied]}
        files.append(output / "execution-summary.json")
    yaml_code = "import sys,json,yaml; configs=json.load(sys.stdin); result={}\nfor name,config in configs.items():\n config['engine'].pop('systems_paths',None)\n for worker in config['engine']['workers'].values(): worker.get('timing',{}).pop('systems_paths',None)\n result[name]=yaml.safe_dump(config,sort_keys=False)\njson.dump(result,sys.stdout)"
    serialized = subprocess.check_output(["docker", "exec", "-i", "aisim-node20-validation-20261027-v2", "python3", "-c", yaml_code], input=json.dumps({name: config for name, (_, config) in selected.items()}), text=True)
    for name, text in json.loads(serialized).items():
        write(DESTINATION / name, text)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    write(DESTINATION / "candidates.csv", stream.getvalue())
    archive = DESTINATION / "search-evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in sorted(files):
            bundle.add(path, arcname=str(path.relative_to(ROOT)))
    identity = json.loads((ROOT / "build/ascend910c-runtime/native-identity.json").read_text())
    report = {
        "date": "2026-10-31", "runtime_source": identity["source_revision"], "native_sha256": identity["sha256"],
        "model": "/data_lib/data/models/glm-q", "logical_devices": 16, "counts": dict(counts),
        "scope": "Single aggregated instance,PP1/CP1,eager,synchronous scheduler;fixed memory/scheduler settings;balanced experts;no latency SLA.",
        "cases": cases, "winners": winners,
        "selected_replays": [{**result, "prediction_config": name} for name, (result, _) in selected.items()],
        "archive": {"path": str(archive.relative_to(ROOT)), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "bytes": archive.stat().st_size},
        "serving_measured": False,
    }
    write(DESTINATION / "report.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps({"counts": dict(counts), "winners": winners}), flush=True)


if __name__ == "__main__":
    main()
