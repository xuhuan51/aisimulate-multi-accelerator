import argparse
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[4]
    args.output = args.output.resolve()
    args.output.mkdir(exist_ok=False, parents=True)
    runtime_uid = int(os.environ.get("SUDO_UID", os.getuid()))
    runtime_gid = int(os.environ.get("SUDO_GID", os.getgid()))

    def own(path):
        if os.getuid() == 0:
            os.chown(path, runtime_uid, runtime_gid)

    own(args.output)
    source = here / "full-request-node25-20261103"
    diagnosis = json.loads((here / "full-request-mixed-diagnosis-20261103.json").read_text())
    arrivals = {row["rid"]: row for path in source.glob("tokenizer-ingress-pid*.jsonl")
                for row in (json.loads(line) for line in path.read_text().splitlines())}
    image_id = subprocess.check_output(["docker", "image", "inspect", "-f", "{{.Id}}", args.image], text=True).strip()
    baseline = here / "full-request-aggregate-8k1k-20261103.yaml"
    base = json.loads(subprocess.check_output([
        "docker", "run", "--rm", "--network", "none", "-i", "--entrypoint", "python3", image_id,
        "-c", "import json,sys,yaml; print(json.dumps(yaml.safe_load(sys.stdin)))",
    ], input=baseline.read_text(), text=True))
    environment = os.environ.copy()
    environment.update(AISIMULATE_910C_IMAGE=image_id, AISIMULATE_910C_MODEL="/data_lib/data/models/glm-q",
                       AISIMULATE_910C_CPUSET="280-295", AISIMULATE_910C_CPUS="4", AISIMULATE_910C_MEMORY="8g")
    manifest = {"status": "running", "image_id": image_id, "cases": [],
                "scope": "Conditional native scheduler replay of externally measured tokenizer-send timestamps. No scheduler receive times, observed batch durations, route assignments or serving residuals are injected. Not an unconditional burst forecast or PD validation.",
                "baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest()}
    target = args.output / "campaign.json"

    def save():
        target.write_text(json.dumps(manifest, indent=2) + "\n")
        own(target)

    save()
    try:
        for original in diagnosis["cases"]:
            concurrency = original["concurrency"]
            selected = [arrivals[row["rid"]] for row in original["request_arrivals"]]
            selected.sort(key=lambda row: row["send_call_start_monotonic"])
            origin = selected[0]["send_call_start_monotonic"]
            trace = args.output / f"c{concurrency}-ingress.jsonl"
            rows = [{"input_length": 8192, "output_length": 1024,
                     "hash_ids": list(range(index * 16, (index + 1) * 16)),
                     "timestamp": 1000 * (row["send_call_start_monotonic"] - origin)}
                    for index, row in enumerate(selected)]
            if len(rows) != concurrency or any(row["prompt_tokens"] != 8192 for row in selected):
                raise ValueError("External input trace conservation failed")
            trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
            own(trace)
            config = copy.deepcopy(base)
            config["traffic"] = {"source": {"type": "trace", "format": "mooncake", "block_size": 512,
                                             "paths": [str(trace)]},
                                 "load": {"type": "trace_timestamps"}, "stop": {}}
            config_path = args.output / f"c{concurrency}.json"
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            own(config_path)
            destination = args.output / f"c{concurrency}"
            command = ["bash", str(repo / "scripts/run-ascend910c.sh"), "predict", "--stack", "engine",
                       "--config", str(config_path), "--capture-per-request", "--detail", "source",
                       "--output-dir", str(destination)]
            case = {"concurrency": concurrency, "arrival_offsets_ms": [row["timestamp"] for row in rows],
                    "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                    "input_records": selected, "command": command, "status": "running"}
            manifest["cases"].append(case)
            save()
            with (args.output / f"c{concurrency}.log").open("w") as stream:
                subprocess.run(command, env=environment, cwd=repo, stdout=stream, stderr=subprocess.STDOUT,
                               timeout=600, check=True)
            result = json.loads((destination / "prediction.json").read_text())
            if (result["completed_requests"] != concurrency or result["total_input_tokens"] != concurrency * 8192
                    or result["total_output_tokens"] != concurrency * 1024
                    or any(result[key] for key in ("failed_requests", "rejected_requests", "canceled_requests"))):
                raise ValueError("Replay request/token conservation failed")
            actual = json.loads((source / "full-request-validation" / f"8192-1024-c{concurrency}" / "requests.jsonl").read_text().splitlines()[-1])
            compared = {}
            for predicted_name, measured_name in (("mean_ttft_ms", "mean_ttft_ms"), ("mean_tpot_ms", "mean_tpot_ms"),
                                                  ("input_throughput_tok_s", "input_throughput"),
                                                  ("output_throughput_tok_s", "output_throughput")):
                compared[predicted_name] = {"measured": actual[measured_name], "predicted": result[predicted_name],
                                           "error_percent": 100 * (result[predicted_name] / actual[measured_name] - 1)}
            case.update(status="completed", prediction=result, comparison=compared)
            save()
            print(json.dumps({"concurrency": concurrency, "comparison": compared}), flush=True)
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
