import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("runtime_layers", "isolated_kernels"), default="runtime_layers")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[4]
    config = here / "full-request-aggregate-8k1k-20261103.yaml"
    image_id = subprocess.check_output(["docker", "image", "inspect", "-f", "{{.Id}}", args.image], text=True).strip()
    args.output.mkdir(parents=True, exist_ok=False)
    if os.getuid() == 0:
        os.chown(args.output, int(os.environ.get("SUDO_UID", "0")), int(os.environ.get("SUDO_GID", "0")))
    manifest = {"scope": "Independent native scheduler replay; fixed observed KV capacity; no serving residual or measured forward injection",
                "image_id": image_id, "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "timing_scope": args.scope, "status": "running", "cases": []}
    target = args.output / "campaign.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    environment = os.environ.copy()
    environment.update(AISIMULATE_910C_IMAGE=image_id, AISIMULATE_910C_MODEL="/data_lib/data/models/glm-q",
                       AISIMULATE_910C_CPUSET="280-295", AISIMULATE_910C_CPUS="4", AISIMULATE_910C_MEMORY="8g")
    try:
        for concurrency in (1, 4, 8):
            destination = args.output / f"c{concurrency}"
            command = ["bash", str(repo / "scripts/run-ascend910c.sh"), "predict", "--stack", "engine",
                       "--config", str(config), "--capture-per-request", "--detail", "time,source",
                       "--output-dir", str(destination),
                       "--set", f"traffic.load.concurrency={concurrency}", "--set", f"traffic.stop.requests={concurrency}",
                       "--set", f"engine.estimator_config.op_level.ascend.decode_timing_scope={args.scope}"]
            case = {"concurrency": concurrency, "requests": concurrency, "command": command, "status": "running"}
            manifest["cases"].append(case)
            target.write_text(json.dumps(manifest, indent=2) + "\n")
            with (args.output / f"c{concurrency}.log").open("w") as stream:
                subprocess.run(command, env=environment, cwd=repo, stdout=stream, stderr=subprocess.STDOUT,
                               timeout=600, check=True)
            prediction = json.loads((destination / "prediction.json").read_text())
            if (prediction["completed_requests"] != concurrency
                    or prediction["total_input_tokens"] != concurrency * 8192
                    or prediction["total_output_tokens"] != concurrency * 1024
                    or any(prediction[key] for key in ("failed_requests", "rejected_requests", "canceled_requests"))):
                raise ValueError("Prediction request/token conservation failed")
            requests = [json.loads(line) for line in (destination / "requests.jsonl").read_text().splitlines()]
            if len(requests) != concurrency or any(
                row["terminal_status"] != "completed" or row["input_length"] != 8192 or row["output_length"] != 1024
                for row in requests
            ):
                raise ValueError("Prediction per-request conservation failed")
            case.update(status="completed", prediction=prediction)
            target.write_text(json.dumps(manifest, indent=2) + "\n")
            print(json.dumps({"concurrency": concurrency, "ttft_ms": prediction["mean_ttft_ms"],
                              "tpot_ms": prediction["mean_tpot_ms"]}), flush=True)
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        target.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
