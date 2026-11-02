import json
import subprocess
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "native-attribution-node20-20261103"
CONTAINER = "aisim-native-attribution-20261103"


def main():
    destination = ROOT / "operator-trace-campaign.json"
    if destination.exists():
        raise FileExistsError(destination)
    name = "operator-trace-8k-c4"
    control = ROOT / "attribution-control.next.json"
    control.write_text(json.dumps({"enabled": False, "case": name}) + "\n")
    control.replace(ROOT / "attribution-control.json")
    command = ["docker", "exec", "-e", "PYTHONUNBUFFERED=1", CONTAINER,
               "python3", "-m", "sglang.benchmark.serving", "--backend", "sglang",
               "--host", "127.0.0.1", "--port", "18080", "--model", "/models/glm-q",
               "--served-model-name", "glm-5", "--dataset-name", "random-ids",
               "--num-prompts", "4", "--max-concurrency", "4", "--random-input-len", "8192",
               "--random-output-len", "160", "--random-range-ratio", "1.0", "--request-rate", "inf",
               "--warmup-requests", "2", "--seed", "910", "--tokenize-prompt", "--output-details",
               "--output-file", f"/campaign/{name}.jsonl"]
    state = {"command": command, "start_unix": time.time(), "status": "running",
             "scope": "Two profiled steady decode iterations; trace durations are diagnostic and instrumented, not serving accuracy ground truth"}
    destination.write_text(json.dumps(state, indent=2) + "\n")
    with (ROOT / f"{name}.log").open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        while process.poll() is None:
            if "profile_request" not in state:
                row = json.loads((ROOT / "paired-forward-rank0.jsonl").read_text().splitlines()[-1])
                if row["launch_unix"] > state["start_unix"] and any(request["prompt_tokens"] == 8192 and 60 <= request["output_tokens"] <= 120 for request in row["requests"]):
                    payload = {"output_dir": "/campaign/operator-trace", "num_steps": 2,
                               "activities": ["CPU", "GPU"], "with_stack": False,
                               "record_shapes": True, "profile_id": "decode-8k-c4",
                               "merge_profiles": False, "detailed_annotations": True}
                    state.update(profile_request=payload, profile_start_unix=time.time(), trigger=row)
                    destination.write_text(json.dumps(state, indent=2) + "\n")
                    request = urllib.request.Request("http://127.0.0.1:18080/start_profile", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(request, timeout=300) as response:
                        state["profile_response"] = {"status": response.status, "body": response.read().decode()}
                    state["profile_return_unix"] = time.time()
                    destination.write_text(json.dumps(state, indent=2) + "\n")
            if time.time() - state["start_unix"] > 900:
                process.terminate()
                raise TimeoutError(name)
            time.sleep(1)
    state.update(returncode=process.returncode, end_unix=time.time())
    state["status"] = "completed" if process.returncode == 0 and "profile_response" in state else "failed"
    destination.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps({key: state.get(key) for key in ("status", "returncode", "profile_response")}))
    if state["status"] != "completed":
        raise RuntimeError(name)


if __name__ == "__main__":
    main()
