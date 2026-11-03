import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path


def latest_row(path):
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 131072))
        data = stream.read()
    lines = data[:data.rfind(b"\n")].splitlines()
    return json.loads(lines[-1]) if lines else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--wait-collection", type=Path, required=True)
    parser.add_argument("--ssh-socket", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    root = args.campaign.resolve()
    remote = json.loads((root / "node1/launch.json").read_text())["remote_root"]
    ssh = ["ssh", "-S", args.ssh_socket, "-o", "BatchMode=yes", "youxingye@npu1-25"]
    state = {"status": "waiting", "scope": "Two steady prefill iterations with real TP32/ADP32/EP32 operator tracing; diagnostic timing only, excluded from serving accuracy and layer training", "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    destination = args.output / "campaign.json"

    def save():
        destination.write_text(json.dumps(state, indent=2) + "\n")

    save()
    process = None
    try:
        deadline = time.monotonic() + 7200
        while time.monotonic() < deadline:
            prior = json.loads(args.wait_collection.read_text())
            if prior["status"] == "completed" and prior.get("probes_disarmed"):
                break
            if prior["status"] == "failed":
                raise RuntimeError("Preceding collection failed")
            time.sleep(5)
        else:
            raise TimeoutError("Preceding collection did not finish")
        if (root / "node0/layer-probe-control.json").exists():
            raise RuntimeError("Node0 layer replay probe is still armed")
        subprocess.run(ssh + [shlex.join(["test", "!", "-e", remote + "/layer-probe-control.json"])], check=True, timeout=15)
        name = args.output.name
        mounted = root / "node0" / name
        mounted.mkdir(exist_ok=False)
        command = ["docker", "exec", "-e", "AISIM_ATTENTION_DP=32", "-e", "AISIM_ROUTED_COUNTS=" + ",".join(["1"] * 32),
                   "aisim-layer-tp32-node0-20261104", "python3", "-u", "/campaign/run_routed_layer_benchmark.py",
                   "--backend", "sglang", "--host", "127.0.0.1", "--port", "19180", "--model", "/models/glm-q",
                   "--served-model-name", "glm-5", "--dataset-name", "random-ids", "--num-prompts", "32",
                   "--max-concurrency", "32", "--random-input-len", "16384", "--random-output-len", "64",
                   "--random-range-ratio", "1.0", "--request-rate", "inf", "--warmup-requests", "0",
                   "--seed", "2810", "--tokenize-prompt", "--output-details", "--output-file", f"/campaign/{name}/requests.jsonl"]
        warmup = command[:]
        for flag, value in (("--num-prompts", "2"), ("--random-output-len", "32"),
                            ("--output-file", f"/campaign/{name}/warmup.jsonl")):
            warmup[warmup.index(flag) + 1] = value
        state.update(status="warming", warmup_command=warmup)
        save()
        with (args.output / "warmup.log").open("w") as stream:
            subprocess.run(warmup, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600)
        shutil.copy2(mounted / "warmup.jsonl", args.output / "warmup.jsonl")
        warm = json.loads((args.output / "warmup.jsonl").read_text().splitlines()[-1])
        if warm["completed"] != 2 or any(warm["errors"]):
            raise ValueError("The two unprofiled warmup requests did not complete")
        state.update(status="running", command=command, start_unix=time.time(),
                     preceding_collection_sha256=hashlib.sha256(args.wait_collection.read_bytes()).hexdigest())
        save()
        with (args.output / "benchmark.log").open("w") as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if "profile_request" not in state:
                    row = latest_row(root / "node0/paired-forward-rank0.jsonl")
                    scheduled = row["scheduled_requests"] if row else {}
                    formal = row and any(request["output_tokens"] == 0 and request["prompt_tokens"] == 16384 for request in row["requests"])
                    if formal and row["launch_unix"] > state["start_unix"] and scheduled["sum_prefill_tokens"] > 0 and scheduled["sum_prefill_kv_tokens"] >= 3072:
                        payload = {"output_dir": f"/campaign/{name}/operator-trace", "num_steps": 2,
                                   "activities": ["CPU", "GPU"], "with_stack": False, "record_shapes": True,
                                   "profile_id": name, "merge_profiles": False, "detailed_annotations": True}
                        state.update(profile_request=payload, profile_start_unix=time.time(), trigger=row)
                        save()
                        request = urllib.request.Request("http://127.0.0.1:19180/start_profile", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
                        with urllib.request.urlopen(request, timeout=600) as response:
                            state["profile_response"] = {"status": response.status, "body": response.read().decode()}
                        state["profile_return_unix"] = time.time()
                        save()
                if time.time() - state["start_unix"] > 1200:
                    raise TimeoutError("Profiled diagnostic workload did not finish")
                time.sleep(0.25)
        state.update(returncode=process.returncode, end_unix=time.time())
        if process.returncode or "profile_response" not in state:
            raise RuntimeError("Profile trigger or diagnostic benchmark failed")
        shutil.copy2(mounted / "requests.jsonl", args.output / "requests.jsonl")
        state.update(status="completed", local_trace=str(mounted / "operator-trace"), remote_trace=remote + "/" + name + "/operator-trace")
    except Exception as error:
        state.update(status="failed", error=repr(error))
        if process is not None and process.poll() is None:
            process.terminate()
        raise
    finally:
        if state["status"] == "failed" and "profile_request" in state:
            try:
                request = urllib.request.Request("http://127.0.0.1:19180/stop_profile", data=b"{}", headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    state["cleanup_profile_response"] = {"status": response.status, "body": response.read().decode()}
            except (OSError, ValueError) as error:
                state["cleanup_profile_error"] = repr(error)
        save()


if __name__ == "__main__":
    main()
