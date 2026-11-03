import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--ssh-socket", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", default="layer-campaign.json")
    args = parser.parse_args()
    root = args.campaign.resolve()
    destination = root / args.manifest
    if destination.exists():
        raise FileExistsError(destination)
    ssh = ["ssh", "-S", args.ssh_socket, "-o", "BatchMode=yes", "youxingye@npu1-25"]
    remote_root = json.loads((root / "node1/launch.json").read_text())["remote_root"]
    plan = json.loads(args.plan.read_text())
    manifest = {"status": "waiting", "cases": [], "plan": plan,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scope": "Real TP32 layer invocation and numeric validation; client latency includes replay pauses and is not serving ground truth"}

    def save():
        destination.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    try:
        for attempt in range(1200):
            state = subprocess.check_output(["docker", "inspect", "-f", "{{.State.Status}}", "aisim-layer-tp32-node0-20261104"], text=True).strip()
            if state != "running":
                raise RuntimeError("Node0 service is not running: " + state)
            try:
                with urllib.request.urlopen("http://127.0.0.1:19180/get_server_info", timeout=2) as response:
                    info = json.load(response)
                break
            except (OSError, ValueError):
                time.sleep(1)
        else:
            raise TimeoutError("Two-node service did not become ready")
        if (info["tp_size"], info["attn_dp_size"], info["ep_size"]) != (32, 32, 32):
            raise ValueError("Unexpected topology")
        if not info["disable_overlap_schedule"] or info["speculative_algorithm"] is not None:
            raise ValueError("Unexpected overlap or speculation")
        if any(info["cuda_graph_config"][phase]["backend"] != "disabled" for phase in ("prefill", "decode")):
            raise ValueError("Unexpected graph mode")
        (root / "server-info.json").write_text(json.dumps(info, indent=2) + "\n")
        manifest["status"] = "running"
        save()
        for planned in plan:
            name, input_length, concurrency, training = planned
            case_root = root / name
            case_root.mkdir(exist_ok=False)
            control = {"id": name, "forward_call": 48, "training": training,
                       "input_length": input_length, "global_requests": concurrency}
            (root / "node0/layer-probe-control.json").unlink(missing_ok=True)
            subprocess.run(ssh + [shlex.join(["rm", "-f", remote_root + "/layer-probe-control.json"])], check=True, timeout=15)
            (root / "node0" / name).mkdir(exist_ok=False)
            command = ["docker", "exec", "aisim-layer-tp32-node0-20261104", "python3", "-u", "-m", "sglang.benchmark.serving",
                       "--backend", "sglang", "--host", "127.0.0.1", "--port", "19180",
                       "--model", "/models/glm-q", "--served-model-name", "glm-5", "--dataset-name", "random-ids",
                       "--num-prompts", str(concurrency), "--max-concurrency", str(concurrency),
                       "--random-input-len", str(input_length), "--random-output-len", "96",
                       "--random-range-ratio", "1.0", "--request-rate", "inf", "--warmup-requests", "0",
                       "--seed", "910", "--tokenize-prompt", "--output-details", "--output-file", f"/campaign/{name}/requests.jsonl"]
            warmup = command[:]
            for flag, value in (("--num-prompts", "2"), ("--random-output-len", "32"),
                                ("--output-file", f"/campaign/{name}/warmup.jsonl")):
                warmup[warmup.index(flag) + 1] = value
            case = {**control, "command": command, "warmup_command": warmup,
                    "start_unix": time.time(), "status": "warming"}
            manifest["cases"].append(case)
            save()
            with (case_root / "warmup.log").open("w") as stream:
                subprocess.run(warmup, stdout=stream, stderr=subprocess.STDOUT, timeout=600, check=True)
            warmup_result = json.loads((root / "node0" / name / "warmup.jsonl").read_text().splitlines()[-1])
            if warmup_result["completed"] != 2 or any(warmup_result["errors"]):
                raise ValueError("Both uninstrumented warmup requests must complete")
            shutil.copy2(root / "node0" / name / "warmup.jsonl", case_root / "warmup.jsonl")
            text = json.dumps(control) + "\n"
            temporary = root / "node0/layer-probe-control.tmp"
            temporary.write_text(text)
            temporary.replace(root / "node0/layer-probe-control.json")
            temporary_remote = remote_root + "/layer-probe-control.tmp"
            remote_control = "cat > " + shlex.quote(temporary_remote) + " && " + shlex.join(["mv", temporary_remote, remote_root + "/layer-probe-control.json"])
            subprocess.run(ssh + [remote_control], input=text, text=True, check=True, timeout=15)
            case.update(start_unix=time.time(), status="running")
            save()
            with (case_root / "benchmark.log").open("w") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=1200)
            case.update(end_unix=time.time(), returncode=result.returncode)
            result.check_returncode()
            shutil.copy2(root / "node0" / name / "requests.jsonl", case_root / "requests.jsonl")
            benchmark = json.loads((case_root / "requests.jsonl").read_text().splitlines()[-1])
            if (benchmark["completed"] != concurrency or any(benchmark["errors"])
                    or benchmark["input_lens"] != [input_length] * concurrency
                    or benchmark["output_lens"] != [96] * concurrency):
                raise ValueError("Incomplete request/token conservation")
            for rank in range(32):
                path = case_root / f"layer-replay-rank{rank}.json"
                if rank < 16:
                    shutil.copy2(root / "node0" / name / path.name, path)
                else:
                    data = subprocess.check_output(ssh + [shlex.join(["cat", remote_root + "/" + name + "/" + path.name])], timeout=15)
                    path.write_bytes(data)
            records = [json.loads((case_root / f"layer-replay-rank{rank}.json").read_text()) for rank in range(32)]
            counts = []
            for rank, record in enumerate(records):
                if (record["rank"] != rank or record["status"] != "completed" or record["control"] != control
                        or record["parallelism"] != {"global_tp": 32, "attention_tp": 1, "attention_dp": 32}
                        or sorted(boundary["layer"] for boundary in record["boundaries"]) != [0, 1, 2, 3, 4, 5, 6, 77]):
                    raise ValueError("Unexpected captured source topology or boundaries")
                layer_counts = {sum(length > 0 for length in boundary["sequence_lengths"]) for boundary in record["boundaries"]}
                if len(layer_counts) != 1:
                    raise ValueError("Request population changed between layers")
                counts.append(layer_counts.pop())
            positive = {count for count in counts if count > 0}
            if sum(counts) != concurrency or len(positive) != 1:
                raise ValueError("Requested uniform active DP population differs from captured population")
            case.update(status="completed", captured_dp_requests=counts)
            save()
            print(json.dumps({"case": name, "status": "completed", "population": counts}), flush=True)
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
