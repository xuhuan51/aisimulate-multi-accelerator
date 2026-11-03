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
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--ssh-socket", required=True)
    args = parser.parse_args()
    root = args.campaign.resolve()
    ssh = ["ssh", "-S", args.ssh_socket, "-o", "BatchMode=yes", "youxingye@npu1-25"]
    remote = json.loads((root / "node1/launch.json").read_text())["remote_root"]
    destination = root / "routing-campaign.json"
    if destination.exists():
        raise FileExistsError(destination)
    plan = json.loads(args.plan.read_text())
    manifest = {"status": "waiting", "cases": [], "plan": plan, "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
                "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scope": "Observed75-layer expert assignment counts, no fitted latency or perf-row replacement. Input lengths and random seeds marked heldout must not enter the training routing distribution."}

    def save():
        destination.write_text(json.dumps(manifest, indent=2) + "\n")

    def disarm():
        (root / "node0/routing-profile-control.json").unlink(missing_ok=True)
        subprocess.run(ssh + [shlex.join(["rm", "-f", remote + "/routing-profile-control.json"])], check=True, timeout=15)

    def validate_case(case):
        groups = {}
        for rank in range(32):
            name = f"routing-profile-rank{rank}.jsonl"
            content = ((root / "node0" / name).read_bytes() if rank < 16 else
                       subprocess.check_output(ssh + [shlex.join(["cat", remote + "/" + name])], timeout=30))
            lines = []
            for line in content.splitlines():
                row = json.loads(line)
                if row["case"] != case["id"]:
                    continue
                if row["tp_rank"] != rank or sorted(map(int, row["layers"])) != list(range(3, 78)):
                    raise ValueError("Missing rank or MoE layer identity")
                group = groups.setdefault(row["model_call"], {})
                if rank in group:
                    raise ValueError("Duplicate rank/forward observation")
                group[rank] = row
                lines.append(line)
            if not lines:
                raise ValueError("Every TP rank must provide observed expert counts")
            (root / case["id"] / name).write_bytes(b"\n".join(lines) + b"\n")
        observations = []
        for iteration, ranks in sorted(groups.items()):
            if len(ranks) != 32:
                raise ValueError("Incomplete32-rank routing observation")
            global_counts = ranks[0]["global_num_tokens_cpu"]
            if any(row["global_num_tokens_cpu"] != global_counts for row in ranks.values()):
                raise ValueError("Ranks disagree on global token volume")
            total_tokens = sum(global_counts)
            if total_tokens <= 0:
                raise ValueError("Captured empty routing population")
            layers = []
            for layer in range(3, 78):
                counts = [count for rank in range(32) for count in ranks[rank]["layers"][str(layer)]["expert_counts"]]
                if len(counts) != 256 or sum(counts) != total_tokens * 8:
                    raise ValueError(f"Expert assignment conservation failed at iteration{iteration}, layer{layer}")
                layers.append({"layer": layer, "expert_counts": counts})
            observations.append({"iteration": iteration, "global_tokens": total_tokens,
                                 "global_num_tokens_cpu": global_counts, "layers": layers})
        (root / case["id"] / "observations.json").write_text(json.dumps({"training": case["training"], "groups": observations}, indent=2) + "\n")
        return len(observations)

    save()
    try:
        deadline = time.monotonic() + 2400
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen("http://127.0.0.1:19180/get_server_info", timeout=2) as response:
                    info = json.load(response)
                break
            except (OSError, ValueError):
                time.sleep(2)
        else:
            raise TimeoutError("Routing observation service did not become ready")
        if (info["tp_size"], info["attn_dp_size"], info["ep_size"]) != (32, 32, 32):
            raise ValueError("Unexpected routing observation topology")
        if not info["disable_overlap_schedule"] or info["speculative_algorithm"] is not None:
            raise ValueError("Unexpected scheduling or speculative execution")
        if any(info["cuda_graph_config"][phase]["backend"] != "disabled" for phase in ("prefill", "decode")):
            raise ValueError("Unexpected graph execution")
        (root / "server-info.json").write_text(json.dumps(info, indent=2) + "\n")
        for name, length, concurrency, prompts, seed, training in plan:
            folder = root / name
            folder.mkdir(exist_ok=False)
            mounted = root / "node0" / name
            mounted.mkdir(exist_ok=False)
            disarm()
            command = ["docker", "exec", "aisim-layer-tp32-node0-20261104", "python3", "-u", "-m", "sglang.benchmark.serving",
                       "--backend", "sglang", "--host", "127.0.0.1", "--port", "19180", "--model", "/models/glm-q",
                       "--served-model-name", "glm-5", "--dataset-name", "random-ids", "--num-prompts", str(prompts),
                       "--max-concurrency", str(concurrency), "--random-input-len", str(length), "--random-output-len", "16",
                       "--random-range-ratio", "1.0", "--request-rate", "inf", "--warmup-requests", "0", "--seed", str(seed),
                       "--tokenize-prompt", "--output-details", "--output-file", f"/campaign/{name}/requests.jsonl"]
            warmup = command[:]
            for flag, value in (("--num-prompts", "2"), ("--random-output-len", "32"), ("--output-file", f"/campaign/{name}/warmup.jsonl")):
                warmup[warmup.index(flag) + 1] = value
            case = {"id": name, "input_length": length, "concurrency": concurrency, "prompts": prompts,
                    "seed": seed, "training": training, "command": command, "warmup_command": warmup, "status": "warming"}
            manifest["cases"].append(case)
            manifest["status"] = "running"
            save()
            with (folder / "warmup.log").open("w") as stream:
                subprocess.run(warmup, check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
            warm = json.loads((mounted / "warmup.jsonl").read_text().splitlines()[-1])
            if warm["completed"] != 2 or any(warm["errors"]):
                raise ValueError("Unprofiled warmups did not complete")
            shutil.copy2(mounted / "warmup.jsonl", folder / "warmup.jsonl")
            control = {"id": name, "training": training, "input_length": length, "seed": seed}
            temporary = root / "node0/routing-profile-control.tmp"
            temporary.write_text(json.dumps(control) + "\n")
            temporary.replace(root / "node0/routing-profile-control.json")
            remote_command = "cat > " + shlex.quote(remote + "/routing-profile-control.tmp") + " && " + shlex.join(["mv", remote + "/routing-profile-control.tmp", remote + "/routing-profile-control.json"])
            subprocess.run(ssh + [remote_command], input=json.dumps(control) + "\n", text=True, check=True, timeout=15)
            case.update(status="running", start_unix=time.time())
            save()
            with (folder / "benchmark.log").open("w") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
            case.update(end_unix=time.time(), returncode=result.returncode)
            result.check_returncode()
            disarm()
            shutil.copy2(mounted / "requests.jsonl", folder / "requests.jsonl")
            measured = json.loads((folder / "requests.jsonl").read_text().splitlines()[-1])
            if measured["completed"] != prompts or any(measured["errors"]) or measured["input_lens"] != [length] * prompts or measured["output_lens"] != [16] * prompts:
                raise ValueError("Request or token conservation failed")
            case.update(groups=validate_case(case), status="completed")
            save()
            print(json.dumps({key: case[key] for key in ("id", "groups", "status")}), flush=True)
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        try:
            disarm()
            manifest["probes_disarmed"] = True
        except Exception as error:
            manifest["probes_disarmed"] = False
            manifest["cleanup_error"] = repr(error)
            manifest["status"] = "failed"
        save()
        if not manifest["probes_disarmed"]:
            raise RuntimeError(manifest["cleanup_error"])


if __name__ == "__main__":
    main()
