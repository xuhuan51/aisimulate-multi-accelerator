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
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--output-name", default="serving-validation")
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name or args.output_name in (".", ".."):
        parser.error("Output name must be a single directory name")
    campaign = args.campaign.resolve()
    qualification = args.qualification or campaign / "qualification-progress.json"
    root = campaign / args.output_name
    root.mkdir(exist_ok=False)
    remote = json.loads((campaign / "node1/launch.json").read_text())["remote_root"]
    ssh = ["ssh", "-S", args.ssh_socket, "-o", "BatchMode=yes", "youxingye@npu1-25"]
    manifest = {"status": "waiting_for_qualified_layer_collection", "cases": [],
                "warmup_requests": 2, "seed": 1810,
                "scope": "Two-node aggregated TP32/ADP32/EP32 serving with all layer replay probes disabled; independent token seed and workload holdouts; not PD TTFT validation",
                "plan": json.loads(args.plan.read_text()),
                "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest()}
    destination = root / "campaign.json"

    def save():
        destination.write_text(json.dumps(manifest, indent=2) + "\n")

    def verify_disarmed():
        if (campaign / "node0/layer-probe-control.json").exists():
            raise ValueError("Node0 layer sampling is still armed")
        subprocess.run(ssh + [shlex.join(["test", "!", "-e", remote + "/layer-probe-control.json"])],
                       check=True, timeout=15)

    save()
    try:
        deadline = time.monotonic() + 7200
        while time.monotonic() < deadline:
            state = json.loads(qualification.read_text())
            if state["status"] == "qualified":
                break
            if state["status"] == "failed":
                raise RuntimeError("Layer qualification failed; independent serving has not started")
            time.sleep(10)
        else:
            raise TimeoutError("Layer qualification did not finish")
        verify_disarmed()
        manifest["layer_qualification"] = {"path": str(qualification),
            "sha256": hashlib.sha256(qualification.read_bytes()).hexdigest(), "declaration": state}
        with urllib.request.urlopen("http://127.0.0.1:19180/get_server_info", timeout=10) as response:
            info = json.load(response)
        if (info["tp_size"], info["attn_dp_size"], info["ep_size"]) != (32, 32, 32):
            raise ValueError("Unexpected serving topology")
        if not info["disable_overlap_schedule"] or info["speculative_algorithm"] is not None:
            raise ValueError("Unexpected overlap or speculation")
        if any(info["cuda_graph_config"][phase]["backend"] != "disabled" for phase in ("prefill", "decode")):
            raise ValueError("Unexpected graph execution")
        (root / "server-info.json").write_text(json.dumps(info, indent=2) + "\n")
        helper = Path(__file__).parent / "run_traced_client.py"
        shutil.copy2(helper, campaign / "node0/run_traced_client.py")
        manifest.update(status="running", client_sha256=hashlib.sha256(helper.read_bytes()).hexdigest())
        save()
        for name, input_length, output_length, concurrency, prompts in manifest["plan"]:
            verify_disarmed()
            case_root = root / name
            case_root.mkdir(exist_ok=False)
            mounted = campaign / "node0" / args.output_name / name
            mounted.mkdir(parents=True, exist_ok=False)
            command = ["docker", "exec", "aisim-layer-tp32-node0-20261104", "python3", "-u", "/campaign/run_traced_client.py",
                       "--backend", "sglang", "--host", "127.0.0.1", "--port", "19180", "--model", "/models/glm-q",
                       "--served-model-name", "glm-5", "--dataset-name", "random-ids", "--num-prompts", str(prompts),
                       "--max-concurrency", str(concurrency), "--random-input-len", str(input_length),
                       "--random-output-len", str(output_length), "--random-range-ratio", "1.0", "--request-rate", "inf",
                       "--warmup-requests", "2", "--seed", str(manifest["seed"]), "--tokenize-prompt", "--output-details",
                       "--output-file", f"/campaign/{args.output_name}/{name}/requests.jsonl"]
            case = {"id": name, "input_length": input_length, "output_length": output_length,
                    "concurrency": concurrency, "prompts": prompts, "command": command,
                    "status": "running", "start_unix": time.time()}
            manifest["cases"].append(case)
            save()
            with (case_root / "benchmark.log").open("w") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=5400)
            case.update(end_unix=time.time(), returncode=result.returncode)
            result.check_returncode()
            for name in ("requests.jsonl", "client-dispatch.jsonl"):
                shutil.copy2(mounted / name, case_root / name)
            measured = json.loads((case_root / "requests.jsonl").read_text().splitlines()[-1])
            if (measured["completed"] != prompts or any(measured["errors"])
                    or measured["input_lens"] != [input_length] * prompts
                    or measured["output_lens"] != [output_length] * prompts):
                raise ValueError("Incomplete serving request or token totals")
            case["status"] = "completed"
            save()
            print(json.dumps({"case": case["id"], "completed": prompts, "ttft_ms": measured["mean_ttft_ms"],
                              "tpot_ms": measured["mean_tpot_ms"]}), flush=True)
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
