import argparse
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--case-prefix", default="")
    parser.add_argument("--manifest", default="component-campaign.json")
    args = parser.parse_args()
    root = args.campaign
    destination = root / args.manifest
    if destination.exists():
        raise FileExistsError(destination)
    plan = [("diagnose-8192-c4", 8192, 4), ("holdout-12288-c4", 12288, 4),
            ("holdout-8192-c8", 8192, 8)]
    manifest = {"status": "starting", "cases": [], "plan": plan,
                "scope": "Component boundary diagnosis; instrumented request latency is not serving ground truth",
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def save():
        destination.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    try:
        with urllib.request.urlopen("http://127.0.0.1:18080/get_server_info", timeout=10) as response:
            info = json.load(response)
        if (info["tp_size"], info["attn_dp_size"], info["ep_size"]) != (16, 4, 16):
            raise ValueError("Unexpected topology")
        if not info["disable_overlap_schedule"] or info["speculative_algorithm"] is not None:
            raise ValueError("Unexpected execution mode")
        if any(info["cuda_graph_config"][phase]["backend"] != "disabled" for phase in ("prefill", "decode")):
            raise ValueError("Unexpected graph mode")
        (root / "server-info.json").write_text(json.dumps(info, indent=2) + "\n")
        manifest["status"] = "running"
        save()
        for name, input_length, concurrency in plan:
            training = name.startswith("diagnose-")
            name = args.case_prefix + name
            case_root = root / name
            case_root.mkdir(exist_ok=False)
            control = {"id": name, "forward_call": 80, "training": training,
                       "input_length": input_length, "global_requests": concurrency}
            temporary = root / "layer-probe-control.tmp"
            temporary.write_text(json.dumps(control) + "\n")
            temporary.replace(root / "layer-probe-control.json")
            command = ["docker", "exec", args.container, "python3", "-m", "sglang.benchmark.serving",
                       "--backend", "sglang", "--host", "127.0.0.1", "--port", "18080",
                       "--model", "/models/glm-q", "--served-model-name", "glm-5", "--dataset-name", "random-ids",
                       "--num-prompts", str(concurrency), "--max-concurrency", str(concurrency),
                       "--random-input-len", str(input_length), "--random-output-len", "128",
                       "--random-range-ratio", "1.0", "--request-rate", "inf", "--warmup-requests", "2",
                       "--seed", "940", "--tokenize-prompt", "--output-details",
                       "--output-file", f"/campaign/{name}/requests.jsonl"]
            case = {**control, "command": command, "start_unix": time.time(), "status": "running"}
            manifest["cases"].append(case)
            save()
            with (case_root / "benchmark.log").open("w") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
            case.update(end_unix=time.time(), returncode=result.returncode)
            result.check_returncode()
            benchmark = json.loads((case_root / "requests.jsonl").read_text().splitlines()[-1])
            if (benchmark["completed"] != concurrency or any(benchmark["errors"])
                    or benchmark["input_lens"] != [input_length] * concurrency
                    or benchmark["output_lens"] != [128] * concurrency):
                raise ValueError(f"Request completion/length mismatch: {name}")
            counts = []
            for rank in range(16):
                path = case_root / f"components-rank{rank}.json"
                observed = json.loads(path.read_text())
                if (observed["status"] != "completed" or observed["rank"] != rank
                        or len(observed["layers"]) != 8 or observed["control"] != control):
                    raise ValueError(f"Invalid rank profile: {path}")
                for layer in observed["layers"]:
                    lengths = [value for value in layer["sequence_lengths"] if value > 0]
                    if (len(lengths) != concurrency // 4
                            or not all(input_length <= value < input_length + 128 for value in lengths)
                            or layer["forward_mode"] != "2"):
                        raise ValueError(f"Captured batch differs from steady decode: {path}")
                    if [population["mode"] for population in layer["populations"]] != [
                            "control_before", "host", "events", "control_after"]:
                        raise ValueError(f"Invalid paired population: {path}")
                    if any(len(population["samples"]) != 12 for population in layer["populations"]):
                        raise ValueError(f"Incomplete sample population: {path}")
                counts.append(len(lengths))
            if sum(counts[::4]) != concurrency:
                raise ValueError(f"Distributed request population mismatch: {counts}")
            case.update(status="completed", captured_dp_requests=counts[::4])
            save()
            print(json.dumps({key: case[key] for key in ("id", "status", "captured_dp_requests")}), flush=True)
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
