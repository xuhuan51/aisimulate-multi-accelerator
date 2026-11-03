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
    args = parser.parse_args()
    root = args.campaign
    manifest = {"status": "waiting", "cases": [], "scope": "Paired continuous-forward event attribution; no fitted latency",
                "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target = root / "prefill-event-campaign.json"

    def save():
        target.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    try:
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen("http://127.0.0.1:18080/health", timeout=5) as response:
                    if response.status == 200:
                        break
            except (OSError, TimeoutError):
                pass
            state = subprocess.check_output(["docker", "inspect", "-f", "{{.State.Running}}", args.container], text=True).strip()
            if state != "true":
                raise RuntimeError("Diagnostic container stopped before health")
            time.sleep(5)
        else:
            raise TimeoutError("Diagnostic service did not become ready")
        with urllib.request.urlopen("http://127.0.0.1:18080/get_server_info", timeout=10) as response:
            info = json.load(response)
        if (info["tp_size"], info["attn_dp_size"], info["ep_size"]) != (16, 4, 16):
            raise ValueError("Unexpected diagnostic topology")
        if not info["disable_overlap_schedule"] or info["speculative_algorithm"] is not None:
            raise ValueError("Unexpected overlap/speculation")
        if any(info["cuda_graph_config"][phase]["backend"] != "disabled" for phase in ("prefill", "decode")):
            raise ValueError("Unexpected graph execution")
        (root / "server-info.json").write_text(json.dumps(info, indent=2) + "\n")
        manifest["status"] = "running"
        for concurrency in (1, 2, 4, 8):
            for repetition, events in enumerate((False, True, False, True)):
                name = f"c{concurrency}-r{repetition}-{'events' if events else 'control'}"
                directory = root / name
                directory.mkdir(exist_ok=False)
                control = root / "prefill-event-control.json"
                temporary = control.with_suffix(".tmp")
                temporary.write_text(json.dumps({"id": name, "events": events}) + "\n")
                temporary.replace(control)
                command = ["docker", "exec", args.container, "python3", "-m", "sglang.benchmark.serving",
                           "--backend", "sglang", "--host", "127.0.0.1", "--port", "18080",
                           "--model", "/models/glm-q", "--served-model-name", "glm-5", "--dataset-name", "random-ids",
                           "--num-prompts", str(concurrency), "--max-concurrency", str(concurrency),
                           "--random-input-len", "8192", "--random-output-len", "8", "--random-range-ratio", "1.0",
                           "--request-rate", "inf", "--warmup-requests", "2" if repetition == 0 else "0",
                           "--seed", "930", "--tokenize-prompt", "--output-details", "--output-file",
                           f"/campaign/{name}/requests.jsonl"]
                case = {"id": name, "concurrency": concurrency, "events": events, "command": command,
                        "status": "running", "start_unix": time.time()}
                manifest["cases"].append(case)
                save()
                with (directory / "benchmark.log").open("w") as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=600, check=True)
                result = json.loads((directory / "requests.jsonl").read_text().splitlines()[-1])
                if (result["completed"] != concurrency or any(result["errors"])
                        or result["input_lens"] != [8192] * concurrency or result["output_lens"] != [8] * concurrency):
                    raise ValueError(f"Request/token conservation failed: {name}")
                case.update(status="completed", end_unix=time.time())
                save()
                print(json.dumps({"id": name, "status": "completed"}), flush=True)
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
