import argparse
import importlib.util
import json
import os
import pathlib
import signal
import subprocess
import time


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SERVING = pathlib.Path("/home/youxingye/youxingye_data/910c_script/PD disaggregation")


def active(pid):
    path = pathlib.Path(f"/proc/{pid}/stat")
    return path.exists() and path.read_text().split()[2] != "Z"


def wait_exit(pid, timeout):
    deadline = time.monotonic() + timeout
    while active(pid):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Process {pid} has not exited")
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-pid", type=int, required=True)
    parser.add_argument("--postprocessor-pid", type=int, required=True)
    parser.add_argument("--client-pid", type=int, required=True)
    args = parser.parse_args()
    controller = pathlib.Path(f"/proc/{args.controller_pid}")
    if b"run-serving-matrix.py" not in (controller / "cmdline").read_bytes():
        raise ValueError("Controller identity does not match")
    environment = {key.decode(): value.decode() for item in (controller / "environ").read_bytes().split(b"\0") if item for key, value in [item.split(b"=", 1)]}
    if not environment.get("PD_SUDO_PASSWORD"):
        raise ValueError("Authorized remote credentials are not retained by the controller")
    command = pathlib.Path(f"/proc/{args.client_pid}/cmdline").read_bytes().split(b"\0")
    if b"sglang.benchmark.serving" not in command or command[command.index(b"--max-concurrency") + 1] != b"128" or command[command.index(b"--num-prompts") + 1] != b"256":
        raise ValueError("Client identity does not match the failed C128 measurement")
    data = SERVING / "results/aisim-comparison"
    directory = data / "control-1p3d-tp16-20261101"
    point = directory / "c128-n256"
    if (point / ".complete").exists() or not (HERE / "control-1p3d-c128-bootstrap-first-event.json").exists():
        raise ValueError("Expected an incomplete point with preserved bootstrap timeout evidence")
    attempt = directory / "failed-attempts" / f"bootstrap-timeout-{time.time_ns()}"
    attempt.mkdir(parents=True)
    progress = {
        "status": "recovering", "started": time.time(),
        "reason": "Observed 600-second bootstrap timeout while waiting for decode KV indices at C128",
        "configuration": "control-1p3d-tp16", "retry_concurrency": 128,
        "failed_attempt": str(attempt), "failed_client_run_interrupted": True,
        "completed_points_preserved": json.loads((data / "matrix-progress.json").read_text())["completed_points"],
        "previous_limits": {"PD_BOOTSTRAP_TIMEOUT": 600, "ROUTER_REQUEST_TIMEOUT": 1800},
        "new_limits": {"PD_BOOTSTRAP_TIMEOUT": 7200, "ROUTER_REQUEST_TIMEOUT": 21600},
        "scope": "Retry C128 and use extended waiting limits for subsequent starts. Completed points retain original limits; no operator/parallelism/graph settings changed. Failed partial attempt is excluded from valid performance comparisons.",
    }

    def save(phase):
        progress.update({"phase": phase, "updated": time.time()})
        path = data / "bootstrap-timeout-recovery.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=2) + "\n")
        temporary.replace(path)
        print(phase, flush=True)

    try:
        save("interrupting_failed_client")
        os.kill(args.client_pid, signal.SIGINT)
        try:
            wait_exit(args.client_pid, 15)
        except TimeoutError:
            os.kill(args.client_pid, signal.SIGTERM)
            wait_exit(args.client_pid, 15)
        wait_exit(args.controller_pid, 90)
        wait_exit(args.postprocessor_pid, 90)
        for name in ("matrix-progress.json", "comparison-progress.json"):
            (attempt / name).write_bytes((data / name).read_bytes())
        if not (point / "metrics-summary.json").exists():
            raise RuntimeError("Metric sampler has not finalized the interrupted trace")
        spec = importlib.util.spec_from_file_location("matrix_recovery", SERVING / "tools/performance/run-serving-matrix.py")
        matrix = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(matrix)
        plan = json.loads((data / "comparison-plan.json").read_text())
        configuration = next(item for item in plan["configurations"] if item["label"] == "control-1p3d-tp16")
        old_environment = os.environ.copy()
        os.environ.update(environment)
        try:
            serving_environment = matrix.environment(configuration, plan["workload"], directory)
        finally:
            os.environ.clear()
            os.environ.update(old_environment)
        save("archiving_failed_deployment_logs")
        matrix.archive_logs(serving_environment, attempt)
        point.rename(attempt / point.name)
        progress["failed_point_archive"] = str(attempt / point.name)
        save("stopping_failed_deployment")
        matrix.execute([str(SERVING / "service/stop-pd.sh")], serving_environment, attempt / "stop.log", 180)
        environment.update({key: str(value) for key, value in progress["new_limits"].items()})
        with pathlib.Path("/tmp/aisim-serving-matrix-20261101.log").open("a") as log:
            controller_process = subprocess.Popen(["python3", str(SERVING / "tools/performance/run-serving-matrix.py"), "--output", str(data), "--resume"], cwd=SERVING, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        progress["controller_pid"] = controller_process.pid
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if controller_process.poll() is not None:
                raise RuntimeError("Replacement controller exited before resuming")
            if json.loads((data / "matrix-progress.json").read_text())["status"] == "running":
                break
            time.sleep(1)
        else:
            raise TimeoutError("Replacement controller did not publish running state")
        with pathlib.Path("/tmp/aisim-comparison-postprocess-20261101.log").open("a") as log:
            postprocessor = subprocess.Popen(["python3", str(HERE / "finish_serving_comparison.py"), "--measurement-root", str(data), "--commit-measurements"], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        progress.update({"postprocessor_pid": postprocessor.pid, "status": "resumed"})
        save("replacement_controller_started")
    except BaseException as error:
        progress.update({"status": "failed", "error": str(error)})
        save("recovery_failed")
        raise


if __name__ == "__main__":
    main()
