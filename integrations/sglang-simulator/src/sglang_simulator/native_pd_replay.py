"""Paired native PD process entry point with one shared virtual event clock."""

import dataclasses
import json
import multiprocessing
import os
from pathlib import Path
import random
import queue
import signal
import sys
import time
import traceback


def run_role(case, cohort, directory, clock, transport, start, status, deadline):
    os.setsid()
    directory = Path(directory)
    directory.mkdir()
    role, worker_id = cohort["role"], cohort["worker_id"]
    worker = case.get("workers", {}).get(role, case)
    identity = dict(worker["predictor_config"])
    if "workers" not in case:
        identity["worker_type"] = role
    if identity["worker_type"] != role:
        raise ValueError("Native PD worker identity differs from its role")
    simulation = {
        "platform": {"accelerator": worker["hardware"], "num_device_per_node": worker["devices_per_node"]},
        "predictor": {"name": "aisimulate", "python_executable": os.environ["SGLANG_SIMULATOR_PREDICTOR_PYTHON"],
                      "timeout_seconds": 120, "config": identity},
        "scheduler": {"tp_size": worker["world_size"], "dp_size": identity["attention_dp"],
                      "ep_size": identity["moe_ep_size"], "backend_name": "sglang",
                      "backend_version": identity["backend_version"]},
        "request_timing": worker.get("request_timing", {}),
    }
    config_path = directory / "sim-config.json"
    config_path.write_text(json.dumps(simulation, indent=2))
    os.environ.update(SGLANG_SIMULATOR_CONFIG_PATH=str(config_path),
                      SGLANG_SIMULATOR_OUTPUT_DIR=str(directory),
                      SGLANG_SIMULATOR_OUTPUT_MODE="OFFLINE", SGLANG_USE_CPU_ENGINE="1")
    engine = None
    try:
        from sglang_simulator.simulation.manager import ConfigManager
        from sglang_simulator.simulation.sglang.engine import SGLangSimulationEngine
        from sglang_simulator.simulation.sglang.runtime_adapter import server_arg_fields, server_arg_values
        from sglang_simulator.native_replay import native_runtime_info, reserve_port_span, simulation_server_args
        from sglang.srt.arg_groups.pd_disaggregation_hook import handle_pd_disaggregation
        from sglang.srt.server_args import ServerArgs

        ConfigManager.reset_config_cache()
        ConfigManager._native_pd_clock = clock
        ConfigManager._native_pd_context = {
            "worker_id": worker_id, "role": role, "transport": transport,
            "attention_tp": identity["tp"], "attention_dp": identity["attention_dp"],
            "prefill_attention_dp": case.get("workers", {}).get("prefill", case)["predictor_config"]["attention_dp"],
            "bootstrap_address": "127.0.0.1:28998",
            # Native CPU KV payload dimensions are compacted by the existing
            # simulator. Transfer volume uses canonical target geometry.
            "kv_page_bytes": worker["recurrent_state_geometry"]["kv_bytes_per_token"] * worker["server_args"]["page_size"],
            "state_bytes_per_request": worker["recurrent_state_geometry"]["bytes_per_request"],
            "shared_admission": True,
            "request_dp_ranks": {rid: ranks[role] for rid, ranks in case.get("request_dp_ranks", {}).items()}
                                if "request_dp_ranks" in case else None,
        }
        fields = server_arg_fields(ServerArgs)
        template_path = os.environ.get("SGLANG_SIMULATOR_NATIVE_ARGS")
        template = json.loads(Path(template_path).read_text()) if template_path else {}
        kwargs = simulation_server_args(worker["server_args"], template, fields)
        kwargs.update(port=reserve_port_span(), nccl_port=None, dist_init_addr=None,
                      disaggregation_mode=role, disaggregation_transfer_backend="mooncake",
                      disaggregation_bootstrap_port=28998,
                      load_balance_method="follow_bootstrap_room")
        if template_path:
            kwargs["chunked_prefill_size"] //= identity["attention_dp"]
        if "cuda_graph_backend_decode" in fields:
            from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig
            kwargs.update(cuda_graph_backend_decode="disabled", cuda_graph_backend_prefill="disabled",
                          cuda_graph_config=CudaGraphConfig())
        original_resolve = ServerArgs.resolve_once

        def resolve_template(self):
            if self.device == "cpu":
                handle_pd_disaggregation(self)
                self._resolution_finished = True
                return
            return original_resolve(self)

        if template_path:
            ServerArgs.resolve_once = resolve_template
        (directory / "launch-args.json").write_text(json.dumps(kwargs, indent=2, default=dataclasses.asdict))
        engine = SGLangSimulationEngine(**kwargs)
        effective = server_arg_values(engine.server_args)
        for name in ("page_size", "max_total_tokens"):
            if effective[name] != worker["server_args"][name]:
                raise ValueError(f"Native {role} resolution changed {name}: {effective[name]}")
        (directory / "resolved-args.json").write_text(json.dumps(effective, indent=2))
        (directory / "runtime.json").write_text(json.dumps(native_runtime_info(ServerArgs, identity), indent=2))
        engine.start_profile()
        from sglang_simulator.simulation.sglang.parallel_hooks import _MANAGERS

        status.put({"role": role, "worker_id": worker_id, "event": "ready", "pid": os.getpid(),
                    "coordinator": _MANAGERS[-1]._simulator_coordinator})
        if not start.wait(max(0, deadline - time.monotonic())):
            raise TimeoutError("Paired native role did not become ready")
        corpus, sampling, ids, rooms = [], [], [], []
        for index in cohort["request_indices"]:
            request = case["requests"][index]
            rng = random.Random(910000 + index)
            corpus.append([rng.randrange(100, 10000) for _ in range(request["input_tokens"])])
            sampling.append({"max_new_tokens": request["output_tokens"], "temperature": 0,
                             "ignore_eos": True, "custom_params": {"simulation": {
                                 "created_time_ms": request["arrival_time_ms"],
                                 "total_request": len(cohort["request_indices"]), "request_index": index,
                                 "max_concurrency": case["concurrency"],
                             }}})
            ids.append(str(request["id"]))
            if len(case.get("cohorts", [])) > 2:
                dp = case["workers"]["prefill"]["predictor_config"]["attention_dp"]
                prefill_dp = case["request_dp_ranks"][str(request["id"])]["prefill"]
                rooms.append((1000 + index) * dp + prefill_dp)
            else:
                rooms.append(1000 + index)
        responses = engine.generate(input_ids=corpus, sampling_params=sampling, rid=ids,
                                    bootstrap_host="127.0.0.1", bootstrap_port=28998,
                                    bootstrap_room=rooms)
        (directory / "responses.json").write_text(json.dumps(responses, indent=2))
        engine.stop_profile()
        status.put({"role": role, "worker_id": worker_id, "event": "completed", "responses": len(responses)})
    except BaseException as error:
        (directory / "failure.txt").write_text(traceback.format_exc())
        status.put({"role": role, "worker_id": worker_id, "event": "failed", "error": repr(error)})
        clock.fail(f"Native {worker_id} failed: {error}")
        raise
    finally:
        try:
            if engine is not None:
                engine.shutdown()
        finally:
            # Native shutdown also reaps its coordinator process. Retire the
            # cohort on the parent-owned clock after its workers have exited.
            clock.stop(worker_id)


def main():
    path = Path(sys.argv[1]).resolve()
    case = json.loads(path.read_text())
    from sglang_simulator.aisimulate_runner import pd_cohorts

    cohorts = pd_cohorts(case)
    if any(request["arrival_time_ms"] != 0 for request in case["requests"]):
        raise ValueError("Initial native PD execution requires requests arriving at zero")
    if os.environ.get("SGLANG_USE_CPU_ENGINE") != "1":
        raise ValueError("Paired native replay requires the CPU simulator")
    # Every virtual rank shares one CPU host. Changes in host free memory do
    # not describe target per-device capacity, which comes from the case.
    os.environ["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"] = "0"
    metadata = json.loads((Path(case["predictor_config"]["model"]) / "config.json").read_text())
    if metadata.get("model_type") == "qwen3_5_moe":
        # Use the native registry's module filter to keep the selected model
        # while reducing immutable imports replicated by CPU fork servers.
        native_root = os.environ.get("SGLANG_SIMULATOR_SGLANG_ROOT")
        if native_root:
            model_modules = Path(native_root) / "python/sglang/srt/models"
        else:
            from importlib.util import find_spec

            package = find_spec("sglang")
            if package is None or not package.submodule_search_locations:
                raise ImportError("Native PD replay requires the selected SGLang runtime")
            model_modules = Path(next(iter(package.submodule_search_locations))) / "srt/models"
        disabled = set(os.environ.get("SGLANG_DISABLED_MODEL_ARCHS", "").split(",")) - {""}
        disabled.update(module.stem for module in model_modules.glob("*.py")
                        if module.stem not in {"__init__", "qwen3_5"})
        os.environ["SGLANG_DISABLED_MODEL_ARCHS"] = ",".join(sorted(disabled))
    (path.parent / "native-environment.json").write_text(json.dumps({
        "architectures": metadata.get("architectures"),
        "SGLANG_DISABLED_MODEL_ARCHS": os.environ.get("SGLANG_DISABLED_MODEL_ARCHS"),
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": os.environ["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"],
    }, indent=2))
    # Import once before creating either role or its RPC threads. The isolated
    # role parents share immutable model-registry pages through copy-on-write.
    # Each role still launches its own native workers with the configured CPU
    # forkserver context and independent runtime state.
    import sglang_simulator.simulation.sglang.cpu_worker_preload
    from sglang_simulator.simulation.parallel.coordinator import CoordinatorManager

    context = multiprocessing.get_context("fork")
    manager = CoordinatorManager(ctx=context)
    manager.start()
    clock = manager.NativePDClock([cohort["worker_id"] for cohort in cohorts], timeout=300)
    clock.configure_requests([str(request["id"]) for request in case["requests"]],
                             case["concurrency"] if case["concurrency"] is not None else len(case["requests"]),
                             "decode", case.get("request_routes"))
    transport = manager.NativePDTransfer(clock, case["pd_bandwidth_bytes_per_second"])
    start, status = context.Event(), context.Queue()
    deadline = time.monotonic() + float(os.environ.get("SGLANG_SIMULATOR_RUN_TIMEOUT", "3600"))
    processes = [context.Process(target=run_role, args=(case, cohort, str(path.parent / cohort["worker_id"]),
                                                       clock, transport, start, status, deadline))
                 for cohort in cohorts]
    events, ready, completed = [], set(), set()
    coordinators = {}
    try:
        # Stagger the two native launch peaks on CPU-only hosts. Both cohorts
        # reach the ready barrier before the shared virtual workload starts.
        processes[0].start()
        while len(completed) < len(processes):
            if time.monotonic() >= deadline:
                raise TimeoutError("Paired native PD replay exceeded its wall-clock deadline")
            try:
                event = status.get(timeout=5)
            except queue.Empty:
                snapshot = {"clock": clock.describe(), "coordinators": {}}
                for role, coordinator in coordinators.items():
                    try:
                        snapshot["coordinators"][role] = coordinator.describe()
                    except (EOFError, BrokenPipeError, ConnectionError):
                        snapshot["coordinators"][role] = "closed"
                (path.parent / "progress.json").write_text(json.dumps(snapshot, indent=2))
                exited = [p for p in processes if p.exitcode is not None and p.exitcode != 0]
                if exited:
                    raise RuntimeError(f"Native PD role processes exited: {[(p.pid, p.exitcode) for p in exited]}")
                continue
            if "coordinator" in event:
                coordinators[event["worker_id"]] = event.pop("coordinator")
            events.append(event)
            (path.parent / "events.json").write_text(json.dumps(events, indent=2))
            if event["event"] == "failed":
                raise RuntimeError(event["error"])
            if event["event"] == "ready":
                ready.add(event["worker_id"])
                if len(ready) < len(processes) and processes[len(ready)].pid is None:
                    processes[len(ready)].start()
                if len(ready) == len(processes):
                    start.set()
            if event["event"] == "completed":
                completed.add(event["worker_id"])
        for process in processes:
            process.join(60)
            if process.exitcode != 0:
                raise RuntimeError(f"Native role failed to shut down cleanly: {process.pid}, {process.exitcode}")
        (path.parent / "transfer.json").write_text(json.dumps(transport.report(), indent=2))
        (path.parent / "admission.json").write_text(json.dumps(clock.request_report(), indent=2))
        from sglang_simulator.aisimulate_runner import summarize_pd_run

        report = summarize_pd_run(path.parent, case)
        (path.parent / "report.json").write_text(json.dumps(report, indent=2))
    finally:
        try:
            (path.parent / "transfer-partial.json").write_text(json.dumps(transport.report(), indent=2))
            (path.parent / "admission-partial.json").write_text(json.dumps(clock.request_report(), indent=2))
        finally:
            close_native_pd(processes, manager, clock)


def close_native_pd(processes, manager, clock):
    try:
        clock.fail("Paired native replay closing")
    finally:
        try:
            for process in processes:
                if process.pid is None:
                    continue
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(process.pid, sig)
                    except ProcessLookupError:
                        pass
                    process.join(15)
        finally:
            manager.shutdown()


if __name__ == "__main__":
    main()
