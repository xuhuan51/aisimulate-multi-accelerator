"""AISimulate search adapter executing the native SGLang simulator in a child.

AISimulate remains responsible for candidate generation and canonical timing /
memory identities. No EngineReplayRunner is used to execute a candidate.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


class NativeSGLangRunnerFactory:
    def capabilities(self):
        from aisimulate.sweeper.replay import RunnerCapabilities
        return RunnerCapabilities(
            supported_backend_topologies=(("sglang", "agg"), ("sglang", "disagg")),
            supports_disaggregated_attention_dp=True,
            supported_trace_formats=("mooncake",),
            supports_agentic_host_offload=False,
            supports_agentic_speculative_decoding=False,
            supported_engine_model_controls=(
                "gemm_quant_mode", "moe_quant_mode", "kvcache_quant_mode",
                "fmha_quant_mode", "comm_quant_mode",
            ),
        )

    def create(self, worker_id):
        return NativeSGLangRunner(worker_id)


def summarize_predictor_provenance(predictors):
    """Expose the executed canonical identity from each replica's saved metrics."""
    identities = []
    for predictor in predictors:
        diagnostics = predictor.get("diagnostics", {})
        provenance = diagnostics.get("provenance", {})
        config = provenance.get("config", {})
        identity = {
            "estimator": provenance.get("selected_estimation_mode"),
            "timing_source": diagnostics.get("source"),
            "data_version": config.get("backend_version"),
            "configuration": config,
            "selected_systems_root": provenance.get("selected_systems_root"),
            "native_sha256": predictor.get("native_sha256"),
            "python_executable": predictor.get("python_executable"),
            "module_path": predictor.get("module_path"),
        }
        if any(not isinstance(identity[key], str) or not identity[key]
               for key in ("estimator", "timing_source", "data_version")):
            return {"agg": {"unavailable_reason": "replica metrics lack canonical predictor provenance"}}
        identities.append(identity)
    if not identities:
        return {"agg": {"unavailable_reason": "no replica predictor metrics were recorded"}}
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("Native replicas used different effective predictor identities")
    return {"agg": {**identities[0], "replicas": [
        {"query_count": predictor.get("query_count"),
         "query_wall_seconds": predictor.get("query_wall_seconds")}
        for predictor in predictors
    ]}}


def compile_case(spec):
    """Lower resolved ReplaySpec to native arguments without new FPM defaults."""
    NativeSGLangRunnerFactory().capabilities().require_compatible(spec)
    deployment = spec.backend_deployment
    if deployment.encoder or deployment.tokenization or spec.adapters:
        raise ValueError("Native runner does not support encoder, tokenization, or adapters")
    if deployment.deployment_mode == "disagg":
        return _compile_pd_case(spec)
    return _compile_worker_case(spec, "agg", deployment.agg_engine_args, deployment.num_workers)


def _compile_worker_case(spec, role, raw, replicas, workload=None):
    from aisimulate.capacity import materialize_aic_num_gpu_blocks
    from aisimulate.runner import _materialize_requests
    from aisimulate_core.sdk.perf_database import load_system_spec

    deployment = spec.backend_deployment
    if not isinstance(raw, dict):
        raise ValueError(f"Missing native {role} engine arguments")
    timing = raw.get("timing_model", {})
    estimator = deployment.forward_pass_estimators.get(role)
    config = dict(estimator.config if estimator is not None else timing.get("config", {}))
    required = ("model", "system", "backend", "backend_version", "tp", "attention_dp", "moe_tp_size", "moe_ep_size", "worker_type")
    if any(key not in config for key in required):
        raise ValueError("Native runner requires the complete canonical forward-pass identity")
    expected_role = "aggregated" if role == "agg" else role
    if config["worker_type"] != expected_role or config.get("pp", 1) != 1:
        raise ValueError(f"Native {role} runner requires worker_type={expected_role} and PP1")
    if config.get("nextn", 0) or config.get("speculation"):
        raise ValueError("Native runner does not support speculative decoding")
    attention_dp = int(config["attention_dp"])
    width = int(config["tp"]) * attention_dp
    if not (
        (config["moe_ep_size"] == 1 and config["moe_tp_size"] == width)
        or (config["moe_ep_size"] == width and config["moe_tp_size"] == 1)
    ):
        raise ValueError("Native runner supports pure expert TP or full EP; hybrid expert parallelism is unqualified")
    if raw.get("startup_time", 0):
        raise ValueError("Native runner has no configured simulated startup cost")
    lowered = materialize_aic_num_gpu_blocks(dict(raw))
    page = int(lowered["block_size"])
    capacity = int(lowered["num_gpu_blocks"]) * page
    if capacity <= 0:
        raise ValueError("Native runner requires positive resolved rank-local KV capacity")
    requests, concurrency = _materialize_requests(spec, page) if workload is None else workload
    # The qualified native scheduler bounds max_req_len by context/capacity-1
    # and then max_new_tokens by max_req_len-input_len-1. Refuse truncation.
    limits = [capacity]
    if raw.get("max_model_len") is not None:
        limits.append(int(raw["max_model_len"]))
    for request in requests:
        required_tokens = request["input_tokens"] + request["output_tokens"] + 2
        if required_tokens > min(limits):
            raise ValueError(
                f"Native SGLang needs {required_tokens} context/KV slots for "
                f"{request['input_tokens']}/{request['output_tokens']} tokens "
                f"including two reserved slots; configured limit is {min(limits)}"
            )
    if not requests or replicas < 1:
        raise ValueError("Native replay requires requests and a positive replica count")
    if replicas > 1:
        # Independent identical workers are equivalent under this explicitly
        # symmetric closed-loop workload. General fleet routing is not claimed.
        shapes = {(r["input_tokens"], r["output_tokens"], r["arrival_time_ms"]) for r in requests}
        if len(shapes) != 1 or next(iter(shapes))[2] != 0 or concurrency is None:
            raise ValueError("Multiple replicas currently require uniform closed-loop requests arriving at zero")
        if concurrency % replicas or len(requests) % replicas:
            raise ValueError("Concurrency and request count must divide evenly over identical replicas")
        if raw.get("enable_prefix_caching"):
            raise ValueError("Symmetric replica replay requires prefix caching disabled")
    roots = config.get("systems_paths")
    system = load_system_spec(config["system"], **({"systems_paths": roots} if roots else {}))
    if not system:
        raise ValueError("Missing canonical hardware specification")
    if width < 2:
        raise ValueError("This runner requires the coordinated native simulator (world size >= 2)")
    scheduler = raw.get("sglang", {})
    recurrent_state = scheduler.get("recurrent_state")
    state_geometry = None
    if recurrent_state is not None:
        from aisimulate.capacity import estimate_state_cache
        from aisimulate.config.engine import RecurrentStateConfig

        recurrent_state = RecurrentStateConfig.model_validate(recurrent_state).model_dump()
        state_geometry = estimate_state_cache(
            config["model"], backend=config["backend"], tp_size=config["tp"],
            pp_size=config["pp"], block_size=page,
            kvcache_quant_mode=config.get("kvcache_quant_mode"),
        )
        if recurrent_state["bytes_per_request"] != state_geometry["bytes_per_request"]:
            raise ValueError("Native recurrent-state bytes differ from the selected model layout")
        if raw.get("enable_prefix_caching"):
            raise ValueError("Native recurrent_state currently requires prefix caching disabled")
    state_args = {}
    if recurrent_state is not None:
        # The simulator's fixed-token configurator bypasses native memory
        # profiling and consumes an already resolved rank-local state size.
        # Its native pool report verifies this mapping after execution.
        state_args = {
            "max_mamba_cache_size": recurrent_state["max_requests"],
            "mamba_ssm_dtype": state_geometry["ssm_dtype"],
        }
    case = {
        "api_version": 1,
        "predictor_config": config,
        "recurrent_state": recurrent_state,
        "recurrent_state_geometry": state_geometry,
        "hardware": {"name": config["system"], "vendor": "ascend" if config["system"].startswith("ascend") else "nvidia", "hbm_capacity_gb": system["gpu"]["mem_capacity"] / 1e9},
        "devices_per_node": system["node"]["num_gpus_per_node"],
        "world_size": width,
        "replicas": replicas,
        "routing": "single_instance" if replicas == 1 else "symmetric_identical_replicas",
        "capacity_source": "explicit_blocks" if "num_gpu_blocks" in raw else "aisimulate_canonical_memory",
        "server_args": {
            "model_path": config["model"],
            "tp_size": width,
            "dp_size": attention_dp,
            "ep_size": config["moe_ep_size"],
            "enable_dp_attention": attention_dp > 1,
            "moe_a2a_backend": "none" if config["moe_ep_size"] == 1 else "deepep",
            "max_total_tokens": capacity,
            "page_size": page,
            "context_length": raw.get("max_model_len"),
            # AISimulate's limit is per rank; SGLang divides its global limit by attention DP.
            "max_running_requests": raw["max_num_seqs"] * attention_dp,
            "chunked_prefill_size": scheduler.get("chunked_prefill_size", raw["max_num_batched_tokens"]),
            "max_prefill_tokens": scheduler.get("max_prefill_tokens", raw["max_num_batched_tokens"]),
            "disable_radix_cache": not raw.get("enable_prefix_caching", False),
            **state_args,
        },
        "requests": requests, "concurrency": concurrency,
    }
    selection_ms = scheduler.get("scheduler_select_time_ms", 0.0)
    from sglang_simulator.simulation.parallel.timing import RequestTimingConfig
    RequestTimingConfig.parse({"scheduler_pre_ms": selection_ms})
    if selection_ms:
        case["request_timing"] = {"scheduler_pre_ms": selection_ms}
        case["server_args"]["disable_overlap_schedule"] = True
    return case


def _compile_pd_case(spec):
    deployment = spec.backend_deployment
    counts = {"prefill": deployment.num_prefill_workers, "decode": deployment.num_decode_workers}
    if any(type(count) is not int or count < 1 for count in counts.values()):
        raise ValueError("Native PD requires positive P/D replica counts")
    if deployment.pd_routing not in (None, "replica_round_robin") or deployment.pd_response not in (None, "direct"):
        raise ValueError("Native PD uses its native bootstrap-room routing and direct Decode response")
    prefill = _compile_worker_case(spec, "prefill", deployment.prefill_engine_args, 1)
    decode = _compile_worker_case(spec, "decode", deployment.decode_engine_args, 1,
                                  workload=(prefill["requests"], prefill["concurrency"]))
    for name in ("model", "system", "tp"):
        if prefill["predictor_config"][name] != decode["predictor_config"][name]:
            raise ValueError(f"Native PD requires matching P/D {name}; state repartition is unqualified")
    if prefill["server_args"]["page_size"] != decode["server_args"]["page_size"]:
        raise ValueError("Native PD requires matching P/D KV page size")
    if any(request["arrival_time_ms"] != 0 for request in prefill["requests"]):
        raise ValueError("Native PD currently implements offline requests arriving at zero")
    for case in (prefill, decode):
        if case["recurrent_state"] is None:
            raise ValueError("Native Qwen PD requires explicit recurrent_state capacity")
    for name in ("bytes_per_request", "kv_bytes_per_token", "ssm_dtype"):
        if prefill["recurrent_state_geometry"][name] != decode["recurrent_state_geometry"][name]:
            raise ValueError(f"Native PD requires matching P/D recurrent-state geometry: {name}")
    bandwidth = deployment.prefill_engine_args.get("kv_transfer_bandwidth")
    if type(bandwidth) not in (int, float) or not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Native PD requires explicit positive kv_transfer_bandwidth in GB/s")
    if deployment.decode_engine_args.get("kv_transfer_bandwidth") != bandwidth:
        raise ValueError("Native P/D transfer bandwidth settings differ")
    for raw in (deployment.prefill_engine_args, deployment.decode_engine_args):
        expected_bytes = prefill["recurrent_state_geometry"]["kv_bytes_per_token"]
        if raw.get("kv_transfer_bytes_per_token", expected_bytes) != expected_bytes:
            raise ValueError("Native PD transfer bytes differ from canonical model geometry")
    case = {**prefill, "mode": "disaggregated", "workers": {"prefill": prefill, "decode": decode},
            "pd_bandwidth_bytes_per_second": bandwidth * 1e9, "routing": "native_bootstrap_room",
            "total_devices": prefill["world_size"] * counts["prefill"] + decode["world_size"] * counts["decode"]}
    if len(case["requests"]) < max(counts.values()):
        raise ValueError("Native PD requires at least one request per cohort")
    cohorts, routes, ranks = [], {}, {}
    for role, count in counts.items():
        dp = case["workers"][role]["predictor_config"]["attention_dp"]
        for replica in range(count):
            worker_id = role if count == 1 else f"{role}-{replica}"
            indices = list(range(replica, len(case["requests"]), count))
            cohorts.append({"worker_id": worker_id, "role": role, "replica": replica,
                            "request_indices": indices})
            for local_index, index in enumerate(indices):
                rid = str(case["requests"][index]["id"])
                routes.setdefault(rid, {})[role] = worker_id
                ranks.setdefault(rid, {})[role] = local_index % dp
    case.update(cohorts=cohorts, request_routes=routes, request_dp_ranks=ranks)
    return case


def pd_cohorts(case):
    """Retain the layout of archived single-pair cases."""
    return case.get("cohorts", [
        {"worker_id": role, "role": role, "replica": 0,
         "request_indices": list(range(len(case["requests"])))}
        for role in ("prefill", "decode")
    ])


def summarize_pd_run(directory, case):
    """Report the Decode client stream and preserve both canonical identities."""
    directory = Path(directory)
    provenance, role_evidence, decode_rows = {}, {}, []
    role_provenance = {"prefill": [], "decode": []}
    for cohort in pd_cohorts(case):
        role, worker_id = cohort["role"], cohort["worker_id"]
        root = directory / worker_id
        worker = case.get("workers", {}).get(role, case)
        metrics = json.loads((root / "metrics.json").read_text())
        if metrics.get("completed") != len(cohort["request_indices"]):
            raise ValueError(f"Native {worker_id} did not finish every assigned request")
        pools = metrics.get("recurrent_state_pools", {})
        if len(pools) != worker["world_size"]:
            raise ValueError(f"Native {role} did not report every state pool")
        expected = worker["recurrent_state"]
        for rank, pool in pools.items():
            if pool["slots"] != expected["max_requests"] or pool["available_slots"] != pool["slots"]:
                raise ValueError(f"Native {role} rank{rank} changed or leaked state slots")
            if pool["bytes_per_request"] != expected["bytes_per_request"]:
                raise ValueError(f"Native {role} rank{rank} changed state geometry")
        forwards = 0
        with (root / "iteration.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("timing_model") != "aisimulate_whole_forward" or row.get("forward_estimate", {}).get("latency_ms") is None:
                    raise ValueError(f"Native {role} did not use the canonical forward predictor")
                forwards += 1
        if forwards == 0:
            raise ValueError(f"Native {role} produced no forwards")
        identity = summarize_predictor_provenance([metrics.get("aisimulate", {})])["agg"]
        if identity.get("configuration", {}).get("worker_type") != role:
            raise ValueError(f"Native {role} canonical cost identity was not recorded")
        role_provenance[role].append(metrics.get("aisimulate", {}))
        runtime_path = root / "runtime.json"
        runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {
            "unavailable_reason": "Historical private replay did not record runtime metadata"
        }
        role_evidence[worker_id] = {"role": role, "replica": cohort["replica"],
                                    "completed_requests": metrics["completed"], "forwards": forwards,
                                    "state_pools": pools, "runtime": runtime}
        if role == "decode":
            origin = metrics.get("request_time_origin_seconds", 0)
            rows = [json.loads(line) for line in (root / "request.jsonl").read_text().splitlines()]
            expected_ids = {str(case["requests"][index]["id"]) for index in cohort["request_indices"]}
            if {row["rid"] for row in rows} != expected_ids:
                raise ValueError(f"Native {worker_id} returned requests owned by another cohort")
            for row in rows:
                row["created_time"] += origin
                row["last_event_time"] += origin
            decode_rows.extend(rows)
    for role, identities in role_provenance.items():
        provenance[role] = summarize_predictor_provenance(identities)["agg"]
    devices = case.get("total_devices", case["world_size"] * 2)
    metrics = summarize_requests(decode_rows, devices=devices, expected=len(case["requests"]))
    admission = json.loads((directory / "admission.json").read_text())
    if admission["active"] or len(admission["completed"]) != len(case["requests"]):
        raise ValueError("Native PD still has unfinished global client requests")
    return {"metrics": metrics, "metadata": {
        "executor": "native_sglang_simulator", "topology": "disagg",
        "routing": "native_bootstrap_room", "hardware_accuracy": "not_validated",
        "prediction_provenance": provenance, "roles": role_evidence,
        "global_peak_concurrency": admission["peak_active"],
        "kv_transfer": {"type": "bandwidth", "bandwidth_bytes_per_second": case["pd_bandwidth_bytes_per_second"],
                        "contention_scope": "source_rank", "payload_storage": "meta", "auxiliary_rows": "cpu_copy"},
    }}


def summarize_requests(rows, *, devices, expected):
    if len(rows) != expected or len({r["rid"] for r in rows}) != expected:
        raise ValueError("Native replay did not complete every distinct request")
    latencies = [r["gen_token_latencies"] for r in rows]
    if any(len(ts) != r["output_length"] or not ts for r, ts in zip(rows, latencies)):
        raise ValueError("Incomplete native token timestamps")
    if any(not math.isfinite(t) or t < 0 for ts in latencies for t in ts):
        raise ValueError("Invalid native token latency")
    duration = max(r["last_event_time"] for r in rows) - min(r["created_time"] for r in rows)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid logical replay duration")
    inputs = sum(r["input_length"] for r in rows)
    outputs = sum(r["output_length"] for r in rows)
    ttft = [ts[0] * 1000 for ts in latencies]
    tpot = [sum(ts[1:]) * 1000 / (len(ts)-1) for ts in latencies if len(ts) > 1]
    e2e = [sum(ts) * 1000 for ts in latencies]
    metrics = {
        "duration_ms": duration*1000, "num_requests": expected,
        "completed_requests": expected, "error_rate": 0.0,
        "total_input_tokens": inputs, "total_output_tokens": outputs,
        "request_throughput_rps": expected/duration,
        "input_throughput_tok_s": inputs/duration,
        "output_throughput_tok_s": outputs/duration,
        "total_throughput_tok_s": (inputs+outputs)/duration,
        "num_total_gpus": devices, "gpu_hours": devices*duration/3600,
    }
    for name, values in (("ttft", ttft), ("tpot", tpot), ("e2e_latency", e2e)):
        metrics[f"num_{name}_samples"] = len(values)
        metrics[f"mean_{name}_ms"] = sum(values)/len(values) if values else 0.0
        ordered = sorted(values)
        for percentile in (50, 90, 95, 99):
            pos = (len(ordered)-1)*percentile/100
            lo, hi = math.floor(pos), math.ceil(pos)
            metrics[f"p{percentile}_{name}_ms"] = (ordered[lo]*(hi-pos) + ordered[hi]*(pos-lo) if lo != hi else ordered[lo]) if ordered else 0.0
    return metrics


class NativeSGLangRunner:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.process = None
        self.pd_run = False

    def run(self, spec, *, output_requirements=None):
        from aisimulate.sweeper.replay import ReplayReport, canonical_json
        case = compile_case(spec)
        if output_requirements and output_requirements.capture_telemetry:
            raise ValueError("Native runner telemetry sampling is not implemented")
        root = os.environ.get("SGLANG_SIMULATOR_RUN_ROOT")
        if root:
            Path(root).mkdir(parents=True, exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix=f"native-w{self.worker_id}-", dir=root))
        (out / "replay-spec.json").write_text(canonical_json(spec))
        (out / "case.json").write_text(json.dumps(case, indent=2))
        runtime = os.environ.get("SGLANG_SIMULATOR_PYTHON", sys.executable)
        source = Path(__file__).resolve().parent.parent
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(source), os.environ.get("SGLANG_SIMULATOR_KERNEL_ROOT"), os.environ.get("PYTHONPATH")]))}
        env["SGLANG_SIMULATOR_PREDICTOR_PYTHON"] = sys.executable
        self.pd_run = case.get("mode") == "disaggregated"
        if self.pd_run:
            env["SGLANG_USE_CPU_ENGINE"] = "1"
        entry = "native_pd_replay.py" if self.pd_run else "native_replay.py"
        timeout = float(os.environ.get("SGLANG_SIMULATOR_RUN_TIMEOUT", "3600" if self.pd_run else "600"))
        with (out / "native.log").open("w") as log:
            self.process = subprocess.Popen([runtime, "-u", str(Path(__file__).with_name(entry)), str(out / "case.json")], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = self.process.wait(timeout=timeout)
                if code:
                    raise RuntimeError(f"Native SGLang child exited {code}; evidence: {out}")
            finally:
                self.close()
        report = json.loads((out / "report.json").read_text())
        return ReplayReport(metrics=report["metrics"], metadata={**report["metadata"], "evidence_directory": str(out)})

    def close(self):
        process, self.process = self.process, None
        if process is None:
            return
        descendants = []
        if self.pd_run and process.poll() is None:
            import psutil

            try:
                descendants = psutil.Process(process.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                pass
        # The child owns its SGLang scheduler/manager descendants, including on
        # timeout. Never leave an orphaned candidate's CPU process tree behind.
        if os.name == "posix":
            try:
                if self.pd_run:
                    if process.poll() is None:
                        # SIGINT lets the paired parent reap its separately
                        # session-owned P/D groups in its finally block.
                        os.kill(process.pid, signal.SIGINT)
                else:
                    os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=75 if self.pd_run else 10)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=10)
        finally:
            if descendants:
                # psutil retains creation times and checks PID reuse before
                # signalling. The snapshot contains only this run's children.
                for child in reversed(descendants):
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(descendants, timeout=5)
                for child in alive:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                psutil.wait_procs(alive, timeout=5)
