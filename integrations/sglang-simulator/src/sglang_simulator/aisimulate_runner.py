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
            supported_backend_topologies=(("sglang", "agg"),),
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


def compile_case(spec):
    """Lower resolved ReplaySpec to native arguments without new FPM defaults."""
    from aisimulate.capacity import materialize_aic_num_gpu_blocks
    from aisimulate.runner import _materialize_requests
    from aisimulate_core.sdk.perf_database import load_system_spec

    NativeSGLangRunnerFactory().capabilities().require_compatible(spec)
    deployment = spec.backend_deployment
    if deployment.deployment_mode != "agg":
        raise ValueError("Native SGLang PD handoff adapter is not implemented yet")
    if deployment.encoder or deployment.tokenization or spec.adapters:
        raise ValueError("Native runner does not support encoder, tokenization, or adapters")
    raw = deployment.agg_engine_args
    if not isinstance(raw, dict):
        raise ValueError("Missing aggregated engine arguments")
    timing = raw.get("timing_model", {})
    estimator = deployment.forward_pass_estimators.get("agg")
    config = dict(estimator.config if estimator is not None else timing.get("config", {}))
    required = ("model", "system", "backend", "backend_version", "tp", "attention_dp", "moe_tp_size", "moe_ep_size", "worker_type")
    if any(key not in config for key in required):
        raise ValueError("Native runner requires the complete canonical forward-pass identity")
    if config["worker_type"] != "aggregated" or config.get("pp", 1) != 1:
        raise ValueError("Native aggregated runner requires worker_type=aggregated and PP1")
    if config.get("nextn", 0) or config.get("speculation"):
        raise ValueError("Native runner does not support speculative decoding")
    width = int(config["tp"]) * int(config["attention_dp"])
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
    requests, concurrency = _materialize_requests(spec, page)
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
    replicas = deployment.num_workers
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
    return {
        "api_version": 1,
        "predictor_config": config,
        "hardware": {"name": config["system"], "vendor": "ascend" if config["system"].startswith("ascend") else "nvidia", "hbm_capacity_gb": system["gpu"]["mem_capacity"] / 1e9},
        "devices_per_node": system["node"]["num_gpus_per_node"],
        "world_size": width,
        "replicas": replicas,
        "routing": "single_instance" if replicas == 1 else "symmetric_identical_replicas",
        "capacity_source": "explicit_blocks" if "num_gpu_blocks" in raw else "aisimulate_canonical_memory",
        "server_args": {
            "model_path": config["model"], "tp_size": width,
            "dp_size": config["attention_dp"], "ep_size": config["moe_ep_size"],
            "enable_dp_attention": config["attention_dp"] > 1,
            "moe_a2a_backend": "none" if config["moe_ep_size"] == 1 else "deepep",
            "max_total_tokens": capacity, "page_size": page,
            "context_length": raw.get("max_model_len"),
            "max_running_requests": raw["max_num_seqs"],
            "chunked_prefill_size": scheduler.get("chunked_prefill_size", raw["max_num_batched_tokens"]),
            "max_prefill_tokens": scheduler.get("max_prefill_tokens", raw["max_num_batched_tokens"]),
            "disable_radix_cache": not raw.get("enable_prefix_caching", False),
        },
        "requests": requests, "concurrency": concurrency,
    }


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
        timeout = float(os.environ.get("SGLANG_SIMULATOR_RUN_TIMEOUT", "600"))
        with (out / "native.log").open("w") as log:
            self.process = subprocess.Popen([runtime, "-u", str(Path(__file__).with_name("native_replay.py")), str(out / "case.json")], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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
        # The child owns its SGLang scheduler/manager descendants, including on
        # timeout. Never leave an orphaned candidate's CPU process tree behind.
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=10)
