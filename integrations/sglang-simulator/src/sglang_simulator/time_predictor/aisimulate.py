"""Whole-forward AISimulate estimates, isolated from the SGLang Python environment."""

import atexit
import json
import math
import os
import selectors
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

from sglang_simulator.time_predictor.base import InferTimePredictor


def batch_to_fpm(batch, dp_rank=0):
    """Snapshot native SGLang work before sampling changes the request lengths.

    Decode FPM KV totals include this step's token, matching SGLang's
    schedule-time seq_lens_cpu. Mixed batches need per-request phase tags;
    one-token prefill must not be mistaken for decode.
    """
    prefill, decode, past_features = [], [], []
    for req in batch.reqs:
        phase = req.is_decode
        if phase is None:
            if batch.forward_mode not in {"DECODE", "EXTEND"}:
                raise ValueError(
                    "AISimulate requires request phase tags for mixed batches"
                )
            phase = batch.forward_mode == "DECODE"
        if req.extend_length < 0 or req.past_kv_length < 0:
            raise ValueError("Negative request length in AISimulate input")
        if phase and req.extend_length != 1:
            raise ValueError(
                "AISimulate simulator adapter supports ordinary decode only"
            )
        (decode if phase else prefill).append(req)
        # The public SGLang FPM hook takes prefix_lens for mixed/extend,
        # but seq_lens_cpu (including the current token) for pure decode.
        past_features.append(
            req.past_kv_length + int(phase and batch.forward_mode == "DECODE")
        )
    prompt_lengths = [
        r.prompt_length
        if r.prompt_length is not None
        else r.extend_length + r.past_kv_length
        for r in prefill
    ]
    decode_lengths = [r.past_kv_length + 1 for r in decode]
    return {
        "version": 1,
        "dp_rank": dp_rank,
        "scheduled_requests": {
            "num_prefill_requests": len(prefill),
            "sum_prefill_tokens": sum(r.extend_length for r in prefill),
            "sum_prefill_kv_tokens": sum(r.past_kv_length for r in prefill),
            "var_prefill_length": statistics.pvariance(prompt_lengths)
            if prompt_lengths
            else 0.0,
            "num_decode_requests": len(decode),
            "sum_decode_kv_tokens": sum(decode_lengths),
            "var_decode_kv_tokens": statistics.pvariance(decode_lengths)
            if decode_lengths
            else 0.0,
            "extend_lengths": [r.extend_length for r in batch.reqs],
            # Match AISimulate's SGLang FPM hook's decode feature convention.
            "past_kv_lengths": past_features,
        },
    }


def make_perf_config(model, hardware, scheduler, overrides):
    """Map simulator topology to the canonical AISimulate identity."""
    topology = {
        "tp": scheduler.attn_tp_size,
        "attention_dp": scheduler.dp_size,
        "moe_tp_size": scheduler.moe_tp_size,
        "moe_ep_size": scheduler.ep_size,
        "pp": scheduler.pp_size,
        "worker_type": "aggregated",
        "nextn": 0,
    }
    for key, value in topology.items():
        if key in overrides and overrides[key] != value:
            raise ValueError(
                f"AISimulate {key}={overrides[key]!r} conflicts with {value!r}"
            )
    if overrides.get("speculation") is not None:
        raise ValueError("Speculative decoding is not supported by this simulator")
    config = {
        "model": model.model_path,
        "system": hardware.name,
        "backend": scheduler.backend_name,
        "backend_version": scheduler.backend_version,
        **topology,
    }
    # Leave model quantization inference and estimator selection to AISimulate.
    # KV dtype belongs to the serving configuration, not weight quantization.
    kv_dtype = getattr(scheduler.kv_cache_data_type, "value", None)
    if kv_dtype is not None:
        kv_modes = {
            "BF16": "bfloat16",
            "FP16": "bfloat16",
            "FP8": "fp8",
            "INT8": "int8",
        }
        if kv_dtype not in kv_modes:
            raise ValueError(f"Unsupported AISimulate KV dtype: {kv_dtype}")
        config["kvcache_quant_mode"] = kv_modes[kv_dtype]
    for key, value in (
        ("moe_quant_mode", scheduler.moe_quant_mode_override),
        ("fmha_quant_mode", scheduler.fmha_quant_mode_override),
        ("comm_quant_mode", scheduler.comm_quant_mode_override),
        ("kv_block_size", scheduler.page_size),
    ):
        if value is not None:
            config[key] = value
    config.update(overrides)
    return config


class AISimulateTimePredictor(InferTimePredictor):
    # Predictor IPC/CPU query time is simulation overhead, not target GPU time.
    exclude_cpu_overhead = True

    def __init__(
        self,
        model,
        hw,
        config,
        *,
        python_executable=None,
        timeout_seconds=120,
        perf_config=None,
    ):
        super().__init__(model, hw, config)
        self.timeout_seconds = float(timeout_seconds)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("AISimulate timeout_seconds must be positive and finite")
        self.perf_config = make_perf_config(model, hw, config, perf_config or {})
        self.lock = threading.Lock()
        self.buffer = b""
        self.process = None
        self.selector = selectors.DefaultSelector()
        self.reset_metrics()
        worker = Path(__file__).with_name("aisimulate_worker.py")
        executable = python_executable or sys.executable
        try:
            # -I prevents the simulator's PYTHONPATH/usercustomize hooks from
            # entering the predictor. The chosen interpreter must have AISimulate
            # installed (an editable install is supported).
            self.process = subprocess.Popen(
                [executable, "-I", "-B", "-u", str(worker)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                bufsize=0,
            )
            self.selector.register(self.process.stdout, selectors.EVENT_READ)
            self.metadata = self._call({"op": "init", "config": self.perf_config})
        except Exception:
            self.close()
            raise
        atexit.register(self.close)

    def _call(self, request):
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError("AISimulate predictor worker is not running")
            payload = (json.dumps(request, allow_nan=False) + "\n").encode()
            try:
                # FileIO.write may be short for large per-request FPM payloads.
                view = memoryview(payload)
                while view:
                    count = self.process.stdin.write(view)
                    if not count:
                        raise BrokenPipeError("AISimulate stdin closed")
                    view = view[count:]
                deadline = time.monotonic() + self.timeout_seconds
                while b"\n" not in self.buffer:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not self.selector.select(remaining):
                        raise TimeoutError("AISimulate predictor query timed out")
                    chunk = os.read(self.process.stdout.fileno(), 65536)
                    if not chunk:
                        raise RuntimeError(
                            "AISimulate predictor worker exited without a reply"
                        )
                    self.buffer += chunk
                line, self.buffer = self.buffer.split(b"\n", 1)
                response = json.loads(line)
            except (OSError, TimeoutError, ValueError, RuntimeError):
                self.close()
                raise
            if "error" in response:
                raise RuntimeError(f"AISimulate: {response['error']}")
            return response["result"]

    def estimate_metrics(self, metrics):
        """Use the same persistent estimator for native batches and recorded FPMs."""
        if len(metrics) != self.config.dp_size:
            raise ValueError("Expected one FPM entry per attention DP group")
        if sorted(m.get("dp_rank", 0) for m in metrics) != list(
            range(self.config.dp_size)
        ):
            raise ValueError(
                "AISimulate FPM rank IDs must cover each DP group exactly once"
            )
        start = time.perf_counter()
        result = self._call({"op": "estimate", "metrics": metrics})
        self.query_seconds += time.perf_counter() - start
        self.query_count += 1
        latency = result.get("latency_ms")
        if latency is None:
            raise RuntimeError("AISimulate has no ready prediction for this batch")
        if isinstance(latency, bool) or not math.isfinite(latency) or latency < 0:
            raise ValueError(f"AISimulate returned invalid latency_ms={latency!r}")
        return result

    def predict_parallel(self, batches, *, padded_tokens=None, padding_mode=None):
        metrics = [batch_to_fpm(b, i) for i, b in enumerate(batches)]
        estimate = self.estimate_metrics(metrics)
        # The public API returns whole-forward time. No extra AIC compute or
        # communication term is added. Native padding is recorded, not silently
        # encoded as extra requests in an API without that input contract.
        return {
            "durations": [estimate["latency_ms"] / 1000] * len(batches),
            "timing_model": "aisimulate_whole_forward",
            "forward_metrics": metrics,
            "forward_estimate": estimate,
            "padding_applied_by_adapter": False,
        }

    def predict_infer_time(self, batch):
        return self.predict_parallel([batch])["durations"][0]

    def get_metrics(self):
        return {
            "aisimulate": {
                **self.metadata,
                "query_count": self.query_count,
                "query_wall_seconds": self.query_seconds,
            },
            "cpu_overhead_model": "excluded",
        }

    def reset_metrics(self):
        self.query_count = 0
        self.query_seconds = 0.0

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            process.stdin.close()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            process.stdout.close()
        self.selector.close()
