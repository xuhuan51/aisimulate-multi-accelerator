"""Opt-in wrappers around real SGLang methods, including spawned workers.

No simulator/model hooks are installed. Patch after import but before manager
construction, so dispatch tables bind the wrapped methods. A source API mismatch
fails at startup instead of silently returning incomplete calibration data.
"""

from __future__ import annotations

import inspect
import time
from functools import wraps

from .trace import ActiveTime, Recorder, measure_awaitable, measured, sha256


def require_methods(target, names):
    missing = [name for name in names if not callable(getattr(target, name, None))]
    if missing:
        raise RuntimeError(
            f"Request timing collector needs adaptation for {target.__name__}: {missing}"
        )


def provenance(recorder, target):
    path = inspect.getfile(target)
    recorder.write("source", path=path, sha256=sha256(path))


def output_fields(obj):
    return {
        "rids": list(obj.rids),
        "counts": list(obj.completion_tokens),
        "finished": [reason is not None for reason in obj.finished_reasons],
    }


def install_tokenizer(target, recorder):
    require_methods(
        target,
        (
            "generate_request",
            "_tokenize_one_request",
            "_send_one_request",
            "_handle_batch_output",
        ),
    )
    original_generate = target.generate_request
    original_tokenize = target._tokenize_one_request
    original_send = target._send_one_request
    original_output = target._handle_batch_output

    @wraps(original_generate)
    async def generate(self, obj, *args, **kwargs):
        if measured(obj.rid):
            if not isinstance(getattr(obj, "text", None), (str, type(None))):
                raise ValueError("Timing collection accepts one request per HTTP call")
            if obj.contains_mm_input() or not obj.stream:
                raise ValueError("Timing collection requires streaming text generation")
            if getattr(self, "async_dynamic_batch_tokenizer", None) is not None:
                raise ValueError(
                    "Async batched tokenization cannot be timed as per-request service"
                )
            recorder.write("api_received", rid=obj.rid, time=time.perf_counter())
        async for item in original_generate(self, obj, *args, **kwargs):
            yield item

    @wraps(original_tokenize)
    async def tokenize(self, obj, *args, **kwargs):
        if not measured(obj.rid):
            return await original_tokenize(self, obj, *args, **kwargs)
        start, cpu = time.perf_counter(), time.thread_time()
        active = ActiveTime()
        result = await measure_awaitable(
            original_tokenize(self, obj, *args, **kwargs), active
        )
        recorder.span(
            "tokenize",
            start,
            cpu,
            rids=[obj.rid],
            input_tokens=len(result.input_ids),
            input_mode="text" if obj.text is not None else "input_ids",
            cpu_end=cpu + active.cpu,
            active_ms=1000 * active.wall,
        )
        return result

    @wraps(original_send)
    async def send(self, obj, *args, **kwargs):
        if not measured(obj.rid):
            return await original_send(self, obj, *args, **kwargs)
        start, cpu = time.perf_counter(), time.thread_time()
        active = ActiveTime()
        result = await measure_awaitable(
            original_send(self, obj, *args, **kwargs), active
        )
        recorder.span(
            "dispatch",
            start,
            cpu,
            rids=[obj.rid],
            cpu_end=cpu + active.cpu,
            active_ms=1000 * active.wall,
        )
        return result

    @wraps(original_output)
    async def output(self, obj, *args, **kwargs):
        if not any(measured(rid) for rid in obj.rids):
            return await original_output(self, obj, *args, **kwargs)
        fields = output_fields(obj)
        start, cpu = time.perf_counter(), time.thread_time()
        active = ActiveTime()
        result = await measure_awaitable(
            original_output(self, obj, *args, **kwargs), active
        )
        recorder.span(
            "api_output",
            start,
            cpu,
            cpu_end=cpu + active.cpu,
            active_ms=1000 * active.wall,
            **fields,
        )
        return result

    target.generate_request = generate
    target._tokenize_one_request = tokenize
    target._send_one_request = send
    target._handle_batch_output = output


def install_serializer(module, recorder):
    original = module.dumps_json

    @wraps(original)
    def dumps(obj, *args, **kwargs):
        meta = obj.get("meta_info", {}) if isinstance(obj, dict) else {}
        rid = meta.get("id")
        if not measured(rid):
            return original(obj, *args, **kwargs)
        start, cpu = time.perf_counter(), time.thread_time()
        result = original(obj, *args, **kwargs)
        recorder.span(
            "serialize",
            start,
            cpu,
            rids=[rid],
            counts=[meta["completion_tokens"]],
            finished=[meta.get("finish_reason") is not None],
        )
        return result

    module.dumps_json = dumps


def install_scheduler(target, recorder):
    require_methods(
        target,
        ("process_input_requests", "get_next_batch_to_run", "process_batch_result"),
    )
    original_input = target.process_input_requests
    original_select = target.get_next_batch_to_run
    original_post = target.process_batch_result

    @wraps(original_input)
    def receive(self, requests, *args, **kwargs):
        now = time.perf_counter()
        rids = []
        for req in requests:
            entries = req if type(req).__name__.startswith("BatchTokenized") else [req]
            rids.extend(
                entry.rid for entry in entries if measured(getattr(entry, "rid", None))
            )
        if rids:
            recorder.write("scheduler_received", rids=rids, time=now)
        return original_input(self, requests, *args, **kwargs)

    # Include ingestion CPU work, but not collective waiting or idle polling.
    ingestion = (
        "ingest_requests"
        if hasattr(target, "ingest_requests")
        else "process_input_requests"
    )
    target.process_input_requests = receive
    original_ingest = getattr(target, ingestion)

    @wraps(original_ingest)
    def ingest(self, *args, **kwargs):
        write_cpu = recorder.write_cpu
        cpu = time.thread_time()
        result = original_ingest(self, *args, **kwargs)
        self._request_timing_ingest_cpu = max(
            0, time.thread_time() - cpu - (recorder.write_cpu - write_cpu)
        )
        return result

    @wraps(original_select)
    def select(self, *args, **kwargs):
        start, cpu = time.perf_counter(), time.thread_time()
        result = original_select(self, *args, **kwargs)
        end, cpu_end = time.perf_counter(), time.thread_time()
        batch = getattr(result, "batch_to_run", result)
        ingestion_cpu = getattr(self, "_request_timing_ingest_cpu", 0)
        self._request_timing_ingest_cpu = 0
        if batch is not None and any(measured(req.rid) for req in batch.reqs):
            self._request_timing_iteration = (
                getattr(self, "_request_timing_iteration", 0) + 1
            )
            recorder.span(
                "scheduler_pre",
                start,
                cpu - ingestion_cpu,
                end=end,
                cpu_end=cpu_end,
                rids=[req.rid for req in batch.reqs],
                ingestion_cpu_ms=1000 * ingestion_cpu,
                mode=batch.forward_mode.name,
                iteration=self._request_timing_iteration,
            )
        return result

    @wraps(original_post)
    def post(self, batch, *args, **kwargs):
        rids = [req.rid for req in batch.reqs]
        if not any(measured(rid) for rid in rids):
            return original_post(self, batch, *args, **kwargs)
        start, cpu = time.perf_counter(), time.thread_time()
        result = original_post(self, batch, *args, **kwargs)
        recorder.span(
            "scheduler_post",
            start,
            cpu,
            rids=rids,
            mode=batch.forward_mode.name,
            iteration=self._request_timing_iteration,
        )
        return result

    setattr(target, ingestion, ingest)
    target.get_next_batch_to_run = select
    target.process_batch_result = post


def install_detokenizer(target, recorder):
    require_methods(target, ("handle_batch_token_id_out",))
    original = target.handle_batch_token_id_out

    @wraps(original)
    def handle(self, obj, *args, **kwargs):
        if not any(measured(rid) for rid in obj.rids):
            return original(self, obj, *args, **kwargs)
        fields = output_fields(obj)
        start, cpu = time.perf_counter(), time.thread_time()
        result = original(self, obj, *args, **kwargs)
        recorder.span("detokenize", start, cpu, **fields)
        return result

    target.handle_batch_token_id_out = handle


def run_scheduler(*args, trace_dir, **kwargs):
    from sglang.srt.managers.scheduler import Scheduler, run_scheduler_process

    recorder = Recorder(trace_dir, "scheduler")
    bound = inspect.signature(run_scheduler_process).bind_partial(*args, **kwargs)
    recorder.write(
        "rank",
        **{
            name: bound.arguments.get(name)
            for name in ("tp_rank", "pp_rank", "dp_rank")
        },
    )
    provenance(recorder, Scheduler)
    install_scheduler(Scheduler, recorder)
    try:
        return run_scheduler_process(*args, **kwargs)
    finally:
        recorder.close()


def run_detokenizer(*args, trace_dir, **kwargs):
    from sglang.srt.managers.detokenizer_manager import (
        DetokenizerManager,
        run_detokenizer_process,
    )

    recorder = Recorder(trace_dir, "detokenizer")
    provenance(recorder, DetokenizerManager)
    install_detokenizer(DetokenizerManager, recorder)
    try:
        return run_detokenizer_process(*args, **kwargs)
    finally:
        recorder.close()
