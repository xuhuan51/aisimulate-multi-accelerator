import dataclasses
import json
import time
import uuid
from pathlib import Path

from sglang.benchmark import serving


ORIGINAL = serving.ASYNC_REQUEST_FUNCS["sglang"]
TRACE = Path("/campaign") / f"client-requests-{time.time_ns()}.jsonl"


async def traced_request(request_func_input, pbar=None):
    request_id = f"aisim-{uuid.uuid4().hex}"
    request = dataclasses.replace(
        request_func_input,
        extra_request_body={**request_func_input.extra_request_body, "rid": request_id},
    )
    started_perf = time.perf_counter()
    started_unix = time.time()
    result = await ORIGINAL(request, pbar=pbar)
    ended_perf = time.perf_counter()
    ended_unix = time.time()
    record = {
        "rid": request_id,
        "requested_input_tokens": request.prompt_len,
        "requested_output_tokens": request.output_len,
        "wrapper_start_perf": started_perf,
        "wrapper_start_unix": started_unix,
        "wrapper_end_perf": ended_perf,
        "wrapper_end_unix": ended_unix,
        "request_start_perf": result.start_time,
        "request_start_unix": started_unix + result.start_time - started_perf,
        "ttft_seconds": result.ttft,
        "latency_seconds": result.latency,
        "itl_seconds": result.itl,
        "input_tokens": result.prompt_len,
        "output_tokens": result.output_len,
        "success": result.success,
        "error": result.error,
    }
    with TRACE.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    return result


serving.ASYNC_REQUEST_FUNCS["sglang"] = traced_request


if __name__ == "__main__":
    serving.cli_main()
