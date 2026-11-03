import dataclasses
import itertools
import json
import os
import time
from pathlib import Path

from sglang.benchmark import serving


ORIGINAL = serving.ASYNC_REQUEST_FUNCS["sglang"]
ORDINAL = itertools.count()


async def traced(request_func_input, pbar=None):
    started = time.perf_counter()
    ordinal = next(ORDINAL)
    request_id = f"ingress-{os.getpid()}-{ordinal}"
    request = dataclasses.replace(request_func_input, extra_request_body={
        **request_func_input.extra_request_body, "rid": request_id})
    result = await ORIGINAL(request, pbar)
    row = {"rid": request_id, "ordinal": ordinal, "task_started": started,
           "client_started": result.start_time, "ttft_ms": result.ttft * 1000,
           "latency_ms": result.latency * 1000, "success": result.success,
           "input_tokens": result.prompt_len, "output_tokens": result.output_len}
    destination = Path(serving.args.output_file).with_name("client-dispatch.jsonl")
    with destination.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    return result


serving.ASYNC_REQUEST_FUNCS["sglang"] = traced

if __name__ == "__main__":
    serving.cli_main()
