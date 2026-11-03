import dataclasses
import hashlib
import inspect
import itertools
import os
from pathlib import Path

from sglang.benchmark import serving


def main():
    source = Path(inspect.getfile(serving))
    if hashlib.sha256(source.read_bytes()).hexdigest() != "365c9a348c16a130a0ab62b8d6141b82dd56cc8542b6baf94cc92d120709cd35":
        raise RuntimeError("Benchmark source differs from the pinned runtime")
    counts = [int(value) for value in os.environ["AISIM_ROUTED_COUNTS"].split(",")]
    attention_dp = int(os.environ.get("AISIM_ATTENTION_DP", "4"))
    if attention_dp not in (4, 32) or len(counts) != attention_dp or not any(counts) or any(value < 0 for value in counts):
        raise ValueError("Expected one nonnegative request count per audited attention DP rank")
    routing = itertools.cycle([rank for row in range(max(counts)) for rank, count in enumerate(counts) if row < count])
    original = serving.ASYNC_REQUEST_FUNCS["sglang"]

    async def routed(request_func_input, pbar=None):
        body = {**request_func_input.extra_request_body, "routed_dp_rank": next(routing)}
        request = dataclasses.replace(request_func_input, extra_request_body=body)
        return await original(request, pbar)

    serving.ASYNC_REQUEST_FUNCS["sglang"] = routed
    serving.cli_main()


if __name__ == "__main__":
    main()
