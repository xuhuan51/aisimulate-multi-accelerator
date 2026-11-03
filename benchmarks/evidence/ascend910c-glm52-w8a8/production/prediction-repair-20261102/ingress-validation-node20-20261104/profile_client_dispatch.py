import argparse
import asyncio
import contextvars
import hashlib
import inspect
import json
import multiprocessing
import os
import random
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import aiohttp
from aiohttp import web
from sglang.benchmark import serving


CURRENT = contextvars.ContextVar("aisim_request")


def serve(connection):
    async def run():
        async def generate(request):
            body = await request.json()
            message = {"text": "a", "meta_info": {"completion_tokens": 1,
                       "prompt_tokens": len(body["input_ids"])}}
            return web.Response(text="data: " + json.dumps(message) + "\n\ndata: [DONE]\n\n",
                                content_type="text/event-stream")

        app = web.Application(client_max_size=16 * 1024 * 1024)
        app.router.add_post("/generate", generate)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        connection.send(site._server.sockets[0].getsockname()[1])
        await asyncio.Event().wait()

    asyncio.run(run())


async def measure(port, repeats):
    original_session = serving._create_bench_client_session
    original_serialize = json.dumps

    def session():
        client = original_session()

        def serialize(value):
            started = time.perf_counter_ns()
            result = original_serialize(value)
            CURRENT.get()["json_cpu_ms"] = (time.perf_counter_ns() - started) / 1e6
            CURRENT.get()["payload_bytes"] = len(result.encode())
            return result

        client._json_serialize = serialize
        return client

    serving._create_bench_client_session = session
    serving.args = SimpleNamespace(temperature=0.0, disable_ignore_eos=False, top_p=1.0,
                                   disable_stream=False, return_logprob=False,
                                   return_routed_experts=False, logprob_start_len=-1,
                                   top_logprobs_num=0, token_ids_logprob=None, header=None)
    generator = random.Random(974)
    cases = []
    for length in (512, 2048, 4096, 8192, 12288, 16384, 32768):
        prompt = [generator.randrange(1000, 150000) for _ in range(length)]
        for population in (4, 8, 16):
            for iteration in range(repeats + 2):
                records = []

                async def issue(ordinal):
                    record = {"ordinal": ordinal, "task_start_ns": time.perf_counter_ns()}
                    CURRENT.set(record)
                    request = serving.RequestFuncInput(
                        prompt=prompt, api_url=f"http://127.0.0.1:{port}/generate", prompt_len=length,
                        output_len=1, model="glm-5", lora_name=None, image_data=None,
                        extra_request_body={})
                    result = await serving.async_request_sglang_generate(request)
                    if not result.success or result.prompt_len != length or result.output_len != 1:
                        raise ValueError(f"Mock endpoint failed: {result.error}")
                    record["client_start_ms"] = result.start_time * 1000
                    records.append(record)

                await asyncio.gather(*(issue(ordinal) for ordinal in range(population)))
                records.sort(key=lambda value: value["ordinal"])
                gaps = [(after["task_start_ns"] - before["task_start_ns"]) / 1e6
                        for before, after in zip(records, records[1:])]
                cases.append({"input_tokens": length, "concurrency": population,
                              "iteration": iteration, "warmup": iteration < 2,
                              "dispatch_intervals_ms": gaps, "requests": records})
    serving._create_bench_client_session = original_session
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=serve, args=(child,))
    process.start()
    try:
        if not parent.poll(60):
            raise TimeoutError("Mock endpoint did not start")
        cases = asyncio.run(measure(parent.recv(), args.repeats))
    finally:
        process.terminate()
        process.join(10)
    summaries = []
    for length in sorted({case["input_tokens"] for case in cases}):
        selected = [case for case in cases if case["input_tokens"] == length and not case["warmup"]]
        gaps = [gap for case in selected for gap in case["dispatch_intervals_ms"]]
        summaries.append({"input_tokens": length, "median_dispatch_interval_ms": statistics.median(gaps),
                          "mean_dispatch_interval_ms": statistics.mean(gaps),
                          "median_json_cpu_ms": statistics.median(
                              request["json_cpu_ms"] for case in selected for request in case["requests"])})
    source = Path(inspect.getfile(serving))
    result = {"schema": "aisimulate-client-dispatch-profile/v1", "status": "completed",
              "scope": "CPU-only original SGLang benchmark client against a separate mock HTTP process; no inference timings or serving holdout arrivals used",
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "source_path": str(source),
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "summaries": summaries, "cases": cases}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
