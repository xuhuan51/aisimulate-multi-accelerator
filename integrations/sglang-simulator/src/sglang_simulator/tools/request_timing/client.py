"""Closed-loop streaming client with stable request IDs and monotonic clocks."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid

from .trace import clock_identity


class SSEDecoder:
    """HTTP byte chunks need not align with SSE lines or events."""

    def __init__(self):
        self.pending = b""
        self.data = []

    def feed(self, chunk):
        self.pending += chunk
        events = []
        while b"\n" in self.pending:
            line, self.pending = self.pending.split(b"\n", 1)
            line = line.rstrip(b"\r")
            if not line:
                if self.data:
                    events.append(b"\n".join(self.data).decode("utf-8"))
                    self.data = []
            elif line.startswith(b"data:"):
                self.data.append(line[5:].lstrip(b" "))
        return events


def load_workload(args):
    if args.requests_jsonl:
        rows = [
            json.loads(line)
            for line in args.requests_jsonl.read_text().splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError("Empty request workload")
    else:
        sizes = [int(size) for size in args.prompt_words.split(",")]
        if not sizes or min(sizes) < 1:
            raise ValueError("--prompt-words must contain positive integers")
        rows = [
            {"text": f"Calibration request {i}. " + "hello " * sizes[i % len(sizes)]}
            for i in range(args.num_requests)
        ]
    inputs = []
    modes = set()
    for row in rows:
        if set(row) - {"text", "input_ids", "sampling_params"}:
            raise ValueError(
                "Workload rows accept only text or input_ids, and sampling_params"
            )
        if ("text" in row) == ("input_ids" in row):
            raise ValueError("Each workload row needs exactly one of text/input_ids")
        if "text" in row and (not isinstance(row["text"], str) or not row["text"]):
            raise ValueError("text must be a nonempty string")
        if "input_ids" in row and (
            not isinstance(row["input_ids"], list)
            or not row["input_ids"]
            or any(type(token) is not int or token < 0 for token in row["input_ids"])
        ):
            raise ValueError(
                "input_ids must be a nonempty list of nonnegative integers"
            )
        modes.add("text" if "text" in row else "input_ids")
        params = {
            "temperature": 0,
            "ignore_eos": True,
            "max_new_tokens": args.output_tokens,
            **row.get("sampling_params", {}),
        }
        if type(params["max_new_tokens"]) is not int or params["max_new_tokens"] < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if params.get("n", 1) != 1:
            raise ValueError("Timing calibration requires n=1")
        inputs.append({**row, "sampling_params": params, "stream": True})
    if len(modes) != 1:
        raise ValueError("Collect text and input_ids workloads in separate runs")
    return inputs


async def send_request(session, base_url, payload, rid):
    # Build the body before timing; don't charge local payload construction to
    # the transport estimate. The request ID is echoed in native meta_info.id.
    body = json.dumps({**payload, "rid": rid})
    row = {"rid": rid, "start": time.perf_counter(), "chunks": [], "success": False}
    count, done, finished_reason = 0, False, False
    try:
        async with session.post(
            base_url.rstrip("/") + "/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        ) as response:
            response.raise_for_status()
            decoder = SSEDecoder()
            async for raw in response.content.iter_any():
                arrived = time.perf_counter()
                for data in decoder.feed(raw):
                    if data == "[DONE]":
                        done = True
                        continue
                    item = json.loads(data)
                    if "error" in item:
                        raise ValueError(f"Server error: {item['error']}")
                    meta = item["meta_info"]
                    if meta["id"] != rid:
                        raise ValueError("Response request ID mismatch")
                    current = meta["completion_tokens"]
                    if type(current) is not int or current < count:
                        raise ValueError("Invalid/non-monotonic response token count")
                    if current > count:
                        row["chunks"].append({"count": current, "time": arrived})
                        count = current
                    reason = meta.get("finish_reason")
                    if isinstance(reason, dict) and reason.get("type") in (
                        "abort",
                        "error",
                    ):
                        raise ValueError(f"Unsuccessful finish reason: {reason}")
                    finished_reason |= reason is not None
            if not done or not finished_reason or not count:
                raise ValueError(
                    "Incomplete SSE response: missing final marker, finish reason or tokens"
                )
        row["success"] = True
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row.update(finished=time.perf_counter(), output_tokens=count)
    return row


async def collect(args):
    import aiohttp

    inputs = load_workload(args)
    if args.output.exists():
        raise FileExistsError(args.output)
    run_id = uuid.uuid4().hex
    output = {
        "schema_version": 1,
        "clock": clock_identity(),
        "run_id": run_id,
        "base_url": args.base_url,
        "concurrency": args.concurrency,
        "think_ms": args.think_ms,
        "warmup_requests": args.warmup_requests,
        "workload_sha256": hashlib.sha256(
            json.dumps(inputs, sort_keys=True).encode()
        ).hexdigest(),
        "workload_source": str(args.requests_jsonl.resolve())
        if args.requests_jsonl
        else "generated_text",
        "requests": [],
    }
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for index in range(args.warmup_requests):
            row = await send_request(
                session,
                args.base_url,
                inputs[index % len(inputs)],
                f"warmup-{run_id}-{index}",
            )
            if not row["success"]:
                raise RuntimeError(f"Warmup failed: {row.get('error')}")
        pending = iter(enumerate(inputs))

        async def worker(slot):
            previous = None
            for index, payload in pending:
                if previous and args.think_ms:
                    await asyncio.sleep(args.think_ms / 1000)
                row = await send_request(
                    session, args.base_url, payload, f"rt-{run_id}-{index}"
                )
                row.update(index=index, slot=slot, previous_rid=previous)
                output["requests"].append(row)
                previous = row["rid"]

        await asyncio.gather(*(worker(slot) for slot in range(args.concurrency)))
    output["requests"].sort(key=lambda row: row["index"])
    output["successful_requests"] = sum(row["success"] for row in output["requests"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(output, file, indent=2, allow_nan=False)
        file.write("\n")
    print(
        f"Saved {output['successful_requests']}/{len(inputs)} successful requests to {args.output}"
    )
    if output["successful_requests"] != len(inputs):
        raise RuntimeError(
            "Some requests failed; see client output. No calibration should use this run."
        )
