# SPDX-License-Identifier: Apache-2.0
# Modified for AISimulate multi-accelerator workload replay.
# Upstream attribution is retained in LICENSE and source-manifest.json.
"""Small, process-local trace recorder; never synchronizes the accelerator."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from types import coroutine

PREFIX = "rt-"


def clock_identity():
    path = Path("/proc/sys/kernel/random/boot_id")
    return {
        "host": platform.node(),
        "boot_id": path.read_text().strip() if path.exists() else None,
        "clock": "perf_counter",
    }


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measured(rid):
    return isinstance(rid, str) and rid.startswith(PREFIX)


@dataclass
class ActiveTime:
    cpu: float = 0.0
    wall: float = 0.0


@coroutine
def measure_awaitable(awaitable, active):
    """Measure execution slices, excluding time yielded to the event loop.

    TokenizerManager._handle_batch_output yields sleep(0) after notifying a
    group of requests. Thread CPU over the whole await would also include
    OTHER requests' tokenization, serialization, and output handling.
    """
    iterator = awaitable.__await__()
    value, error = None, None
    try:
        while True:
            wall, cpu = time.perf_counter(), time.thread_time()
            try:
                yielded = (
                    iterator.throw(error) if error is not None else iterator.send(value)
                )
            except StopIteration as stop:
                return stop.value
            finally:
                active.cpu += time.thread_time() - cpu
                active.wall += time.perf_counter() - wall
            try:
                value = yield yielded
                error = None
            except BaseException as exc:
                error = exc
    finally:
        iterator.close()


class Recorder:
    def __init__(self, directory, role):
        self.path = Path(directory) / f"{role}-{os.getpid()}.jsonl"
        self.output = self.path.open("x", buffering=1)
        self.write_cpu = 0.0
        self.write("metadata", role=role, pid=os.getpid(), clock=clock_identity())

    def write(self, kind, **values):
        cpu = time.thread_time()
        self.output.write(json.dumps({"kind": kind, **values}, allow_nan=False) + "\n")
        self.write_cpu += time.thread_time() - cpu

    def span(self, name, start, cpu_start, *, end=None, cpu_end=None, **values):
        # Capture clocks before JSON encoding or file IO. Scheduler service uses
        # CPU time to exclude blocking GPU/IPC waits. Tokenizers additionally
        # use active wall time to include work on native tokenizer threads.
        end = time.perf_counter() if end is None else end
        cpu_end = time.thread_time() if cpu_end is None else cpu_end
        self.write(
            "span",
            name=name,
            start=start,
            end=end,
            cpu_ms=1000 * (cpu_end - cpu_start),
            wall_ms=1000 * (end - start),
            active_ms=values.pop("active_ms", 1000 * (end - start)),
            **values,
        )

    def close(self):
        self.output.close()
