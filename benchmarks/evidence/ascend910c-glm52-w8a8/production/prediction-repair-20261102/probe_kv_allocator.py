import argparse
import functools
import hashlib
import inspect
import json
import time
from pathlib import Path

import torch
import torch_npu
import triton.compiler as triton_compiler

from sgl_kernel_npu.mem_cache.allocator import alloc_extend_kernel
from sglang.srt.hardware_backend.npu.allocator_npu import NPUPagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 16, 32, 128, 4096, 8192, 16384, 32768])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(length <= 0 or length > 65536 for length in args.lengths):
        raise ValueError("Probe lengths must fit the isolated metadata pool")
    torch.npu.set_device(0)
    device = "npu:0"
    page_size = 128
    request_pool = ReqToTokenPool(1, 180000, device, False)
    torch.npu.synchronize()
    kernel_calls = []
    original_run = alloc_extend_kernel.run
    compile_calls = []
    original_compile = triton_compiler.compile

    @functools.wraps(original_compile)
    def measured_compile(*arguments, **keywords):
        started = time.perf_counter()
        result = original_compile(*arguments, **keywords)
        compile_calls.append({"host_ms": 1000 * (time.perf_counter() - started)})
        return result

    @functools.wraps(original_run)
    def measured_kernel(*arguments, **keywords):
        started = time.perf_counter()
        result = original_run(*arguments, **keywords)
        kernel_calls.append({"host_ms": 1000 * (time.perf_counter() - started)})
        return result

    alloc_extend_kernel.run = measured_kernel
    triton_compiler.compile = measured_compile
    rows = []
    for length in args.lengths:
        for repetition in range(3):
            allocator = NPUPagedTokenToKVPoolAllocator(65536, page_size, torch.bfloat16, device, None, False)
            prefix_cpu = torch.tensor([0], dtype=torch.int64)
            sequence_cpu = torch.tensor([length], dtype=torch.int64)
            prefix = prefix_cpu.to(device)
            sequence = sequence_cpu.to(device)
            previous = torch.tensor([-1], dtype=torch.int64, device=device)
            torch.npu.synchronize()
            kernel_calls.clear()
            compile_calls.clear()
            started = time.perf_counter()
            locations = allocator.alloc_extend(prefix, prefix_cpu, sequence, sequence_cpu, previous, length)
            host_finished = time.perf_counter()
            torch.npu.synchronize()
            device_finished = time.perf_counter()
            write_started = time.perf_counter()
            request_pool.write((1, slice(0, length)), locations)
            write_host_finished = time.perf_counter()
            torch.npu.synchronize()
            write_device_finished = time.perf_counter()
            expected = torch.arange(page_size, page_size + length, dtype=torch.int32)
            actual = locations.cpu()
            written = request_pool.req_to_token[1, :length].cpu()
            if not torch.equal(actual, expected) or not torch.equal(written, expected):
                raise ValueError(f"Invalid KV indices: length={length} repetition={repetition}")
            rows.append({
                "input_tokens": length,
                "repetition": repetition,
                "new_pages": (length + page_size - 1) // page_size,
                "allocator_host_ms": 1000 * (host_finished - started),
                "allocator_synchronized_ms": 1000 * (device_finished - started),
                "write_host_ms": 1000 * (write_host_finished - write_started),
                "write_synchronized_ms": 1000 * (write_device_finished - write_started),
                "kernel_launch_calls": list(kernel_calls),
                "kernel_compile_calls": list(compile_calls),
                "numerical_check": "exact_contiguous_unique_indices_and_request_pool_write",
            })
            print(json.dumps(rows[-1]), flush=True)
    alloc_extend_kernel.run = original_run
    alloc_extend_kernel.compile = original_compile
    triton_compiler.compile = original_compile
    sources = {}
    for implementation in (NPUPagedTokenToKVPoolAllocator, ReqToTokenPool, original_run, original_compile, alloc_extend_kernel.fn):
        path = Path(inspect.getsourcefile(implementation))
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {
        "scope": "Isolated runtime metadata allocator and request map, no model weights or distributed communication. Synchronizes only this probe to separate host launch and completion. First use is observed in a fresh process; kernel.run includes JIT/load/launch and is not pure device kernel latency. Not an end-to-end timing calibration.",
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "page_size": page_size,
        "allocator_class": "NPUPagedTokenToKVPoolAllocator",
        "sources_sha256": sources,
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
