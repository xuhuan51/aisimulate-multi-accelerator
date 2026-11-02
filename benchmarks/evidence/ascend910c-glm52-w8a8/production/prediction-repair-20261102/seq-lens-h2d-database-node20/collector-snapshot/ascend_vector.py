"""Measure declared vector boundaries with actual eager and captured replay checks."""

import argparse
import gc
import hashlib
import importlib
import json
import os
import socket
import statistics
import subprocess
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--serving-setup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--case-filter")
    args = parser.parse_args()
    if args.output.exists() or not 0 <= args.shard_index < args.num_shards:
        parser.error("Use fresh output and a valid shard index")
    plan = json.loads(args.plan.read_text())
    protocol = plan["protocol"]
    if plan["schema"] != "aisimulate-ascend-vector-plan/v1" or min(protocol.values()) < 1 or protocol["repeats"] < 3:
        raise ValueError("Invalid vector plan/protocol")
    setup = json.loads(args.serving_setup.read_text())
    for name, value in setup["environment"].items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    os.sched_setaffinity(0, setup["worker_cpu_affinity"][str(args.device)])
    import torch
    import torch_npu

    importlib.import_module("sglang.srt.layers.quantization")
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend

    if __package__:
        from .ascend_vector_kernels import prepare
    else:
        from ascend_vector_kernels import prepare
    revision = subprocess.check_output(["git", "-C", "/sgl-workspace/sglang", "rev-parse", "HEAD"], text=True).strip()
    if revision != "baa9befa381511cd4d56f83980951e5a45fffc6d":
        raise ValueError("Reaudit vector dispatch for this runtime")
    init_npu_backend()
    torch.npu.set_device(args.device)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    filtered = [
        (index, case)
        for index, case in enumerate(plan["cases"])
        if not args.case_filter or args.case_filter in json.dumps(case, sort_keys=True)
    ]
    selected = [entry for position, entry in enumerate(filtered) if position % args.num_shards == args.shard_index]
    if not selected:
        raise ValueError("Empty selected plan")
    sources = {"collector": Path(__file__), "kernels": Path(__file__).with_name("ascend_vector_kernels.py")}
    record = {
        "schema": "aisimulate-ascend-vector-observations/v1",
        "status": "running",
        "scope": (
            "Source-matched standalone vector boundaries; synthetic values and warm buffers; not whole-forward accuracy"
        ),
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "collector_source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()
        },
        "protocol": protocol,
        "selected_plan_indices": [index for index, _ in selected],
        "cases": [],
        "runtime": {
            "framework_commit": revision,
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device": torch.npu.get_device_name(args.device),
            "host": socket.gethostname(),
            "device_index": args.device,
            "driver_version_info": Path("/usr/local/Ascend/driver/version.info").read_text(),
            "serving_setup": setup,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(record, indent=2) + "\n")

    def describe(value):
        if isinstance(value, torch.Tensor):
            return {
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "storage_offset": value.storage_offset(),
                "dtype": str(value.dtype),
                "format": torch_npu.get_npu_format(value) if value.device.type != "cpu" else None,
                **({"device": "cpu", "pinned": value.is_pinned()} if value.device.type == "cpu" else {}),
            }
        if isinstance(value, (list, tuple)):
            return [describe(item) for item in value]
        return value

    def collect(index, case):
        torch.manual_seed(910 + index)
        prepared = prepare({**case, "iterations": protocol["iterations"]}, torch, torch_npu)
        input_signature = {name: describe(value) for name, value in prepared.inputs.items()}

        def batch():
            return [prepared.invoke() for _ in range(protocol["iterations"])]

        for _ in range(protocol["warmup"]):
            prepared.reset()
            batch()
        torch.npu.synchronize()
        graph, outputs = None, None
        if case["execution_mode"] == "graph":
            prepared.reset()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                outputs = batch()
            for _ in range(protocol["warmup"]):
                prepared.reset()
                graph.replay()
            torch.npu.synchronize()
        elif case["execution_mode"] != "eager":
            raise ValueError("Unsupported vector execution mode")

        def execute():
            if graph is None:
                return batch()
            graph.replay()
            return outputs

        def verify():
            prepared.perturb()
            torch.npu.synchronize()
            expected = prepared.reference()
            observed = execute()
            torch.npu.synchronize()
            checked = observed[-1:] if prepared.stateful or case["role"] == "cache_scatter" else observed
            count, maximum = 0, 0.0
            for output in checked:
                elements, error = prepared.check(output, expected)
                count += elements
                maximum = max(maximum, error)
            return {"sampled_elements": count, "max_abs_error": maximum}, describe(observed[-1])

        first, output_signature = verify()
        values, wall = [], []
        for _ in range(protocol["repeats"]):
            prepared.reset()
            torch.npu.synchronize()
            begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            started = time.perf_counter()
            begin.record()
            result = execute()
            end.record()
            end.synchronize()
            values.append(begin.elapsed_time(end) / protocol["iterations"])
            wall.append((time.perf_counter() - started) * 1000 / protocol["iterations"])
            del result
        second, after_signature = verify()
        if after_signature != output_signature:
            raise ValueError("Vector output signature changed during timing")
        row = {
            **case,
            "plan_index": index,
            "kernel_source": prepared.kernel_source,
            "input_signature": input_signature,
            "output_signature": output_signature,
            "checks": [first, second],
            "execution_verification": "captured_graph_replay" if graph is not None else "eager_invocation",
            "samples_ms": values,
            "wall_samples_ms": wall,
            "latency_ms": statistics.median(values),
            "cv": statistics.pstdev(values) / statistics.mean(values),
        }
        record["cases"].append(row)
        save()
        print(
            json.dumps({"plan_index": index, "role": case["role"], "latency_ms": row["latency_ms"], "cv": row["cv"]}),
            flush=True,
        )

    save()
    try:
        for index, case in selected:
            collect(index, case)
            gc.collect()
            torch.npu.empty_cache()
        record["status"] = "completed"
    except Exception:
        record.update(status="failed", failed_plan_index=index, error=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
