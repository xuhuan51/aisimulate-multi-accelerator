"""Measure the three source-matched MLA BMM paths with actual replay checks."""

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


def validate_runtime(revision, expected_revision=None):
    audited = {
        "baa9befa381511cd4d56f83980951e5a45fffc6d",
        "76f9213a411018547f4fd6a75f36feaa4d6bed58",
    }
    if revision not in audited or (expected_revision is not None and revision != expected_revision):
        raise ValueError("Reaudit BMM dispatch for the installed framework")


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
    if (
        plan["schema"] != "aisimulate-ascend-bmm-plan/v1"
        or min(protocol.values()) < 1
        or protocol["repeats"] < 3
        or protocol["iterations"] % protocol["weight_copies"]
    ):
        raise ValueError("Invalid BMM plan/protocol")
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

    revision = subprocess.check_output(["git", "-C", "/sgl-workspace/sglang", "rev-parse", "HEAD"], text=True).strip()
    # native76 MLA source uses transpose_batchmatmul at line466 for Q-nope,
    # line551 for Prefill output, and batch_matmul_transpose at line565 for
    # Decode output. Same operand strides and APIs as the original audit.
    validate_runtime(revision, plan.get("framework_commit"))
    init_npu_backend()
    torch.npu.set_device(args.device)
    # Same NPU worker setting as ModelRunner.load_model, native76:1191-1195.
    torch.set_num_threads(1)
    selected = [
        (index, case)
        for index, case in enumerate(plan["cases"])
        if index % args.num_shards == args.shard_index
        and (not args.case_filter or args.case_filter in json.dumps(case, sort_keys=True))
    ]
    if not selected:
        raise ValueError("Empty selected plan")
    source = Path(
        "/sgl-workspace/sglang/python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py"
    )
    record = {
        "schema": "aisimulate-ascend-bmm-observations/v1",
        "status": "running",
        "scope": "Source-matched BF16 MLA strides and kernels; synthetic values; not whole-forward accuracy",
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "collector_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol": protocol,
        "selected_plan_indices": [index for index, _ in selected],
        "cases": [],
        "runtime": {
            "framework_commit": revision,
            "host": socket.gethostname(),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device": torch.npu.get_device_name(args.device),
            "device_index": args.device,
            "driver_version_info": Path("/usr/local/Ascend/driver/version.info").read_text(),
            "serving_setup": setup,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "torch_num_threads": torch.get_num_threads(),
            "dispatch_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(record, indent=2) + "\n")

    def collect(index, case):
        torch.manual_seed(910 + index)
        tokens, heads = case["tokens"], case["local_heads"]
        inner, width = case["input_width"], case["output_width"]
        query = case["role"] == "query_nope"
        storage = torch.randn(tokens, heads, inner + 64 if query else inner, device="npu", dtype=torch.bfloat16)
        operand = storage[..., :inner]
        weights = []
        for _ in range(protocol["weight_copies"]):
            weight = (
                torch.randn(heads, width, inner, device="npu", dtype=torch.bfloat16).transpose(1, 2)
                if query
                else torch.randn(heads, inner, width, device="npu", dtype=torch.bfloat16)
            )
            weight.mul_(inner**-0.5)
            weights.append(weight)
        if (
            list(operand.stride()) != case["input_stride"]
            or any(list(weight.stride()) != case["weight_stride"] for weight in weights)
            or torch_npu.get_npu_format(operand) != case["format"]
            or any(torch_npu.get_npu_format(weight) != case["format"] for weight in weights)
        ):
            raise ValueError("BMM tensor stride/format mismatch")
        custom = case["role"] == "output_decode"
        expected_kernel = "torch.ops.npu.batch_matmul_transpose" if custom else "torch_npu.npu_transpose_batchmatmul"
        if case["kernel_source"] != expected_kernel or case["dtype"] != "bfloat16":
            raise ValueError("BMM kernel/precision mismatch")

        def batch():
            outputs = []
            for iteration in range(protocol["iterations"]):
                weight = weights[iteration % len(weights)]
                if custom:
                    output = torch.empty(tokens, heads, width, device="npu", dtype=torch.bfloat16)
                    torch.ops.npu.batch_matmul_transpose(operand, weight, output)
                else:
                    output = torch_npu.npu_transpose_batchmatmul(
                        operand, weight, perm_x1=(1, 0, 2), perm_x2=(0, 1, 2), perm_y=(1, 0, 2)
                    )
                outputs.append(output)
            return outputs

        for _ in range(protocol["warmup"]):
            batch()
        torch.npu.synchronize()
        graph, outputs = None, None
        if case["execution_mode"] == "graph":
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                outputs = batch()
            for _ in range(protocol["warmup"]):
                graph.replay()
            torch.npu.synchronize()
        elif case["execution_mode"] != "eager":
            raise ValueError("Unsupported execution mode")

        def execute():
            if graph is None:
                return batch()
            graph.replay()
            return outputs

        row_ids = sorted({0, tokens // 2, tokens - 1})
        columns = list(range(0, width, max(1, width // 16)))
        references = [weight[:, :, columns].float().cpu() for weight in weights]

        def verify():
            operand.neg_()
            observed = execute()
            torch.npu.synchronize()
            input_cpu = operand[row_ids].float().cpu().transpose(0, 1)
            maximum_error = 0.0
            for iteration, output in enumerate(observed):
                expected = torch.bmm(input_cpu, references[iteration % len(weights)]).transpose(0, 1)
                actual = output[row_ids][:, :, columns].float().cpu()
                error = (actual - expected).abs()
                if not torch.all(error <= 0.01 + 0.02 * expected.abs()):
                    raise ValueError("BMM execution numerical check failed")
                maximum_error = max(maximum_error, error.max().item())
            return {
                "sampled_elements": len(observed) * len(row_ids) * heads * len(columns),
                "max_abs_error": maximum_error,
            }

        checks = [verify()]
        values, wall = [], []
        for _ in range(protocol["repeats"]):
            begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            started = time.perf_counter()
            begin.record()
            result = execute()
            end.record()
            end.synchronize()
            values.append(begin.elapsed_time(end) / protocol["iterations"])
            wall.append((time.perf_counter() - started) * 1000 / protocol["iterations"])
            del result
        checks.append(verify())
        row = {
            **case,
            "plan_index": index,
            "checks": checks,
            "weight_copies": len(weights),
            "execution_verification": "captured_graph_replay" if graph is not None else "eager_invocation",
            "samples_ms": values,
            "wall_samples_ms": wall,
            "latency_ms": statistics.median(values),
            "cv": statistics.pstdev(values) / statistics.mean(values),
        }
        record["cases"].append(row)
        save()
        print(json.dumps({"plan_index": index, "latency_ms": row["latency_ms"], "cv": row["cv"]}), flush=True)

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
