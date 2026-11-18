"""Measure declared Ascend linear cases, including actual graph output checks.

The pinned ModelSlimW8A8Int8 builder selects the installed static/dynamic
methods. Weight/scale geometry mirrors that builder's create_weights and
process_weights_after_loading. RowParallel rank identity mirrors
linear_method_npu.py:106-121 at baa9befa381511cd4d56f83980951e5a45fffc6d.
BF16 sites use the observed ND weights and torch.nn.functional.linear.
Numerical inputs and weights are seeded controls, not checkpoint values.
"""

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
from types import SimpleNamespace


def validate_runtime(revision, expected_revision=None):
    audited = {
        "baa9befa381511cd4d56f83980951e5a45fffc6d",
        "76f9213a411018547f4fd6a75f36feaa4d6bed58",
    }
    if revision not in audited or (expected_revision is not None and revision != expected_revision):
        raise ValueError("Reaudit linear dispatch for the installed framework")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--serving-setup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--case-filter")
    parser.add_argument("--weight-copies", type=int)
    parser.add_argument("--iterations", type=int)
    args = parser.parse_args()
    if args.output.exists() or not 0 <= args.shard_index < args.num_shards:
        parser.error("Use fresh output and a valid shard index")
    plan = json.loads(args.plan.read_text())
    if plan["schema"] != "aisimulate-ascend-linear-plan/v1":
        raise ValueError("Unsupported plan")
    protocol = dict(plan["protocol"])
    if args.weight_copies is not None:
        protocol["weight_copies"] = args.weight_copies
    if args.iterations is not None:
        protocol["iterations"] = args.iterations
    if min(protocol.values()) < 1 or protocol["repeats"] < 3:
        raise ValueError("Invalid measurement protocol")
    if protocol["iterations"] % protocol["weight_copies"]:
        raise ValueError("Iterations must be a multiple of weight copies")
    setup = json.loads(args.serving_setup.read_text())
    if setup["schema"] != "sglang-runtime-setup/v1":
        raise ValueError("Unsupported serving setup")
    for name, value in setup["environment"].items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    affinity = setup["worker_cpu_affinity"][str(args.device)]
    if not affinity or not set(affinity) <= os.sched_getaffinity(0):
        raise ValueError("Serving CPU mask unavailable")
    os.sched_setaffinity(0, affinity)
    selected = [
        (index, case)
        for index, case in enumerate(plan["cases"])
        if index % args.num_shards == args.shard_index
        and (not args.case_filter or args.case_filter in json.dumps(case, sort_keys=True))
    ]
    if not selected:
        raise ValueError("No cases selected")

    import torch
    import torch_npu

    importlib.import_module("sglang.srt.layers.quantization")
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend
    from sglang.srt.layers.linear import RowParallelLinear
    from sglang.srt.layers.quantization.modelslim.schemes.modelslim_w8a8_int8 import ModelSlimW8A8Int8

    revision = subprocess.check_output(["git", "-C", "/sgl-workspace/sglang", "rev-parse", "HEAD"], text=True).strip()
    validate_runtime(revision, plan.get("framework_commit"))
    init_npu_backend()
    torch.npu.set_device(args.device)
    # ModelRunner.load_model uses one CPU thread on NPU in both audited builds.
    # native76 model_executor/model_runner.py:1191-1195, pinned image
    # sha256:773ac29170d2dc4943f035bff2be2250e9c782555bd16ff72ff9ec891c66dfab.
    torch.set_num_threads(1)
    torch.manual_seed(910)
    result = {
        "schema": "aisimulate-ascend-linear-observations/v1",
        "status": "running",
        "canonical_fpm_data": False,
        "input_values": "Seeded BF16 inputs, INT8[-3,3]/BF16 weights; scales1/8,input1/32,nonzero INT32bias",
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "collector_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol": protocol,
        "selected_plan_indices": [index for index, _ in selected],
        "shard": {"index": args.shard_index, "count": args.num_shards},
        "runtime": {
            "framework_commit": revision,
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device": torch.npu.get_device_name(args.device),
            "device_index": args.device,
            "driver_version_info": Path("/usr/local/Ascend/driver/version.info").read_text(),
            "host": socket.gethostname(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "torch_num_threads": torch.get_num_threads(),
            "serving_setup": setup,
        },
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def prepare(case):
        output_width, input_width = case["n"], case["k"]
        quant = case["quant_mode"]
        if quant not in {"bfloat16", "float32_bfloat16", "w8a8", "w8a8_dynamic"} or (
            case["quant_bias"] and quant != "w8a8"
        ):
            raise ValueError("Unsupported declared precision/bias")
        raw = (
            torch.randn(output_width, input_width, device="npu", dtype=torch.bfloat16) / input_width**0.5
            if quant in {"bfloat16", "float32_bfloat16"}
            else torch.randint(-3, 4, (output_width, input_width), device="npu", dtype=torch.int8)
        )
        bias = ((torch.arange(output_width, device="npu", dtype=torch.int32) % 11) - 5) * 128
        layers = []
        scheme = None
        for _ in range(protocol["weight_copies"]):
            if quant in {"bfloat16", "float32_bfloat16"}:
                layer = SimpleNamespace(weight=raw.clone())
            else:
                layer = RowParallelLinear.__new__(RowParallelLinear)
                torch.nn.Module.__init__(layer)
                layer.tp_group = SimpleNamespace(rank_in_group=0 if case["quant_bias"] else 1)
                # native76 linear_method_npu.py:109-115 reads tp_rank directly.
                # Both attributes describe the same rank for the audited paths.
                layer.tp_rank = layer.tp_group.rank_in_group
                scheme = ModelSlimW8A8Int8({"linear.weight": "W8A8" if quant == "w8a8" else "W8A8_DYNAMIC"}, "linear")
                with torch.device("npu"):
                    scheme.create_weights(layer, input_width, [output_width], input_width, output_width, torch.bfloat16)
                layer.weight.data.copy_(raw)
                layer.weight_scale.data.fill_(0.125)
                layer.weight_offset.data.zero_()
                if quant == "w8a8":
                    layer.input_scale.data.fill_(1 / 32)
                    layer.input_offset.data.zero_()
                    layer.deq_scale.data.fill_(1 / 256)
                    layer.quant_bias.data.copy_(bias)
                scheme.process_weights_after_loading(layer)
            layers.append(layer)
        columns = list(range(0, output_width, max(1, output_width // 32)))
        reference = raw[columns].float().cpu().transpose(0, 1)
        bias_reference = bias[columns].float().cpu() if case["quant_bias"] else None
        kernel_source = (
            type(scheme.kernel).__module__ + "." + type(scheme.kernel).__name__
            if scheme
            else "torch.nn.functional.linear"
        )
        return layers, scheme, columns, reference, bias_reference, kernel_source

    def collect(index, case, prepared):
        layers, scheme, columns, reference, bias_reference, kernel_source = prepared
        tokens, input_width = case["m"], case["k"]
        boundary_dtype = torch.float32 if case["quant_mode"] == "float32_bfloat16" else torch.bfloat16
        inputs = torch.randn(tokens, input_width, device="npu", dtype=boundary_dtype)
        row_indices = sorted({0, tokens // 2, tokens - 1})

        def invoke(layer):
            return scheme.apply_weights(layer, inputs) if scheme else torch.nn.functional.linear(inputs, layer.weight)

        def batch():
            return [invoke(layers[index % len(layers)]) for index in range(protocol["iterations"])]

        for _ in range(protocol["warmup"]):
            batch()
        torch.npu.synchronize()
        graph = None
        captured = None
        if case["execution_mode"] == "graph":
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                captured = batch()
            for _ in range(protocol["warmup"]):
                graph.replay()
            torch.npu.synchronize()
        elif case["execution_mode"] != "eager":
            raise ValueError("Unsupported execution mode")

        def execute():
            if graph is None:
                return batch()
            graph.replay()
            return captured

        def verify():
            inputs.neg_()
            original = inputs[row_indices].float().cpu()
            quant_bin_error, scale_error = 0.0, 0.0
            if case["quant_mode"] in {"bfloat16", "float32_bfloat16"}:
                expected = original @ reference
            else:
                scale = (
                    original.abs().amax(dim=1) / 127
                    if case["quant_mode"] == "w8a8_dynamic"
                    else torch.full((len(row_indices),), 1 / 32)
                )
                quantized = (original / scale[:, None]).round().clamp(-128, 127)
                if case["quant_mode"] == "w8a8_dynamic":
                    actual_quant, actual_scale = torch_npu.npu_dynamic_quant(inputs)
                    quant_sample = actual_quant[row_indices].float().cpu()
                    scale_sample = actual_scale[row_indices].float().cpu()
                    quant_bin_error = (quant_sample - quantized).abs().max().item()
                    scale_error = (scale_sample - scale).abs().max().item()
                    torch.testing.assert_close(scale_sample, scale, rtol=1e-5, atol=1e-7)
                    if quant_bin_error > 1:
                        raise ValueError("Dynamic quantization differs by more than one INT8 bin")
                    quantized, scale = quant_sample, scale_sample
                expected = quantized @ reference
                if bias_reference is not None:
                    expected += bias_reference
                expected *= scale[:, None] * 0.125
            outputs = execute()
            torch.npu.synchronize()
            maximum = 0.0
            for output in outputs:
                if output.dtype != boundary_dtype or list(output.shape) != [tokens, case["n"]]:
                    raise ValueError("Unexpected linear output")
                observed = output[row_indices][:, columns].float().cpu()
                difference = (observed - expected).abs()
                if not torch.all(difference <= 0.02 + 0.012 * expected.abs()):
                    raise ValueError(f"Linear numerical check failed, max error={difference.max().item()}")
                maximum = max(maximum, difference.max().item())
            return {
                "sampled_elements": expected.numel() * len(outputs),
                "max_abs_error": maximum,
                "quant_max_bin_error": quant_bin_error,
                "quant_max_scale_error": scale_error,
            }

        checks = [verify()]
        samples, wall_samples = [], []
        for _ in range(protocol["repeats"]):
            begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            started = time.perf_counter()
            begin.record()
            outputs = execute()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) / protocol["iterations"])
            wall_samples.append((time.perf_counter() - started) * 1000 / protocol["iterations"])
            del outputs
        checks.append(verify())
        row = {
            **case,
            "plan_index": index,
            "weight_format": torch_npu.get_npu_format(layers[0].weight),
            "input_dtype": str(boundary_dtype).removeprefix("torch."),
            "output_dtype": str(boundary_dtype).removeprefix("torch."),
            "kernel_source": kernel_source,
            "execution_verification": "captured_graph_replay" if graph is not None else "eager_invocation",
            "checks": checks,
            "samples_ms": samples,
            "wall_samples_ms": wall_samples,
            "latency_ms": statistics.median(samples),
            "cv": statistics.pstdev(samples) / statistics.mean(samples),
            "weight_copies": len(layers),
            "logical_weight_working_set_bytes": sum(
                layer.weight.numel() * layer.weight.element_size() for layer in layers
            ),
        }
        result["cases"].append(row)
        save()
        print(json.dumps({"plan_index": index, "latency_ms": row["latency_ms"], "cv": row["cv"]}), flush=True)

    save()
    current, prepared = None, None
    try:
        for index, case in selected:
            geometry = tuple(case[field] for field in ("quant_mode", "n", "k", "quant_bias"))
            if geometry != current:
                prepared = None
                gc.collect()
                torch.npu.empty_cache()
                prepared, current = prepare(case), geometry
            collect(index, case, prepared)
        result["status"] = "completed"
    except Exception:
        result["status"] = "failed"
        result["failed_plan_index"] = index
        result["error"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
