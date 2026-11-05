"""Measure the installed Ascend tensor-MoE chain with independent numerical checks.

SGLang baa9befa381511cd4d56f83980951e5a45fffc6d selects AscendTPDispatcher
for the NPU none backend: INT8 init-routing-v2, grouped matmul, NPUSwigluQuant,
grouped matmul, and NPUFinalizeRouting(drop_pad_mode=2). This collector calls
those installed components; it does not copy their implementations.
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
        parser.error("Use a fresh output and a valid shard")
    plan = json.loads(args.plan.read_text())
    protocol = plan["protocol"]
    if (
        plan["schema"] != "aisimulate-ascend-moe-tensor-plan/v1"
        or min(protocol.values()) < 1
        or protocol["repeats"] < 3
        or protocol["iterations"] % protocol["weight_copies"]
    ):
        raise ValueError("Invalid tensor-MoE plan or timing protocol")
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
    from sglang.srt.hardware_backend.npu.moe.activation import NPUSwigluQuant
    from sglang.srt.hardware_backend.npu.moe.finalize_routing import NPUFinalizeRouting
    from sglang.srt.hardware_backend.npu.moe.init_routing import NPUMoEInitRouting_v2
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUW8A8Int8MoEMethod
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend, npu_format_cast

    revision = subprocess.check_output(["git", "-C", "/sgl-workspace/sglang", "rev-parse", "HEAD"], text=True).strip()
    if revision != "baa9befa381511cd4d56f83980951e5a45fffc6d":
        raise ValueError("Reaudit tensor-MoE dispatch for a different runtime")
    init_npu_backend()
    torch.npu.set_device(args.device)
    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    routing, activation = NPUMoEInitRouting_v2(quant_mode=1), NPUSwigluQuant()
    matmul, finalizer = NPUW8A8Int8MoEMethod(), NPUFinalizeRouting(drop_pad_mode=2)
    selected = [
        (index, case)
        for index, case in enumerate(plan["cases"])
        if not args.case_filter or args.case_filter in json.dumps(case, sort_keys=True)
    ]
    selected = [entry for position, entry in enumerate(selected) if position % args.num_shards == args.shard_index]
    if not selected:
        raise ValueError("Empty selected tensor-MoE plan")
    result = {
        "schema": "aisimulate-ascend-moe-tensor-observations/v1",
        "status": "running",
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "collector_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol": protocol,
        "selected_plan_indices": [index for index, _ in selected],
        "scope": plan["scope"],
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
        },
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def collect(index, case):
        hidden, experts = case["hidden_size"], case["num_experts"]
        tokens, top_k, intermediate = case["tokens"], case["top_k"], case["intermediate_size"]
        if (
            (hidden, experts, top_k) != (6144, 256, 8)
            or intermediate not in (64, 128, 256, 512)
            or case["distribution"] not in ("balanced", "balanced_subset")
            or tokens < 1
        ):
            raise ValueError("Unqualified tensor-MoE geometry")
        active_experts = experts if case["distribution"] == "balanced" else case["active_experts"]
        if not isinstance(active_experts, int) or not top_k <= active_experts <= experts:
            raise ValueError("Active expert count must be between top-k and the model expert count")
        if case["distribution"] == "balanced_subset" and active_experts > tokens * top_k:
            raise ValueError("The requested active expert count exceeds the number of assignments")
        torch.manual_seed(case.get("random_seed", 910 + index))
        values = (torch.randint(-31, 32, (tokens, hidden), device="npu").float() / 32).bfloat16()
        ids_cpu = torch.arange(tokens * top_k, dtype=torch.int32).reshape(tokens, top_k) % active_experts
        ids = ids_cpu.npu()
        probabilities = torch.full((tokens, top_k), 1 / top_k, device="npu", dtype=torch.bfloat16)
        raw_up = torch.randint(-3, 4, (experts, hidden, intermediate * 2), device="npu", dtype=torch.int8)
        raw_down = torch.randint(-3, 4, (experts, intermediate, hidden), device="npu", dtype=torch.int8)
        weights = [
            SimpleNamespace(
                w13_weight=npu_format_cast(raw_up.clone()),
                w2_weight=npu_format_cast(raw_down.clone()),
                w13_weight_scale=torch.full((experts, intermediate * 2), 0.125, device="npu", dtype=torch.bfloat16),
                w2_weight_scale=torch.full((experts, hidden), 0.125, device="npu", dtype=torch.bfloat16),
            )
            for _ in range(protocol["weight_copies"])
        ]
        if any(
            torch_npu.get_npu_format(weight.w13_weight) != 29 or torch_npu.get_npu_format(weight.w2_weight) != 29
            for weight in weights
        ):
            raise ValueError("Tensor-MoE weights must use the serving NZ format")
        columns = list(range(0, hidden, hidden // 32))
        selected_tokens = sorted({0, tokens // 2, tokens - 1})
        references = []
        inputs_cpu = values[selected_tokens].float().cpu()
        for position, token in enumerate(selected_tokens):
            for expert in ids_cpu[token].tolist():
                references.append(
                    (
                        token,
                        expert,
                        inputs_cpu[position],
                        raw_up[expert].float().cpu(),
                        raw_down[expert][:, columns].float().cpu(),
                    )
                )
        del raw_up, raw_down

        def invoke(weight):
            routed, row_mapping, counts, input_scales = routing._init_routing(values, ids, experts, top_k)
            gate = matmul.apply(weight, routed, counts, input_scales, torch.bfloat16, "w13", 1)
            activated, scales = activation._apply_activation(gate)
            down = matmul.apply(weight, activated, counts, scales, torch.bfloat16, "w2", 1)
            output = finalizer._finalize_routing(down, probabilities, row_mapping, ids)
            return routed, row_mapping, counts, input_scales, gate, activated, scales, down, output

        def batch():
            return [invoke(weights[iteration % len(weights)]) for iteration in range(protocol["iterations"])]

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
            raise ValueError("Unsupported tensor-MoE execution mode")

        def execute():
            if graph is None:
                return batch()
            graph.replay()
            return outputs

        sign = 1

        def verify():
            nonlocal sign
            sign = -sign
            values.neg_()
            observed = execute()
            torch.npu.synchronize()
            checked, maximum = 0, 0.0
            expected_counts = torch.bincount(ids_cpu.flatten().long(), minlength=experts)
            for tensors in observed:
                routed, row_mapping, counts, input_scales, gate, activated, scales, down, output = tensors
                counts_cpu = counts.cpu()
                if not torch.equal(counts_cpu, expected_counts):
                    raise ValueError("Tensor-MoE execution numerical check failed: expert counts")
                mapping = row_mapping.cpu().long()
                if mapping.numel() != tokens * top_k or not torch.equal(
                    mapping.sort().values, torch.arange(tokens * top_k)
                ):
                    raise ValueError("Tensor-MoE execution numerical check failed: routing permutation")
                expected_final = {token: torch.zeros(len(columns)) for token in selected_tokens}
                for token, expert, original, up_weight, down_weight in references:
                    topk_slot = int((ids_cpu[token] == expert).nonzero()[0, 0])
                    destination = int(mapping[token * top_k + topk_slot])
                    actual_input = routed[destination].float().cpu()
                    actual_input_scale = input_scales[destination].float().cpu()
                    original = original * sign
                    expected_scale = original.abs().max() / 127
                    expected_input = (original / expected_scale).round().clamp(-128, 127)
                    if (
                        not torch.allclose(actual_input_scale, expected_scale, atol=1e-7, rtol=2e-4)
                        or (actual_input - expected_input).abs().max().item() > 1
                    ):
                        raise ValueError("Tensor-MoE execution numerical check failed: routing/quantization")
                    expected_gate = (actual_input @ up_weight * actual_input_scale * 0.125).bfloat16().float()
                    actual_gate = gate[destination].float().cpu()
                    error = (actual_gate - expected_gate).abs()
                    if not torch.all(error <= 0.02 + 0.012 * expected_gate.abs()):
                        raise ValueError("Tensor-MoE execution numerical check failed: GMM1")
                    gate_half, up_half = actual_gate.chunk(2)
                    expected_activation = torch.nn.functional.silu(gate_half) * up_half
                    expected_scale = expected_activation.abs().max() / 127
                    actual_scale = scales[destination].float().cpu()
                    quantized = activated[destination].float().cpu()
                    expected_quantized = (expected_activation / expected_scale).round().clamp(-128, 127)
                    if (
                        not torch.allclose(actual_scale, expected_scale, atol=1e-7, rtol=2e-4)
                        or (quantized - expected_quantized).abs().max().item() > 1
                    ):
                        raise ValueError("Tensor-MoE execution numerical check failed: activation quantization")
                    expected_down = quantized @ down_weight * actual_scale * 0.125
                    actual_down = down[destination, columns].float().cpu()
                    down_error = (actual_down - expected_down).abs()
                    if not torch.all(down_error <= 0.02 + 0.012 * expected_down.abs()):
                        raise ValueError("Tensor-MoE execution numerical check failed: GMM2")
                    expected_final[token] += actual_down / top_k
                    maximum = max(maximum, error.max().item(), down_error.max().item())
                    checked += hidden + intermediate * 3 + len(columns)
                for token, expected in expected_final.items():
                    difference = (output[token, columns].float().cpu() - expected).abs()
                    if not torch.all(difference <= 0.02 + 0.012 * expected.abs()):
                        raise ValueError("Tensor-MoE execution numerical check failed: finalization")
                    maximum = max(maximum, difference.max().item())
                    checked += len(columns)
            return {"sampled_elements": checked, "max_abs_error": maximum, "input_mutation": sign}

        checks = [verify()]
        samples, wall_samples = [], []
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        for _ in range(protocol["repeats"]):
            started = time.perf_counter()
            begin.record()
            measured = execute()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) / protocol["iterations"])
            wall_samples.append((time.perf_counter() - started) * 1000 / protocol["iterations"])
            del measured
        checks.append(verify())
        row = {
            **case,
            "plan_index": index,
            "status": "passed",
            "weight_format": 29,
            "input_dtype": "bfloat16",
            "output_dtype": "bfloat16",
            "group_list_type": 1,
            "quant_mode": 1,
            "drop_pad_mode": 2,
            "weight_copies": len(weights),
            "kernel_sources": [
                "sglang.srt.hardware_backend.npu.moe.init_routing.NPUMoEInitRouting_v2",
                "sglang.srt.hardware_backend.npu.quantization.moe_methods.NPUW8A8Int8MoEMethod",
                "sglang.srt.hardware_backend.npu.moe.activation.NPUSwigluQuant",
                "sglang.srt.hardware_backend.npu.moe.finalize_routing.NPUFinalizeRouting",
            ],
            "execution_verification": "captured_graph_replay" if graph is not None else "eager_invocation",
            "checks": checks,
            "samples_ms": samples,
            "wall_samples_ms": wall_samples,
            "latency_ms": statistics.median(samples),
            "cv": statistics.pstdev(samples) / statistics.mean(samples),
        }
        result["cases"].append(row)
        save()
        print(json.dumps({"plan_index": index, "latency_ms": row["latency_ms"], "cv": row["cv"]}), flush=True)

    save()
    try:
        for index, case in selected:
            collect(index, case)
            gc.collect()
            torch.npu.empty_cache()
        result["status"] = "completed"
    except Exception:
        result.update(status="failed", failed_plan_index=index, error=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
