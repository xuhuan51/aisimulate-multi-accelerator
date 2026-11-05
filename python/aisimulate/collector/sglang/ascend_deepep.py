"""Measure actual SGLang DeepEP dispatch/combine with full-payload verification."""

import argparse
import datetime
import faulthandler
import gc
import hashlib
import importlib.metadata
import json
import os
import socket
import statistics
import subprocess
import time
import traceback
from pathlib import Path


def checked_dispatch_precision(output):
    precision = {
        "dispatch_dtype": str(output.hidden_states.dtype).removeprefix("torch."),
        "dispatch_scale_dtype": (
            None if output.hidden_states_scale is None else str(output.hidden_states_scale.dtype).removeprefix("torch.")
        ),
    }
    if precision != {"dispatch_dtype": "int8", "dispatch_scale_dtype": "float32"}:
        raise ValueError(f"DeepEP actual dispatch precision differs from the audited INT8 path: {precision}")
    return precision


def checked_combine_precision(output):
    dtype = str(output.dtype).removeprefix("torch.")
    # torch.equal checks values but accepts equal values with different dtypes.
    if dtype != "bfloat16":
        raise ValueError(f"DeepEP actual combine precision differs from the audited BF16 path: {dtype}")
    return {"combine_dtype": dtype}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--model-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--negative-control", choices=("dispatch", "combine"))
    args = parser.parse_args()
    faulthandler.dump_traceback_later(90, repeat=True)
    if args.output.exists():
        parser.error("Preserve prior observations; use a fresh output")
    plan, setup = json.loads(args.plan.read_text()), json.loads(args.setup.read_text())
    protocol = plan["protocol"]
    if plan["schema"] != "aisimulate-ascend-deepep-plan/v1" or min(protocol.values()) < 1 or protocol["repeats"] < 3:
        raise ValueError("Invalid transport plan/protocol")
    for name, value in setup["environment"].items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    local_rank = int(os.environ["LOCAL_RANK"])
    os.sched_setaffinity(0, setup["worker_cpu_affinity"][str(local_rank)])
    import torch
    import torch.distributed as distributed
    import torch_npu
    from sglang.srt.distributed.parallel_state import get_torch_distributed_pg_options
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPBuffer,
        _DeepEPDispatcherImplLowLatency,
        _DeepEPDispatcherImplNormal,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.moe.utils import DeepEPMode
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs

    init_npu_backend()
    torch.npu.set_device(local_rank)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    revision = subprocess.check_output(["git", "-C", "/sgl-workspace/sglang", "rev-parse", "HEAD"], text=True).strip()
    if revision != "baa9befa381511cd4d56f83980951e5a45fffc6d":
        raise ValueError("Reaudit DeepEP source for another framework revision")
    distributed.init_process_group(
        "hccl", timeout=datetime.timedelta(seconds=300), pg_options=get_torch_distributed_pg_options()
    )
    control = distributed.new_group(backend="gloo", timeout=datetime.timedelta(seconds=300))
    rank, world = distributed.get_rank(), distributed.get_world_size()
    local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    selected = [
        (index, case)
        for index, case in enumerate(plan["cases"])
        if case["num_ranks"] == world
        and case["ranks_per_node"] == local_world
        and index >= args.start_index
        and (args.case_index is None or index == args.case_index)
    ]
    if not selected or (args.negative_control and (len(selected) != 1 or selected[0][1]["execution_mode"] != "graph")):
        raise ValueError("Empty selection or invalid graph negative control")
    publish(
        ServerArgs(
            model_path=str(args.model_metadata),
            device="npu",
            tp_size=world,
            ep_size=world,
            dp_size=world // 8,
            nnodes=world // local_world,
            node_rank=rank // local_world,
            enable_dp_attention=True,
            enable_dp_lm_head=True,
            moe_a2a_backend="deepep",
            quantization="modelslim",
            dtype="bfloat16",
            context_length=180000,
            disable_shared_experts_fusion=True,
        ),
        role="scheduler",
    )
    common = dict(
        group=distributed.group.WORLD,
        router_topk=8,
        permute_fusion=False,
        num_experts=256,
        num_local_experts=256 // world,
        hidden_size=6144,
        params_dtype=torch.bfloat16,
        deepep_mode=DeepEPMode.AUTO,
    )
    dispatchers = {
        "normal": _DeepEPDispatcherImplNormal(async_finish=True, **common),
        "low_latency": _DeepEPDispatcherImplLowLatency(return_recv_hook=True, **common),
    }
    for dispatcher in dispatchers.values():
        dispatcher.set_quant_config({"dispatcher_output_dtype": "int8"})
    captured_graphs = []
    identities = [None] * world
    distributed.all_gather_object(
        identities,
        {
            "rank": rank,
            "local_rank": local_rank,
            "host": socket.gethostname(),
            "device": torch.npu.get_device_name(local_rank),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "deep_ep": importlib.metadata.version("deep_ep"),
            "driver_version_info": Path("/usr/local/Ascend/driver/version.info").read_text(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
        group=control,
    )
    record = {
        "schema": "aisimulate-ascend-deepep-observations/v1",
        "status": "running",
        "scope": plan["scope"],
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "collector_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "framework_commit": revision,
        "protocol": protocol,
        "timing_method": "per_call_outer_events_barrier_between_calls",
        "serving_setup": setup,
        "devices": identities,
        "selected_plan_indices": [index for index, _ in selected],
        "cases": [],
        "negative_control": args.negative_control,
    }

    def save():
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    def barrier():
        distributed.barrier(group=control)

    def collect(index, case):
        if rank == 0:
            print(json.dumps({"plan_index": index, "stage": "initial_dispatch"}), flush=True)
        tokens = case["tokens_per_rank"]
        if (
            case["hidden_size"] != 6144
            or case["num_experts"] != 256
            or case["topk"] != 8
            or case["max_dispatch_tokens_per_rank"] != 32
        ):
            raise ValueError("Unsupported DeepEP geometry")
        dispatcher = dispatchers[case["phase"]]
        global_ids = torch.arange(tokens, device="npu", dtype=torch.int64) + rank * tokens
        columns = torch.arange(6144, device="npu", dtype=torch.int64)
        inputs = ((global_ids[:, None] * 13 + columns[None, :]) % 15 - 7).to(torch.bfloat16)
        inputs[:, 0] = (global_ids % 127).to(torch.bfloat16)
        inputs[:, 1] = (global_ids // 127 % 127).to(torch.bfloat16)
        inputs[:, 2] = (global_ids // (127 * 127)).to(torch.bfloat16)
        inputs[:, -1] = 127
        if case["distribution"] == "balanced":
            ids = (global_ids[:, None] * 17 + torch.arange(8, device="npu", dtype=torch.int64) * 31) % 256
        elif case["distribution"] == "single_expert_group":
            ids = torch.arange(8, device="npu", dtype=torch.int64).expand(tokens, 8).contiguous()
        else:
            raise ValueError("Unknown routing distribution")
        weights = torch.full((tokens, 8), 0.125, device="npu", dtype=torch.float32)
        topk = StandardTopKOutput(
            weights, ids.to(torch.int32), torch.empty(tokens, 256, device="npu", dtype=torch.bfloat16)
        )
        cpu_global = torch.arange(world * tokens, dtype=torch.int64)
        all_ids = (
            (cpu_global[:, None] * 17 + torch.arange(8, dtype=torch.int64) * 31) % 256
            if case["distribution"] == "balanced"
            else torch.arange(8, dtype=torch.int64).expand(world * tokens, 8)
        )
        local_experts = 256 // world
        expected_ids = [
            cpu_global[(all_ids == rank * local_experts + expert).any(-1)] for expert in range(local_experts)
        ]
        expected_counts = [len(values) for values in expected_ids]
        saved = {}

        def dispatch():
            output = dispatcher.dispatch_b(*dispatcher.dispatch_a(inputs, topk))
            saved["handle"] = dispatcher.handle
            if case["phase"] == "low_latency":
                saved["counts"] = dispatcher.packed_recv_count
            return output

        first = dispatch()
        torch.npu.synchronize()
        dispatch_precision = checked_dispatch_precision(first)
        if rank == 0:
            print(json.dumps({"plan_index": index, "stage": "initial_verification"}), flush=True)
        combine_values = (first.hidden_states.float() * first.hidden_states_scale.reshape(-1, 1)).to(torch.bfloat16)

        def combine():
            dispatcher.handle = saved["handle"]
            if case["phase"] == "low_latency":
                dispatcher.packed_recv_count = saved["counts"]
                return dispatcher.combine_b(*dispatcher.combine_a(combine_values, ids, weights))
            return dispatcher.combine_b(*dispatcher.combine_a(combine_values, first.topk_ids, first.topk_weights))

        def check_dispatch(output, sign):
            checked_dispatch_precision(output)
            raw_counts = output.num_recv_tokens_per_expert if case["phase"] == "normal" else output.masked_m
            counts = raw_counts.cpu().tolist() if isinstance(raw_counts, torch.Tensor) else list(raw_counts)
            if (
                counts != expected_counts
                or output.hidden_states.dtype != torch.int8
                or output.hidden_states_scale.dtype != torch.float32
            ):
                raise ValueError("DeepEP dispatch numerical check failed: counts or precision")
            valid = sum(counts)
            actual = output.hidden_states[:valid].cpu() * sign
            scales = output.hidden_states_scale[:valid].cpu()
            if not torch.equal(scales, torch.ones_like(scales)):
                raise ValueError("DeepEP dispatch numerical check failed: unit quantization scales")
            offset = 0
            for count, wanted_ids in zip(counts, expected_ids, strict=True):
                if count:
                    values = actual[offset : offset + count]
                    markers = values[:, :3].to(torch.int64)
                    observed_ids = markers[:, 0] + markers[:, 1] * 127 + markers[:, 2] * 127 * 127
                    order = observed_ids.argsort()
                    if not torch.equal(observed_ids[order], wanted_ids):
                        raise ValueError("DeepEP dispatch numerical check failed: token routing")
                    for start in range(0, count, 256):
                        chunk_ids = wanted_ids[start : start + 256]
                        expected = ((chunk_ids[:, None] * 13 + torch.arange(6144)[None, :]) % 15 - 7).to(torch.int8)
                        expected[:, 0] = (chunk_ids % 127).to(torch.int8)
                        expected[:, 1] = (chunk_ids // 127 % 127).to(torch.int8)
                        expected[:, 2] = (chunk_ids // (127 * 127)).to(torch.int8)
                        expected[:, -1] = 127
                        if not torch.equal(values[order[start : start + 256]], expected):
                            raise ValueError("DeepEP dispatch numerical check failed: full payload")
                offset += count
            return {
                "checked_elements": valid * 6144,
                "checked_experts": len(counts),
                "valid_rows": valid,
                "physical_rows": output.hidden_states.shape[0],
            }

        check_dispatch(first, 1)
        barrier()
        combined = combine()
        combine_precision = checked_combine_precision(combined)
        if not torch.equal(combined, inputs):
            raise ValueError("DeepEP initial roundtrip numerical check failed")
        del combined
        measured = []
        for operation, invoke in (("dispatch", dispatch), ("combine", combine)):
            if rank == 0:
                print(json.dumps({"plan_index": index, "operation": operation, "stage": "warmup"}), flush=True)
            barrier()
            for _ in range(protocol["warmup"]):
                invoke()
                torch.npu.synchronize()
                barrier()
            torch.npu.synchronize()
            barrier()
            graph, outputs = None, None

            def batch():
                return [invoke()]

            if case["execution_mode"] == "graph":
                if rank == 0:
                    print(json.dumps({"plan_index": index, "operation": operation, "stage": "capture"}), flush=True)
                if case["phase"] != "low_latency":
                    raise ValueError("Normal dispatch graph execution is not a serving branch")
                graph = torch.npu.NPUGraph()
                captured_graphs.append(graph)
                with torch.npu.graph(graph):
                    outputs = batch()
                barrier()
                for _ in range(protocol["warmup"]):
                    graph.replay()
                    torch.npu.synchronize()
                    barrier()
                torch.npu.synchronize()

            def execute():
                if graph is None:
                    return batch()
                if args.negative_control != operation:
                    graph.replay()
                return outputs

            sign = 1

            def verify():
                nonlocal sign
                sign = -sign
                if operation == "dispatch":
                    inputs.neg_()
                else:
                    combine_values.neg_()
                torch.npu.synchronize()
                barrier()
                actual = execute()
                torch.npu.synchronize()
                checks = []
                for output in actual:
                    if operation == "dispatch":
                        checks.append(check_dispatch(output, sign))
                    else:
                        checked_combine_precision(output)
                        if not torch.equal(output, inputs * sign):
                            raise ValueError("DeepEP combine numerical check failed: full payload")
                        checks.append({"checked_elements": output.numel()})
                return {"checked_elements": sum(check["checked_elements"] for check in checks), "calls": len(actual)}

            checks = [verify()]
            if rank == 0:
                print(json.dumps({"plan_index": index, "operation": operation, "stage": "timing"}), flush=True)
            samples, wall = [], []
            for _ in range(protocol["repeats"]):
                elapsed, wall_elapsed = 0.0, 0.0
                for _ in range(protocol["iterations"]):
                    barrier()
                    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                    started = time.perf_counter()
                    begin.record()
                    actual = execute()
                    end.record()
                    end.synchronize()
                    elapsed += begin.elapsed_time(end)
                    wall_elapsed += (time.perf_counter() - started) * 1000
                    del actual
                samples.append(elapsed / protocol["iterations"])
                wall.append(wall_elapsed / protocol["iterations"])
            checks.append(verify())
            rank_results = [None] * world
            distributed.all_gather_object(
                rank_results,
                {
                    "rank": rank,
                    "samples_ms": samples,
                    "wall_samples_ms": wall,
                    "checks": checks,
                    "expected_counts": expected_counts,
                    "received_shape": list(first.hidden_states.shape),
                    **dispatch_precision,
                    **combine_precision,
                },
                group=control,
            )
            critical = [
                max(result["samples_ms"][sample] for result in rank_results) for sample in range(protocol["repeats"])
            ]
            measured.append(
                {
                    "operation": operation,
                    "rank_results": rank_results,
                    "latency_ms": statistics.median(critical),
                    "cv": statistics.pstdev(critical) / statistics.mean(critical),
                    "execution_verification": "captured_graph_replay" if graph is not None else "eager_invocation",
                }
            )
        record["cases"].append(
            {
                **case,
                "plan_index": index,
                "measurements": measured,
                "kernel_source": type(dispatcher).__module__ + "." + type(dispatcher).__name__,
            }
        )
        save()
        if rank == 0:
            print(
                json.dumps(
                    {
                        "plan_index": index,
                        "phase": case["phase"],
                        "tokens": tokens,
                        "measurements": [
                            {"operation": value["operation"], "latency_ms": value["latency_ms"], "cv": value["cv"]}
                            for value in measured
                        ],
                    }
                ),
                flush=True,
            )

    save()
    try:
        for index, case in selected:
            collect(index, case)
            gc.collect()
        record["status"] = "completed"
    except Exception:
        record.update(status="failed", failed_plan_index=index, error=traceback.format_exc())
        raise
    finally:
        save()
    barrier()
    DeepEPBuffer.clean_buffer()
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
