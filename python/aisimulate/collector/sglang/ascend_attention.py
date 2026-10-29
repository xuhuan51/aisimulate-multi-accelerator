"""Measure source-matched DSA kernels with active/padded sequence checks.

Pinned SGLang baa9befa381511cd4d56f83980951e5a45fffc6d:
dsa_npu_indexer.py selects lightning_indexer on arch2201;
ascend_backend.py selects sparse_flash_attention with TND/PA_BSND;
memory_pool_npu.py:680-720 allocates five-dimensional BF16 cache tensors.
The one-layer view below retains the observed NCDHW format30 and strides.
Synthetic tensor values are numerical controls, not checkpoint activation values.
"""

import argparse
import gc
import hashlib
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
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not 0 <= args.shard_index < args.num_shards:
        parser.error("Use a fresh output and valid shard index")
    plan = json.loads(args.plan.read_text())
    setup = json.loads(args.serving_setup.read_text())
    protocol = plan["protocol"]
    if plan["schema"] != "aisimulate-ascend-attention-plan/v1" or min(protocol.values()) < 1 or protocol["repeats"] < 3:
        raise ValueError("Invalid attention plan/protocol")
    for name, value in setup["environment"].items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    os.sched_setaffinity(0, setup["worker_cpu_affinity"][str(args.device)])
    import torch
    import torch_npu

    torch.npu.set_device(args.device)
    torch.set_num_threads(8)
    revision = subprocess.check_output(["git", "-C", "/sgl-workspace/sglang", "rev-parse", "HEAD"], text=True).strip()
    if revision != "baa9befa381511cd4d56f83980951e5a45fffc6d":
        raise ValueError("Reaudit dispatch for the installed framework")
    selected = [
        (index, case)
        for index, case in enumerate(plan["cases"])
        if index % args.num_shards == args.shard_index
        and (not args.case_filter or args.case_filter in json.dumps(case, sort_keys=True))
    ]
    if not selected:
        raise ValueError("Empty selected plan")
    record = {
        "schema": "aisimulate-ascend-attention-observations/v1",
        "status": "running",
        "scope": "Syntheticvalues;actualsource/layout/active-paddingcontract;notwholeforwardaccuracy",
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "collector_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol": protocol,
        "measurement_warmup": "after_numerical_verification_with_timing_events",
        "selected_plan_indices": [index for index, _ in selected],
        "cases": [],
        "failures": [],
        "runtime": {
            "framework_commit": revision,
            "host": socket.gethostname(),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device": torch.npu.get_device_name(args.device),
            "driver_version_info": Path("/usr/local/Ascend/driver/version.info").read_text(),
            "device_index": args.device,
            "serving_setup": setup,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(record, indent=2) + "\n")

    def collect(index, case):
        torch.manual_seed(910 + index)
        tokens, active, padded = case["tokens"], case["active_batch"], case["padded_batch"]
        page_size, length, prefix = case["page_size"], case["sequence_length"], case["prefix"]
        query_tokens = tokens // active if case["phase"] == "prefill" else 1
        if case["phase"] == "prefill" and (tokens % active or length != query_tokens + prefix):
            raise ValueError("Prefill requires equal query lengths per active request")
        pages_per_request = (length + page_size - 1) // page_size
        table = torch.zeros(padded, case["block_table_width"], device="npu", dtype=torch.int32)
        for request in range(active):
            table[request, :pages_per_request] = torch.arange(
                1 + request * pages_per_request, 1 + (request + 1) * pages_per_request, device="npu", dtype=torch.int32
            )
        query_lengths = (
            torch.arange(query_tokens, tokens + 1, query_tokens, device="npu", dtype=torch.int32)
            if case["phase"] == "prefill"
            else torch.arange(1, padded + 1, device="npu", dtype=torch.int32)
        )
        key_lengths = torch.tensor([length] * active + [0] * (padded - active), device="npu", dtype=torch.int32)
        heads = case["local_heads"]
        width = case["index_dim"] if case["operation"] == "lightning_indexer" else case["kv_lora_rank"]
        query = torch.randn(tokens, heads, width, device="npu", dtype=torch.bfloat16)
        cache_owner = torch.randn(1, case["pool_pages"], page_size, 1, width, device="npu", dtype=torch.bfloat16)
        cache = cache_owner[0]
        if torch_npu.get_npu_format(cache) != 30:
            raise ValueError("Cache format differs from observed layer view")
        weights = torch.rand(tokens, case["index_heads"], device="npu", dtype=torch.bfloat16)
        query_rope = torch.randn(tokens, heads, case["rope_dim"], device="npu", dtype=torch.bfloat16)
        rope_owner = torch.randn(
            1, case["pool_pages"], page_size, 1, case["rope_dim"], device="npu", dtype=torch.bfloat16
        )
        rope_cache = rope_owner[0]
        indices_cpu = torch.full((tokens, 1, case["sparse_topk"]), -1, dtype=torch.int32)
        for row in range(tokens if case["phase"] == "prefill" else active):
            visible = prefix + row % query_tokens + 1 if case["phase"] == "prefill" else length
            count = min(visible, case["sparse_topk"])
            indices_cpu[row, 0, :count] = torch.arange(max(0, visible - count), visible, dtype=torch.int32)
        indices = indices_cpu.to("npu")

        def invoke():
            if case["operation"] == "lightning_indexer":
                return torch_npu.npu_lightning_indexer(
                    query=query,
                    key=cache,
                    weights=weights,
                    actual_seq_lengths_query=query_lengths,
                    actual_seq_lengths_key=key_lengths,
                    block_table=table,
                    layout_query="TND",
                    layout_key="PA_BSND",
                    sparse_count=case["sparse_topk"],
                    sparse_mode=3,
                )[0]
            return torch_npu.npu_sparse_flash_attention(
                query=query,
                key=cache,
                value=cache,
                query_rope=query_rope,
                key_rope=rope_cache,
                sparse_indices=indices,
                scale_value=0.0625,
                actual_seq_lengths_query=query_lengths,
                actual_seq_lengths_kv=key_lengths,
                block_table=table,
                sparse_block_size=1,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                attention_mode=2,
                return_softmax_lse=False,
            )[0]

        def batch():
            return [invoke() for _ in range(protocol["iterations"])]

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

        def verify():
            query.neg_()
            observed = execute()
            torch.npu.synchronize()
            count = tokens if case["phase"] == "prefill" else active
            row_ids = sorted({0, count // 2, count - 1})
            maximum_error, minimum_overlap = 0.0, 1.0
            for row in row_ids:
                request = row // query_tokens if case["phase"] == "prefill" else row
                visible = prefix + row % query_tokens + 1 if case["phase"] == "prefill" else length
                start = 1 + request * pages_per_request
                keys = cache[start : start + pages_per_request].reshape(-1, width)[:visible].float().cpu()
                query_cpu = query[row].float().cpu()
                if case["operation"] == "lightning_indexer":
                    scores = ((query_cpu @ keys.T).relu() * weights[row].float().cpu()[:, None]).sum(0)
                    expected = set(scores.topk(min(visible, case["sparse_topk"])).indices.tolist())
                    for result in observed:
                        raw = result[row].flatten().cpu().tolist()
                        selected_indices = [value for value in raw if value >= 0]
                        actual = set(selected_indices)
                        overlap = len(actual & expected) / len(expected)
                        if (
                            len(actual) != len(expected)
                            or len(actual) != len(selected_indices)
                            or overlap < 0.99
                            or any(value >= visible for value in actual)
                        ):
                            raise ValueError("Indexer execution numerical check failed")
                        minimum_overlap = min(minimum_overlap, overlap)
                else:
                    chosen = indices_cpu[row, 0]
                    chosen = chosen[chosen >= 0].long()
                    keys = keys[chosen]
                    ropes = (
                        rope_cache[start : start + pages_per_request]
                        .reshape(-1, case["rope_dim"])[:visible]
                        .float()
                        .cpu()[chosen]
                    )
                    logits = (query_cpu @ keys.T + query_rope[row].float().cpu() @ ropes.T) * 0.0625
                    expected = logits.softmax(-1) @ keys
                    for result in observed:
                        actual = result[row].float().cpu()
                        error = (actual - expected).abs()
                        if not torch.all(error <= 0.02 + 0.012 * expected.abs()):
                            tolerance = 0.02 + 0.012 * expected.abs()
                            location = (error - tolerance).flatten().argmax().item()
                            details = {
                                "query_row": row,
                                "failing_elements": int((error > tolerance).sum().item()),
                                "checked_elements": error.numel(),
                                "max_abs_error": error.max().item(),
                                "rms_error": error.square().mean().sqrt().item(),
                                "reference_rms": expected.square().mean().sqrt().item(),
                                "worst_actual": actual.flatten()[location].item(),
                                "worst_expected": expected.flatten()[location].item(),
                                "worst_tolerance": tolerance.flatten()[location].item(),
                                "head": location // expected.shape[-1],
                                "dimension": location % expected.shape[-1],
                            }
                            eager_control = invoke()
                            torch.npu.synchronize()
                            details["eager_control"] = eager_control[row].float().cpu().flatten()[location].item()
                            details["raw_bf16_cpu_control"] = result[row].cpu().float().flatten()[location].item()
                            dense_weights = logits.softmax(-1)
                            details["reference_float64"] = (
                                (logits.double().softmax(-1) @ keys.double()).flatten()[location].item()
                            )
                            details["reference_bf16_weights"] = (
                                (dense_weights.to(torch.bfloat16).float() @ keys).flatten()[location].item()
                            )
                            raise ValueError(
                                "Sparse attention execution numerical check failed: " + json.dumps(details)
                            )
                        maximum_error = max(maximum_error, error.max().item())
            return {
                "sampled_queries": len(row_ids) * len(observed),
                "max_abs_error": maximum_error,
                "minimum_topk_overlap": minimum_overlap,
                "padding_kv_lengths": key_lengths[active:].cpu().tolist(),
            }

        checks = [verify()]
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        for _ in range(protocol["warmup"]):
            begin.record()
            result = execute()
            end.record()
            end.synchronize()
            del result
        values, wall = [], []
        for _ in range(protocol["repeats"]):
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
            "cache_format": torch_npu.get_npu_format(cache),
            "cache_stride": list(cache.stride()),
            "kernel_source": "torch_npu.npu_" + case["operation"].replace("sparse_attention", "sparse_flash_attention"),
            "execution_verification": "captured_graph_replay" if graph is not None else "eager_invocation",
            "checks": checks,
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
            try:
                collect(index, case)
            except ValueError as error:
                if "execution numerical check failed" not in str(error):
                    raise
                record["failures"].append(
                    {
                        "plan_index": index,
                        "case": case,
                        "classification": "unexpected",
                        "exception_type": type(error).__name__,
                        "error": traceback.format_exc(),
                    }
                )
                save()
                if args.fail_fast:
                    raise
                same_family = [
                    failure for failure in record["failures"] if failure["case"]["operation"] == case["operation"]
                ]
                attempts = len(same_family) + sum(row["operation"] == case["operation"] for row in record["cases"])
                if len(same_family) >= 3 and len(same_family) / attempts >= 0.1:
                    raise RuntimeError("Systematic numerical failures require investigation") from error
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
