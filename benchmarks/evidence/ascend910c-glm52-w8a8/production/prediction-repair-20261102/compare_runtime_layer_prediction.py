import argparse
import collections
import copy
import hashlib
import json
import statistics
from pathlib import Path

from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel


def configuration(scope, attention_tp=4, attention_dp=4, moe_ep=16, tp_lm_head_all_to_all=False):
    return {
        "model": "/data_lib/data/models/glm-q", "system": "ascend910c", "backend": "sglang",
        "backend_version": "0.5.22", "worker_type": "decode", "tp": attention_tp, "attention_dp": attention_dp,
        "moe_tp_size": 1, "moe_ep_size": moe_ep, "kv_block_size": 128, "gemm_quant_mode": "w8a8",
        "moe_quant_mode": "w8a8_dynamic", "kvcache_quant_mode": "bfloat16", "fmha_quant_mode": "bfloat16",
        "estimation_mode": "op_level", "estimator_config": {"op_level": {"ascend": {
            "decode_execution_mode": "eager", "shape_policy": "anchored", "enable_dp_lm_head": False,
            "decode_timing_scope": scope,
            "enable_tp_lm_head_all_to_all": tp_lm_head_all_to_all,
        }}},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", default="chain-campaign.json")
    parser.add_argument("--allow-asymmetric", action="store_true")
    parser.add_argument("--attention-tp", type=int, default=4)
    parser.add_argument("--attention-dp", type=int, default=4)
    parser.add_argument("--moe-ep", type=int, default=16)
    parser.add_argument("--tp-lm-head-all-to-all", action="store_true")
    parser.add_argument("--output-length", type=int)
    parser.add_argument("--telemetry-dir", type=Path, action="append")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    campaign = args.campaign
    manifest = json.loads((campaign / args.manifest).read_text())
    if manifest["status"] != "completed":
        raise ValueError("Only completed diagnostic campaigns can be compared")
    group_rows = collections.defaultdict(dict)
    sources = {}
    telemetry_files = [path for folder in (args.telemetry_dir or [campaign])
                       for path in folder.glob("paired-forward-rank*.jsonl")]
    for path in sorted(telemetry_files):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            row = json.loads(line)
            group = group_rows[row["timed_forward_iteration"]]
            if row["dp_rank"] in group:
                raise ValueError("Duplicate paired rank/iteration")
            group[row["dp_rank"]] = row
    topology = (args.attention_tp, args.attention_dp, args.moe_ep, args.tp_lm_head_all_to_all)
    raw_config, layer_config = configuration("isolated_kernels", *topology), configuration("runtime_layers", *topology)
    raw_model = RustForwardPassPerfModel.best_available(raw_config)
    layer_model = RustForwardPassPerfModel.best_available(layer_config)
    results = []
    output_length = args.output_length or manifest.get("output_length", 192)
    minimum_output = min(40, output_length // 4)
    maximum_output = output_length - 8
    for case in manifest["cases"]:
        probe = json.loads((campaign / case["id"] / "layer-replay-rank0.json").read_text())
        probe_iteration = probe["model_call"]
        retained = []
        gaps = collections.Counter()
        excluded = collections.Counter()
        for iteration, rank_rows in sorted(group_rows.items()):
            if len(rank_rows) != args.attention_dp:
                continue
            rows = [rank_rows[rank] for rank in range(args.attention_dp)]
            if not all(case["start_unix"] <= row["launch_unix"] <= case["end_unix"] for row in rows):
                continue
            if abs(iteration - probe_iteration) <= 4:
                excluded["isolated_probe_and_neighbor_steps"] += 1
                continue
            requests = [request for row in rows for request in row["requests"]]
            if (len(requests) != case["global_requests"]
                    or any(row["scheduled_requests"]["sum_prefill_tokens"] for row in rows)
                    or any(request["prompt_tokens"] != case["input_length"]
                           or not minimum_output <= request["output_tokens"] <= maximum_output for request in requests)):
                excluded["prefill_warmup_or_unsteady_requests"] += 1
                continue
            if not args.allow_asymmetric and any(
                    row["scheduled_requests"]["num_decode_requests"] != case["global_requests"] // args.attention_dp for row in rows):
                excluded["unbalanced_dp"] += 1
                continue
            metrics = [{"scheduled_requests": row["scheduled_requests"]} for row in rows]
            result = {"iteration": iteration, "metrics": metrics,
                      "measured_ms": max(row["elapsed_ms"] for row in rows),
                      "isolated_kernel_prediction_ms": raw_model.estimate_forward_pass_time_ms(metrics)}
            try:
                result["runtime_layer_prediction_ms"] = layer_model.estimate_forward_pass_time_ms(metrics)
            except Exception as error:
                if not any(message in str(error) for message in (
                    "measured request/KV brackets", "unmeasured Ascend runtime layer",
                    "skewed active-rank profiles",
                    "lacks a common measured anchor",
                )):
                    raise
                result["runtime_layer_unavailable"] = str(error)
                gaps["unqualified_population_or_request_or_kv_brackets"] += 1
            retained.append(result)
        if len(retained) < min(30, output_length // 3):
            raise ValueError(f"Insufficient paired unprofiled steps for {case['id']}: {len(retained)}")
        supported = [row for row in retained if "runtime_layer_prediction_ms" in row]
        summary = {
            "unprofiled_groups": len(retained), "runtime_layer_supported_groups": len(supported),
            "measured_ms": statistics.median(row["measured_ms"] for row in retained),
            "isolated_kernel_prediction_ms": statistics.median(row["isolated_kernel_prediction_ms"] for row in retained),
            "isolated_kernel_mape_percent": statistics.mean(
                abs(row["isolated_kernel_prediction_ms"] / row["measured_ms"] - 1) * 100 for row in retained),
        }
        if supported:
            summary.update(
                matched_measured_ms=statistics.median(row["measured_ms"] for row in supported),
                runtime_layer_prediction_ms=statistics.median(row["runtime_layer_prediction_ms"] for row in supported),
                runtime_layer_mape_percent=statistics.mean(
                    abs(row["runtime_layer_prediction_ms"] / row["measured_ms"] - 1) * 100 for row in supported),
            )
        results.append({"case": copy.deepcopy(case), "summary": summary,
                        "exclusions": dict(excluded), "coverage_gaps": dict(gaps), "forwards": retained})
        print(json.dumps({"case": case["id"], **summary}), flush=True)
    args.output.write_text(json.dumps({
        "scope": "Unprofiled matched steady decode steps; excludes module replay and neighboring steps; no serving residual fit",
        "limitations": "Not a full8192/1024 request TTFT/TPOT/throughput validation. Unmeasured request/KV shapes remain unsupported.",
        "configurations": {"isolated_kernels": raw_config, "runtime_layers": layer_config},
        "source_sha256": sources, "cases": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
