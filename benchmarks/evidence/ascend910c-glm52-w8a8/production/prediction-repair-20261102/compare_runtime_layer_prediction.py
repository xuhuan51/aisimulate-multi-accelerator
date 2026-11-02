import argparse
import collections
import copy
import hashlib
import json
import statistics
from pathlib import Path

from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel


def configuration(scope):
    return {
        "model": "/data_lib/data/models/glm-q", "system": "ascend910c", "backend": "sglang",
        "backend_version": "0.5.22", "worker_type": "decode", "tp": 4, "attention_dp": 4,
        "moe_tp_size": 1, "moe_ep_size": 16, "kv_block_size": 128, "gemm_quant_mode": "w8a8",
        "moe_quant_mode": "w8a8_dynamic", "kvcache_quant_mode": "bfloat16", "fmha_quant_mode": "bfloat16",
        "estimation_mode": "op_level", "estimator_config": {"op_level": {"ascend": {
            "decode_execution_mode": "eager", "shape_policy": "anchored", "enable_dp_lm_head": False,
            "decode_timing_scope": scope,
        }}},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign
    manifest = json.loads((campaign / "chain-campaign.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Only completed diagnostic campaigns can be compared")
    group_rows = collections.defaultdict(dict)
    sources = {}
    for path in sorted(campaign.glob("paired-forward-rank*.jsonl")):
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            row = json.loads(line)
            group = group_rows[row["timed_forward_iteration"]]
            if row["dp_rank"] in group:
                raise ValueError("Duplicate paired rank/iteration")
            group[row["dp_rank"]] = row
    raw_config, layer_config = configuration("isolated_kernels"), configuration("runtime_layers")
    raw_model = RustForwardPassPerfModel.best_available(raw_config)
    layer_model = RustForwardPassPerfModel.best_available(layer_config)
    results = []
    for case in manifest["cases"]:
        probe = json.loads((campaign / case["id"] / "layer-replay-rank0.json").read_text())
        probe_iteration = probe["model_call"]
        retained = []
        gaps = collections.Counter()
        excluded = collections.Counter()
        for iteration, rank_rows in sorted(group_rows.items()):
            if len(rank_rows) != 4:
                continue
            rows = [rank_rows[rank] for rank in range(4)]
            if not all(case["start_unix"] <= row["launch_unix"] <= case["end_unix"] for row in rows):
                continue
            if abs(iteration - probe_iteration) <= 4:
                excluded["isolated_probe_and_neighbor_steps"] += 1
                continue
            requests = [request for row in rows for request in row["requests"]]
            if (len(requests) != case["global_requests"]
                    or any(row["scheduled_requests"]["sum_prefill_tokens"] for row in rows)
                    or any(request["prompt_tokens"] != case["input_length"]
                           or not 40 <= request["output_tokens"] <= 184 for request in requests)):
                excluded["prefill_warmup_or_unsteady_requests"] += 1
                continue
            if any(row["scheduled_requests"]["num_decode_requests"] != case["global_requests"] // 4 for row in rows):
                excluded["unbalanced_dp"] += 1
                continue
            metrics = [{"scheduled_requests": row["scheduled_requests"]} for row in rows]
            result = {"iteration": iteration, "metrics": metrics,
                      "measured_ms": max(row["elapsed_ms"] for row in rows),
                      "isolated_kernel_prediction_ms": raw_model.estimate_forward_pass_time_ms(metrics)}
            try:
                result["runtime_layer_prediction_ms"] = layer_model.estimate_forward_pass_time_ms(metrics)
            except Exception as error:
                if "measured request/KV brackets" not in str(error):
                    raise
                result["runtime_layer_unavailable"] = str(error)
                gaps["outside_measured_request_or_kv_brackets"] += 1
            retained.append(result)
        if len(retained) < 30:
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
