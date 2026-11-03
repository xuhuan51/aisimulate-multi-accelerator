import argparse
import copy
import json
from pathlib import Path

from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel


def configuration(model, scope, policy):
    return {
        "model": model, "system": "ascend910c", "backend": "sglang", "backend_version": "0.5.22",
        "worker_type": "decode", "tp": 4, "attention_dp": 4, "moe_tp_size": 1, "moe_ep_size": 16,
        "kv_block_size": 128, "gemm_quant_mode": "w8a8", "moe_quant_mode": "w8a8_dynamic",
        "kvcache_quant_mode": "bfloat16", "fmha_quant_mode": "bfloat16", "estimation_mode": "op_level",
        "estimator_config": {"op_level": {"ascend": {
            "decode_execution_mode": "eager", "shape_policy": policy, "enable_dp_lm_head": False,
            "decode_timing_scope": scope}}}}


def metrics(counts, context):
    return [{"scheduled_requests": {
        "num_prefill_requests": 0, "sum_prefill_tokens": 0, "var_prefill_length": 0.0,
        "sum_prefill_kv_tokens": 0, "num_decode_requests": count,
        "sum_decode_kv_tokens": count * context, "var_decode_kv_tokens": 0.0}} for count in counts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for scope, policy in (("isolated_kernels", "anchored"), ("runtime_layers", "measured"),
                          ("runtime_layers", "anchored")):
        config = configuration(args.model, scope, policy)
        model = RustForwardPassPerfModel.best_available(copy.deepcopy(config))
        for counts in ([1, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]):
            for context in (8192, 10240, 12288, 16384, 32768):
                row = {"config": config, "counts": counts, "context": context}
                try:
                    batch = metrics(counts, context)
                    latency = model.estimate_forward_pass_time_ms(batch)
                    operations = model.estimate_ascend_dp_operations(batch)
                    if abs(sum(op[1] for op in operations) - latency) > 1e-6:
                        raise AssertionError("Per-operation breakdown does not conserve total latency")
                    row.update(status="predicted", latency_ms=latency, operations=operations)
                except (ValueError, RuntimeError) as error:
                    row.update(status="rejected", error=str(error))
                rows.append(row)
    result = {"scope": "Canonical-model coverage and attribution check, not serving accuracy", "rows": rows}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        settings = row["config"]["estimator_config"]["op_level"]["ascend"]
        if settings["shape_policy"] == "anchored" and row["status"] != "predicted":
            raise AssertionError(f"Expected supported anchored query: {row}")
        if (settings["decode_timing_scope"] == "runtime_layers" and settings["shape_policy"] == "anchored"
                and row["context"] == 32768
                and not any(op[0] == "generation_decoder_layers" and op[3] == "estimated" for op in row["operations"])):
            raise AssertionError("Context transfer must be explicitly attributed as estimated")
    print(json.dumps({"queries": len(rows), "predicted": sum(row["status"] == "predicted" for row in rows),
                      "rejected": sum(row["status"] == "rejected" for row in rows)}))


if __name__ == "__main__":
    main()
