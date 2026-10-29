import argparse
import json
from pathlib import Path

from aisimulate_core.sdk import RustForwardPassPerfModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    records = []
    for global_size in (16, 32):
        for local_tp in (1, 2, 4, 8, 16):
            attention_dp = global_size // local_tp
            for tensor_moe in (False, True):
                for mode in ("graph", "eager"):
                    config = {
                        "model": args.model,
                        "system": "ascend910c",
                        "backend": "sglang",
                        "backend_version": "0.5.22",
                        "worker_type": "decode",
                        "tp": local_tp,
                        "attention_dp": attention_dp,
                        "moe_tp_size": global_size if tensor_moe else 1,
                        "moe_ep_size": 1 if tensor_moe else global_size,
                        "gemm_quant_mode": "w8a8",
                        "moe_quant_mode": "w8a8_dynamic",
                        "fmha_quant_mode": "bfloat16",
                        "kvcache_quant_mode": "bfloat16",
                        "comm_quant_mode": "half",
                        "kv_block_size": 128,
                        "estimation_mode": "op_level",
                        "fallback_policy": "deny",
                        "estimator_config": {
                            "op_level": {"ascend": {"decode_execution_mode": mode, "shape_policy": "anchored"}}
                        },
                    }
                    model = None
                    construction_error = None
                    try:
                        model = RustForwardPassPerfModel.best_available(config)
                    except (ValueError, RuntimeError) as error:
                        construction_error = str(error)
                    for kv_tokens in (8193, 32769):
                        for batch in (1, 4, 16, 32):
                            record = {"config": config, "batch_per_rank": batch, "kv_tokens": kv_tokens}
                            if construction_error is not None:
                                record["error"] = construction_error
                            else:
                                metrics = [
                                    {
                                        "version": 1,
                                        "worker_id": "execution-mode-probe",
                                        "dp_rank": rank,
                                        "counter_id": 0,
                                        "scheduled_requests": {
                                            "num_prefill_requests": 0,
                                            "sum_prefill_tokens": 0,
                                            "sum_prefill_kv_tokens": 0,
                                            "num_decode_requests": batch,
                                            "sum_decode_kv_tokens": batch * kv_tokens,
                                        },
                                    }
                                    for rank in range(attention_dp)
                                ]
                                try:
                                    latency = model.estimate_forward_pass_time_ms(metrics)
                                    if latency is None or not 0 < latency < float("inf"):
                                        raise ValueError(f"Invalid latency: {latency}")
                                    record["latency_ms"] = latency
                                except (ValueError, RuntimeError) as error:
                                    record["error"] = str(error)
                            records.append(record)
    failures = sum("error" in record for record in records)
    report = {
        "scope": (
            "Canonical forward queries with uniform active batches on every DP rank; not serving or accuracy validation"
        ),
        "attempted": len(records),
        "successful": len(records) - failures,
        "failed": failures,
        "records": records,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}))
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
