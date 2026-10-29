"""Probe complete per-rank batches through the canonical Ascend estimator."""

import argparse
import hashlib
import json
from pathlib import Path

import aisimulate._runtime as runtime
from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for mode in ("eager", "graph"):
        for world in (16, 32):
            for tensor_moe in (False, True):
                for tp in (1, 2, 4, 8, 16):
                    dp = world // tp
                    identity = {"mode": mode, "world": world, "tensor_moe": tensor_moe, "tp": tp, "dp": dp}
                    config = {
                        "model": args.model,
                        "system": "ascend910c",
                        "backend": "sglang",
                        "backend_version": "0.5.22",
                        "worker_type": "decode",
                        "tp": tp,
                        "attention_dp": dp,
                        "moe_tp_size": world if tensor_moe else 1,
                        "moe_ep_size": 1 if tensor_moe else world,
                        "kv_block_size": 128,
                        "gemm_quant_mode": "w8a8",
                        "moe_quant_mode": "w8a8_dynamic",
                        "kvcache_quant_mode": "bfloat16",
                        "fmha_quant_mode": "bfloat16",
                        "estimation_mode": "op_level",
                        "estimator_config": {
                            "op_level": {"ascend": {"decode_execution_mode": mode, "shape_policy": "anchored"}}
                        },
                    }
                    model = RustForwardPassPerfModel.best_available(config)
                    counts = [[1] * dp, [3] * dp, [1] + [0] * (dp - 1), [17] + [0] * (dp - 1)]
                    counts += [[1, maximum] + [0] * (dp - 2) for maximum in (9, 17, 31)] if dp > 1 else []
                    counts += [[17] * ((dp + 1) // 2) + [0] * (dp // 2)]
                    if dp > 1:
                        counts += [[1] + [maximum] * (dp - 1) for maximum in (2, 3, 5, 7, 9, 13, 17, 23, 31)]
                    for batch in counts:
                        for kv_length in (8193, 32769):
                            record = {**identity, "batch": batch, "kv_length": kv_length}
                            metrics = [
                                {
                                    "scheduled_requests": {
                                        "num_decode_requests": active,
                                        "sum_decode_kv_tokens": active * kv_length,
                                    }
                                }
                                for active in batch
                            ]
                            try:
                                record["latency_ms"] = model.estimate_forward_pass_time_ms(metrics)
                            except Exception as error:
                                record["error"] = str(error)
                            records.append(record)
    output = {
        "scope": "Shape coverage only; nonuniform transport and end-to-end accuracy remain unqualified",
        "native_sha256": hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest(),
        "queries": len(records),
        "successful": sum("latency_ms" in record for record in records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "records"}))


if __name__ == "__main__":
    main()
