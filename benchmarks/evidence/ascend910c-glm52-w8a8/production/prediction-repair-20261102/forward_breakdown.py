import argparse
import json
from pathlib import Path

from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-logits', action='store_true')
    args = parser.parse_args()
    options = {'decode_execution_mode': 'eager', 'shape_policy': 'anchored'}
    if args.runtime_logits:
        options['enable_dp_lm_head'] = False
    configuration = {
        'model': '/data_lib/data/models/glm-q', 'system': 'ascend910c', 'backend': 'sglang',
        'backend_version': '0.5.22', 'worker_type': 'decode', 'tp': 4, 'attention_dp': 4,
        'moe_tp_size': 1, 'moe_ep_size': 16, 'kv_block_size': 128,
        'gemm_quant_mode': 'w8a8', 'moe_quant_mode': 'w8a8_dynamic',
        'kvcache_quant_mode': 'bfloat16', 'fmha_quant_mode': 'bfloat16',
        'estimation_mode': 'op_level', 'estimator_config': {'op_level': {'ascend': options}},
    }
    model = RustForwardPassPerfModel.best_available(configuration)
    result = {'configuration': configuration, 'uniform_ranks': {}, 'active_ranks': {}}
    for batch in [1, 2, 4, 8]:
        metrics = {'scheduled_requests': {'num_decode_requests': batch, 'sum_decode_kv_tokens': 8193 * batch}}
        result['uniform_ranks'][str(batch)] = {
            'forward_ms': model.estimate_forward_pass_time_ms([metrics] * 4),
            'operators': model.static_phase_diagnostics(batch_size=batch, context_length=8192, prefill=False),
        }
    for active in [1, 2, 3, 4]:
        metrics = [{'scheduled_requests': {'num_decode_requests': 1, 'sum_decode_kv_tokens': 8193}}] * active
        metrics.extend([{'scheduled_requests': {}}] * (4 - active))
        result['active_ranks'][str(active)] = model.estimate_forward_pass_time_ms(metrics)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'active_ranks': result['active_ranks'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
