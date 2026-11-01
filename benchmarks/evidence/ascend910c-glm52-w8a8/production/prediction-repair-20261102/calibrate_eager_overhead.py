import argparse
import gzip
import hashlib
import json
import statistics
from pathlib import Path

from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel


ROOT = Path(__file__).resolve().parent


def grouped_forwards(records, case):
    selected = [row for row in records if case['start_unix'] <= row['received_unix'] <= case['end_unix'] and row['metrics']['wall_time'] > 0]
    groups = []
    for row in sorted(selected, key=lambda item: item['received_unix']):
        if not groups or row['received_unix'] - groups[-1][-1]['received_unix'] > 0.2:
            groups.append([])
        groups[-1].append(row)
    retained = []
    for group in groups:
        if len(group) != 4 or {row['metrics']['dp_rank'] for row in group} != set(range(4)):
            continue
        spread = group[-1]['received_unix'] - group[0]['received_unix']
        if spread > min(0.2, 0.5 * min(row['metrics']['wall_time'] for row in group)):
            continue
        ordered = sorted(group, key=lambda row: row['metrics']['dp_rank'])
        metrics = [row['metrics']['scheduled_requests'] for row in ordered]
        if any(metric['sum_prefill_tokens'] > 0 for metric in metrics):
            continue
        active = [metric for metric in metrics if metric['num_decode_requests'] > 0]
        if not active:
            continue
        positions = [metric['sum_decode_kv_tokens'] / metric['num_decode_requests'] - case['input_length'] for metric in active]
        if not all(40 <= position <= case['output_length'] - 8 for position in positions):
            continue
        retained.append({
            'metrics': [{'scheduled_requests': metric} for metric in metrics],
            'measured_ms': max(row['metrics']['wall_time'] for row in ordered) * 1000,
            'source_rows': ordered,
        })
    return retained


def configuration(overhead=None):
    options = {'decode_execution_mode': 'eager', 'shape_policy': 'anchored', 'enable_dp_lm_head': False}
    if overhead is not None:
        options['eager_runtime_overhead'] = overhead
    return {
        'model': '/data_lib/data/models/glm-q', 'system': 'ascend910c', 'backend': 'sglang',
        'backend_version': '0.5.22', 'worker_type': 'decode', 'tp': 4, 'attention_dp': 4,
        'moe_tp_size': 1, 'moe_ep_size': 16, 'kv_block_size': 128, 'gemm_quant_mode': 'w8a8',
        'moe_quant_mode': 'w8a8_dynamic', 'kvcache_quant_mode': 'bfloat16', 'fmha_quant_mode': 'bfloat16',
        'estimation_mode': 'op_level', 'estimator_config': {'op_level': {'ascend': options}},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    campaign = ROOT / 'forward-metrics-node20'
    state = json.loads((campaign / 'progress.json').read_text())
    if state['status'] != 'completed':
        raise ValueError('Freeze a completed campaign before calibrating')
    raw_path = campaign / 'forward-metrics.jsonl'
    raw = raw_path.read_bytes() if raw_path.exists() else gzip.decompress(raw_path.with_suffix('.jsonl.gz').read_bytes())
    records = [json.loads(line) for line in raw.splitlines()]
    model = RustForwardPassPerfModel.best_available(configuration())
    cases = {case['name']: case for case in state['cases']}
    calibration = grouped_forwards(records, cases['calibration-c1'])
    if len(calibration) < 50 or any(sum(rank['scheduled_requests']['num_decode_requests'] for rank in row['metrics']) != 1 for row in calibration):
        raise ValueError('C1 calibration requires at least 50 complete single-request DP groups')
    for row in calibration:
        row['raw_prediction_ms'] = model.estimate_forward_pass_time_ms(row['metrics'])
        row['residual_ms'] = row['measured_ms'] - row['raw_prediction_ms']
    offset = statistics.median(row['residual_ms'] for row in calibration)
    if offset < 0:
        raise ValueError('An additive nonnegative overhead cannot represent this campaign')
    identity = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()
    overhead = {'calibration_id': f'node20-c1-fpm-{identity[:16]}', 'attention_tp': 4,
                'attention_dp': 4, 'moe_ep_size': 16, 'prefill_ms': 0.0, 'decode_ms': offset}
    report = {
        'configuration': configuration(), 'profile': overhead, 'calibration_case': 'calibration-c1',
        'scope': 'Additive eager forward residual for TP16/ADP4/EP16 and the pinned runtime. Prefill is not calibrated.',
        'rule': 'Median(measured NPU-event forward ms minus native operator estimate), only C1 formal-decode positions40..120. Warmup is capped at32 output tokens and excluded.',
        'grouping': 'Four distinct DP ranks arriving within200ms and less than half the shortest reported forward duration; per-publisher counters are not global step IDs. Incomplete or duplicate-rank groups are excluded.',
        'source_sha256': hashlib.sha256(raw).hexdigest(), 'calibration_rows': calibration,
        'limitations': ['NPU-event elapsed time includes stream idle and host-submission gaps, not just kernel work.',
                        'Residual is not exclusively CPU overhead; operator omissions may contribute.',
                        'Idle ranks publish about130..145ms later than the active rank. Bounded arrival grouping is diagnostic, not a framework-issued common iteration ID.',
                        'No claimed validity for graph, other software, parallel layouts, or arbitrary model weights.'],
    }
    profile_path = ROOT / 'eager-overhead-tp16-adp4.json'
    encoded = json.dumps(overhead, indent=2) + '\n'
    if profile_path.exists() and profile_path.read_text() != encoded:
        raise ValueError('Calibration changed; preserve the prior profile under its identity')
    profile_path.write_text(encoded)
    if not args.prepare_only:
        calibrated = RustForwardPassPerfModel.best_available(configuration(overhead))
        report['validation'] = []
        for name, case in cases.items():
            if name == 'calibration-c1':
                continue
            rows = grouped_forwards(records, case)
            if len(rows) < 10:
                raise ValueError(f'Insufficient validation groups for {name}: {len(rows)}')
            for row in rows:
                row.pop('source_rows')
                row['raw_prediction_ms'] = model.estimate_forward_pass_time_ms(row['metrics'])
                row['calibrated_prediction_ms'] = calibrated.estimate_forward_pass_time_ms(row['metrics'])
                row['raw_error_percent'] = (row['raw_prediction_ms'] / row['measured_ms'] - 1) * 100
                row['calibrated_error_percent'] = (row['calibrated_prediction_ms'] / row['measured_ms'] - 1) * 100
            errors = sorted(abs(row['calibrated_error_percent']) for row in rows)
            report['validation'].append({
                'case': name, 'forwards': len(rows), 'median_measured_ms': statistics.median(row['measured_ms'] for row in rows),
                'mean_absolute_error_percent': statistics.mean(errors), 'p95_absolute_error_percent': errors[int(0.95 * (len(errors) - 1))],
                'raw_mean_absolute_error_percent': statistics.mean(abs(row['raw_error_percent']) for row in rows), 'rows': rows,
            })
    (ROOT / ('eager-overhead-calibration.json' if args.prepare_only else 'eager-overhead-forward-validation.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'profile': overhead, 'calibration_forwards': len(calibration), 'validation': [{key: value for key, value in row.items() if key != 'rows'} for row in report.get('validation', [])]}))


if __name__ == '__main__':
    main()
