import argparse
import concurrent.futures
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'four-node-serving-20261101/compare_serving_predictions.py'
spec = importlib.util.spec_from_file_location('matched_serving_comparison', SOURCE)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--measurement-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image', default='aisimulate-910c:logits-repair-20261102')
    parser.add_argument('--concurrencies', type=int, nargs='+')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--overhead-profile', type=Path, action='append', default=[])
    parser.add_argument('--configurations', nargs='+')
    args = parser.parse_args()
    os.environ['AISIMULATE_910C_IMAGE'] = args.image
    jobs = comparison.make_jobs(args.measurement_root, args.output)
    if args.concurrencies:
        jobs = [job for job in jobs if job['concurrency'] in args.concurrencies]
    if args.configurations:
        jobs = [job for job in jobs if job['configuration'] in args.configurations]
    profiles = [json.loads(path.read_text()) for path in args.overhead_profile]
    if profiles:
        plan = json.loads((args.measurement_root / 'comparison-plan.json').read_text())
        layouts = {entry['label']: (entry['DECODE_TP_SIZE'] // entry['DECODE_DP_SIZE'], entry['DECODE_DP_SIZE'], entry['DECODE_EP_SIZE']) for entry in plan['configurations']}
        for job in jobs:
            matches = [profile for profile in profiles if (profile['attention_tp'], profile['attention_dp'], profile['moe_ep_size']) == layouts[job['configuration']]]
            if len(matches) != 1:
                raise ValueError(f"Need exactly one explicit overhead profile for {job['configuration']}")
            encoded = json.dumps(matches[0], separators=(',', ':'))
            job['command'].extend(['--set', f'engine.workers.decode.timing.estimator_config.op_level.ascend.eager_runtime_overhead={encoded}'])
            job['eager_runtime_overhead'] = matches[0]
    for job in jobs:
        job['original_logits_layout_audit'] = copy.deepcopy(job['logits_layout_audit'])
        for audit in job['logits_layout_audit']:
            audit['modeled_lm_head_tp'] = audit['actual_lm_head_tp']
            audit['aligned'] = True
            audit['scope'] = 'Effective repaired replay: role timing overrides match the audited server flags'
        for role in ('prefill', 'decode'):
            flags = {
                (row['actual_enable_dp_lm_head'], row['actual_enable_tp_lm_head_all_to_all'])
                for row in job['logits_layout_audit'] if row['role'] == role
            }
            if len(flags) != 1:
                raise ValueError(f'Nonuniform runtime flags for {role}: {flags}')
            dp_head, all_to_all = flags.pop()
            for field, value in [('enable_dp_lm_head', dp_head), ('enable_tp_lm_head_all_to_all', all_to_all)]:
                setting = f'engine.workers.{role}.timing.estimator_config.op_level.ascend.{field}={str(value).lower()}'
                job['command'].extend(['--set', setting])
        job['repair_scope'] = 'Runtime LM-head partition, conditional all-to-all, unpadded eager logits and any explicitly selected additive forward calibration'
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'comparison-jobs.json').write_text(json.dumps(
        [{key: value for key, value in job.items() if key != 'measured'} for job in jobs], indent=2,
    ) + '\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(comparison.run_job, jobs))
    (args.output / 'comparison.json').write_text(json.dumps({
        'scope': 'Repaired offline replay against original completed serving measurements; not independent serving validation',
        'error_formula': '100 * (predicted / measured - 1)', 'rows': rows,
    }, indent=2) + '\n')
    with (args.output / 'comparison.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
