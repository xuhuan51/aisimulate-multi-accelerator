"""Recompute Eager residuals against the current operator contract using frozen independent forwards."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import statistics

import aisimulate._runtime as native
from aisimulate_core.sdk import RustForwardPassPerfModel
import yaml


root = Path('/opt/pd-chunk-20261108')
prepared = json.loads((root/'prepared.json').read_text())
reports, profiles = [], {}
for topology, filename, slope_case, attention_dp in (
    ('tp16-adp4', 'paired-forward-analysis.json', 'validation-c4', 4),
    ('tp32-adp32', 'tp32-paired-forward-analysis.json.gz', 'validation-c8', 32),
):
    source = root/'profile-source'/filename
    payload = source.read_bytes()
    data = json.loads(gzip.decompress(payload) if source.suffix=='.gz' else payload)
    cases = {case['case']:case['decode_rows'] for case in data['cases']} if 'cases' in data else data['forwards']
    assert len(cases['calibration-c1']) >= 100 and len(cases[slope_case]) >= 100
    job = next(job for job in prepared['jobs'] if job['mode']=='K' and
               yaml.safe_load(Path(job['config']).read_text())['engine']['workers']['decode']['parallelism']['attention_data']==attention_dp)
    public = yaml.safe_load(Path(job['config']).read_text())['engine']
    worker = public['workers']['decode']
    parallel = worker['parallelism']
    config = dict(model=public['model'],system=public['hardware'],backend=public['backend'],
                  backend_version=public['backend_version'],worker_type='decode',tp=parallel['tensor'],
                  attention_dp=attention_dp,moe_tp_size=parallel['moe_tensor'],moe_ep_size=parallel['moe_expert'],
                  kv_block_size=128,nextn=0,estimation_mode='op_level',fallback_policy='deny',
                  database_mode='SILICON',systems_paths=public['systems_paths'],
                  estimator_config=copy.deepcopy(worker['timing']['estimator_config']))
    for name in ('gemm_quant_mode','moe_quant_mode','fmha_quant_mode','kvcache_quant_mode','comm_quant_mode'):
        config[name] = worker['timing'].get(name,public[name])
    settings = config['estimator_config']['op_level']['ascend']
    assert settings['decode_execution_mode']=='eager' and settings['decode_timing_scope']=='isolated_kernels'
    assert 'eager_runtime_overhead' not in settings
    model = RustForwardPassPerfModel.best_available(config)
    evaluated = {}
    try:
        for name, rows in cases.items():
            values = []
            for row in rows:
                assert len(row['metrics']) == attention_dp
                assert all(rank['scheduled_requests']['num_prefill_requests']==0 for rank in row['metrics'])
                prediction = model.estimate_forward_pass_time_ms(row['metrics'])
                active = sum(rank['scheduled_requests']['num_decode_requests']>0 for rank in row['metrics'])
                values.append(dict(metrics=row['metrics'],active_ranks=active,measured_ms=row['measured_ms'],
                                   prediction_ms=prediction,residual_ms=row['measured_ms']-prediction))
            evaluated[name] = values
            print(topology, name, len(values), flush=True)
    finally:
        model.close()
    assert all(row['active_ranks']==1 for row in evaluated['calibration-c1'])
    base = statistics.median(row['residual_ms'] for row in evaluated['calibration-c1'])
    training = evaluated[slope_case]
    assert all(row['active_ranks']>1 for row in training)
    slope = statistics.median((row['residual_ms']-base)/(row['active_ranks']-1) for row in training)
    assert base >= 0 and slope >= 0
    identity = hashlib.sha256(json.dumps(dict(config=config,base_rows=evaluated['calibration-c1'],
                                              rank_rows=training),sort_keys=True).encode()).hexdigest()
    profile = dict(calibration_id='pd-eager-recomputed-v2-'+identity[:16],operator_contract='glm52_eager_metadata/v2',
                   attention_tp=parallel['tensor'],attention_dp=attention_dp,moe_ep_size=parallel['moe_expert'],
                   prefill_ms=0.0,decode_ms=base,decode_per_extra_active_rank_ms=slope)
    profiles[attention_dp] = profile
    corrected_config = copy.deepcopy(config)
    corrected_config['estimator_config']['op_level']['ascend']['eager_runtime_overhead'] = profile
    corrected = RustForwardPassPerfModel.best_available(corrected_config)
    summaries = []
    try:
        for name, rows in evaluated.items():
            for row in rows:
                row['corrected_ms'] = corrected.estimate_forward_pass_time_ms(row['metrics'])
            summaries.append(dict(case=name,forwards=len(rows),used_for_fit=name in ('calibration-c1',slope_case),
                                  before_mape=statistics.mean(abs(row['prediction_ms']/row['measured_ms']-1)*100 for row in rows),
                                  after_mape=statistics.mean(abs(row['corrected_ms']/row['measured_ms']-1)*100 for row in rows)))
    finally:
        corrected.close()
    reports.append(dict(topology=topology,profile=profile,config=config,fit_base_case='calibration-c1',
                        fit_rank_case=slope_case,source_sha256=hashlib.sha256(payload).hexdigest(),
                        source=str(source),cases=summaries,forwards=evaluated))
    print(json.dumps(dict(topology=topology,profile=profile,cases=summaries)),flush=True)

jobs = []
for original in prepared['jobs']:
    if original['mode'] != 'K':
        continue
    job = copy.deepcopy(original)
    config = yaml.safe_load(Path(job['config']).read_text())
    worker = config['engine']['workers']['decode']
    worker['timing']['estimator_config']['op_level']['ascend']['eager_runtime_overhead'] = profiles[worker['parallelism']['attention_data']]
    job['job'] = 'R2-'+original['job'][2:]
    job['mode'] = 'R2'
    path = root/'configs'/f"{job['job']}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    job['config'] = str(path)
    jobs.append(job)
(root/'runtime-v2-prepared.json').write_text(json.dumps(dict(jobs=jobs,unique_jobs=len(jobs)),indent=2))
report = dict(status='recalibrated',operator_contract='glm52_eager_metadata/v2',topologies=reports,
              native_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
              scope='New residuals computed from independent C1 and named active-rank training forwards against the current operator model. '
                    'The historical PD serving values are excluded from fitting. Prefill residual stays zero. '
                    'Residual includes stream idle, host submission and operator omissions; its components are not individually identified.')
(root/'runtime-v2-calibration.json').write_text(json.dumps(report,indent=2))
