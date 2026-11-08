"""Audit missing Graph identities and attempt both other topologies via the CLI."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess

import pyarrow.parquet as parquet
import yaml

from aisimulate_core.sdk.models.ascend_qwen35 import physical_specs
from aisimulate_core.qwen_modelslim import read_qwen35_modelslim


root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--output-dir',type=Path,required=True)
output = parser.parse_args().output_dir
output.mkdir(exist_ok=False)
systems = Path('/opt/qwen-graph-systems/data/ascend910c')
version = 'q397-graph-major-pilot-20261108'
custom_path = systems/'sglang'/version/'ascend_qwen_perf.parquet'
linear_path = systems/'sglang'/version/'ascend_linear_perf.parquet'
hccl_path = systems/'comm/hccl'/version/'hccl_perf.parquet'
custom = parquet.read_table(custom_path).to_pylist()
linear = parquet.read_table(linear_path).to_pylist()
hccl = parquet.read_table(hccl_path).to_pylist()
metadata = json.loads((Path('/runtime/qwen-model/config.json')).read_text())
metadata['modelslim_metadata'] = read_qwen35_modelslim('/runtime/qwen-model',metadata)

graph_roles = sorted({row['role'] for row in custom
                      if row['phase']=='decode' and 'graph' in row['boundary_contract']
                      and row['tp']==4})
linear_graph = {(row['quant_mode'],row['n'],row['k']) for row in linear
                if row['execution_mode']=='graph'}
hccl_graph = {(row['operation'],row['num_ranks']) for row in hccl
              if row['execution_mode']=='graph'}
jobs = []
for tp, dp in ((2,8),(8,2)):
    name = f'tp{tp}-dp{dp}-ep16'
    config = yaml.safe_load((root/'qwen-graph.yaml').read_text())
    parallel = config['engine']['workers']['aggregated']['parallelism']
    parallel.update(tensor=tp,attention_data=dp)
    config['engine']['context_length'] = 10240
    config['engine']['workers']['aggregated']['scheduler']['max_sequences'] = 32//dp
    config_path = output/f'{name}.yaml'
    config_path.write_text(yaml.safe_dump(config,sort_keys=False))

    # Both execution branches use the same matrix N/K geometry. Only inspect
    # those identities here. Eager timings never become Graph measurements.
    specs = physical_specs(metadata,tp,'decode',attention_dp=dp)
    required = sorted({(spec['AscendLinear']['key']['quant_mode'],
                        spec['AscendLinear']['key']['n'],
                        spec['AscendLinear']['key']['k'])
                       for spec in specs if 'AscendLinear' in spec})
    missing_linear = [dict(quant_mode=quant,n=n,k=k)
                      for quant,n,k in required if (quant,n,k) not in linear_graph]
    coverage = dict(
        tp=tp,dp=dp,ep=16,
        graph_custom_roles_missing_exact_tp=[role for role in graph_roles
            if not any(row['phase']=='decode' and row['role']==role and row['tp']==tp
                       and 'graph' in row['boundary_contract'] for row in custom)],
        required_linear_identities=len(required),
        present_linear_identities=len(required)-len(missing_linear),
        missing_linear_identities=missing_linear,
        missing_hccl_curves=[dict(operation=operation,num_ranks=tp)
            for operation in ('reduce_scatter','all_gather')
            if (operation,tp) not in hccl_graph],
        source_config=str(config_path),
        scope='Identity coverage only. Present identities still require token/byte-axis and runtime-signature verification.')
    jobs.append(dict(name=name,config_path=config_path,coverage=coverage))


def attempt(job):
    name = job['name']
    command = ['/opt/aisimulate-venv/bin/aisimulate','predict','-c',str(job['config_path']),
               '--output-dir',str(output/(name+'-prediction')),'--format','json']
    log_path = output/(name+'.log')
    with log_path.open('w') as log:
        process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                   start_new_session=True,env=os.environ.copy())
        try:
            code = process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            code = 'timeout'
    log = log_path.read_text()
    rejected = 'Qwen Graph calibration currently requires TP4/DP4/EP16' in log
    return dict(**job['coverage'],command=command,exit_code=code,
                status='blocked_by_current_graph_contract' if rejected else 'inspect_cli_result',
                log=log_path.name,prediction_metrics=None)


with ThreadPoolExecutor(max_workers=2) as pool:
    attempts = list(pool.map(attempt,jobs))
report = dict(status='coverage_audited',attempts=attempts,
              table_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in (custom_path,linear_path,hccl_path)},
              graph_custom_rows=sum('graph' in row['boundary_contract'] for row in custom),
              graph_linear_rows=sum(row['execution_mode']=='graph' for row in linear),
              graph_hccl_rows=sum(row['execution_mode']=='graph' for row in hccl),
              expert_graph_tokens=sorted({row['tokens'] for row in custom
                  if 'graph' in row['boundary_contract'] and row['role']=='routed_experts'}),
              limitations=[
                  'Both ordinary CLI attempts retain the production topology guard and deny fallback.',
                  'GDN TP2/TP8 Graph state/Conv/norm geometries need qualified data.',
                  'Shape-independent vector roles are reuse candidates pending physical signature and token-axis audit.',
                  'EP16 M1 expert and transport rows are potential reuse candidates. TP2 may also require M2. Current native token mapping is explicitly TP4-specific.',
                  'No target serving throughput or unmeasured Graph latency was inserted.'])
(output/'coverage.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
