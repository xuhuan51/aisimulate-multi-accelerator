"""Publish qualified raw Graph observations into an isolated pilot profile.

Existing schemas and key columns are preserved. Small Decode roles and Prefill
retain explicitly recorded historical proxies. They are not relabeled Graph.
"""
import hashlib
import json
from pathlib import Path
import statistics
import shutil

import pyarrow as arrow
import pyarrow.parquet as parquet
import yaml


root = Path('/runtime')
raw = Path('/opt/qwen-graph-raw')
target = Path('/opt/qwen-graph-systems')
version = 'q397-graph-major-pilot-20261108'
contract = 'qwen35_397b_modelslim_fp32_state_graph_tp4_dp4_ep16/v1'
runtime = dict(image='sha256:ed48c44e8da49820e3f935d38a8cdb623e184b27361afdf2d1e6e3ecd4e741e0',
               framework_commit='a6b542813ff46de8ba3856e0236f1af35f4fca54',
               torch='2.10.0+cpu',torch_npu='2.10.0',deep_ep='1.0.0+d9261669.cann.9.0.0.b250',
               model='Qwen3.5-397B-A17B',tp=4,dp=4,ep=16,ssm_dtype='float32',capacity=16)
identity = hashlib.sha256(json.dumps(runtime,sort_keys=True).encode()).hexdigest()
sources = {}
new_rows = dict(qwen=[],linear=[],hccl=[])
manifest = json.loads((raw/'graph-raw-manifest.json').read_text())
assert manifest['runtime_image'] == runtime['image']
assert manifest['framework_commit'] == runtime['framework_commit']
assert (manifest['tp'], manifest['dp'], manifest['ep']) == (4, 4, 16)
assert manifest['ssm_dtype'] == 'float32' and manifest['max_dispatch_tokens_per_rank'] == 16
expected_sources = {entry['path']:entry['sha256'] for entry in manifest['files']}
expected_sources.update({
    'ep-graph-qualified-rows-10.json': 'afb2dd4e45934d08f0b5eeeb3f4e89c917c5e6e5670e35e08dfd7af8178eef51',
    'qwen-pair-m1-r4/results.json': '1acd03de662a7c9f6ee48818f6fe881482fabe0dc84131f258ed1f7171e137a5',
})
for relative, expected in expected_sources.items():
    assert hashlib.sha256((raw/relative).read_bytes()).hexdigest() == expected, relative


def read(relative):
    path = raw/relative
    sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sources[relative] == expected_sources[relative], relative
    return json.loads(path.read_text())


def verify_checks(value):
    if isinstance(value, dict):
        for key,item in value.items():
            if key in ('pass','passed','untouched','inactive_unchanged','noop_rejected','no_op_rejected'):
                assert item is True, (key,item)
            verify_checks(item)
    elif isinstance(value, list):
        for item in value:
            verify_checks(item)


def common(row, filename, index, checks):
    assert checks
    verify_checks(checks)
    samples = row['samples_ms']
    assert len(samples) >= 10 and all(value>0 for value in samples)
    cv = statistics.stdev(samples)/statistics.mean(samples)
    assert cv <= .1, (filename,index,cv)
    assert abs(statistics.median(samples)-row.get('latency_ms',row.get('median_ms'))) < 1e-12
    return dict(latency_ms=statistics.median(samples),cv=cv,sample_count=len(samples),
                runtime_identity_sha256=identity,execution_verification='captured_graph_replay',
                source_sha256=sources[filename],source_file=filename,source_row=index)


def custom(row, filename, index, checks):
    value = {key:row[key] for key in ('role','tp','phase','tokens','history')}
    value.update(common(row,filename,index,checks),boundary_contract=contract,
                 numerical_checks=len(checks),replication_reason=None,
                 signature_sha256=hashlib.sha256(json.dumps([sources[filename],index,value],sort_keys=True).encode()).hexdigest())
    new_rows['qwen'].append(value)


for tag in ('gdn-graph-ascend-r1','conv-graph-ascend-r1','conv-norm-graph-independent-r0','vector-graph-independent-r1'):
    filename=tag+'/rows.json'
    for index,row in enumerate(read(filename)):
        if tag=='conv-norm-graph-independent-r0' and row['role']=='gdn_conv':
            continue
        assert row.get('status','measured') in ('measured','stable')
        assert row['no_op_rejected']
        checks=row.get('checks',[row.get('before'),row.get('after')])
        custom(row,filename,index,checks)

filename='expert-graph-independent-r0.json'
expert=read(filename)
assert expert['status']=='completed' and len(expert['cases'])==1
row=expert['cases'][0]
assert row['physical_rows']==2560 and row['received_tokens']==10 and row['execution_mode']=='graph'
assert all(check['sampled_elements']>0 for check in row['checks'])
row.update(role='routed_experts',history=0)
custom(row,filename,0,row['checks'])

# The fresh10-repeat communication campaign satisfies the Qwen loader's
# existing sample gate. Earlier7-repeat diagnostics remain in raw evidence.
filename='ep-graph-qualified-rows-10.json'
communication=read(filename)
assert communication['status']=='qualified'
for index,row in enumerate(communication['points']):
    assert row['qualified'] and row['tokens']==1 and row['sample_count']==10
    measured=read(row['source_file'])
    measured_case,=measured['cases']
    assert measured['status']=='completed' and measured_case['validation']=='passed'
    assert len(measured_case['rank_results'])==16
    assert all(item['checked_elements']>0 for item in measured_case['rank_results'])
    custom(row,filename,index,measured_case['rank_results'])

filename='linear-graph-independent-r0/rows.json'
for index,row in enumerate(read(filename)):
    assert row['status']=='stable' and row['no_op_rejected']
    value={key:row[key] for key in ('quant_mode','n','k','m','weight_format')}
    value.update(common(row,filename,index,row['checks']),quant_bias=False,execution_mode='graph',
                 input_dtype='bfloat16',output_dtype='bfloat16',numerical_checks=len(row['checks']),
                 kernel_source='torch.nn.functional.linear' if row['quant_mode']=='bfloat16'
                 else 'sglang.srt.hardware_backend.npu.quantization.linear_method_npu.NPUW8A8Int8DynamicLinearMethod')
    new_rows['linear'].append(value)

filename='hccl-graph-independent-r0/rows.json'
for index,row in enumerate(read(filename)):
    assert row['no_op_rejected'] and row['checked_elements_per_rank']>0 and len(row['rank_results'])==16
    value={key:row[key] for key in ('dtype','operation','input_bytes_per_rank','num_ranks','ranks_per_node','rank_layout','execution_mode')}
    value.update(common(row,filename,index,row['rank_results']),correctness_checks=16*3,
                 kernel_source='torch.distributed.hccl')
    new_rows['hccl'].append(value)

major_roles={row['role'] for row in new_rows['qwen']}
counts={}
for family,filename,kind in [('sglang','ascend_qwen_perf.parquet','qwen'),
                             ('sglang','ascend_linear_perf.parquet','linear'),
                             ('comm/hccl','hccl_perf.parquet','hccl')]:
    original,= (root/'qwen-systems/data/ascend910c'/family).rglob(filename)
    table=parquet.read_table(original)
    retained=table.to_pylist()
    if kind=='qwen':
        retained=[row for row in retained if not (row['phase']=='decode' and row['role'] in major_roles
                   and row['tp']==(16 if row['role'] in ('routed_experts','expert_dispatch','expert_combine') else 4))]
    output=target/'data/ascend910c'/family/version/filename
    output.parent.mkdir(parents=True,exist_ok=True)
    parquet.write_table(arrow.Table.from_pylist(retained+new_rows[kind],schema=table.schema),output)
    counts[kind]=dict(retained=len(retained),added=len(new_rows[kind]),sha256=hashlib.sha256(output.read_bytes()).hexdigest())
spec=yaml.safe_load((root/'qwen-systems/ascend910c.yaml').read_text())
spec['misc']['hccl_version']=version
(target/'ascend910c.yaml').write_text(yaml.safe_dump(spec))
shutil.copytree(target,root/'qwen-graph-systems',dirs_exist_ok=True)
case=json.loads((root/'qwen-local-case.json').read_text())
case['predictor_config'].update(backend_version=version,systems_paths=[str(target)])
case['predictor_config']['estimator_config']['op_level']['ascend'].update(
    decode_execution_mode='graph',shape_policy='measured',shared_expert_overlap=False)
(root/'qwen-graph-case.json').write_text(json.dumps(case,indent=2))
report=dict(status='published_pilot',runtime=runtime,runtime_identity_sha256=identity,counts=counts,sources=sources,
            scope='TP4/DP4/EP16,1..4 requests/DP. Major Decode kernels calibrated on current Graph runtime. '
                  'Small GQA/embedding/logits roles retain historical Eager proxies. Prefill retains BF16-state proxy. '
                  'Whole-model accuracy is pending formal replay, independent operator qualification does not certify it.')
(root/'qwen-graph-publication.json').write_text(json.dumps(report,indent=2))
config=yaml.safe_load((root/'local-qwen-smoke.yaml').read_text())
config['traffic'].update(source=dict(type='synthetic',input_tokens=8192,output_tokens=1024),
                          load=dict(type='concurrency',concurrency=32),stop=dict(requests=128))
config['engine'].update(backend_version=version,systems_paths=[str(target)])
config['engine']['estimator_config']['op_level']['ascend'].update(decode_execution_mode='graph')
config['engine']['workers']['aggregated']['scheduler'].update(
    chunked_prefill_size=8192, max_prefill_tokens=16384, ignore_eos=True)
(root/'qwen-graph-formal.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
print(json.dumps(report,indent=2))
