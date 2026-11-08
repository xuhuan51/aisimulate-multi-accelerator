"""Prepare a finite four-node PD and chunk search using the revalidated Eager profile."""
import argparse
import json
from pathlib import Path

import yaml


root = Path('/opt/pd-chunk-20261108')
parser = argparse.ArgumentParser()
parser.add_argument('--wide-decode', action='store_true')
args = parser.parse_args()
prepared = json.loads((root/'runtime-v2-prepared.json').read_text())
source_group = 'A' if args.wide_decode else 'B'
source = next(job for job in prepared['jobs'] if job['group']==source_group and job['concurrency']==16)
config = yaml.safe_load(Path(source['config']).read_text())
assert config['engine'].pop('decoder_replay',False) is False
config['traffic']['load']['concurrency'] = {'choices':[4,8,16,32,64]}
config['traffic']['stop'] = {'requests_per_load_unit':2}
for role in ('prefill','decode'):
    worker = config['engine']['workers'][role]
    worker['parallelism']['preset'] = False
    worker['parallelism']['replicas'] = {'choices':[1,2,3]}
    # Historical prediction files contain materialized default fields that the
    # recommendation schema leaves implicit. Preserve their effective values.
    assert worker['scheduler'].pop('prefill_schedule_interval',1)==1
    assert worker['scheduler'].pop('prefill_decode_interval',0)==0
    assert worker['kv_cache']['capacity'].pop('cuda_graph_reserved_bytes',0)==0
    for name in ('gemm_quant_mode','moe_quant_mode','fmha_quant_mode','kvcache_quant_mode','comm_quant_mode'):
        value = worker['timing'].pop(name,None)
        assert value in (None,config['engine'][name])
    assert worker['timing'].pop('attention_backend',None) is None
config['engine']['workers']['prefill']['scheduler']['chunked_prefill_size'] = {'choices':[4096,8192,16384]}
config['optimization'] = dict(target='throughput',constraints=dict(min_candidate_gpus=64,max_candidate_gpus=64))
config['optimizer'] = dict(algorithm='random',max_trials=100,parallelism=2,candidate_timeout_seconds=180,seed=910)
path = root/('pd-chunk-search-wide-decode.yaml' if args.wide_decode else 'pd-chunk-search.yaml')
path.write_text(yaml.safe_dump(config,sort_keys=False))
print(path)
print('Expected finite domain:',15 if args.wide_decode else 45,'candidates')
