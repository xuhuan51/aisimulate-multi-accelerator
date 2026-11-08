"""Verify finite-domain coverage, candidate wiring and saved top1 replay."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config.cli import CorePredictionConfig


root = Path('/opt/pd-chunk-20261108')
records, campaigns = [], []
for directory, expected, replay in (
    ('joint-search-qualified',45,'top1-local-replay'),
    ('joint-search-wide-decode',15,'top1-wide-replay'),
):
    result = json.loads((root/directory/'recommendation.json').read_text())
    counts = result['counts']
    assert counts['evaluated']==counts['feasible']==expected
    assert all(counts[key]==0 for key in ('failed','infeasible','resource_limited','timed_out','unsupported'))
    assert len(result['candidates'])==expected
    for candidate in result['candidates']:
        assert candidate['status']=='feasible' and candidate['used_gpus']==64
        sample = candidate['config']
        chunk = sample['prefill_chunked_prefill_size']
        assert chunk in (4096,8192,16384)
        metrics = candidate['metrics']
        assert metrics['completed_requests']==2*sample['concurrency'] and metrics['failed_requests']==0
        assert candidate['score']==metrics['output_throughput_tok_s']
        saved = CorePredictionConfig.model_validate(candidate['prediction_config'])
        assert saved.engine.workers.prefill.scheduler.chunked_prefill_size==chunk
        deployment = prediction_to_replay_spec(saved).backend_deployment
        assert deployment.prefill_engine_args['sglang']['chunked_prefill_size']==chunk
        profile = saved.engine.workers.decode.timing.estimator_config['op_level']['ascend']['eager_runtime_overhead']
        assert profile['operator_contract']=='glm52_eager_metadata/v2'
        assert profile['attention_tp']==sample['decode_tp']
        assert profile['attention_dp']==sample['decode_attention_dp']
        assert profile['moe_ep_size']==sample['decode_moe_ep']
        records.append(dict(campaign=directory,candidate_id=candidate['candidate_id'],
                            prefill_replicas=sample['prefill_replicas'],decode_replicas=sample['decode_replicas'],
                            decode_tp=sample['decode_tp'],decode_dp=sample['decode_attention_dp'],
                            chunk=chunk,concurrency=sample['concurrency'],
                            predicted_throughput=metrics['output_throughput_tok_s'],
                            predicted_ttft_ms=metrics['mean_ttft_ms'],predicted_tpot_ms=metrics['mean_tpot_ms']))
    best = max(result['candidates'],key=lambda row:row['score'])
    repeated = json.loads((root/replay/'prediction.json').read_text())
    for field in ('output_throughput_tok_s','mean_ttft_ms','mean_tpot_ms','completed_requests'):
        assert repeated[field]==best['metrics'][field], field
    campaigns.append(dict(directory=directory,counts=counts,top1=best['candidate_id'],
                          top1_replay_identical=True))

assert len(records)==60
unique={(row['prefill_replicas'],row['decode_replicas'],row['decode_tp'],row['decode_dp'],row['chunk'],row['concurrency']) for row in records}
assert len(unique)==60
exploratory = json.loads((root/'joint-search-r1/recommendation.json').read_text())
failures=[row for row in exploratory['candidates'] if row['status']!='feasible']
assert len(failures)==15 and all(row['config']['prefill_chunked_prefill_size']==32768 for row in failures)
assert all('outside measured' in str(row['reason']) for row in failures)
report=dict(status='passed',campaigns=campaigns,candidates=records,
            best_predicted=max(records,key=lambda row:row['predicted_throughput']),
            exploratory_missing_data_candidates=15,
            limitations=['Global chunk32768 produced per-rank DeepEP payload2048 outside the measured1to1024 range. Its15 failures remain in exploratory evidence.',
                         'Qualified domain4096/8192/16384 has60 completed candidates in two topology-specific calibration campaigns.',
                         'Historical serving data verify chunk16384. Other chunk choices have operator-data coverage and functional replay evidence; their serving ranking remains unmeasured.',
                         'Search uses known fixed per-role KV capacities. Other hardware, runtime, model weights and new capacity policies require separate verification.'])
(root/'joint-search-verification.json').write_text(json.dumps(report,indent=2))

labels=['2P1D / D32','1P3D','2P2D','3P1D']
def matches(row,label):
    return (row['decode_dp']==32 if label==labels[0] else
            row['decode_dp']==4 and f"{row['prefill_replicas']}P{row['decode_replicas']}D"==label)

fig,axes=plt.subplots(1,2,figsize=(10.5,4.4))
for axis,field,title in zip(axes,('predicted_throughput','predicted_ttft_ms'),
                            ('Predicted output throughput (token/s)','Predicted mean TTFT (ms)')):
    values=np.array([[next(row[field] for row in records if matches(row,label) and row['concurrency']==64 and row['chunk']==chunk)
                      for chunk in (4096,8192,16384)] for label in labels])
    matrix=axis.imshow(values,aspect='auto',cmap='Blues' if field=='predicted_throughput' else 'YlOrBr')
    for i in range(4):
        for j in range(3):
            axis.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=9,
                      color='white' if values[i,j]>values.max()*.65 else '#172b4d')
    axis.set_xticks(range(3),['4096','8192','16384'])
    axis.set_yticks(range(4),labels)
    axis.set_xlabel('Global chunk tokens, attention DP4')
    axis.set_title(title,fontsize=10)
    fig.colorbar(matrix,ax=axis,fraction=.04)
fig.suptitle('GLM-5.2 · four-node budget · C64/N128 ·60-candidate search')
fig.text(.5,.015,'Other chunk choices have prediction evidence. Whole-serving validation uses historical chunk16384.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.045,1,.94))
for suffix in ('png','svg'):
    fig.savefig(root/f'pd-chunk-search.{suffix}',dpi=180)
print(json.dumps(dict(status='passed',completed=60,best=report['best_predicted'],
                      top1_replay_identical=True,exploratory_missing_data_candidates=15),indent=2))
