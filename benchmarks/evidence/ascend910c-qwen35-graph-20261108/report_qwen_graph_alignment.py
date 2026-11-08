"""Compare canonical CPU predictions with the completed Qwen serving run."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from aisimulate_core.sdk import RustForwardPassPerfModel


root = Path('/runtime')
raw = Path('/opt/qwen-graph-raw')
real_root = Path('/opt/qwen-graph-complete/qwen-graph-fp32-r1/formal-validation-r0')
real = json.loads((real_root/'formal-metrics.json').read_text())
requests = [json.loads(line) for line in (real_root/'requests.jsonl').read_text().splitlines()]
assert real['status'] == 'passed' and len(requests) == 128
assert len({row['meta']['id'] for row in requests}) == 128
assert all(row['input_tokens'] == 8192 and row['output_tokens'] == 1024 for row in requests)
assert all(row['meta']['num_retractions'] == 0 for row in requests)
assert hashlib.sha256((real_root/'formal-corpus.jsonl').read_bytes()).hexdigest() == real['corpus_sha256']
predictions = {}
for label, directory, case_file in [('old', 'qwen-eager-formal-control-r2', 'qwen-local-case.json'),
                                    ('repaired', 'qwen-graph-formal-r5', 'qwen-graph-case.json')]:
    metrics = json.loads((root/directory/'prediction.json').read_text())
    assert metrics['completed_requests'] == 128 and metrics['failed_requests'] == 0
    assert metrics['total_input_tokens'] == 1048576 and metrics['total_output_tokens'] == 131072
    case = json.loads((root/case_file).read_text())
    model = RustForwardPassPerfModel.best_available(case['predictor_config'])
    try:
        operations = model.static_phase_diagnostics(batch_size=3, context_length=8192, prefill=False)
    finally:
        model.close()
    predictions[label] = dict(throughput_token_s=metrics['output_throughput_tok_s'],
                              throughput_error_percent=100*(metrics['output_throughput_tok_s']/real['output_throughput']-1),
                              median_ttft_ms=metrics['median_ttft_ms'], median_tpot_ms=metrics['median_tpot_ms'],
                              fixed_forward_ms=sum(row['latency_ms'] for row in operations), operations=operations)

measured_ttft = statistics.median(row['ttft_ms'] for row in requests)
measured_tpot = statistics.median(row['tpot_ms'] for row in requests)
for prediction in predictions.values():
    prediction['ttft_error_percent'] = 100*(prediction['median_ttft_ms']/measured_ttft-1)
    prediction['tpot_error_percent'] = 100*(prediction['median_tpot_ms']/measured_tpot-1)

summary = json.loads((raw/'steady-profile-r0/steady-kernel-summary.json').read_text())
timeline = json.loads(gzip.open(raw/'steady-profile-r0/steady-kernel-timeline.json.gz', 'rt').read())
report = dict(status='formal_shape_matched_validation_completed', model='Qwen3.5-397B-A17B ModelSlim',
              attention_tp=4, attention_dp=4, moe_ep=16, concurrency=32, requests=128,
              input_tokens=8192, output_tokens=1024, kv_tokens_per_dp=32768,
              global_chunk_tokens=8192, per_dp_chunk_tokens=2048, ignore_eos=True,
              measured_throughput_token_s=real['output_throughput'], measured_p50_ttft_ms=measured_ttft,
              measured_p50_tpot_ms=measured_tpot, predictions=predictions,
              throughput_target_passed=abs(predictions['repaired']['throughput_error_percent']) < 20,
              real_corpus_sha256=real['corpus_sha256'],
              limitations=['Major Decode data use the current Graph/FP32-state runtime. Prefill and six small Decode roles retain historical proxies.',
                           'CPU synthetic replay matches shapes and scheduling bounds. It does not execute the actual prompt token contents or replay their expert routes.',
                           'Historical Eager/BF16 prediction is an old-model control against Graph/FP32 serving.',
                           'The profiler timeline is diagnostic. Category durations overlap and contain synchronization waits.',
                           'Only TP4/DP4/EP16 and 1..4 real requests per DP are calibrated.'],
              raw_archive_sha256='70540229f9c0dc3a1532759392fdf385f85ad86161ae5b51deac6dca5d48a16d')
(root/'qwen-graph-formal-comparison.json').write_text(json.dumps(report, indent=2)+'\n')

fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1))
values = [predictions['old']['throughput_token_s'], predictions['repaired']['throughput_token_s'], real['output_throughput']]
bars = axes[0].bar(['Old prediction', 'Repaired', 'Real serving'], values, color=['#91a4b8','#2578ad','#ef9f36'])
for bar, value in zip(bars, values):
    axes[0].text(bar.get_x()+bar.get_width()/2, value+8, f'{value:.2f}', ha='center')
axes[0].set_ylim(0, max(values)*1.24)
axes[0].set_ylabel('Output throughput (token/s)')
axes[0].set_title(f"Error {predictions['old']['throughput_error_percent']:.2f}% to {predictions['repaired']['throughput_error_percent']:.2f}%")
names = ['Conv', 'GDN update', 'Experts', 'Dispatch', 'Combine']
suffixes = ['_conv', '_scan', '_experts', '_dispatch', '_combine']
for index, label in enumerate(('old','repaired')):
    costs = [sum(op['latency_ms'] for op in predictions[label]['operations'] if op['name'].endswith(suffix)) for suffix in suffixes]
    axes[1].bar([i+index*.27-.27 for i in range(5)], costs, width=.27, label=label, color=['#91a4b8','#2578ad'][index])
kernels = timeline['ranks'][0]['kernels']
actual = [sum(k['duration_us'] for k in kernels if 'causal_conv1d' in k['type'])/1000,
          sum(k['duration_us'] for k in kernels if 'gating_delta_rule_update' in k['type'])/1000,
          summary['ranks'][0]['categories_union_ms']['routed_experts'],
          sum(k['duration_us'] for k in kernels if k['type']=='MoeLowLatencyDispatchV2')/1000,
          sum(k['duration_us'] for k in kernels if k['type']=='MoeLowLatencyCombineV2')/1000]
axes[1].bar([i+.27 for i in range(5)], actual, width=.27, label='profile rank0', color='#ef9f36')
axes[1].set_xticks(range(5), names)
axes[1].set_ylabel('Selected Decode operations (ms)')
axes[1].set_title('Diagnostic costs across distinct timing boundaries')
axes[1].legend(fontsize=8)
for axis in axes:
    axis.spines[['top','right']].set_visible(False)
fig.suptitle('Qwen3.5-397B-A17B · TP4/DP4/EP16 · 8K/1K · C32/N128')
fig.text(.5,.01,'Left uses full serving throughput. Right is diagnostic and includes waits. Prefill proxies remain.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.045,1,.94))
for suffix in ('png','svg'):
    fig.savefig(root/f'qwen-graph-formal-comparison.{suffix}',dpi=180)

colors = {'Dispatch':'#d85a57','Combine':'#e8a13a','HCCL':'#8b6db3','GDN':'#2f9f99','Expert':'#4479b4','Other':'#b7c4d2'}
def category(kernel):
    kind=kernel['type']
    if 'Dispatch' in kind:
        return 'Dispatch'
    if 'Combine' in kind:
        return 'Combine'
    if any(part in kind.lower() for part in ('allgather','all_gather','reducescatter','reduce_scatter')):
        return 'HCCL'
    if any(part in kind for part in ('causal_conv1d','delta_rule')):
        return 'GDN'
    if 'GroupedMatmul' in kind:
        return 'Expert'
    return 'Other'

figure, axis = plt.subplots(figsize=(12,6.2))
events=[]
for rank in timeline['ranks']:
    grouped=collections.defaultdict(list)
    for kernel in rank['kernels']:
        tag=category(kernel)
        grouped[tag].append((kernel['start_us']/1000,kernel['duration_us']/1000))
        events.append(dict(name=kernel['name'],cat=tag,ph='X',ts=kernel['start_us'],dur=kernel['duration_us'],
                           pid=rank['rank'],tid=kernel['stream'],args=dict(type=kernel['type'],shape=kernel['input_shapes'])))
    for tag,segments in grouped.items():
        axis.broken_barh(segments,(rank['rank']-.35,.7),facecolors=colors[tag],linewidth=0)
axis.set_yticks(range(16))
axis.set_ylabel('Rank, all streams overlaid within each lane')
axis.set_xlabel('Time from aligned cycle origin (ms)')
axis.set_title('Qwen real Decode timeline · one common complete cycle · all16 ranks')
axis.legend(handles=[plt.Line2D([0],[0],color=color,lw=5,label=name) for name,color in colors.items()],
            loc='upper center',bbox_to_anchor=(.5,-.10),ncol=6,fontsize=9)
figure.tight_layout(rect=(0,.03,1,1))
figure.savefig(root/'qwen-decode-all16-timeline.png',dpi=180)
(root/'qwen-decode-chrome-trace.json').write_text(json.dumps(dict(traceEvents=events,displayTimeUnit='ms')))
print(json.dumps(dict(measured=real['output_throughput'],
                      results={key:{k:v for k,v in row.items() if k!='operations'} for key,row in predictions.items()}),indent=2))
