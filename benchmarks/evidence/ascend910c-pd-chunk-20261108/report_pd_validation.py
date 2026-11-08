"""Audit matched PD runs, report absolute errors and rank agreement, and plot them."""
import collections
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


root = Path('/opt/pd-chunk-20261108')
manifest = json.loads(next((root/'inherited').rglob('manifest.json')).read_text())
baseline = json.loads((root/'baseline-all/status.json').read_text())
repaired = json.loads((root/'runtime-v2-all/status.json').read_text())
assert baseline['completed']==baseline['total']==62
assert repaired['completed']==repaired['total']==31
jobs = {row['job']:row for row in baseline['results']+repaired['results']}
rows = []
for record in manifest['records']:
    historical = record['historical']
    actual = record['actual']
    for mode in ('K','L','R2'):
        key = record['jobs']['K' if mode=='R2' else mode]
        if mode=='R2':
            key = 'R2-'+key[2:]
        job = jobs[key]
        row = dict(number=record['number'],nodes=4 if historical['campaign']=='four-node' else 2,
                   configuration=historical['configuration'],concurrency=historical['concurrency'],
                   requests=historical['requests'],input_length=historical['input_length'],
                   output_length=historical['output_length'],mode=mode,status=job['status'],job=key,
                   measurement_sha256=record['measurement_sha256'],source_config_sha256=job['original_config_sha256'])
        if job['status']=='passed':
            metrics = job['metrics']
            assert metrics['completed_requests']==historical['requests'] and metrics['failed_requests']==0
            assert metrics['total_input_tokens']==historical['input_length']*historical['requests']
            assert metrics['total_output_tokens']==historical['output_length']*historical['requests']
            for name, prediction_key, measured_key in (
                ('throughput','output_throughput_tok_s','output_tok_s'),
                ('ttft','mean_ttft_ms','ttft_ms'),('tpot','mean_tpot_ms','tpot_ms'),
            ):
                row[name+'_predicted'] = metrics[prediction_key]
                row[name+'_measured'] = actual[measured_key]
                row[name+'_error_percent'] = 100*(metrics[prediction_key]/actual[measured_key]-1)
        rows.append(row)

summaries = []
for nodes in (4,2,None):
    for mode in ('K','L','R2'):
        selected = [row for row in rows if row['mode']==mode and (nodes is None or row['nodes']==nodes)]
        successful = [row for row in selected if row['status']=='passed']
        summary = dict(nodes=nodes,mode=mode,completed=len(successful),total=len(selected))
        for metric in ('throughput','ttft','tpot'):
            values = [row[metric+'_error_percent'] for row in successful]
            summary[metric+'_mape'] = statistics.mean(map(abs,values))
            summary[metric+'_error_range_percent'] = [min(values),max(values)]
        summary['throughput_below20_count'] = sum(abs(row['throughput_error_percent'])<20 for row in successful)
        summaries.append(summary)

def ranks(values):
    return [sum(other>value for other in values)+(sum(other==value for other in values)+1)/2 for value in values]


rank_groups = collections.defaultdict(list)
for row in rows:
    if row['nodes']==4 and row['status']=='passed':
        rank_groups[(row['mode'],row['input_length'],row['output_length'],row['concurrency'],row['requests'])].append(row)
rankings=[]
for key,selected in rank_groups.items():
    if len(selected)<2:
        continue
    actual = [row['throughput_measured'] for row in selected]
    predicted = [row['throughput_predicted'] for row in selected]
    winner = max(range(len(selected)),key=lambda i:predicted[i])
    actual_winner = max(range(len(selected)),key=lambda i:actual[i])
    rankings.append(dict(mode=key[0],input_length=key[1],output_length=key[2],concurrency=key[3],requests=key[4],
                         candidates=len(selected),spearman=statistics.correlation(ranks(actual),ranks(predicted)),
                         top1_agrees=winner==actual_winner,
                         throughput_regret_percent=100*(max(actual)-actual[winner])/max(actual),
                         predicted_order=[row['configuration'] for row in sorted(selected,key=lambda row:row['throughput_predicted'],reverse=True)],
                         actual_order=[row['configuration'] for row in sorted(selected,key=lambda row:row['throughput_measured'],reverse=True)]))

report = dict(status='historical_pd_validation_completed',model='GLM-5.2 ModelSlim W8A8',
              runtime='Ascend910C / SGLang0.5.22, Eager, schedule overlap disabled, shared-expert overlap enabled',
              source_archive_sha256='3cf9767854f5436168aea80df599128ab834c80fcd405388c7caa51791f2961d',
              summaries=summaries,rows=rows,rankings=rankings,
              calibration_scope='R2 refits only independent forward C1 plus the predeclared active-rank cases against operator contract v2. '
                                'Historical PD serving measurements are validation targets. It preserves existing native estimation logic.',
              limitations=['Historical serving data include39 observations of31 unique workloads. Four-node subset has23 observations.',
                           'L uses measured runtime-layer composition and covers20of23 four-node observations. It stays a separate reference.',
                           'TTFT cold allocator/JIT and first nonempty text remain outside the steady calibration.',
                           'Eager residual combines host submission, stream idle and operator omissions. Their individual costs are unresolved.',
                           'Fixed role capacities preserve the historical minimum observed capacity per role. Per-replica differences remain approximated.'])
(root/'pd-validation.json').write_text(json.dumps(report,indent=2))

selected = [row for row in rows if row['mode']=='R2' and row['nodes']==4]
colors = {'recommended-2p1d-dtp32':'#7254a0','control-1p3d-tp16':'#2a80b9',
          'control-2p2d-tp16':'#2b9d8f','control-3p1d-tp16':'#d78935'}
fig,axes = plt.subplots(1,3,figsize=(12.8,4.4))
for axis,metric,label in zip(axes,('throughput','tpot','ttft'),('Output throughput (token/s)','Mean TPOT (ms)','Mean TTFT (ms)')):
    values = [row[metric+'_measured'] for row in selected]
    lower,upper = min(values)*.7,max(values)*1.2
    axis.plot([lower,upper],[lower,upper],color='#586878',lw=1)
    axis.fill_between([lower,upper],[lower*.8,upper*.8],[lower*1.2,upper*1.2],color='#e6edf4')
    for name,color in colors.items():
        points=[row for row in selected if row['configuration']==name]
        axis.scatter([row[metric+'_measured'] for row in points],
                     [row[metric+'_predicted'] for row in points],color=color,label=name,s=24,zorder=3)
    if metric in ('throughput','ttft'):
        axis.set_xscale('log')
        axis.set_yscale('log')
    axis.set_xlabel('Measured '+label)
    axis.set_ylabel('Predicted '+label)
    mape=next(summary[metric+'_mape'] for summary in summaries if summary['nodes']==4 and summary['mode']=='R2')
    axis.set_title(f'MAPE {mape:.2f}%')
    axis.spines[['top','right']].set_visible(False)
fig.suptitle('GLM-5.2 · four-node PD ·23 historical observations · current Eager calibration')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8)
fig.tight_layout(rect=(0,.06,1,.94))
for suffix in ('png','svg'):
    fig.savefig(root/f'pd-validation.{suffix}',dpi=180)

fig,axes=plt.subplots(1,2,figsize=(10.5,4.1))
concurrency16=[row for row in rows if row['nodes']==4 and row['concurrency']==16]
names=list(colors)
for index,(mode,label,color) in enumerate((('K','Kernel sum','#91a4b8'),('R2','Recalibrated Eager','#2578ad'))):
    values=[next(row['throughput_predicted'] for row in concurrency16 if row['configuration']==name and row['mode']==mode) for name in names]
    axes[0].bar([i+index*.25-.25 for i in range(4)],values,width=.25,label=label,color=color)
actual=[next(row['throughput_measured'] for row in concurrency16 if row['configuration']==name) for name in names]
axes[0].bar([i+.25 for i in range(4)],actual,width=.25,label='Real serving',color='#ef9f36')
axes[0].set_xticks(range(4),['2P1D / D32','1P3D','2P2D','3P1D'])
axes[0].set_ylabel('Output throughput (token/s)')
axes[0].set_title('Matched C16/N32,8K/1K')
axes[0].legend(fontsize=8)
for index,(mode,color) in enumerate((('K','#91a4b8'),('R2','#2578ad'))):
    summary=next(row for row in summaries if row['nodes']==4 and row['mode']==mode)
    axes[1].bar([i+index*.32-.16 for i in range(3)],
                [summary[name+'_mape'] for name in ('throughput','tpot','ttft')],width=.32,label=mode,color=color)
axes[1].set_xticks(range(3),['Throughput','TPOT','TTFT'])
axes[1].set_ylabel('MAPE (%)')
axes[1].set_title('Same23 four-node observations')
axes[1].legend()
for axis in axes:
    axis.spines[['top','right']].set_visible(False)
fig.tight_layout()
for suffix in ('png','svg'):
    fig.savefig(root/f'pd-before-after.{suffix}',dpi=180)
print(json.dumps(dict(summaries=summaries,rankings=rankings),indent=2))
