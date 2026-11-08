"""Audit all matched GLM formal repeats without mixing forward and serving errors."""
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


root = Path('/runtime')
source = root / 'glm-ep-six-results-20261108'
rows = []
for topology in ('E4', 'E2'):
    for repeat in range(3):
        predicted_path = source / 'simulation' / f'{topology}-r{repeat}' / 'result.json'
        real_path = source / 'real' / f'{topology}-r{repeat}.json'
        predicted = json.loads(predicted_path.read_text())
        real = json.loads(real_path.read_text())
        metrics = predicted['metrics']
        assert predicted['status'] == real['status'] == 'passed'
        assert predicted['input_sha256'] == real['input_sha256']
        assert metrics['completed'] == real['completed'] == 128
        assert metrics['total_input'] == real['total_input'] == 1048576
        assert metrics['total_output'] == real['total_output'] == 131072
        assert metrics['aisimulate']['native_sha256'] == '0af85559427bc22f44f74700752136ef0eb566fd9137f020e41470b8f8330b8a'
        row = dict(topology=topology, tp=real['attention_tp'], dp=real['dp'], ep=16,
                   repeat=repeat, predicted_token_s=metrics['output_throughput'],
                   real_token_s=real['output_throughput'], corpus_sha256=real['input_sha256'],
                   throughput_error_percent=100*(metrics['output_throughput']/real['output_throughput']-1),
                   ttft_error_percent=100*(metrics['median_ttft_ms']/real['p50_ttft_ms']-1),
                   tpot_error_percent=100*(metrics['median_tpot_ms']/real['p50_tpot_ms']-1),
                   sources=[dict(path=str(path.relative_to(root)),
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                            for path in (predicted_path, real_path)])
        rows.append(row)

summaries = []
for topology in ('E4', 'E2'):
    selected = [row for row in rows if row['topology'] == topology]
    predicted = statistics.median(row['predicted_token_s'] for row in selected)
    measured = statistics.median(row['real_token_s'] for row in selected)
    summaries.append(dict(topology=topology, tp=selected[0]['tp'], dp=selected[0]['dp'], ep=16,
                          predicted_median_token_s=predicted, real_median_token_s=measured,
                          error_of_medians_percent=100*(predicted/measured-1),
                          repeat_error_range_percent=[min(row['throughput_error_percent'] for row in selected),
                                                      max(row['throughput_error_percent'] for row in selected)],
                          median_ttft_error_percent=statistics.median(row['ttft_error_percent'] for row in selected),
                          median_tpot_error_percent=statistics.median(row['tpot_error_percent'] for row in selected),
                          all_repeats_below_20_percent=all(abs(row['throughput_error_percent']) < 20 for row in selected)))

report = dict(status='six_matched_formal_replays_completed', model='GLM-5.2 ModelSlim W8A8',
              hardware='Ascend910C, 16 logical devices', input_tokens=8192, output_tokens=1024,
              concurrency=32, requests=128, kv_tokens_per_dp=32768, summaries=summaries, results=rows,
              scope='CPU native-SGLang replay against archived real serving with matched per-repeat corpus hashes. '
                    'Three new predictions per topology. Throughput target does not qualify TTFT or TPOT.',
              measurement_caveat='Communication steady curve excludes the first pair after observing its outlier. '
                                 'Raw all75 failed-CV data are preserved. Earlier old-package replay is baseline only.')
(root / 'glm-ep-formal-comparison-three-repeats.json').write_text(json.dumps(report, indent=2)+'\n')

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
for axis, summary in zip(axes, summaries):
    selected = [row for row in rows if row['topology'] == summary['topology']]
    values = [summary['predicted_median_token_s'], summary['real_median_token_s']]
    bars = axis.bar(['Repaired prediction', 'Real serving'], values,
                    color=['#2578ad', '#ef9f36'], width=.55)
    for i, key in enumerate(('predicted_token_s', 'real_token_s')):
        axis.scatter([i-.04, i, i+.04], [row[key] for row in selected], color='#25364a', s=17, zorder=3)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x()+bar.get_width()/2, value+max(values)*.03, f'{value:.2f}', ha='center')
    axis.set_ylim(0, max(values)*1.27)
    axis.set_ylabel('Output throughput (token/s)')
    axis.set_title(f"TP{summary['tp']} / DP{summary['dp']} / EP16")
    axis.text(.5, .92, f"Median error {summary['error_of_medians_percent']:.2f}%", ha='center', transform=axis.transAxes)
    axis.spines[['top', 'right']].set_visible(False)
fig.suptitle('GLM-5.2 · 8K/1K · C32/N128 · three matched repeats', fontsize=12)
fig.text(.5, .01, 'Bars show medians. Dots show individual repeats. Archived real serving vs CPU replay.', ha='center', fontsize=9)
fig.tight_layout(rect=(0, .045, 1, .94))
for suffix in ('png', 'svg'):
    fig.savefig(root / f'glm-ep-formal-comparison-three-repeats.{suffix}', dpi=180)
print(json.dumps(summaries, indent=2))
