import argparse
import collections
import csv
import json
from pathlib import Path


def union_duration(intervals):
    total, end = 0.0, float('-inf')
    for start, stop in sorted(intervals):
        if stop > end:
            total += stop - max(start, end)
            end = stop
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    events = json.loads((args.directory / 'trace_view.json').read_text())
    if isinstance(events, dict):
        events = events['traceEvents']
    spans = [event for event in events if event.get('ph') == 'X' and event.get('name') == 'scheduler.run_batch']
    kernels = list(csv.DictReader((args.directory / 'kernel_details.csv').open()))
    rows = []
    for index, span in enumerate(spans):
        start, duration = float(span['ts']), float(span['dur'])
        end = start + duration
        selected = [row for row in kernels if start <= float(row['Start Time(us)']) < end]
        intervals = [(float(row['Start Time(us)']), min(end, float(row['Start Time(us)']) + float(row['Duration(us)']))) for row in selected]
        types = collections.Counter()
        for row in selected:
            types[row['Type']] += float(row['Duration(us)']) / 1000
        host = collections.Counter()
        counts = collections.Counter()
        for event in events:
            if event.get('ph') != 'X' or event.get('cat') != 'cpu_op' or event.get('name') == 'scheduler.run_batch':
                continue
            if start <= float(event['ts']) < end and event.get('tid') == span['tid']:
                host[event['name']] += float(event.get('dur', 0)) / 1000
                counts[event['name']] += 1
        rows.append({
            'index': index, 'start_us': start, 'run_batch_ms': duration / 1000,
            'kernel_union_ms': union_duration(intervals) / 1000,
            'outside_kernel_union_ms': (duration - union_duration(intervals)) / 1000,
            'kernel_duration_sum_ms': sum(types.values()), 'kernel_count': len(selected),
            'kernel_type_duration_sums_ms': dict(types.most_common()),
            'host_inclusive_duration_ms': dict(host.most_common(40)),
            'host_invocations': dict(counts.most_common(40)),
        })
    result = {
        'scope': 'Profiler-observed execution; profiler perturbation prevents using these durations as unprofiled latency truth',
        'outside_kernel_union': 'Includes host submission, device idle, synchronization, and uncaptured runtime work; not an isolated CPU overhead measurement',
        'host_inclusive': 'Nested CPU spans; values must not be summed as exclusive wall time',
        'source': str(args.directory), 'rows': rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps([{key: row[key] for key in ['index', 'run_batch_ms', 'kernel_union_ms', 'outside_kernel_union_ms', 'kernel_count']} for row in rows]))


if __name__ == '__main__':
    main()
