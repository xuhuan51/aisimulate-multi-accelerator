import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent / 'forward-metrics-node20'
CONTAINER = 'aisim-forward-metrics-20261102'
CASES = [
    ('telemetry-control', 8192, 64, 4),
    ('calibration-c1', 8192, 128, 1),
    ('validation-c4', 8192, 128, 4),
    ('validation-c8', 8192, 128, 8),
    ('calibration-prefill4k', 4096, 64, 4),
    ('validation-prefill16k', 16384, 64, 4),
    ('validation-c12', 8192, 64, 12),
]


def save(state):
    (ROOT / 'progress.json').write_text(json.dumps(state, indent=2) + '\n')


def main():
    state = {'status': 'waiting_for_health', 'scope': 'Dedicated single-node forward telemetry; original four-node PD measurements remain unchanged', 'cases': []}
    save(state)
    deadline = time.monotonic() + 900
    while True:
        try:
            urllib.request.urlopen('http://127.0.0.1:18080/health', timeout=5).close()
            break
        except (urllib.error.URLError, TimeoutError):
            if time.monotonic() > deadline:
                raise TimeoutError('Diagnostic service did not become ready')
            time.sleep(5)
    for name, input_length, output_length, concurrency in CASES:
        if (ROOT / f'{name}.complete').exists():
            raise FileExistsError(f'Campaign already has {name}; inspect instead of overwriting')
        command = [
            'docker', 'exec', '-e', 'PYTHONUNBUFFERED=1', CONTAINER,
            'python3', '-m', 'sglang.benchmark.serving', '--backend', 'sglang',
            '--host', '127.0.0.1', '--port', '18080', '--model', '/models/glm-q',
            '--served-model-name', 'glm-5', '--dataset-name', 'random-ids',
            '--num-prompts', str(concurrency * 2), '--max-concurrency', str(concurrency),
            '--random-input-len', str(input_length), '--random-output-len', str(output_length),
            '--random-range-ratio', '1.0', '--request-rate', 'inf', '--warmup-requests', '2',
            '--seed', '910', '--tokenize-prompt', '--output-details', '--output-file', f'/campaign/{name}.jsonl',
        ]
        point = {'name': name, 'input_length': input_length, 'output_length': output_length, 'concurrency': concurrency,
                 'command': command, 'start_unix': time.time(), 'status': 'running'}
        state['status'] = 'running'
        state['cases'].append(point)
        save(state)
        with (ROOT / f'{name}.log').open('w') as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=1800)
        point.update({'end_unix': time.time(), 'returncode': result.returncode})
        if result.returncode:
            point['status'] = state['status'] = 'failed'
            save(state)
            raise RuntimeError(f'Failed benchmark {name}')
        measured = json.loads((ROOT / f'{name}.jsonl').read_text().splitlines()[-1])
        if measured['completed'] != concurrency * 2 or measured['input_lens'] != [input_length] * (concurrency * 2) or measured['output_lens'] != [output_length] * (concurrency * 2) or any(measured['errors']):
            raise ValueError(f'Incomplete workload {name}')
        point['status'] = 'completed'
        point['metrics'] = {key: measured[key] for key in ['mean_ttft_ms', 'mean_tpot_ms', 'output_throughput']}
        (ROOT / f'{name}.complete').write_text(json.dumps(point, indent=2) + '\n')
        save(state)
        print(json.dumps(point['metrics'] | {'name': name}), flush=True)
    state['status'] = 'completed'
    save(state)


if __name__ == '__main__':
    main()
