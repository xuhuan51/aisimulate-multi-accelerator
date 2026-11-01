import argparse
import json
import os
import socket
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
IMAGE = 'quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443'
CONTAINER = 'aisim-forward-diagnostic-20261102'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--forward-metrics', action='store_true')
    parser.add_argument('--paired-forward-metrics', action='store_true')
    args = parser.parse_args()
    if args.paired_forward_metrics:
        args.forward_metrics = True
    container = 'aisim-forward-paired-20261102' if args.paired_forward_metrics else ('aisim-forward-metrics-20261102' if args.forward_metrics else CONTAINER)
    inventory = subprocess.check_output(['npu-smi', 'info'], text=True)
    if inventory.count('No running processes found in NPU') != 8:
        raise RuntimeError('NPU processes already exist; preserve their workloads')
    destination = ROOT / ('forward-paired-node20' if args.paired_forward_metrics else ('forward-metrics-node20' if args.forward_metrics else 'forward-diagnostic-node20'))
    if destination.exists():
        raise FileExistsError(destination)
    with socket.socket() as probe:
        probe.bind(('0.0.0.0', 18080))
    destination.mkdir()
    if args.paired_forward_metrics:
        shutil.copyfile(ROOT / 'launch_paired_telemetry.py', destination / 'launch_paired_telemetry.py')
    (destination / 'npu-before.txt').write_text(inventory)
    setup = json.loads((ROOT.parent / 'serving-setup.json').read_text())
    environment = {key: value for key, value in setup['environment'].items() if value is not None}
    environment.update({
        'SGLANG_ENABLE_OVERLAP_PLAN_STREAM': '0',
        'SGLANG_NPU_PROFILING': '1',
        'HCCL_IF_BASE_PORT': '39000',
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
    })
    arguments = [
        '--model-path', '/models/glm-q', '--host', '0.0.0.0', '--port', '18080',
        '--nnodes', '1', '--node-rank', '0', '--tp-size', '16', '--attn-dp-size', '4',
        '--ep-size', '16', '--mem-fraction-static', '0.90', '--max-running-requests', '128',
        '--attention-backend', 'ascend', '--device', 'npu', '--quantization', 'modelslim',
        '--served-model-name', 'glm-5', '--watchdog-timeout', '9000', '--context-length', '180000',
        '--tokenizer-worker-num', '4', '--disable-shared-experts-fusion', '--dtype', 'bfloat16',
        '--load-balance-method', 'round_robin', '--page-size', '128',
        '--enable-cache-report', '--enable-metrics', '--enable-metrics-for-all-schedulers',
        '--trust-remote-code', '--moe-a2a-backend', 'deepep', '--deepep-mode', 'auto',
        '--cuda-graph-backend-prefill', 'disabled', '--cuda-graph-backend-decode', 'disabled',
        '--disable-overlap-schedule', '--disable-radix-cache',
        '--chunked-prefill-size', '16384', '--max-prefill-tokens', '32768',
    ]
    if args.forward_metrics:
        environment['SGLANG_NPU_PROFILING'] = '0'
        arguments.extend([
            '--enable-forward-pass-metrics', '--forward-pass-metrics-worker-id', container,
            '--forward-pass-metrics-ipc-name', 'ipc:///campaign/forward-metrics',
        ])
    command = [
        'docker', 'run', '-d', '--name', container, '--network', 'host', '--privileged',
        '--shm-size', '16g', '--mount', 'type=bind,src=/data_lib/data/models/glm-q,dst=/models/glm-q,readonly',
        '--mount', f'type=bind,src={destination},dst=/campaign',
    ]
    for path in ('/usr/local/Ascend/driver', '/usr/local/Ascend/firmware', '/usr/local/sbin', '/etc/ascend_install.info'):
        if Path(path).exists():
            command.extend(['--mount', f'type=bind,src={path},dst={path},readonly'])
    for key, value in environment.items():
        command.extend(['-e', f'{key}={value}'])
    entrypoint = ['/campaign/launch_paired_telemetry.py'] if args.paired_forward_metrics else ['-m', 'sglang.launch_server']
    command.extend(['--entrypoint', 'python3', IMAGE, *entrypoint, *arguments])
    result = subprocess.run(command, capture_output=True, text=True)
    record = {
        'scope': 'Single-node TP16/ADP4/EP16 colocated forward diagnostic; not a replacement for measured four-node PD results',
        'decode_comparability': 'Same decoder geometry, global TP LM head, eager execution and low-latency DeepEP branch; no cross-host KV transfer',
        'environment': environment, 'command': command,
        'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
        'host': socket.gethostname(),
    }
    (destination / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')
    os.chown(destination, 5034, 5034)
    print(json.dumps({'container': container, 'returncode': result.returncode, 'evidence': str(destination)}))
    result.check_returncode()


if __name__ == '__main__':
    main()
