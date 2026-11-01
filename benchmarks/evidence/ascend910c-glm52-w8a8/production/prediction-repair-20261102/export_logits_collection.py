import json
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    observation = root / 'linear-observations-node20.json'
    raw = json.loads(observation.read_text())
    runtime = raw['runtime']
    identity = {key: runtime[key] for key in ('framework_commit', 'torch', 'torch_npu', 'driver_version_info')}
    identity.update({
        'image_digest': 'quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443',
        'hardware': runtime['device'], 'collector_source_sha256': raw['collector_source_sha256'],
    })
    identity_path = root / 'linear-runtime-identity.json'
    identity_path.write_text(json.dumps(identity, indent=2) + '\n')
    subprocess.run([
        sys.executable, '-m', 'collector.sglang.ascend_linear_export', str(observation),
        '--plan', str(root / 'linear-plan.json'), '--runtime-identity', str(identity_path),
        '--output-dir', str(root / 'linear-published'),
        '--collector-ref', '2a5c7c6b:python/aisimulate/collector/sglang/ascend_linear.py',
    ], check=True)


if __name__ == '__main__':
    main()
