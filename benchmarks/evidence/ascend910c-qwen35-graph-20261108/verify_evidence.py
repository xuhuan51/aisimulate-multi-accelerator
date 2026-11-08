"""Verify frozen Qwen observations and reproduce10-repeat communication costs."""
import hashlib
import json
from pathlib import Path
import tempfile
import tarfile


directory = Path(__file__).resolve().parent
archive = directory/'qwen-graph-evidence-complete-20261108.tar.gz'
assert archive.stat().st_size == 8033813
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '70540229f9c0dc3a1532759392fdf385f85ad86161ae5b51deac6dca5d48a16d'


def extract(source, target):
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source) as stream:
        for member in stream.getmembers():
            assert member.isfile() or member.isdir(), member.name
            assert (target/member.name).resolve().is_relative_to(target.resolve()), member.name
        stream.extractall(target)


with tempfile.TemporaryDirectory(prefix='qwen-graph-evidence-') as scratch:
    root = Path(scratch)
    extract(archive, root)
    original = root/'qwen-graph-qualified-raw-20261108.tar.gz'
    assert hashlib.sha256(original.read_bytes()).hexdigest() == 'f30f4981b52986419b928726faa1b7501c0a47f65e19696d2e554bc5f0794230'
    raw = root/'raw'
    extract(original, raw)
    manifest = json.loads((raw/'graph-raw-manifest.json').read_text())
    for entry in manifest['files']:
        assert hashlib.sha256((raw/entry['path']).read_bytes()).hexdigest() == entry['sha256'], entry['path']
    communication_root = root/'qwen-graph-fp32-r1'
    frozen = json.loads((communication_root/'ep-graph-qualified-rows-10.json').read_text())
    for entry in frozen['sources']:
        path = communication_root/entry['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], entry['path']
    collector = root/'qualify_qwen_ep_graph_10.py'
    assert hashlib.sha256(collector.read_bytes()).hexdigest() == frozen['collector_source_sha256']
    # Only relocate files for this audit. Execute the frozen numerical logic and
    # retain the collector's original source hash in its recomputed report.
    code = collector.read_text().replace(
        '/home/wuliang/aisimulate-repair-20261108/qwen-graph-fp32-r1', str(communication_root))
    code = code.replace("root/'ep-graph-qualified-rows-10.json'", "root/'recomputed.json'")
    exec(compile(code, str(collector), 'exec'), {'__file__':str(collector)})
    rebuilt = json.loads((communication_root/'recomputed.json').read_text())
    assert rebuilt['status'] == frozen['status'] == 'qualified'
    assert len(rebuilt['points']) == len(frozen['points']) == 2
    for actual, expected in zip(rebuilt['points'], frozen['points']):
        assert actual.keys() == expected.keys()
        for key in actual:
            if key == 'cv':
                assert abs(actual[key]-expected[key]) <= max(1e-18, abs(expected[key])*1e-12)
            else:
                assert actual[key] == expected[key], key
    print(json.dumps(dict(status='passed',manifest_files=len(manifest['files']),
                          communication_csv_files=len(frozen['sources']),points=2)))
