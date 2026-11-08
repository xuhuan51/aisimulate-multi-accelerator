"""Verify inherited PD evidence and prepare matched CPU reruns."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile

import yaml


repository = Path('/workspace')
prefix = repository/'benchmarks/evidence/ascend910c-glm52-w8a8/production/prediction-repair-20261102'
archive_directory = prefix/'historical-modes-rerun-evidence-archive-20261104'
archive = archive_directory/'evidence.tar.gz'
manifest = json.loads((archive_directory/'manifest.json').read_text())
assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest['archive_sha256']
root = Path('/opt/pd-chunk-20261108')
inherited = root/'inherited'
inherited.mkdir(parents=True, exist_ok=True)
with tarfile.open(archive) as stream:
    members = {member.name:member for member in stream.getmembers()}
    for entry in manifest['members']:
        member = members[entry['member']]
        assert member.isfile() and member.size == entry['bytes']
        assert (inherited/member.name).resolve().is_relative_to(inherited.resolve())
        content = stream.extractfile(member).read()
        assert hashlib.sha256(content).hexdigest() == entry['sha256'], member.name
        destination = inherited/member.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)

historical = inherited/prefix.relative_to(repository)/'historical-kernel-layer-rerun-20261104-1430'
rows = list(csv.DictReader((historical/'comparison.csv').open(encoding='utf-8-sig')))
assert len(rows) == 78
jobs = {}
for row in rows:
    original_config = inherited/row['config']
    job = original_config.parent.name
    if job in jobs:
        jobs[job]['comparison_numbers'].append(int(row['number']))
        continue
    config = yaml.safe_load(original_config.read_text())
    config['engine']['model'] = '/runtime/glm52-model'
    config['engine']['systems_paths'] = ['/opt/combined-main-runtime/aisimulate_core/systems']
    destination = root/'configs'/f'{job}.yaml'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False))
    jobs[job] = dict(job=job, mode=row['mode'], group=row['group'], nodes=int(row['nodes']),
                     concurrency=int(row['concurrency']), requests=int(row['requests']),
                     comparison_numbers=[int(row['number'])], config=str(destination),
                     source_config=str(original_config),
                     original_config_sha256=hashlib.sha256(original_config.read_bytes()).hexdigest())
report = dict(status='prepared', source_archive_sha256=manifest['archive_sha256'],
              verified_members=len(manifest['members']), measured_rows=39, comparison_rows=len(rows),
              unique_jobs=len(jobs), jobs=list(jobs.values()),
              rewrite_scope='Only model metadata and equivalent systems paths relocated. All timing modes, precision, workload and scheduling choices preserved.')
(root/'prepared.json').write_text(json.dumps(report, indent=2))
print(json.dumps({key:value for key,value in report.items() if key!='jobs'}, indent=2))
