"""Freeze the verified CPU campaign and its historical inputs for reproduction."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


repository = Path('/workspace')
destination = repository/'benchmarks/evidence/ascend910c-pd-chunk-20261108'
root = Path('/opt/pd-chunk-20261108')
results = destination/'results'
results.mkdir(exist_ok=True)
calibration = json.loads((root/'runtime-v2-calibration.json').read_text())
profiles = destination/'profiles'
profiles.mkdir(exist_ok=True)
for topology in calibration['topologies']:
    (profiles/f"eager-v2-{topology['topology']}.json").write_text(
        json.dumps(topology['profile'], indent=2)+'\n')
with gzip.open(results/'runtime-v2-calibration.json.gz','wb') as stream:
    stream.write((root/'runtime-v2-calibration.json').read_bytes())
for name in ('legacy-source-verification.json','pd-validation.json','joint-search-verification.json',
             'pd-validation.png','pd-validation.svg','pd-before-after.png','pd-before-after.svg',
             'pd-chunk-search.png','pd-chunk-search.svg'):
    shutil.copy2(root/name,results/name)

main = json.loads((root/'main-cli-wide-replay/prediction.json').read_text())
frozen = json.loads((root/'top1-wide-replay/prediction.json').read_text())
for field in ('completed_requests','failed_requests','output_throughput_tok_s','mean_ttft_ms','mean_tpot_ms'):
    assert main[field] == frozen[field], field
shutil.copy2(root/'main-cli-wide-replay/prediction.json',results/'main-cli-wide-prediction.json')

logs = destination/'logs'
logs.mkdir(exist_ok=True)
for name in ('pd-chunk-unit-r1.log','pd-chunk-search-regression-r3.log',
             'pd-ascend-rust-tests-r1.log','pd-hccl-rust-tests.log',
             'pd-chunk-frozen-parity.log','pd-main-cli-wide.log'):
    shutil.copy2(Path('/runtime')/name,logs/name)

files = []
for name in ('configs','profile-source','baseline-all','runtime-v2-all','runtime-six',
             'joint-search-qualified','joint-search-wide-decode','joint-search-r1',
             'top1-local-replay','top1-wide-replay','main-cli-wide-replay'):
    files.extend((path,str(path.relative_to(root))) for path in sorted((root/name).rglob('*')) if path.is_file())
for path in sorted(root.iterdir()):
    if path.is_file():
        files.append((path,path.name))
source = repository/'benchmarks/evidence/ascend910c-glm52-w8a8/production/prediction-repair-20261102/historical-modes-rerun-evidence-archive-20261104'
for name in ('evidence.tar.gz','manifest.json'):
    files.append((source/name,'sources/historical-'+name))
for path in sorted(destination.glob('*.py')):
    files.append((path,'scripts/'+path.name))
for path in sorted(logs.iterdir()):
    files.append((path,'logs/'+path.name))

production = (
    'python/aisimulate/src/aisimulate/config/engine.py',
    'python/aisimulate/src/aisimulate/sweeper/config.py',
    'python/aisimulate/src/aisimulate/sweeper/search_space.py',
    'python/aisimulate/src/aisimulate/recommend.py',
    'python/aisimulate/tests/unit/test_modern_recommend_context.py',
    'tests/sweeper/test_engine_request.py',
    'crates/core/src/perfmodel/perf_database/hccl.rs',
    'crates/core/src/perfmodel/operators/ascend_eager_overhead.rs',
    'crates/core/src/perfmodel/fpm/config.rs',
    'crates/core/src/perfmodel/fpm/estimator.rs',
    'crates/core/src/engine/scheduler/rank.rs',
)
for name in production:
    files.append((repository/name,'source/'+name))

members = [dict(member=name,bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path,name in files]
temporary = Path('/opt/pd-chunk-evidence-20261108.tar.xz')
with tarfile.open(temporary,'w:xz',preset=6) as stream:
    for path,name in files:
        stream.add(path,arcname=name,recursive=False)
archive_sha = hashlib.sha256(temporary.read_bytes()).hexdigest()
shutil.copy2(temporary,destination/'evidence.tar.xz')
manifest = dict(status='frozen',archive='evidence.tar.xz',archive_sha256=archive_sha,
                archive_bytes=temporary.stat().st_size,native_sha256=calibration['native_sha256'],
                source_head='f134fa6f',members=members,
                historical_archive_sha256='3cf9767854f5436168aea80df599128ab834c80fcd405388c7caa51791f2961d',
                verification='Historical archive377 members verified before use. Current bundle members hashed before archival.',
                runtime='CPU only, Docker4CPU/5GiB, schema36, no NPU device requests',
                scope='Archived GLM-5.2 Eager serving data. Current PD values did not enter calibration. Other chunk choices have no new serving validation.')
(destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({key:value for key,value in manifest.items() if key!='members'},indent=2))
print('members',len(members))
