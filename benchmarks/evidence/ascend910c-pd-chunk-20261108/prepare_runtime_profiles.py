"""Verify inherited independent Eager calibration and stage its frozen source data."""
import gzip
import hashlib
import json
from pathlib import Path
import statistics


root = Path('/opt/pd-chunk-20261108')
checkout = Path('/workspace/benchmarks/evidence/ascend910c-glm52-w8a8/production/prediction-repair-20261102')
source = root/'profile-source'
source.mkdir(exist_ok=True)
for filename in (
    'active-rank-repair-manifest.json', 'eager-overhead-active-tp16-adp4.json',
    'eager-overhead-active-tp32-adp32.json', 'eager-overhead-tp16-adp4.json',
    'eager-overhead-tp32-adp32.json', 'paired-forward-analysis.json',
    'tp32-paired-forward-analysis.json.gz',
):
    payload = (checkout/filename).read_bytes()
    if filename.endswith('.json'):
        # Git's Windows checkout translates text to CRLF. Restore canonical LF
        # before comparing the immutable Linux-campaign hashes and identities.
        payload = payload.replace(b'\r\n', b'\n')
    (source/filename).write_bytes(payload)
manifest = json.loads((source/'active-rank-repair-manifest.json').read_text())
evidence = []
for topology, analysis_name, fit_case in (
    ('tp16-adp4', 'paired-forward-analysis.json', 'validation-c4'),
    ('tp32-adp32', 'tp32-paired-forward-analysis.json.gz', 'validation-c8'),
):
    profile_path = source/f'eager-overhead-active-{topology}.json'
    assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == manifest['artifacts'][profile_path.name]
    profile = json.loads(profile_path.read_text())
    base_path = source/f'eager-overhead-{topology}.json'
    base = json.loads(base_path.read_text())
    analysis_path = source/analysis_name
    raw = analysis_path.read_bytes()
    data = json.loads(gzip.decompress(raw) if analysis_path.suffix=='.gz' else raw)
    cases = {case['case']:case['decode_rows'] for case in data['cases']} if 'cases' in data else data['forwards']
    fitting = cases[fit_case]
    assert len(fitting) >= 100
    coefficients, fitted_rows = [], []
    for row in fitting:
        active = sum(rank['scheduled_requests']['num_decode_requests']>0 for rank in row['metrics'])
        assert active>1
        coefficients.append((row['residual_ms']-base['decode_ms'])/(active-1))
        fitted_rows.append(dict(metrics=row['metrics'],measured_ms=row['measured_ms'],
                                residual_ms=row['residual_ms'],active_ranks=active))
    slope = statistics.median(coefficients)
    assert abs(slope-profile['decode_per_extra_active_rank_ms']) < 1e-12
    identity = hashlib.sha256(json.dumps(dict(
        base_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),fit_rows=fitted_rows),sort_keys=True).encode()).hexdigest()
    assert profile['calibration_id'] == f'paired-active-{topology}-{identity[:16]}'
    evidence.append(dict(topology=topology,fit_case=fit_case,fit_forwards=len(fitting),
                         profile=profile,profile_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                         analysis_sha256=hashlib.sha256(raw).hexdigest()))

report = dict(status='legacy_sources_verified',independent_calibration=evidence,
              scope='Legacy profile identities and training-source arithmetic verified. These legacy values predate '
                    'operator contract v2 and require recalibration before use with the current model.')
(root/'legacy-source-verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
