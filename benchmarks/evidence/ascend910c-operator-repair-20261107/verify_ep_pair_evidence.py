"""Recompute the 14 published EP rows from frozen raw CSVs and checks."""
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import runpy
import tarfile
import tempfile

import pyarrow.parquet as parquet


root = Path(__file__).resolve().parent
repository = root.parents[2]
archive = root/'ep-pair-evidence-20261108.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '8788747c72547dd9a9fbeff9301d60ab0dc926f32774f9f4490d047c825bd996'
table_path = repository/'python/aisimulate/src/aisimulate_core/systems/data/ascend910c/moe/sglang/0.5.22/ascend_deepep_perf.parquet'


def same_row(actual, expected):
    # Python 3.9 and3.11 statistics produce a one-ULP CV difference at M1.
    # Coordinates, identities, source hashes and latencies remain exact.
    return actual.keys() == expected.keys() and all(
        math.isclose(actual[key], value, rel_tol=1e-12, abs_tol=1e-18)
        if key == 'cv' else actual[key] == value for key, value in expected.items())


with tempfile.TemporaryDirectory(prefix='aisimulate-ep-pair-') as directory:
    destination = Path(directory).resolve()
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            assert member.isfile() or member.isdir(), member.name
            assert (destination/member.name).resolve().is_relative_to(destination), member.name
        source.extractall(destination)
    script, = destination.rglob('qualify_ep_pair_steady.py')
    measured = script.parent
    expected = json.loads((measured/'ep-pair-steady-qualified-rows.json').read_text())
    namespace = runpy.run_path(str(script))
    main = namespace['main']
    main.__globals__['ROOT'] = measured
    with contextlib.redirect_stdout(io.StringIO()):
        main()
    regenerated = json.loads((measured/'ep-pair-steady-qualified-rows.json').read_text())
    report = json.loads((measured/'ep-pair-steady-qualification.json').read_text())
    assert report['status'] == 'qualified' and not report['rejected']
    assert len(regenerated) == len(expected) == 14
    assert all(same_row(actual, wanted) for actual, wanted in zip(regenerated, expected))
    exported = parquet.read_table(table_path).to_pylist()
    identity = report['identity_sha256']
    published = [row for row in exported if row['runtime_identity_sha256'] == identity]
    assert len(published) == 14
    for row in regenerated:
        matches = [item for item in published if item['operation'] == row['operation']
                   and item['tokens_per_rank'] == row['tokens_per_rank']]
        match, = matches
        assert same_row(match, row), row
print(json.dumps(dict(status='passed', recomputed_rows=14, table_rows=len(exported),
                      source_csv_files=len(report['sources']), runtime_identity_sha256=identity,
                      scope='Frozen numerical records and raw per-stage intervals. '
                            'First-pair exclusion was chosen after observing the all75 anomaly. '
                            'This verifier does not establish new serving accuracy.')))
