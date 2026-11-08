"""Exercise curve-local identity isolation through the canonical public FPM."""
import copy
import json
from pathlib import Path
import shutil
import tempfile

import pyarrow as arrow
import pyarrow.parquet as parquet
from aisimulate_core.sdk import RustForwardPassPerfModel


case = json.loads(Path('/runtime/qwen-graph-case.json').read_text())
with tempfile.TemporaryDirectory(prefix='qwen-hccl-identity-') as scratch:
    root = Path(scratch)/'systems'
    shutil.copytree('/opt/qwen-graph-systems', root)
    path, = root.rglob('hccl_perf.parquet')
    table = parquet.read_table(path)
    original = table.to_pylist()
    for mode in ('separate_complete_curve', 'mixed_within_curve'):
        rows = copy.deepcopy(original)
        candidates = [row for row in rows if row['execution_mode']=='graph' and
                      row['operation']==('reduce_scatter' if mode=='separate_complete_curve' else 'all_gather')]
        assert len(candidates) == (1 if mode=='separate_complete_curve' else 2)
        candidates[0]['runtime_identity_sha256'] = 'a'*64
        parquet.write_table(arrow.Table.from_pylist(rows, schema=table.schema), path)
        config = copy.deepcopy(case['predictor_config'])
        config['systems_paths'] = [str(root)]
        model = None
        try:
            model = RustForwardPassPerfModel.best_available(config)
            result = model.static_phase_diagnostics(batch_size=3, context_length=8192, prefill=False)
            assert mode == 'separate_complete_curve', 'mixed identity unexpectedly accepted'
            assert abs(sum(row['latency_ms'] for row in result)-33.25211611464436) < 1e-9
        except Exception as error:
            assert mode == 'mixed_within_curve' and 'mixes runtime/hardware identities' in str(error), str(error)
        finally:
            if model is not None:
                model.close()
        print(mode, 'passed')
