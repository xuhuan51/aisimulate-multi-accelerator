import hashlib
import json
from pathlib import Path

from collector.sglang.ascend_linear_cases import get_ascend_linear_cases


destination = Path(__file__).resolve().parent
plan = get_ascend_linear_cases('/data_lib/data/models/glm-q')
selected = [case for case in plan['cases'] if case['quant_mode'] == 'bfloat16' and case['n'] == 4840 and case['k'] == 6144]
if not selected:
    raise ValueError('Global TP32 LM head is absent from the declared model cases')
plan['cases'] = selected
plan['scope'] = 'Only the newly declared global TP32 BF16 LM-head curve; no other operator coverage claim'
plan['recipe_invocations'] = len(selected)
plan['unique_invocations'] = len(selected)
plan['generator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(destination / 'linear-plan.json').write_text(json.dumps(plan, indent=2) + '\n')
print(json.dumps({'cases': len(selected), 'output': str(destination / 'linear-plan.json')}))
