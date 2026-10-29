"""Execute pinned SGLang DP selection and alignment functions without a device."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import sys
from enum import IntEnum, auto
from pathlib import Path
from types import ModuleType, SimpleNamespace

SOURCE_HASHES = {
    "layers/dp_attention.py": "f128669ebd206148f9ef4eb5cca9f4481702cf65177ed8a279a219b84c466915",
    "utils/common.py": "5423d1bb153e72852306934ab56ee0b3eccab8b3ec6395dc9da0fea79f7e74da",
    "model_executor/forward_batch_info.py": "167b747b28ee6030bc30e612c6da5c0a3dc0a5611a7324799427606655b6efcc",
    "layers/layer_boundary/exit.py": "c6bfc6897c494f5335a5e422b786d493c86db4ea19ef168dbcc4f7f3bc0dfd97",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    hashes = {name: hashlib.sha256((args.source_root / name).read_bytes()).hexdigest() for name in SOURCE_HASHES}
    if hashes != SOURCE_HASHES:
        raise ValueError("Source does not match the pinned SGLang Ascend 0.5.22 revision")
    source = ast.parse((args.source_root / "layers/dp_attention.py").read_text())
    padding = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "DpPaddingMode")
    helpers = ast.parse((args.source_root / "utils/common.py").read_text())
    align = [
        node for node in helpers.body if isinstance(node, ast.FunctionDef) and node.name in {"ceil_align", "ceil_div"}
    ]
    gather_width = [2]
    namespace = {
        "IntEnum": IntEnum,
        "auto": auto,
        "dp_gather_width": lambda: gather_width[0],
        "get_flags": lambda: SimpleNamespace(dp=SimpleNamespace(max_len_with_idle=False)),
        "_USE_ROCM700A_WA": False,
    }
    module = ModuleType("sglang.srt.layers.moe.utils")
    module.get_moe_a2a_backend = lambda: SimpleNamespace(is_pplx=lambda: False, is_deepep_v2=lambda: False)
    sys.modules[module.__name__] = module
    tree = ast.Module(body=[*ast.parse("from __future__ import annotations").body, padding, *align], type_ignores=[])
    exec(compile(tree, str(args.source_root), "exec"), namespace)
    policy, ceil_align = namespace["DpPaddingMode"], namespace["ceil_align"]
    generator = random.Random(20261030)
    cases = []
    for world in (16, 32):
        for tp in (1, 2, 4, 8, 16):
            dp = world // tp
            if dp == 1:
                continue
            gather_width[0] = dp
            for prefill in (False, True):
                vectors = [[0] * dp, [1] + [0] * (dp - 1), [9] + [1] * (dp - 1), [1] * dp]
                vectors.extend([[1] * (dp // 2) + [0] * (dp - dp // 2)])
                vectors.extend([[generator.choice((0, 0, 1, 3, 9, 17, 31, 32)) for _ in range(dp)] for _ in range(12)])
                for tokens in vectors:
                    if prefill:
                        tokens = [count * 1024 + int(count > 0) for count in tokens]
                    aligned = [ceil_align(count, tp) for count in tokens]
                    mode = policy.get_dp_padding_mode(prefill, aligned)
                    local = [max(aligned)] * dp if mode.is_max_len() else aligned
                    cases.append(
                        {
                            "tp": tp,
                            "prefill": prefill,
                            "tokens": tokens,
                            "mode": mode.name.lower(),
                            "local": local,
                            "global": sum(local),
                        }
                    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "revision": "baa9befa381511cd4d56f83980951e5a45fffc6d",
                "source_sha256": hashes,
                "contract": "eager, DeepEP v1 or tensor MoE, no hybrid SSM, no elastic world DP, no gatherv",
                "graph_default": policy.get_default_mode_in_cuda_graph().name.lower(),
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"output": str(args.output), "cases": len(cases)}))


if __name__ == "__main__":
    main()
