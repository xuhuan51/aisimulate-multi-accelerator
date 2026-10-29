"""Execute pinned SGLang admission methods against a deterministic token pool."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from enum import Enum, auto
from itertools import product
from pathlib import Path
from types import SimpleNamespace

SOURCE_HASHES = {
    "managers/schedule_policy.py": "03ba021fdd225cc4e7d0d5eb822a725fe6e3907e40157ec6461390f23b7f2cb9",
    "mem_cache/prefill_budget.py": "837ac7c099166d061b9a76e24825d0555f3845e1a3bb53e785c5ae8e6f0092a2",
}


class Request(SimpleNamespace):
    @property
    def extend_len(self):
        return self.extend_end - self.prefix_len


def request(input_tokens, output_tokens, materialized=0):
    return Request(
        origin_input_ids=[1] * input_tokens,
        full_untruncated_fill_ids=[1] * input_tokens,
        output_ids=[],
        prefix_len=materialized,
        extend_end=input_tokens,
        sampling_params=SimpleNamespace(max_new_tokens=output_tokens, ignore_eos=True),
        retracted_stain=False,
    )


def load_policy(source_root):
    hashes = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest() for name in SOURCE_HASHES}
    if hashes != SOURCE_HASHES:
        raise ValueError("Source hashes do not match the pinned 0.5.22 Ascend runtime")
    methods = {
        "_get_running_request_total_token_offset",
        "rem_total_tokens",
        "cur_rem_tokens",
        "_swa_new_tokens",
        "ceil_paged_tokens",
        "budget_state",
        "_update_prefill_budget",
        "add_one_req_ignore_eos",
        "add_chunked_req",
    }
    policy_path = source_root / "managers/schedule_policy.py"
    budget_path = source_root / "mem_cache/prefill_budget.py"
    policy_ast = ast.parse(policy_path.read_text())
    budget_ast = ast.parse(budget_path.read_text())
    adder = next(node for node in policy_ast.body if isinstance(node, ast.ClassDef) and node.name == "PrefillAdder")
    adder.body = [node for node in adder.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    assert {node.name for node in adder.body} == methods
    status = next(node for node in policy_ast.body if isinstance(node, ast.ClassDef) and node.name == "AddReqResult")
    budget = next(node for node in budget_ast.body if isinstance(node, ast.ClassDef) and node.name == "PrefillBudget")
    namespace = {"Enum": Enum, "auto": auto, "CLIP_MAX_NEW_TOKENS": 4096, "IGNORE_EOS_RESERVE_TOKENS": 1}
    module = ast.Module(
        body=[*ast.parse("from __future__ import annotations").body, status, budget, adder], type_ignores=[]
    )
    exec(compile(module, str(policy_path), "exec"), namespace)
    return namespace, hashes


def evaluate(namespace, case):
    policy = namespace["PrefillAdder"].__new__(namespace["PrefillAdder"])
    block = case["block_size"]
    running = [request(*entry) for entry in case["running"]]
    continuation = request(*case["continuation"]) if case.get("continuation") else None
    occupied = sum((len(entry.origin_input_ids) + block - 1) // block * block for entry in running)
    occupied += continuation.prefix_len if continuation else 0
    allocator = SimpleNamespace(page_size=block, available_size=lambda: case["capacity"] - occupied)
    cache = SimpleNamespace(supports_mamba=lambda: False, evictable_size=lambda: 0)
    policy.memory_budget = namespace["PrefillBudget"](allocator, cache)
    policy.new_token_ratio = 0.1
    policy.memory_budget.total_offset = sum(policy._get_running_request_total_token_offset(entry) for entry in running)
    policy.page_size = policy.per_req_token_overhead = block
    policy.running_batch = SimpleNamespace(reqs=running)
    policy.can_run_list = []
    policy.req_states = None
    policy.rem_chunk_tokens = case["chunk"]
    policy.rem_input_tokens = case["input_budget"]
    policy.is_hybrid_swa = policy.exact_chunk_fill = False
    policy.prefill_delayer_single_pass = policy.dllm_config = policy.rem_mamba_slots = None
    policy.chunked_req_limit = None
    policy.kv_shard_granule = 0
    policy.log_hit_tokens = policy.log_input_tokens = 0
    policy._mamba_gap_budget_for_req = lambda *args, **kwargs: 0
    policy._request_slot_reserve_for_req = lambda *args, **kwargs: 0
    policy._kv_shard_reserve_scratch = lambda **kwargs: True
    policy._check_prefill_tile_budget = lambda tokens: None
    if continuation:
        policy.add_chunked_req(continuation)
    for entry in case["waiting"]:
        if policy.budget_state() != namespace["AddReqResult"].CONTINUE:
            break
        result = policy.add_one_req_ignore_eos(request(*entry))
        if result != namespace["AddReqResult"].CONTINUE:
            break
    return [entry.extend_end for entry in policy.can_run_list]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    namespace, hashes = load_policy(args.source_root)
    cases = []
    for capacity, chunk, inputs, outputs, running_input in product(
        [32, 40, 64, 100], [8, 32], [4, 16, 40], [1, 16], [0, 16]
    ):
        cases.append(
            {
                "capacity": capacity,
                "block_size": 4,
                "chunk": chunk,
                "input_budget": 64,
                "running": [[running_input, 4]] if running_input else [],
                "waiting": [[inputs, outputs]],
            }
        )
    for output_tokens in [4, 25]:
        cases.append(
            {
                "capacity": 100,
                "block_size": 1,
                "chunk": 100,
                "input_budget": 100,
                "running": [[60, output_tokens]],
                "waiting": [[10, 30]],
            }
        )
    cases.append(
        {"capacity": 64, "block_size": 4, "chunk": 32, "input_budget": 8, "running": [], "waiting": [[16, 4], [4, 1]]}
    )
    for materialized in [8, 16, 24, 32]:
        cases.append(
            {
                "capacity": 64,
                "block_size": 4,
                "chunk": 8,
                "input_budget": 64,
                "running": [],
                "waiting": [],
                "continuation": [40, 16, materialized],
            }
        )
    for case in cases:
        case["admitted_ends"] = evaluate(namespace, case)
    result = {
        "revision": "baa9befa381511cd4d56f83980951e5a45fffc6d",
        "source_sha256": hashes,
        "scope": "Actual pinned admission methods; homogeneous ignore_eos; single pool; no cache/SWA/Mamba/DCP",
        "constants": {"clip_max_new_tokens": 4096, "ignore_eos_reserve_tokens": 1},
        "cases": cases,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "source_sha256": hashes}))


if __name__ == "__main__":
    main()
