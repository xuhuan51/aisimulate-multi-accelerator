"""Run pinned SGLang PD restoration and queue admission bodies with CPU stubs."""

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

SOURCES = {
    "disaggregation/decode.py": "7a106070dee807c4cc5498dc6d8092b229b23e573bf2dfa44f201ee695815718",
    "mem_cache/common.py": "e8d3ff0a8a0357e04e048bd18bd56f109ac59d6e44b205d366d9c617b8effc7d",
}
NAMES = {
    "resume_retracted_reqs",
    "_pre_alloc_fill_len",
    "_prealloc_kv_lens",
    "_prealloc_required_tokens",
    "_prealloc_reservation_fits",
    "_allocatable_token_budgets",
    "_active_req_count",
    "_active_reserved_tokens",
    "_need_space_for_single_req",
    "process_decode_queue",
    "backup_kv_cache",
    "restore_kv_cache",
}


def load_code(root):
    functions = []
    for relative, digest in SOURCES.items():
        content = (root / relative).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"Reaudit changed upstream source: {relative}")
        for node in ast.walk(ast.parse(content)):
            if isinstance(node, ast.FunctionDef) and node.name in NAMES:
                function = copy.deepcopy(node)
                function.decorator_list = []
                functions.append(function)
    assert {function.name for function in functions} == NAMES
    assert len(functions) == len(NAMES)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *functions,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    return compile(module, "pinned-sglang-pd-retraction", "exec")


def run_case(code, name, available, active, shapes, expected, fresh_allowed):
    events = []
    disagg = SimpleNamespace(
        disaggregation_decode_retraction_backup="cpu_tensor",
        disaggregation_decode_enable_radix_cache=False,
        disaggregation_decode_enable_offload_kvcache=False,
        disaggregation_decode_polling_interval=1,
        disaggregation_decode_host_receive_threshold=1,
    )
    namespace = {
        "get_disagg": lambda: disagg,
        "get_memory": lambda: SimpleNamespace(disable_radix_cache=True),
        "ceil_align": lambda value, alignment: (value + alignment - 1)
        // alignment
        * alignment,
        "ensure_mamba_capacity": lambda *args: True,
        "dsv4_needs_rebootstrap": lambda allocator: False,
        "CLIP_MAX_NEW_TOKEN": 4096,
    }
    exec(code, namespace)
    allocator = SimpleNamespace(
        page_size=4,
        free=available,
        prealloc_fits_assumes_reclaim=lambda: False,
        prealloc_fits=lambda tree,
        full,
        swa,
        full_budget_tokens,
        swa_budget_tokens: full <= full_budget_tokens,
    )
    allocator.available_size = lambda: allocator.free
    request_pool = SimpleNamespace(available_size=lambda: 100)
    transfer = SimpleNamespace(
        queue=[],
        resolve_deferred_releases=lambda: events.append("deferred"),
        extend=lambda requests: None,
        pop_transferred=lambda: [],
    )
    scheduler = SimpleNamespace(
        enable_hisparse=False,
        enable_decode_hicache=False,
        waiting_queue=[],
        running_batch=SimpleNamespace(reqs=[None] * active),
        last_batch=None,
        disagg_decode_transfer_queue=transfer,
    )
    queue = SimpleNamespace(
        scheduler=scheduler,
        req_to_token_pool=request_pool,
        token_to_kv_pool_allocator=allocator,
        transfer_queue=transfer,
        tree_cache=None,
        num_reserved_decode_tokens=4,
        retracted_queue=[],
        _uses_swa_reservation=lambda: False,
        _uses_swa_tail_prealloc=lambda: False,
    )
    for function in NAMES - {
        "process_decode_queue",
        "backup_kv_cache",
        "restore_kv_cache",
        "_pre_alloc_fill_len",
    }:
        setattr(queue, function, MethodType(namespace[function], queue))
    queue._pre_alloc_fill_len = namespace["_pre_alloc_fill_len"]

    def allocate(request):
        length = queue._pre_alloc_fill_len(request)
        allocator.free -= (length + 3) // 4 * 4
        events.append(["allocate", request.rid, length])

    def pop_preallocated():
        events.append("fresh")
        return [], []

    queue._pre_alloc = allocate
    queue.pop_preallocated = pop_preallocated
    scheduler.disagg_decode_prealloc_queue = queue
    for identifier, (prompt, output) in enumerate(shapes):
        request = SimpleNamespace(
            rid=identifier,
            origin_input_ids=[1] * prompt,
            output_ids=list(range(output)),
            is_retracted=True,
            load_kv_cache=lambda *args, identifier=identifier: events.append(
                ["restore", identifier]
            ),
            offload_kv_cache=lambda *args, identifier=identifier: events.append(
                ["backup", identifier]
            ),
        )
        assert namespace["backup_kv_cache"](
            request, None, request_pool, allocator, "cpu_tensor"
        )
        queue.retracted_queue.append(request)
    outputs_before = [request.output_ids[:] for request in queue.retracted_queue]
    namespace["process_decode_queue"](scheduler)
    resumed = [request.rid for request in scheduler.waiting_queue]
    assert resumed == expected, (name, resumed, expected)
    assert ("fresh" in events) == fresh_allowed, (name, events)
    assert [
        event[1]
        for event in events
        if isinstance(event, list) and event[0] == "restore"
    ] == expected
    for request in scheduler.waiting_queue + queue.retracted_queue:
        assert request.output_ids == outputs_before[request.rid]
        assert request.is_retracted == (request.rid not in expected)
    return {
        "name": name,
        "available_before": available,
        "active_requests": active,
        "prompt_output_lengths": shapes,
        "resumed": resumed,
        "waiting_for_restore": [request.rid for request in queue.retracted_queue],
        "available_after": allocator.free,
        "events": events,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    args = parser.parse_args()
    code = load_code(args.source_root)
    cases = [
        ("pending_sample_has_no_kv", 16, 0, [(8, 5)], [0], True),
        ("active_reserve_exact_fit", 20, 1, [(8, 5)], [0], True),
        ("active_reserve_blocks_restore", 19, 1, [(8, 5)], [], False),
        ("fifo_does_not_skip_large_head", 20, 0, [(20, 1), (8, 1)], [], False),
        ("each_resumed_request_reserves_growth", 32, 0, [(8, 5), (8, 5)], [0, 1], True),
        ("partial_restore_still_blocks_fresh", 28, 0, [(8, 5), (8, 5)], [0], False),
    ]
    print(
        json.dumps(
            {
                "source_revision": "baa9befa381511cd4d56f83980951e5a45fffc6d",
                "source_sha256": SOURCES,
                "scope": "CPU source behavior only; no KV tensor copies or timing measured",
                "cases": [run_case(code, *case) for case in cases],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
