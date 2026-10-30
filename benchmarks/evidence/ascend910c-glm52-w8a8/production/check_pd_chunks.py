"""Run pinned SGLang chunk dispatch bodies with CPU-only transport and KV stubs."""

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

SOURCE = "disaggregation/prefill.py"
SOURCE_SHA256 = "386c5d007000eecead04f3527d2c780f26b14ca5e1e438346afa40eceba666d9"


class Request(SimpleNamespace):
    __hash__ = object.__hash__


class TokenRows:
    def __getitem__(self, index):
        token_slice = index[1]
        return list(range(token_slice.start or 0, token_slice.stop))


def load_functions(source_root):
    content = (source_root / SOURCE).read_bytes()
    if hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
        raise ValueError("Reaudit changed SGLang prefill source before running this oracle")
    names = {"process_prefill_chunk", "_send_kv_chunk"}
    functions = []
    for node in ast.walk(ast.parse(content)):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            function = copy.deepcopy(node)
            function.decorator_list = []
            functions.append(function)
    assert {function.name for function in functions} == names
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    return compile(module, "pinned-sglang-chunks", "exec")


def run_case(code, prompt, boundaries, prefix, expected, overlap=False, bootstrap=True):
    events = []
    sends = []
    page_size = 128
    state_names = [
        "MAMBA",
        "QSA_PENDING",
        "QSA_COMPRESSED",
        "SWA",
        "DSA",
        "DSA_TAIL",
        "MINIMAX_INDEX_K",
        "MINIMAX_DENSE_KV",
        "SWA_RING",
        "DSV4_REQUEST_STATE",
        "BLOCK_SCALE",
        "BLOCK_SCALE_SWA",
    ]
    namespace = {
        "StateType": SimpleNamespace(**{name: name for name in state_names}),
        "_is_npu": True,
        "DeepSeekV4TokenToKVPool": type("UnrelatedPool", (), {}),
        "kv_to_page_indices": lambda indices, size: sorted({index // size for index in indices}),
        "logger": SimpleNamespace(debug=lambda *args: None),
        "get_disagg": lambda: SimpleNamespace(disaggregation_decode_allocation_policy="prefill_complete"),
        "is_aborted": lambda request: False,
    }
    exec(code, namespace)

    def send(pages, states, num_kv_tokens):
        entry = {
            "ready_tokens": request.extend_end,
            "pages": pages,
            "tokens": num_kv_tokens,
            "last": states is not None,
        }
        sends.append(entry)
        events.append(["send", request.extend_end, num_kv_tokens])

    sender = SimpleNamespace(
        get_max_transfer_tokens=lambda: None,
        should_send_kv_chunk=lambda pages, last: pages > 0,
        send=send,
    )
    request = Request(
        rid="request",
        start_send_idx=prefix,
        disagg_decode_prefix_len=prefix,
        origin_input_ids=list(range(prompt)),
        extend_end=0,
        disagg_kv_sender=sender,
        kv=SimpleNamespace(req_pool_idx=0),
    )
    scheduler = SimpleNamespace(
        chunked_req=request,
        enable_overlap=overlap,
        enable_staging=False,
        token_to_kv_pool_allocator=SimpleNamespace(
            page_size=page_size,
            translate_kv_indices_for_transfer=lambda indices: indices,
            get_kvcache=object,
        ),
        req_to_token_pool=SimpleNamespace(req_to_token=TokenRows()),
        disagg_metadata_buffers=SimpleNamespace(
            set_buf=lambda request: events.append(["metadata", request.extend_end])
        ),
        disagg_prefill_bootstrap_queue=SimpleNamespace(
            kv_manager=SimpleNamespace(kv_args=SimpleNamespace(state_types=[]))
        ),
        disagg_prefill_pending_chunk_rids=set(),
        checkpoint_disagg_prefill=lambda request: events.append(["checkpoint", request.extend_end]),
        check_bootstrap=lambda request: bootstrap,
    )
    scheduler.send_kv_chunk = lambda request: namespace["_send_kv_chunk"](scheduler, request)
    running = SimpleNamespace(batch_is_full=True)
    for end_tokens in boundaries[:-1]:
        events.append(["forward_complete", end_tokens])
        request.extend_end = end_tokens
        namespace["process_prefill_chunk"](scheduler, None, running)
        events.append(["next_forward", end_tokens])
    request.extend_end = boundaries[-1]
    namespace["_send_kv_chunk"](scheduler, request, last_chunk=True)
    assert [(entry["ready_tokens"], entry["tokens"]) for entry in sends] == expected
    assert all(not entry["last"] for entry in sends[:-1])
    assert sends[-1]["last"]
    assert events.count(["metadata", prompt]) == 1
    assert not scheduler.disagg_prefill_pending_chunk_rids
    if not overlap and bootstrap:
        for entry in sends[:-1]:
            sent = events.index(["send", entry["ready_tokens"], entry["tokens"]])
            following = events.index(["next_forward", entry["ready_tokens"]])
            assert sent < following
    return {
        "prompt_tokens": prompt,
        "boundaries": boundaries,
        "decode_prefix": prefix,
        "overlap": overlap,
        "bootstrap": bootstrap,
        "sends": sends,
        "events": events,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    code = load_functions(args.source_root)
    specifications = [
        (384, [128, 256, 384], 0, [(128, 128), (256, 128), (384, 128)]),
        (257, [127, 255, 257], 0, [(255, 128), (257, 129)]),
        (512, [128, 256, 384, 512], 256, [(384, 128), (512, 128)]),
        (385, [256, 384, 385], 128, [(256, 128), (384, 128), (385, 1)]),
        (129, [64, 129], 0, [(129, 129)]),
        (384, [128, 256, 384], 0, [(384, 384)], True),
        (384, [128, 256, 384], 0, [(384, 384)], False, False),
    ]
    results = [run_case(code, *specification) for specification in specifications]
    report = {
        "upstream_commit": "baa9befa381511cd4d56f83980951e5a45fffc6d",
        "source": SOURCE,
        "source_sha256": SOURCE_SHA256,
        "scope": (
            "Unmodified function bodies; CPU stubs for KV indices, metadata and sender. "
            "Standard non-staging transport, no DCP pack limit. "
            "No measured network timing or serving accuracy claim."
        ),
        "cases_passed": len(results),
        "cases": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases_passed": len(results), "output": str(args.output)}))


if __name__ == "__main__":
    main()
