"""Execute the pinned SGLang source-side transfer completion poll without devices."""

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

SOURCE_SHA256 = "386c5d007000eecead04f3527d2c780f26b14ca5e1e438346afa40eceba666d9"


def run_case(poll, statuses):
    events = []
    requests = []
    for index, status in enumerate(statuses):
        rid = f"request-{index}"
        sender = SimpleNamespace(
            status=status,
            clear=lambda rid=rid: events.append({"event": "clear_sender", "rid": rid}),
            get_transfer_metric=lambda: {},
        )
        requests.append(
            SimpleNamespace(
                rid=rid,
                pending_bootstrap=False,
                return_routed_experts=False,
                return_logprob=False,
                finished_reason=None,
                bootstrap_host="registered-peer",
                cache_request_handle=rid,
                disagg_kv_sender=sender,
                time_stats=SimpleNamespace(
                    set_prefill_kv_transfer_finish_time=lambda: None,
                    set_completion_time=lambda: None,
                    compute_and_observe_kv_transfer_metrics=lambda value: {},
                ),
            )
        )
    namespace = poll.__globals__
    namespace["release_kv_cache"] = lambda req, cache, checkpoint: events.append(
        {"event": "release_kv", "rid": req.rid, "checkpoint": checkpoint}
    )
    namespace["maybe_release_metadata_buffer"] = lambda req, allocator: events.append(
        {"event": "release_metadata", "rid": req.rid}
    )
    context = SimpleNamespace(
        disagg_prefill_inflight_queue=requests,
        attn_cp_cpu_group=None,
        attn_tp_cpu_group=None,
        req_to_metadata_buffer_idx_allocator=None,
        tree_cache=SimpleNamespace(finish=lambda handle, outcome: None),
        output_streamer=SimpleNamespace(
            stream_output=lambda done, logprob, extra: events.append(
                {"event": "stream_output", "rids": [req.rid for req in done]}
            )
        ),
    )
    completed = poll(context)
    expected = [f"request-{index}" for index, status in enumerate(statuses) if status == "success"]
    assert [req.rid for req in completed] == expected
    assert [event["rid"] for event in events if event["event"] == "release_kv"] == expected
    assert all(event["checkpoint"] for event in events if event["event"] == "release_kv")
    assert [req.rid for req in context.disagg_prefill_inflight_queue] == [
        f"request-{index}" for index, status in enumerate(statuses) if status != "success"
    ]
    previous_releases = sum(event["event"] == "release_kv" for event in events)
    assert poll(context) == []
    assert sum(event["event"] == "release_kv" for event in events) == previous_releases
    return {"poll_states": statuses, "completed": expected, "events": events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = args.source.read_bytes()
    if hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
        raise ValueError("Reaudit changed SGLang source before using this oracle")
    node = next(
        node
        for node in ast.walk(ast.parse(content))
        if isinstance(node, ast.FunctionDef) and node.name == "process_disagg_prefill_inflight_queue"
    )
    node = copy.deepcopy(node)
    node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "KVPoll": SimpleNamespace(
            Success="success", Failed="failed", WaitingForInput="waiting", Transferring="transferring"
        ),
        "FINISH_ABORT": type("FinishAbort", (), {}),
        "FINISH_LENGTH": lambda length: SimpleNamespace(length=length),
        "FAKE_BOOTSTRAP_HOST": "fake-bootstrap",
        "CacheRequestOutcome": SimpleNamespace(SUCCESS="success"),
        "poll_and_all_reduce_attn_cp_tp_group": lambda senders, cp_group, tp_group: [
            sender.status for sender in senders
        ],
    }
    exec(compile(module, str(args.source), "exec"), namespace)
    poll = namespace[node.name]
    cases = [
        [],
        ["waiting"],
        ["transferring"],
        ["success"],
        ["success", "transferring"],
        ["transferring", "success"],
        ["success"] * 3,
    ]
    report = {
        "source_revision": "baa9befa381511cd4d56f83980951e5a45fffc6d",
        "source_sha256": SOURCE_SHA256,
        "function": node.name,
        "scope": (
            "Original source poll body with stubbed transport/cache/telemetry; "
            "destination scheduler timing is covered by separate native integration tests"
        ),
        "cases": [run_case(poll, statuses) for statuses in cases],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": len(cases), "source_sha256": SOURCE_SHA256}))


if __name__ == "__main__":
    main()
