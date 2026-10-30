"""Execute pinned SGLang prebuilt-result and decode-selection bodies without devices."""

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

SOURCES = {
    "disaggregation/decode.py": "7a106070dee807c4cc5498dc6d8092b229b23e573bf2dfa44f201ee695815718",
    "managers/scheduler_components/batch_result_processor.py": (
        "bf45be863809db610712b68679a69ba9f9c011e63753f4f2244e3ca824ab1432"
    ),
}


class Request:
    def __init__(self, request_id, maximum, events):
        self.request_id = request_id
        self.maximum = maximum
        self.output_ids = [101]
        self.complete = False
        self.events = events
        self.time_stats = SimpleNamespace(
            set_decode_prebuilt_finish_time=lambda: events.append(["prebuilt", request_id]),
            set_quick_finish_time=lambda: events.append(["finished", request_id]),
        )

    def update_finish_state(self):
        self.complete = len(self.output_ids) >= self.maximum

    def finished(self):
        return self.complete


class Batch:
    def __init__(self, requests):
        self.reqs = requests
        self.return_logprob = False

    def filter_batch(self):
        self.reqs = [request for request in self.reqs if not request.finished()]

    def is_empty(self):
        return not self.reqs

    def merge_batch(self, other):
        self.reqs.extend(other.reqs)


def load_functions(source_root):
    functions = []
    names = {"process_batch_result_prebuilt", "get_next_disagg_decode_batch_to_run"}
    for relative, expected in SOURCES.items():
        content = (source_root / relative).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Reaudit changed SGLang source before using this oracle: {relative}")
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
    return compile(module, "pinned-sglang-prebuilt", "exec")


def run_case(code, maximum, incoming_count, running_count, radix_cache):
    events = []
    incoming = [Request(f"incoming-{index}", maximum, events) for index in range(incoming_count)]
    existing = [Request(f"running-{index}", 10, events) for index in range(running_count)]
    namespace = {
        "DisaggregationMode": SimpleNamespace(DECODE="decode"),
        "get_disagg": lambda: SimpleNamespace(disaggregation_decode_enable_radix_cache=radix_cache),
        "get_memory": lambda: SimpleNamespace(enable_hisparse=False),
        "release_kv_cache": lambda request, cache, checkpoint: events.append(["release", request.request_id]),
        "NextBatchPlan": SimpleNamespace,
        "set_schedule_time_batch": lambda batch: events.append(["schedule", len(batch.reqs)]),
    }
    exec(code, namespace)
    processor = SimpleNamespace(
        disaggregation_mode="decode",
        token_to_kv_pool_allocator=SimpleNamespace(
            free_group_begin=lambda: events.append(["free_group_begin"]),
            free_group_end=lambda: events.append(["free_group_end"]),
        ),
        tree_cache=object(),
        output_streamer=SimpleNamespace(
            stream_output=lambda requests, logprob: events.append(
                ["stream", [[request.request_id, list(request.output_ids)] for request in requests]]
            )
        ),
    )

    def update_running(batch):
        events.append(["prepare_decode", [request.request_id for request in batch.reqs]])
        return batch

    scheduler = SimpleNamespace(
        get_new_prebuilt_batch=lambda running: Batch(incoming),
        chunked_req=None,
        enable_hisparse=False,
        batch_result_processor=SimpleNamespace(
            process_batch_result_prebuilt=lambda batch: namespace["process_batch_result_prebuilt"](processor, batch)
        ),
        update_running_batch=update_running,
        dp_attn_adapter=SimpleNamespace(maybe_prepare_mlp_sync_batch=lambda batch: batch),
    )
    result = namespace["get_next_disagg_decode_batch_to_run"](scheduler, Batch(existing))
    streamed = next(index for index, event in enumerate(events) if event[0] == "stream")
    preparation = next((index for index, event in enumerate(events) if event[0] == "prepare_decode"), None)
    expected_running = running_count + (incoming_count if maximum > 1 else 0)
    assert (result.batch_to_run is not None) == (expected_running > 0)
    assert preparation is None or streamed < preparation
    assert len(result.running_batch.reqs) == expected_running
    assert sum(event[0] == "release" for event in events) == (incoming_count if maximum == 1 else 0)
    assert all(len(request.output_ids) == 1 for request in incoming)
    return {
        "maximum": maximum,
        "incoming": incoming_count,
        "running": running_count,
        "radix_cache": radix_cache,
        "events": events,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    code = load_functions(args.source_root)
    cases = [
        run_case(code, maximum, incoming, running, radix_cache)
        for maximum in [1, 2, 4]
        for incoming in [1, 2]
        for running in [0, 1]
        for radix_cache in [False, True]
    ]
    report = {
        "source_revision": "baa9befa381511cd4d56f83980951e5a45fffc6d",
        "source_sha256": SOURCES,
        "scope": "Original prebuilt-result and decode-selection bodies; tensor, allocator and network calls stubbed",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": len(cases)}))


if __name__ == "__main__":
    main()
