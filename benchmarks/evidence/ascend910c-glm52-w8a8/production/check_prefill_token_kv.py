"""Execute pinned prefill-result and decode-preparation bodies with CPU stubs."""

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

SOURCES = {
    "managers/scheduler_components/batch_result_processor.py": (
        "bf45be863809db610712b68679a69ba9f9c011e63753f4f2244e3ca824ab1432"
    ),
    "managers/schedule_batch.py": "c382bf0066d70e19b4b9e38c92ff0029d39996238f8f0685f4c1dadbfa4e8a96",
}


class Lengths(list):
    def __add__(self, increment):
        return Lengths(value + increment for value in self)


class Request:
    def __init__(self, prompt, maximum):
        self.maximum = maximum
        self.output_ids = []
        self.kv = SimpleNamespace(kv_committed_len=prompt)
        self.is_retracted = False
        self.inflight_middle_chunks = 0
        self.discard_output_reason = None
        self.return_sampling_mask = False
        self.beam_group = None
        self.grammar = None
        self.decode_batch_idx = 0
        self.time_stats = SimpleNamespace(
            set_prefill_finished_time=lambda: None,
            set_completion_time=lambda: None,
        )

    def finished(self):
        return len(self.output_ids) >= self.maximum

    def update_finish_state(self):
        return None


def load_functions(source_root):
    names = {"process_batch_result_prefill", "prepare_for_decode"}
    functions = []
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
    assert len(functions) == 2
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    return compile(module, "pinned-sglang-prefill-kv", "exec")


def run_case(code, prompt, maximum, count):
    events = []
    requests = [Request(prompt, maximum) for _ in range(count)]

    def allocate(batch, token_per_req):
        events.append(["allocate_decode", len(batch.reqs), token_per_req])
        return object()

    namespace = {
        "get_memory": lambda: SimpleNamespace(enable_hisparse=False),
        "get_exec": lambda: SimpleNamespace(mamba=SimpleNamespace(enable_mamba_extra_buffer=False)),
        "checkpoint_kv_cache": lambda request, cache: events.append(["checkpoint", request.kv.kv_committed_len]),
        "release_kv_cache": lambda request, cache, checkpoint: events.append(["release", request.kv.kv_committed_len]),
        "ForwardMode": SimpleNamespace(DECODE="decode"),
        "strip_beam_tail": lambda batch: None,
        "append_beam_tail": lambda batch: None,
        "alloc_for_decode": allocate,
    }
    exec(code, namespace)
    batch = SimpleNamespace(
        reqs=requests,
        return_hidden_states=False,
        return_logprob=False,
        decoding_reqs=None,
        prefill_stats=None,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        sampling_info=SimpleNamespace(penalizer_orchestrator=SimpleNamespace(is_required=False)),
        model_config=SimpleNamespace(is_encoder_decoder=False),
        seq_lens=Lengths([prompt] * count),
        seq_lens_cpu=Lengths([prompt] * count),
        orig_seq_lens=Lengths([prompt] * count),
        hisparse_coordinator=None,
    )
    result = SimpleNamespace(
        copy_done=None,
        auxiliary_host_output=None,
        routed_experts_output=None,
        indexer_topk_output=None,
        logits_output=None,
        next_token_ids=SimpleNamespace(tolist=lambda: [101] * count),
        extend_input_len_per_req=None,
        extend_logprob_start_len_per_req=None,
        can_run_cuda_graph=False,
    )
    scheduler = SimpleNamespace(
        is_generation=True,
        token_to_kv_pool_allocator=SimpleNamespace(
            free_group_begin=lambda: events.append(["free_group_begin"]),
            free_group_end=lambda: events.append(["free_group_end"]),
        ),
        snapshot_auxiliary_output_starts=lambda batch, result: None,
        move_logprobs_to_cpu=lambda **kwargs: None,
        materialize_sampling_mask_output=lambda *args: None,
        _validate_pp_skip_output_comm=lambda *args: None,
        _get_prefill_hidden_capture_mode=lambda *args: None,
        _maybe_update_reasoning_tokens=lambda *args: None,
        _maybe_collect_routed_experts=lambda *args: None,
        _maybe_collect_indexer_topk=lambda *args: None,
        _maybe_collect_customized_info=lambda *args: None,
        tree_cache=object(),
        output_streamer=SimpleNamespace(
            stream_output=lambda requests, logprob, skip: events.append(
                ["stream", [list(request.output_ids) for request in requests]]
            )
        ),
    )
    namespace["process_batch_result_prefill"](scheduler, batch, result)
    assert all(request.output_ids == [101] for request in requests)
    assert all(request.kv.kv_committed_len == prompt for request in requests)
    assert list(batch.seq_lens) == [prompt] * count
    assert not any(event[0] == "allocate_decode" for event in events)
    stream_index = next(index for index, event in enumerate(events) if event[0] == "stream")
    if maximum > 1:
        namespace["prepare_for_decode"](batch)
        assert events[-1] == ["allocate_decode", count, 1]
        assert list(batch.seq_lens) == [prompt + 1] * count
        assert all(request.decode_batch_idx == 1 for request in requests)
        assert stream_index < len(events) - 1
    else:
        assert sum(event[0] == "release" for event in events) == count
    return {"prompt_tokens": prompt, "max_output_tokens": maximum, "requests": count, "events": events}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    code = load_functions(args.source_root)
    cases = [run_case(code, prompt, maximum, count) for prompt in [3, 4] for maximum in [1, 2, 4] for count in [1, 2]]
    report = {
        "upstream_revision": "baa9befa381511cd4d56f83980951e5a45fffc6d",
        "source_sha256": SOURCES,
        "scope": (
            "Unmodified ordinary-prefill result and non-speculative decode preparation bodies. "
            "CPU allocator/cache/streaming stubs witness call ordering; no NPU execution or memory measurement."
        ),
        "cases_passed": len(cases),
        "cases": cases,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases_passed": len(cases), "output": str(args.output)}))


if __name__ == "__main__":
    main()
