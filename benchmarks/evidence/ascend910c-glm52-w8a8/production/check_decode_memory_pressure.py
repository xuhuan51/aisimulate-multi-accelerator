"""Execute pinned decode retraction bodies with CPU request and allocator stubs."""

import argparse
import ast
import copy
import hashlib
import json
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace

SOURCES = {
    "managers/schedule_batch.py": "c382bf0066d70e19b4b9e38c92ff0029d39996238f8f0685f4c1dadbfa4e8a96",
    "managers/scheduler_components/new_token_ratio_tracker.py": (
        "d6ad7ede11e9262318649cc5427ee44fb14973d22f8ada26b6d5383a1e77a271"
    ),
}


def load_functions(source_root):
    names = {"_get_decode_retraction_order", "retract_decode", "estimate_new_token_ratio_after_retract"}
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
    assert len(functions) == len(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    return compile(module, "pinned-sglang-decode-memory-pressure", "exec")


def run_case(code, shapes, survivor_fits, expected_retracted, expected_aborted, expected_survivors):
    releases = []
    namespace = {
        "get_schedule": lambda: SimpleNamespace(retraction_policy="length"),
        "beam_retraction_order": lambda indices, requests: indices,
        "FINISH_ABORT": lambda message, status_code: SimpleNamespace(message=message, status_code=status_code),
        "HTTPStatus": HTTPStatus,
        "logger": SimpleNamespace(warning=lambda *args: None),
        "envs": SimpleNamespace(SGLANG_RETRACT_DECODE_STEPS=SimpleNamespace(get=lambda: 20)),
    }
    exec(code, namespace)
    namespace["NewTokenRatioTracker"] = SimpleNamespace(
        estimate_new_token_ratio_after_retract=namespace["estimate_new_token_ratio_after_retract"]
    )
    requests = [
        SimpleNamespace(
            rid=index,
            output_ids=[1] * output,
            origin_input_ids=[2] * prompt,
            sampling_params=SimpleNamespace(max_new_tokens=128),
            beam_group=None,
            to_finish=None,
        )
        for index, (prompt, output) in enumerate(shapes)
    ]
    batch = SimpleNamespace(reqs=requests)

    def release(index, remaining, offload_kv=True):
        releases.append({"request": requests[index].rid, "remaining": remaining, "offload_kv": offload_kv})
        return True

    batch._get_decode_retraction_order = namespace["_get_decode_retraction_order"]
    batch.check_decode_mem = lambda selected_indices: len(selected_indices) <= 1 and survivor_fits
    batch.release_req = release
    batch.filter_batch = lambda keep_indices: setattr(batch, "reqs", [requests[index] for index in keep_indices])
    retracted, ratio, aborted = namespace["retract_decode"](batch)
    assert [request.rid for request in retracted] == expected_retracted
    assert [request.rid for request in aborted] == expected_aborted
    assert [request.rid for request in batch.reqs] == expected_survivors
    assert all(request.to_finish.status_code == HTTPStatus.INTERNAL_SERVER_ERROR for request in aborted)
    assert ratio == min(
        1.0,
        (sum(len(request.output_ids) for request in batch.reqs) + 20 * len(batch.reqs)) / (128 * len(batch.reqs) + 1),
    )
    if aborted:
        assert releases[-1]["offload_kv"] is False
    return {
        "prompt_and_output_lengths": shapes,
        "last_survivor_fits": survivor_fits,
        "retracted": expected_retracted,
        "aborted": expected_aborted,
        "survivors": expected_survivors,
        "new_token_ratio": ratio,
        "releases": releases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    code = load_functions(args.source_root)
    cases = [
        run_case(code, [(4, 1), (8, 1)], True, [1], [], [0]),
        run_case(code, [(4, 1), (4, 1)], True, [1], [], [0]),
        run_case(code, [(4, 1), (8, 2)], True, [0], [], [1]),
        run_case(code, [(4, 1), (4, 1), (4, 1)], True, [2, 1], [], [0]),
        run_case(code, [(4, 1)], False, [], [0], []),
        run_case(code, [(4, 1), (8, 1)], False, [1], [0], []),
    ]
    report = {
        "upstream_revision": "baa9befa381511cd4d56f83980951e5a45fffc6d",
        "source_sha256": SOURCES,
        "scope": (
            "Unmodified default-length retraction and ratio bodies with CPU capacity/release stubs. "
            "No priority, beam, host backup, NPU execution or serving accuracy validation."
        ),
        "cases_passed": len(cases),
        "cases": cases,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases_passed": len(cases), "output": str(args.output)}))


if __name__ == "__main__":
    main()
