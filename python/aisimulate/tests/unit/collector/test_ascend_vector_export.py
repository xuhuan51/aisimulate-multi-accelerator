import hashlib
import json
import sys
from pathlib import Path

import pytest


from collector.sglang.ascend_vector import runtime_version, validate_runtime_case
from collector.sglang.ascend_vector_export import expected_invocation, matches_signature, validated_rows

pytestmark = pytest.mark.unit


def test_legacy_vector_runtime_is_limited_to_audited_roles():
    legacy = "76f9213a411018547f4fd6a75f36feaa4d6bed58"
    assert runtime_version(legacy) == "0.5.21.dev223+g76f9213a4"
    for role in ("rope_attention", "split_qk_norm"):
        validate_runtime_case(legacy, {"role": role})
    with pytest.raises(ValueError, match="serving audit"):
        validate_runtime_case(legacy, {"role": "cache_scatter"})
    with pytest.raises(ValueError, match="Reaudit"):
        runtime_version("unknown-commit")
    with pytest.raises(ValueError, match="frozen vector plan"):
        runtime_version("baa9befa381511cd4d56f83980951e5a45fffc6d", legacy)


@pytest.mark.parametrize(
    ("revision", "version"),
    [
        ("baa9befa381511cd4d56f83980951e5a45fffc6d", "0.5.22"),
        ("76f9213a411018547f4fd6a75f36feaa4d6bed58", "0.5.21.dev223+g76f9213a4"),
    ],
)
def test_vector_export_records_the_measured_framework_version(campaign, tmp_path, monkeypatch, revision, version):
    from collector import provenance
    from collector.sglang import ascend_vector_export

    plan, observation, identity, record = campaign
    declared = json.loads(plan.read_text())
    for index, case in enumerate(declared["cases"]):
        case.update(role="split_qk_norm", dimensions=[2048, 512, 64])
        inputs, outputs, kernel = expected_invocation(case)
        record["cases"][index].update(**case, input_signature=inputs, output_signature=outputs, kernel_source=kernel)
    plan.write_text(json.dumps(declared))
    record["plan_sha256"] = hashlib.sha256(plan.read_bytes()).hexdigest()
    identity["framework_commit"] = record["runtime"]["framework_commit"] = revision
    source = Path(ascend_vector_export.__file__).parent
    identity["collector_source_sha256"] = {
        name: hashlib.sha256((source / filename).read_bytes()).hexdigest()
        for name, filename in {"collector": "ascend_vector.py", "kernels": "ascend_vector_kernels.py"}.items()
    }
    record["collector_source_sha256"] = identity["collector_source_sha256"]
    observation.write_text(json.dumps(record))
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(identity))
    published = {}
    monkeypatch.setattr(
        provenance, "write_collection_meta", lambda directory, runtime, tables, **kwargs: published.update(runtime)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ascend_vector_export",
            str(observation),
            "--plan",
            str(plan),
            "--runtime-identity",
            str(identity_path),
            "--output-dir",
            str(tmp_path / "published"),
            "--collector-ref",
            "test",
        ],
    )
    ascend_vector_export.main()
    assert published["version"] == version
    assert published["source_commit"] == revision


@pytest.fixture
def campaign(tmp_path):
    cases = [
        {"role": "cache_scatter", "dimensions": [53376, 128], "tokens": 8, "execution_mode": mode}
        for mode in ("eager", "graph")
    ]
    protocol = {"iterations": 16, "warmup": 5, "repeats": 3}
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": "aisimulate-ascend-vector-plan/v1", "cases": cases, "protocol": protocol}))
    identity = dict.fromkeys(
        ("image_digest", "framework_commit", "driver_version_info", "hardware", "torch", "torch_npu"), "fixture"
    )
    identity["collector_source_sha256"] = {"collector": "first", "kernels": "second"}
    rows = []
    for index, case in enumerate(cases):
        inputs, outputs, kernel = expected_invocation(case)
        rows.append(
            {
                **case,
                "plan_index": index,
                "input_signature": inputs,
                "output_signature": outputs,
                "kernel_source": kernel,
                "execution_verification": "captured_graph_replay"
                if case["execution_mode"] == "graph"
                else "eager_invocation",
                "checks": [{"sampled_elements": 128, "max_abs_error": 0.0}] * 2,
                "samples_ms": [float(index + 1)] * 3,
            }
        )
    record = {
        "schema": "aisimulate-ascend-vector-observations/v1",
        "status": "completed",
        "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
        "collector_source_sha256": identity["collector_source_sha256"],
        "protocol": protocol,
        "runtime": {
            "framework_commit": "fixture",
            "driver_version_info": "fixture",
            "device": "fixture",
            "torch": "fixture",
            "torch_npu": "fixture",
            "device_index": 0,
            "cpu_affinity": [0],
            "serving_setup": {"worker_cpu_affinity": {"0": [0]}},
        },
        "selected_plan_indices": [0, 1],
        "cases": rows,
    }
    observation = tmp_path / "observations.json"
    observation.write_text(json.dumps(record))
    return plan, observation, identity, record


def test_cache_index_dtype_and_ncdhw_views_are_not_interchangeable(campaign):
    plan, observation, identity, record = campaign
    rows, report = validated_rows(plan, [observation], identity)
    assert report["qualified"] == 2 and rows[0]["dimensions"] == "[53376,128]"
    assert rows[0]["signature_sha256"] != rows[1]["signature_sha256"]
    record["cases"][1]["input_signature"]["indices"]["dtype"] = "torch.int64"
    observation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="consumer contract"):
        validated_rows(plan, [observation], identity)


def test_singleton_stride_is_irrelevant_but_query_head_stride_and_offset_are_not():
    case = {"role": "rope_attention", "dimensions": [8, 64, 256, 64], "tokens": 1, "execution_mode": "graph"}
    expected, _, _ = expected_invocation(case)
    actual = json.loads(json.dumps(expected))
    actual["query"]["stride"][0] = 512
    assert matches_signature(actual, expected)
    actual["query"]["stride"][1] = 64
    assert not matches_signature(actual, expected)
    actual = json.loads(json.dumps(expected))
    actual["query"]["storage_offset"] = 0
    assert not matches_signature(actual, expected)


def test_sequence_length_copy_requires_cpu_int32_unpinned_input():
    case = {"role": "seq_lens_h2d", "dimensions": [4], "tokens": 3, "execution_mode": "eager"}
    inputs, outputs, kernel = expected_invocation(case)
    assert outputs["shape"] == [3] and outputs["dtype"] == "torch.int32"
    assert kernel == "torch.Tensor.to" and matches_signature(inputs, inputs)
    for field, wrong in (("device", "npu"), ("pinned", True), ("dtype", "torch.int64"), ("format", 2)):
        altered = json.loads(json.dumps(inputs))
        altered["lengths"][field] = wrong
        assert not matches_signature(altered, inputs)
    with pytest.raises(ValueError, match="Unsupported vector"):
        expected_invocation({**case, "execution_mode": "graph"})


def test_unstable_rows_are_reported_and_partial_grid_cannot_publish(campaign):
    plan, observation, identity, record = campaign
    record["cases"][0]["samples_ms"] = [1.0, 1.0, 9.0]
    observation.write_text(json.dumps(record))
    rows, report = validated_rows(plan, [observation], identity)
    assert len(rows) == 1 and report["unqualified"][0]["plan_index"] == 0
    record["cases"].pop()
    record["selected_plan_indices"] = [0]
    observation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="complete frozen plan"):
        validated_rows(plan, [observation], identity)


def test_wrong_kernel_actual_execution_and_runtime_fail_closed(campaign):
    plan, observation, identity, record = campaign
    with pytest.raises(ValueError, match="Duplicate"):
        validated_rows(plan, [observation, observation], identity)
    record["cases"][0]["kernel_source"] = "another_scatter"
    observation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="consumer contract"):
        validated_rows(plan, [observation], identity)
    record["cases"][0]["kernel_source"] = "torch_npu.npu_scatter_nd_update_"
    record["cases"][1]["execution_verification"] = "eager_invocation"
    observation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Actual execution"):
        validated_rows(plan, [observation], identity)
    record["runtime"]["torch_npu"] = "other"
    observation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Runtime"):
        validated_rows(plan, [observation], identity)
