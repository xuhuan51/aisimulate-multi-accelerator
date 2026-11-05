import hashlib
import json
from types import SimpleNamespace

import pytest
from collector.sglang.ascend_deepep import checked_combine_precision, checked_dispatch_precision
from collector.sglang.ascend_deepep_export import validated_rows

pytestmark = pytest.mark.unit


def test_combine_precision_requires_bf16_even_when_values_match():
    assert checked_combine_precision(SimpleNamespace(dtype="torch.bfloat16")) == {"combine_dtype": "bfloat16"}
    with pytest.raises(ValueError, match="actual combine precision"):
        checked_combine_precision(SimpleNamespace(dtype="torch.float32"))


@pytest.mark.parametrize("dtype,scale", [("torch.bfloat16", None), ("torch.int8", "torch.bfloat16")])
def test_dispatch_precision_is_checked_before_dequantization(dtype, scale):
    output = SimpleNamespace(
        hidden_states=SimpleNamespace(dtype=dtype),
        hidden_states_scale=None if scale is None else SimpleNamespace(dtype=scale),
    )
    with pytest.raises(ValueError, match="actual dispatch precision"):
        checked_dispatch_precision(output)


def test_recorded_dispatch_precision_matches_actual_output():
    output = SimpleNamespace(
        hidden_states=SimpleNamespace(dtype="torch.int8"),
        hidden_states_scale=SimpleNamespace(dtype="torch.float32"),
    )
    assert checked_dispatch_precision(output) == {"dispatch_dtype": "int8", "dispatch_scale_dtype": "float32"}


@pytest.fixture
def campaign(tmp_path):
    case = {
        "phase": "low_latency",
        "num_ranks": 16,
        "ranks_per_node": 16,
        "rank_layout": "contiguous",
        "tokens_per_rank": 1,
        "distribution": "single_expert_group",
        "execution_mode": "graph",
        "hidden_size": 6144,
        "num_experts": 256,
        "topk": 8,
        "max_dispatch_tokens_per_rank": 32,
    }
    protocol = {"iterations": 8, "warmup": 5, "repeats": 3}
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": "aisimulate-ascend-deepep-plan/v1", "cases": [case], "protocol": protocol}))
    identity = dict.fromkeys(
        (
            "torch",
            "torch_npu",
            "deep_ep",
            "driver_version_info",
            "image_digest",
            "framework_commit",
            "hardware",
            "collector_source_sha256",
        ),
        "fixture",
    )
    setup = {"worker_cpu_affinity": {str(rank): [rank] for rank in range(16)}}
    devices = [
        {
            "rank": rank,
            "local_rank": rank,
            "host": "node",
            "device": "fixture",
            "cpu_affinity": [rank],
            **dict.fromkeys(("torch", "torch_npu", "deep_ep", "driver_version_info"), "fixture"),
        }
        for rank in range(16)
    ]
    measurements = []
    for operation in ("dispatch", "combine"):
        rank_results = []
        for rank in range(16):
            counts = [16] * 8 + [0] * 8 if rank == 0 else [0] * 16
            elements = sum(counts) * 6144 if operation == "dispatch" else 6144
            rank_results.append(
                {
                    "rank": rank,
                    "samples_ms": [4.0 if rank == 15 else 1.0] * 3,
                    "wall_samples_ms": [5.0] * 3,
                    "checks": [{"checked_elements": elements, "calls": 1}] * 2,
                    "expected_counts": counts,
                    "received_shape": [4096, 6144],
                }
            )
        measurements.append(
            {
                "operation": operation,
                "rank_results": rank_results,
                "latency_ms": 4.0,
                "cv": 0.0,
                "execution_verification": "captured_graph_replay",
            }
        )
    record = {
        "schema": "aisimulate-ascend-deepep-observations/v1",
        "status": "completed",
        "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
        "collector_source_sha256": "fixture",
        "framework_commit": "fixture",
        "protocol": protocol,
        "serving_setup": setup,
        "devices": devices,
        "timing_method": "per_call_outer_events_barrier_between_calls",
        "selected_plan_indices": [0],
        "negative_control": None,
        "cases": [
            {
                **case,
                "plan_index": 0,
                "measurements": measurements,
                "kernel_source": "sglang.srt.layers.moe.token_dispatcher.deepep._DeepEPDispatcherImplLowLatency",
            }
        ],
    }
    observation = tmp_path / "observations.json"
    observation.write_text(json.dumps(record))
    return plan, observation, identity, record


def test_rank_critical_path_and_empty_receivers_keep_full_checks(campaign):
    plan, observation, identity, _ = campaign
    rows, report = validated_rows(plan, [observation], identity)
    assert report["qualified"] == 2
    assert [row["latency_ms"] for row in rows] == [4.0, 4.0]
    assert rows[0]["numerical_checks"] == 128 * 6144 * 2
    assert rows[1]["numerical_checks"] == 16 * 6144 * 2
    assert {row["dispatch_dtype"] for row in rows} == {"int8"}


@pytest.mark.parametrize(
    "precision",
    [
        {"dispatch_dtype": "bfloat16", "dispatch_scale_dtype": None},
        {"dispatch_dtype": "int8"},
        {"dispatch_dtype": "int8", "dispatch_scale_dtype": "bfloat16"},
    ],
)
def test_export_rejects_precision_mislabeled_as_int8(campaign, precision):
    plan, observation, identity, record = campaign
    record["cases"][0]["measurements"][0]["rank_results"][0].update(precision)
    observation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Observed dispatch precision"):
        validated_rows(plan, [observation], identity)


def test_export_accepts_explicit_precision_from_every_rank(campaign):
    plan, observation, identity, record = campaign
    for measurement in record["cases"][0]["measurements"]:
        for rank in measurement["rank_results"]:
            rank.update(dispatch_dtype="int8", dispatch_scale_dtype="float32", combine_dtype="bfloat16")
    observation.write_text(json.dumps(record))
    rows, _ = validated_rows(plan, [observation], identity)
    assert {row["dispatch_dtype"] for row in rows} == {"int8"}


def test_export_rejects_combine_output_mislabeled_as_bf16(campaign):
    plan, observation, identity, record = campaign
    record["cases"][0]["measurements"][1]["rank_results"][0]["combine_dtype"] = "float32"
    observation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Observed combine precision"):
        validated_rows(plan, [observation], identity)


@pytest.mark.parametrize("failure", ("missing_rank", "unchecked_output", "wrong_routing", "rank_runtime", "negative"))
def test_incomplete_or_mislabeled_transport_cannot_publish(campaign, failure):
    plan, observation, identity, record = campaign
    results = record["cases"][0]["measurements"][0]["rank_results"]
    if failure == "missing_rank":
        results.pop()
    elif failure == "unchecked_output":
        results[0]["checks"][0]["checked_elements"] -= 1
    elif failure == "wrong_routing":
        results[0]["expected_counts"] = [8] * 16
    elif failure == "rank_runtime":
        record["devices"][3]["deep_ep"] = "another-build"
    else:
        record["negative_control"] = "dispatch"
    observation.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        validated_rows(plan, [observation], identity)


def test_missing_or_duplicate_cases_fail_the_publication_gate(campaign):
    plan, observation, identity, record = campaign
    with pytest.raises(ValueError, match="Duplicate"):
        validated_rows(plan, [observation, observation], identity)
    record["cases"] = []
    record["selected_plan_indices"] = []
    observation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="complete frozen plan"):
        validated_rows(plan, [observation], identity)


def test_failed_execution_is_preserved_without_fabricating_transport_rows(campaign):
    plan, observation, identity, record = campaign
    planned = json.loads(plan.read_text())
    planned["cases"].append({**planned["cases"][0], "tokens_per_rank": 2})
    plan.write_text(json.dumps(planned))
    record.update(
        status="failed",
        failed_plan_index=1,
        error="RuntimeError: operator tiling failed before execution",
        selected_plan_indices=[0, 1],
        plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
    )
    observation.write_text(json.dumps(record))
    rows, report = validated_rows(plan, [observation], identity)
    assert len(rows) == 2 and report["attempted"] == 4
    assert {row["tokens_per_rank"] for row in rows} == {1}
    assert report["unqualified"][0]["classification"] == "unexpected"
