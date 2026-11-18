import json

import pytest
from collector.sglang.ascend_hccl_export import validated_rows

pytestmark = pytest.mark.unit


def runtime():
    return dict.fromkeys(
        (
            "image_digest",
            "framework_commit",
            "cann_hccl_build",
            "driver_versions",
            "hardware",
            "collector_source_sha256",
        ),
        "test-fixture",
    )


def campaign(tmp_path, name, samples=None, iterations=16):
    path = tmp_path / name
    row = {
        "dtype": "bfloat16",
        "operation": "all_reduce",
        "execution_mode": "graph",
        "input_bytes_per_rank": 1024,
        "full_payload_correctness_checks": 4,
        "execution_verification": "captured_graph_replay",
        "verification_buffers_per_rank": iterations,
        "rank_samples": [{"device_ms": [1.0] * 3}, {"device_ms": samples or [2.0] * 3}],
    }
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "warmup": 10,
                "repeats": 3,
                "iterations": iterations,
                "schema": "aisimulate-hccl-observations/v3",
                "warmup_scope": "batches_of_iterations",
                "all_reduce_buffer_policy": "one_buffer_per_invocation",
                "collector_source_sha256": "test-fixture",
                "framework_commit": "test-fixture",
                "runtime_environment": {"TASK_QUEUE_ENABLE": "0", "HCCL_BUFFSIZE": "650"},
                "hccl_group_options": {"hccl_buffer_size": 650},
                "serving_setup": {
                    "schema": "sglang-runtime-setup/v1",
                    "environment": {"TASK_QUEUE_ENABLE": "0", "HCCL_BUFFSIZE": "650"},
                    "worker_cpu_affinity": {"0": [0], "1": [1]},
                },
                "num_ranks": 2,
                "ranks_per_node": 2,
                "rank_layout": "contiguous",
                "devices": [
                    {"rank": 0, "local_rank": 0, "cpu_affinity": [0]},
                    {"rank": 1, "local_rank": 1, "cpu_affinity": [1]},
                ],
                "cases": [row],
            }
        )
    )
    return path


def test_critical_rank_timing_and_exact_transport_identity(tmp_path):
    rows, report = validated_rows([campaign(tmp_path, "valid.json")], runtime())
    assert rows[0]["latency_ms"] == 2.0
    assert rows[0]["kernel_source"] == "torch.distributed.hccl"
    assert rows[0]["dtype"] == "bfloat16"
    assert rows[0]["correctness_checks"] == 4
    assert rows[0]["execution_verification"] == "captured_graph_replay"
    assert report["unqualified"] == []


def test_rejects_duplicates_and_protocol_mixing(tmp_path):
    first = campaign(tmp_path, "first.json")
    with pytest.raises(ValueError, match="Duplicate HCCL physical key"):
        validated_rows([first, first], runtime())
    second = campaign(tmp_path, "second.json", iterations=64)
    with pytest.raises(ValueError, match="different measurement protocols"):
        validated_rows([first, second], runtime())


def test_unstable_measurement_cannot_be_published(tmp_path):
    path = campaign(tmp_path, "unstable.json", samples=[2.0, 2.0, 10.0])
    with pytest.raises(ValueError, match="No qualified HCCL rows"):
        validated_rows([path], runtime())
    value = json.loads(path.read_text())
    value["cases"][0]["full_payload_correctness_checks"] = 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Missing pre/post correctness"):
        validated_rows([path], runtime())


def test_unrecorded_or_mismatched_runtime_options_cannot_be_published(tmp_path):
    path = campaign(tmp_path, "runtime.json")
    value = json.loads(path.read_text())
    value["hccl_group_options"]["hccl_buffer_size"] = 200
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="buffer differs"):
        validated_rows([path], runtime())
    value["schema"] = "aisimulate-hccl-observations/v1"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="requires v3"):
        validated_rows([path], runtime())


def test_eager_checks_cannot_qualify_graph_timings(tmp_path):
    path = campaign(tmp_path, "graph.json")
    value = json.loads(path.read_text())
    value["cases"][0]["execution_verification"] = "eager_invocation"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="actual timed buffers/execution path"):
        validated_rows([path], runtime())
    value["cases"][0]["execution_verification"] = "captured_graph_replay"
    value["cases"][0]["verification_buffers_per_rank"] = 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="actual timed buffers/execution path"):
        validated_rows([path], runtime())
    value["schema"] = "aisimulate-hccl-observations/v2"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="requires v3"):
        validated_rows([path], runtime())


def test_runtime_source_must_match_observation(tmp_path):
    path = campaign(tmp_path, "source.json")
    value = json.loads(path.read_text())
    value["framework_commit"] = "another-commit"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="source differs"):
        validated_rows([path], runtime())


def test_unset_runtime_option_is_preserved_in_pinned_identity(tmp_path):
    path = campaign(tmp_path, "unset.json")
    _, explicit = validated_rows([path], runtime())
    value = json.loads(path.read_text())
    value["runtime_environment"]["TASK_QUEUE_ENABLE"] = None
    value["serving_setup"]["environment"]["TASK_QUEUE_ENABLE"] = None
    path.write_text(json.dumps(value))
    rows, unset = validated_rows([path], runtime())
    assert rows[0]["latency_ms"] == 2.0
    assert unset["identity"]["measurement_protocol"]["runtime_environment"]["TASK_QUEUE_ENABLE"] is None
    assert unset["runtime_identity_sha256"] != explicit["runtime_identity_sha256"]

    value["serving_setup"]["environment"]["TASK_QUEUE_ENABLE"] = "1"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="environment differs"):
        validated_rows([path], runtime())


@pytest.mark.parametrize("option", ["2", "", False, 0])
def test_invalid_runtime_option_cannot_be_published(tmp_path, option):
    path = campaign(tmp_path, "invalid.json")
    value = json.loads(path.read_text())
    value["runtime_environment"]["TASK_QUEUE_ENABLE"] = option
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="task-queue policy"):
        validated_rows([path], runtime())


def test_missing_runtime_option_cannot_be_published(tmp_path):
    path = campaign(tmp_path, "missing.json")
    value = json.loads(path.read_text())
    del value["runtime_environment"]["TASK_QUEUE_ENABLE"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="task-queue policy"):
        validated_rows([path], runtime())
