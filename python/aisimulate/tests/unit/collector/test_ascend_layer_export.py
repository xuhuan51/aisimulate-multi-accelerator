import copy

import pytest

from collector.sglang.ascend_layer_export import MODEL_SHA256, MODEL_SOURCE, qualified_boundary

pytestmark = pytest.mark.unit


def observations():
    control = {"id": "fixture", "global_requests": 4, "input_length": 8192}
    records = []
    for rank in range(16):
        records.append(
            {
                "rank": rank,
                "status": "completed",
                "control": control,
                "source_sha256": {MODEL_SOURCE: MODEL_SHA256},
                "boundaries": [
                    {
                        "module": "DeepseekV2DecoderLayer",
                        "layer": 4,
                        "function_profile": None,
                        "sequence_lengths": [8237, 0, 0, 0],
                        "forward_mode": "2",
                        "batch_size": 4,
                        "global_num_tokens_cpu": [4] * 4,
                        "samples": [
                            {"device_ms": 2.0 if rank == 15 else 1.0, "warmup": repeat < 3, "repeat": repeat}
                            for repeat in range(12)
                        ],
                    }
                ],
            }
        )
    return records, control


def test_critical_rank_is_measured_without_summing_parallel_ranks():
    records, control = observations()
    row, samples = qualified_boundary(records, 4, control)
    assert row["latency_ms"] == 2.0
    assert samples == [2.0] * 9
    assert row["cv"] == 0.0
    assert row["numerical_checks"] == 192


@pytest.mark.parametrize(
    "field,value",
    [
        ("function_profile", [{"self_seconds": 0.1}]),
        ("forward_mode", "1"),
        ("sequence_lengths", [8237, 8237, 0, 0]),
        ("global_num_tokens_cpu", [4, 0, 0, 0]),
    ],
)
def test_unqualified_boundaries_cannot_be_published(field, value):
    records, control = observations()
    records[0]["boundaries"][0][field] = value
    with pytest.raises(ValueError):
        qualified_boundary(records, 4, control)


def test_source_identity_and_every_timed_sample_are_required():
    records, control = observations()
    other = copy.deepcopy(records)
    other[0]["source_sha256"][MODEL_SOURCE] = "0" * 64
    with pytest.raises(ValueError, match="source"):
        qualified_boundary(other, 4, control)
    records[0]["boundaries"][0]["samples"].pop()
    with pytest.raises(ValueError, match="nine"):
        qualified_boundary(records, 4, control)
