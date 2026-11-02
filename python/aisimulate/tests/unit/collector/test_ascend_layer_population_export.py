import copy

import pytest

from collector.sglang.ascend_layer_export import MODEL_SHA256, MODEL_SOURCE
from collector.sglang.ascend_layer_population_export import qualified_population

pytestmark = pytest.mark.unit


def population_observations(counts, padded):
    control = {"id": "population", "input_length": 8192, "global_requests": sum(counts)}
    records = []
    for rank in range(16):
        requests = counts[rank // 4]
        boundary = {
            "module": "DeepseekV2DecoderLayer",
            "layer": 4,
            "function_profile": None,
            "forward_mode": "2" if requests else "4",
            "batch_size": padded[rank // 4],
            "global_num_tokens_cpu": padded,
            "sequence_lengths": [8200] * requests + [0] * (padded[rank // 4] - requests),
            "samples": [
                {
                    "repeat": repeat,
                    "warmup": repeat < 3,
                    "device_ms": 2.0,
                    "topk_validation": {"hidden": [{"valid_rows": requests}], "topk": {"valid_rows": requests}},
                }
                for repeat in range(12)
            ],
        }
        records.append(
            {
                "rank": rank,
                "status": "completed",
                "control": control,
                "source_sha256": {MODEL_SOURCE: MODEL_SHA256},
                "boundaries": [boundary],
                "numerical_contract": "all_logical_hidden_values_and_all_topk_columns_for_positive_kv_rows/v3",
            }
        )
    return records, control


@pytest.mark.parametrize(
    "counts,padded,active,requests",
    [([0, 0, 1, 0], [0, 0, 4, 0], 1, 1), ([1, 0, 1, 0], [4] * 4, 2, 1), ([2, 2, 2, 2], [4] * 4, 4, 2)],
)
def test_idle_population_and_physical_padding_remain_separate(counts, padded, active, requests):
    records, control = population_observations(counts, padded)
    row, samples = qualified_population(records, 4, control)
    assert row["active_attention_dp"] == active
    assert row["requests_per_dp"] == requests
    assert row["latency_ms"] == 2.0
    assert samples == [2.0] * 9


def test_ignoring_a_real_request_or_using_decode_for_idle_fails():
    records, control = population_observations([1, 0, 1, 0], [4] * 4)
    unchecked = copy.deepcopy(records)
    unchecked[0]["boundaries"][0]["samples"][3]["topk_validation"]["topk"]["valid_rows"] = 0
    with pytest.raises(ValueError, match="numerically checked"):
        qualified_population(unchecked, 4, control)
    records[4]["boundaries"][0]["forward_mode"] = "2"
    with pytest.raises(ValueError, match="idle/decode"):
        qualified_population(records, 4, control)


def test_unqualified_skew_and_missing_tp_rank_cannot_be_published():
    records, control = population_observations([1, 2, 0, 0], [4] * 4)
    with pytest.raises(ValueError, match="uniform"):
        qualified_population(records, 4, control)
    with pytest.raises(ValueError, match="sixteen"):
        qualified_population(records[:-1], 4, control)
