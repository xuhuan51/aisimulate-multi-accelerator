import copy

import pytest
from collector.sglang.ascend_layer_export import MODEL_SHA256, MODEL_SOURCE
from collector.sglang.ascend_layer_population_export import qualified_population

pytestmark = pytest.mark.unit


def population_observations(counts, padded, attention_tp=4):
    control = {"id": "population", "input_length": 8192, "global_requests": sum(counts)}
    records = []
    for rank in range(len(counts) * attention_tp):
        requests = counts[rank // attention_tp]
        boundary = {
            "module": "DeepseekV2DecoderLayer",
            "layer": 4,
            "function_profile": None,
            "forward_mode": "2" if requests else "4",
            "batch_size": padded[rank // attention_tp],
            "global_num_tokens_cpu": padded,
            "sequence_lengths": [8200] * requests + [0] * (padded[rank // attention_tp] - requests),
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
                "parallelism": {
                    "global_tp": len(counts) * attention_tp,
                    "attention_tp": attention_tp,
                    "attention_dp": len(counts),
                },
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
    with pytest.raises(ValueError, match="rank observations"):
        qualified_population(records[:-1], 4, control)


@pytest.mark.parametrize("active,padded", [(4, [1] * 4 + [0] * 28), (16, [1] * 32), (32, [1] * 32)])
def test_tp32_population_uses_all_ranks_and_preserves_valid_rows(active, padded):
    records, control = population_observations([1] * active + [0] * (32 - active), padded, attention_tp=1)
    records[31]["boundaries"][0]["samples"][5]["device_ms"] = 3.0
    row, samples = qualified_population(records, 4, control, attention_tp=1, attention_dp=32, moe_ep=32)
    assert (row["attention_tp"], row["attention_dp"], row["moe_ep"]) == (1, 32, 32)
    assert row["active_attention_dp"] == active
    assert row["numerical_checks"] == 384
    assert samples == [2.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    records[31]["parallelism"]["attention_tp"] = 4
    with pytest.raises(ValueError, match="Unqualified population"):
        qualified_population(records, 4, control, attention_tp=1, attention_dp=32, moe_ep=32)
