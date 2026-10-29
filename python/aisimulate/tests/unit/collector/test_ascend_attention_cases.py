import json
from pathlib import Path

import pytest
from collector.sglang.ascend_attention_cases import get_ascend_attention_cases

pytestmark = pytest.mark.unit


@pytest.fixture
def model(tmp_path):
    config = Path(__file__).resolve().parents[3] / "src/aisimulate_core/model_configs/zai-org--GLM-5.2_config.json"
    (tmp_path / "config.json").write_bytes(config.read_bytes())
    return tmp_path


def test_active_requests_and_graph_padding_remain_separate(model):
    cases = get_ascend_attention_cases(model)["cases"]
    case = next(
        row
        for row in cases
        if row["operation"] == "lightning_indexer"
        and row["phase"] == "decode"
        and row["execution_mode"] == "graph"
        and row["active_batch"] == 5
        and row["sequence_length"] == 9216
    )
    assert case["tokens"] == case["padded_batch"] == 8
    assert case["block_table_width"] == 1407
    assert case["pool_pages"] == 417
    assert case["page_size"] == 128
    assert len({json.dumps(row, sort_keys=True) for row in cases}) == len(cases)


def test_prefix_cache_and_head_geometry_follow_declared_model(model):
    cases = get_ascend_attention_cases(model)["cases"]
    assert {row["local_heads"] for row in cases if row["operation"] == "sparse_attention"} == {4, 8, 16, 32, 64}
    assert {row["local_heads"] for row in cases if row["operation"] == "lightning_indexer"} == {32}
    case = next(
        row
        for row in cases
        if row["phase"] == "prefill" and row["active_batch"] == 1 and row["tokens"] == 8192 and row["prefix"] == 8192
    )
    assert case["sequence_length"] == 16384 and case["block_table_width"] == 128
    config = json.loads((model / "config.json").read_text())
    config["kv_lora_rank"] = 256
    (model / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="kv_lora_rank"):
        get_ascend_attention_cases(model)


def test_multi_request_prefill_and_lower_graph_bucket_edges_are_covered(model):
    cases = get_ascend_attention_cases(model)["cases"]
    prefill = next(
        row
        for row in cases
        if row["phase"] == "prefill" and row["active_batch"] == 4 and row["tokens"] == 8192 and row["prefix"] == 8192
    )
    assert prefill["sequence_length"] == 10240 and prefill["padded_batch"] == 4
    assert all(row["tokens"] <= 32768 for row in cases if row["phase"] == "prefill")
    assert any(
        row["phase"] == "prefill"
        and row["tokens"] == 16384
        and row["active_batch"] == 1
        and row["prefix"] == 16384
        and row["sequence_length"] == 32768
        for row in cases
    )
    assert any(row["phase"] == "decode" and row["sequence_length"] == 33792 for row in cases)
    for active, padded in ((9, 16), (17, 24), (25, 32)):
        assert any(
            row["phase"] == "decode"
            and row["active_batch"] == active
            and row["padded_batch"] == padded
            and row["execution_mode"] == "graph"
            for row in cases
        )


def test_larger_minimum_graph_bucket_covers_tp16_without_duplicate_work(model):
    cases = get_ascend_attention_cases(model)["cases"]
    for operation, heads in (("lightning_indexer", 32), ("sparse_attention", 4)):
        for active, padded in ((1, 16), (8, 16), (17, 32), (24, 32)):
            matching = [
                case
                for case in cases
                if case["operation"] == operation
                and case["local_heads"] == heads
                and case["phase"] == "decode"
                and case["execution_mode"] == "graph"
                and case["active_batch"] == active
                and case["padded_batch"] == padded
                and case["sequence_length"] == 9216
            ]
            assert len(matching) == 1
    assert len(cases) == len({json.dumps(case, sort_keys=True) for case in cases})


def test_eager_decode_keeps_tp_padding_distinct_from_active_requests(model):
    cases = get_ascend_attention_cases(model)["cases"]
    for heads, active, expected in ((8, 1, {1, 8}), (16, 5, {5, 8}), (4, 17, {17, 32}), (64, 5, {5})):
        matching = [
            case
            for case in cases
            if case["operation"] == "sparse_attention"
            and case["local_heads"] == heads
            and case["phase"] == "decode"
            and case["execution_mode"] == "eager"
            and case["active_batch"] == active
            and case["sequence_length"] == 9216
        ]
        assert {case["padded_batch"] for case in matching} == expected
        assert all(case["tokens"] == case["padded_batch"] for case in matching)
    indexer = {
        case["padded_batch"]
        for case in cases
        if case["operation"] == "lightning_indexer"
        and case["phase"] == "decode"
        and case["execution_mode"] == "eager"
        and case["active_batch"] == 1
    }
    assert indexer == {1, 2, 4, 8, 16}
