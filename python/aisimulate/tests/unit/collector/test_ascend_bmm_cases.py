import json

import pytest
from collector.sglang.ascend_bmm import validate_runtime
from collector.sglang.ascend_bmm_cases import get_ascend_bmm_cases

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "revision",
    [
        "baa9befa381511cd4d56f83980951e5a45fffc6d",
        "76f9213a411018547f4fd6a75f36feaa4d6bed58",
    ],
)
def test_bmm_runtime_must_match_an_audited_frozen_plan(revision):
    validate_runtime(revision)
    validate_runtime(revision, revision)
    with pytest.raises(ValueError, match="Reaudit BMM dispatch"):
        validate_runtime(revision, "different-planned-revision")
    with pytest.raises(ValueError, match="Reaudit BMM dispatch"):
        validate_runtime("unknown-runtime")


def test_all_roles_preserve_the_observed_mla_views(tmp_path):
    config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "num_attention_heads": 64,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    plan = get_ascend_bmm_cases(tmp_path)
    assert len(plan["cases"]) == 450
    assert len({json.dumps(case, sort_keys=True) for case in plan["cases"]}) == 450
    assert {case["local_heads"] for case in plan["cases"]} == {4, 8, 16, 32, 64}
    for case in plan["cases"]:
        if case["role"] == "output_decode":
            assert case["tokens"] <= 32
        if case["role"] == "query_nope":
            assert case["input_stride"] == [case["local_heads"] * 256, 256, 1]
            assert case["weight_stride"] == [98304, 1, 192]
            assert case["output_width"] == 512
        else:
            assert case["input_stride"] == [case["local_heads"] * 512, 512, 1]
            assert case["weight_stride"] == [131072, 256, 1]
            assert case["output_width"] == 256
        assert case["kernel_source"] == (
            "torch.ops.npu.batch_matmul_transpose"
            if case["role"] == "output_decode"
            else "torch_npu.npu_transpose_batchmatmul"
        )
    bounded = get_ascend_bmm_cases(tmp_path, device_memory_bytes=64 * 1024**3)
    excluded = bounded["memory_feasibility"]["excluded"]
    assert excluded and len(bounded["cases"]) + len(excluded) == len(plan["cases"])
    assert all(row["estimated_bytes"] > 64 * 1024**3 * 0.75 for row in excluded)
    assert any(case["tokens"] == 32768 for case in bounded["cases"])
    config["qk_rope_head_dim"] = 32
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="qk_rope_head_dim"):
        get_ascend_bmm_cases(tmp_path)
