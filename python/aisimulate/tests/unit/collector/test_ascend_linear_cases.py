import json
from pathlib import Path

import pytest
from collector.sglang.ascend_linear import validate_runtime
from collector.sglang.ascend_linear_cases import PHYSICAL_FIELDS, get_ascend_linear_cases

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "revision",
    [
        "baa9befa381511cd4d56f83980951e5a45fffc6d",
        "76f9213a411018547f4fd6a75f36feaa4d6bed58",
    ],
)
def test_linear_runtime_must_match_an_audited_frozen_plan(revision):
    validate_runtime(revision)
    validate_runtime(revision, revision)
    with pytest.raises(ValueError, match="Reaudit linear dispatch"):
        validate_runtime(revision, "different-planned-revision")
    with pytest.raises(ValueError, match="Reaudit linear dispatch"):
        validate_runtime("unknown-runtime")


@pytest.fixture
def artifact(tmp_path):
    source = Path(__file__).resolve().parents[3] / "src/aisimulate_core/model_configs/zai-org--GLM-5.2_config.json"
    config = json.loads(source.read_text())
    (tmp_path / "config.json").write_text(json.dumps(config))
    description = {}
    for module in ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj", "indexer.wq_b"):
        description[f"model.layers.0.self_attn.{module}.weight"] = "W8A8"
    for module in ("indexer.wk", "indexer.weights_proj"):
        description[f"model.layers.0.self_attn.{module}.weight"] = "FLOAT"
    description["lm_head.weight"] = "FLOAT"
    description["model.layers.3.mlp.gate.weight"] = "FLOAT"
    for projection in ("gate_proj", "up_proj", "down_proj"):
        description[f"model.layers.0.mlp.{projection}.weight"] = "W8A8_DYNAMIC"
        description[f"model.layers.3.mlp.shared_experts.{projection}.weight"] = "W8A8_DYNAMIC"
    (tmp_path / "quant_model_description.json").write_text(json.dumps(description))
    return tmp_path


def test_real_profile_deduplicates_equivalent_invocations_without_merging_bias(artifact):
    plan = get_ascend_linear_cases(artifact)
    assert plan["recipe_invocations"] > plan["unique_invocations"]
    cases = plan["cases"]
    assert len({tuple(case[field] for field in PHYSICAL_FIELDS) for case in cases}) == len(cases)
    projection = next(
        case
        for case in cases
        if (case["quant_mode"], case["n"], case["k"], case["m"], case["execution_mode"])
        == ("w8a8", 4096, 2048, 8, "graph")
    )
    assert set(projection["owners"]) == {"q_b.tp4", "indexer_q"}
    output = [
        case
        for case in cases
        if (case["quant_mode"], case["n"], case["k"], case["m"], case["execution_mode"])
        == ("w8a8", 6144, 2048, 8, "graph")
    ]
    assert {case["quant_bias"] for case in output} == {True, False}


def test_dense_and_shared_shapes_follow_their_actual_parallel_axes(artifact):
    cases = get_ascend_linear_cases(artifact)["cases"]
    for case in cases:
        if "dense_up.tp16" in case["owners"]:
            assert (case["n"], case["k"], case["quant_mode"]) == (1536, 6144, "w8a8_dynamic")
        if "shared_up" in case["owners"]:
            assert (case["n"], case["k"]) == (4096, 6144)
        if "shared_tensor_up.tp16" in case["owners"]:
            assert (case["n"], case["k"]) == (256, 6144)
        if "shared_tensor_down.tp32" in case["owners"]:
            assert (case["n"], case["k"]) == (6144, 64)
        if any(owner.startswith("lm_head") for owner in case["owners"]):
            assert case["m"] <= 64
        if "indexer_weights" in case["owners"]:
            assert case["quant_mode"] == "float32_bfloat16"


def test_artifact_mismatch_never_uses_declared_default_shapes(artifact):
    path = artifact / "config.json"
    config = json.loads(path.read_text())
    config["n_shared_experts"] = 2
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="n_shared_experts"):
        get_ascend_linear_cases(artifact)


def test_wrong_precision_cannot_be_named_w8a8(artifact):
    path = artifact / "quant_model_description.json"
    description = json.loads(path.read_text())
    description["model.layers.0.self_attn.q_b_proj.weight"] = "W4A8"
    path.write_text(json.dumps(description))
    with pytest.raises(ValueError, match="precision"):
        get_ascend_linear_cases(artifact)
