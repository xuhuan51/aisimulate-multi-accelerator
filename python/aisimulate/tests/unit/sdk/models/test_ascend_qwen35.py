import copy
from collections import Counter

import pytest

from aisimulate_core.sdk.models.ascend_qwen35 import physical_specs, validate_checkpoint

pytestmark = pytest.mark.unit


def raw_config():
    text = dict(
        hidden_size=4096,
        num_hidden_layers=60,
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=256,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=1024,
        shared_expert_intermediate_size=1024,
        vocab_size=248320,
        linear_num_key_heads=16,
        linear_num_value_heads=64,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    )
    kinds = ["linear_attention"] * 3 + ["full_attention"]
    text["layer_types"] = kinds * 15
    layers = []
    for i, kind in enumerate(text["layer_types"]):
        attn = (
            dict.fromkeys(("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "conv1d", "norm"), "FLOAT")
            if kind == "linear_attention"
            else dict(
                q_proj="W8A8_DYNAMIC",
                k_proj="W8A8_DYNAMIC",
                v_proj="W8A8_DYNAMIC",
                o_proj="FLOAT",
                q_norm="FLOAT",
                k_norm="FLOAT",
            )
        )
        layers.append(
            dict(
                layer=i,
                kind=kind,
                attention=attn,
                norms=dict(input_layernorm="FLOAT", post_attention_layernorm="FLOAT"),
                mlp=dict(
                    experts=512,
                    routed=dict.fromkeys(("gate_proj", "up_proj", "down_proj"), "W8A8_DYNAMIC"),
                    router="FLOAT",
                    shared=dict.fromkeys(("gate_proj", "up_proj", "down_proj"), "FLOAT"),
                    shared_gate="FLOAT",
                ),
            )
        )
    return dict(
        text_config=text,
        modelslim_metadata=dict(
            schema="aisimulate-qwen35-modelslim-metadata/v1", layers=layers, embedding="FLOAT", lm_head="FLOAT"
        ),
    )


@pytest.mark.parametrize("tp", [8, 16])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_hybrid_graph_covers_all_layers_and_replication(tp, phase):
    raw = raw_config()
    before = copy.deepcopy(raw)
    specs = physical_specs(raw, tp, phase)
    assert raw == before
    qwen = [x["AscendQwen"] for x in specs if "AscendQwen" in x]
    count = Counter(x["key"]["role"] for x in qwen)
    assert count["gdn_scan"] == 45 and count["gqa_attention"] == 15
    assert count["routed_experts"] == 60 and count["expert_dispatch"] == 60 and count["expert_combine"] == 60
    assert count["gdn_state_io"] == (45 if phase == "prefill" else 0)
    assert count["rms_norm"] == 1 and count["add_rms_norm"] == 120
    linears = {x["AscendLinear"]["name"]: x["AscendLinear"] for x in specs if "AscendLinear" in x}
    # Qwen's gate doubles Q width; the single rank-local KV head is replicated.
    qkv = linears[f"{phase}_layer3_qkv"]["key"]
    assert qkv["n"] == (2560 if tp == 8 else 1536)
    assert qkv["quant_mode"] == "w8a8_dynamic"
    assert linears[f"{phase}_layer0_qkvz"]["key"]["quant_mode"] == "bfloat16"
    assert linears[f"{phase}_layer3_out"]["key"]["quant_mode"] == "bfloat16"
    # DeepEP shared expert is replicated (1024 intermediate), not TP-sharded.
    assert linears[f"{phase}_layer0_shared_up"]["key"]["n"] == 2048
    assert linears[f"{phase}_logits"]["token_axis"] == "requests"
    assert all(not x.get("Overlap") for x in specs)
    names = [next(iter(x.values()))["name"] for x in specs]
    assert len(names) == len(set(names))
    assert (
        names.index(f"{phase}_layer2_scan")
        < names.index(f"{phase}_layer3_attention")
        < names.index(f"{phase}_layer4_scan")
    )


@pytest.mark.parametrize("mutation", ["geometry", "precision", "layer_count"])
def test_reject_unqualified_checkpoint(mutation):
    raw = raw_config()
    if mutation == "geometry":
        raw["text_config"]["num_experts"] = 256
    elif mutation == "precision":
        raw["modelslim_metadata"]["layers"][59]["attention"]["q_proj"] = "FLOAT"
    else:
        raw["modelslim_metadata"]["layers"].pop()
    with pytest.raises(ValueError):
        validate_checkpoint(raw)


def test_native_wire_roundtrip():
    import json

    import aisimulate_core._native as core

    for spec in physical_specs(raw_config(), 8, "decode"):
        assert json.loads(core.op_from_spec_json(json.dumps(spec))._spec_json()) == spec


@pytest.mark.parametrize(
    "overlap,policy,accepted", [(False, "measured", True), (True, "measured", False), (False, "anchored", False)]
)
def test_resolved_estimator_controls_are_not_silently_ignored(overlap, policy, accepted):
    import json
    from types import SimpleNamespace

    from aisimulate_core.sdk.config_builders import build_model_config
    from aisimulate_core.sdk.fpm_config import OpLevelCompileConfig
    from aisimulate_core.sdk.models.ascend_qwen35 import configure_ascend_qwen35

    config = build_model_config(
        tp_size=8,
        pp_size=1,
        attention_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=8,
        gemm_quant_mode="bfloat16",
        moe_quant_mode="w8a8_dynamic",
        kvcache_quant_mode="bfloat16",
        system="ascend910c",
    )
    options = {"ascend": {"shared_expert_overlap": overlap, "shape_policy": policy, "decode_execution_mode": "eager"}}
    config.op_level_config = OpLevelCompileConfig(options, system="ascend910c", backend="sglang")
    saved = config.op_level_config.cache_identity()
    restored = OpLevelCompileConfig(json.loads(saved), system="ascend910c", backend="sglang")
    assert restored.cache_identity() == saved
    model = SimpleNamespace(config=config, _backend_name="sglang")
    if accepted:
        configure_ascend_qwen35(model, raw_config())
        assert model._mamba_ssm_dtype == "bfloat16"
        assert model.execution_assumptions["shared_expert_overlap"] is False
        assert config.op_level_config.cache_identity() == saved
    else:
        with pytest.raises(ValueError, match="explicit shared_expert_overlap=false"):
            configure_ascend_qwen35(model, raw_config())


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_attention_dp_uses_local_attention_and_global_expert_groups(phase):
    specs = physical_specs(raw_config(), 8, phase, attention_dp=2)
    leaves = [s.get("TokenScale", {}).get("op", s) for s in specs]
    qwen = [s["AscendQwen"] for s in leaves if "AscendQwen" in s]
    assert {s["key"]["tp"] for s in qwen if s["key"]["role"] == "gdn_scan"} == {8}
    assert {s["key"]["tp"] for s in qwen if s["key"]["role"] == "routed_experts"} == {16}
    assert {s["Hccl"]["key"]["num_ranks"] for s in specs if "Hccl" in s} == {8}
    wrapped = [s["TokenScale"] for s in specs if "TokenScale" in s]
    assert len(wrapped) == (180 if phase == "prefill" else 0)
    assert all(s["numerator"] == 2 and s["denominator"] == 1 for s in wrapped)


def test_single_attention_rank_retains_both_kv_heads():
    specs = physical_specs(raw_config(), 1, "decode", attention_dp=8)
    qkv = next(s["AscendLinear"] for s in specs if s.get("AscendLinear", {}).get("name") == "decode_layer3_qkv")
    assert qkv["key"]["n"] == 2 * 32 * 256 + 2 * 2 * 256
    with pytest.raises(ValueError):
        physical_specs(raw_config(), 8, "decode", attention_dp=3)


def test_graph_decode_matches_observed_tp_scatter_and_collectives():
    # All four observed loads (1..4 requests/DP) have attention M4 and MoE M1.
    specs = physical_specs(raw_config(), 4, "decode", attention_dp=4, decode_execution_mode="graph")
    linears = {s["AscendLinear"]["name"]: s["AscendLinear"] for s in specs if "AscendLinear" in s}
    assert linears["decode_layer0_qkvz"]["token_multiple"] == 4
    assert linears["decode_layer0_router_gemm"]["token_split"] == 4
    assert linears["decode_layer0_shared_up"]["token_split"] == 4
    assert linears["decode_logits"]["key"]["n"] == 62080
    assert linears["decode_logits"]["token_multiple"] == 4
    collectives = [s["Hccl"] for s in specs if "Hccl" in s]
    counts = Counter(s["key"]["operation"] for s in collectives)
    assert counts == {"reduce_scatter": 60, "all_gather": 61}
    assert all(s["key"]["execution_mode"] == "graph" for s in collectives)
    router = next(s["AscendQwen"] for s in specs if s.get("AscendQwen", {}).get("name") == "decode_layer0_router")
    assert router["boundary_contract"].endswith("fp32_state_graph_tp4_dp4_ep16/v1")
    attention = next(s["AscendQwen"] for s in specs if s.get("AscendQwen", {}).get("name") == "decode_layer3_attention")
    assert attention["boundary_contract"].endswith("bf16_state_eager/v1")
    with pytest.raises(ValueError, match="TP4/DP4/EP16"):
        physical_specs(raw_config(), 8, "decode", attention_dp=2, decode_execution_mode="graph")


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_dp_embedding_matches_pinned_qwen_replication_hook(phase):
    # Qwen3_5ForCausalLM._build_embed_tokens passes enable_tp=False for DP.
    specs = physical_specs(raw_config(), 8, phase, attention_dp=2)
    embedding = next(
        s["AscendQwen"] for s in specs if s.get("AscendQwen", {}).get("key", {}).get("role") == "embedding"
    )
    assert embedding["key"]["tp"] == 1
    assert not any(s.get("Hccl", {}).get("name") == phase + "_embedding_sum" for s in specs)
    baseline = physical_specs(raw_config(), 8, phase)
    assert any(s.get("Hccl", {}).get("name") == phase + "_embedding_sum" for s in baseline)
