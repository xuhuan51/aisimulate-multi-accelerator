import copy
from collections import Counter

import pytest

from aisimulate_core.sdk.models.ascend_qwen35 import physical_specs, validate_checkpoint

pytestmark = pytest.mark.unit


def test_host_cpu_bound_preserves_device_path_and_architectural_counts():
    import json

    import aisimulate_core._native as core

    options = {"attention_dp": 4, "qwen_runtime_contract": "modelslim_fp32_tp_scatter"}
    original = physical_specs(raw_config(), 4, "decode", **options)
    specs = physical_specs(raw_config(), 4, "decode", qwen_eager_host_cpu_bound=True, **options)
    root = specs[0]["Overlap"]
    assert root["group_a"] == original
    assert json.loads(core.op_from_spec_json(json.dumps(specs[0]))._spec_json()) == specs[0]
    hosts = []

    def visit(entries):
        for entry in entries:
            if "Overlap" in entry:
                visit(entry["Overlap"]["group_a"])
                visit(entry["Overlap"]["group_b"])
            else:
                hosts.append(entry["AscendQwen"])

    visit(root["group_b"])
    assert len(hosts) == 16 * 6
    for rank in range(16):
        observed = {op["key"]["role"]: op["scale_factor"] for op in hosts if op["name"].endswith(f"_rank{rank}")}
        assert observed == {
            f"host_cpu_{component}_rank{rank}": count
            for component, count in {
                "linear_attention": 45.0,
                "linear_layer_host": 45.0,
                "full_attention_layer_host": 15.0,
                "moe": 60.0,
                "layer_norms": 60.0,
                "outer_forward": 1.0,
            }.items()
        }


@pytest.mark.parametrize("tp,dp", [(2, 8), (8, 2)])
def test_host_cpu_transfer_is_explicit_and_retains_its_tp4_profile_identity(tp, dp):
    options = dict(attention_dp=dp, qwen_runtime_contract="modelslim_fp32_tp_scatter")
    baseline = physical_specs(raw_config(), tp, "decode", **options)
    candidate = physical_specs(
        raw_config(), tp, "decode", qwen_eager_host_cpu_bound=True, qwen_eager_host_cpu_transfer=True, **options
    )
    assert candidate[0]["Overlap"]["group_a"] == baseline
    import json

    assert '"tp": 4' in json.dumps(candidate[0]["Overlap"]["group_b"])
    with pytest.raises(ValueError, match="Host CPU transfer requires"):
        physical_specs(raw_config(), tp, "decode", qwen_eager_host_cpu_transfer=True, **options)


@pytest.mark.parametrize(
    "tp,dp,phase,mode,contract",
    [
        (2, 8, "decode", "eager", "modelslim_fp32_tp_scatter"),
        (4, 4, "prefill", "eager", "modelslim_fp32_tp_scatter"),
        (4, 4, "decode", "graph", "modelslim_fp32_tp_scatter"),
        (4, 4, "decode", "eager", "historical"),
    ],
)
def test_host_cpu_bound_rejects_unmeasured_execution(tp, dp, phase, mode, contract):
    with pytest.raises(ValueError, match="host CPU bound"):
        physical_specs(
            raw_config(),
            tp,
            phase,
            attention_dp=dp,
            decode_execution_mode=mode,
            qwen_runtime_contract=contract,
            qwen_eager_host_cpu_bound=True,
        )


@pytest.mark.parametrize("tp,dp", [(2, 8), (4, 4), (8, 2)])
def test_normal_transport_preserves_each_layer_vector_and_replaces_its_two_boundaries(tp, dp):
    workload = dict(input_tokens=8192, layer_expert_counts=[[160] * 512 for _ in range(60)])
    workload["layer_expert_counts"][1][0] += 1
    workload["layer_expert_counts"][1][1] -= 1
    specs = physical_specs(
        raw_config(),
        tp,
        "prefill",
        dp,
        prefill_expert_workload=workload,
        qwen_prefill_current_roles=["normal_transport"],
    )
    selected = [
        spec["TokenScale"] for spec in specs if "AscendNormalTransport" in spec.get("TokenScale", {}).get("op", {})
    ]
    assert len(selected) == 120
    for index, wrapped in enumerate(selected):
        op = wrapped["op"]["AscendNormalTransport"]
        assert wrapped["numerator"] == dp and wrapped["denominator"] == 1
        assert op["expert_counts"] == workload["layer_expert_counts"][index // 2]
        assert op["reference_tokens"] == 8192
        assert op["key"] == dict(
            role="normal_dispatch_ready" if index % 2 == 0 else "normal_combine", tp=16, phase="prefill"
        )


def test_normal_transport_requires_an_explicit_workload_reference():
    from aisimulate_core.sdk.models.ascend_qwen35 import InvalidEngineConfigurationError

    with pytest.raises(InvalidEngineConfigurationError, match="expert workload"):
        physical_specs(raw_config(), 4, "prefill", 4, qwen_prefill_current_roles=["normal_transport"])


def test_prefill_int8_expert_reference_keeps_its_input_boundary_explicit():
    specs = physical_specs(raw_config(), 4, "prefill", 4, qwen_prefill_current_roles=["routed_experts"])
    selected = []
    for spec in specs:
        op = spec.get("TokenScale", {}).get("op", spec).get("AscendQwen")
        if op is not None and op["key"]["role"] == "routed_experts":
            selected.append(op)
            assert op["boundary_contract"] == "qwen35_397b_modelslim_prefill_int8_expert_chain/v1"
            assert op["key"] == dict(role="routed_experts", tp=16, phase="prefill")
    assert len(selected) == 60


@pytest.mark.parametrize("tp,dp", [(2, 8), (4, 4), (8, 2)])
def test_fp32_eager_mixed_prefill_keeps_decode_fp32_and_labels_historical_costs(tp, dp):
    roles = ["gdn_conv", "gdn_scan", "gdn_state_io", "gqa_attention", "expert_dispatch"]
    current = f"qwen35_397b_modelslim_fp32_state_eager_tp{tp}_dp{dp}_ep16/v1"
    specs = physical_specs(
        raw_config(),
        tp,
        "prefill",
        dp,
        qwen_runtime_contract="modelslim_fp32_tp_scatter",
        qwen_prefill_current_roles=roles,
    )
    for spec in specs:
        op = spec.get("TokenScale", {}).get("op", spec).get("AscendQwen")
        if op is not None:
            expected = current if op["key"]["role"] in roles else "qwen35_397b_modelslim_bf16_state_eager/v1"
            assert op["boundary_contract"] == expected
    decode = physical_specs(
        raw_config(),
        tp,
        "decode",
        dp,
        "eager",
        qwen_runtime_contract="modelslim_fp32_tp_scatter",
    )
    for spec in decode:
        op = spec.get("TokenScale", {}).get("op", spec).get("AscendQwen")
        if op is not None:
            assert op["boundary_contract"] == current


@pytest.mark.parametrize("tp,dp", [(2, 8), (4, 4), (8, 2)])
def test_current_prefill_components_select_only_their_physical_contracts(tp, dp):
    import json
    from types import SimpleNamespace

    from aisimulate_core.sdk.config_builders import build_model_config
    from aisimulate_core.sdk.fpm_config import OpLevelCompileConfig
    from aisimulate_core.sdk.models.ascend_qwen35 import configure_ascend_qwen35

    roles = ["gdn_scan", "gdn_state_io", "gqa_attention", "expert_dispatch"]
    config = build_model_config(
        tp_size=tp,
        pp_size=1,
        attention_dp_size=dp,
        moe_tp_size=1,
        moe_ep_size=16,
        gemm_quant_mode="bfloat16",
        moe_quant_mode="w8a8_dynamic",
        kvcache_quant_mode="bfloat16",
        system="ascend910c",
    )
    baseline = SimpleNamespace(config=copy.deepcopy(config), _backend_name="sglang")
    configure_ascend_qwen35(baseline, raw_config())
    config.op_level_config = OpLevelCompileConfig(
        {"ascend": {"qwen_prefill_current_roles": roles}},
        system="ascend910c",
        backend="sglang",
    )
    identity = config.op_level_config.cache_identity()
    restored = OpLevelCompileConfig(json.loads(identity), system="ascend910c", backend="sglang")
    assert restored.cache_identity() == identity
    assert set(restored.options["ascend"]["qwen_prefill_current_roles"]) == set(roles)
    current = SimpleNamespace(config=config, _backend_name="sglang")
    configure_ascend_qwen35(current, raw_config())
    selected = Counter()
    for before, after in zip(baseline.context_ops, current.context_ops, strict=True):
        previous, actual = json.loads(before._spec_json()), json.loads(after._spec_json())
        old_op = previous.get("TokenScale", {}).get("op", previous)
        new_op = actual.get("TokenScale", {}).get("op", actual)
        role = new_op.get("AscendQwen", {}).get("key", {}).get("role")
        if role in roles:
            expected = f"qwen35_397b_modelslim_fp32_state_eager_tp{tp}_dp{dp}_ep16/v1"
            assert new_op["AscendQwen"]["boundary_contract"] == expected
            new_op["AscendQwen"]["boundary_contract"] = old_op["AscendQwen"]["boundary_contract"]
            selected[role] += 1
        assert actual == previous
    assert selected == Counter(gdn_scan=45, gdn_state_io=45, gqa_attention=15, expert_dispatch=60)
    assert [op._spec_json() for op in current.generation_ops] == [op._spec_json() for op in baseline.generation_ops]
    assert sum(op.get_weights() for op in current.context_ops) == sum(op.get_weights() for op in baseline.context_ops)
    assert current.execution_assumptions["qwen_prefill_current_roles"] == roles


@pytest.mark.parametrize("tp,dp", [(2, 8), (4, 4), (8, 2)])
def test_observed_prefill_workload_changes_only_expert_compute_shape(tp, dp):
    import json
    from types import SimpleNamespace

    from aisimulate_core.sdk.config_builders import build_model_config
    from aisimulate_core.sdk.fpm_config import OpLevelCompileConfig
    from aisimulate_core.sdk.models.ascend_qwen35 import configure_ascend_qwen35

    raw = raw_config()
    raw["modelslim_metadata"]["source_sha256"] = "b" * 64
    balanced = ([1] * 20 + [0] * 12) * 16
    concentrated = [32] * 10 + [0] * 502
    workload = dict(
        observation_id="synthetic32-token-top10",
        quantization_sha256="b" * 64,
        input_tokens=32,
        layer_expert_counts=[concentrated] + [balanced[:] for _ in range(59)],
    )
    config = build_model_config(
        tp_size=tp,
        pp_size=1,
        attention_dp_size=dp,
        moe_tp_size=1,
        moe_ep_size=16,
        gemm_quant_mode="bfloat16",
        moe_quant_mode="w8a8_dynamic",
        kvcache_quant_mode="bfloat16",
        system="ascend910c",
    )
    baseline = SimpleNamespace(config=copy.deepcopy(config), _backend_name="sglang")
    configure_ascend_qwen35(baseline, raw)
    config.op_level_config = OpLevelCompileConfig(
        {"ascend": {"prefill_expert_workload": workload}}, system="ascend910c", backend="sglang"
    )
    saved = config.op_level_config.cache_identity()
    restored = OpLevelCompileConfig(json.loads(saved), system="ascend910c", backend="sglang")
    assert restored.cache_identity() == saved
    assert restored.options["ascend"]["prefill_expert_workload"] == workload
    observed = SimpleNamespace(config=config, _backend_name="sglang")
    configure_ascend_qwen35(observed, raw)
    baseline_specs = [json.loads(op._spec_json()) for op in baseline.context_ops]
    observed_specs = [json.loads(op._spec_json()) for op in observed.context_ops]
    for index, (before, after) in enumerate(zip(baseline_specs, observed_specs, strict=True)):
        if (
            before.get("TokenScale", {}).get("op", {}).get("AscendQwen", {}).get("key", {}).get("role")
            == "routed_experts"
        ):
            actual = after["TokenScale"]
            # Layer0 sends320 assignments to one rank,16 times the balanced20.
            # Other layers send20 to every rank and preserve the original axis.
            expected = dp * (16 if actual["op"]["AscendQwen"]["name"] == "prefill_layer0_experts" else 1)
            assert (actual["numerator"], actual["denominator"]) == (expected, 1)
            assert actual["op"] == before["TokenScale"]["op"]
        else:
            assert after == before, index
    assert sum(op.get_weights() for op in observed.context_ops) == sum(op.get_weights() for op in baseline.context_ops)
    assert [op._spec_json() for op in observed.generation_ops] == [op._spec_json() for op in baseline.generation_ops]
    assert observed.execution_assumptions["prefill_expert_workload"]["observation_id"] == workload["observation_id"]
    config.op_level_config = OpLevelCompileConfig(
        {"ascend": {"prefill_expert_workload": {**workload, "quantization_sha256": "c" * 64}}},
        system="ascend910c",
        backend="sglang",
    )
    with pytest.raises(ValueError, match="matching ModelSlim"):
        configure_ascend_qwen35(SimpleNamespace(config=config, _backend_name="sglang"), raw)


def test_observed_graph_experts_use_received_volume_without_changing_default_roles():
    raw = raw_config()
    workload = {"observation_id": "source-pinned-counts"}
    specs = physical_specs(raw, 4, "decode", 4, "graph", True, decode_expert_workload=workload)
    experts = [entry["AscendMoeChain"] for entry in specs if "AscendMoeChain" in entry]
    assert len(experts) == 60
    assert all(
        entry["key"] == dict(phase="decode", local_experts=32, distribution="balanced", execution_mode="graph")
        for entry in experts
    )
    assert all(
        entry["token_numerator"] == 40
        and entry["token_denominator"] == 16
        and entry["input_token_multiple"] == 4
        and entry["decode_workload"] == workload
        for entry in experts
    )
    baseline = physical_specs(raw, 4, "decode", 4, "graph", True)
    assert not any("AscendMoeChain" in entry for entry in baseline)
    current_prefill_runtime = physical_specs(
        raw,
        4,
        "decode",
        4,
        "graph",
        True,
        qwen_runtime_contract="modelslim_fp32_tp_scatter",
        decode_expert_workload=workload,
    )
    assert current_prefill_runtime == specs
    for tp, dp, phase, mode, contract in [
        (2, 8, "decode", "graph", None),
        (4, 4, "prefill", "graph", None),
        (4, 4, "decode", "eager", None),
    ]:
        with pytest.raises(ValueError, match="Observed Qwen expert curves"):
            physical_specs(raw, tp, phase, dp, mode, False, contract, workload)


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
    with pytest.raises(ValueError, match="with EP16"):
        physical_specs(raw_config(), 8, "decode", attention_dp=1, decode_execution_mode="graph")


@pytest.mark.parametrize("tp,dp", [(2, 8), (4, 4), (8, 2)])
def test_fp32_eager_runtime_uses_observed_tp_scatter(tp, dp):
    # The pinned FP32 eager trace has padded attention rows, TP-local MoE
    # rows and 60 RS + 61 AG calls, including the logits exchange.
    specs = physical_specs(
        raw_config(),
        tp,
        "decode",
        attention_dp=dp,
        qwen_runtime_contract="modelslim_fp32_tp_scatter",
    )
    linears = {spec["AscendLinear"]["name"]: spec["AscendLinear"] for spec in specs if "AscendLinear" in spec}
    assert linears["decode_layer0_qkvz"]["token_multiple"] == tp
    assert linears["decode_layer0_router_gemm"]["token_split"] == tp
    assert linears["decode_layer0_shared_up"]["token_split"] == tp
    assert all(value["key"]["execution_mode"] == "eager" for value in linears.values())
    collectives = [spec["Hccl"] for spec in specs if "Hccl" in spec]
    assert Counter(value["key"]["operation"] for value in collectives) == {"reduce_scatter": 60, "all_gather": 61}
    assert all(value["key"]["execution_mode"] == "eager" for value in collectives)
    expected = f"qwen35_397b_modelslim_fp32_state_eager_tp{tp}_dp{dp}_ep16/v1"
    assert all(spec["AscendQwen"]["boundary_contract"] == expected for spec in specs if "AscendQwen" in spec)


def test_fp32_runtime_roundtrip_preserves_state_capacity_and_phase_contracts():
    import json
    from types import SimpleNamespace

    from aisimulate_core.sdk.config_builders import build_model_config
    from aisimulate_core.sdk.fpm_config import OpLevelCompileConfig
    from aisimulate_core.sdk.models.ascend_qwen35 import configure_ascend_qwen35

    config = build_model_config(
        tp_size=4,
        pp_size=1,
        attention_dp_size=4,
        moe_tp_size=1,
        moe_ep_size=16,
        gemm_quant_mode="bfloat16",
        moe_quant_mode="w8a8_dynamic",
        kvcache_quant_mode="bfloat16",
        system="ascend910c",
    )
    config.op_level_config = OpLevelCompileConfig(
        {"ascend": {"decode_execution_mode": "eager", "qwen_runtime_contract": "modelslim_fp32_tp_scatter"}},
        system="ascend910c",
        backend="sglang",
    )
    saved = config.op_level_config.cache_identity()
    restored = OpLevelCompileConfig(json.loads(saved), system="ascend910c", backend="sglang")
    assert restored.cache_identity() == saved
    model = SimpleNamespace(config=config, _backend_name="sglang")
    configure_ascend_qwen35(model, raw_config())
    assert model._mamba_ssm_dtype == "float32"
    assert model.execution_assumptions["max_dispatch_tokens_per_rank"] == 16
    assert model.execution_assumptions["qwen_runtime_contract"] == "modelslim_fp32_tp_scatter"
    for operation in model.context_ops + model.generation_ops:
        specification = json.loads(operation._spec_json())
        leaf = specification.get("TokenScale", {}).get("op", specification)
        if "AscendQwen" in leaf:
            assert "fp32_state_eager_tp4_dp4_ep16" in leaf["AscendQwen"]["boundary_contract"]


@pytest.mark.parametrize("tp,dp", [(2, 8), (4, 4), (8, 2)])
@pytest.mark.parametrize("decode_mode", ["eager", "graph"])
def test_fp32_prefill_uses_observed_reduce_scatter_before_moe(tp, dp, decode_mode):
    # Both fresh Prefill forwards contain60 RS and60 layer AG calls. The
    # logits exchange adds one AG. Prefill keeps eager collective execution.
    specs = physical_specs(
        raw_config(),
        tp,
        "prefill",
        attention_dp=dp,
        decode_execution_mode=decode_mode,
        qwen_runtime_contract="modelslim_fp32_tp_scatter",
    )
    collectives = [spec["Hccl"] for spec in specs if "Hccl" in spec]
    assert Counter(value["key"]["operation"] for value in collectives) == {"reduce_scatter": 60, "all_gather": 61}
    assert all(value["key"]["execution_mode"] == "eager" for value in collectives)
    reductions = [value for value in collectives if value["key"]["operation"] == "reduce_scatter"]
    assert all(value["elements_per_token"] == 4096 * tp and value["token_split"] == tp for value in reductions)
    historical = physical_specs(raw_config(), tp, "prefill", attention_dp=dp, decode_execution_mode=decode_mode)
    assert Counter(spec["Hccl"]["key"]["operation"] for spec in historical if "Hccl" in spec) == {
        "all_reduce": 60,
        "all_gather": 61,
    }


@pytest.mark.parametrize("tp,dp", [(2, 8), (4, 4), (8, 2)])
def test_small_graph_roles_require_their_graph_contract(tp, dp):
    # The six replacement roles were independently measured with padded Graph
    # inputs. They cannot consume the historical eager contract in this mode.
    specs = physical_specs(
        raw_config(), tp, "decode", attention_dp=dp, decode_execution_mode="graph", qwen_graph_small_ops=True
    )
    qwen = [s["AscendQwen"] for s in specs if "AscendQwen" in s]
    expected = f"qwen35_397b_modelslim_fp32_state_graph_tp{tp}_dp{dp}_ep16/v1"
    assert all(s["boundary_contract"] == expected for s in qwen)
    assert next(s for s in qwen if s["key"]["role"] == "embedding")["key"]["tp"] == 1
    with pytest.raises(ValueError, match="requires Graph Decode"):
        physical_specs(raw_config(), tp, "decode", attention_dp=dp, qwen_graph_small_ops=True)


@pytest.mark.parametrize("tp,dp", [(2, 8), (4, 4), (8, 2)])
def test_graph_topology_contract_roundtrip(tp, dp):
    import json

    import aisimulate_core._native as core

    specs = physical_specs(raw_config(), tp, "decode", attention_dp=dp, decode_execution_mode="graph")
    for spec in specs:
        assert json.loads(core.op_from_spec_json(json.dumps(spec))._spec_json()) == spec
    qwen = [s["AscendQwen"] for s in specs if "AscendQwen" in s]
    expert = next(s for s in qwen if s["key"]["role"] == "routed_experts")
    assert expert["key"]["tp"] == 16
    assert expert["boundary_contract"].endswith(f"graph_tp{tp}_dp{dp}_ep16/v1")
    linear = next(s["AscendLinear"] for s in specs if s.get("AscendLinear", {}).get("name") == "decode_layer0_qkvz")
    assert linear["key"]["n"] == 20480 // tp
    assert linear["token_multiple"] == tp


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


@pytest.mark.parametrize("tp,dp", [(2, 8), (4, 4), (8, 2)])
def test_fp32_eager_pair_is_one_serial_transport_charge_per_layer(tp, dp):
    import json

    import aisimulate_core._native as core

    specs = physical_specs(
        raw_config(),
        tp,
        "decode",
        attention_dp=dp,
        qwen_runtime_contract="modelslim_fp32_tp_scatter",
        qwen_eager_ep_pair=True,
    )
    custom = [spec["AscendQwen"] for spec in specs if "AscendQwen" in spec]
    roles = Counter(spec["key"]["role"] for spec in custom)
    assert roles["expert_transport_pair"] == roles["routed_experts"] == 60
    assert roles["expert_dispatch"] == roles["expert_combine"] == 0
    pairs = [spec for spec in custom if spec["key"]["role"] == "expert_transport_pair"]
    assert all(spec["key"]["tp"] == 16 for spec in pairs)
    for spec in specs:
        assert json.loads(core.op_from_spec_json(json.dumps(spec))._spec_json()) == spec
    with pytest.raises(ValueError, match="FP32 TP-scatter eager"):
        physical_specs(
            raw_config(), tp, "decode", attention_dp=dp, decode_execution_mode="graph", qwen_eager_ep_pair=True
        )
