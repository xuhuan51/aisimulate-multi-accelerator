import json

import pytest

from aisimulate_core.sdk.config_builders import build_model_config
from aisimulate_core.sdk.engine import _engine_config_dict
from aisimulate_core.sdk.errors import InvalidEngineConfigurationError, PerfDataNotAvailableError
from aisimulate_core.sdk.fpm_config import OpLevelCompileConfig
from aisimulate_core.sdk.models import get_model
from tests.unit.sdk.ascend_fixtures import write_glm52_modelslim

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    return write_glm52_modelslim(tmp_path_factory.mktemp("ascend-forward-model"))


def configuration():
    return build_model_config(
        tp_size=8,
        pp_size=1,
        attention_dp_size=2,
        moe_tp_size=1,
        moe_ep_size=16,
        gemm_quant_mode="w8a8",
        moe_quant_mode="w8a8_dynamic",
        kvcache_quant_mode="bfloat16",
        system="ascend910c",
    )


def flattened(operations):
    output = {}

    def visit(spec):
        kind, value = next(iter(spec.items()))
        if kind == "Overlap":
            for child in value["group_a"] + value["group_b"]:
                visit(child)
        elif kind == "TokenScale":
            visit(value["op"])
        else:
            output[value["name"]] = (kind, value)

    for operation in operations:
        visit(json.loads(operation._spec_json()))
    return output


def test_resident_inventory_matches_loaded_united_parameters(model_path):
    model = get_model(model_path, configuration(), "sglang")
    assert model.get_resident_weights_bytes() == 52_471_997_948
    context, generation = flattened(model.context_ops), flattened(model.generation_ops)
    assert context["context_indexer"][1]["scale_factor"] == 21
    assert context["context_attention"][1]["scale_factor"] == 78
    assert context["context_dense_gate_up"][1]["scale_factor"] == 3
    assert context["context_routed_experts"][1]["scale_factor"] == 75
    assert context["context_rope_cache"][1]["scale_factor"] == 2
    assert context["context_rope_cache"][1]["resident_copies"] == 1
    assert context["context_dense_global_sums"][1]["scale_factor"] == 6
    assert generation["generation_dense_input_gather"][1]["scale_factor"] == 3
    assert generation["generation_dense_output_reduce_scatter"][1]["scale_factor"] == 3
    assert generation["generation_sparse_reduce_scatter"][1]["scale_factor"] == 75
    assert generation["generation_dense_attention_reduce_scatter"][1]["scale_factor"] == 3
    assert context["context_logits_gather"][1]["token_axis"] == "requests"
    assert context["context_logits_finalize"][1]["token_axis"] == "requests"


@pytest.mark.parametrize("attention_tp,attention_dp,expert_ep", [(4, 4, 16), (1, 32, 32)])
def test_observed_prefill_counts_change_compute_shape_without_adding_latency(
    model_path, attention_tp, attention_dp, expert_ep
):
    config = configuration()
    config.tp_size, config.attention_dp_size, config.moe_ep_size = attention_tp, attention_dp, expert_ep
    baseline = get_model(model_path, config, "sglang")
    workload = {
        "observation_id": "synthetic32-token-top8-workload",
        "quantization_sha256": baseline.raw_config["modelslim_metadata"]["source_sha256"],
        "input_tokens": 32,
        "layer_expert_counts": [[32] * 8 + [0] * 248 for _ in range(75)],
    }
    config.op_level_config = OpLevelCompileConfig(
        {"ascend": {"shape_policy": "anchored", "prefill_expert_workload": workload}},
        system="ascend910c",
        backend="sglang",
    )
    observed = get_model(model_path, config, "sglang")
    operations = flattened(observed.context_ops)
    experts = [spec for kind, spec in operations.values() if kind == "AscendMoeChain"]
    assert len(experts) == 75
    for spec in experts:
        assert spec["scale_factor"] == 1
        assert spec["token_numerator"] % attention_dp == 0
        assert spec["token_numerator"] == 8 * attention_dp * spec["token_denominator"]
    assert sum(op.get_weights() for op in observed.context_ops) == sum(op.get_weights() for op in baseline.context_ops)
    assert [op._spec_json() for op in observed.generation_ops] == [op._spec_json() for op in baseline.generation_ops]
    assert observed.execution_assumptions["prefill_expert_workload"] == workload
    transports = [spec for kind, spec in operations.values() if kind == "AscendDeepEp"]
    assert len(transports) == 150
    assert all(spec["scale_factor"] == 1 and spec["key"]["distribution"] == "balanced" for spec in transports)


def test_observed_prefill_workload_rejects_another_quantization(model_path):
    config = configuration()
    config.op_level_config = OpLevelCompileConfig(
        {
            "ascend": {
                "shape_policy": "anchored",
                "prefill_expert_workload": {
                    "observation_id": "wrong-checkpoint",
                    "quantization_sha256": "a" * 64,
                    "input_tokens": 32,
                    "layer_expert_counts": [[1] * 256 for _ in range(75)],
                },
            }
        },
        system="ascend910c",
        backend="sglang",
    )
    with pytest.raises(InvalidEngineConfigurationError, match="observed ModelSlim"):
        get_model(model_path, config, "sglang")


@pytest.mark.parametrize("attention_tp,attention_dp,expert_ep", [(4, 4, 16), (1, 32, 32)])
def test_runtime_layers_replace_timing_and_preserve_resident_weights(model_path, attention_tp, attention_dp, expert_ep):
    config = configuration()
    config.tp_size, config.attention_dp_size, config.moe_ep_size = attention_tp, attention_dp, expert_ep
    settings = {"decode_execution_mode": "eager"}
    config.op_level_config = OpLevelCompileConfig({"ascend": settings}, system="ascend910c", backend="sglang")
    isolated = get_model(model_path, config, "sglang")
    settings = {**settings, "decode_timing_scope": "runtime_layers"}
    config.op_level_config = OpLevelCompileConfig({"ascend": settings}, system="ascend910c", backend="sglang")
    replacement = get_model(model_path, config, "sglang")
    assert [op._spec_json() for op in replacement.context_ops] == [op._spec_json() for op in isolated.context_ops]
    assert sum(op.get_weights() for op in replacement.generation_ops) == sum(
        op.get_weights() for op in isolated.generation_ops
    )
    specs = flattened(replacement.generation_ops)
    assert "generation_qkv_a" not in specs
    assert "generation_attention_seq_lens_h2d" not in specs
    assert "generation_global_logits" not in specs
    assert "generation_logits_gemm" in specs
    kind, layers = specs["generation_decoder_layers"]
    assert kind == "AscendDecoderLayers"
    assert (layers["attention_tp"], layers["attention_dp"], layers["moe_ep"]) == (attention_tp, attention_dp, expert_ep)
    assert layers["covered_ops"]
    assert "generation_eager_runtime_overhead" not in specs


def test_runtime_layer_profile_cannot_migrate_to_unmeasured_topology(model_path):
    config = configuration()
    config.op_level_config = OpLevelCompileConfig(
        {"ascend": {"decode_execution_mode": "eager", "decode_timing_scope": "runtime_layers"}},
        system="ascend910c",
        backend="sglang",
    )
    with pytest.raises(InvalidEngineConfigurationError, match="TP16/ADP4/EP16/DCP1"):
        get_model(model_path, config, "sglang")


def test_decode_padding_preserves_global_dense_and_rank_local_expert_shapes(model_path):
    model = get_model(model_path, configuration(), "sglang")
    decoded = [json.loads(operation._spec_json()) for operation in model.generation_ops]
    dense = next(
        spec["TokenScale"]
        for spec in decoded
        if "TokenScale" in spec
        and spec["TokenScale"]["op"].get("AscendLinear", {}).get("name") == "generation_dense_gate_up"
    )
    assert dense["numerator"] == 2 and dense["denominator"] == 1
    assert dense["op"]["AscendLinear"]["token_multiple"] == 16
    ops = flattened(model.generation_ops)
    assert ops["generation_shared_gate_up"][1]["token_split"] == 8
    assert ops["generation_moe_dispatch"][1]["input_token_multiple"] == 8
    assert ops["generation_indexer_weights"][1]["key"]["quant_mode"] == "float32_bfloat16"
    assert ops["generation_final_norm"][1]["token_split"] == 1
    assert ops["generation_sparse_input_norms"][1]["token_split"] == 8


def test_tp16_matches_observed_single_dp_boundaries_and_graph_padding(model_path):
    config = configuration()
    config.tp_size, config.attention_dp_size = 16, 1
    model = get_model(model_path, config, "sglang")
    assert model.get_resident_weights_bytes() == 51_292_091_900
    for phase, operations in (("context", model.context_ops), ("generation", model.generation_ops)):
        ops = flattened(operations)
        assert ops[phase + "_dense_global_sums"][1]["scale_factor"] == 6
        assert ops[phase + "_dense_post_norm"][1]["key"]["role"] == "add_rms_bias"
        assert phase + "_dense_residual_add" not in ops
        assert phase + "_dense_input_gather" not in ops
        assert ops[phase + "_sparse_reduce_scatter"][1]["scale_factor"] == 75
        assert ops[phase + "_sparse_output_gather"][1]["scale_factor"] == 75
    generated = flattened(model.generation_ops)
    assert generated["generation_attention"][1]["graph_buckets"] == [16, 32]
    assert generated["generation_sparse_output_gather"][1]["elements_per_token"] == 6144
    assert generated["generation_sparse_output_gather"][1]["token_split"] == 16
    assert generated["generation_logits_gemm"][1]["key"]["n"] == 9680
    assert generated["generation_logits_gemm"][1]["token_multiple"] == 16


def test_tensor_moe_matches_loaded_weights_and_observed_all_reduce_boundaries(model_path):
    config = configuration()
    config.tp_size, config.attention_dp_size = 16, 1
    config.moe_tp_size, config.moe_ep_size = 16, 1
    model = get_model(model_path, config, "sglang")
    assert model.get_resident_weights_bytes() == 49_300_283_900
    assert model.execution_assumptions["moe_a2a_backend"] == "none"
    assert model.execution_assumptions["prefill_shared_expert_overlap"] is False
    assert all("Overlap" not in json.loads(operation._spec_json()) for operation in model.context_ops)
    assert any("Overlap" in json.loads(operation._spec_json()) for operation in model.generation_ops)
    for phase, operations in (("context", model.context_ops), ("generation", model.generation_ops)):
        ops = flattened(operations)
        assert ops[phase + "_routed_experts"][0] == "AscendMoeTensor"
        assert ops[phase + "_routed_experts"][1]["key"]["intermediate_size"] == 128
        assert ops[phase + "_shared_gate_up"][1]["key"]["n"] == 256
        assert ops[phase + "_shared_down"][1]["key"]["k"] == 128
        assert ops[phase + "_shared_gate_up"][1]["token_split"] == 1
        assert ops[phase + "_final_norm"][1]["key"]["role"] == "add_rms_bias"
        assert phase + "_final_residual_add" not in ops
        assert all(kind != "AscendDeepEp" for kind, _ in ops.values())
        assert (
            sum(
                spec["scale_factor"]
                for kind, spec in ops.values()
                if kind == "Hccl" and spec["key"]["operation"] == "all_reduce"
            )
            == 157
        )


@pytest.mark.parametrize("local_tp,attention_dp,resident", [(8, 2, 50_480_189_948), (4, 4, 52_840_002_044)])
def test_tensor_moe_with_attention_dp_matches_observed_global_ffn_boundaries(
    model_path, local_tp, attention_dp, resident
):
    config = configuration()
    config.tp_size, config.attention_dp_size = local_tp, attention_dp
    config.moe_tp_size, config.moe_ep_size = 16, 1
    model = get_model(model_path, config, "sglang")
    assert model.get_resident_weights_bytes() == resident
    context, generation = flattened(model.context_ops), flattened(model.generation_ops)
    assert context["context_sparse_global_sums"][1]["scale_factor"] == 150
    assert context["context_sparse_global_sums"][1]["key"]["num_ranks"] == 16
    assert generation["generation_sparse_attention_reduce_scatter"][1]["key"]["num_ranks"] == local_tp
    assert generation["generation_sparse_input_gather"][1]["key"]["num_ranks"] == 16
    assert generation["generation_sparse_output_reduce_scatter"][1]["key"]["num_ranks"] == 16
    assert generation["generation_sparse_output_gather"][1]["key"]["num_ranks"] == local_tp
    assert generation["generation_sparse_post_norm"][1]["key"]["role"] == "rms_bias"
    assert generation["generation_routed_experts"][1]["input_token_multiple"] == 8 * attention_dp
    assert generation["generation_shared_gate_up"][1]["token_multiple"] == 8 * attention_dp
    assert generation["generation_shared_gate_up"][1]["key"]["n"] == 256
    assert generation["generation_shared_down"][1]["key"]["k"] == 128


@pytest.mark.parametrize("local_tp,attention_dp", [(4, 8), (8, 4), (16, 2)])
def test_tensor_moe_global32_uses_measured_64_wide_shards(model_path, local_tp, attention_dp):
    config = configuration()
    config.tp_size, config.attention_dp_size = local_tp, attention_dp
    config.moe_tp_size, config.moe_ep_size = 32, 1
    model = get_model(model_path, config, "sglang")
    ops = flattened(model.generation_ops)
    assert ops["generation_routed_experts"][1]["key"]["intermediate_size"] == 64
    assert ops["generation_shared_down"][1]["key"]["k"] == 64
    assert ops["generation_sparse_input_gather"][1]["key"]["num_ranks"] == 32
    assert ops["generation_sparse_output_reduce_scatter"][1]["key"]["num_ranks"] == 32


def test_dcp_composes_storage_once_and_matches_observed_split_kv_pool(model_path):
    config = configuration()
    config.dcp_size = 2
    model = get_model(model_path, config, "sglang")
    for operations in (model.context_ops, model.generation_ops):
        entries = flattened(operations)
        attention = [spec for kind, spec in entries.values() if kind == "AscendDcpAttention"]
        assert {(item["key"]["remap_topk"], item["scale_factor"]) for item in attention} == {
            (True, 21.0),
            (False, 57.0),
        }
        assert not any(
            kind == "AscendVector" and spec["key"]["role"] == "cache_scatter" for kind, spec in entries.values()
        )
        assert all(
            spec["key"]["operation"] == "lightning_indexer"
            for kind, spec in entries.values()
            if kind == "AscendAttention"
        )
    assert model.get_resident_weights_bytes() == 52_471_997_948
    assert model.get_kvcache_rank_bytes_per_sequence(1) == 64_896
    assert model.get_kvcache_pool_overhead_bytes() == 16_613_376
    assert model.get_kvcache_rank_bytes_per_sequence(49_664) + model.get_kvcache_pool_overhead_bytes() == 3_239_608_320
    assert model.get_kvcache_rank_batch_capacity(3_239_608_320, 32) == 49_664
    assert model.get_kvcache_rank_batch_capacity(3_239_608_319, 32) == 49_408


def test_attention_tp1_has_no_local_collective_and_keeps_terminal_residual_for_norm(model_path):
    config = configuration()
    config.tp_size, config.attention_dp_size, config.moe_ep_size = 1, 32, 32
    model = get_model(model_path, config, "sglang")
    for phase, operations in (("context", model.context_ops), ("generation", model.generation_ops)):
        entries = flattened(operations)
        assert all(spec["key"]["num_ranks"] > 1 for kind, spec in entries.values() if kind == "Hccl")
        assert phase + "_embedding_all_reduce" not in entries
        assert phase + "_sparse_output_gather" not in entries
        assert phase + "_logits_gather" not in entries
        assert phase + "_final_residual_add" not in entries
        assert entries[phase + "_final_norm"][1]["key"]["role"] == "add_rms_bias"
        assert entries[phase + "_q_b"][1]["key"]["n"] == 16384
        assert entries[phase + "_logits_gemm"][1]["key"]["n"] == 154880


def test_explicit_serial_shared_experts_survive_saved_engine_identity(model_path):
    config = configuration()
    config.op_level_config = OpLevelCompileConfig(
        {"ascend": {"shared_expert_overlap": False}}, system="ascend910c", backend="sglang"
    )
    model = get_model(model_path, config, "sglang")
    assert all("Overlap" not in json.loads(operation._spec_json()) for operation in model.generation_ops)
    assert model.get_resident_weights_bytes() == 52_471_997_948
    engine = _engine_config_dict(
        model=model,
        model_path=model_path,
        system="ascend910c",
        backend="sglang",
        backend_version="0.5.22",
        kv_block_size=128,
        systems_path=None,
        nextn=0,
    )
    saved = json.loads(engine["extra"]["estimator_config"])
    assert saved["op_level"]["ascend"] == {
        "shared_expert_overlap": False,
        "decode_execution_mode": "graph",
        "decode_timing_scope": "isolated_kernels",
        "enable_dp_lm_head": True,
        "enable_tp_lm_head_all_to_all": False,
        "routing": "balanced",
        "kv_transfer_page_layout": "contiguous",
    }


def test_eager_decode_uses_eager_profiles_and_preserves_partitioning(model_path):
    config = configuration()
    config.op_level_config = OpLevelCompileConfig(
        {"ascend": {"decode_execution_mode": "eager"}}, system="ascend910c", backend="sglang"
    )
    model = get_model(model_path, config, "sglang")
    baseline = get_model(model_path, configuration(), "sglang")
    assert flattened(model.context_ops) == flattened(baseline.context_ops)
    assert model.execution_assumptions["decode_execution_mode"] == "eager"
    assert model.execution_assumptions["decode_graph_buckets"] == []
    ops = flattened(model.generation_ops)
    assert all(spec["key"]["execution_mode"] == "eager" for _, spec in ops.values())
    assert ops["generation_attention"][1]["graph_buckets"] == []
    assert ops["generation_attention"][1]["eager_token_multiple"] == 8
    assert ops["generation_qkv_a"][1]["token_multiple"] == 8
    assert ops["generation_dense_gate_up"][1]["token_multiple"] == 16
    assert ops["generation_moe_dispatch"][1]["input_token_multiple"] == 8
    assert ops["generation_shared_gate_up"][1]["token_split"] == 8
    assert ops["generation_dense_input_gather"][1]["elements_per_token"] == 6144
    assert ops["generation_dense_input_gather"][1]["token_split"] == 8
    assert model.get_resident_weights_bytes() == baseline.get_resident_weights_bytes()


def test_sequence_lengths_follow_prefill_requests_and_padded_decode_rows(model_path):
    config = configuration()
    config.op_level_config = OpLevelCompileConfig(
        {"ascend": {"decode_execution_mode": "eager"}}, system="ascend910c", backend="sglang"
    )
    model = get_model(model_path, config, "sglang")
    context, generation = flattened(model.context_ops), flattened(model.generation_ops)
    for site, layers in (("attention", 78), ("indexer", 21)):
        prefill = context[f"context_{site}_seq_lens_h2d"][1]
        decode = generation[f"generation_{site}_seq_lens_h2d"][1]
        assert prefill["token_axis"] == "requests" and prefill["token_multiple"] == 1
        assert decode["token_axis"] == "tokens" and decode["token_multiple"] == 8
        assert prefill["scale_factor"] == decode["scale_factor"] == layers
    graph = get_model(model_path, configuration(), "sglang")
    assert not any("seq_lens_h2d" in name for name in flattened(graph.generation_ops))


@pytest.mark.parametrize("global_tp,attention_dp", [(16, 4), (32, 32)])
def test_global_logits_retains_runtime_partition_and_canonical_options(model_path, global_tp, attention_dp):
    config = configuration()
    config.tp_size = global_tp // attention_dp
    config.attention_dp_size = attention_dp
    config.moe_ep_size = global_tp
    config.op_level_config = OpLevelCompileConfig(
        {
            "ascend": {
                "decode_execution_mode": "eager",
                "enable_dp_lm_head": False,
                "enable_tp_lm_head_all_to_all": global_tp == 32,
            }
        },
        system="ascend910c",
        backend="sglang",
    )
    model = get_model(model_path, config, "sglang")
    entries = flattened(model.generation_ops)
    kind, logits = entries["generation_global_logits"]
    assert kind == "AscendLogits"
    assert logits["global_tp"] == global_tp
    assert logits["attention_dp"] == attention_dp
    assert logits["request_counts"] is None
    assert logits["vocabulary"] // logits["global_tp"] == (9680 if global_tp == 16 else 4840)
    assert "generation_logits_gemm" not in entries
    assert "generation_logits_gather" not in entries
    assert model.execution_assumptions["enable_dp_lm_head"] is False
    engine = _engine_config_dict(
        model=model,
        model_path=model_path,
        system="ascend910c",
        backend="sglang",
        backend_version="0.5.22",
        kv_block_size=128,
        systems_path=None,
        nextn=0,
    )
    restored = json.loads(engine["extra"]["estimator_config"])["op_level"]["ascend"]
    assert restored["enable_dp_lm_head"] is False
    assert restored["enable_tp_lm_head_all_to_all"] == (global_tp == 32)


def test_eager_decode_rejects_unmeasured_dcp_profiles(model_path):
    config = configuration()
    config.dcp_size = 2
    config.op_level_config = OpLevelCompileConfig(
        {"ascend": {"decode_execution_mode": "eager"}}, system="ascend910c", backend="sglang"
    )
    with pytest.raises(InvalidEngineConfigurationError, match="eager DCP profiles"):
        get_model(model_path, config, "sglang")


@pytest.mark.parametrize(
    "local_tp,attention_dp,tensor_moe,alignment",
    [(2, 8, False, 2), (4, 4, False, 4), (8, 2, True, 8), (16, 1, False, 16), (16, 1, True, 1)],
)
def test_eager_alignment_follows_mlp_sync_requirement(model_path, local_tp, attention_dp, tensor_moe, alignment):
    config = configuration()
    config.tp_size, config.attention_dp_size = local_tp, attention_dp
    config.moe_tp_size, config.moe_ep_size = (16, 1) if tensor_moe else (1, 16)
    config.op_level_config = OpLevelCompileConfig(
        {"ascend": {"decode_execution_mode": "eager"}}, system="ascend910c", backend="sglang"
    )
    model = get_model(model_path, config, "sglang")
    ops = flattened(model.generation_ops)
    assert model.execution_assumptions["decode_eager_token_multiple"] == alignment
    assert ops["generation_attention"][1]["eager_token_multiple"] == alignment
    assert ops["generation_qkv_a"][1]["token_multiple"] == alignment
    assert ops["generation_attention"][1]["graph_buckets"] == []
    assert ops["generation_logits_gemm"][1]["token_multiple"] == 1
    assert ops["generation_logits_finalize"][1]["token_multiple"] == 1


def test_anchored_shape_policy_survives_model_and_engine_identity(model_path):
    config = configuration()
    config.op_level_config = OpLevelCompileConfig(
        {"ascend": {"shape_policy": "anchored"}}, system="ascend910c", backend="sglang"
    )
    model = get_model(model_path, config, "sglang")
    engine = _engine_config_dict(
        model=model,
        model_path=model_path,
        system="ascend910c",
        backend="sglang",
        backend_version="0.5.22",
        kv_block_size=128,
        systems_path=None,
        nextn=0,
    )
    saved = json.loads(engine["extra"]["estimator_config"])
    assert saved["op_level"]["ascend"]["shape_policy"] == "anchored"
    bounded = configuration()
    assert get_model(model_path, bounded, "sglang").get_resident_weights_bytes() == model.get_resident_weights_bytes()


def test_model_name_and_w8a8_label_cannot_replace_checkpoint_metadata():
    with pytest.raises(InvalidEngineConfigurationError, match="quant_model_description"):
        get_model("zai-org/GLM-5.2", configuration(), "sglang")


def test_default_search_requires_the_audited_checkpoint(model_path):
    from aisimulate_core.sdk.models.ascend_glm52 import ascend_glm52_default_parallel_shapes
    from aisimulate_core.sdk.utils import get_model_config_from_model_path

    raw = get_model_config_from_model_path(model_path)["raw_config"]
    assert ascend_glm52_default_parallel_shapes(raw) == (
        (1, 16, 1, 16, 1),
        (2, 8, 1, 16, 1),
        (4, 4, 1, 16, 1),
        (8, 2, 1, 16, 1),
        (16, 1, 1, 16, 1),
        (1, 32, 1, 32, 1),
        (2, 16, 1, 32, 1),
        (4, 8, 1, 32, 1),
        (8, 4, 1, 32, 1),
        (16, 2, 1, 32, 1),
        (1, 16, 16, 1, 1),
        (2, 8, 16, 1, 1),
        (4, 4, 16, 1, 1),
        (8, 2, 16, 1, 1),
        (16, 1, 16, 1, 1),
        (1, 32, 32, 1, 1),
        (2, 16, 32, 1, 1),
        (4, 8, 32, 1, 1),
        (8, 4, 32, 1, 1),
        (16, 2, 32, 1, 1),
    )
    with pytest.raises(InvalidEngineConfigurationError, match="quant_model_description"):
        ascend_glm52_default_parallel_shapes({**raw, "modelslim_metadata": None})


def transfer_model(model_path, role, **overrides):
    from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

    return RustForwardPassPerfModel.best_available(
        {
            "model": model_path,
            "system": "ascend910c",
            "backend": "sglang",
            "backend_version": "0.5.22",
            "worker_type": role,
            "tp": 8,
            "attention_dp": 2,
            "moe_tp_size": 1,
            "moe_ep_size": 16,
            "kv_block_size": 128,
            "gemm_quant_mode": "w8a8",
            "moe_quant_mode": "w8a8_dynamic",
            "kvcache_quant_mode": "bfloat16",
            "fmha_quant_mode": "bfloat16",
            "estimation_mode": "op_level",
            **overrides,
        }
    )


@pytest.mark.parametrize("execution_mode", ["eager", "graph"])
def test_synchronized_dp_estimator_preserves_uniform_cost_and_rank_permutation(model_path, execution_mode):
    model = transfer_model(
        model_path,
        "decode",
        estimator_config={
            "op_level": {"ascend": {"decode_execution_mode": execution_mode, "shape_policy": "anchored"}}
        },
    )

    def metrics(count):
        return {"scheduled_requests": {"num_decode_requests": count, "sum_decode_kv_tokens": count * 8193}}

    uniform = model.estimate_forward_pass_time_ms([metrics(2), metrics(2)])
    assert uniform == pytest.approx(model.estimate_forward_pass_time_ms(metrics(2)), rel=1e-12)
    for counts in ((1, 9), (0, 9), (0, 0)):
        duration = model.estimate_forward_pass_time_ms([metrics(count) for count in counts])
        assert duration == model.estimate_forward_pass_time_ms([metrics(count) for count in reversed(counts)])
        assert duration >= 0
    with pytest.raises(ValueError, match="all 2 rank metrics"):
        model.estimate_forward_pass_time_ms([metrics(1)] * 3)


def test_synchronized_sum_len_executes_with_idle_and_mixed_ranks(model_path):
    model = transfer_model(
        model_path,
        "decode",
        tp=4,
        attention_dp=4,
        estimator_config={"op_level": {"ascend": {"decode_execution_mode": "eager", "shape_policy": "anchored"}}},
    )
    active = {"scheduled_requests": {"num_decode_requests": 1, "sum_decode_kv_tokens": 32769}}
    idle = {"scheduled_requests": {}}
    duration = model.estimate_forward_pass_time_ms([active, idle, idle, idle])
    assert duration > 0
    assert duration == model.estimate_forward_pass_time_ms([idle, idle, idle, active])
    prefill = {"scheduled_requests": {"num_prefill_requests": 1, "sum_prefill_tokens": 4096}}
    mixed = model.estimate_forward_pass_time_ms([active, prefill, idle, idle])
    assert mixed > duration
    assert mixed == model.estimate_forward_pass_time_ms([idle, idle, prefill, active])
    within_rank = {"scheduled_requests": {**active["scheduled_requests"], **prefill["scheduled_requests"]}}
    with pytest.raises(ValueError, match="within the same rank"):
        model.estimate_forward_pass_time_ms([within_rank, idle, idle, idle])


def test_mixed_ranks_expose_estimated_native_operation_breakdown(model_path):
    ranks = [
        {"scheduled_requests": {"num_prefill_requests": 1, "sum_prefill_tokens": 4096, "sum_prefill_kv_tokens": 4096}},
        {"scheduled_requests": {"num_decode_requests": 1, "sum_decode_kv_tokens": 8193}},
        {"scheduled_requests": {}},
        {"scheduled_requests": {}},
    ]
    model = transfer_model(
        model_path,
        "aggregated",
        tp=4,
        attention_dp=4,
        estimator_config={"op_level": {"ascend": {"decode_execution_mode": "eager", "shape_policy": "anchored"}}},
    )
    duration = model.estimate_forward_pass_time_ms(ranks)
    operations = model.estimate_ascend_dp_operations(ranks)
    assert sum(entry[1] for entry in operations) == pytest.approx(duration, rel=1e-12)
    assert all(entry[3] == "estimated" for entry in operations)
    assert duration > 0
    strict = transfer_model(
        model_path,
        "aggregated",
        tp=4,
        attention_dp=4,
        estimator_config={"op_level": {"ascend": {"decode_execution_mode": "eager", "shape_policy": "measured"}}},
    )
    with pytest.raises(ValueError, match="mixed timing requires shape_policy=anchored"):
        strict.estimate_forward_pass_time_ms(ranks)


def test_eager_runtime_overhead_is_once_per_forward_with_explicit_topology(model_path):
    overhead = {
        "calibration_id": "synthetic-7.5ms-forward-residual",
        "operator_contract": "glm52_eager_metadata/v2",
        "attention_tp": 4,
        "attention_dp": 4,
        "moe_ep_size": 16,
        "prefill_ms": 0.0,
        "decode_ms": 7.5,
    }
    options = {"decode_execution_mode": "eager", "shape_policy": "anchored", "enable_dp_lm_head": False}
    baseline = transfer_model(
        model_path, "decode", tp=4, attention_dp=4, estimator_config={"op_level": {"ascend": options}}
    )
    calibrated = transfer_model(
        model_path,
        "decode",
        tp=4,
        attention_dp=4,
        estimator_config={"op_level": {"ascend": {**options, "eager_runtime_overhead": overhead}}},
    )
    for counts in [(1, 0, 0, 0), (2, 2, 2, 2), (9, 1, 0, 0)]:
        metrics = [
            {"scheduled_requests": {"num_decode_requests": count, "sum_decode_kv_tokens": count * 8193}}
            for count in counts
        ]
        assert calibrated.estimate_forward_pass_time_ms(metrics) - baseline.estimate_forward_pass_time_ms(
            metrics
        ) == pytest.approx(7.5)
    assert calibrated.estimate_forward_pass_time_ms([{"scheduled_requests": {}}] * 4) == 0
    diagnostics = calibrated.static_phase_diagnostics(batch_size=1, context_length=8192, prefill=False)
    contribution = next(row for row in diagnostics if row["name"] == "generation_eager_runtime_overhead")
    assert contribution["latency_ms"] == 7.5
    assert contribution["source"] == "estimated"
    for changed in (
        {"attention_tp": 8},
        {"decode_ms": -1.0},
        {"decode_per_extra_active_rank_ms": -1.0},
        {"calibration_id": ""},
        {"operator_contract": ""},
        {"operator_contract": "glm52_eager_metadata/v1"},
    ):
        with pytest.raises(ValueError, match="overhead"):
            transfer_model(
                model_path,
                "decode",
                tp=4,
                attention_dp=4,
                estimator_config={
                    "op_level": {"ascend": {**options, "eager_runtime_overhead": {**overhead, **changed}}}
                },
            )
    with pytest.raises(ValueError, match="overhead"):
        transfer_model(
            model_path,
            "decode",
            tp=4,
            attention_dp=4,
            estimator_config={
                "op_level": {
                    "ascend": {**options, "decode_execution_mode": "graph", "eager_runtime_overhead": overhead}
                }
            },
        )


def test_eager_extra_active_rank_overhead_ignores_padding_and_request_count(model_path):
    options = {"decode_execution_mode": "eager", "shape_policy": "anchored", "enable_dp_lm_head": False}
    profile = {
        "calibration_id": "hand-derived-base-2.5-extra-rank-0.75",
        "operator_contract": "glm52_eager_metadata/v2",
        "attention_tp": 4,
        "attention_dp": 4,
        "moe_ep_size": 16,
        "prefill_ms": 0.0,
        "decode_ms": 2.5,
        "decode_per_extra_active_rank_ms": 0.75,
    }
    baseline = transfer_model(
        model_path, "decode", tp=4, attention_dp=4, estimator_config={"op_level": {"ascend": options}}
    )
    calibrated = transfer_model(
        model_path,
        "decode",
        tp=4,
        attention_dp=4,
        estimator_config={"op_level": {"ascend": {**options, "eager_runtime_overhead": profile}}},
    )
    for counts, expected in [((1, 0, 0, 0), 2.5), ((2, 2, 0, 0), 3.25), ((9, 1, 0, 0), 3.25), ((1, 1, 1, 1), 4.75)]:
        metrics = [
            {"scheduled_requests": {"num_decode_requests": count, "sum_decode_kv_tokens": count * 8193}}
            for count in counts
        ]
        assert calibrated.estimate_forward_pass_time_ms(metrics) - baseline.estimate_forward_pass_time_ms(
            metrics
        ) == pytest.approx(expected)
    assert calibrated.estimate_forward_pass_time_ms([{"scheduled_requests": {}}] * 4) == 0


def test_pd_group_timing_is_independent_of_diagnostic_capture(model_path):
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.config.cli import CorePredictionConfig
    from aisimulate.runner import EngineReplayRunnerFactory
    from aisimulate.sweeper import ReplayOutputRequirements

    public = CorePredictionConfig.model_validate(
        {
            "traffic": {
                "source": {"type": "synthetic", "input_tokens": 8192, "output_tokens": 4},
                "load": {"type": "concurrency", "concurrency": 2},
                "stop": {"requests": 2},
            },
            "engine": {
                "model": model_path,
                "hardware": "ascend910c",
                "backend": "sglang",
                "backend_version": "0.5.22",
                "mode": "disaggregated",
                "context_length": 16384,
                "gemm_quant_mode": "w8a8",
                "moe_quant_mode": "w8a8_dynamic",
                "fmha_quant_mode": "bfloat16",
                "kvcache_quant_mode": "bfloat16",
                "estimation_mode": "op_level",
                "kv_transfer": {"type": "measured"},
                "estimator_config": {
                    "op_level": {"ascend": {"decode_execution_mode": "eager", "shape_policy": "anchored"}}
                },
                "workers": {
                    role: {
                        "parallelism": {"tensor": 8, "attention_data": 2, "moe_tensor": 1, "moe_expert": 16},
                        "scheduler": {"ignore_eos": True, "chunked_prefill_size": 16384, "max_prefill_tokens": 32768},
                        "kv_cache": {
                            "block_size": 128,
                            "prefix_caching": False,
                            "capacity": {"type": "fixed", "blocks": 512},
                        },
                    }
                    for role in ("prefill", "decode")
                },
            },
        }
    )
    reports = [
        EngineReplayRunnerFactory()
        .create(0)
        .run(
            prediction_to_replay_spec(public),
            output_requirements=ReplayOutputRequirements(capture_performance_diagnostics=capture),
        )
        for capture in (False, True)
    ]
    for field in ("completed_requests", "mean_ttft_ms", "mean_tpot_ms", "output_throughput_tok_s"):
        assert reports[0].metrics[field] == pytest.approx(reports[1].metrics[field], rel=1e-12)
    assert reports[0].metrics["completed_requests"] == 2
    assert "native_report" not in reports[0].metadata
    batches = reports[1].metadata["native_report"]["forward_pass_batches"]
    decode_groups = [row for row in batches if row["stage"] == "decode" and row["dp_rank"] is None]
    decode_ranks = [row for row in batches if row["stage"] == "decode" and row["dp_rank"] is not None]
    assert sum(row["request_instances"] for row in decode_groups) == 6
    assert sum(row["request_instances"] for row in decode_ranks) == 6
    assert all(row["phase"] == "decode" for row in decode_groups)
    assert all(sum(row["batch_histogram"].values()) == row["passes"] for row in batches)
    assert all(row["max_batch_size"] <= 2 for row in batches)


def test_aggregated_replay_prices_the_actual_active_dp_population(model_path):
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.config.cli import CorePredictionConfig
    from aisimulate.runner import EngineReplayRunnerFactory

    estimator = {"op_level": {"ascend": {"decode_execution_mode": "eager", "shape_policy": "anchored"}}}
    public = CorePredictionConfig.model_validate(
        {
            "traffic": {
                "source": {"type": "synthetic", "input_tokens": 8192, "output_tokens": 4},
                "load": {"type": "concurrency", "concurrency": 1},
                "stop": {"requests": 1},
            },
            "engine": {
                "model": model_path,
                "hardware": "ascend910c",
                "backend": "sglang",
                "backend_version": "0.5.22",
                "mode": "aggregated",
                "context_length": 16384,
                "gemm_quant_mode": "w8a8",
                "moe_quant_mode": "w8a8_dynamic",
                "fmha_quant_mode": "bfloat16",
                "kvcache_quant_mode": "bfloat16",
                "estimation_mode": "op_level",
                "estimator_config": estimator,
                "workers": {
                    "aggregated": {
                        "parallelism": {"tensor": 8, "attention_data": 2, "moe_tensor": 1, "moe_expert": 16},
                        "scheduler": {"ignore_eos": True, "chunked_prefill_size": 16384, "max_prefill_tokens": 32768},
                        "kv_cache": {
                            "block_size": 128,
                            "prefix_caching": False,
                            "capacity": {"type": "fixed", "blocks": 512},
                        },
                    }
                },
            },
        }
    )
    oracle = transfer_model(model_path, "aggregated", estimator_config=estimator)
    prefill = [
        {"scheduled_requests": {"num_prefill_requests": 1, "sum_prefill_tokens": 8192}},
        {"scheduled_requests": {}},
    ]
    expected_ttft = oracle.estimate_forward_pass_time_ms(prefill)
    expected_tpot = (
        sum(
            oracle.estimate_forward_pass_time_ms(
                [
                    {"scheduled_requests": {"num_decode_requests": 1, "sum_decode_kv_tokens": length}},
                    {"scheduled_requests": {}},
                ]
            )
            for length in (8193, 8194, 8195)
        )
        / 3
    )
    report = EngineReplayRunnerFactory().create(0).run(prediction_to_replay_spec(public))
    assert report.metrics["completed_requests"] == 1
    assert report.metrics["mean_ttft_ms"] == pytest.approx(expected_ttft, abs=1e-6)
    assert report.metrics["mean_tpot_ms"] == pytest.approx(expected_tpot, abs=1e-6)


def test_tensor_moe_routing_profile_reaches_canonical_model_and_preserves_identity(model_path):
    from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

    # Hand-written calibration. All real tokens choose experts 0..7,
    # padding chooses 8..15. The measured grid is 1 ms at 8 experts,
    # 3 ms at 16 experts. This tests routing/count semantics, not accuracy.
    layer = {"expert_probabilities": [1.0] * 8 + [0.0] * 248, "padding_experts": list(range(8, 16))}
    profile = {
        "model": model_path,
        "backend_version": "0.5.22",
        "evidence_sha256": "a" * 64,
        "intermediate_size": 128,
        "layers": [layer] * 75,
        "costs": [
            {"physical_tokens": 16, "active_experts": 8, "latency_ms": 1.0},
            {"physical_tokens": 16, "active_experts": 16, "latency_ms": 3.0},
        ],
    }
    options = {"op_level": {"ascend": {"tensor_moe_routing_profile": profile}}}
    model = transfer_model(model_path, "aggregated", moe_tp_size=16, moe_ep_size=1, estimator_config=options)
    saved = json.loads(json.dumps(model.diagnostics()["provenance"]["config"]))
    assert saved["estimator_config"]["op_level"]["ascend"]["tensor_moe_routing_profile"] == profile
    restored = RustForwardPassPerfModel.best_available(saved)
    metrics = {"scheduled_requests": {"num_decode_requests": 3, "sum_decode_kv_tokens": 3 * 8192}}
    assert model.estimate_forward_pass_time_ms(metrics) == restored.estimate_forward_pass_time_ms(metrics)
    assert model.estimate_forward_pass_time_ms(metrics) >= 75 * 3.0
    for changes in ({"backend_version": "other"}, {"moe_tp_size": 1, "moe_ep_size": 16}, {"model": "other"}):
        with pytest.raises(ValueError, match="routing profile requires"):
            RustForwardPassPerfModel.normalize_config({**saved, **changes})

    config = configuration()
    config.moe_tp_size, config.moe_ep_size = 16, 1
    config.op_level_config = OpLevelCompileConfig(options["op_level"], system="ascend910c", backend="sglang")
    composed = get_model(model_path, config, "sglang")
    assert flattened(composed.context_ops)["context_routed_experts"][1]["routing_profile"] is None
    assert flattened(composed.generation_ops)["generation_routed_experts"][1]["routing_profile"] == profile


def test_canonical_pd_model_selects_measured_mapping_and_rejects_unknown_geometry(model_path):
    source = transfer_model(model_path, "prefill")
    target = transfer_model(model_path, "decode")
    metadata = source.kv_transfer_metadata(target)
    assert metadata["key"] == {"source_tp": 8, "destination_tp": 8, "page_layout": "contiguous"}
    assert metadata["bytes_per_token_per_replica"] == 109824
    assert source.estimate_kv_transfer_time_ms(target, 0) == 0
    latency = source.estimate_kv_transfer_time_ms(target, 8192)
    assert 7 < latency < 9
    assert source.estimate_kv_transfer_time_ms(target, 8191) == latency
    with pytest.raises(PerfDataNotAvailableError, match="outside measured"):
        source.estimate_kv_transfer_time_ms(target, 32768)
    with pytest.raises(ValueError, match="matching Ascend"):
        target.estimate_kv_transfer_time_ms(source, 8192)
    multi_node = transfer_model(model_path, "decode", attention_dp=4, moe_ep_size=32)
    assert source.estimate_kv_transfer_time_ms(multi_node, 8192) == latency
    assert source.kv_transfer_metadata(multi_node)["key"] == metadata["key"]
    fragmented = transfer_model(
        model_path,
        "decode",
        estimator_config={"op_level": {"ascend": {"kv_transfer_page_layout": "fragmented"}}},
    )
    with pytest.raises(ValueError, match="matching role page-layout"):
        source.estimate_kv_transfer_time_ms(fragmented, 8192)


def test_canonical_anchored_pd_preserves_measured_values_and_requires_matching_policy(model_path):
    options = {"op_level": {"ascend": {"shape_policy": "anchored"}}}
    source = transfer_model(model_path, "prefill", estimator_config=options)
    target = transfer_model(model_path, "decode", estimator_config=options)
    bounded_source = transfer_model(model_path, "prefill")
    bounded_target = transfer_model(model_path, "decode")
    assert source.estimate_kv_transfer_time_ms(target, 8192) == bounded_source.estimate_kv_transfer_time_ms(
        bounded_target, 8192
    )
    assert source.estimate_kv_transfer_time_ms(target, 65536) > source.estimate_kv_transfer_time_ms(target, 32768) > 0
    assert source.kv_transfer_metadata(target)["shape_policy"] == "anchored"
    with pytest.raises(ValueError, match="shape-policy"):
        source.estimate_kv_transfer_time_ms(bounded_target, 65536)
    with pytest.raises(PerfDataNotAvailableError):
        bounded_source.estimate_kv_transfer_time_ms(bounded_target, 65536)
