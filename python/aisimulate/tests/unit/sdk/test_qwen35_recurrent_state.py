# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical Qwen3.5 state geometry for SGLang PD, with no device allocation."""

import pytest

from aisimulate_core.sdk import models
from aisimulate_core.sdk.memory.state import estimate_state_cache


@pytest.fixture
def qwen_info(monkeypatch):
    raw = {
        "text_config": {
            "num_hidden_layers": 60,
            "layer_types": ["linear_attention"] * 45 + ["full_attention"] * 15,
            "dtype": "bfloat16",
            "mamba_ssm_dtype": "float32",
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 64,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_key_value_heads": 2,
            "head_dim": 256,
        }
    }
    monkeypatch.setattr(
        models,
        "_get_model_info",
        lambda _: {
            "architecture": "Qwen3_5MoeForConditionalGeneration",
            "raw_config": raw,
        },
    )
    return raw


@pytest.mark.parametrize("tp,expected", [(2, 96030720), (4, 48015360), (8, 24007680)])
def test_qwen_state_size_tracks_tp_without_page_padding(qwen_info, tp, expected):
    result = estimate_state_cache("local", backend="sglang", tp_size=tp)
    assert result["bytes_per_request"] == expected
    assert result["ssm_bytes_per_request"] == 188743680 // tp
    assert result["conv_bytes_per_request"] == 3317760 // tp
    assert result["kv_bytes_per_token"] == 15360
    assert result["allocation"] == "separate_request_slots"
    assert result["padding_bytes_per_request"] == 0
    assert result["conv_shape_per_layer"][1] == 3


def test_qwen_state_precision_and_gqa_replication(qwen_info):
    qwen_info["text_config"]["mamba_ssm_dtype"] = "bfloat16"
    result = estimate_state_cache("local", backend="sglang", tp_size=1)
    assert result["kv_bytes_per_token"] == 30720
    assert result["bytes_per_request"] == 188743680 // 2 + 3317760


@pytest.mark.parametrize(
    "overrides",
    [
        {"backend": "vllm"},
        {"tp_size": 3},
        {"pp_size": True},
        {"num_speculative_tokens": 1},
        {"kvcache_quant_mode": "fp8"},
        {"kv_bytes_per_token": 109824},
    ],
)
def test_qwen_state_rejects_incompatible_geometry(qwen_info, overrides):
    settings = {"backend": "sglang", "tp_size": 4, **overrides}
    with pytest.raises(ValueError):
        estimate_state_cache("local", **settings)


def test_public_prediction_preserves_separate_state_pool():
    from aisimulate.compiler import _worker_engine_args
    from aisimulate.config.engine import EnginePredictionConfig

    config = EnginePredictionConfig.model_validate(
        {
            "model": "local",
            "hardware": "ascend910c",
            "backend": "sglang",
            "context_length": 10240,
            "workers": {
                "aggregated": {
                    "kv_cache": {
                        "block_size": 128,
                        "prefix_caching": False,
                        "capacity": {"type": "fixed", "blocks": 256},
                        "recurrent_state": {"bytes_per_request": 48015360, "max_requests": 32},
                    },
                    "timing": {"type": "fixed", "prefill_ms": 1, "decode_ms": 1},
                }
            },
        }
    )
    args = _worker_engine_args(config, config.workers.aggregated, "aggregated", transfer_bytes_per_token=None)
    assert args["num_gpu_blocks"] == 256
    assert args["sglang"]["recurrent_state"] == {"bytes_per_request": 48015360, "max_requests": 32}
    raw = config.model_dump(mode="json", exclude_none=True)
    raw["workers"]["aggregated"]["kv_cache"]["prefix_caching"] = True
    with pytest.raises(ValueError, match="prefix snapshots"):
        EnginePredictionConfig.model_validate(raw)
