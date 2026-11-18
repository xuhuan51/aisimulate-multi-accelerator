# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve hybrid token/state cache geometry without loading accelerator weights."""

from __future__ import annotations

from typing import Any


def estimate_state_cache(
    model_path: str,
    *,
    backend: str = "vllm",
    tp_size: int = 1,
    pp_size: int = 1,
    block_size: int | None = None,
    kv_bytes_per_token: int | None = None,
    kvcache_quant_mode: str | None = None,
    mamba_cache_dtype: str = "auto",
    indexer_cache_dtype: str = "auto",
    num_speculative_tokens: int = 0,
) -> dict[str, Any]:
    """Resolve the physical block size and ONE state copy per rank.

    Qwen3.5 with SGLang: the returned state belongs to a separate request-slot
    pool. Token pages keep their requested granularity. BF16 convolution and
    configured temporal-state dtype determine the state bytes. Supply matching
    runtime dtype overrides through the model metadata.

    DeepSeek V4: see ``deepseek_v4``. Block size and token bytes follow
    vLLM's allocation; ``indexer_cache_dtype`` selects its FP8 (``auto``) or
    MXFP4 indexer, and draft tokens widen each sliding window.

    Kimi K3: see ``kimi_k3``. Uses the existing Kimi model for target-only
    token KV geometry. Draft KV and extra state copies belong to their runtime
    pools, not this one-copy size. Model configuration loading follows the
    SDK's normal model loader.

    ``block_size`` is the requested page granularity (default 64).
    It must satisfy the attention kernel's alignment. Automatic sizing follows
    the pinned vLLM KDA none/align layout and Triton MLA's 16-token minimum;
    supply a larger granularity for kernels requiring it (e.g. 128 for CUTLASS).
    The returned block size grows to fit a recurrent state. An explicit token
    byte rate overrides the model's rate, including caller-normalized geometry.

    State bytes include per-layer page padding, before pool-block rounding.
    ``num_speculative_tokens`` is metadata: it does not change a KDA copy's shape.
    """
    from ..models import _architecture_to_model_family, _get_model_info
    from .deepseek_v4 import BLOCK_SIZE, deepseek_v4_state_cache
    from .kimi_k3 import kimi_k3_state_cache

    info = _get_model_info(model_path)
    family = _architecture_to_model_family(info["architecture"])
    if family == "QWEN35":
        return _qwen35_sglang_state_cache(
            info["raw_config"],
            backend=backend,
            tp_size=tp_size,
            pp_size=pp_size,
            block_size=block_size,
            kv_bytes_per_token=kv_bytes_per_token,
            kvcache_quant_mode=kvcache_quant_mode,
            mamba_cache_dtype=mamba_cache_dtype,
            indexer_cache_dtype=indexer_cache_dtype,
            num_speculative_tokens=num_speculative_tokens,
        )
    if family not in ("KIMIK3", "DEEPSEEKV4"):
        raise ValueError("automatic state sizing supports Kimi K3 and DeepSeek V4; supply manual state geometry")
    if backend != "vllm" or pp_size != 1:
        raise ValueError("automatic state sizing requires vllm and PP=1")
    for name, value, minimum in (
        ("tp_size", tp_size, 1),
        ("pp_size", pp_size, 1),
        ("num_speculative_tokens", num_speculative_tokens, 0),
    ):
        _integer(name, value, minimum)
    if family == "DEEPSEEKV4":
        if block_size not in (None, BLOCK_SIZE):
            raise ValueError(f"vLLM's DeepSeek V4 kernels and compressor states use {BLOCK_SIZE}-token blocks")
        if kv_bytes_per_token is not None:
            raise ValueError("DeepSeek V4 token bytes follow vLLM's cache groups; leave bytes_per_token as auto")
        if kvcache_quant_mode not in (None, "fp8"):
            raise ValueError("vLLM's DeepSeek V4 FlashMLA backend stores fp8_ds_mla KV; use kvcache_quant_mode fp8")
        if mamba_cache_dtype != "auto":
            raise ValueError("mamba_cache_dtype applies to Kimi K3 only")
        # KV is replicated on every TP rank, so per-rank geometry does not depend on TP.
        return deepseek_v4_state_cache(
            info["extra_params"], indexer_cache_dtype=indexer_cache_dtype, num_speculative_tokens=num_speculative_tokens
        )
    if indexer_cache_dtype != "auto":
        raise ValueError("indexer_cache_dtype applies to DeepSeek V4 only")
    return kimi_k3_state_cache(
        model_path,
        info,
        backend=backend,
        tp_size=tp_size,
        block_size=block_size,
        kv_bytes_per_token=kv_bytes_per_token,
        kvcache_quant_mode=kvcache_quant_mode,
        mamba_cache_dtype=mamba_cache_dtype,
        num_speculative_tokens=num_speculative_tokens,
    )


def _integer(name: str, value: int, minimum: int) -> None:
    if type(value) is not int or not minimum <= value <= (1 << 64) - 1:
        raise ValueError(f"{name} must be an integer from {minimum} through u64 max")


# Physical geometry derived from SGLang at immutable revision a6b542813ff46de8ba3856e0236f1af35f4fca54.
# See THIRD_PARTY_NOTICES.md for the source paths and Apache-2.0 attribution.
def _qwen35_sglang_state_cache(
    raw: dict[str, Any],
    *,
    backend: str,
    tp_size: int,
    pp_size: int,
    block_size: int | None,
    kv_bytes_per_token: int | None,
    kvcache_quant_mode: str | None,
    mamba_cache_dtype: str,
    indexer_cache_dtype: str,
    num_speculative_tokens: int,
) -> dict[str, Any]:
    _integer("tp_size", tp_size, 1)
    _integer("pp_size", pp_size, 1)
    if backend != "sglang" or pp_size != 1:
        raise ValueError("Qwen3.5 recurrent state sizing requires SGLang and PP=1")
    _integer("num_speculative_tokens", num_speculative_tokens, 0)
    if num_speculative_tokens:
        raise ValueError("Qwen3.5 speculative state copies are not implemented")
    if indexer_cache_dtype != "auto":
        raise ValueError("Qwen3.5 GQA does not use an indexer cache")
    text = raw.get("text_config", raw)
    layers = text.get("layer_types", [])
    state_layers = layers.count("linear_attention")
    attention_layers = layers.count("full_attention")
    if not state_layers or not attention_layers or len(layers) != text.get("num_hidden_layers"):
        raise ValueError("Qwen3.5 requires a complete hybrid layer_types layout")
    names = (
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
        "num_key_value_heads",
        "head_dim",
    )
    for name in names:
        _integer(name, text.get(name), 1)
    key_heads = text["linear_num_key_heads"]
    value_heads = text["linear_num_value_heads"]
    if key_heads % tp_size or value_heads % tp_size:
        raise ValueError("Qwen3.5 linear key/value heads must be divisible by attention TP")
    kernel = text["linear_conv_kernel_dim"]
    if kernel < 2:
        raise ValueError("Qwen3.5 convolution kernel must retain at least one history element")
    width = {"bfloat16": 2, "float16": 2, "float32": 4}
    model_dtype = text.get("dtype", text.get("torch_dtype"))
    conv_dtype = model_dtype if mamba_cache_dtype == "auto" else mamba_cache_dtype
    ssm_dtype = text.get("mamba_ssm_dtype", "float32")
    if conv_dtype not in width or ssm_dtype not in width:
        raise ValueError("Qwen3.5 requires explicit supported convolution and SSM dtypes")
    kv_dtype = model_dtype if kvcache_quant_mode is None else kvcache_quant_mode
    if kv_dtype not in ("bfloat16", "float16"):
        raise ValueError("Qwen3.5 SGLang automatic KV sizing currently supports BF16/FP16")
    block_size = 128 if block_size is None else block_size
    _integer("block_size", block_size, 1)
    kv_heads = text["num_key_value_heads"]
    if kv_heads % tp_size and tp_size % kv_heads:
        raise ValueError("GQA KV heads and attention TP must divide one another")
    token_bytes = attention_layers * 2 * max(1, kv_heads // tp_size) * text["head_dim"] * width[kv_dtype]
    if kv_bytes_per_token is not None and kv_bytes_per_token != token_bytes:
        raise ValueError("Qwen3.5 KV byte override differs from the physical SGLang GQA layout")
    key_dim, value_dim = text["linear_key_head_dim"], text["linear_value_head_dim"]
    conv_groups = [key_heads * key_dim // tp_size, key_heads * key_dim // tp_size, value_heads * value_dim // tp_size]
    conv_shape = [sum(conv_groups), kernel - 1]
    temporal_shape = [value_heads // tp_size, value_dim, key_dim]
    conv_bytes = state_layers * conv_shape[0] * conv_shape[1] * width[conv_dtype]
    temporal_bytes = state_layers * temporal_shape[0] * value_dim * key_dim * width[ssm_dtype]
    state_bytes = conv_bytes + temporal_bytes
    _integer("bytes_per_request", state_bytes, 1)
    _integer("block_bytes", block_size * token_bytes, 1)
    return {
        "source": "inferred",
        "allocation": "separate_request_slots",
        "block_size": block_size,
        "kv_bytes_per_token": token_bytes,
        "bytes_per_request": state_bytes,
        "raw_bytes_per_request": state_bytes,
        "padding_bytes_per_request": 0,
        "conv_bytes_per_request": conv_bytes,
        "ssm_bytes_per_request": temporal_bytes,
        "conv_shape_per_layer": conv_shape,
        "temporal_shape_per_layer": temporal_shape,
        "conv_shard_groups_per_rank": conv_groups,
        "state_layers": state_layers,
        "attention_layers": attention_layers,
        "conv_dtype": conv_dtype,
        "ssm_dtype": ssm_dtype,
        "attention_tp": tp_size,
        "backend_revision": "a6b542813ff46de8ba3856e0236f1af35f4fca54",
    }
