"""Invoke installed NPU implementations; independently check sampled CPU values.

Contracts: SGLang baa9befa381511cd4d56f83980951e5a45fffc6d ModelSlim
normalization, layernorm.py, deepseek_v2.py and memory_pool_npu.py; companion
vector-state-report.json records actual dtypes, strides and call arguments.
No serving kernel implementation is copied into this module.
"""

from types import SimpleNamespace


def prepare(case, torch, torch_npu):
    role, dimensions, tokens = case["role"], case["dimensions"], case["tokens"]
    rows = sorted({0, tokens // 2, tokens - 1})
    inputs = {}
    stateful = False

    def random(shape, dtype=None):
        return torch.randn(*shape, device="npu", dtype=dtype or torch.bfloat16)

    def cpu(value):
        return value[rows].cpu().float()

    def reset():
        return None

    if role == "dp_buffer_zero":
        if dimensions != [6144]:
            raise ValueError("DP buffer zero requires BF16 hidden width 6144")
        values = random((tokens, dimensions[0]))
        inputs = {"buffer": values}
        kernel = "torch.Tensor.fill_"

        def invoke():
            return values.fill_(0)

        def reference():
            return torch.zeros(len(rows), dimensions[0])

        def perturb():
            values.fill_(3)

    elif role in {"dp_gather_copy", "dp_scatter_copy"}:
        # SGLang baa9befa dp_attention.memcpy delegates to the installed
        # memcpy_triton. Its grid is min(src.numel(), dst.numel()), and its
        # device scalars select a contiguous row range. Use a nonzero offset
        # in both directions and check the untouched destination rows too.
        from sglang.srt.layers.dp_attention import memcpy, memcpy_func

        if dimensions != [6144]:
            raise ValueError("DP buffer copy requires BF16 hidden width 6144")
        if memcpy_func.__module__ != "sglang.kernels.ops.memory.memcpy_triton":
            raise ValueError("Expected SGLang's NPU Triton DP copy dispatch")
        scatter = role == "dp_scatter_copy"
        source = random((tokens * (2 if scatter else 1), dimensions[0]))
        target = torch.full((tokens * (1 if scatter else 2), dimensions[0]), -7, device="npu", dtype=torch.bfloat16)
        offset = torch.tensor(tokens, device="npu", dtype=torch.int32)
        size = torch.tensor(tokens, device="npu", dtype=torch.int32)
        inputs = {"source": source, "target": target, "offset": offset, "size": size}
        kernel = "sglang.srt.layers.dp_attention.memcpy"

        def invoke():
            memcpy(target, source, 0, offset, size, scatter)
            return target

        def reference():
            start = tokens if scatter else 0
            return source[[start + row for row in rows]].cpu().float()

        def sample_output(output):
            if not scatter and not torch.all(output[:tokens] == -7):
                raise ValueError("DP gather copy changed rows outside its destination range")
            start = 0 if scatter else tokens
            return output[[start + row for row in rows]].cpu().float()

        def perturb():
            source.neg_()
            target.fill_(-7)

    elif role == "seq_lens_h2d":
        if dimensions != [4] or case["execution_mode"] != "eager":
            raise ValueError("Sequence-length transfers require eager CPU INT32 metadata")
        values = torch.arange(tokens, dtype=torch.int32, device="cpu") + 16384
        inputs = {"lengths": values}
        kernel = "torch.Tensor.to"

        def invoke():
            return values.to(device="npu", dtype=torch.int32)

        def reference():
            return cpu(values)

        def perturb():
            values.add_(1)

    elif role in {"rms_bias", "add_rms_bias"}:
        from sglang.srt.layers.quantization.modelslim.modelslim import npu_wrapper_rmsnorm_forward

        width = dimensions[0]
        values = random((tokens, width))
        weight = torch.ones(width, device="npu", dtype=torch.bfloat16)
        bias = random((width,)) * 0.125
        residual = random((tokens, width)) if role == "add_rms_bias" else None
        norm = SimpleNamespace(weight=weight, bias=bias, variance_epsilon=1e-5)
        method = npu_wrapper_rmsnorm_forward(None)
        inputs = {"input": values, "weight": weight, "bias": bias, "residual": residual}
        kernel = "sglang.srt.layers.quantization.modelslim.modelslim.npu_wrapper_rmsnorm_forward"

        def invoke():
            return method(norm, values, residual)

        def reference():
            original = cpu(values)
            if residual is not None:
                original = (original + cpu(residual)).to(torch.bfloat16).float()
            normalized = original * (original.square().mean(-1, keepdim=True) + 1e-5).rsqrt()
            normalized *= weight.float().cpu()
            if residual is None:
                normalized = normalized.to(torch.bfloat16).float()
            normalized = (normalized + bias.float().cpu()).to(torch.bfloat16).float()
            return (normalized, original) if residual is not None else normalized

        def perturb():
            values.neg_()

    elif role == "split_qk_norm":
        from sgl_kernel_npu.norm.fused_split_qk_norm import fused_split_qk_norm

        query_width, key_width, rope_width = dimensions
        values = random((tokens, sum(dimensions)))
        query_norm = SimpleNamespace(weight=random((query_width,)), bias=random((query_width,)) * 0.125)
        key_norm = SimpleNamespace(weight=random((key_width,)), bias=random((key_width,)) * 0.125)
        inputs = {
            "input": values,
            "query_weight": query_norm.weight,
            "query_bias": query_norm.bias,
            "key_weight": key_norm.weight,
            "key_bias": key_norm.bias,
        }
        kernel = "sgl_kernel_npu.norm.fused_split_qk_norm.fused_split_qk_norm"

        def invoke():
            return fused_split_qk_norm(values, query_norm, key_norm, *dimensions, eps=1e-5)

        def reference():
            query, key, rope = cpu(values).split(dimensions, dim=-1)

            def normalized(value, norm):
                return (
                    (
                        (value * (value.square().mean(-1, keepdim=True) + 1e-5).rsqrt()) * norm.weight.float().cpu()
                        + norm.bias.float().cpu()
                    )
                    .to(torch.bfloat16)
                    .float()
                )

            return normalized(query, query_norm), normalized(key, key_norm).unsqueeze(1), rope.unsqueeze(1)

        def perturb():
            values.neg_()

    elif role == "indexer_norm":
        from sglang.srt.layers.layernorm import LayerNorm

        values = random((tokens, dimensions[0]))
        with torch.device("npu"):
            norm = LayerNorm(dimensions[0], eps=1e-6, dtype=torch.float32)
        norm.weight.data.fill_(1.125)
        norm.bias.data.fill_(0.0625)
        inputs = {"input": values, "weight": norm.weight, "bias": norm.bias}
        kernel = "sglang.srt.layers.layernorm.LayerNorm.forward_npu"

        def invoke():
            return norm.forward_npu(values)

        def reference():
            sample = cpu(values)
            centered = sample - sample.mean(-1, keepdim=True)
            return (
                (centered * (centered.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * 1.125 + 0.0625)
                .to(torch.bfloat16)
                .float()
            )

        def perturb():
            values.neg_()

    elif role in {"rope_attention", "rope_indexer"}:
        from sgl_kernel_npu.norm.fused_rope_qk_mqa import fused_rope_qk_mqa

        heads, width, query_storage, key_storage = dimensions
        query_start = 0 if role == "rope_indexer" else query_storage - width
        query = random((tokens, heads, query_storage))[..., query_start : query_start + width]
        key = random((tokens, 1, key_storage))[..., :width]
        angles = torch.randn(tokens, width // 2, device="npu", dtype=torch.float32)
        cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
        inputs = {"query": query, "key": key, "cos_sin": cache}
        kernel = "sgl_kernel_npu.norm.fused_rope_qk_mqa.fused_rope_qk_mqa"

        def invoke():
            return fused_rope_qk_mqa(query, key, cache, width, False)

        def reference():
            cosine, sine = cpu(cache).chunk(2, dim=-1)

            def rotate(value):
                sample = cpu(value)
                result = torch.empty_like(sample)
                result[..., 0::2] = sample[..., 0::2] * cosine[:, None] - sample[..., 1::2] * sine[:, None]
                result[..., 1::2] = sample[..., 0::2] * sine[:, None] + sample[..., 1::2] * cosine[:, None]
                return result.to(torch.bfloat16).float()

            return rotate(query), rotate(key)

        def perturb():
            query.neg_()
            key.neg_()

    elif role in {"concat_indexer_q", "concat_indexer_k"}:
        heads, width, storage = dimensions
        leading = random((tokens, heads, width))
        trailing = random((tokens, heads, storage))[..., width:]
        if role == "concat_indexer_k":
            leading, trailing = leading.squeeze(1), trailing.squeeze(1)
        inputs = {"leading": leading, "trailing": trailing}
        kernel = "torch.cat"

        def invoke():
            return torch.cat((leading, trailing), dim=-1)

        def reference():
            return torch.cat((cpu(leading), cpu(trailing)), dim=-1)

        def perturb():
            leading.neg_()
            trailing.neg_()

    elif role == "cache_scatter":
        capacity, width = dimensions
        owner = torch.zeros(1, capacity // 128, 128, 1, width, device="npu", dtype=torch.bfloat16)
        cache = owner[0].view(-1, 1, width)
        indices = (
            torch.arange(tokens, device="npu", dtype=torch.int32 if case["execution_mode"] == "graph" else torch.int64)
            + 128
        ).view(-1, 1)
        values = random((tokens, 1, width))
        if torch_npu.get_npu_format(cache) != 30 or capacity % 128 or tokens + 128 > capacity:
            raise ValueError("Cache allocation/format differs from the declared contract")
        inputs = {"cache": cache, "indices": indices, "updates": values}
        kernel = "torch_npu.npu_scatter_nd_update_"

        def invoke():
            return torch_npu.npu_scatter_nd_update_(cache, indices, values)

        def reference():
            return cpu(values)

        def sample_output(output):
            return output[[row + 128 for row in rows]].float().cpu()

        def perturb():
            values.neg_()
            cache.zero_()

    elif role == "swiglu":
        values = random((tokens, 2 * dimensions[0]))
        inputs = {"input": values}
        kernel = "torch_npu.npu_swiglu"

        def invoke():
            return torch_npu.npu_swiglu(values)

        def reference():
            gate, up = cpu(values).chunk(2, dim=-1)
            return (gate * gate.sigmoid() * up).to(torch.bfloat16).float()

        def perturb():
            values.neg_()

    elif role == "router_topk":
        experts, topk = dimensions
        values = random((tokens, experts))
        bias = random((experts,), torch.float32) * 0.01
        inputs = {"logits": values, "bias": bias}
        kernel = "torch.ops.npu.npu_moe_gating_top_k"

        def invoke():
            return torch.ops.npu.npu_moe_gating_top_k(
                values.float(),
                k=topk,
                bias=bias,
                k_group=1,
                group_count=1,
                group_select_mode=1,
                renorm=True,
                norm_type=1,
                routed_scaling_factor=1,
                eps=1e-20,
            )

        def reference():
            scores = cpu(values).sigmoid()
            selected = (scores + bias.cpu()).topk(topk, dim=-1).indices
            weights = scores.gather(1, selected)
            return weights / weights.sum(-1, keepdim=True), selected

        def perturb():
            values.neg_()

    elif role in {"cast_hidden_fp32", "cast_weights_bf16"}:
        values = random((tokens, dimensions[0]), torch.bfloat16 if role == "cast_hidden_fp32" else torch.float32)
        target = torch.float32 if role == "cast_hidden_fp32" else torch.bfloat16
        inputs = {"input": values}
        kernel = "torch.Tensor.to"

        def invoke():
            return values.to(target)

        def reference():
            return cpu(values).to(target).float()

        def perturb():
            values.neg_()

    elif role == "logits_finalize":
        from sglang.srt.layers.logits_processor import LogitsProcessor

        parallel, vocabulary = dimensions
        gathered = random((parallel, tokens, vocabulary // parallel))
        buffer = (
            torch.empty((32, vocabulary), device="npu", dtype=torch.float32)[:tokens]
            if case["execution_mode"] == "graph"
            else None
        )
        metadata = SimpleNamespace(next_token_logits_buffer=buffer)
        processor = SimpleNamespace(vocab_size=vocabulary)
        method = LogitsProcessor._copy_logits_to_buffer
        inputs = {"gathered_logits": gathered, "logits_buffer": buffer}
        kernel = method.__module__ + "." + method.__qualname__

        def invoke():
            logits = gathered.permute(1, 0, 2).reshape(tokens, vocabulary)
            return method(processor, logits, metadata)

        def reference():
            return torch.stack(
                [torch.cat([gathered[rank, row].cpu().float() for rank in range(parallel)]) for row in rows]
            )

        def perturb():
            gathered.neg_()

    elif role in {"embedding", "rope_cache_gather"}:
        vocabulary, width = dimensions
        weight = random((vocabulary, width))
        indices = torch.arange(tokens, device="npu", dtype=torch.int64) % vocabulary
        inputs = {"indices": indices, "weight": weight}
        kernel = "torch.nn.functional.embedding" if role == "embedding" else "torch.Tensor.index_select"

        def invoke():
            return (
                torch.nn.functional.embedding(indices, weight)
                if role == "embedding"
                else weight.index_select(0, indices)
            )

        def reference():
            selected = indices[rows].cpu().tolist()
            return torch.stack([weight[index].float().cpu() for index in selected])

        def perturb():
            indices.add_(1).remainder_(vocabulary)

    elif role in {"routed_scale", "shared_add"}:
        values = random((tokens, dimensions[0]))
        initial = values.clone()
        shared = random(tuple(values.shape)) if role == "shared_add" else None
        inputs = {"input": values, "shared": shared}
        stateful = True
        kernel = "torch.Tensor.mul_" if role == "routed_scale" else "torch.Tensor.add_"

        def invoke():
            return values.mul_(2.5) if role == "routed_scale" else values.add_(shared)

        def reset():
            values.copy_(initial)

        def reference():
            expected = cpu(values)
            for _ in range(case["iterations"]):
                expected = (
                    (expected * 2.5 if role == "routed_scale" else expected + cpu(shared)).to(torch.bfloat16).float()
                )
            return expected

        def perturb():
            initial.neg_()
            reset()

    else:
        raise ValueError(f"Unknown vector role: {role}")

    if role not in {"cache_scatter", "dp_gather_copy", "dp_scatter_copy"}:
        sample_output = cpu

    def check(observed, expected):
        if role == "router_topk":
            actual_weights, actual_indices = cpu(observed[0]), observed[1][rows].long().cpu()
            expected_weights, expected_indices = expected
            actual_order, expected_order = actual_indices.argsort(-1), expected_indices.argsort(-1)
            actual_weights = actual_weights.gather(1, actual_order)
            expected_weights = expected_weights.gather(1, expected_order)
            if not torch.equal(
                actual_indices.gather(1, actual_order), expected_indices.gather(1, expected_order)
            ) or not torch.allclose(actual_weights, expected_weights, rtol=2e-4, atol=1e-5):
                raise ValueError("Vector execution numerical check failed: router_topk")
            return actual_weights.numel() + actual_indices.numel(), (
                actual_weights - expected_weights
            ).abs().max().item()
        actual = observed if isinstance(observed, tuple) else (observed,)
        wanted = expected if isinstance(expected, tuple) else (expected,)
        if len(actual) != len(wanted):
            raise ValueError("Vector output arity mismatch")
        count, maximum = 0, 0.0
        for tensor, target in zip(actual, wanted, strict=True):
            value = sample_output(tensor)
            error = (value - target).abs()
            if role in {"logits_finalize", "seq_lens_h2d"} and not torch.equal(value, target):
                raise ValueError(f"Vector execution exact numerical check failed: {role}")
            if not torch.all(error <= 0.02 + 0.02 * target.abs()):
                raise ValueError(f"Vector execution numerical check failed: {role}, max={error.max().item()}")
            count += value.numel()
            maximum = max(maximum, error.max().item())
        return count, maximum

    return SimpleNamespace(
        invoke=invoke,
        reference=reference,
        perturb=perturb,
        reset=reset,
        check=check,
        inputs=inputs,
        kernel_source=kernel,
        stateful=stateful,
    )
