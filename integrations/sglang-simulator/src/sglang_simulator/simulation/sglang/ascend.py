# SPDX-License-Identifier: Apache-2.0
# Modified for AISimulate multi-accelerator workload replay.
# Upstream attribution is retained in LICENSE and source-manifest.json.
"""CPU execution boundary for the dataclass-based Ascend SGLang runtime.

Keep native scheduling, radix-cache, pool configuration and allocation. Only
model execution and unused KV payload dimensions are replaced. This is not an
NPU kernel implementation or a hardware-memory capacity estimator.
"""

import torch

from sglang_simulator.hook import BaseHook


def install_scheduler_observation(target):
    original = target.get_internal_state

    def get_internal_state(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        allocator = self.token_to_kv_pool_allocator
        result.internal_state["simulator_native_kv"] = {
            "capacity": int(self.max_total_num_tokens),
            "available": int(allocator.available_size()),
            "evictable": int(self.tree_cache.evictable_size()),
            "protected": int(self.tree_cache.protected_size()),
            "pool_class": type(allocator.get_kvcache()).__name__,
            "allocator_class": type(allocator).__name__,
        }
        return result

    target.get_internal_state = get_internal_state


def receive_requests(scheduler, original, *args, **kwargs):
    """Deliver logical arrivals before native attention-TP/control broadcasts.

    The older Scheduler owns its sockets directly rather than a request receiver.
    Only the native socket boundary is adapted; native broadcast and request
    processing still execute in the original recv_requests method.
    """
    import zmq

    from sglang_simulator.simulation.parallel.client import get_client

    leader = scheduler.attn_tp_rank == 0 and scheduler.attn_cp_rank == 0
    incoming = []
    socket = scheduler.recv_from_tokenizer
    if leader:
        while True:
            try:
                incoming.append(socket.recv_pyobj(zmq.NOBLOCK))
            except zmq.Again:
                break
    ready = get_client().receive(incoming)
    if not leader:
        return original(scheduler, *args, **kwargs)
    due = getattr(scheduler, "_simulator_pending_requests", []) + ready

    class LogicalSocket:
        def recv_pyobj(self, flags=0):
            if due:
                return due.pop(0)
            raise zmq.Again()

        def __getattr__(self, name):
            return getattr(socket, name)

    scheduler.recv_from_tokenizer = LogicalSocket()
    try:
        return original(scheduler, *args, **kwargs)
    finally:
        scheduler.recv_from_tokenizer = socket
        # The native receiver can stop at its per-poll limit or skip a poll.
        # Keep undelivered arrivals for the next native receive boundary.
        scheduler._simulator_pending_requests = due


def install_model_runner(target):
    from sglang_simulator.simulation.manager import ConfigManager
    from sglang_simulator.simulation.sglang.model_runner import _MockModel
    from sglang_simulator.simulation.sglang.utils import resolve_model_info

    def load_model(self):
        self.model = _MockModel()
        self.dtype = self.model_config.dtype
        self.sliding_window_size = None
        self.weight_load_mem_usage = 0
        if ConfigManager.get_model_info() is None:
            ConfigManager.set_model_info(resolve_model_info(self.model_config))

    def forward(self, forward_batch, *args, **kwargs):
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.model_executor.model_runner import ModelRunnerOutput

        return ModelRunnerOutput(
            logits_output=LogitsProcessorOutput(
                next_token_logits=torch.empty(
                    (forward_batch.batch_size, self.model_config.vocab_size),
                    device=self.device,
                )
            ),
            can_run_graph=False,
        )

    def sample(self, logits_output, *args, **kwargs):
        return torch.ones(
            logits_output.next_token_logits.shape[0], dtype=torch.int64, device=self.device
        )

    def init_attention_backend(self):
        self.attn_backend = None

    target.load_model = load_model
    target.forward = forward
    target.sample = sample
    target.init_attention_backend = init_attention_backend


class C_AscendKVCacheHook(BaseHook):
    HOOK_CLASS_NAME = "ModelRunnerKVCacheMixin"
    HOOK_MODULE_NAME = "sglang.srt.model_executor.model_runner_kv_cache_mixin"

    @classmethod
    def hook(cls, target):
        original_init_pools = target._init_pools

        def profile_max_num_token(self, pre_model_load_memory):
            capacity = self.server_args.max_total_tokens
            if type(capacity) is not int or capacity <= 0:
                raise ValueError("Ascend simulation requires explicit --max-total-tokens")
            return capacity

        def init_pools(self, *args, **kwargs):
            # Preserve native capacity/page bookkeeping while avoiding large,
            # unread model-specific payload tensors on the CPU host.
            attrs = ("head_dim", "v_head_dim", "kv_lora_rank", "qk_rope_head_dim")
            geometry = {
                name: getattr(self.model_config, name)
                for name in attrs
                if hasattr(self.model_config, name)
            }
            try:
                for name in geometry:
                    setattr(self.model_config, name, 1)
                result = original_init_pools(self, *args, **kwargs)
            finally:
                for name, value in geometry.items():
                    setattr(self.model_config, name, value)
            pool = self.token_to_kv_pool
            for name, value in geometry.items():
                if hasattr(pool, name):
                    setattr(pool, name, value)
            if hasattr(pool, "kv_cache_dim"):
                pool.kv_cache_dim = self.calculate_mla_kv_cache_dim()
            return result

        target.profile_max_num_token = profile_max_num_token
        target._init_pools = init_pools
