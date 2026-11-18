"""Native Mamba payload geometry and request-slot accounting."""
import math
import unittest
from types import SimpleNamespace

from sglang_simulator.simulation.sglang.runtime_adapter import (
    recurrent_state_pool_info,
    recurrent_state_transfer_info,
)


class Tensor:
    def __init__(self, shape, item_size, dtype):
        self.shape, self.item_size, self.dtype, self.device = shape, item_size, dtype, "meta"

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return self.item_size


class NativeRecurrentStateTests(unittest.TestCase):
    def test_observed_shapes_and_slot_peak_are_preserved(self):
        free = [2]
        state = SimpleNamespace(
            conv=[Tensor([45, 3, 3072, 3], 2, "torch.bfloat16")],
            temporal=Tensor([45, 3, 16, 128, 128], 4, "torch.float32"),
        )
        pool = SimpleNamespace(mamba_pool=SimpleNamespace(size=2, mamba_cache=state),
                               mamba_allocator=SimpleNamespace(available_size=lambda: free[0]))
        result = recurrent_state_pool_info(pool)
        self.assertEqual(result["bytes_per_request"], 48015360)
        self.assertEqual(result["padding_slots"], 1)
        self.assertEqual(result["peak_used_slots"], 0)
        free[0] = 0
        self.assertEqual(recurrent_state_pool_info(pool)["peak_used_slots"], 2)
        free[0] = 2
        result = recurrent_state_pool_info(pool)
        self.assertEqual(result["peak_used_slots"], 2)
        self.assertEqual(result["available_slots"], 2)
        self.assertEqual(result["payload_storage"], "meta")

    def test_profile_exports_then_resets_the_native_peak(self):
        from sglang_simulator.simulation.parallel.client import RankClient

        free = [1]
        state = SimpleNamespace(conv=[Tensor([1, 2, 3, 2], 2, "torch.bfloat16")],
                                temporal=Tensor([1, 2, 1, 2, 2], 4, "torch.float32"))
        pool = SimpleNamespace(mamba_pool=SimpleNamespace(size=1, mamba_cache=state),
                               mamba_allocator=SimpleNamespace(available_size=lambda: free[0]))
        free[0] = 0
        recurrent_state_pool_info(pool)
        free[0] = 1
        captured = []
        coordinator = SimpleNamespace(exchange=lambda *args: captured.append(args[-1]) or {})
        client = RankClient(coordinator, 0)
        client.scheduler = SimpleNamespace(req_to_token_pool=pool)
        client.profile([])
        self.assertEqual(captured[0]["recurrent_state_pool"]["peak_used_slots"], 1)
        self.assertEqual(recurrent_state_pool_info(pool)["peak_used_slots"], 0)

    def test_nonhybrid_pool_has_no_recurrent_payload(self):
        self.assertIsNone(recurrent_state_pool_info(None))
        self.assertIsNone(recurrent_state_pool_info(SimpleNamespace()))

    def transfer_pool(self, item_bytes=(12, 16)):
        state = SimpleNamespace(
            conv=[Tensor([1, 3, 3, 2], 2, "torch.bfloat16")],
            temporal=Tensor([1, 3, 1, 2, 2], 4, "torch.float32"),
        )
        pool = SimpleNamespace(
            size=2,
            mamba_cache=state,
            get_contiguous_buf_infos=lambda: (
                [0, 0], [item * 3 for item in item_bytes], list(item_bytes)
            ),
            get_state_dim_per_tensor=lambda: [3, 1],
            get_state_layer_ids=lambda: [0, 0],
            get_state_slice_outer_counts=lambda: [1, 1],
            get_state_conv_shard_groups=lambda: [[1, 1, 1], None],
        )
        return SimpleNamespace(
            mamba_pool=pool,
            mamba_allocator=SimpleNamespace(available_size=lambda: 2),
        )

    def test_native_transfer_rows_match_one_state_slot(self):
        result = recurrent_state_transfer_info(self.transfer_pool())
        self.assertTrue(result["available"])
        self.assertEqual(result["buffer_count"], 2)
        self.assertEqual(result["bytes_per_request"], 28)
        self.assertEqual(result["conv_shard_groups"], [[1, 1, 1], None])

    def test_native_transfer_rejects_wrong_payload_size(self):
        with self.assertRaisesRegex(ValueError, "one request"):
            recurrent_state_transfer_info(self.transfer_pool(item_bytes=(12, 32)))

    def test_native_transfer_rejects_incomplete_layer_descriptors(self):
        pool = self.transfer_pool()
        pool.mamba_pool.get_state_layer_ids = lambda: [0]
        with self.assertRaisesRegex(ValueError, "inconsistent lengths"):
            recurrent_state_transfer_info(pool)

    def test_native_transfer_rejects_incorrect_slot_span(self):
        pool = self.transfer_pool()
        pool.mamba_pool.get_contiguous_buf_infos = lambda: ([0, 0], [24, 32], [12, 16])
        with self.assertRaisesRegex(ValueError, "state slots"):
            recurrent_state_transfer_info(pool)

    def test_older_aggregated_runtime_reports_missing_pd_descriptors(self):
        pool = self.transfer_pool()
        del pool.mamba_pool.get_state_conv_shard_groups
        result = recurrent_state_transfer_info(pool)
        self.assertFalse(result["available"])
        self.assertEqual(result["missing_native_methods"], ["get_state_conv_shard_groups"])
        self.assertIsNone(recurrent_state_transfer_info(None))
