from dataclasses import dataclass
import unittest

from sglang_simulator.simulation.sglang.runtime_adapter import server_arg_fields


class ServerArgFieldsTests(unittest.TestCase):
    def test_dataclass_fields_match_native_replay_filtering(self):
        @dataclass
        class ServerArgs:
            dp_size: int = 1
            enable_dp_attention: bool = False

        self.assertEqual(server_arg_fields(ServerArgs), {"dp_size", "enable_dp_attention"})

    def test_msgspec_fields_preserve_the_existing_contract(self):
        class ServerArgs:
            __struct_fields__ = ("tp_size", "attn_dp_size")

        self.assertEqual(server_arg_fields(ServerArgs), {"tp_size", "attn_dp_size"})

    def test_unknown_representation_fails_explicitly(self):
        with self.assertRaisesRegex(TypeError, "Unsupported SGLang ServerArgs"):
            server_arg_fields(object)
