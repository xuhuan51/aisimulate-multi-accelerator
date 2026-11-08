"""Metadata-only tests; no device or model weights required."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pytest

from aisimulate_core.modelslim import read_modelslim

pytestmark = pytest.mark.unit


class QwenModelSlimTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": {
                "dtype": "bfloat16",
                "num_hidden_layers": 2,
                "num_experts": 2,
                "layer_types": ["linear_attention", "full_attention"],
                "shared_expert_intermediate_size": 8,
                "mamba_ssm_dtype": "float32",
            },
        }
        self.quant = {
            "model_quant_type": "W8A8_DYNAMIC",
            "model.language_model.embed_tokens.weight": "FLOAT",
            "lm_head.weight": "FLOAT",
        }
        for i, kind in enumerate(self.config["text_config"]["layer_types"]):
            prefix = f"model.language_model.layers.{i}."
            attn = ("linear_attn", ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "conv1d", "norm"])
            if kind == "full_attention":
                attn = ("self_attn", ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"])
            for name in attn[1]:
                quant = "W8A8_DYNAMIC" if i == 1 and name in ("q_proj", "k_proj", "v_proj") else "FLOAT"
                self.quant[prefix + attn[0] + "." + name + ".weight"] = quant
            for name in ("input_layernorm", "post_attention_layernorm", "mlp.gate", "mlp.shared_expert_gate"):
                self.quant[prefix + name + ".weight"] = "FLOAT"
            for proj in ("gate_proj", "up_proj", "down_proj"):
                self.quant[prefix + f"mlp.shared_expert.{proj}.weight"] = "FLOAT"
                for e in range(2):
                    self.quant[prefix + f"mlp.experts.{e}.{proj}.weight"] = "W8A8_DYNAMIC"

    def read(self):
        (self.root / "quant_model_description.json").write_text(json.dumps(self.quant))
        return read_modelslim(str(self.root), self.config)

    def test_per_operation_precision_and_roundtrip(self):
        before = copy.deepcopy(self.config)
        metadata = self.read()
        self.assertEqual(metadata["layers"][0]["attention"]["in_proj_qkv"], "FLOAT")
        self.assertEqual(metadata["layers"][1]["attention"]["q_proj"], "W8A8_DYNAMIC")
        self.assertEqual(metadata["layers"][1]["attention"]["o_proj"], "FLOAT")
        self.assertEqual(metadata["layers"][1]["mlp"]["shared"]["up_proj"], "FLOAT")
        self.assertEqual(metadata["layers"][1]["mlp"]["routed"]["up_proj"], "W8A8_DYNAMIC")
        self.assertEqual(metadata["declared_mamba_ssm_dtype"], "float32")
        self.assertEqual(json.loads(json.dumps(metadata)), metadata)
        self.assertEqual(before, self.config)
        self.assertEqual(
            metadata["source_sha256"],
            hashlib.sha256((self.root / "quant_model_description.json").read_bytes()).hexdigest(),
        )

    def test_missing_final_expert_is_rejected(self):
        del self.quant["model.language_model.layers.1.mlp.experts.1.down_proj.weight"]
        with self.assertRaisesRegex(ValueError, "experts.1.down_proj"):
            self.read()

    def test_changed_precision_is_not_silently_defaulted(self):
        for key in (
            "model.language_model.layers.1.self_attn.q_proj.weight",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            "model.language_model.layers.1.mlp.experts.1.up_proj.weight",
        ):
            with self.subTest(key=key):
                old = self.quant[key]
                self.quant[key] = "FLOAT" if old != "FLOAT" else "W8A8_DYNAMIC"
                with self.assertRaisesRegex(ValueError, "precision"):
                    self.read()
                self.quant[key] = old

    def test_bad_structure_rejected(self):
        for key, value in [("num_hidden_layers", 3), ("num_experts", 0), ("layer_types", ["bad", "full_attention"])]:
            with self.subTest(key=key):
                old = self.config["text_config"][key]
                self.config["text_config"][key] = value
                with self.assertRaisesRegex(ValueError, "structure"):
                    self.read()
                self.config["text_config"][key] = old

    def test_no_sidecar_is_not_modelslim(self):
        self.assertIsNone(read_modelslim(str(self.root), self.config))

    def test_local_config_loader_preserves_partition(self):
        from aisimulate_core.sdk.utils import _load_model_config_from_model_path

        self.read()
        (self.root / "config.json").write_text(json.dumps(self.config))
        raw = _load_model_config_from_model_path(str(self.root))
        self.assertEqual(raw["quant_algo"], "modelslim")
        self.assertEqual(raw["modelslim_metadata"]["full_attention_qkv_mode"], "w8a8_dynamic")
        self.assertEqual(raw["text_config"]["mamba_ssm_dtype"], "float32")

    def test_generic_graph_cannot_consume_unqualified_metadata(self):
        from aisimulate_core.sdk.models.qwen35 import Qwen35Model

        with self.assertRaisesRegex(ValueError, "Ascend physical operator graph"):
            Qwen35Model(raw_config={"modelslim_metadata": self.read()}, backend_name="sglang")

    def test_dtype_and_rotation_require_explicit_support(self):
        self.config["text_config"]["dtype"] = "float16"
        with self.assertRaisesRegex(ValueError, "dtype"):
            self.read()
        self.config["text_config"]["dtype"] = "bfloat16"
        self.quant["is_rot_used"] = True
        with self.assertRaisesRegex(ValueError, "rotation"):
            self.read()

    def test_unknown_architecture_not_discarded(self):
        self.config["architectures"] = ["UnknownModel"]
        with self.assertRaisesRegex(ValueError, "ModelSlim"):
            self.read()


if __name__ == "__main__":
    unittest.main()
