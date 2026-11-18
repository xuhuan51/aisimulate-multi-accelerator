# SPDX-License-Identifier: Apache-2.0
# Modified for AISimulate multi-accelerator workload replay.
# Upstream attribution is retained in LICENSE and source-manifest.json.
"""Keep weight-loader metadata lightweight for the CPU mock model."""

import os
from pathlib import Path

from sglang_simulator.hook import BaseHook


class C_ModelConfigHook(BaseHook):
    HOOK_CLASS_NAME = "ModelConfig"
    HOOK_MODULE_NAME = "sglang.srt.configs.model_config"

    @classmethod
    def hook(cls, target):
        original = target._find_quant_modelslim_config

        def find_modelslim(self):
            if os.environ.get("SGLANG_USE_CPU_ENGINE") != "1":
                return original(self)
            # The native CPU runner replaces the weight loader and forward.
            # Quantization verification needs the format marker. Per-weight
            # packing entries belong to the replaced loader and can be large.
            if self.is_draft_model:
                return None
            if Path(self.model_path, "quant_model_description.json").is_file():
                return {"quant_method": "modelslim"}
            return None

        target._find_quant_modelslim_config = find_modelslim


class C_MambaPoolHook(BaseHook):
    HOOK_CLASS_NAME = "MambaPool"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.memory_pool"

    @classmethod
    def hook(cls, target):
        original_init = target.__init__

        def initialize(self, *args, **kwargs):
            if os.environ.get("SGLANG_USE_CPU_ENGINE") != "1" or kwargs.get("device") != "cpu":
                return original_init(self, *args, **kwargs)
            # The forward stub never reads recurrent-state payloads. Meta
            # tensors retain their shapes, dtypes and logical byte accounting.
            # Slot allocation and request-to-state mappings live in the native
            # CPU request pool and allocator outside this payload constructor.
            kwargs["device"] = "meta"
            original_init(self, *args, **kwargs)
            self.device = "cpu"

        target.__init__ = initialize
