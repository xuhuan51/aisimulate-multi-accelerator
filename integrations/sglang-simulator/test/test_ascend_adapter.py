# SPDX-License-Identifier: Apache-2.0
# Modified for AISimulate multi-accelerator workload replay.
# Upstream attribution is retained in LICENSE and source-manifest.json.
"""Contract tests for choosing a source runtime without altering upstream defaults."""

import json
import sys
from dataclasses import dataclass

import pytest

from sglang_simulator.compat import (
    SGLangCompatibilityError,
    simulator_parallel_args,
    validate_simulator_server_args,
)
from sglang_simulator.simulation.sglang.runtime_adapter import (
    request_prefix_length,
    select_source_runtime,
    server_arg_values,
)


def config(monkeypatch, tmp_path, flavor, **scheduler):
    path = tmp_path / "sim.json"
    path.write_text(json.dumps({"scheduler": scheduler}))
    monkeypatch.setenv("SGLANG_SIMULATOR_CONFIG_PATH", str(path))
    monkeypatch.setenv("SGLANG_SIMULATOR_RUNTIME", flavor)
    return path


def test_upstream_attention_dp_mapping_is_unchanged(monkeypatch, tmp_path):
    config(monkeypatch, tmp_path, "upstream", tp_size=8, dp_size=4, ep_size=8)
    args = simulator_parallel_args()
    assert args["attn_dp_size"] == 4
    assert args["dp_size"] == 1
    assert args["tp_size"] == args["ep_size"] == 8


def test_ascend_uses_native_dataclass_fields(monkeypatch, tmp_path):
    config(monkeypatch, tmp_path, "ascend", tp_size=1, dp_size=1, ep_size=1)
    args = simulator_parallel_args()
    assert args["dp_size"] == 1
    assert args["enable_dp_attention"] is False
    assert "attn_dp_size" not in args


@pytest.mark.parametrize("scheduler", [{"tp_size": 3, "dp_size": 2}, {"pp_size": 2}, {"cp_size": 2}])
def test_unsupported_ascend_topology_fails_early(monkeypatch, tmp_path, scheduler):
    config(monkeypatch, tmp_path, "ascend", **scheduler)
    with pytest.raises(SGLangCompatibilityError):
        simulator_parallel_args()


def test_ascend_attention_dp_maps_to_legacy_dp(monkeypatch, tmp_path):
    config(monkeypatch, tmp_path, "ascend", tp_size=4, dp_size=2, ep_size=2)
    args = simulator_parallel_args()
    assert args["tp_size"] == 4
    assert args["dp_size"] == 2
    assert args["enable_dp_attention"] is True
    assert "attn_dp_size" not in args


def test_server_arg_representations():
    @dataclass
    class Legacy:
        tp_size: int = 1

    class Current:
        def resolved_dict(self):
            return {"tp_size": 8}

    assert server_arg_values(Legacy()) == {"tp_size": 1}
    assert server_arg_values(Current()) == {"tp_size": 8}


def test_prefix_length_in_both_runtimes():
    class Legacy:
        prefix_indices = [1, 2, 3]

    class Current:
        prefix_len = 4

    assert request_prefix_length(Legacy()) == 3
    assert request_prefix_length(Current()) == 4


def test_invalid_explicit_source_does_not_fall_back(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_SIMULATOR_SGLANG_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="SGLang source not found"):
        select_source_runtime()


def test_source_selection_rejects_already_loaded_other_checkout(monkeypatch, tmp_path):
    from types import SimpleNamespace

    source = tmp_path / "python/sglang/srt/server_args.py"
    source.parent.mkdir(parents=True)
    source.touch()
    monkeypatch.setenv("SGLANG_SIMULATOR_SGLANG_ROOT", str(tmp_path))
    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(__file__="/another/sglang/__init__.py"))
    with pytest.raises(RuntimeError, match="Already imported SGLang"):
        select_source_runtime()
