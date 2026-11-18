import json
from pathlib import Path

import pytest
from collector.sglang.ascend_deepep import LEGACY_REVISION, NATIVE_REVISION, validate_runtime
from collector.sglang.ascend_deepep_cases import get_ascend_deepep_cases

pytestmark = pytest.mark.unit


def test_dispatcher_runtime_guards_preserve_audited_phase_contracts():
    validate_runtime(LEGACY_REVISION, "normal")
    validate_runtime(LEGACY_REVISION, "low_latency")
    validate_runtime(NATIVE_REVISION, "low_latency")
    with pytest.raises(ValueError, match="payload-precision audit"):
        validate_runtime(NATIVE_REVISION, "normal")
    with pytest.raises(ValueError, match="Reaudit DeepEP"):
        validate_runtime("unknown-runtime", "low_latency")


def test_transport_plan_preserves_placement_and_serving_branches(tmp_path):
    config = Path(__file__).resolve().parents[3] / "src/aisimulate_core/model_configs/zai-org--GLM-5.2_config.json"
    (tmp_path / "config.json").write_bytes(config.read_bytes())
    quant = tmp_path / "quant_model_description.json"
    quant.write_text(json.dumps({"model.layers.3.mlp.experts.0.gate_proj.weight": "W8A8_DYNAMIC"}))
    plan = get_ascend_deepep_cases(tmp_path)
    cases = plan["cases"]
    assert len(cases) == len({json.dumps(case, sort_keys=True) for case in cases})
    assert {(case["num_ranks"], case["ranks_per_node"]) for case in cases} == {(16, 16), (32, 16)}
    assert {case["execution_mode"] for case in cases if case["phase"] == "normal"} == {"eager"}
    assert {case["execution_mode"] for case in cases if case["phase"] == "low_latency"} == {"eager", "graph"}
    assert max(case["tokens_per_rank"] for case in cases if case["phase"] == "low_latency") == 32
    quant.write_text(json.dumps({"model.layers.3.mlp.experts.0.gate_proj.weight": "FLOAT"}))
    with pytest.raises(ValueError, match="dynamic W8A8"):
        get_ascend_deepep_cases(tmp_path)
