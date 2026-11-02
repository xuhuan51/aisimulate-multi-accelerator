import json
from pathlib import Path

import pytest
from collector.sglang.ascend_vector_cases import get_ascend_vector_cases

pytestmark = pytest.mark.unit


def test_declared_vector_geometry_preserves_parallelism_and_precision_boundaries(tmp_path):
    config = Path(__file__).resolve().parents[3] / "src/aisimulate_core/model_configs/zai-org--GLM-5.2_config.json"
    (tmp_path / "config.json").write_bytes(config.read_bytes())
    (tmp_path / "quant_model_description.json").write_text(
        json.dumps({"model.layers.0.self_attn.q_a_proj.weight": "W8A8"})
    )
    plan = get_ascend_vector_cases(tmp_path)
    cases = plan["cases"]
    assert len(cases) == len({json.dumps(case, sort_keys=True) for case in cases})

    def shapes(role):
        return {tuple(case["dimensions"]) for case in cases if case["role"] == role}

    assert shapes("swiglu") == {(64,), (128,), (256,), (384,), (512,), (768,), (1536,), (2048,), (3072,)}
    assert shapes("rope_attention") == {(heads, 64, 256, 64) for heads in (4, 8, 16, 32, 64)}
    assert shapes("rope_indexer") == {(32, 64, 128, 128)}
    assert shapes("embedding") == {(vocabulary, 6144) for vocabulary in (154880, 77440, 38720, 19360, 9680)}
    assert shapes("cache_scatter") == {(53376, 64), (53376, 128), (53376, 512)}
    assert shapes("router_topk") == {(256, 8)}
    assert shapes("logits_finalize") == {(parallel, 154880) for parallel in (1, 2, 4, 8, 16)}
    assert max(case["tokens"] for case in cases if case["role"] == "logits_finalize") == 32
    transfers = [case for case in cases if case["role"] == "seq_lens_h2d"]
    assert {case["execution_mode"] for case in transfers} == {"eager"}
    assert {tuple(case["dimensions"]) for case in transfers} == {(4,)}
    assert {case["tokens"] for case in transfers} == {1, 2, 3, 4, 8, 16, 24, 32}
    altered = json.loads(config.read_text())
    altered["rms_norm_eps"] = 1e-6
    (tmp_path / "config.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="rms_norm_eps"):
        get_ascend_vector_cases(tmp_path)
