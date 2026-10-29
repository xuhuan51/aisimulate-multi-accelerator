import copy
import hashlib
import json

import pytest
from collector.sglang.ascend_curve_append import (
    append_curves,
    collection_identities,
    extend_attention_points,
    extend_points,
)

pytestmark = pytest.mark.unit


def campaign(width, latency, protocol):
    identity = {
        "runtime": {"version": "0.5.22", "hardware": "Ascend910_9382"},
        "protocol": protocol,
        "serving_setup": {"environment": {}},
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "report": {"identity": identity},
        "metadata": {"runtime": identity["runtime"]},
        "rows": [
            {
                "role": "swiglu",
                "dimensions": json.dumps([width]),
                "execution_mode": "graph",
                "tokens": tokens,
                "latency_ms": latency,
                "runtime_identity_sha256": digest,
            }
            for tokens in (8, 16)
        ],
    }


def test_disjoint_curves_keep_original_timings_and_collection_protocols():
    first, second = campaign(128, 2.0, {"iterations": 8}), campaign(256, 1.0, {"iterations": 16})
    rows, curves, counts = append_curves("vector", [first, second])
    assert rows == first["rows"] + second["rows"]
    assert len(curves) == 2 and counts == {0: 2, 1: 2}
    assert curves[0]["collection_identity_sha256"] != curves[1]["collection_identity_sha256"]


def test_overlapping_curve_cannot_replace_slower_points():
    with pytest.raises(ValueError, match="Overlapping curves"):
        append_curves("vector", [campaign(128, 2.0, {}), campaign(128, 1.0, {})])


def test_vector_extension_preserves_existing_points_and_rejects_protocol_changes():
    first = campaign(128, 2.0, {"iterations": 16})
    added = campaign(128, 4.0, {"iterations": 16})
    for row in added["rows"]:
        row["tokens"] *= 2048
    rows, _, counts, _ = extend_points("vector", [first, added])
    assert [row["latency_ms"] for row in rows] == [2.0, 2.0, 4.0, 4.0]
    assert counts == {0: 2, 1: 2}
    different = campaign(128, 4.0, {"iterations": 8})
    for row in different["rows"]:
        row["tokens"] *= 2048
    with pytest.raises(ValueError, match="identical runtime, protocol"):
        extend_points("vector", [first, different])


def test_unverified_identity_and_different_serving_setup_are_rejected():
    first, second = campaign(128, 2.0, {}), campaign(256, 1.0, {})
    invalid = copy.deepcopy(second)
    invalid["rows"][0]["runtime_identity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="verified collection"):
        append_curves("vector", [first, invalid])
    invalid = copy.deepcopy(second)
    invalid["report"]["identity"]["serving_setup"]["environment"]["different"] = "1"
    with pytest.raises(ValueError, match="serving setups"):
        append_curves("vector", [first, invalid])


def test_attention_grid_keeps_distinct_padding_and_prefix_coordinates():
    source = campaign(128, 2.0, {})
    source["rows"] = [
        {
            "operation": "sparse_attention",
            "phase": "decode",
            "local_heads": 32,
            "execution_mode": "graph",
            "active_batch": active,
            "padded_batch": padded,
            "sequence_length": 8192,
            "prefix": 0,
            "latency_ms": 1.0,
            "runtime_identity_sha256": source["rows"][0]["runtime_identity_sha256"],
        }
        for active, padded in [(1, 8), (1, 16), (2, 8)]
    ]
    rows, grids, counts = append_curves("attention", [source])
    assert len(rows) == 3 and len(grids) == 1 and counts == {0: 3}
    duplicate = copy.deepcopy(source)
    duplicate["rows"].append(dict(duplicate["rows"][0]))
    with pytest.raises(ValueError, match="duplicate points"):
        append_curves("attention", [duplicate])


def attention_campaign(length, plan, protocol=None):
    source = campaign(128, float(length), protocol or {"iterations": 8})
    identity = source["report"]["identity"]
    identity["plan_sha256"] = plan
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    source["rows"] = [
        {
            "operation": "sparse_attention",
            "phase": "decode",
            "local_heads": 8,
            "execution_mode": "graph",
            "active_batch": 1,
            "padded_batch": 8,
            "sequence_length": length,
            "prefix": 0,
            "latency_ms": float(length),
            "source_sha256": plan,
            "runtime_identity_sha256": digest,
        }
    ]
    return source


def test_attention_extension_preserves_measurements_and_records_composite_provenance():
    old, added = attention_campaign(8192, "old"), attention_campaign(32768, "new")
    rows, curves, counts, identities = extend_attention_points([old, added])
    assert [(row["latency_ms"], row["source_sha256"]) for row in rows] == [(8192.0, "old"), (32768.0, "new")]
    assert counts == {0: 1, 1: 1}
    assert len(curves) == 1 and len(curves[0]["sources"]) == 2
    assert len({row["runtime_identity_sha256"] for row in rows}) == 1
    report = {"merged_identities": identities, "campaigns": [{"report": old["report"]}, {"report": added["report"]}]}
    assert rows[0]["runtime_identity_sha256"] in collection_identities(report)
    merged = {"rows": rows, "report": report, "metadata": old["metadata"]}
    extended, _, counts, _ = extend_attention_points([merged, attention_campaign(33792, "next")])
    assert len(extended) == 3 and counts == {0: 2, 1: 1}


def deepep_campaign(tokens, plan, mode="eager", protocol=None):
    source = attention_campaign(8192, plan, protocol)
    digest = source["rows"][0]["runtime_identity_sha256"]
    source["rows"] = [
        {
            "phase": "low_latency",
            "operation": "combine",
            "num_ranks": 32,
            "ranks_per_node": 16,
            "distribution": "balanced",
            "execution_mode": mode,
            "tokens_per_rank": tokens,
            "latency_ms": tokens / 10,
            "source_sha256": plan,
            "runtime_identity_sha256": digest,
        }
    ]
    return source


def test_deepep_extension_keeps_one_identity_across_unchanged_and_extended_curves():
    old = deepep_campaign(1, "old")
    old["rows"].append({**old["rows"][0], "execution_mode": "graph", "latency_ms": 0.05})
    added = deepep_campaign(32, "new")
    rows, curves, counts, identities = extend_points("deepep", [old, added])
    assert [
        (row["execution_mode"], row["tokens_per_rank"], row["latency_ms"], row["source_sha256"]) for row in rows
    ] == [
        ("eager", 1, 0.1, "old"),
        ("eager", 32, 3.2, "new"),
        ("graph", 1, 0.05, "old"),
    ]
    assert counts == {0: 2, 1: 1}
    assert len({row["runtime_identity_sha256"] for row in rows}) == 1
    assert len({curve["collection_identity_sha256"] for curve in curves}) == 1
    merged = {
        "rows": rows,
        "metadata": old["metadata"],
        "report": {
            "merged_identities": identities,
            "campaigns": [{"report": old["report"]}, {"report": added["report"]}],
        },
    }
    extended, _, counts, _ = extend_points("deepep", [merged, deepep_campaign(16, "next")])
    assert len(extended) == 4 and counts == {0: 3, 1: 1}
    assert len({row["runtime_identity_sha256"] for row in extended}) == 1


def test_deepep_extension_rejects_overlaps_and_cross_curve_protocol_changes():
    old = deepep_campaign(1, "old")
    with pytest.raises(ValueError, match="Overlapping physical points"):
        extend_points("deepep", [old, deepep_campaign(1, "new")])
    with pytest.raises(ValueError, match="identical runtime, protocol"):
        extend_points("deepep", [old, deepep_campaign(1, "new", mode="graph", protocol={"iterations": 16})])


def test_attention_extension_rejects_overlaps_and_changed_protocols():
    old = attention_campaign(8192, "old")
    with pytest.raises(ValueError, match="Overlapping physical points"):
        extend_attention_points([old, attention_campaign(8192, "new")])
    with pytest.raises(ValueError, match="identical runtime, protocol"):
        extend_attention_points([old, attention_campaign(32768, "new", {"iterations": 16})])
    changed = attention_campaign(32768, "new")
    changed["report"]["identity"]["protocol"]["iterations"] = 16
    with pytest.raises(ValueError, match="verified collection identity"):
        extend_attention_points([old, changed])
