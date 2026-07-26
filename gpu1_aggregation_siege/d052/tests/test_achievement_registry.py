"""GATE 3 — official Craftax-67 achievement registry.

Verifies:
  * exactly 67 achievements, canonical ids 0..66
  * canonical_id == goal_vector_index (multi-hot index alignment)
  * known anchor values (first 5, defeat_kobold=41, last 5)
  * DRIFT DETECTOR: the committed official_achievements.json matches the canonical
    source module (dicode_src/auction/craftax_achievements.py) EXACTLY
  * explicit alias allow-list (defeat_orc_soldier -> defeat_orc_solider / id 38)
  * fail-closed: unknown target -> error; empty goal -> error; case-sensitive
"""
import importlib.util
import json
import os

import pytest

from d052.achievements import (
    ACHIEVEMENT_SCHEMA,
    NUM_ACHIEVEMENTS,
    REGISTRY,
    AchievementError,
)

# repo root = 3 levels up from this test file (d052/tests/.. -> gpu1 -> repo root)
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SRC = os.path.join(_REPO_ROOT, "dicode_src", "auction", "craftax_achievements.py")
_REGISTRY_JSON = os.path.join(
    os.path.dirname(__file__), "..", "achievements", "official_achievements.json")


def _load_source_module():
    spec = importlib.util.spec_from_file_location("_src_ach", _SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_count_and_id_range():
    assert NUM_ACHIEVEMENTS == 67
    assert REGISTRY.count == 67
    assert len(REGISTRY.names) == 67
    ids = sorted(REGISTRY.canonical_id(n) for n in REGISTRY.names)
    assert ids == list(range(67))


def test_anchor_values():
    anchors = {
        "collect_wood": 0, "place_table": 1, "eat_cow": 2,
        "collect_sapling": 3, "collect_drink": 4,
        "defeat_kobold": 41,
        "drink_potion": 62, "enchant_sword": 63, "enchant_armour": 64,
        "defeat_knight": 65, "defeat_archer": 66,
    }
    for name, cid in anchors.items():
        assert REGISTRY.canonical_id(name) == cid, name


def test_canonical_id_equals_goal_vector_index():
    # Setting a single target must light exactly the bit at its canonical_id.
    for name in list(REGISTRY.names)[:10] + ["defeat_kobold", "defeat_archer"]:
        vec = REGISTRY.to_goal_vector([name])
        cid = REGISTRY.canonical_id(name)
        assert vec[cid] == 1.0
        assert sum(vec) == 1.0
        assert len(vec) == 67


def test_drift_detector_committed_json_matches_source_exactly():
    """The committed registry MUST equal the canonical source. Any silent edit to
    either side fails this gate."""
    src = _load_source_module()
    src_pairs = sorted((n, v) for n, v in src._ACHIEVEMENTS_ORDERED)

    committed = json.load(open(_REGISTRY_JSON, encoding="utf-8"))
    committed_pairs = sorted(
        (e["name"], e["canonical_id"]) for e in committed["achievements"])

    assert committed_pairs == src_pairs, (
        "committed official_achievements.json drifted from canonical source")
    assert committed["num_achievements"] == src.NUM_ACHIEVEMENTS == 67
    assert committed["achievement_schema"] == ACHIEVEMENT_SCHEMA
    # every committed goal_vector_index equals its canonical_id
    for e in committed["achievements"]:
        assert e["goal_vector_index"] == e["canonical_id"]
    # source provenance points at the canonical single source
    assert committed["source"]["path"].endswith("auction/craftax_achievements.py")
    assert committed["source"]["git_blob_sha1"]  # recorded
    assert committed["source"]["environment_version"]["craftax"] == "1.4.5"


def test_source_blob_sha_is_the_canonical_one():
    committed = json.load(open(_REGISTRY_JSON, encoding="utf-8"))
    # blob sha of dicode_src/auction/craftax_achievements.py at baseline a2726e3
    assert committed["source"]["git_blob_sha1"].startswith("5bb881a6")


def test_explicit_alias_resolves():
    # correct English spelling -> canonical misspelling, same id 38
    assert REGISTRY.resolve("defeat_orc_soldier") == "defeat_orc_solider"
    assert REGISTRY.canonical_id("defeat_orc_solider") == 38
    vec = REGISTRY.to_goal_vector(["defeat_orc_soldier"])
    assert vec[38] == 1.0 and sum(vec) == 1.0


def test_unknown_target_is_an_error():
    with pytest.raises(AchievementError) as ei:
        REGISTRY.resolve("defeat_dragon")
    assert ei.value.code == AchievementError.UNKNOWN_ACHIEVEMENT
    with pytest.raises(AchievementError):
        REGISTRY.to_goal_vector(["collect_wood", "defeat_dragon"])


def test_empty_goal_is_an_error():
    with pytest.raises(AchievementError) as ei:
        REGISTRY.to_goal_vector([])
    assert ei.value.code == AchievementError.EMPTY_GOAL_SET
    with pytest.raises(AchievementError):
        REGISTRY.canonicalize_targets([])


@pytest.mark.parametrize("bad", ["Collect_Wood", "COLLECT_WOOD", "collect_wood ",
                                 " collect_wood", 0, None, ["collect_wood"]])
def test_matching_is_case_sensitive_and_exact(bad):
    # NO case-folding / NO trimming coercion at the schema boundary.
    with pytest.raises(AchievementError):
        REGISTRY.resolve(bad)


def test_goal_vector_deterministic_and_dedup():
    a = REGISTRY.to_goal_vector(["collect_wood", "defeat_archer"])
    b = REGISTRY.to_goal_vector(["defeat_archer", "collect_wood"])
    c = REGISTRY.to_goal_vector(["collect_wood", "collect_wood", "defeat_archer"])
    assert a == b == c
    assert len(a) == 67
    assert sum(a) == 2.0
    assert a[0] == 1.0 and a[66] == 1.0


def test_canonicalize_returns_sorted_canonical_names():
    out = REGISTRY.canonicalize_targets(["defeat_archer", "collect_wood"])
    assert out == ["collect_wood", "defeat_archer"]  # sorted by canonical_id


def test_design_layer_metadata_present():
    assert REGISTRY.depth_tier("collect_wood") == 1
    assert REGISTRY.depth_tier("defeat_archer") == 4
    assert REGISTRY.family("defeat_archer") == "COMBAT"
    assert REGISTRY.family("collect_wood") == "GATHER"
