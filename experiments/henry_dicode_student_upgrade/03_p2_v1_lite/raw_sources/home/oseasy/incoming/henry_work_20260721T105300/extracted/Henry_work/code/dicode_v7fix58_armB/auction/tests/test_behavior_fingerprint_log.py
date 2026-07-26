"""Offline tests for BehaviorFingerprintLog (v6 problem-2, 2026-07-05).

Pure-python: no jax/craftax. Verifies cross-session accumulation, the relative-SR guard, name/action
remap, resume idempotency, and the movement-folded render. Mirrors test_cooccurrence_log.py's shape.
"""
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_AUCTION = os.path.dirname(_HERE)
if _AUCTION not in sys.path:
    sys.path.insert(0, _AUCTION)

from behavior_fingerprint_log import (  # noqa: E402
    ACTION_NAMES,
    MIN_SR,
    NUM_ACTIONS,
    BehaviorFingerprintLog,
)
from craftax_achievements import ACHIEVEMENT_TO_VALUE, NUM_ACHIEVEMENTS  # noqa: E402


def _zeros_act():
    return [[0.0] * NUM_ACTIONS for _ in range(NUM_ACHIEVEMENTS)]


def _idx(name):
    return ACHIEVEMENT_TO_VALUE[name.lower()]


def _a(name):
    return ACTION_NAMES.index(name)


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "behav.json")


def test_empty_fingerprint_is_none(path):
    log = BehaviorFingerprintLog(path)
    assert log.fingerprint("make_iron_pickaxe") is None
    assert log.render_fingerprint_hint("make_iron_pickaxe") == ""


def test_basic_accumulate_and_mean(path):
    log = BehaviorFingerprintLog(path)
    act = _zeros_act()
    steps = [0.0] * NUM_ACHIEVEMENTS
    reached = [0] * NUM_ACHIEVEMENTS
    deep = _idx("make_iron_pickaxe")
    # 10 winning episodes reached iron pickaxe; across them: DO used 300x, PLACE_STONE 20x,
    # MAKE_IRON_PICKAXE 10x; total 840 steps.
    act[deep][_a("DO")] = 300.0
    act[deep][_a("PLACE_STONE")] = 20.0
    act[deep][_a("MAKE_IRON_PICKAXE")] = 10.0
    steps[deep] = 840.0
    reached[deep] = 10
    log.add_session(1, act, steps, reached, total=1000)  # SR = 10/1000 = 1%... below MIN_SR 3%
    assert log.fingerprint("make_iron_pickaxe") is None  # guarded: too rare

    # add another session so SR clears 3%.
    log.add_session(2, act, steps, reached, total=200)  # now n=20, total=1200 -> SR 1.67%, still < 3%
    assert log.fingerprint("make_iron_pickaxe") is None
    log.add_session(3, act, steps, reached, total=50)   # n=30, total=1250 -> 2.4%, still <3%
    log.add_session(4, act, steps, reached, total=50)   # n=40, total=1300 -> 3.08% >= 3%
    fp = log.fingerprint("make_iron_pickaxe")
    assert fp is not None
    assert fp["n"] == 40
    # mean per winning episode: DO 300/10=30 per session-batch; accumulated 4x => 1200 DO / 40 ep = 30.
    assert fp["actions"]["DO"] == pytest.approx(30.0, abs=0.1)
    assert fp["actions"]["PLACE_STONE"] == pytest.approx(2.0, abs=0.1)
    assert fp["actions"]["MAKE_IRON_PICKAXE"] == pytest.approx(1.0, abs=0.1)
    assert fp["avg_len"] == pytest.approx(84.0, abs=0.1)  # 3360 steps / 40 ep


def test_movement_folded_into_move_frac(path):
    log = BehaviorFingerprintLog(path)
    act = _zeros_act()
    steps = [0.0] * NUM_ACHIEVEMENTS
    reached = [0] * NUM_ACHIEVEMENTS
    deep = _idx("collect_wood")
    # a single winning episode: 60 LEFT/RIGHT movement + 30 DO + 10 PLACE_STONE = 100 actions.
    act[deep][_a("LEFT")] = 30.0
    act[deep][_a("RIGHT")] = 30.0
    act[deep][_a("DO")] = 30.0
    act[deep][_a("PLACE_STONE")] = 10.0
    steps[deep] = 100.0
    reached[deep] = 100  # 100 winning eps of a common skill
    log.add_session(1, act, steps, reached, total=1000)  # SR 10% >= 3%
    fp = log.fingerprint("collect_wood")
    assert fp is not None
    # movement (LEFT+RIGHT = 60) folded into move_frac; DO/PLACE_STONE stay in actions.
    assert fp["move_frac"] == pytest.approx(0.60, abs=0.01)
    assert "LEFT" not in fp["actions"] and "RIGHT" not in fp["actions"]
    assert "DO" in fp["actions"] and "PLACE_STONE" in fp["actions"]


def test_render_hint_text(path):
    log = BehaviorFingerprintLog(path)
    act = _zeros_act()
    steps = [0.0] * NUM_ACHIEVEMENTS
    reached = [0] * NUM_ACHIEVEMENTS
    deep = _idx("defeat_skeleton")
    act[deep][_a("DO")] = 500.0
    act[deep][_a("PLACE_TORCH")] = 40.0
    steps[deep] = 700.0
    reached[deep] = 50
    log.add_session(1, act, steps, reached, total=1000)  # SR 5%
    hint = log.render_fingerprint_hint("defeat_skeleton")
    assert "defeat_skeleton" in hint
    assert "50 winning episodes" in hint
    assert "PLACE_TORCH" in hint and "DO" in hint
    # v7fix5.0: counts are labelled as PRESSES (incl. failed attempts) — the s207 misdiagnosis
    # read winners' craft-press spam as "winners build diamond gear" while the craft SR sat at 3-5%.
    assert "ACTION PRESSES" in hint and "NOT a successful craft" in hint


def test_name_and_action_remap(path):
    """Eval hands rows/cols in ITS OWN order + labels; add_session remaps into canonical order."""
    log = BehaviorFingerprintLog(path)
    # a tiny 2-achievement, 3-action source in a scrambled order.
    src_ach = ["make_iron_pickaxe", "collect_wood"]
    src_actions = ["PLACE_STONE", "DO", "MAKE_IRON_PICKAXE"]
    act = [[5.0, 100.0, 2.0], [1.0, 50.0, 0.0]]  # rows = src_ach, cols = src_actions
    steps = [200.0, 120.0]
    reached = [40, 200]
    log.add_session(1, act, steps, reached, names=src_ach, action_names=src_actions, total=1000)
    fp = log.fingerprint("make_iron_pickaxe")
    assert fp is not None
    # DO: 100/40 = 2.5 per ep; PLACE_STONE 5/40 = 0.125; MAKE_IRON_PICKAXE 2/40 = 0.05
    assert fp["actions"]["DO"] == pytest.approx(2.5, abs=0.01)
    assert fp["actions"]["PLACE_STONE"] == pytest.approx(0.125, abs=0.01)


def test_unknown_names_skipped(path):
    log = BehaviorFingerprintLog(path)
    act = [[10.0], [20.0]]
    steps = [50.0, 60.0]
    reached = [40, 40]
    # first row a real skill, second an unknown name (dropped); single real action.
    log.add_session(
        1, act, steps, reached,
        names=["make_iron_pickaxe", "not_a_real_skill"],
        action_names=["DO"], total=1000,
    )
    assert log.support("make_iron_pickaxe") == 40
    fp = log.fingerprint("make_iron_pickaxe")
    assert fp["actions"]["DO"] == pytest.approx(0.25, abs=0.01)  # 10/40


def test_resume_idempotent(path):
    log = BehaviorFingerprintLog(path)
    act = _zeros_act()
    act[_idx("make_iron_pickaxe")][_a("DO")] = 400.0
    steps = [0.0] * NUM_ACHIEVEMENTS
    steps[_idx("make_iron_pickaxe")] = 800.0
    reached = [0] * NUM_ACHIEVEMENTS
    reached[_idx("make_iron_pickaxe")] = 40
    log.add_session(5, act, steps, reached, total=1000)
    log.add_session(5, act, steps, reached, total=1000)  # same session -> skipped
    assert log.support("make_iron_pickaxe") == 40  # not doubled

    # reload from disk -> state survives resume.
    log2 = BehaviorFingerprintLog(path)
    assert log2.support("make_iron_pickaxe") == 40
    fp = log2.fingerprint("make_iron_pickaxe")
    assert fp["actions"]["DO"] == pytest.approx(10.0, abs=0.1)


def test_corrupt_file_falls_back_empty(path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("{ not valid json")
    log = BehaviorFingerprintLog(path)
    assert log.total_finished() == 0
    assert log.fingerprint("make_iron_pickaxe") is None


def test_stale_shape_file_falls_back_empty(path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"act": [[0.0, 0.0]], "steps": [0.0], "n": [0], "total": 5}, f)  # wrong shapes
    log = BehaviorFingerprintLog(path)
    assert log.total_finished() == 0  # rejected, not crashed


def test_action_names_length():
    assert NUM_ACTIONS == 43  # craftax action_dim, pinned so an eval-side mismatch is caught
    assert ACTION_NAMES[0] == "NOOP" and ACTION_NAMES[13] == "MAKE_IRON_PICKAXE"
