"""Tests for Endorsement and AmbitionGain bid terms (both must be MODULAR)."""

import pytest

from auction.ambition import ambition_gain
from auction.endorsement import endorsement_scores
from auction.proposal import Proposal


def _p(pid, proposer, achs=("collect_wood",)):
    return Proposal(pid, proposer, "task_0", "d", "r", frozenset(achs))


# ---------------------------------------------------------------------------
# Endorsement
# ---------------------------------------------------------------------------

def test_endorsement_mean_of_others():
    props = [_p("a", "qwen"), _p("b", "deepseek")]
    ratings = {
        "qwen": {"b": 0.8},        # qwen rates b
        "deepseek": {"a": 0.4},    # deepseek rates a
    }
    s = endorsement_scores(props, ratings)
    assert s["a"] == pytest.approx(0.4)
    assert s["b"] == pytest.approx(0.8)


def test_endorsement_excludes_self():
    props = [_p("a", "qwen")]
    ratings = {"qwen": {"a": 1.0}}  # only a self-vote exists
    s = endorsement_scores(props, ratings, exclude_self=True)
    assert s["a"] == 0.0  # self-vote ignored, no others -> 0


def test_endorsement_averages_multiple_voters():
    props = [_p("a", "qwen")]
    ratings = {"deepseek": {"a": 0.6}, "glm": {"a": 1.0}}
    s = endorsement_scores(props, ratings)
    assert s["a"] == pytest.approx(0.8)


def test_endorsement_rejects_out_of_range_vote():
    props = [_p("a", "qwen")]
    with pytest.raises(ValueError):
        endorsement_scores(props, {"deepseek": {"a": 1.5}})


def test_endorsement_is_modular():
    # Modular = a proposal's score is independent of which other proposals are present.
    a = _p("a", "qwen")
    b = _p("b", "deepseek")
    c = _p("c", "glm")
    ratings = {
        "qwen": {"b": 0.5, "c": 0.7},
        "deepseek": {"a": 0.3, "c": 0.9},
        "glm": {"a": 0.1, "b": 0.6},
    }
    s_all = endorsement_scores([a, b, c], ratings)
    s_sub = endorsement_scores([a], ratings)
    assert s_all["a"] == pytest.approx(s_sub["a"])  # a's score unchanged by presence of b,c


# ---------------------------------------------------------------------------
# AmbitionGain
# ---------------------------------------------------------------------------

def test_ambition_zero_when_student_aces():
    p = _p("a", "qwen", achs=("collect_wood",))
    assert ambition_gain(p, {"collect_wood": 0.0}) == 0.0  # gap 0 -> no ambition


def test_ambition_high_for_deep_unmastered():
    shallow = _p("s", "qwen", achs=("collect_wood",))     # depth 1
    deep = _p("d", "qwen", achs=("defeat_necromancer",))  # depth 4
    gap = {"collect_wood": 1.0, "defeat_necromancer": 1.0}
    # same full gap, but deep achievement weighted by depth 4 vs depth 1
    assert ambition_gain(deep, gap) > ambition_gain(shallow, gap)
    assert ambition_gain(deep, gap) == pytest.approx(4.0)
    assert ambition_gain(shallow, gap) == pytest.approx(1.0)


def test_ambition_missing_gap_is_zero():
    p = _p("a", "qwen", achs=("collect_wood", "defeat_archer"))
    # only collect_wood has a gap entry; defeat_archer missing -> treated as gap 0
    assert ambition_gain(p, {"collect_wood": 0.5}) == pytest.approx(0.5)  # 0.5 * depth1


def test_ambition_no_depth_weight():
    deep = _p("d", "qwen", achs=("defeat_necromancer",))
    assert ambition_gain(deep, {"defeat_necromancer": 1.0}, use_depth_weight=False) == 1.0


def test_ambition_rejects_bad_gap():
    p = _p("a", "qwen")
    with pytest.raises(ValueError):
        ambition_gain(p, {"collect_wood": 1.5})
    with pytest.raises(ValueError):
        ambition_gain(p, {"bogus": 0.5})
