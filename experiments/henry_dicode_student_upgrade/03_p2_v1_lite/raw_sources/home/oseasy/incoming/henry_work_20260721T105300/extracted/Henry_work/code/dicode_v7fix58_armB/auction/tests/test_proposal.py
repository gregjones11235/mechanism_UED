"""Tests for the Proposal / SelfReport data contract."""

import pytest

from auction.proposal import Proposal, SelfReport


def _p(**kw):
    base = dict(
        proposal_id="p1",
        proposer_id="qwen",
        parent_task_id="task_0",
        docstring="build a torch then mine deep",
        reasoning="...",
        achievements=frozenset({"make_torch", "collect_iron"}),
    )
    base.update(kw)
    return Proposal(**base)


def test_valid_proposal_constructs():
    p = _p()
    assert p.proposal_id == "p1"
    assert p.is_v2 is False
    assert "make_torch" in p.achievements


def test_unknown_achievement_rejected():
    with pytest.raises(ValueError, match="unknown achievement"):
        _p(achievements=frozenset({"make_torch", "not_a_real_achievement"}))


def test_empty_ids_rejected():
    with pytest.raises(ValueError):
        _p(proposal_id="")
    with pytest.raises(ValueError):
        _p(proposer_id="")


def test_empty_achievements_is_allowed():
    # A proposal that teaches nothing in the ground-truth set is legal (Coverage gives it 0).
    p = _p(achievements=frozenset())
    assert p.achievements == frozenset()


def test_self_report_makes_v2():
    sr = SelfReport(
        claimed_achievements=frozenset({"make_torch"}),
        confidence=0.8,
        claimed_learning_gain=1.2,
    )
    p = _p(self_report=sr)
    assert p.is_v2 is True


def test_self_report_validation():
    with pytest.raises(ValueError, match=r"in \[0,1\]"):
        SelfReport(frozenset({"make_torch"}), confidence=1.5, claimed_learning_gain=0.0)
    with pytest.raises(ValueError, match="learning_gain"):
        SelfReport(frozenset({"make_torch"}), confidence=0.5, claimed_learning_gain=-1.0)
    with pytest.raises(ValueError, match="unknown names"):
        SelfReport(frozenset({"bogus"}), confidence=0.5, claimed_learning_gain=0.0)


def test_frozen_immutable():
    p = _p()
    with pytest.raises(Exception):
        p.proposal_id = "x"  # type: ignore[misc]
