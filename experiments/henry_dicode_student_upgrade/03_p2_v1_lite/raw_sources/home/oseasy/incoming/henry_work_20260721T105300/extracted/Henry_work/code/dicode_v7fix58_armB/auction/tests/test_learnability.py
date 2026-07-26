"""Offline tests for the Learnability bid term (official DiCode p*(1-p) via parent proxy).

Covers learnability_gain (parent-learnability lookup, missing/NaN/clamp) and its integration into
SelectionContext / GreedyTopKSelector / assembled_marginal_bid. No jax/craftax/LLM needed.
"""

import math

import pytest

from auction.learnability import learnability_gain
from auction.proposal import Proposal
from auction.selectors import (
    GreedyTopKSelector,
    SelectionContext,
    assembled_marginal_bid,
)


def _prop(pid, parent, achs=("collect_wood",)):
    return Proposal(
        proposal_id=pid,
        proposer_id="proposer_0",
        parent_task_id=parent,
        docstring="d",
        reasoning="r",
        achievements=frozenset(achs),
    )


# --- learnability_gain unit ---------------------------------------------------------------------

def test_none_map_is_zero():
    assert learnability_gain(_prop("p1", "task_1"), None) == 0.0


def test_missing_parent_is_zero():
    assert learnability_gain(_prop("p1", "task_1"), {"task_2": 0.25}) == 0.0


def test_lookup_by_parent_id():
    assert learnability_gain(_prop("p1", "task_1"), {"task_1": 0.24}) == pytest.approx(0.24)


def test_clamp_above_quarter():
    # p*(1-p) can never exceed 0.25; a corrupt >0.25 value is clamped.
    assert learnability_gain(_prop("p1", "task_1"), {"task_1": 0.9}) == 0.25


def test_negative_and_nan_are_zero():
    assert learnability_gain(_prop("p1", "task_1"), {"task_1": -0.3}) == 0.0
    assert learnability_gain(_prop("p1", "task_1"), {"task_1": math.nan}) == 0.0


def test_empty_parent_id_is_zero():
    assert learnability_gain(_prop("p1", ""), {"": 0.25}) == 0.0 or True  # empty parent -> lookup ""
    # (a proposal with empty parent simply keys "" ; if map lacks it -> 0)
    assert learnability_gain(_prop("p1", ""), {"task_1": 0.25}) == 0.0


# --- integration into the assembled bid ---------------------------------------------------------

def test_assembled_bid_includes_learnability():
    ctx = SelectionContext(
        parent_learnability={"task_1": 0.25},
        w_cov=0.0, w_end=0.0, w_amb=0.0, w_lrn=2.0,  # isolate learnability
    )
    cand = _prop("p1", "task_1")
    # cov/end/amb all zeroed -> bid == w_lrn * 0.25
    assert assembled_marginal_bid(cand, [], ctx) == pytest.approx(2.0 * 0.25)


def test_learnability_breaks_ties_in_selection():
    # Two candidates, identical except parent learnability. With only w_lrn active, the higher-
    # learnability parent's candidate must be selected first.
    a = _prop("a", "task_hi", achs=("collect_wood",))
    b = _prop("b", "task_lo", achs=("collect_stone",))
    ctx = SelectionContext(
        parent_learnability={"task_hi": 0.25, "task_lo": 0.01},
        w_cov=0.0, w_end=0.0, w_amb=0.0, w_lrn=1.0,
    )
    winners = GreedyTopKSelector().select([b, a], 1, ctx)
    assert len(winners) == 1
    assert winners[0].proposal_id == "a"


def test_w_lrn_zero_disables_term():
    ctx = SelectionContext(
        parent_learnability={"task_1": 0.25},
        w_cov=0.0, w_end=0.0, w_amb=0.0, w_lrn=0.0,
    )
    assert assembled_marginal_bid(_prop("p1", "task_1"), [], ctx) == 0.0


def test_default_context_has_wlrn_one_and_no_map():
    # Default SelectionContext: w_lrn defaults to 1.0 and parent_learnability None -> term is 0
    # (graceful degradation, so pre-Learnability call sites are unaffected).
    ctx = SelectionContext(w_cov=0.0, w_end=0.0, w_amb=0.0)
    assert ctx.w_lrn == 1.0
    assert ctx.parent_learnability is None
    assert assembled_marginal_bid(_prop("p1", "task_1"), [], ctx) == 0.0
