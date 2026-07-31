"""CC1 audit fix1 §8 — alpha_front is structurally < 1.

The global-regret component weight (1 - alpha_front) must ALWAYS be strictly
positive: a GLOBAL-scope UED method can never degenerate into a front-only
scorer. Enforced at three layers:

  1. schema: EnvironmentScoreBundle.alpha_front uses lt=1.0 (strict);
  2. runtime: assert_alpha_front_bounds() — three invariants checked on
     every dry run;
  3. structural proof: the Soft Copeland global component channel can never
     be annihilated (weight > 0, coefficient > 0).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from d052.bagr_ued import constants as C
from d052.bagr_ued.controller import ALPHA_FRONT, assert_alpha_front_bounds
from d052.bagr_ued.soft_copeland import (
    EnvironmentScoreBundle,
    soft_copeland_rank,
)


def _bundle(env_id, *, alpha=0.5, front=0.4, glob=0.6, gap=0.2):
    return EnvironmentScoreBundle(
        environment_id=env_id,
        front_regret=front,
        global_regret=glob,
        behavioral_gap=gap,
        learning_progress=0.3,
        learnability=0.4,
        diversity=0.5,
        global_retention=0.6,
        critic_penalty=0.0,
        alpha_front=alpha)


# ---------------------------------------------------------------------------
# layer 1 — schema: strict upper bound
# ---------------------------------------------------------------------------

def test_schema_rejects_alpha_front_equal_one():
    with pytest.raises(ValidationError):
        _bundle("e1", alpha=1.0)


def test_schema_rejects_alpha_front_above_one():
    with pytest.raises(ValidationError):
        _bundle("e1", alpha=1.5)


def test_schema_rejects_negative_alpha_front():
    with pytest.raises(ValidationError):
        _bundle("e1", alpha=-0.1)


def test_schema_accepts_default_and_interior_values():
    assert _bundle("e1", alpha=0.5).alpha_front == 0.5
    assert _bundle("e1", alpha=0.999).alpha_front == 0.999
    assert _bundle("e1", alpha=0.0).alpha_front == 0.0


# ---------------------------------------------------------------------------
# layer 2 — runtime assertions
# ---------------------------------------------------------------------------

def test_runtime_assert_passes_on_defaults():
    # module default ALPHA_FRONT and the auditable constant window
    assert_alpha_front_bounds()
    assert ALPHA_FRONT == 0.5
    assert C.ALPHA_FRONT_MIN == 0.0
    assert C.ALPHA_FRONT_MAX == 0.75
    assert 0.0 <= ALPHA_FRONT <= C.ALPHA_FRONT_MAX < 1.0


def test_runtime_assert_rejects_alpha_front_one():
    with pytest.raises(AssertionError, match="ALPHA_FRONT_OUT_OF_BOUNDS"):
        assert_alpha_front_bounds(alpha_front=1.0)


def test_runtime_assert_rejects_alpha_max_one():
    with pytest.raises(AssertionError, match="ALPHA_WINDOW_OUT_OF_BOUNDS"):
        assert_alpha_front_bounds(alpha_min=0.0, alpha_max=1.0)


def test_runtime_assert_rejects_inverted_window():
    with pytest.raises(AssertionError, match="ALPHA_WINDOW_OUT_OF_BOUNDS"):
        assert_alpha_front_bounds(alpha_min=0.8, alpha_max=0.7)


# ---------------------------------------------------------------------------
# layer 3 — structural proof: the global component can never vanish
#
# CC3 fix2 contract update (documented in the fix2 report): the criterion-
# wise Soft Copeland no longer carries a static "one_minus_alpha_global_
# regret" weight entry — the global pairwise weight is resolved AT THE
# PAIRWISE LEVEL as (1 - alpha_front) from the run-level alpha. The three
# structural proofs below are the new-contract equivalents: the global
# criterion always exists, always has strictly positive weight, and always
# contributes (neutrally on a constant column, decisively on a varying one).
# ---------------------------------------------------------------------------

def test_global_component_coefficient_always_strictly_positive():
    # for every schema-admissible alpha_front, (1 - alpha) > 0
    for alpha in (0.0, 0.25, 0.5, 0.75, 0.9, 0.999):
        b = _bundle("e", alpha=alpha)
        assert (1.0 - b.alpha_front) > 0.0


def test_global_pairwise_weight_is_strictly_positive_for_every_alpha():
    for alpha in (0.0, 0.25, 0.5, 0.75, 0.999):
        ranking = soft_copeland_rank([_bundle("a", alpha=alpha),
                                      _bundle("b", alpha=alpha)])
        assert ranking.dimension_weights["front_regret"] == \
            pytest.approx(alpha)
        assert ranking.dimension_weights["global_regret"] == \
            pytest.approx(1.0 - alpha)
        assert ranking.dimension_weights["global_regret"] > 0.0


def test_global_component_nonzero_on_constant_column():
    # degenerate (constant) global_regret -> normalized 0.5 each (neutral
    # pairwise contribution) while the criterion keeps strictly positive
    # pairwise weight (1 - alpha) — the channel exists and is weighted
    ranking = soft_copeland_rank([_bundle("a", alpha=0.99, glob=0.7),
                                  _bundle("b", alpha=0.99, glob=0.7)])
    for e in ranking.entries:
        assert e.components["global_regret"] == pytest.approx(0.5)
    assert ranking.dimension_weights["global_regret"] == pytest.approx(0.01)
    assert ranking.dimension_weights["global_regret"] > 0.0


def test_global_component_channel_alive_on_varying_column():
    # varying global_regret: the criterion produces a NON-NEUTRAL pairwise
    # preference — with front and all other dimensions equal, the candidate
    # with the higher global regret wins the comparison
    ranking = soft_copeland_rank([_bundle("a", alpha=0.5, glob=0.1),
                                  _bundle("b", alpha=0.5, glob=0.9)])
    ia = ranking.environment_order.index("a")
    ib = ranking.environment_order.index("b")
    assert ranking.pairwise_preference[ib][ia] > 0.5
    top = next(e for e in ranking.entries if e.rank == 1)
    assert top.environment_id == "b"
