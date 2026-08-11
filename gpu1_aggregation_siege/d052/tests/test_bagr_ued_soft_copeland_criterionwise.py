"""CC3 audit fix2 §1-§3 — criterion-wise Soft Copeland + alpha effectiveness.

fix2 rewrote Soft Copeland because the v1 implementation (a) multiplied
alpha_front into the RAW front/global regret BEFORE min-max normalization
(any uniform positive alpha is canceled by normalization and cannot move the
FRONT/GLOBAL preference) and (b) ranked a single weighted-sum "strength"
pairwise (= weighted-sum ranking, not criterion-wise Copeland).

This file proves the NEW contract:

  §1  each RAW dimension is normalized independently with NO alpha before
      normalization; pairwise preferences are computed criterion by criterion
      (sigmoid soft comparison with per-dimension temperatures); dimension
      weights are applied AT THE PAIRWISE LEVEL with front_weight=alpha_front
      and global_weight=1-alpha_front; NO pre-aggregated strength drives the
      ranking; the full audit trail (normalization provenance, weights,
      temperatures, pairwise preference + margin matrices, Copeland scores,
      tie-break provenance) is recorded and hash-bound.
  §2  alpha_front ACTUALLY MOVES the ranking: a FRONT-strong candidate wins
      at alpha=0.75 and LOSES at alpha=0.25 against a GLOBAL-strong candidate
      (pairwise preference flips, rank flips).
  §3  boundary cases A-H (I/J live in test_bagr_ued_legality_filtering.py):
      A criterion conflict resolved pairwise, B alpha flip, C constant
      dimension neutral, D complete tie stable, E permutation invariance,
      F NaN/Inf fail closed, G duplicate ids fail closed, H missing
      criterion fail closed.
"""
from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from d052.bagr_ued.soft_copeland import (
    DIMENSION_WEIGHTS,
    RAW_DIMENSIONS,
    TIE_BREAK_RULE,
    EnvironmentScoreBundle,
    soft_copeland_rank,
)


def _bundle(env_id, *, alpha=0.5, front=0.4, glob=0.6, gap=0.2, lp=0.3,
            learn=0.4, div=0.5, ret=0.6, penalty=0.0):
    return EnvironmentScoreBundle(
        environment_id=env_id,
        front_regret=front,
        global_regret=glob,
        behavioral_gap=gap,
        learning_progress=lp,
        learnability=learn,
        diversity=div,
        global_retention=ret,
        critic_penalty=penalty,
        alpha_front=alpha)


def _idx(ranking, env_id):
    return ranking.environment_order.index(env_id)


# ===========================================================================
# §1 — structure & audit trail of the criterion-wise ranking
# ===========================================================================

def test_no_alpha_before_normalization_provenance_records_raw_minmax():
    # front_regret raw values 0.1/0.9 at alpha=0.25 must normalize to 0.0/1.0
    # over the RAW dimension — if alpha were premultiplied (v1 bug) the
    # recorded min/max would be 0.025/0.225 instead
    r = soft_copeland_rank([_bundle("a", alpha=0.25, front=0.1),
                            _bundle("b", alpha=0.25, front=0.9)])
    prov = r.normalization_provenance["front_regret"]
    assert prov["min"] == pytest.approx(0.1)
    assert prov["max"] == pytest.approx(0.9)
    ia, ib = _idx(r, "a"), _idx(r, "b")
    assert prov["normalized"][ia] == pytest.approx(0.0)
    assert prov["normalized"][ib] == pytest.approx(1.0)
    # the entry components carry the NORMALIZED dimension values (no alpha
    # premultiplied) — a's normalized front_regret is 0.0, not 0.25*0.1
    comp_a = next(e for e in r.entries if e.environment_id == "a").components
    assert comp_a["front_regret"] == pytest.approx(0.0)


def test_criterion_wise_pairwise_matrices_and_weights_recorded():
    r = soft_copeland_rank([_bundle("a", alpha=0.6), _bundle("b", alpha=0.6)])
    # candidate-id-aligned matrices
    assert r.environment_order == ["a", "b"]
    assert len(r.pairwise_preference) == 2 and \
        len(r.pairwise_preference[0]) == 2
    assert len(r.pairwise_margin) == 2
    # diagonal is the neutral preference; margins are preference - 0.5
    for i in range(2):
        assert r.pairwise_preference[i][i] == pytest.approx(0.5)
        for j in range(2):
            assert r.pairwise_margin[i][j] == pytest.approx(
                r.pairwise_preference[i][j] - 0.5)
    # antisymmetry of the soft preference around 0.5
    assert r.pairwise_preference[0][1] + r.pairwise_preference[1][0] == \
        pytest.approx(1.0)
    # weights applied AT the pairwise level: front=alpha, global=1-alpha
    assert r.alpha_front_used == pytest.approx(0.6)
    assert r.dimension_weights["front_regret"] == pytest.approx(0.6)
    assert r.dimension_weights["global_regret"] == pytest.approx(0.4)
    assert r.dimension_weights["behavioral_gap"] == \
        DIMENSION_WEIGHTS["behavioral_gap"]
    # temperatures + tie-break provenance recorded and hash-bound
    assert set(r.dimension_temperatures) == set(RAW_DIMENSIONS)
    assert r.tie_break_rule == TIE_BREAK_RULE
    assert len(r.ranking_hash) == 64
    # NO pre-aggregated strength drives the entries
    for e in r.entries:
        assert "strength" not in e.components
        assert not hasattr(e, "strength") or "strength" not in \
            e.model_fields_set


def test_copeland_score_is_mean_pairwise_preference():
    r = soft_copeland_rank([_bundle("a", front=0.9, glob=0.9),
                            _bundle("b", front=0.5, glob=0.5),
                            _bundle("c", front=0.1, glob=0.1)])
    for e in r.entries:
        i = _idx(r, e.environment_id)
        n = len(r.environment_order)
        mean_pref = sum(r.pairwise_preference[i][j] for j in range(n)
                        if j != i) / (n - 1)
        assert e.copeland_score == pytest.approx(mean_pref, abs=1e-6)
        margin_sum = sum(r.pairwise_margin[i][j] for j in range(n)
                         if j != i)
        assert e.copeland_margin_sum == pytest.approx(margin_sum, abs=1e-6)
        assert e.copeland_score == pytest.approx(
            0.5 + e.copeland_margin_sum / (n - 1), abs=1e-6)
    assert [e.rank for e in r.entries] == [1, 2, 3]


def test_all_eight_raw_dimensions_are_normalized_independently():
    r = soft_copeland_rank([_bundle("a"), _bundle("b")])
    assert set(r.normalization_provenance) == set(RAW_DIMENSIONS)
    assert len(RAW_DIMENSIONS) == 8
    for dim in RAW_DIMENSIONS:
        prov = r.normalization_provenance[dim]
        assert {"min", "max", "constant", "normalized"} <= set(prov)


# ===========================================================================
# §2 — alpha_front ACTUALLY MOVES the ranking (the fix1 bug it must not have)
# ===========================================================================

def _front_vs_global_pair(alpha):
    # A: strong FRONT, weak GLOBAL. B: the mirror image. Every OTHER
    # dimension is IDENTICAL (constant column -> neutral 0.5 pairwise
    # contribution), so the FRONT/GLOBAL criteria alone decide.
    a = _bundle("A", alpha=alpha, front=1.0, glob=0.0)
    b = _bundle("B", alpha=alpha, front=0.0, glob=1.0)
    return soft_copeland_rank([a, b])


def test_alpha_075_prefers_front_strong_candidate():
    r = _front_vs_global_pair(0.75)
    ia, ib = _idx(r, "A"), _idx(r, "B")
    assert r.pairwise_preference[ia][ib] > 0.5
    rank = {e.environment_id: e.rank for e in r.entries}
    assert rank["A"] < rank["B"]


def test_alpha_025_prefers_global_strong_candidate():
    r = _front_vs_global_pair(0.25)
    ia, ib = _idx(r, "A"), _idx(r, "B")
    assert r.pairwise_preference[ia][ib] < 0.5
    rank = {e.environment_id: e.rank for e in r.entries}
    assert rank["B"] < rank["A"]


def test_alpha_changes_pairwise_preference_and_ranking():
    hi = _front_vs_global_pair(0.75)
    lo = _front_vs_global_pair(0.25)
    ia_hi, ib_hi = _idx(hi, "A"), _idx(hi, "B")
    ia_lo, ib_lo = _idx(lo, "A"), _idx(lo, "B")
    p_hi = hi.pairwise_preference[ia_hi][ib_hi]
    p_lo = lo.pairwise_preference[ia_lo][ib_lo]
    # the pairwise preference literally crosses 0.5 between the two alphas
    assert p_hi > 0.5 > p_lo
    assert p_hi != pytest.approx(p_lo)
    # ... and the ranking flips with it
    order_hi = [e.environment_id for e in hi.entries]
    order_lo = [e.environment_id for e in lo.entries]
    assert order_hi == ["A", "B"]
    assert order_lo == ["B", "A"]
    assert hi.ranking_hash != lo.ranking_hash


def test_alpha_effect_not_canceled_by_normalization():
    # the exact fix1 failure mode: under the v1 math, premultiplying alpha
    # before min-max leaves the normalized columns (and therefore the
    # ranking) IDENTICAL for any uniform positive alpha. Prove that is gone:
    # the normalized front/global columns are alpha-INDEPENDENT ...
    hi = _front_vs_global_pair(0.75)
    lo = _front_vs_global_pair(0.25)
    assert hi.normalization_provenance["front_regret"]["normalized"] == \
        lo.normalization_provenance["front_regret"]["normalized"]
    # ... while the ranking IS alpha-dependent (alpha enters at the pairwise
    # weight level instead)
    assert [e.environment_id for e in hi.entries] != \
        [e.environment_id for e in lo.entries]


# ===========================================================================
# §3 — boundary cases A-H
# ===========================================================================

def test_case_a_criterion_conflict_resolved_pairwise_not_presynthesized():
    # A dominates on front_regret, B dominates on global_regret, all else
    # equal: the conflict is settled by the WEIGHTED pairwise preferences of
    # the two criteria (alpha=0.7 -> front-leaning -> A), not by comparing a
    # pre-synthesized strength scalar
    a = _bundle("A", alpha=0.7, front=0.95, glob=0.05)
    b = _bundle("B", alpha=0.7, front=0.05, glob=0.95)
    r = soft_copeland_rank([a, b])
    ia, ib = _idx(r, "A"), _idx(r, "B")
    p = r.pairwise_preference[ia][ib]
    # front and global criteria push in OPPOSITE directions (neither 0 nor 1)
    assert 0.5 < p < 1.0
    assert next(e for e in r.entries if e.rank == 1).environment_id == "A"


def test_case_b_alpha_preference_flip():
    # the §3-B mirror of §2: preference flips between alpha 0.25 and 0.75
    hi = _front_vs_global_pair(0.75)
    lo = _front_vs_global_pair(0.25)
    ia_hi, ib_hi = _idx(hi, "A"), _idx(hi, "B")
    ia_lo, ib_lo = _idx(lo, "A"), _idx(lo, "B")
    assert hi.pairwise_preference[ia_hi][ib_hi] > 0.5
    assert lo.pairwise_preference[ia_lo][ib_lo] < 0.5


def test_case_c_constant_dimension_neutral_no_order_bias():
    # a constant dimension normalizes to 0.5 for everyone and contributes the
    # neutral pairwise preference 0.5 — it cannot bias the order
    bundles = [_bundle(f"e{i}", div=0.7) for i in range(4)]   # diversity const
    r = soft_copeland_rank(bundles)
    prov = r.normalization_provenance["diversity"]
    assert prov["constant"] is True
    assert all(v == pytest.approx(0.5) for v in prov["normalized"])
    # two candidates identical on every varying dimension stay tied at 0.5
    twin = [_bundle("x", front=0.5, glob=0.5), _bundle("y", front=0.5,
                                                       glob=0.5)]
    rt = soft_copeland_rank(twin)
    ix, iy = _idx(rt, "x"), _idx(rt, "y")
    assert rt.pairwise_preference[ix][iy] == pytest.approx(0.5)


def test_case_d_complete_tie_is_stable_and_deterministic():
    twins = [_bundle(f"t{i}", front=0.5, glob=0.5, gap=0.3) for i in range(3)]
    r1 = soft_copeland_rank(twins)
    r2 = soft_copeland_rank(list(reversed(twins)))
    assert r1.ranking_hash == r2.ranking_hash
    for e in r1.entries:
        assert e.copeland_score == pytest.approx(0.5)
    # deterministic tie-break: copeland_score DESC, environment_id ASC
    assert [e.environment_id for e in r1.entries] == ["t0", "t1", "t2"]
    assert [e.rank for e in r1.entries] == [1, 2, 3]


def test_case_e_permutation_invariance_id_aligned_matrices():
    bundles = [_bundle(f"env{i}", front=0.1 * i, glob=0.9 - 0.1 * i,
                       gap=0.05 * i, lp=0.1 * i) for i in range(6)]
    import random
    shuffled = bundles[:]
    random.Random(11).shuffle(shuffled)
    r1 = soft_copeland_rank(bundles)
    r2 = soft_copeland_rank(shuffled)
    assert r1.ranking_hash == r2.ranking_hash
    assert r1.environment_order == r2.environment_order == \
        [f"env{i}" for i in range(6)]
    assert r1.pairwise_preference == r2.pairwise_preference
    assert [e.environment_id for e in r1.entries] == \
        [e.environment_id for e in r2.entries]


def test_case_f_nan_inf_fail_closed():
    # layer 1 — the schema refuses non-finite criteria at construction
    with pytest.raises(ValidationError):
        _bundle("bad", front=float("nan"))
    with pytest.raises(ValidationError):
        _bundle("bad", glob=float("inf"))

    # layer 2 — defense in depth: the rank function itself refuses
    # non-finite inputs even on bundles smuggled past the schema
    # (duck-typed, attribute-access only)
    from types import SimpleNamespace

    def fake(eid, front=0.5):
        return SimpleNamespace(
            environment_id=eid, front_regret=front, global_regret=0.5,
            behavioral_gap=0.5, learning_progress=0.5, learnability=0.5,
            diversity=0.5, global_retention=0.5, critic_penalty=0.0,
            alpha_front=0.5)

    with pytest.raises(ValueError, match="NON_FINITE_SCORE_INPUT"):
        soft_copeland_rank([fake("ok"), fake("smuggled",
                                             front=float("nan"))])


def test_case_g_duplicate_candidate_ids_fail_closed():
    with pytest.raises(ValueError, match="DUPLICATE_ENVIRONMENT_SCORES"):
        soft_copeland_rank([_bundle("same"), _bundle("same")])


def test_case_h_missing_criterion_fail_closed():
    # every criterion is a required field — an incomplete bundle cannot be
    # constructed, so no criterion can silently drop out of the comparison
    with pytest.raises(ValidationError):
        EnvironmentScoreBundle(environment_id="incomplete",
                               front_regret=0.5,
                               # global_regret + the rest missing
                               alpha_front=0.5)
    with pytest.raises(ValueError, match="EMPTY_ENVIRONMENT_SCORES"):
        soft_copeland_rank([])


def test_non_uniform_alpha_fails_closed():
    with pytest.raises(ValueError, match="ALPHA_FRONT_NOT_UNIFORM"):
        soft_copeland_rank([_bundle("a", alpha=0.3), _bundle("b", alpha=0.6)])


def test_critic_penalty_is_lower_is_better_criterion():
    # identical on every other dimension: the LOWER penalty wins the pairwise
    # comparison (sign-flipped criterion, weight applied at pairwise level)
    clean = _bundle("clean", penalty=0.0)
    penalized = _bundle("pen", penalty=0.9)
    r = soft_copeland_rank([clean, penalized])
    ic, ip = _idx(r, "clean"), _idx(r, "pen")
    assert r.pairwise_preference[ic][ip] > 0.5
    assert next(e for e in r.entries if e.rank == 1).environment_id == "clean"
