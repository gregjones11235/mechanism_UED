"""GATE 7 — rank_percentile_v1 normalization: bounded [0,1], raw preserved,
deterministic ties, per-role independence, no empty/non-finite."""
import pytest

from d052.normalization import (
    NormalizationError,
    normalize_role_matrix,
    normalized_map,
    rank_percentile_v1,
)
from d052.schemas.roles import ScoringRole


def test_single_candidate_is_neutral():
    col = rank_percentile_v1(ScoringRole.TUTOR, [("a", 0.7)])
    assert col.entries[0].normalized == 0.5


def test_min_max_and_monotonic():
    col = rank_percentile_v1(ScoringRole.TUTOR, [("a", 1.0), ("b", 2.0), ("c", 3.0)])
    m = {e.candidate_id: e.normalized for e in col.entries}
    assert m["a"] == 0.0
    assert m["c"] == 1.0
    assert m["a"] < m["b"] < m["c"]


def test_ties_share_value_rank_and_group():
    col = rank_percentile_v1(
        ScoringRole.EXPLORER, [("a", 1.0), ("b", 2.0), ("c", 2.0), ("d", 3.0)])
    e = {x.candidate_id: x for x in col.entries}
    # n=4: a=0/3, b=c=1/3, d=3/3
    assert e["a"].normalized == 0.0
    assert e["b"].normalized == e["c"].normalized == pytest.approx(1 / 3)
    assert e["d"].normalized == 1.0
    assert e["b"].rank == e["c"].rank == 1
    assert e["b"].tie_group == e["c"].tie_group == 1
    assert e["a"].tie_group == 0 and e["d"].tie_group == 2


def test_deterministic_under_shuffle():
    base = [("a", 1.0), ("b", 2.0), ("c", 2.0), ("d", 3.0), ("e", 0.5)]
    col1 = rank_percentile_v1(ScoringRole.TUTOR, base)
    col2 = rank_percentile_v1(ScoringRole.TUTOR, list(reversed(base)))
    assert normalized_map(col1) == normalized_map(col2)


def test_raw_preserved_and_bounded():
    raw = [("a", -5.0), ("b", 0.0), ("c", 100.0)]
    col = rank_percentile_v1(ScoringRole.CRITIC, raw)
    by_id = {e.candidate_id: e for e in col.entries}
    assert by_id["a"].raw == -5.0 and by_id["c"].raw == 100.0
    for e in col.entries:
        assert 0.0 <= e.normalized <= 1.0


def test_all_tied_is_deterministic():
    col = rank_percentile_v1(ScoringRole.TUTOR, [("a", 1.0), ("b", 1.0), ("c", 1.0)])
    vals = {e.normalized for e in col.entries}
    assert len(vals) == 1  # all identical -> deterministic single value
    assert all(e.rank == 0 for e in col.entries)


def test_per_role_independence():
    matrix = {
        ScoringRole.TUTOR: [("a", 1.0), ("b", 2.0)],
        ScoringRole.EXPLORER: [("a", 100.0), ("b", 0.0)],  # opposite ordering
    }
    out = normalize_role_matrix(matrix)
    t = normalized_map(out[ScoringRole.TUTOR])
    x = normalized_map(out[ScoringRole.EXPLORER])
    assert t["a"] < t["b"]
    assert x["a"] > x["b"]  # normalized independently per role


def test_empty_column_rejected():
    with pytest.raises(NormalizationError) as ei:
        rank_percentile_v1(ScoringRole.TUTOR, [])
    assert ei.value.code == NormalizationError.EMPTY_COLUMN


def test_non_finite_rejected():
    with pytest.raises(NormalizationError) as ei:
        rank_percentile_v1(ScoringRole.TUTOR, [("a", float("nan"))])
    assert ei.value.code == NormalizationError.NON_FINITE


def test_normalization_method_label():
    col = rank_percentile_v1(ScoringRole.TUTOR, [("a", 1.0)])
    assert col.normalization == "rank_percentile_v1"
