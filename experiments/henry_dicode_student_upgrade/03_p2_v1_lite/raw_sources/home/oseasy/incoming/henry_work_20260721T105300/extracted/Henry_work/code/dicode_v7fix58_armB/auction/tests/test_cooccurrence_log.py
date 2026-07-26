"""Offline tests for CooccurrenceLog (v6 §3.8 (c) co-occurrence signal).

Pins: cross-session accumulation (never averages away), resume idempotency, the RELATIVE-SR guard
(MIN_SR, user 2026-07-05 — replaces the old absolute MIN_SUPPORT count), prereq frequency =
cooc[deep][j]/count[deep], the accumulated SR denominator (total_finished), and the prompt hint text.
"""

import json

import pytest

from auction.cooccurrence_log import MIN_SR, CooccurrenceLog
from auction.craftax_achievements import ACHIEVEMENT_TO_VALUE, NUM_ACHIEVEMENTS


def _zeros_mat():
    n = NUM_ACHIEVEMENTS
    return [[0] * n for _ in range(n)]


def _zeros_vec():
    return [0] * NUM_ACHIEVEMENTS


def _idx(name):
    return ACHIEVEMENT_TO_VALUE[name]


def _episode_multihot(names):
    """A single successful episode reaching `names`: contributes to count[i] and cooc[i][j] for all
    pairs. Returns (count_vec, cooc_mat) for ONE episode so tests can add several."""
    vec = _zeros_vec()
    mat = _zeros_mat()
    idxs = [_idx(n) for n in names]
    for i in idxs:
        vec[i] += 1
        for j in idxs:
            mat[i][j] += 1
    return vec, mat


def _add_episodes(log, session, episodes, total=None):
    """episodes = list of name-lists; accumulate them into one session's arrays and add.

    ``total`` = the session's FINISHED-episode denominator (SR guard). Default None -> use len(episodes)
    so the deep skill's empirical SR is ~100% (every finished episode succeeded), which keeps the
    frequency/accumulation tests above the MIN_SR guard. SR-guard tests pass an explicit large total.
    """
    vec = _zeros_vec()
    mat = _zeros_mat()
    for names in episodes:
        v, m = _episode_multihot(names)
        for i in range(NUM_ACHIEVEMENTS):
            vec[i] += v[i]
            for j in range(NUM_ACHIEVEMENTS):
                mat[i][j] += m[i][j]
    log.add_session(session, vec, mat, total=(len(episodes) if total is None else total))


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "cooc.json")


def test_empty_log(path):
    log = CooccurrenceLog(path)
    assert log.support("defeat_gnome_warrior") == 0
    assert log.total_finished() == 0
    assert log.prereq_frequencies("defeat_gnome_warrior") == {}


def test_basic_cooccurrence_frequency(path):
    log = CooccurrenceLog(path)
    # 10 episodes reach defeat_gnome_warrior; in 9 of them place_torch too, in 5 collect_diamond.
    episodes = []
    for k in range(10):
        names = ["defeat_gnome_warrior"]
        if k < 9:
            names.append("place_torch")
        if k < 5:
            names.append("collect_diamond")
        episodes.append(names)
    _add_episodes(log, 1, episodes)  # total defaults to 10 -> deep SR 100% >> MIN_SR

    assert log.support("defeat_gnome_warrior") == 10
    freqs = log.prereq_frequencies("defeat_gnome_warrior")
    assert abs(freqs["place_torch"] - 0.9) < 1e-9
    assert abs(freqs["collect_diamond"] - 0.5) < 1e-9
    # sorted high->low
    assert list(freqs.keys())[0] == "place_torch"
    # the deep skill itself is excluded
    assert "defeat_gnome_warrior" not in freqs


def test_cross_session_accumulation(path):
    log = CooccurrenceLog(path)
    _add_episodes(log, 1, [["defeat_troll", "place_torch"]] * 3)
    _add_episodes(log, 2, [["defeat_troll"]] * 3)  # 3 more troll wins, no torch
    # support accumulates: 6 troll episodes total
    assert log.support("defeat_troll") == 6
    # torch co-reached in 3 of 6 -> 0.5
    assert abs(log.prereq_frequencies("defeat_troll", min_freq=0.0)["place_torch"] - 0.5) < 1e-9


def test_total_finished_accumulates(path):
    log = CooccurrenceLog(path)
    _add_episodes(log, 1, [["defeat_troll"]] * 3, total=1000)
    _add_episodes(log, 2, [["defeat_troll"]] * 3, total=1000)
    assert log.total_finished() == 2000
    # deep SR = 6 troll wins / 2000 finished = 0.3%  -> below MIN_SR (3%) -> guarded empty
    assert log.deep_sr("defeat_troll") < MIN_SR
    assert log.prereq_frequencies("defeat_troll") == {}


def test_resume_idempotent_session(path):
    log = CooccurrenceLog(path)
    _add_episodes(log, 5, [["defeat_troll", "collect_iron"]] * 4, total=100)
    # re-adding session 5 must NOT double-count (neither counts nor the total denominator)
    _add_episodes(log, 5, [["defeat_troll", "collect_iron"]] * 4, total=100)
    assert log.support("defeat_troll") == 4
    assert log.total_finished() == 100


def test_persistence_roundtrip(path):
    log = CooccurrenceLog(path)
    _add_episodes(log, 1, [["defeat_troll", "place_torch"]] * 7, total=100)
    log2 = CooccurrenceLog(path)  # reopen from disk
    assert log2.support("defeat_troll") == 7
    assert log2.total_finished() == 100  # denominator persisted + reloaded
    assert abs(log2.prereq_frequencies("defeat_troll")["place_torch"] - 1.0) < 1e-9


def test_relative_sr_guard(path):
    """The guard is RELATIVE (count[deep]/total >= MIN_SR), not an absolute count."""
    log = CooccurrenceLog(path)
    # 20 deep wins but a huge denominator -> SR 2% < MIN_SR(3%) -> guarded empty despite 20 successes.
    _add_episodes(log, 1, [["defeat_necromancer", "make_diamond_sword"]] * 20, total=1000)
    assert log.support("defeat_necromancer") == 20  # NOT sparse in absolute terms
    assert log.deep_sr("defeat_necromancer") == pytest.approx(0.02)
    assert log.prereq_frequencies("defeat_necromancer") == {}  # but SR too low -> guarded
    # add a session where the same skill is solved much more often -> SR crosses MIN_SR.
    _add_episodes(log, 2, [["defeat_necromancer", "make_diamond_sword"]] * 50, total=100)
    # now 70 wins / 1100 finished = 6.4% >= 3%
    assert log.deep_sr("defeat_necromancer") >= MIN_SR
    assert log.prereq_frequencies("defeat_necromancer") != {}


def test_guard_empty_without_denominator(path):
    """A record added WITHOUT a total (older caller) has no denominator -> SR 0 -> guarded empty."""
    log = CooccurrenceLog(path)
    vec, mat = _zeros_vec(), _zeros_mat()
    for names in [["defeat_troll", "place_torch"]] * 10:
        v, m = _episode_multihot(names)
        for i in range(NUM_ACHIEVEMENTS):
            vec[i] += v[i]
            for j in range(NUM_ACHIEVEMENTS):
                mat[i][j] += m[i][j]
    log.add_session(1, vec, mat)  # no total= -> _total stays 0
    assert log.support("defeat_troll") == 10
    assert log.total_finished() == 0
    assert log.deep_sr("defeat_troll") == 0.0
    assert log.prereq_frequencies("defeat_troll") == {}


def test_min_freq_filter(path):
    log = CooccurrenceLog(path)
    episodes = []
    for k in range(10):
        names = ["defeat_troll"]
        if k < 2:  # 20% co-occurrence
            names.append("collect_ruby")
        if k < 8:  # 80%
            names.append("place_torch")
        episodes.append(names)
    _add_episodes(log, 1, episodes)  # total=10 -> deep SR 100%, passes the guard
    freqs = log.prereq_frequencies("defeat_troll", min_freq=0.5)
    assert "place_torch" in freqs        # 0.8 >= 0.5
    assert "collect_ruby" not in freqs   # 0.2 < 0.5


def test_render_hint_text(path):
    log = CooccurrenceLog(path)
    _add_episodes(log, 1, [["defeat_gnome_warrior", "place_torch"]] * 10, total=20)
    hint = log.render_prereq_hint("defeat_gnome_warrior")
    assert "defeat_gnome_warrior" in hint
    assert "place_torch" in hint
    assert "100%" in hint          # torch co-occurrence
    assert "50% SR" in hint        # 10 wins / 20 finished = 50% empirical SR shown to the modeler


def test_render_hint_lift_excludes_universal(path):
    """v6fix9 P0.5-#1: a universal achievement (reached in ~every episode -> lift ~1) must not eat
    a prerequisite slot; a genuinely disproportionate skill (lift >= MIN_LIFT) stays."""
    log = CooccurrenceLog(path)
    episodes = [["defeat_gnome_warrior", "place_torch", "wake_up"]] * 10 + [["wake_up"]] * 10
    _add_episodes(log, 1, episodes, total=20)
    hint = log.render_prereq_hint("defeat_gnome_warrior")
    assert "place_torch" in hint   # 100% in wins vs 50% base -> lift 2.0, genuine prerequisite
    assert "wake_up" not in hint   # 100% in wins vs 100% base -> lift 1.0, base-rate noise
    assert "lift" in hint          # the hint teaches the lift semantics, not "High % = prereq"


def test_render_hint_empty_when_sr_too_low(path):
    log = CooccurrenceLog(path)
    # 10 wins but a large denominator -> 1% SR < MIN_SR(3%) -> hint suppressed, modeler uses mechanics.
    _add_episodes(log, 1, [["defeat_gnome_warrior", "place_torch"]] * 10, total=1000)
    assert log.render_prereq_hint("defeat_gnome_warrior") == ""


def test_corrupt_file_falls_back(path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not valid json")
    log = CooccurrenceLog(path)
    assert log.support("defeat_troll") == 0  # did not crash
    assert log.total_finished() == 0


def test_stale_shape_file_falls_back(path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"count": [1, 2, 3], "cooc": [[1]], "sessions": [1]}, f)  # wrong shape
    log = CooccurrenceLog(path)
    assert log.support("defeat_troll") == 0  # rejected, empty


def test_legacy_file_without_total_loads(path):
    """A pre-SR-guard file (no ``total`` key) must load, defaulting the denominator to 0."""
    n = NUM_ACHIEVEMENTS
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"count": [0] * n, "cooc": [[0] * n for _ in range(n)], "sessions": [1]}, f
        )
    log = CooccurrenceLog(path)
    assert log.total_finished() == 0  # missing key -> 0, no crash


def test_add_session_with_names_remap(path):
    # eval hands us small arrays in ITS OWN order + names; add_session remaps by name into canonical.
    log = CooccurrenceLog(path)
    names = ["defeat_gnome_warrior", "place_torch", "eat_cow"]
    # 6 successful episodes; every one reached all three -> count 6, cooc 6 per pair. total 6 -> SR 100%.
    count_vec = [6, 6, 6]
    cooc_mat = [[6, 6, 6], [6, 6, 6], [6, 6, 6]]
    log.add_session(1, count_vec, cooc_mat, names=names, total=6)
    assert log.support("defeat_gnome_warrior") == 6
    assert log.support("place_torch") == 6
    freqs = log.prereq_frequencies("defeat_gnome_warrior")
    assert abs(freqs["place_torch"] - 1.0) < 1e-9
    assert abs(freqs["eat_cow"] - 1.0) < 1e-9


def test_add_session_names_skips_unknown(path):
    log = CooccurrenceLog(path)
    names = ["defeat_troll", "not_a_real_achievement", "collect_iron"]
    count_vec = [5, 9, 5]
    cooc_mat = [[5, 4, 5], [4, 9, 4], [5, 4, 5]]
    log.add_session(1, count_vec, cooc_mat, names=names, total=5)  # unknown column dropped, no crash
    assert log.support("defeat_troll") == 5
    assert log.support("collect_iron") == 5
    freqs = log.prereq_frequencies("defeat_troll")
    assert abs(freqs["collect_iron"] - 1.0) < 1e-9
    assert "not_a_real_achievement" not in freqs
