"""Offline tests for ChainOrderLog (v6fix7 P2 — directed chain order + break-link mining).

Pins the temporal upgrade of (c):
  - directed 2-gram accumulation from time-sorted successful episodes + the greedy dominant path,
    guarded by the same relative MIN_SR rule as the co-occurrence hint;
  - 3-gram sparse counts persisted (paper/ablation material);
  - break-link mining on FAILING episodes w.r.t. a wall's ORDERED chain: modal deepest link, missing
    links, mean depth (= COUNT of achieved links, order-invariant);
  - the frontier-advance detector (mean depth must rise >= FRONTIER_ADVANCE_LINKS over the best
    previous comparable session, both sides with >= FRONTIER_MIN_FAILS fails, same link SET — a
    reorder of the tree must not reset the history) — the P1a patience / blacklist-escape signal;
  - persistence: resume idempotency, roundtrip, corrupt-file fallback, name remapping;
  - the SiegeNotebook side: chain_targets() (active foci + retired walls), note_chain_progress on a
    retired wall setting the registry flag, _has_new_evidence honouring it, and a fresh retirement
    clearing the stale flag while snapshotting the ordered chain.

No jax/craftax/LLM needed.
"""

import json

import pytest

from auction.chain_order_log import (
    FRONTIER_ADVANCE_LINKS,
    FRONTIER_MIN_FAILS,
    ChainOrderLog,
)
from auction.craftax_achievements import ACHIEVEMENT_TO_VALUE, NUM_ACHIEVEMENTS
from auction.siege_notebook import (
    LADDER_L4,
    MATURITY_MIN_SNAPSHOTS,
    SiegeNotebook,
)
from auction.tests.test_siege_notebook import _mature_profile


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "chain.json")


def _row(steps: dict[str, int]) -> list[int]:
    """One episode's first-step row in canonical order: {ach_name: step}, -1 elsewhere."""
    row = [-1] * NUM_ACHIEVEMENTS
    for name, step in steps.items():
        row[ACHIEVEMENT_TO_VALUE[name]] = int(step)
    return row


def _add(log, session, episodes, unfinished=0, chain_targets=None):
    """episodes = list of {name: step}; each is one FINISHED episode. ``unfinished`` appends that
    many horizon-truncated (finished=False) rows, which must be ignored entirely."""
    rows = [_row(ep) for ep in episodes] + [_row({"collect_wood": 0})] * unfinished
    finished = [True] * len(episodes) + [False] * unfinished
    log.add_session(session, rows, finished, chain_targets=chain_targets)


# ---- directed success chains --------------------------------------------------------------------

def test_empty_log(path):
    log = ChainOrderLog(path)
    assert log.support("defeat_troll") == 0
    assert log.total_finished() == 0
    assert log.dominant_path("defeat_troll") == []
    assert log.render_chain_hint("defeat_troll") == ""


def test_dominant_path_follows_time_order(path):
    log = ChainOrderLog(path)
    # 10 wins, always iron(1) -> sword(5) -> troll(9). SR 100% -> passes the MIN_SR guard.
    eps = [{"collect_iron": 1, "make_iron_sword": 5, "defeat_troll": 9}] * 10
    _add(log, 1, eps)
    assert log.support("defeat_troll") == 10
    path_ = log.dominant_path("defeat_troll")
    # shallow -> deep, each hop with its predecessor share
    assert [p[0] for p in path_] == ["collect_iron", "make_iron_sword"]
    assert all(abs(frac - 1.0) < 1e-9 for _, frac in path_)


def test_order_matters_not_just_cooccurrence(path):
    log = ChainOrderLog(path)
    # Same SET of skills but iron comes AFTER the troll kill -> iron is NOT a predecessor of troll.
    _add(log, 1, [{"defeat_troll": 2, "collect_iron": 8, "make_iron_sword": 1}] * 10)
    path_ = log.dominant_path("defeat_troll")
    assert [p[0] for p in path_] == ["make_iron_sword"]  # only the true predecessor


def test_dominant_path_guarded_below_min_sr(path):
    log = ChainOrderLog(path)
    # 10 wins drowned in 990 finished failures -> troll SR 1% < MIN_SR(3%) -> no trusted path.
    eps = [{"collect_iron": 1, "defeat_troll": 9}] * 10 + [{"collect_wood": 0}] * 990
    _add(log, 1, eps)
    assert log.support("defeat_troll") == 10
    assert log.dominant_path("defeat_troll") == []


def test_gram3_persisted(path):
    log = ChainOrderLog(path)
    _add(log, 1, [{"collect_iron": 1, "make_iron_sword": 5, "defeat_troll": 9}] * 7)
    data = json.load(open(path, encoding="utf-8"))
    assert data["gram3"]["collect_iron>make_iron_sword>defeat_troll"] == 7


def test_unfinished_episodes_ignored(path):
    log = ChainOrderLog(path)
    _add(log, 1, [{"collect_wood": 0}] * 3, unfinished=50)
    assert log.total_finished() == 3  # the 50 truncated rows contributed nothing
    assert log.support("collect_wood") == 3


def test_names_remap(path):
    log = ChainOrderLog(path)
    # Columns in a scrambled 3-wide order + one unknown name (dropped, no crash).
    names = ["defeat_troll", "bogus_skill", "collect_iron"]
    rows = [[9, 3, 1]] * 5  # iron@1 -> troll@9 in every episode
    log.add_session(1, rows, [True] * 5, names=names)
    assert log.support("defeat_troll") == 5
    assert log.support("collect_iron") == 5
    assert [p[0] for p in log.dominant_path("defeat_troll")] == ["collect_iron"]


# ---- break-link mining on failures ---------------------------------------------------------------

_TROLL_CHAIN = {"defeat_troll": ["collect_iron", "make_iron_sword"]}


def test_break_link_distribution(path):
    log = ChainOrderLog(path)
    eps = (
        [{"collect_iron": 2}] * 100                          # died after link 1
        + [{"collect_iron": 2, "make_iron_sword": 6}] * 20   # died after link 2
        + [{"collect_iron": 2, "make_iron_sword": 6, "defeat_troll": 9}] * 10  # succeeded
    )
    _add(log, 1, eps, chain_targets=_TROLL_CHAIN)
    entry = log.latest_fail_summary("defeat_troll")
    assert entry["n_fail"] == 120 and entry["n_succ"] == 10
    assert entry["last_link"] == {"collect_iron": 100, "make_iron_sword": 20}
    assert entry["missing"] == {"make_iron_sword": 100}
    assert entry["mean_depth"] == pytest.approx((100 * 1 + 20 * 2) / 120, abs=1e-3)


def test_break_link_none_bucket(path):
    log = ChainOrderLog(path)
    _add(log, 1, [{"collect_wood": 0}] * 80, chain_targets=_TROLL_CHAIN)
    entry = log.latest_fail_summary("defeat_troll")
    assert entry["last_link"] == {"(none)": 80}
    assert entry["mean_depth"] == 0.0


def test_frontier_advance_detection(path):
    log = ChainOrderLog(path)
    # session 1: everyone dies after link 1 (mean depth 1.0)
    _add(log, 1, [{"collect_iron": 2}] * 100, chain_targets=_TROLL_CHAIN)
    assert not log.frontier_advanced("defeat_troll")  # single entry: nothing to compare
    # session 2: half the cohort now reaches link 2 (mean 1.5 >= 1.0 + 0.25) -> ADVANCED
    _add(
        log, 2,
        [{"collect_iron": 2}] * 50 + [{"collect_iron": 2, "make_iron_sword": 6}] * 50,
        chain_targets=_TROLL_CHAIN,
    )
    assert log.frontier_advanced("defeat_troll")
    # session 3: back to shallow deaths -> latest no longer beats the best previous -> not advanced
    _add(log, 3, [{"collect_iron": 2}] * 100, chain_targets=_TROLL_CHAIN)
    assert not log.frontier_advanced("defeat_troll")


def test_frontier_needs_min_fails(path):
    log = ChainOrderLog(path)
    small = FRONTIER_MIN_FAILS - 1
    _add(log, 1, [{"collect_iron": 2}] * small, chain_targets=_TROLL_CHAIN)
    _add(log, 2, [{"collect_iron": 2, "make_iron_sword": 6}] * small, chain_targets=_TROLL_CHAIN)
    assert not log.frontier_advanced("defeat_troll")  # both sessions under the support floor


def test_frontier_requires_same_chain(path):
    log = ChainOrderLog(path)
    _add(log, 1, [{"collect_iron": 2}] * 100, chain_targets=_TROLL_CHAIN)
    # the chain MEMBERSHIP changed between sessions (a link added) -> the depth denominator
    # changed -> not comparable -> no advance
    other = {"defeat_troll": ["collect_iron", "place_torch", "make_iron_sword"]}
    _add(
        log, 2,
        [{"collect_iron": 2, "make_iron_sword": 6}] * 100,
        chain_targets=other,
    )
    assert not log.frontier_advanced("defeat_troll")


def test_frontier_survives_link_reorder(path):
    """The LLM re-submits the prereq_tree every session; a mere REORDER of the same links must NOT
    reset frontier comparability (depth counts achieved links, order-invariant; comparability is by
    link SET). This was the fail-quiet hazard found in the 2026-07-07 audit."""
    log = ChainOrderLog(path)
    _add(log, 1, [{"collect_iron": 2}] * 100, chain_targets=_TROLL_CHAIN)
    reordered = {"defeat_troll": ["make_iron_sword", "collect_iron"]}  # same set, new order
    _add(
        log, 2,
        [{"collect_iron": 2, "make_iron_sword": 6}] * 100,  # now 2 links achieved per failure
        chain_targets=reordered,
    )
    assert log.frontier_advanced("defeat_troll")  # 1.0 -> 2.0 achieved links, still comparable


def test_empty_chain_target_skipped(path):
    """A freshly-opened focus has no prereq_tree yet: mining it would only yield a junk
    '(none) 100%' entry that renders as a misleading 'first link is the wall' line -> skip."""
    log = ChainOrderLog(path)
    _add(log, 1, [{"collect_wood": 0}] * 80, chain_targets={"defeat_troll": []})
    assert log.latest_fail_summary("defeat_troll") is None
    assert log.render_chain_hint("defeat_troll") == ""


def test_frontier_threshold_boundary(path):
    log = ChainOrderLog(path)
    _add(log, 1, [{"collect_iron": 2}] * 100, chain_targets=_TROLL_CHAIN)
    # mean depth 1.0 + just UNDER the threshold -> not an advance
    n_deep = int(round(100 * (FRONTIER_ADVANCE_LINKS - 0.05)))
    _add(
        log, 2,
        [{"collect_iron": 2}] * (100 - n_deep)
        + [{"collect_iron": 2, "make_iron_sword": 6}] * n_deep,
        chain_targets=_TROLL_CHAIN,
    )
    assert not log.frontier_advanced("defeat_troll")


# ---- persistence ----------------------------------------------------------------------------------

def test_resume_idempotent_session(path):
    log = ChainOrderLog(path)
    _add(log, 5, [{"collect_iron": 1, "defeat_troll": 9}] * 4, chain_targets=_TROLL_CHAIN)
    _add(log, 5, [{"collect_iron": 1, "defeat_troll": 9}] * 4, chain_targets=_TROLL_CHAIN)
    assert log.support("defeat_troll") == 4
    assert log.total_finished() == 4
    assert len([e for e in log._fail_hist if e["target"] == "defeat_troll"]) == 1


def test_persistence_roundtrip(path):
    log = ChainOrderLog(path)
    _add(log, 1, [{"collect_iron": 2}] * 100, chain_targets=_TROLL_CHAIN)
    log2 = ChainOrderLog(path)  # reopen from disk
    assert log2.total_finished() == 100
    assert log2.latest_fail_summary("defeat_troll")["n_fail"] == 100
    _add(
        log2, 2,
        [{"collect_iron": 2}] * 50 + [{"collect_iron": 2, "make_iron_sword": 6}] * 50,
        chain_targets=_TROLL_CHAIN,
    )
    assert log2.frontier_advanced("defeat_troll")  # history survived the reopen


def test_corrupt_file_falls_back(path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not valid json")
    log = ChainOrderLog(path)
    assert log.total_finished() == 0  # did not crash


# ---- prompt rendering -------------------------------------------------------------------------------

def test_render_chain_hint_full_block(path):
    log = ChainOrderLog(path)
    # session 1: shallow failures + a real success cohort (SR 10/110 = 9% >= MIN_SR)
    _add(
        log, 1,
        [{"collect_iron": 2}] * 100
        + [{"collect_iron": 2, "make_iron_sword": 6, "defeat_troll": 9}] * 10,
        chain_targets=_TROLL_CHAIN,
    )
    # session 2: failures die deeper -> frontier line must appear
    _add(
        log, 2,
        [{"collect_iron": 2, "make_iron_sword": 6}] * 100
        + [{"collect_iron": 2, "make_iron_sword": 6, "defeat_troll": 9}] * 10,
        chain_targets=_TROLL_CHAIN,
    )
    hint = log.render_chain_hint("defeat_troll")
    assert "collect_iron" in hint and "make_iron_sword" in hint  # dominant path rendered
    # v6fix9 P0: the modal deepest link here is the chain's TAIL (make_iron_sword) — the hint must
    # say "reaches the whole chain and still fails" instead of claiming a snapped link.
    assert "STILL fail the wall" in hint
    assert "ADVANCING" in hint                                   # frontier movement rendered


def test_render_break_line_mid_chain(path):
    """A genuinely located mid-chain break (not tail, not universal) keeps the snapped-link wording,
    and the v6fix9 forensic missing-histogram line is rendered alongside it."""
    log = ChainOrderLog(path)
    # 40 fails die before ANY link, 60 die after collect_iron, 10 succeed -> modal = collect_iron
    # (60%), not the tail; deep_sr(collect_iron) = 70/110 = 64% < the universal bar.
    _add(
        log, 1,
        [{}] * 40 + [{"collect_iron": 2}] * 60
        + [{"collect_iron": 2, "make_iron_sword": 6, "defeat_troll": 9}] * 10,
        chain_targets=_TROLL_CHAIN,
    )
    hint = log.render_chain_hint("defeat_troll")
    assert "break AFTER collect_iron" in hint
    assert "before make_iron_sword" in hint
    assert "make_iron_sword missing in 100%" in hint  # forensic line


def test_render_artifact_guard_universal_link(path):
    """v6fix9 P0 (the make_iron_armour misdiagnosis shape): a near-universal link deep in the
    proposed chain is the modal deepest-achieved link of ~every failure — the hint must NOT claim
    the chain snaps there, and the missing histogram (the disambiguating field) must be rendered."""
    log = ChainOrderLog(path)
    chain = {"make_iron_armour": ["collect_iron", "place_table"]}
    # every episode places a table (universal, deep_sr = 100%); only 30% of failures touch iron.
    _add(
        log, 1,
        [{"place_table": 1}] * 70
        + [{"collect_iron": 3, "place_table": 1}] * 30
        + [{"collect_iron": 3, "place_table": 1, "make_iron_armour": 9}] * 10,
        chain_targets=chain,
    )
    hint = log.render_chain_hint("make_iron_armour")
    assert "collect_iron missing in 70%" in hint       # forensic histogram reaches the prompt
    assert "break AFTER place_table" not in hint       # the tautological artifact line is gone
    assert "STILL fail the wall" in hint               # honest final-step wording instead
    assert "QUANTITY" in hint                          # points at the binary-achievement blind spot


def test_render_hint_empty_without_trustworthy_data(path):
    log = ChainOrderLog(path)
    # a handful of fails (< FRONTIER_MIN_FAILS) and SR 0 -> neither line has support -> empty
    _add(log, 1, [{"collect_iron": 2}] * 5, chain_targets=_TROLL_CHAIN)
    assert log.render_chain_hint("defeat_troll") == ""


# ---- SiegeNotebook integration (chain_targets / retired-wall frontier escape) ---------------------

_TREE = [{"skill": "collect_iron", "role": "gear"}, {"skill": "make_iron_sword", "role": "gear"}]

# v7fix4: the notebook-integration wall must be a NATIVE-FLOOR-0 skill — a floor-3+ wall
# (defeat_troll's habitat is floor 5) is deep-locked out of ordinary opens, and any floor-bound
# wall gets entrance links autofilled into its chain, changing chain_targets. These tests pin the
# generic chain machinery, so they use a gear wall the habitat gates leave untouched.
_NB_WALL = "make_iron_armour"


def _open_focus_with_tree(nb, session=1, skill=_NB_WALL):
    nb.apply_llm_update(
        session, _mature_profile({skill: 3.0}),
        {"foci": [{"skill": skill, "prereq_tree": _TREE}]},
        num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )


def test_notebook_chain_targets_active_focus(tmp_path):
    nb = SiegeNotebook(str(tmp_path / "nb.json"))
    _open_focus_with_tree(nb)
    assert nb.chain_targets() == {_NB_WALL: ["collect_iron", "make_iron_sword"]}


def test_notebook_chain_targets_keeps_retired_wall(tmp_path):
    nb = SiegeNotebook(str(tmp_path / "nb.json"))
    _open_focus_with_tree(nb)
    for s in range(2, 2 + LADDER_L4 + 1):  # freeze through the ladder -> retirement
        nb.apply_llm_update(
            s, _mature_profile({_NB_WALL: 3.0}), None, num_snapshots=MATURITY_MIN_SNAPSHOTS,
        )
    assert nb.focus_skills() == []  # retired
    # the retired wall stays under chain watch, with the chain snapshotted at retirement (ordered)
    assert nb.chain_targets() == {_NB_WALL: ["collect_iron", "make_iron_sword"]}
    reg = nb._nb["retired"][_NB_WALL]
    assert reg["links_at_retirement"] == ["collect_iron", "make_iron_sword"]
    assert "chain_frontier_advanced" not in reg  # retirement starts with a clean flag


def test_note_chain_progress_reaches_retired_wall_and_unlocks_blacklist(tmp_path):
    nb = SiegeNotebook(str(tmp_path / "nb.json"))
    _open_focus_with_tree(nb)
    for s in range(2, 2 + LADDER_L4 + 1):
        nb.apply_llm_update(
            s, _mature_profile({_NB_WALL: 3.0}), None, num_snapshots=MATURITY_MIN_SNAPSHOTS,
        )
    reg = nb._nb["retired"][_NB_WALL]
    # flat SR everywhere -> no SR-based new evidence
    assert not nb._has_new_evidence(_NB_WALL, reg, {_NB_WALL: 3.0})
    # the P2 frontier signal lands on the RETIRED wall's registry entry...
    nb.note_chain_progress(_NB_WALL)
    assert reg.get("chain_frontier_advanced") is True
    # ...and counts as new evidence (the blacklist escape hatch), SR still flat
    assert nb._has_new_evidence(_NB_WALL, reg, {_NB_WALL: 3.0})


def test_note_chain_progress_prefers_active_focus(tmp_path):
    nb = SiegeNotebook(str(tmp_path / "nb.json"))
    _open_focus_with_tree(nb)
    nb.note_chain_progress(_NB_WALL)
    foc = nb.foci()[0]
    assert foc.get("chain_frontier_advanced") is True  # landed on the focus, not a registry entry
    assert _NB_WALL not in (nb._nb.get("retired") or {})
