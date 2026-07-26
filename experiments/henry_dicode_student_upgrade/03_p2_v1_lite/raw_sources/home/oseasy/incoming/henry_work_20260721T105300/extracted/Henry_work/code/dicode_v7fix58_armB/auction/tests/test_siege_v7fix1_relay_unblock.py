"""v7fix1 — offline tests for the RELAY TRIGGER-PATH UNBLOCK (first-run s23 post-mortem).

v7 first run (job 3773598): all fix10 gates worked, but RELAY-OPEN=0 / RUNG=0 through s23 even
though the LLM targeted the exact deep walls relays exist for (defeat_kobold s11-13, defeat_troll
s15-23). Root-cause chain: the ⑦ chain-incomplete latch in _reconcile_foci sat BEFORE the relay
branch and silently swallowed every relay proposal for a latched wall; its rejection message only
said "EXPAND the prereq_tree", never naming the relay exit; the LLM's deep chains were
hallucinated filler (troll: eat_cow / collect_drink), so ⑦ latched permanently.

These tests pin the fix:
  1. a relay proposal (relay_r0_floor set) on a ⑦-latched wall is ADMITTED (opened_relay) — the
     rung ladder + R1/R3/R6 replace chain completeness structurally;
  2. a NON-relay proposal on a ⑦-latched wall is still refused, and the message now teaches the
     relay exit (relay_r0_floor) + the closed vocabulary;
  3. when the relay slot is full, a ⑦-latched wall's relay ask may NOT fall through to a normal
     open (⑦ was only waived because a relay was requested);
  4. the shipped prompt states the exemption (rule (3) ★ + RELAY RULES (f)).

No jax/craftax/LLM needed.
"""

import pytest

from auction.modeler import MODELER_SIEGE_SYSTEM_PROMPT
from auction.siege_notebook import (
    MATURITY_MIN_MASTERED,
    MATURITY_MIN_SNAPSHOTS,
    MATURITY_SKILL_SR,
    RELAY_MAX,
    SiegeNotebook,
)


def _mature_profile(extra: dict | None = None) -> dict:
    prof = {f"filler_{i}": MATURITY_SKILL_SR + 10 for i in range(MATURITY_MIN_MASTERED)}
    if extra:
        prof.update(extra)
    return prof


@pytest.fixture
def nb_path(tmp_path):
    return str(tmp_path / "siege_notebook.json")


def _update(nb, session, profile, foci=None, ranked=None, forensics=None, incomplete=None):
    proposal = {}
    if foci is not None:
        proposal["foci"] = foci
    if ranked is not None:
        proposal["ranked_walls"] = ranked
    return nb.apply_llm_update(
        session, profile, proposal, num_snapshots=MATURITY_MIN_SNAPSHOTS,
        forensics=forensics, chain_incomplete=incomplete,
    )


def test_relay_proposal_passes_chain_incomplete_latch(nb_path):
    """The headline bug: kobold/troll relay proposals must reach the relay branch, not die at ⑦."""
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({"defeat_kobold": 0.0})
    _update(nb, 1, prof,
            foci=[{"skill": "defeat_kobold", "prereq_tree": [], "relay_r0_floor": 3}],
            forensics={}, incomplete={"defeat_kobold"})
    assert "opened_relay(defeat_kobold" in nb.last_focus_decision
    assert nb.focus_skills() == ["defeat_kobold"]
    assert nb.last_relay_open is not None


def test_non_relay_proposal_still_latched_with_relay_hint(nb_path):
    # v7fix4: the latched wall is a floor-2 gnome — a floor-3+ wall now gets deep_locked (with the
    # same relay teaching) BEFORE ⑦ can rule, so ⑦'s own message is pinned on a shallow wall.
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({"defeat_gnome_warrior": 0.0})
    _update(nb, 1, prof, foci=[{"skill": "defeat_gnome_warrior", "prereq_tree": []}],
            forensics={}, incomplete={"defeat_gnome_warrior"})
    assert nb.focus_skills() == []
    assert "chain_incomplete(defeat_gnome_warrior" in nb.last_focus_decision
    assert "relay_r0_floor" in nb.last_focus_decision
    assert "EXACT achievement names" in nb.last_focus_decision


def test_capacity_full_latched_wall_does_not_fall_through_to_normal_open(nb_path):
    assert RELAY_MAX == 1  # the fall-through guard below assumes a single slot
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({"defeat_troll": 0.0, "defeat_kobold": 0.0})
    _update(nb, 1, prof,
            foci=[{"skill": "defeat_troll", "prereq_tree": [], "relay_r0_floor": 4}],
            forensics={})
    assert nb.focus_skills() == ["defeat_troll"]
    # expand slack: the running relay's wall now reads a solid SR so a second seat is allowed —
    # the second relay ask must still be refused on CAPACITY, and ⑦ must veto the normal-open
    # fallback (this was a fresh hole the reorder would otherwise open).
    prof2 = _mature_profile({"defeat_troll": 55.0, "defeat_kobold": 0.0})
    _update(nb, 3, prof2,
            foci=[{"skill": "defeat_kobold", "prereq_tree": [], "relay_r0_floor": 3}],
            forensics={}, incomplete={"defeat_kobold"})
    assert "relay_refused(defeat_kobold" in nb.last_focus_decision
    assert "keep the relay proposal" in nb.last_focus_decision
    assert nb.focus_skills() == ["defeat_troll"]


def test_capacity_full_unlatched_wall_still_falls_through(nb_path):
    """A wall NOT ⑦-latched keeps the original fall-through semantics (normal open path)."""
    nb = SiegeNotebook(nb_path)
    prof = _mature_profile({"defeat_troll": 0.0, "defeat_kobold": 0.0})
    _update(nb, 1, prof,
            foci=[{"skill": "defeat_troll", "prereq_tree": [], "relay_r0_floor": 4}],
            forensics={})
    # v7fix4: the falling-through wall must be floor-<3 (a deep wall now gets relay_refused with
    # no ordinary fall-through at all) — the gnome keeps the original semantics.
    prof2 = _mature_profile({"defeat_troll": 55.0, "defeat_gnome_warrior": 0.0})
    _update(nb, 3, prof2,
            foci=[{"skill": "defeat_gnome_warrior", "prereq_tree": [], "relay_r0_floor": 2}],
            forensics={"defeat_gnome_warrior": {"missing_top": []}})
    assert "relay_refused(defeat_gnome_warrior" in nb.last_focus_decision
    assert "treating this as a normal focus proposal" in nb.last_focus_decision
    # no door, forensics present -> opens as a NORMAL focus (no relay state on it)
    assert "opened(defeat_gnome_warrior)" in nb.last_focus_decision


def test_prompt_states_the_exemption_and_the_exit():
    assert "EXEMPT from the chain-incomplete refusal" in MODELER_SIEGE_SYSTEM_PROMPT
    assert "relay IS the sanctioned exit" in MODELER_SIEGE_SYSTEM_PROMPT
    assert "CLOSED VOCABULARY" in MODELER_SIEGE_SYSTEM_PROMPT
