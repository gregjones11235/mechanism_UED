"""v7fix4.2: deep-wall relay trigger — autoconvert + journal directive + K-session force.

fix4-run post-mortem (@s77, 2026-07-12): the modeler NEVER proposed a floor-3+ wall in 77
sessions — zero deep_locked refusals, zero relay_r0_floor asks. fix4's deep lock had closed the
one relay on-ramp any run actually used (fix3's lizard: ordinary open -> zero-win -> in-place
upgrade), leaving a cold-start path no LLM ever takes, while auto-open deliberately excludes
deep walls. The trigger pins the entry in code (the fix7 lesson, one level up):

  (1) AUTOCONVERT  — an ordinary proposal of a floor-3..(tier4-1) wall converts IN PLACE into a
      relay ask and rides the EXISTING explicit-relay pipeline (decision ``relay_converted``);
      tier-4 keeps ``tier_locked``; a wall with real held-out wins keeps ``deep_locked``; a busy
      relay slot keeps ``deep_locked`` with a slot-busy teaching clause.
  (2) JOURNAL DIRECTIVE — while (mature, no live relay, free slot, eligible wall exists) the
      journal renders ★RELAY TRIGGER naming the eligible walls, deterministic best-first.
  (3) K-SESSION FORCE — the directive ignored ``relay_trigger_force_sessions`` consecutive
      armed decisions -> the system opens the top candidate itself (``relay_forced``).
      Reproducibility backstop: relay start must not depend on the LLM's mood.

Pure python (SiegeNotebook), no jax/LLM — same fixture style as the .sh functional smokes.
"""

import pytest

pytestmark = pytest.mark.relay_trigger  # opt back in: this suite IS about the trigger

from auction.siege_notebook import (
    MATURITY_MIN_MASTERED,
    MATURITY_MIN_SNAPSHOTS,
    MATURITY_SKILL_SR,
    SiegeNotebook,
)

K = 3  # mirrors RELAY_TRIGGER_FORCE_SESSIONS (asserted in test_defaults below)


def _mature_prof(**targets):
    prof = {f"f{i}": MATURITY_SKILL_SR + 10 for i in range(MATURITY_MIN_MASTERED)}
    prof.update(targets)
    return prof


def _nb(tmp_path, name="nb.json"):
    return SiegeNotebook(str(tmp_path / name))


def _apply(nb, session, prof, foci):
    return nb.apply_llm_update(
        session, prof, {"foci": foci}, num_snapshots=MATURITY_MIN_SNAPSHOTS, forensics={}
    )


# ---- defaults wired ---------------------------------------------------------------------------

def test_defaults(tmp_path):
    nb = _nb(tmp_path)
    assert nb.th.deep_wall_autoconvert is True
    assert nb.th.relay_trigger_hint is True
    assert nb.th.relay_trigger_force_sessions == K


# ---- (1) autoconvert --------------------------------------------------------------------------

def test_deep_ordinary_proposal_autoconverts_on_original_wall(tmp_path):
    nb = _nb(tmp_path)
    prof = _mature_prof(defeat_lizard=0.0)
    _apply(nb, 1, prof, [{"skill": "defeat_lizard", "prereq_tree": []}])
    assert "relay_converted(defeat_lizard" in nb.last_focus_decision
    assert "opened_relay(defeat_lizard" in nb.last_focus_decision
    # the conversion acted on the ORIGINAL wall, r0 anchored to its habitat (floor 3, Sewers) —
    # NOT door-substituted onto enter_sewers.
    assert nb.relay_walls() == ["defeat_lizard"]
    assert nb.required_spawn_floor("defeat_lizard") == 3
    assert all(f.get("skill") != "enter_sewers" for f in nb.foci())


def test_tier4_keeps_tier_locked_no_conversion(tmp_path):
    nb = _nb(tmp_path)
    prof = _mature_prof(defeat_pigman=0.0)
    _apply(nb, 1, prof, [{"skill": "defeat_pigman", "prereq_tree": []}])
    assert "tier_locked(defeat_pigman" in nb.last_focus_decision
    assert nb.relay_walls() == []


def test_no_conversion_while_relay_slot_busy(tmp_path):
    nb = _nb(tmp_path)
    prof = _mature_prof(defeat_lizard=0.0, defeat_kobold=0.0)
    _apply(nb, 1, prof, [{"skill": "defeat_lizard", "prereq_tree": []}])
    assert nb.relay_walls() == ["defeat_lizard"]
    _apply(nb, 3, prof, [{"skill": "defeat_kobold", "prereq_tree": []}])
    assert "deep_locked(defeat_kobold" in nb.last_focus_decision
    assert "slot" in nb.last_focus_decision  # the busy clause teaches to keep proposing
    assert nb.relay_walls() == ["defeat_lizard"]


def test_no_conversion_for_wall_with_heldout_wins(tmp_path):
    nb = _nb(tmp_path)
    prof = _mature_prof(defeat_lizard=30.0)
    _apply(nb, 1, prof, [{"skill": "defeat_lizard", "prereq_tree": []}])
    assert "deep_locked(defeat_lizard" in nb.last_focus_decision
    assert "relay_converted" not in nb.last_focus_decision
    assert nb.relay_walls() == []


def test_flag_off_restores_old_refusal(tmp_path):
    nb = _nb(tmp_path)
    nb.th.deep_wall_autoconvert = False
    prof = _mature_prof(defeat_lizard=0.0)
    _apply(nb, 1, prof, [{"skill": "defeat_lizard", "prereq_tree": []}])
    assert "deep_locked(defeat_lizard" in nb.last_focus_decision
    assert nb.relay_walls() == []


# ---- (2) journal directive --------------------------------------------------------------------

def test_trigger_arms_and_renders(tmp_path):
    nb = _nb(tmp_path)
    _apply(nb, 1, _mature_prof(), [])
    journal = nb.render_for_prompt()
    assert "★RELAY TRIGGER" in journal
    assert "relay_converted" in journal and "relay_forced" in journal
    assert "0/3" in journal
    # deterministic best-first: fight walls before entrances.
    cands = (nb._nb.get("relay_trigger") or {}).get("candidates") or []
    assert cands and cands[0].startswith("defeat_")


def test_trigger_not_armed_when_immature(tmp_path):
    nb = _nb(tmp_path)
    nb.apply_llm_update(1, _mature_prof(), {"foci": []}, num_snapshots=1, forensics={})
    assert "★RELAY TRIGGER" not in nb.render_for_prompt()


def test_trigger_not_armed_when_hint_flag_off(tmp_path):
    nb = _nb(tmp_path)
    nb.th.relay_trigger_hint = False
    _apply(nb, 1, _mature_prof(), [])
    assert "★RELAY TRIGGER" not in nb.render_for_prompt()


def test_trigger_disarmed_while_relay_live(tmp_path):
    nb = _nb(tmp_path)
    prof = _mature_prof(defeat_lizard=0.0)
    _apply(nb, 1, prof, [{"skill": "defeat_lizard", "prereq_tree": []}])
    assert nb.relay_walls() == ["defeat_lizard"]
    assert "★RELAY TRIGGER" not in nb.render_for_prompt()


def test_trigger_disarmed_when_no_free_slot(tmp_path):
    # intent: capacity is a hard condition — with zero openable slots the trigger must not arm
    # (and therefore can never force). max_focus=0 isolates the slot condition from the door/
    # admission gates an ordinary focus-open would otherwise drag into this test.
    nb = _nb(tmp_path)
    nb.th.max_focus = 0
    _apply(nb, 1, _mature_prof(), [])
    assert "★RELAY TRIGGER" not in nb.render_for_prompt()
    st = nb._nb.get("relay_trigger") or {}
    assert not st.get("armed") and st.get("ignored") == 0


# ---- (3) K-session force ----------------------------------------------------------------------

def test_force_opens_top_candidate_after_k_ignored(tmp_path):
    nb = _nb(tmp_path)
    prof = _mature_prof()
    _apply(nb, 1, prof, [])  # arms (ignored 0)
    for s, expected_ignored in ((2, 1), (3, 2)):
        _apply(nb, s, prof, [])
        st = nb._nb.get("relay_trigger") or {}
        assert st.get("armed") and st.get("ignored") == expected_ignored
    _apply(nb, 4, prof, [])  # 3rd ignored armed decision -> force
    assert nb.relay_walls(), "force must open a relay campaign"
    forced = nb.relay_walls()[0]
    assert forced.startswith("defeat_")
    assert nb.required_spawn_floor(forced) is not None
    assert nb.last_relay_open and "relay_forced(" in nb.last_relay_open
    assert "★RELAY TRIGGER" not in nb.render_for_prompt()  # disarmed, relay live
    # the forced campaign got its entrance chain autofilled the SAME session (step 5 runs after).
    tree = [l["skill"] for l in nb.foci()[0].get("prereq_tree") or []]
    assert "enter_dungeon" in tree


def test_counter_resets_when_answered(tmp_path):
    nb = _nb(tmp_path)
    prof = _mature_prof(defeat_lizard=0.0)
    _apply(nb, 1, prof, [])
    _apply(nb, 2, prof, [])
    assert (nb._nb.get("relay_trigger") or {}).get("ignored") == 1
    # answering with a deep wall (here: one that converts) resets everything.
    _apply(nb, 3, prof, [{"skill": "defeat_lizard", "prereq_tree": []}])
    st = nb._nb.get("relay_trigger") or {}
    assert st.get("ignored") == 0 and not st.get("armed")
    assert nb.relay_walls() == ["defeat_lizard"]


def test_trigger_state_is_resume_safe(tmp_path):
    nb = _nb(tmp_path)
    prof = _mature_prof()
    _apply(nb, 1, prof, [])
    _apply(nb, 2, prof, [])
    nb2 = SiegeNotebook(str(tmp_path / "nb.json"))
    st = nb2._nb.get("relay_trigger") or {}
    assert st.get("armed") and st.get("ignored") == 1
    assert "★RELAY TRIGGER" in nb2.render_for_prompt()
    assert "1/3" in nb2.render_for_prompt()


# ---- candidate ordering -----------------------------------------------------------------------

def test_candidates_combat_first_then_floor_then_name(tmp_path):
    nb = _nb(tmp_path)
    cands = nb._relay_trigger_candidates({}, session_idx=None)
    assert cands, "floor-3+ non-tier4 walls must exist"
    from auction.craftax_achievements import native_floor_of, tier_of

    assert all(native_floor_of(s) >= nb.th.deep_wall_relay_floor for s in cands)
    assert all(tier_of(s) < 4 for s in cands)
    first_noncombat = next(
        (i for i, s in enumerate(cands) if not s.startswith("defeat_")), len(cands)
    )
    assert all(s.startswith("defeat_") for s in cands[:first_noncombat])
    assert not any(s.startswith("defeat_") for s in cands[first_noncombat:])


def test_candidates_exclude_walls_with_wins_and_active(tmp_path):
    nb = _nb(tmp_path)
    prof = _mature_prof(defeat_lizard=0.0, defeat_kobold=25.0)
    _apply(nb, 1, prof, [{"skill": "defeat_lizard", "prereq_tree": []}])  # lizard now active relay
    cands = nb._relay_trigger_candidates(prof, session_idx=1)
    assert "defeat_lizard" not in cands  # active
    assert "defeat_kobold" not in cands  # has real held-out wins
