"""Offline tests for the SiegeNotebook (v6 §3.5) — the B-layer hard constraints.

The whole point of the A+B hybrid is that the LLM (A) can PROPOSE anything, but the CODE (B) enforces
invariants the LLM cannot violate. These tests pin every B-layer rule:

  - persistence: atomic save + resume round-trip, malformed file -> empty schema, missing keys filled.
  - §3.2 SCOPE: a saturated/easy skill is REFUSED as a focus (siege only serves real walls).
  - focus-switch minimum condition: the LLM cannot thrash the focus session-to-session; only after
    FOCUS_MIN_STALL_SESSIONS non-improving sessions (or conquest) may it switch.
  - mastery flags are CODE-owned: an LLM "this link is mastered" claim that SR contradicts is ignored.
  - success experience is CODE-driven & INCREMENTAL (§2① user 2026-07-05): an SR rise of >=
    RECORD_DELTA_PP writes/updates (dedup-by-target, combat/enabler categorised) a verified_chains
    entry + protects target/links; noise jitter records nothing; the focus is NOT retired by an SR
    threshold (only by stall, §2.6).
  - unmastered_links (consumed by the §3.4 Completed gate) reflects LIVE SR, unioned over all foci.

No jax/craftax/LLM needed.
"""

import json

import pytest

from auction.siege_notebook import (
    FOCUS_EXPAND_SR,
    FOCUS_MIN_STALL_SESSIONS,
    MASTERED_SR,
    MATURITY_MIN_MASTERED,
    MATURITY_MIN_SNAPSHOTS,
    MATURITY_SKILL_SR,
    RECORD_DELTA_PP,
    SATURATED_SR,
    UNMASTERED_SR,
    SiegeNotebook,
    mastery_from_sr,
)


def _mature_profile(extra: dict | None = None) -> dict:
    """A held-out profile of a student PAST the early ramp: MATURITY_MIN_MASTERED skills at decent SR."""
    prof = {f"filler_{i}": MATURITY_SKILL_SR + 10 for i in range(MATURITY_MIN_MASTERED)}
    if extra:
        prof.update(extra)
    return prof


@pytest.fixture
def nb_path(tmp_path):
    return str(tmp_path / "siege_notebook.json")


# ---- mastery_from_sr thresholds ----------------------------------------------------------------

def test_mastery_from_sr_thresholds():
    assert mastery_from_sr(None) == "UNKNOWN"
    assert mastery_from_sr(MASTERED_SR) == "CONSOLIDATED"
    assert mastery_from_sr(MASTERED_SR + 10) == "CONSOLIDATED"
    assert mastery_from_sr(UNMASTERED_SR) == "UNMASTERED"
    assert mastery_from_sr(0.0) == "UNMASTERED"
    # between the two bands -> RISING
    mid = (UNMASTERED_SR + MASTERED_SR) / 2
    assert mastery_from_sr(mid) == "RISING"


# ---- persistence -------------------------------------------------------------------------------

def test_empty_on_fresh_path(nb_path):
    nb = SiegeNotebook(nb_path)
    snap = nb.snapshot()
    assert snap["foci"] == []              # §2.6: foci list, no top-level single focus
    assert nb.focus is None                # primary-focus accessor
    assert nb.prereq_links() == []
    assert snap["verified_chains"] == []
    assert snap["protected_set"] == []


def test_resume_roundtrip(nb_path):
    nb = SiegeNotebook(nb_path)
    # Set a focus that is a legal wall (SR low), student mature.
    nb.apply_llm_update(
        1, _mature_profile({"defeat_gnome_warrior": 5.0}),
        {"focus": "defeat_gnome_warrior"}, num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    assert nb.focus == "defeat_gnome_warrior"
    # Re-open from disk -> focus survives.
    nb2 = SiegeNotebook(nb_path)
    assert nb2.focus == "defeat_gnome_warrior"


def test_malformed_file_falls_back_to_empty(nb_path):
    with open(nb_path, "w", encoding="utf-8") as f:
        f.write("{ this is not json ]")
    nb = SiegeNotebook(nb_path)
    assert nb.focus is None  # did not crash; empty schema


def test_missing_keys_filled_on_load(nb_path):
    # A partial OLD-schema file (top-level "focus") must be coerced AND migrated to the foci list.
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump({"focus": "defeat_troll"}, f)
    nb = SiegeNotebook(nb_path)
    snap = nb.snapshot()
    assert nb.focus == "defeat_troll"       # migrated into foci[0]
    assert nb.focus_skills() == ["defeat_troll"]
    assert nb.prereq_links() == []          # filled
    assert snap["protected_set"] == []      # filled


# ---- §3.2 SCOPE hard constraint ----------------------------------------------------------------

def test_saturated_skill_refused_as_focus(nb_path):
    nb = SiegeNotebook(nb_path)
    # collect_wood is mastered (SR high) -> forbidden as a siege focus (student mature otherwise).
    nb.apply_llm_update(
        1, _mature_profile({"collect_wood": SATURATED_SR + 5}),
        {"focus": "collect_wood"}, num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    assert nb.focus is None  # refused


def test_real_wall_accepted_as_focus(nb_path):
    nb = SiegeNotebook(nb_path)
    nb.apply_llm_update(
        1, _mature_profile({"defeat_orc_mage": 8.0}),
        {"focus": "defeat_orc_mage"}, num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    assert nb.focus == "defeat_orc_mage"


# ---- EARLY-TRAINING guard: low SR early is NORMAL_EARLY, not a wall (user 2026-07-04) -----------

def test_no_focus_when_too_few_snapshots(nb_path):
    nb = SiegeNotebook(nb_path)
    # Even a fully-mature-looking profile: with too few snapshots we can't read a trend -> no siege.
    nb.apply_llm_update(
        1, _mature_profile({"make_wood_pickaxe": 3.0}),
        {"focus": "make_wood_pickaxe"}, num_snapshots=MATURITY_MIN_SNAPSHOTS - 1,
    )
    assert nb.focus is None  # refused — too early


def test_no_focus_when_student_immature(nb_path):
    nb = SiegeNotebook(nb_path)
    # Enough snapshots, but the WHOLE student is still weak (few skills at decent SR): a low SR on an
    # easy skill is NORMAL_EARLY, not a wall. The LLM proposing to siege it must be refused in code.
    weak_profile = {f"basic_{i}": 8.0 for i in range(MATURITY_MIN_MASTERED + 2)}
    weak_profile["make_wood_pickaxe"] = 3.0
    nb.apply_llm_update(
        1, weak_profile, {"focus": "make_wood_pickaxe"},
        num_snapshots=MATURITY_MIN_SNAPSHOTS + 5,
    )
    assert nb.focus is None  # refused — student not past the early ramp


def test_focus_allowed_once_mature(nb_path):
    nb = SiegeNotebook(nb_path)
    # Same easy-ish target, but now the student is genuinely mature (many skills solid) AND has
    # enough snapshots: a persistently-low skill can now legitimately be a wall.
    nb.apply_llm_update(
        1, _mature_profile({"defeat_gnome_warrior": 4.0}),
        {"focus": "defeat_gnome_warrior"}, num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    assert nb.focus == "defeat_gnome_warrior"


# ---- §2.6 multi-focus: expand gate + stall-retire (replaces the old single-focus switch) --------

def _set_focus(nb, session, focus, extra_sr=None, ns=MATURITY_MIN_SNAPSHOTS):
    nb.apply_llm_update(session, _mature_profile(extra_sr or {focus: 8.0}), {"focus": focus}, num_snapshots=ns)


def test_second_focus_refused_before_expand_sr(nb_path):
    # A new parallel focus may open only once an existing focus reaches FOCUS_EXPAND_SR. The first
    # focus is only at ~9%, so proposing a second wall is refused -> still just the one focus.
    nb = SiegeNotebook(nb_path)
    _set_focus(nb, 1, "defeat_orc_mage")
    nb.apply_llm_update(
        2, _mature_profile({"defeat_orc_mage": 9.0, "defeat_troll": 3.0}),
        {"foci": [{"skill": "defeat_orc_mage"}, {"skill": "defeat_troll"}]},
        num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    assert nb.focus_skills() == ["defeat_orc_mage"]  # 2nd wall REFUSED (no slack yet)


def test_second_focus_allowed_once_expand_sr_reached(nb_path):
    # Once the first focus is going well (>= FOCUS_EXPAND_SR), there is slack to open a second wall.
    nb = SiegeNotebook(nb_path)
    _set_focus(nb, 1, "defeat_orc_mage")
    nb.apply_llm_update(
        2, _mature_profile({"defeat_orc_mage": FOCUS_EXPAND_SR + 5, "defeat_troll": 3.0}),
        {"foci": [{"skill": "defeat_orc_mage"}, {"skill": "defeat_troll"}]},
        num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    assert nb.focus_skills() == ["defeat_orc_mage", "defeat_troll"]  # both walls active


def test_at_most_max_focus_parallel_walls(nb_path):
    # Even with all existing foci going well (>= expand SR), no more than MAX_FOCUS (3) may be active.
    from auction.siege_notebook import MAX_FOCUS
    nb = SiegeNotebook(nb_path)
    walls = ["defeat_orc_mage", "defeat_troll", "defeat_gnome_warrior", "defeat_skeleton"]
    # session 1: open the first wall.
    nb.apply_llm_update(1, _mature_profile({walls[0]: 8.0}), {"focus": walls[0]},
                        num_snapshots=MATURITY_MIN_SNAPSHOTS)
    # keep every active wall above the expand SR AND rising (so none stalls out), proposing all four.
    # v6fix7: stay BELOW mastered_sr (70) — holding at mastered for 2 snapshots now CONQUERS a wall
    # and retires it gracefully, which is not what this cap test is about.
    for i, s in enumerate(range(2, 6)):
        base = FOCUS_EXPAND_SR + 1 + i * 4  # 51, 55, 59, 63: rising, above expand, below mastered
        prof = _mature_profile({w: base for w in walls})
        nb.apply_llm_update(s, prof, {"foci": [{"skill": w} for w in walls]},
                            num_snapshots=MATURITY_MIN_SNAPSHOTS)
    assert len(nb.focus_skills()) == MAX_FOCUS  # capped at 3, the 4th never gets a slot


def test_stalled_focus_retires_and_frees_slot(nb_path):
    # v6fix7 P1a: retirement now fires after LADDER_L4 consecutive FROZEN sessions (whole-tree
    # no-progress), not the legacy stall ratchet; it is still never retired by any SR threshold.
    # After retirement a DIFFERENT wall can be opened at once (the retired one is in cooldown).
    from auction.siege_notebook import LADDER_L4
    nb = SiegeNotebook(nb_path)
    _set_focus(nb, 1, "defeat_orc_mage")
    for s in range(2, 2 + LADDER_L4 + 1):
        nb.apply_llm_update(
            s, _mature_profile({"defeat_orc_mage": 8.0}), None, num_snapshots=MATURITY_MIN_SNAPSHOTS,
        )
    assert nb.focus_skills() == []  # frozen through the ladder -> retired, slot free
    nb.apply_llm_update(
        99, _mature_profile({"defeat_troll": 3.0}),
        {"foci": [{"skill": "defeat_troll"}]}, num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    assert nb.focus_skills() == ["defeat_troll"]


def test_improvement_resets_stall(nb_path):
    nb = SiegeNotebook(nb_path)
    _set_focus(nb, 1, "defeat_orc_mage")
    # A few flat sessions...
    nb.apply_llm_update(2, _mature_profile({"defeat_orc_mage": 8.0}), None, num_snapshots=MATURITY_MIN_SNAPSHOTS)
    nb.apply_llm_update(3, _mature_profile({"defeat_orc_mage": 8.0}), None, num_snapshots=MATURITY_MIN_SNAPSHOTS)
    # ...then a real improvement resets the (per-focus) stall counter.
    nb.apply_llm_update(4, _mature_profile({"defeat_orc_mage": 30.0}), None, num_snapshots=MATURITY_MIN_SNAPSHOTS)
    assert nb.foci()[0]["stall_sessions"] == 0
    # defeat_orc_mage is still only 30% (< FOCUS_EXPAND_SR) so no slack to open a 2nd wall -> refused,
    # the primary focus is unchanged.
    nb.apply_llm_update(
        5, _mature_profile({"defeat_orc_mage": 30.0, "defeat_troll": 2.0}),
        {"foci": [{"skill": "defeat_orc_mage"}, {"skill": "defeat_troll"}]},
        num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    assert nb.focus_skills() == ["defeat_orc_mage"]


# ---- mastery flags are code-owned --------------------------------------------------------------

def test_link_flag_ignores_llm_claim_uses_sr(nb_path):
    nb = SiegeNotebook(nb_path)
    # LLM claims collect_diamond is a mastered enabler, but SR says it's near zero.
    nb.apply_llm_update(
        1,
        _mature_profile({"defeat_gnome_warrior": 5.0, "collect_diamond": 2.0}),
        {
            "focus": "defeat_gnome_warrior",
            "prereq_tree": [
                {"skill": "collect_diamond", "role": "enabler", "state": "CONSOLIDATED"},  # lie
            ],
        },
        num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    tree = nb.prereq_links()
    assert len(tree) == 1
    assert tree[0]["skill"] == "collect_diamond"
    assert tree[0]["state"] == "UNMASTERED"  # code overrode the LLM's CONSOLIDATED claim


def test_focus_not_duplicated_as_its_own_prereq(nb_path):
    nb = SiegeNotebook(nb_path)
    nb.apply_llm_update(
        1,
        _mature_profile({"defeat_troll": 4.0, "collect_iron": 80.0}),
        {
            "focus": "defeat_troll",
            "prereq_tree": [
                {"skill": "defeat_troll", "role": "self"},   # illegal self-link -> dropped
                {"skill": "collect_iron", "role": "gear"},
            ],
        },
        num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    skills = [l["skill"] for l in nb.prereq_links()]
    assert skills == ["collect_iron"]


# ---- §2① success experience is code-driven, incremental, categorised, deduped-by-target ---------

def test_experience_recorded_incrementally_and_protects(nb_path):
    # A combat focus with a prereq chain. The focus is NOT retired at any SR threshold; instead its
    # success experience is recorded (categorised combat_milestone) and target+links are protected.
    nb = SiegeNotebook(nb_path)
    nb.apply_llm_update(
        1,
        _mature_profile({"defeat_gnome_warrior": 12.0, "collect_diamond": 80.0, "make_diamond_sword": 78.0}),
        {
            "focus": "defeat_gnome_warrior",
            "prereq_tree": [
                {"skill": "collect_diamond", "role": "reach"},
                {"skill": "make_diamond_sword", "role": "gear"},
            ],
        },
        num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    # Second session: the focus survives (still active) and is recorded now that it has an SR reading.
    nb.apply_llm_update(
        2, _mature_profile({"defeat_gnome_warrior": 12.0, "collect_diamond": 80.0, "make_diamond_sword": 78.0}),
        None, num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    snap = nb.snapshot()
    assert nb.focus == "defeat_gnome_warrior"          # NOT retired by SR (only the ladder retires)
    assert len(snap["verified_chains"]) == 1
    chain = snap["verified_chains"][0]
    assert chain["target"] == "defeat_gnome_warrior"
    assert chain["category"] == "combat_milestone"     # §2.5 combat -> milestone
    assert chain["status"] == "progress"               # v6fix7 (#8): progress, NOT a conquest
    assert set(chain["links"]) == {"collect_diamond", "make_diamond_sword"}
    # v6fix7 (#8): nothing protected until the wall HOLDS at mastered for 2 consecutive snapshots —
    # fix4 poisoned protected/verified at 44% via exactly this path.
    assert snap["protected_set"] == []


def test_experience_dedup_by_target_updates_in_place(nb_path):
    # The same target rising 12% -> 12% (noise) records once; a real +RECORD_DELTA_PP rise UPDATES the
    # single entry (dedup by target), it does not append a near-duplicate.
    nb = SiegeNotebook(nb_path)
    prof0 = {"defeat_gnome_warrior": 12.0}
    nb.apply_llm_update(1, _mature_profile(prof0), {"focus": "defeat_gnome_warrior"}, num_snapshots=MATURITY_MIN_SNAPSHOTS)
    nb.apply_llm_update(2, _mature_profile(prof0), None, num_snapshots=MATURITY_MIN_SNAPSHOTS)  # records 12
    nb.apply_llm_update(3, _mature_profile({"defeat_gnome_warrior": 12.0 + RECORD_DELTA_PP - 1}), None,
                        num_snapshots=MATURITY_MIN_SNAPSHOTS)  # jitter < delta -> no update
    chains = nb.verified_chains()
    assert len(chains) == 1 and chains[0]["last_recorded_sr"] == 12.0
    # a real jump updates the SAME entry in place.
    nb.apply_llm_update(4, _mature_profile({"defeat_gnome_warrior": 40.0}), None, num_snapshots=MATURITY_MIN_SNAPSHOTS)
    chains = nb.verified_chains()
    assert len(chains) == 1
    assert chains[0]["last_recorded_sr"] == 40.0 and chains[0]["first_recorded_sr"] == 12.0


def test_enabler_focus_categorised_enabler(nb_path):
    # A non-combat (gear) focus records under the enabler category, not combat_milestone (§2.5).
    nb = SiegeNotebook(nb_path)
    prof = {"make_iron_pickaxe": 20.0}
    nb.apply_llm_update(1, _mature_profile(prof), {"focus": "make_iron_pickaxe"}, num_snapshots=MATURITY_MIN_SNAPSHOTS)
    nb.apply_llm_update(2, _mature_profile(prof), None, num_snapshots=MATURITY_MIN_SNAPSHOTS)
    chains = nb.verified_chains()
    assert len(chains) == 1 and chains[0]["category"] == "enabler"


# ---- §3.1 self-style note: the transferable attack know-how carried per target -------------------

def test_style_note_recorded_into_verified_chain(nb_path):
    # The LLM's style_note for an active focus is folded into that target's verified_chains entry.
    nb = SiegeNotebook(nb_path)
    prof = {"defeat_gnome_warrior": 12.0}
    nb.apply_llm_update(
        1, _mature_profile(prof),
        {"focus": "defeat_gnome_warrior", "style_note": "pull one gnome at a time; kite along wall"},
        num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    # 2nd session records the experience (focus now has an SR reading); note rides along.
    nb.apply_llm_update(
        2, _mature_profile(prof),
        {"focus": "defeat_gnome_warrior", "style_note": "pull one gnome at a time; kite along wall"},
        num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    chain = nb.verified_chains()[0]
    assert chain["style_note"] == "pull one gnome at a time; kite along wall"


def test_style_note_non_empty_updates_empty_keeps_prior(nb_path):
    # A fresh non-empty note OVERWRITES; a silent (empty) session KEEPS the prior know-how, so style
    # accumulates rather than being blanked.
    nb = SiegeNotebook(nb_path)
    base = {"defeat_gnome_warrior": 12.0}
    nb.apply_llm_update(1, _mature_profile(base),
                        {"focus": "defeat_gnome_warrior", "style_note": "note-A"},
                        num_snapshots=MATURITY_MIN_SNAPSHOTS)
    nb.apply_llm_update(2, _mature_profile(base),
                        {"focus": "defeat_gnome_warrior", "style_note": "note-A"},
                        num_snapshots=MATURITY_MIN_SNAPSHOTS)
    assert nb.verified_chains()[0]["style_note"] == "note-A"
    # a real SR jump + refined note updates the note in place.
    nb.apply_llm_update(3, _mature_profile({"defeat_gnome_warrior": 40.0}),
                        {"focus": "defeat_gnome_warrior", "style_note": "note-B refined"},
                        num_snapshots=MATURITY_MIN_SNAPSHOTS)
    assert nb.verified_chains()[0]["style_note"] == "note-B refined"
    # another real jump but EMPTY note this session -> prior note is kept, not erased.
    nb.apply_llm_update(4, _mature_profile({"defeat_gnome_warrior": 40.0 + RECORD_DELTA_PP + 1}),
                        {"focus": "defeat_gnome_warrior", "style_note": ""},
                        num_snapshots=MATURITY_MIN_SNAPSHOTS)
    assert nb.verified_chains()[0]["style_note"] == "note-B refined"


def test_style_note_rendered_into_prompt(nb_path):
    # The stored note must be rendered back for the next modeler turn (stored-but-unread == wasted).
    nb = SiegeNotebook(nb_path)
    prof = {"defeat_gnome_warrior": 12.0}
    nb.apply_llm_update(1, _mature_profile(prof),
                        {"focus": "defeat_gnome_warrior", "style_note": "kite-along-wall tactic"},
                        num_snapshots=MATURITY_MIN_SNAPSHOTS)
    text = nb.render_for_prompt()
    assert "kite-along-wall tactic" in text


def test_style_note_long_kept_intact(nb_path):
    # No length cap (user 2026-07-05): a long note is stored verbatim, not truncated.
    nb = SiegeNotebook(nb_path)
    prof = {"defeat_gnome_warrior": 12.0}
    long_note = "x" * 900
    nb.apply_llm_update(1, _mature_profile(prof),
                        {"focus": "defeat_gnome_warrior", "style_note": long_note},
                        num_snapshots=MATURITY_MIN_SNAPSHOTS)
    nb.apply_llm_update(2, _mature_profile(prof),
                        {"focus": "defeat_gnome_warrior", "style_note": long_note},
                        num_snapshots=MATURITY_MIN_SNAPSHOTS)
    assert nb.verified_chains()[0]["style_note"] == long_note


# ---- unmastered_links feeds the §3.4 gate, from LIVE SR ----------------------------------------

def test_unmastered_links_reflects_live_sr(nb_path):
    nb = SiegeNotebook(nb_path)
    nb.apply_llm_update(
        1,
        _mature_profile({"defeat_gnome_warrior": 5.0, "collect_diamond": 80.0, "place_torch": 10.0}),
        {
            "focus": "defeat_gnome_warrior",
            "prereq_tree": [
                {"skill": "collect_diamond", "role": "reach"},   # mastered
                {"skill": "place_torch", "role": "survive-dark"},  # unmastered
            ],
        },
        num_snapshots=MATURITY_MIN_SNAPSHOTS,
    )
    # place_torch is below MASTERED_SR -> in the unmastered set; collect_diamond is not.
    live = {"collect_diamond": 80.0, "place_torch": 10.0, "defeat_gnome_warrior": 5.0}
    unmastered = nb.unmastered_links(live)
    assert "place_torch" in unmastered
    assert "collect_diamond" not in unmastered
    # If place_torch later becomes mastered, the gate sees it drop out (live, not the stored flag).
    live2 = {"collect_diamond": 80.0, "place_torch": MASTERED_SR + 5, "defeat_gnome_warrior": 5.0}
    assert "place_torch" not in nb.unmastered_links(live2)
