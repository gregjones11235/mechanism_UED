"""SiegeNotebook — the modeler's PERSISTENT siege journal (v6, v6_design.md §3.5).

v6's central data structure. Where v5's StudentProfileLog is a passive time series the modeler
re-reads from scratch every session, the SiegeNotebook is a *growing* journal the modeler maintains
ACROSS sessions — like a human player's mental notes when stuck on a boss: which fight I'm currently
attacking, which prerequisites in its chain I've already nailed vs still can't do, which chains I've
already conquered, and which skills I must keep rehearsing so I don't forget them.

WHY PERSISTENT (v6_design.md §3.5, not re-derived each session):
  1. (c) co-occurrence evidence is sparse (tier3 SR ~10% -> few successful episodes per session) —
     must accumulate across sessions to be stable.
  2. conquering one hard fight (with its backtracked sub-goals) spans MANY sessions — must remember
     "which fight I'm attacking / which link of the chain / which links are already consolidated",
     else the curriculum thrashes back into v5's spread-thin disease.
  3. H1's "self-style" accumulation is itself memory-dependent.

A+B HYBRID (v6_design.md §3.5, user 2026-07-04):
  - B (CODE guarantees, anti-drift backstop): schema + cross-session persistence, the §3.2 SCOPE
    hard constraint (focus may NOT be an easy/saturated skill), per-link "consolidated/unmastered"
    flags computed by an SR THRESHOLD (NOT trusted from the LLM's self-report), and the minimum
    condition for switching focus (K consecutive non-improving sessions). These CANNOT be violated
    by the LLM.
  - A (LLM judgement, "player intuition"): reads the previous notebook + this session's new evidence
    and PROPOSES an updated notebook (schema-constrained). It does not rewrite from zero.

This module owns B. The LLM (A) produces a raw proposed-update dict; ``apply_llm_update`` runs it
through every B-layer rule before it is allowed to persist. So a hallucinated "focus = collect_wood"
or a "this link is mastered" claim that the SR data contradicts is silently corrected here, not
trusted.

Storage: a plain JSON file ``siege_notebook.json`` in the run's cwd (same place as
``student_profile_history.json`` / ``task_graph.graphml``), atomic-written so an interrupted write
can't corrupt it, and reloaded on resume.

NOTE ON SR SOURCE (v6_design.md §3.8): mastery flags this phase are computed from the held-out
AGGREGATE per-skill SR (from StudentProfileLog). The (c) per-episode co-occurrence signal does not
exist yet (evaluation throws away per-env multi-hot at .sum()); wiring it in is a later phase. The
mastery-flag computation is deliberately funnelled through ``_mastery_from_sr`` so that later phase
can enrich it without touching callers.
"""

from __future__ import annotations

import json
import os

_DEFAULT_PATH = "siege_notebook.json"

# --- B-layer thresholds (the "code guarantees") --------------------------------------------------
# These module constants are the DEFAULTS. Every one is now overridable per-run via config
# (dicode_manager.siege_* keys) — see SiegeNotebook.__init__ / SiegeThresholds and gen_manager, which
# passes the config in. The constants stay as the fallback so tests and any code that constructs a
# SiegeNotebook() with no thresholds keep the documented default behaviour byte-for-byte.
# ★ These are the highest-leverage knobs (user 2026-07-05): MASTERED_SR gates link compression,
#   and MATURITY_* gates whether a siege ever opens.

# A prereq LINK is CONSOLIDATED/MASTERED (safe to compress into Completed, per §3.4) only when its
# held-out SR is at/above this. Mid-chain links below this MUST stay Relevant (trained), never
# compressed — that was exactly the v5 chain-severing disease.
# NOTE: this judges a LINK ("has this prerequisite been learned"); it is INDEPENDENT of SATURATED_SR,
# which judges a FOCUS candidate ("is this skill too easy to be a wall"). They are different questions
# on different objects and need not share a value (user 2026-07-05).
MASTERED_SR = 70.0

# A skill is UNMASTERED (a live siege link that must be trained in-level) when its SR is at/below this.
# Between UNMASTERED_SR and MASTERED_SR is the "rising / in-progress" band.
UNMASTERED_SR = 20.0

# §3.2 SCOPE: the focus must be a genuine hard wall, NEVER an easy/saturated skill. A skill at/above
# this SR is "already-held ground", forbidden as a focus.
# ★ user 2026-07-05 (2nd round): combat has NO 40% ceiling (a human expert beats a hard boss near
#   100%), so "saturated / already ground" is a HIGH bar for BOTH combat and gear — set to 80, NOT
#   bound to MASTERED_SR. A combat skill at 40% is still a wall worth attacking, not saturated.
SATURATED_SR = 80.0

# --- EARLY-TRAINING guard (user 2026-07-04): "low SR" has TWO causes and only ONE is a real wall ---
# (A) genuine wall: student PAST the early phase, prerequisites in place, STILL can't do it (siege it).
# (B) early-immature: whole student still weak, this skill just hasn't had its turn (NORMAL_EARLY —
#     never siege it). Until the student crosses a maturity threshold, NO focus may be set (a low-SR
#     skill early is immaturity, not a wall). Judged purely from SR statistics — no tier table.
MATURITY_MIN_SNAPSHOTS = 4     # fewer snapshots than this = too early to call anything a wall
# ★ user 2026-07-05: relaxed so the siege can OPEN in the real tier3~low-ceiling regime (the old
#   12×50 could mean the siege never opens). A "decent" skill now counts at 35, and 10 of them suffice.
# ★ user 2026-07-05, from REAL v5 curve (dicode-v5yA-s0-r2, job 3640867): tier1 has 15 skills at
#   SR>=50 by session ~9 and tier2 climbs to ~10 by session ~17 (the burst->plateau knee, return ~30).
#   20 = 15(tier1) + ~5(tier2) at SR>=50 -> siege opens right at that knee, once the base is laid.
MATURITY_MIN_MASTERED = 20     # fewer skills at/above MATURITY_SKILL_SR = still in the early ramp
MATURITY_SKILL_SR = 50.0       # a skill counts toward maturity once it is at/above this SR

# §3.5 focus-switch minimum condition: switch only after the focus has gone this many consecutive
# recorded sessions without improving (anti-thrash guard).
# v6fix7 P1a: retained for the LEGACY stall counter (logging/back-compat); ladder escalation and
# retirement are driven by the FROZEN counter below, not by this.
FOCUS_MIN_STALL_SESSIONS = 10

# How much the focus's SR must rise between two recorded sessions to count as "improving" (resets the
# stall counter). Small, to tolerate eval noise but still register real progress.
FOCUS_IMPROVE_PP = 3.0

# --- v6fix7 P1a: ESCALATION LADDER (adaptive patience; user 2026-07-06) ----------------------------
# The old binary retire/reopen loop watched ONLY the focus's own SR: a tier4 wall whose SR sits at 0
# while its foundations climb was mis-read as "stalled", and fix4's iron-pickaxe siege showed the
# retire->same-session-reopen cycle. The ladder counts FROZEN sessions instead — a session is frozen
# only when the WHOLE attack tree shows no progress (focus SR flat AND every link flat AND no
# chain-frontier advance from the failure-episode signal). Patience is therefore unbounded while
# anything measurable is moving; the ladder escalates the response only on total freeze:
#   L1 (>= LADDER_L1 frozen): asked to switch form OR defend staying with a NEW plan (accountability);
#   L2 (>= LADDER_L2): form switch FORCED (DEPTH<->CONSOLIDATE) — code overrides;
#   L3 (>= LADDER_L3): tactic revision FORCED (style_note must be materially different);
#   L4 (>= LADDER_L4): retire + cooldown; repeated retirements blacklist until NEW chain evidence.
LADDER_L1 = 3
LADDER_L2 = 6
LADDER_L3 = 9
LADDER_L4 = 12
# Windowed-slope progress signal (defeats the fix4 best-SR ratchet: oscillation under an old best is
# NOT progress; a genuine climb shows a positive recent slope even without a new best).
SLOPE_WINDOW = 6           # readings in the regression window
SLOPE_MIN_PP = 1.0         # pp/session; slope above this counts as live progress
# Retirement aftermath.
COOLDOWN_SESSIONS = 6      # a retired wall may not reopen for this many sessions
BLACKLIST_RETIREMENTS = 2  # retired this many times -> blacklisted until new chain evidence
# Conquest (#8 fix): verified_chains "verified" status + protected_set admission require the wall to
# HOLD at mastered_sr for this many consecutive snapshots — a one-session +delta record is progress,
# never conquest (fix4 wrote make_iron_pickaxe into verified_chains/protected at 44%).
CONQUEST_CONSECUTIVE = 2
# v6fix7 P1c (AutoManual-lite): a style_note that goes this many sessions without the modeler's
# evidence_check reporting "supported" is marked STALE — the journal then demands re-derivation
# instead of further refinement of an unverified guess (self-reinforcing-tactic loop breaker).
NOTE_STALE_SESSIONS = 4

# --- §2.6 MULTI-FOCUS (user 2026-07-05, 3rd round): parallel sieges + cross-achievement assist ------
# The siege may attack up to MAX_FOCUS walls at once. A NEW focus may be opened only once ANY existing
# focus has reached FOCUS_EXPAND_SR — "one is going well, so there's slack to also carry a new one".
# The new focus may be ANY tier3+ achievement (incl. gear: upgrading gear helps later attacks).
MAX_FOCUS = 3
FOCUS_EXPAND_SR = 50.0

# --- §2① incremental success-experience recording (user 2026-07-05, 2nd round) --------------------
# CONQUERED is NOT a one-shot 80% retire gate anymore (conquered != saturated: a wall going 15%->40%
# is real progress worth recording). Instead, whenever a focus's SR rises by >= RECORD_DELTA_PP versus
# the LAST time it was recorded, we write/UPDATE (dedup by target) its success-experience entry. Noise
# jitter below this delta does NOT record -> no redundant near-duplicate entries. Focus retirement is
# governed purely by stall (FOCUS_MIN_STALL_SESSIONS), never by hitting an SR threshold.
RECORD_DELTA_PP = 10.0

# --- §2⑥ rehearsal FORGETTING prefilter thresholds (family-split, user 2026-07-05) ----------------
# A skill is a rehearsal candidate once it peaked >= *_MIN_PEAK then fell >= *_DROP_PP. COMBAT
# milestones peak LOW (~15%), so they get a low bar — else a forgetting combat milestone (exactly what
# §2.5 records and §3.6 must protect) would never be flagged. Gear/gather/explore keep the higher bar.
FORGETTING_MIN_PEAK = 40.0          # non-combat peak bar
FORGETTING_DROP_PP = 20.0           # non-combat drop bar
FORGETTING_COMBAT_MIN_PEAK = 15.0   # combat milestones peak low -> low bar (=combat record floor)
FORGETTING_COMBAT_DROP_PP = 10.0    # a 10pp slip off a low plateau is already a real slide


class SiegeThresholds:
    """All B-layer numeric thresholds for one SiegeNotebook, resolved from config (or defaults).

    Grouping them in one object keeps ``SiegeNotebook`` methods reading ``self.th.mastered_sr`` etc.
    instead of module globals, so a run's ``dicode_manager`` config can tune every knob. ``from_config``
    reads the ``siege_*`` keys off a config-manager object (any object with attributes / .get), falling
    back to the module-constant default for each missing key — so partial config or no config is fine.
    """

    __slots__ = (
        "mastered_sr", "unmastered_sr", "saturated_sr",
        "maturity_min_snapshots", "maturity_min_mastered", "maturity_skill_sr",
        "focus_min_stall_sessions", "focus_improve_pp",
        # §2.6 multi-focus + §2① incremental recording (user 2026-07-05)
        "max_focus", "focus_expand_sr", "record_delta_pp",
        # §2⑥ family-split rehearsal FORGETTING prefilter (user 2026-07-05)
        "forgetting_min_peak", "forgetting_drop_pp",
        "forgetting_combat_min_peak", "forgetting_combat_drop_pp",
        # v6fix7 P1a escalation ladder + conquest
        "ladder_l1", "ladder_l2", "ladder_l3", "ladder_l4",
        "slope_window", "slope_min_pp",
        "cooldown_sessions", "blacklist_retirements", "conquest_consecutive",
        # v6fix7 P1c style_note lifecycle
        "note_stale_sessions",
    )

    def __init__(
        self,
        mastered_sr=MASTERED_SR,
        unmastered_sr=UNMASTERED_SR,
        saturated_sr=None,
        maturity_min_snapshots=MATURITY_MIN_SNAPSHOTS,
        maturity_min_mastered=MATURITY_MIN_MASTERED,
        maturity_skill_sr=MATURITY_SKILL_SR,
        focus_min_stall_sessions=FOCUS_MIN_STALL_SESSIONS,
        focus_improve_pp=FOCUS_IMPROVE_PP,
        max_focus=MAX_FOCUS,
        focus_expand_sr=FOCUS_EXPAND_SR,
        record_delta_pp=RECORD_DELTA_PP,
        forgetting_min_peak=FORGETTING_MIN_PEAK,
        forgetting_drop_pp=FORGETTING_DROP_PP,
        forgetting_combat_min_peak=FORGETTING_COMBAT_MIN_PEAK,
        forgetting_combat_drop_pp=FORGETTING_COMBAT_DROP_PP,
        ladder_l1=LADDER_L1,
        ladder_l2=LADDER_L2,
        ladder_l3=LADDER_L3,
        ladder_l4=LADDER_L4,
        slope_window=SLOPE_WINDOW,
        slope_min_pp=SLOPE_MIN_PP,
        cooldown_sessions=COOLDOWN_SESSIONS,
        blacklist_retirements=BLACKLIST_RETIREMENTS,
        conquest_consecutive=CONQUEST_CONSECUTIVE,
        note_stale_sessions=NOTE_STALE_SESSIONS,
    ):
        self.mastered_sr = float(mastered_sr)
        self.unmastered_sr = float(unmastered_sr)
        # saturated now defaults to the module SATURATED_SR (80, user 2026-07-05), NOT mastered_sr.
        self.saturated_sr = float(saturated_sr if saturated_sr is not None else SATURATED_SR)
        self.maturity_min_snapshots = int(maturity_min_snapshots)
        self.maturity_min_mastered = int(maturity_min_mastered)
        self.maturity_skill_sr = float(maturity_skill_sr)
        self.focus_min_stall_sessions = int(focus_min_stall_sessions)
        self.focus_improve_pp = float(focus_improve_pp)
        # §2.6 multi-focus + §2① incremental recording.
        self.max_focus = int(max_focus)
        self.focus_expand_sr = float(focus_expand_sr)
        self.record_delta_pp = float(record_delta_pp)
        # §2⑥ family-split forgetting prefilter.
        self.forgetting_min_peak = float(forgetting_min_peak)
        self.forgetting_drop_pp = float(forgetting_drop_pp)
        self.forgetting_combat_min_peak = float(forgetting_combat_min_peak)
        self.forgetting_combat_drop_pp = float(forgetting_combat_drop_pp)
        # v6fix7 P1a escalation ladder + conquest.
        self.ladder_l1 = int(ladder_l1)
        self.ladder_l2 = int(ladder_l2)
        self.ladder_l3 = int(ladder_l3)
        self.ladder_l4 = int(ladder_l4)
        self.slope_window = int(slope_window)
        self.slope_min_pp = float(slope_min_pp)
        self.cooldown_sessions = int(cooldown_sessions)
        self.blacklist_retirements = int(blacklist_retirements)
        self.conquest_consecutive = int(conquest_consecutive)
        self.note_stale_sessions = int(note_stale_sessions)

    @classmethod
    def from_config(cls, dm) -> "SiegeThresholds":
        """Build from a dicode_manager config object; each missing ``siege_*`` key keeps its default."""
        def g(key, default):
            if dm is None:
                return default
            # Prefer .get (OmegaConf DictConfig / plain dict both support it, and it won't raise on a
            # missing key the way OmegaConf attribute access does).
            getter = getattr(dm, "get", None)
            if callable(getter):
                try:
                    val = getter(key, default)
                    return val if val is not None else default
                except Exception:  # noqa: BLE001
                    return default
            val = getattr(dm, key, default)
            return val if val is not None else default
        return cls(
            mastered_sr=g("siege_mastered_sr", MASTERED_SR),
            unmastered_sr=g("siege_unmastered_sr", UNMASTERED_SR),
            saturated_sr=g("siege_saturated_sr", None),
            maturity_min_snapshots=g("siege_maturity_min_snapshots", MATURITY_MIN_SNAPSHOTS),
            maturity_min_mastered=g("siege_maturity_min_mastered", MATURITY_MIN_MASTERED),
            maturity_skill_sr=g("siege_maturity_skill_sr", MATURITY_SKILL_SR),
            focus_min_stall_sessions=g("siege_focus_min_stall_sessions", FOCUS_MIN_STALL_SESSIONS),
            focus_improve_pp=g("siege_focus_improve_pp", FOCUS_IMPROVE_PP),
            max_focus=g("siege_max_focus", MAX_FOCUS),
            focus_expand_sr=g("siege_focus_expand_sr", FOCUS_EXPAND_SR),
            record_delta_pp=g("siege_record_delta_pp", RECORD_DELTA_PP),
            forgetting_min_peak=g("siege_forgetting_min_peak", FORGETTING_MIN_PEAK),
            forgetting_drop_pp=g("siege_forgetting_drop_pp", FORGETTING_DROP_PP),
            forgetting_combat_min_peak=g("siege_forgetting_combat_min_peak", FORGETTING_COMBAT_MIN_PEAK),
            forgetting_combat_drop_pp=g("siege_forgetting_combat_drop_pp", FORGETTING_COMBAT_DROP_PP),
            ladder_l1=g("siege_ladder_l1", LADDER_L1),
            ladder_l2=g("siege_ladder_l2", LADDER_L2),
            ladder_l3=g("siege_ladder_l3", LADDER_L3),
            ladder_l4=g("siege_ladder_l4", LADDER_L4),
            slope_window=g("siege_slope_window", SLOPE_WINDOW),
            slope_min_pp=g("siege_slope_min_pp", SLOPE_MIN_PP),
            cooldown_sessions=g("siege_cooldown_sessions", COOLDOWN_SESSIONS),
            blacklist_retirements=g("siege_blacklist_retirements", BLACKLIST_RETIREMENTS),
            conquest_consecutive=g("siege_conquest_consecutive", CONQUEST_CONSECUTIVE),
            note_stale_sessions=g("siege_note_stale_sessions", NOTE_STALE_SESSIONS),
        )


# Mastery-flag vocabulary attached to every prereq-tree link.
LINK_STATES = ("CONSOLIDATED", "RISING", "UNMASTERED", "UNKNOWN")


def mastery_from_sr(
    sr: float | None,
    mastered_sr: float = MASTERED_SR,
    unmastered_sr: float = UNMASTERED_SR,
) -> str:
    """Map a held-out SR (0..100) to a link mastery flag by the B-layer thresholds.

    Single source of truth for "consolidated vs unmastered" — the LLM's self-report is never trusted.
    Thresholds default to the module constants but callers (SiegeNotebook) pass their config-resolved
    values so the flags honour a tuned MASTERED_SR/UNMASTERED_SR.
    """
    if sr is None:
        return "UNKNOWN"
    if sr >= mastered_sr:
        return "CONSOLIDATED"
    if sr <= unmastered_sr:
        return "UNMASTERED"
    return "RISING"


def _empty_notebook() -> dict:
    """The guaranteed-valid empty notebook schema (session 0, nothing sieged yet).

    §2.6 (user 2026-07-05): the single top-level ``focus`` became a ``foci`` LIST (up to MAX_FOCUS
    parallel sieges). Each focus element carries its OWN per-focus bookkeeping + prereq tree:
        {"skill", "started_session", "best_sr", "stall_sessions",
         "last_recorded_sr",          # §2① SR at the last success-experience record (dedup/delta base)
         "prereq_tree": [ {"skill","state","sr","role"}, ... ]}
    """
    return {
        "foci": [],                  # list of focus dicts (see docstring); empty = no active siege
        "verified_chains": [],       # §2.5 success-experience entries (dedup by target; categorised)
        "protected_set": [],         # skill names to rehearse (§3.6; v6fix7: VERIFIED conquests only)
        "last_session": None,        # last session index this notebook was updated at
        "history": [],               # append-only log of focus changes, for offline inspection
        # v6fix7 P1a: retirement registry — skill -> {count, last_session, sr_at_retirement,
        # link_sr_at_retirement, failed_notes[<=3]}. Drives cooldown / blacklist / "what's different
        # this time" checks in _reconcile_foci, and is rendered to the modeler (history made READABLE).
        "retired": {},
    }


def _empty_focus(skill: str, session_idx: int, sr: float | None) -> dict:
    """A fresh focus element for ``skill`` newly opened at ``session_idx``."""
    return {
        "skill": skill,
        "started_session": session_idx,
        "best_sr": sr,
        "stall_sessions": 0,         # legacy ratchet counter (logging/back-compat only)
        "last_recorded_sr": None,    # §2① no success-experience recorded for it yet
        "prereq_tree": [],           # backtracked chain, filled by the LLM proposal + code flags
        "style_note": "",            # §3.1 self-style: LLM free-text attack know-how for this wall
        # v6fix7 P1a ladder state:
        "sr_history": [] if sr is None else [float(sr)],  # recent focus SR readings (slope signal)
        "link_best": {},             # per-link best SR seen (foundation-progress signal)
        "frozen_sessions": 0,        # consecutive whole-tree no-progress sessions (drives the ladder)
        "ladder_level": 0,           # 0 none / 1 defend-or-switch / 2 forced form / 3 forced tactic
        "last_siege_type": "",       # TYPE of the latest siege level built for this wall (from level_meta)
        "consecutive_mastered": 0,   # consecutive snapshots at/above mastered_sr (conquest, #8 fix)
        # v6fix7 P1c style_note lifecycle (AutoManual-lite):
        "note_status": "active",     # active | stale (long unsupported) | contradicted (must rewrite)
        "note_last_supported_session": None,  # last session evidence_check said "supported"
    }


class SiegeNotebook:
    """Persistent siege journal. Owns the B-layer hard constraints; the LLM (A) only proposes."""

    def __init__(self, path: str | None = None, thresholds: "SiegeThresholds | None" = None):
        self.path = path or _DEFAULT_PATH
        # B-layer thresholds: config-resolved when passed, else the module-constant defaults.
        self.th = thresholds or SiegeThresholds()
        self._nb: dict = self._load()
        # Diagnostics for the caller to log (why a focus was/wasn't set; whether a conquest fired this
        # session). Set fresh each apply_llm_update; None between calls.
        self.last_focus_decision: str | None = None
        self.last_conquest: str | None = None

    # ---- persistence (mirrors StudentProfileLog: atomic write, resume-safe) --------------------

    def _load(self) -> dict:
        if not os.path.exists(self.path):
            return _empty_notebook()
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            return self._coerce(data)
        except (json.JSONDecodeError, OSError):
            return _empty_notebook()

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._nb, f, indent=1)
        os.replace(tmp, self.path)

    @staticmethod
    def _coerce(data: object) -> dict:
        """Force a loaded object into the guaranteed schema, filling any missing key with its empty.

        §2.6 backward-compat: a notebook written by the OLD single-focus schema (top-level ``focus``
        string + ``focus_*`` fields + a single ``prereq_tree``) is migrated into the new ``foci`` list
        so old on-disk journals / old unit-test fixtures don't break.
        """
        base = _empty_notebook()
        if not isinstance(data, dict):
            return base
        for key in base:
            if key in data:
                base[key] = data[key]
        # migrate an old single-focus notebook that predates the foci list.
        if not data.get("foci") and data.get("focus"):
            migrated = _empty_focus(
                str(data["focus"]).lower(),
                data.get("focus_started_session"),
                data.get("focus_best_sr"),
            )
            migrated["stall_sessions"] = int(data.get("focus_stall_sessions", 0) or 0)
            tree = data.get("prereq_tree")
            if isinstance(tree, list):
                migrated["prereq_tree"] = tree
            base["foci"] = [migrated]
        return base

    # ---- read accessors ------------------------------------------------------------------------

    def snapshot(self) -> dict:
        """A deep-ish copy of the current notebook (safe to hand to prompt builders / tests)."""
        return json.loads(json.dumps(self._nb))

    def foci(self) -> list[dict]:
        """All active focus elements (§2.6). Empty list = no active siege."""
        return list(self._nb.get("foci", []))

    def focus_skills(self) -> list[str]:
        """Just the skill names of the active foci, in order (primary first)."""
        return [f["skill"] for f in self._nb.get("foci", []) if isinstance(f.get("skill"), str)]

    @property
    def focus(self) -> str | None:
        """Backward-compat single-focus accessor: the PRIMARY (first) active focus, or None.

        Kept so the many `.focus` read sites (logging, co-occurrence hint) keep working; new code that
        needs all parallel sieges should use ``focus_skills()`` / ``foci()``.
        """
        foci = self._nb.get("foci", [])
        return foci[0]["skill"] if foci else None

    def protected_set(self) -> list[str]:
        return list(self._nb.get("protected_set", []))

    def verified_chains(self) -> list[dict]:
        return list(self._nb.get("verified_chains", []))

    # ---- v6fix7 P1a: ladder accessors + external progress/type signals --------------------------

    def retired_registry(self) -> dict:
        """skill -> retirement record (count / last_session / failed_notes / SR snapshots)."""
        return dict(self._nb.get("retired", {}))

    def ladder_level_of(self, skill: str) -> int:
        """Current escalation-ladder level (0-3) of an active focus; 0 if not an active focus."""
        for foc in self._nb.get("foci", []):
            if str(foc.get("skill", "")).lower() == str(skill).lower():
                return int(foc.get("ladder_level", 0))
        return 0

    def required_form(self, skill: str) -> str | None:
        """L2 forced form switch: the level TYPE siege levels for this wall MUST use now.

        Returns the OPPOSITE of the form that froze (DEPTH<->CONSOLIDATE) once the focus is at
        ladder level >= 2 and we know which form was being used; None otherwise (no constraint).
        """
        for foc in self._nb.get("foci", []):
            if str(foc.get("skill", "")).lower() != str(skill).lower():
                continue
            if int(foc.get("ladder_level", 0)) < 2:
                return None
            last = str(foc.get("last_siege_type", "")).upper()
            if last == "CONSOLIDATE":
                return "DEPTH"
            if last == "DEPTH":
                return "CONSOLIDATE"
            return None  # form unknown -> cannot force a flip
        return None

    def note_siege_level_type(self, skill: str, level_type: str) -> None:
        """Record the TYPE of the latest siege level built for this wall (from <level_meta>).

        This is what makes the L2 forced flip computable — without knowing which form has been
        attacking the wall, "switch form" is unenforceable.
        """
        changed = False
        for foc in self._nb.get("foci", []):
            if str(foc.get("skill", "")).lower() == str(skill).lower():
                lt = str(level_type or "").upper()
                if lt and foc.get("last_siege_type") != lt:
                    foc["last_siege_type"] = lt
                    changed = True
        if changed:
            self._save()

    def note_chain_progress(self, skill: str, advanced: bool = True) -> None:
        """P2 chain-frontier signal: failure episodes are dying DEEPER along this wall's chain.

        For an ACTIVE focus: consumed (and cleared) by the next _update_focus_stall — an advance
        counts as live progress, so the ladder never escalates while the student is measurably
        getting closer (SR may still be 0; this is the tier4 patience signal).
        For a RETIRED wall: recorded on its retirement-registry entry, where _has_new_evidence reads
        it as the blacklist escape hatch (design §P1a: "blacklisted until new chain evidence OR the
        break-link frontier advances"). Cleared on the wall's NEXT retirement (stale by then)."""
        if not advanced:
            return
        for foc in self._nb.get("foci", []):
            if str(foc.get("skill", "")).lower() == str(skill).lower():
                foc["chain_frontier_advanced"] = True
                self._save()
                return
        for rskill, reg in (self._nb.get("retired") or {}).items():
            if str(rskill).lower() == str(skill).lower() and isinstance(reg, dict):
                reg["chain_frontier_advanced"] = True
                self._save()
                return

    def _is_conquered_and_held(self, skill: str, latest_profile: dict[str, float]) -> bool:
        """True when this skill has a VERIFIED conquest entry and its current SR still holds at
        mastered — conquered ground is not a wall; if it later slips, sieging it again is legal."""
        for c in self._nb.get("verified_chains", []):
            if isinstance(c, dict) and str(c.get("target", "")).lower() == str(skill).lower() \
                    and c.get("status") == "verified":
                sr = latest_profile.get(str(skill).lower())
                return sr is not None and sr >= self.th.mastered_sr
        return False

    def _has_new_evidence(self, skill: str, reg: dict, latest_profile: dict[str, float]) -> bool:
        """Blacklist escape hatch: has ANYTHING on this wall's chain moved since retirement?

        True if the wall's own SR, or any link SR snapshotted at retirement, has risen by
        >= focus_improve_pp — or if the P2 break-link frontier advanced (failure episodes dying
        measurably deeper along the chain, delivered via note_chain_progress; SR may still be 0) —
        new evidence earns the wall another attempt."""
        if reg.get("chain_frontier_advanced"):
            return True
        base_sr = reg.get("sr_at_retirement")
        cur = latest_profile.get(str(skill).lower())
        if cur is not None and base_sr is not None and cur >= float(base_sr) + self.th.focus_improve_pp:
            return True
        for lskill, lsr0 in (reg.get("link_sr_at_retirement") or {}).items():
            cur_l = latest_profile.get(str(lskill).lower())
            if cur_l is not None and lsr0 is not None and cur_l >= float(lsr0) + self.th.focus_improve_pp:
                return True
        return False

    def chain_targets(self) -> dict[str, list[str]]:
        """P2: the walls whose failure episodes the ChainOrderLog should break-link-mine, each with
        its ORDERED prereq chain (shallow -> deep): every ACTIVE focus (prereq_tree order) plus every
        RETIRED wall (its chain snapshotted at retirement — kept under watch so a frontier advance
        can unlock the blacklist via note_chain_progress even while nobody is sieging it). Retired
        entries predating the snapshot field fall back to link_sr_at_retirement's key order (built in
        prereq_tree order, dict order preserved)."""
        out: dict[str, list[str]] = {}
        for foc in self._nb.get("foci", []):
            skill = foc.get("skill")
            if isinstance(skill, str) and skill:
                out[skill.lower()] = [
                    str(l.get("skill")).lower()
                    for l in foc.get("prereq_tree", []) if isinstance(l.get("skill"), str)
                ]
        for rskill, reg in (self._nb.get("retired") or {}).items():
            sl = str(rskill).lower()
            if sl in out or not isinstance(reg, dict):
                continue
            links = reg.get("links_at_retirement")
            if not links:
                links = list((reg.get("link_sr_at_retirement") or {}).keys())
            out[sl] = [str(l).lower() for l in links]
        return out

    def prereq_links(self) -> list[dict]:
        """Prereq-tree links across ALL active foci (deduped by skill), each with ``skill``/``role``.

        §2.6: was the single focus's tree; now the union over foci. Dedup keeps the first occurrence
        so a shared enabler (e.g. make_iron_pickaxe) appears once even if several foci list it.
        """
        out: list[dict] = []
        seen: set[str] = set()
        for foc in self._nb.get("foci", []):
            for link in foc.get("prereq_tree", []):
                skill = link.get("skill")
                if isinstance(skill, str) and skill.lower() not in seen:
                    seen.add(skill.lower())
                    out.append(link)
        return out

    def unmastered_links(self, latest_profile: dict[str, float]) -> set[str]:
        """Skills across ALL active foci's prereq trees that are NOT yet consolidated (per live SR).

        §2.6: the §3.4 Completed-admission gate wants the UNION over every active siege — any such link
        appearing in a proposer's ``Completed`` list is an illegal compression and must be pulled back
        to ``Relevant``. Computed from LIVE SR (not the stored flag) so it is always current even
        between LLM updates. A set, so a shared enabler across foci is naturally deduped.
        """
        out: set[str] = set()
        for foc in self._nb.get("foci", []):
            for link in foc.get("prereq_tree", []):
                skill = link.get("skill")
                if not isinstance(skill, str):
                    continue
                sr = latest_profile.get(skill.lower())
                if mastery_from_sr(sr, self.th.mastered_sr, self.th.unmastered_sr) != "CONSOLIDATED":
                    out.add(skill.lower())
        return out

    # ---- B-layer: recompute link flags from live SR (never trust the LLM for this) -------------

    def _refresh_link_flags(self, latest_profile: dict[str, float]) -> None:
        """Overwrite every focus's prereq-tree link ``state``/``sr`` from the current held-out profile."""
        for foc in self._nb.get("foci", []):
            for link in foc.get("prereq_tree", []):
                skill = link.get("skill")
                if not isinstance(skill, str):
                    link["state"] = "UNKNOWN"
                    link["sr"] = None
                    continue
                sr = latest_profile.get(skill.lower())
                link["sr"] = sr
                link["state"] = mastery_from_sr(sr, self.th.mastered_sr, self.th.unmastered_sr)

    # ---- B-layer: focus scope + stall bookkeeping ----------------------------------------------

    def _is_valid_focus(self, skill: object, latest_profile: dict[str, float]) -> bool:
        """§3.2 SCOPE hard constraint: a legal focus is a real wall, not an easy/saturated skill.

        A focus must (a) be a known achievement name (str), and (b) NOT already be at/above the
        saturation SR. We do NOT here require it to be combat-only or tier3 — the modeler picks WHICH
        wall from evidence (§3.2); code only forbids wasting the siege on ground already held.
        """
        if not isinstance(skill, str) or not skill:
            return False
        sr = latest_profile.get(skill.lower())
        if sr is not None and sr >= self.th.saturated_sr:
            return False  # already mastered ground — not a wall, forbidden as focus
        return True

    def _student_is_mature(self, latest_profile: dict[str, float], num_snapshots: int) -> bool:
        """EARLY-TRAINING guard: is the student past the early ramp so a low SR can mean a WALL?

        Pure SR statistics, no tier table (user 2026-07-04). Immature => low SR is NORMAL_EARLY (B),
        not a wall (A), and NO focus may be set. Mature once there are enough snapshots to read a
        trend AND enough skills already at a decent SR that "still weak everywhere" is ruled out.
        """
        if num_snapshots < self.th.maturity_min_snapshots:
            return False
        n_decent = sum(
            1 for v in latest_profile.values() if v is not None and v >= self.th.maturity_skill_sr
        )
        return n_decent >= self.th.maturity_min_mastered

    @staticmethod
    def _slope(readings: list[float]) -> float:
        """Least-squares slope (pp per session) of a short SR window. <2 points -> 0."""
        pts = [float(r) for r in readings if isinstance(r, (int, float))]
        n = len(pts)
        if n < 2:
            return 0.0
        xm = (n - 1) / 2.0
        ym = sum(pts) / n
        denom = sum((i - xm) ** 2 for i in range(n))
        if denom == 0:
            return 0.0
        return sum((i - xm) * (pts[i] - ym) for i in range(n)) / denom

    @staticmethod
    def _notes_similar(a: str, b: str, threshold: float = 0.7) -> bool:
        """Token-Jaccard similarity check for 'is this tactic materially different?' (L3 / reopen)."""
        ta = {w for w in str(a or "").lower().split() if len(w) > 2}
        tb = {w for w in str(b or "").lower().split() if len(w) > 2}
        if not ta or not tb:
            return False  # an empty note is never 'the same tactic'
        return len(ta & tb) / len(ta | tb) >= threshold

    def _ladder_level(self, frozen: int) -> int:
        if frozen >= self.th.ladder_l3:
            return 3
        if frozen >= self.th.ladder_l2:
            return 2
        if frozen >= self.th.ladder_l1:
            return 1
        return 0

    def _update_focus_stall(self, latest_profile: dict[str, float]) -> None:
        """v6fix7 P1a: per-focus stall — FROZEN counter over the WHOLE attack tree (adaptive patience).

        A session is frozen for a focus only if NONE of three progress signals fires:
          (a) focus SR — new best (>= +improve_pp) OR positive windowed slope (> slope_min_pp/session;
              defeats the fix4 ratchet where oscillation under an old best deferred retirement);
          (b) any prereq link SR making a new best — foundations rising = the siege is biting even
              while the wall itself sits at 0% (tier4 patience, user 2026-07-06);
          (c) chain-frontier advance (P2 failure-episode signal) delivered via note_chain_progress().
        frozen_sessions drives the escalation ladder + retirement; the legacy stall_sessions ratchet
        is still maintained for logging/back-compat but decides nothing.
        Also tracks consecutive_mastered for the conquest gate (#8 fix).
        """
        for foc in self._nb.get("foci", []):
            skill = foc.get("skill")
            sr = latest_profile.get(skill.lower()) if isinstance(skill, str) else None
            prev_best = foc.get("best_sr")
            progress = False

            # --- legacy ratchet counter (logging only) ---
            if sr is None:
                foc["stall_sessions"] = int(foc.get("stall_sessions", 0)) + 1
            elif prev_best is None or sr >= prev_best + self.th.focus_improve_pp:
                foc["best_sr"] = sr
                foc["stall_sessions"] = 0
            else:
                foc["stall_sessions"] = int(foc.get("stall_sessions", 0)) + 1

            # --- (a) focus SR: new best or live windowed slope ---
            if sr is not None:
                hist = [h for h in foc.get("sr_history", []) if isinstance(h, (int, float))]
                hist.append(float(sr))
                foc["sr_history"] = hist[-max(2 * self.th.slope_window, 12):]
                if prev_best is None or sr >= prev_best + self.th.focus_improve_pp:
                    progress = True
                elif self._slope(foc["sr_history"][-self.th.slope_window:]) > self.th.slope_min_pp:
                    progress = True
                # conquest bookkeeping (#8): consecutive snapshots holding at mastered.
                if sr >= self.th.mastered_sr:
                    foc["consecutive_mastered"] = int(foc.get("consecutive_mastered", 0)) + 1
                else:
                    foc["consecutive_mastered"] = 0

            # --- (b) foundation progress: any link making a new best ---
            link_best = foc.setdefault("link_best", {})
            for link in foc.get("prereq_tree", []):
                lskill = link.get("skill")
                if not isinstance(lskill, str):
                    continue
                lsr = latest_profile.get(lskill.lower())
                if lsr is None:
                    continue
                lb = link_best.get(lskill.lower())
                if lb is None:
                    link_best[lskill.lower()] = float(lsr)  # first reading = baseline, not progress
                elif lsr >= lb + self.th.focus_improve_pp:
                    link_best[lskill.lower()] = float(lsr)
                    progress = True

            # --- (c) external chain-frontier advance (consumed once) ---
            if foc.pop("chain_frontier_advanced", False):
                progress = True

            foc["frozen_sessions"] = 0 if progress else int(foc.get("frozen_sessions", 0)) + 1
            foc["ladder_level"] = self._ladder_level(int(foc["frozen_sessions"]))

    def _verify_conquests(self, session_idx: int, latest_profile: dict[str, float]) -> None:
        """v6fix7 P1a (#8 fix): a wall is CONQUERED only after holding mastered_sr for
        conquest_consecutive consecutive snapshots. Only then does its verified_chains entry get
        status='verified', its skills enter the protected_set (rehearsal), and the focus retires
        gracefully. A one-session +delta record stays status='progress' and protects nothing —
        fix4 wrote make_iron_pickaxe into verified/protected at 44%, poisoning the tier4 base."""
        from auction.craftax_achievements import family_of

        kept = []
        for foc in self._nb.get("foci", []):
            skill = foc.get("skill")
            if not isinstance(skill, str) or int(foc.get("consecutive_mastered", 0)) < self.th.conquest_consecutive:
                kept.append(foc)
                continue
            sl = skill.lower()
            sr = latest_profile.get(sl)
            links = [
                l.get("skill") for l in foc.get("prereq_tree", []) if isinstance(l.get("skill"), str)
            ]
            category = "combat_milestone" if family_of(sl) == "COMBAT" else "enabler"
            self._upsert_experience(
                session_idx, sl, links, sr if sr is not None else float(self.th.mastered_sr),
                category, style_note=str(foc.get("style_note", "")), status="verified",
            )
            protect = {sl} | {l.lower() for l in links if l}
            self._nb["protected_set"] = sorted(set(self._nb.get("protected_set", [])) | protect)
            self._nb.setdefault("history", []).append(
                {"session": session_idx, "event": "focus_conquered", "focus": sl,
                 "sr": sr, "held_snapshots": int(foc.get("consecutive_mastered", 0))}
            )
            self.last_conquest = f"{sl} CONQUERED (SR {sr}%, held {foc.get('consecutive_mastered')} snapshots)"
        self._nb["foci"] = kept

    def _retire_stalled_foci(self, session_idx: int) -> None:
        """v6fix7 P1a L4: retirement fires only after ladder_l4 consecutive FROZEN sessions
        (whole-tree no-progress — see _update_focus_stall), never from the legacy ratchet.

        Retirement archives the failed tactic + an SR snapshot of the chain into the ``retired``
        registry, which _reconcile_foci uses for the cooldown / blacklist / "what's different this
        time" checks, and which render_for_prompt shows the modeler (history made READABLE — the
        fix4 bug was retire→same-session-reopen because the modeler never saw its own retirement).
        """
        kept = []
        for foc in self._nb.get("foci", []):
            frozen = int(foc.get("frozen_sessions", 0))
            if frozen >= self.th.ladder_l4:
                skill = str(foc.get("skill") or "")
                reg = self._nb.setdefault("retired", {}).setdefault(
                    skill, {"count": 0, "failed_notes": []}
                )
                reg["count"] = int(reg.get("count", 0)) + 1
                reg["last_session"] = session_idx
                reg["sr_at_retirement"] = foc.get("best_sr")
                reg["link_sr_at_retirement"] = dict(foc.get("link_best") or {})
                # P2: an ordered chain snapshot (prereq_tree order, shallow -> deep) so break-link
                # mining keeps tracking this wall while it rests, and any pre-retirement frontier
                # flag is stale by definition — the escape hatch must be earned AFTER this point.
                reg["links_at_retirement"] = [
                    str(l.get("skill")).lower()
                    for l in foc.get("prereq_tree", []) if isinstance(l.get("skill"), str)
                ]
                reg.pop("chain_frontier_advanced", None)
                note = str(foc.get("style_note", "")).strip()
                if note:
                    reg["failed_notes"] = ([*reg.get("failed_notes", []), note])[-3:]
                self._nb.setdefault("history", []).append(
                    {"session": session_idx, "event": "focus_retired_stalled",
                     "focus": foc.get("skill"), "frozen_sessions": frozen,
                     "best_sr": foc.get("best_sr")}
                )
            else:
                kept.append(foc)
        self._nb["foci"] = kept

    def _may_open_new_focus(self, latest_profile: dict[str, float]) -> bool:
        """§2.6 expand gate: may a NEW focus be opened this session?

        True iff (a) there is a free slot (fewer than ``max_focus`` active foci) AND (b) either there
        is NO active focus yet (the first wall is always allowed) OR ANY existing focus has reached
        ``focus_expand_sr`` — "one is going well, so there's slack to carry a new one" (user 2026-07-05).
        """
        foci = self._nb.get("foci", [])
        if len(foci) >= self.th.max_focus:
            return False
        if not foci:
            return True  # first wall
        return any(
            (latest_profile.get(f["skill"].lower()) or 0.0) >= self.th.focus_expand_sr
            for f in foci if isinstance(f.get("skill"), str)
        )

    def _merge_style_notes(self, proposed_foci: list[dict], session_idx: int | None = None) -> None:
        """§3.1: copy each proposed focus's fresh ``style_note`` onto the matching ACTIVE focus dict.

        Only non-empty notes overwrite (a session where the LLM says nothing new keeps the prior note,
        so know-how accumulates rather than being blanked). Matching is by skill name, so this naturally
        follows the same dedup-by-target rule as the experience log.

        v6fix7 P1c (AutoManual-lite lifecycle): each proposed focus also carries ``evidence_check``
        (supported / contradicted / no_evidence — the modeler's self-audit of its tactic against this
        session's REAL evidence). The notebook turns that into a code-tracked note status:
          - supported     -> note_status=active, note_last_supported_session=now;
          - contradicted  -> the note MUST be rewritten (materially different — same-tactic rephrase
                             rejected, like ladder L3); until rewritten it renders as CONTRADICTED;
          - no_evidence   -> ages toward STALE after note_stale_sessions unsupported sessions.
        """
        if not proposed_foci:
            proposed_foci = []
        by_skill = {
            str(pf.get("skill", "")).lower(): pf
            for pf in proposed_foci
            if isinstance(pf, dict) and pf.get("skill")
        }
        for foc in self._nb.get("foci", []):
            pf = by_skill.get(str(foc.get("skill", "")).lower())
            note = str(pf.get("style_note", "")).strip() if pf else ""
            evidence = str(pf.get("evidence_check", "")).strip().lower() if pf else ""
            old_note = str(foc.get("style_note", ""))

            # --- lifecycle bookkeeping (runs even when the note itself is empty) ---
            if session_idx is not None and pf is not None:
                if evidence == "supported":
                    foc["note_last_supported_session"] = session_idx
                    foc["note_status"] = "active"
                elif evidence == "contradicted":
                    foc["note_status"] = "contradicted"
                else:  # no_evidence / missing -> age toward stale
                    anchor = foc.get("note_last_supported_session")
                    if anchor is None:
                        anchor = foc.get("started_session") or session_idx
                    if old_note and (session_idx - int(anchor)) >= self.th.note_stale_sessions \
                            and foc.get("note_status") != "contradicted":
                        foc["note_status"] = "stale"

            if not note:
                continue
            # v6fix7 P1a L3 + P1c contradicted: when a REVISION is demanded (ladder >= 3, or the
            # modeler itself judged the tactic contradicted by evidence), a note materially the same
            # as the current one is rejected (kept old + history event) — "refine" is not enough.
            revision_required = int(foc.get("ladder_level", 0)) >= 3 or \
                foc.get("note_status") == "contradicted"
            if revision_required and self._notes_similar(note, old_note):
                self._nb.setdefault("history", []).append(
                    {"session": session_idx if session_idx is not None else self._nb.get("last_session"),
                     "event": "tactic_revision_rejected", "focus": foc.get("skill")}
                )
                continue
            if foc.get("note_status") == "contradicted" and note != old_note:
                # rewrite accepted: back to active (still unsupported until evidence says otherwise).
                foc["note_status"] = "active"
            foc["style_note"] = note

    def _record_experience(self, session_idx: int, latest_profile: dict[str, float]) -> None:
        """§2① + §2.5: incrementally record/UPDATE each active focus's success experience.

        For every active focus whose SR has risen by >= ``record_delta_pp`` versus the LAST time it was
        recorded (or first crossing above 0-ish), write/update (DEDUP BY TARGET) a verified_chains entry
        categorised combat_milestone vs enabler. Noise jitter below the delta records nothing -> no
        near-duplicate spam. The focus is NOT retired here (that is stall-driven, §2⑤); a focus keeps
        being attacked and its evidence keeps upgrading ("15% -> 40%, getting stronger").

        NOTE (ordering): this runs BEFORE new foci are opened in apply_llm_update, so a focus is first
        recorded the session AFTER it opens (once it has a real SR reading as an active focus) — this
        deliberately avoids recording a focus the same session it is proposed but before any training.
        """
        from auction.craftax_achievements import family_of

        for foc in self._nb.get("foci", []):
            skill = foc.get("skill")
            if not isinstance(skill, str):
                continue
            sr = latest_profile.get(skill.lower())
            if sr is None:
                continue
            last = foc.get("last_recorded_sr")
            if last is not None and sr < last + self.th.record_delta_pp:
                continue  # not enough new progress — skip (dedup/anti-noise)
            links = [
                link.get("skill")
                for link in foc.get("prereq_tree", [])
                if isinstance(link.get("skill"), str)
            ]
            category = "combat_milestone" if family_of(skill.lower()) == "COMBAT" else "enabler"
            # v6fix7 P1a (#8 fix): a +delta record is PROGRESS, not conquest. It no longer touches
            # the protected_set and its chain entry carries status='progress' — only
            # _verify_conquests (mastered_sr held for conquest_consecutive snapshots) upgrades an
            # entry to 'verified', protects it, and reports a conquest.
            self._upsert_experience(
                session_idx, skill.lower(), links, sr, category,
                style_note=str(foc.get("style_note", "")), status="progress",
            )
            foc["last_recorded_sr"] = sr

    def _upsert_experience(
        self, session_idx: int, target: str, links: list[str], sr: float, category: str,
        style_note: str = "", status: str = "progress",
    ) -> None:
        """Insert or, if ``target`` already recorded, UPDATE its verified_chains entry (dedup by target).

        ``style_note`` (§3.1) is the transferable attack know-how. On UPDATE it is overwritten only when
        the new note is non-empty, so a silent session keeps the prior know-how instead of erasing it.
        v6fix7 (#8): ``status`` distinguishes 'progress' (delta-recorded, NOT reusable as a tier4 base)
        from 'verified' (conquest gate passed). A 'verified' entry is never downgraded."""
        chains = self._nb.setdefault("verified_chains", [])
        evidence = f"held-out SR {sr}% (recorded at s{session_idx})"
        note = str(style_note or "").strip()
        for c in chains:
            if isinstance(c, dict) and str(c.get("target", "")).lower() == target:
                # update in place: keep first_recorded_sr, refresh the rest, merge links.
                c["last_recorded_sr"] = sr
                c["last_recorded_session"] = session_idx
                c["category"] = category
                merged_links = list(dict.fromkeys([*c.get("links", []), *links]))
                c["links"] = merged_links
                c["evidence"] = evidence
                if status == "verified" or c.get("status") == "verified":
                    c["status"] = "verified"  # never downgrade a verified conquest
                else:
                    c["status"] = "progress"
                if note:  # keep prior know-how if this session added none
                    c["style_note"] = note
                return
        chains.append({
            "target": target,
            "links": links,
            "category": category,
            "first_recorded_sr": sr,
            "last_recorded_sr": sr,
            "last_recorded_session": session_idx,
            "evidence": evidence,
            "style_note": note,
            "status": status,
        })

    # ---- the one public write path: apply an LLM-proposed update through B-layer rules ----------

    def apply_llm_update(
        self,
        session_idx: int,
        latest_profile: dict[str, float],
        proposed: dict | None,
        num_snapshots: int = 999,
    ) -> dict:
        """Fold one LLM proposed-update into the notebook, enforcing every B-layer rule, then persist.

        ``proposed`` (the A-layer output; may be None/empty if the LLM produced nothing usable) may
        contain EITHER the multi-focus form or the legacy single-focus form:
          - multi (§2.6):  "foci": [ {"skill", "prereq_tree":[{"skill","role"},...]}, ... ]
          - legacy:        "focus": <skill>, "prereq_tree": [ {"skill","role"}, ... ]
        The legacy form is normalised into a one-element ``foci`` list here, so old callers/tests keep
        working. The LLM supplies skill + role only; the CODE fills every mastery flag/state.

        B-layer enforcement order (§2.6 multi-focus):
          1. refresh existing link flags from live SR (LLM never sets mastery),
          2. §2① incrementally record each focus's success experience (dedup by target, categorised),
          3. per-focus stall bookkeeping, then retire any STALLED focus (never by SR threshold),
          4. reconcile the LLM's desired foci set: keep still-wanted existing foci; OPEN a new focus
             only if MATURE AND it passes §3.2 scope AND the §2.6 expand gate (free slot + one focus
             already going well) allows it,
          5. attach each accepted focus's prereq_tree, recomputing every link's mastery flag from SR.

        ``num_snapshots`` = how many held-out snapshots exist so far; used by the maturity gate.

        Returns the resulting notebook snapshot.
        """
        proposed = self._normalise_proposal(proposed)
        latest_profile = {k.lower(): v for k, v in (latest_profile or {}).items()}
        # reset per-call diagnostics (the caller reads these to log).
        self.last_focus_decision = None
        self.last_conquest = None

        # 1. existing links: flags come from data, not the model.
        self._refresh_link_flags(latest_profile)

        # 1b. §3.1 self-style: fold this session's fresh notes onto the EXISTING active foci first, so
        #     _record_experience carries the LATEST know-how into verified_chains (not last session's).
        #     A newly-opened focus is merged again in step 6 (it isn't active yet here).
        self._merge_style_notes(proposed.get("foci") or [], session_idx=session_idx)

        # 2. §2① incremental success-experience recording (replaces the old conquered-retire gate).
        self._record_experience(session_idx, latest_profile)

        # 3. per-focus stall (frozen counter + ladder level), conquest verification (#8), then retire
        #    exhausted foci (L4; frees slots). Conquest runs BEFORE retirement so a wall that holds
        #    mastered long enough exits as CONQUERED, never as stalled.
        self._update_focus_stall(latest_profile)
        self._verify_conquests(session_idx, latest_profile)
        self._retire_stalled_foci(session_idx)

        # 4. reconcile the desired foci set against the B-layer gates.
        mature = self._student_is_mature(latest_profile, num_snapshots)
        proposed_foci = proposed.get("foci") or []
        if not proposed_foci:
            self.last_focus_decision = "no_proposal"
        elif not mature:
            n_decent = sum(
                1 for v in latest_profile.values()
                if v is not None and v >= self.th.maturity_skill_sr
            )
            self.last_focus_decision = (
                f"immature (snapshots={num_snapshots}<{self.th.maturity_min_snapshots} or "
                f"decent_skills={n_decent}<{self.th.maturity_min_mastered}) -> NORMAL_EARLY, no siege"
            )
        else:
            self._reconcile_foci(session_idx, latest_profile, proposed_foci)

        # 5. attach each accepted focus's prereq_tree (code owns every mastery flag).
        self._attach_prereq_trees(latest_profile, proposed_foci)

        # 6. §3.1 self-style: fold the LLM's fresh attack know-how onto EVERY current focus (existing +
        #    just-opened). Done last so a newly-opened focus also captures its note this session; it is
        #    then carried into verified_chains by _record_experience NEXT session. LLM owns this text
        #    (A-layer); code only stores the latest non-empty note per target.
        self._merge_style_notes(proposed.get("foci") or [], session_idx=session_idx)

        self._nb["last_session"] = session_idx
        self._save()
        return self.snapshot()

    @staticmethod
    def _normalise_proposal(proposed: dict | None) -> dict:
        """Normalise a proposal into the multi-focus ``{"foci": [{skill, prereq_tree}, ...]}`` form.

        Accepts the legacy single-focus ``{"focus", "prereq_tree"}`` shape and wraps it into a
        one-element foci list, so old callers/tests need no change.
        """
        proposed = proposed or {}
        if proposed.get("foci"):
            return proposed
        focus = proposed.get("focus")
        if isinstance(focus, str) and focus.strip():
            return {"foci": [{
                "skill": focus,
                "prereq_tree": proposed.get("prereq_tree") or [],
                "style_note": proposed.get("style_note", ""),  # carry legacy top-level note through
            }]}
        return {"foci": []}

    def _reconcile_foci(
        self, session_idx: int, latest_profile: dict[str, float], proposed_foci: list[dict]
    ) -> None:
        """§2.6: keep still-wanted existing foci; open NEW foci through maturity/scope/expand/cap gates."""
        active = {f["skill"].lower() for f in self._nb.get("foci", []) if isinstance(f.get("skill"), str)}
        decisions: list[str] = []
        for pf in proposed_foci:
            skill = pf.get("skill")
            if not isinstance(skill, str) or not skill.strip():
                continue
            sl = skill.lower()
            if sl in active:
                decisions.append(f"kept({sl})")
                continue
            # a NEW focus: must be a real wall (scope) and there must be room + slack (expand gate).
            if not self._is_valid_focus(sl, latest_profile):
                decisions.append(
                    f"scope_rejected({sl}:SR{latest_profile.get(sl)}>=sat{self.th.saturated_sr})"
                )
                continue
            # v6fix7 P1a: a VERIFIED (conquered) wall still held at mastered SR may not reopen — it
            # is protected ground (rehearsal guards it). Siege it again only if it genuinely slipped.
            if self._is_conquered_and_held(sl, latest_profile):
                decisions.append(f"conquered_held({sl}: verified and SR still >= mastered)")
                continue
            # v6fix7 P1a: retirement aftermath gates — cooldown, blacklist, and "what's different".
            reg = (self._nb.get("retired") or {}).get(sl)
            if reg:
                last_ret = int(reg.get("last_session", -10**9))
                cooldown_left = self.th.cooldown_sessions - (session_idx - last_ret)
                if cooldown_left > 0:
                    decisions.append(f"cooldown_rejected({sl}: {cooldown_left} session(s) left)")
                    continue
                if int(reg.get("count", 0)) >= self.th.blacklist_retirements and not \
                        self._has_new_evidence(sl, reg, latest_profile):
                    decisions.append(
                        f"blacklisted({sl}: retired {reg.get('count')}x, no new chain evidence)"
                    )
                    continue
                new_note = str(pf.get("style_note", "")).strip()
                if not new_note or any(
                    self._notes_similar(new_note, old) for old in reg.get("failed_notes", [])
                ):
                    decisions.append(
                        f"reopen_needs_new_tactic({sl}: proposal's tactic is empty or repeats an "
                        "archived failed tactic — state what is DIFFERENT this time)"
                    )
                    self._nb.setdefault("history", []).append(
                        {"session": session_idx, "event": "reopen_rejected_same_tactic", "focus": sl}
                    )
                    continue
            if not self._may_open_new_focus(latest_profile):
                decisions.append(
                    f"expand_refused({sl}: {len(self._nb.get('foci', []))}/{self.th.max_focus} foci, "
                    f"none>={self.th.focus_expand_sr}%)"
                )
                continue
            self._nb.setdefault("foci", []).append(
                _empty_focus(sl, session_idx, latest_profile.get(sl))
            )
            active.add(sl)
            self._nb.setdefault("history", []).append(
                {"session": session_idx, "event": "focus_opened", "focus": sl}
            )
            decisions.append(f"opened({sl})")
        self.last_focus_decision = "; ".join(decisions) if decisions else "no_valid_focus"

    def _attach_prereq_trees(self, latest_profile: dict[str, float], proposed_foci: list[dict]) -> None:
        """For each ACTIVE focus that the proposal supplied a prereq_tree for, install it (code flags)."""
        trees = {}
        for pf in proposed_foci:
            skill = pf.get("skill")
            tree = pf.get("prereq_tree")
            if isinstance(skill, str) and isinstance(tree, list):
                trees[skill.lower()] = tree
        for foc in self._nb.get("foci", []):
            fskill = foc.get("skill")
            if not isinstance(fskill, str) or fskill.lower() not in trees:
                continue
            new_tree = []
            seen: set[str] = set()
            for item in trees[fskill.lower()]:
                if not isinstance(item, dict):
                    continue
                skill = item.get("skill")
                if not isinstance(skill, str) or not skill:
                    continue
                sl = skill.lower()
                if sl in seen or sl == fskill.lower():
                    continue  # dedupe; the focus itself is not a prereq link of itself
                seen.add(sl)
                sr = latest_profile.get(sl)
                new_tree.append({
                    "skill": sl,
                    "role": str(item.get("role", ""))[:120],
                    "state": mastery_from_sr(sr, self.th.mastered_sr, self.th.unmastered_sr),
                    "sr": sr,
                })
            if new_tree:
                foc["prereq_tree"] = new_tree

    # ---- rendering for the modeler prompt (the previous-notebook context handed back to the LLM) --

    def render_for_prompt(self) -> str:
        """Compact text of the current notebook to feed back into the modeler next session.

        This is the 'read the previous page of your journal' input for the A-layer LLM update.
        """
        nb = self._nb
        foci = nb.get("foci", [])
        if not foci and not nb.get("verified_chains"):
            return "(empty siege notebook — no active focus yet; pick the first wall this session.)"
        lines = []
        # §2.6: render every active focus (up to max_focus parallel sieges).
        if foci:
            lines.append(f"ACTIVE FOCI ({len(foci)}/{self.th.max_focus}) — the hard walls being attacked:")
            for foc in foci:
                lines.append(
                    f"  * {foc.get('skill')} "
                    f"(started s{foc.get('started_session')}, best SR {foc.get('best_sr')}%, "
                    f"frozen {foc.get('frozen_sessions', 0)} session(s) [whole-tree no-progress], "
                    f"form so far: {foc.get('last_siege_type') or 'unknown'})"
                )
                # v6fix7 P1a: the escalation ladder speaks — the modeler MUST see and obey its level.
                lvl = int(foc.get("ladder_level", 0))
                if lvl == 1:
                    lines.append(
                        "      ★LADDER L1: the whole attack tree has been frozen "
                        f"{foc.get('frozen_sessions')} sessions. Either switch the attack FORM "
                        "(DEPTH<->CONSOLIDATE) now, or DEFEND staying: give a concrete NEW reason and "
                        "plan in this focus's style_note (silently continuing is not allowed)."
                    )
                elif lvl == 2:
                    lines.append(
                        "      ★LADDER L2 (FORCED): still frozen after your defence. The attack FORM "
                        "is now switched by the system — recommend the OTHER form for this wall; the "
                        "proposer will be required to build it."
                    )
                elif lvl >= 3:
                    lines.append(
                        "      ★LADDER L3 (FORCED): form switch did not unfreeze it. You MUST write a "
                        "MATERIALLY DIFFERENT style_note (new tactic — a rephrase will be rejected) "
                        "stating what you abandon and what you try instead. At "
                        f"{self.th.ladder_l4} frozen sessions this focus retires with a cooldown."
                    )
                if foc.get("style_note"):
                    status = str(foc.get("note_status", "active"))
                    if status == "contradicted":
                        tag = " [★CONTRADICTED by evidence — you MUST rewrite this tactic (a rephrase is rejected)]"
                    elif status == "stale":
                        tag = (f" [STALE — unsupported for >= {self.th.note_stale_sessions} sessions: "
                               "re-derive it from mechanics + the latest data, do not keep refining it]")
                    else:
                        tag = ""
                    lines.append(f"      style-so-far{tag}: {foc['style_note']}")
                tree = foc.get("prereq_tree") or []
                if tree:
                    for link in tree:
                        lines.append(
                            f"      - {link.get('skill')}: {link.get('state')} "
                            f"(SR {link.get('sr')}%) role={link.get('role') or '-'}"
                        )
                else:
                    lines.append("      (prereq chain not yet backtracked — propose it this session)")
        else:
            lines.append("ACTIVE FOCI: (none — pick the next wall this session)")

        # §2.5: success experience split into two zones so the LLM sees the H1 milestones vs the
        # background enablers, instead of one flat undifferentiated list.
        chains = nb.get("verified_chains", [])
        milestones = [c for c in chains if c.get("category") == "combat_milestone"]
        enablers = [c for c in chains if c.get("category") != "combat_milestone"]
        if milestones:
            lines.append(
                f"COMBAT MILESTONE EXPERIENCE ({len(milestones)}) — YOUR self-style, the transferable "
                f"wins to build on. ★Only entries marked VERIFIED are conquered ground that may be "
                f"legally compressed as a base for deeper walls; PROGRESS entries are NOT yet won:"
            )
            for c in milestones[-6:]:
                tag = "VERIFIED" if c.get("status") == "verified" else "PROGRESS"
                lines.append(
                    f"    - [{tag}] {c.get('target')} via [{', '.join(c.get('links', []))}] "
                    f"(SR {c.get('last_recorded_sr')}%, s{c.get('last_recorded_session')})"
                )
                if c.get("style_note"):
                    lines.append(f"        style: {c['style_note']}")
        if enablers:
            lines.append(
                f"ENABLER GROUND ({len(enablers)}) — gear/prereqs experience (scaffolding, NOT "
                f"self-style). ★Only VERIFIED entries are held ground; PROGRESS is still contested:"
            )
            for c in enablers[-6:]:
                tag = "VERIFIED" if c.get("status") == "verified" else "PROGRESS"
                lines.append(
                    f"    - [{tag}] {c.get('target')} via [{', '.join(c.get('links', []))}] "
                    f"(SR {c.get('last_recorded_sr')}%, s{c.get('last_recorded_session')})"
                )
                if c.get("style_note"):
                    lines.append(f"        note: {c['style_note']}")
        if nb.get("protected_set"):
            lines.append(f"PROTECTED (rehearsal) skills: {', '.join(nb['protected_set'])}")
        # v6fix7 P1a: retirement history made READABLE — the fix4 bug was retire→same-session-reopen
        # because the modeler never saw its own retirements or why they failed.
        retired = nb.get("retired") or {}
        if retired:
            lines.append("RETIRED WALLS (failed sieges — do NOT reopen with the same tactic):")
            for skill, reg in list(retired.items())[-4:]:
                lines.append(
                    f"    - {skill}: retired {reg.get('count')}x, last s{reg.get('last_session')}, "
                    f"best SR reached {reg.get('sr_at_retirement')}%. Reopening requires the cooldown "
                    "to pass AND a tactic that is genuinely different from the failed ones below."
                )
                for note in (reg.get("failed_notes") or [])[-2:]:
                    lines.append(f"        failed tactic: {note}")
        return "\n".join(lines)
