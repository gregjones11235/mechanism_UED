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
FOCUS_MIN_STALL_SESSIONS = 10

# How much the focus's SR must rise between two recorded sessions to count as "improving" (resets the
# stall counter). Small, to tolerate eval noise but still register real progress.
FOCUS_IMPROVE_PP = 3.0

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
        "protected_set": [],         # list of skill names to rehearse (§3.6; populated on recording)
        "last_session": None,        # last session index this notebook was updated at
        "history": [],               # append-only log of focus changes, for offline inspection
    }


def _empty_focus(skill: str, session_idx: int, sr: float | None) -> dict:
    """A fresh focus element for ``skill`` newly opened at ``session_idx``."""
    return {
        "skill": skill,
        "started_session": session_idx,
        "best_sr": sr,
        "stall_sessions": 0,
        "last_recorded_sr": None,    # §2① no success-experience recorded for it yet
        "prereq_tree": [],           # backtracked chain, filled by the LLM proposal + code flags
        "style_note": "",            # §3.1 self-style: LLM free-text attack know-how for this wall
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

    def _update_focus_stall(self, latest_profile: dict[str, float]) -> None:
        """Per-focus (§2.6): update each active focus's best_sr / stall_sessions from its current SR."""
        for foc in self._nb.get("foci", []):
            skill = foc.get("skill")
            sr = latest_profile.get(skill.lower()) if isinstance(skill, str) else None
            if sr is None:
                # no reading this session — treat as a non-improving session (counts toward stall)
                foc["stall_sessions"] = int(foc.get("stall_sessions", 0)) + 1
                continue
            best = foc.get("best_sr")
            if best is None or sr >= best + self.th.focus_improve_pp:
                foc["best_sr"] = sr
                foc["stall_sessions"] = 0  # real improvement -> reset stall
            else:
                foc["stall_sessions"] = int(foc.get("stall_sessions", 0)) + 1

    def _retire_stalled_foci(self, session_idx: int) -> None:
        """§2.6 + §2⑤: a focus RETIRES only when it has stalled long enough (never by SR threshold).

        Stalled foci are dropped from ``foci`` (their success experience, if any, already lives in
        verified_chains via §2① incremental recording). This frees a slot for a new focus.
        """
        kept = []
        for foc in self._nb.get("foci", []):
            if int(foc.get("stall_sessions", 0)) >= self.th.focus_min_stall_sessions:
                self._nb.setdefault("history", []).append(
                    {"session": session_idx, "event": "focus_retired_stalled", "focus": foc.get("skill")}
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

    def _merge_style_notes(self, proposed_foci: list[dict]) -> None:
        """§3.1: copy each proposed focus's fresh ``style_note`` onto the matching ACTIVE focus dict.

        Only non-empty notes overwrite (a session where the LLM says nothing new keeps the prior note,
        so know-how accumulates rather than being blanked). Matching is by skill name, so this naturally
        follows the same dedup-by-target rule as the experience log."""
        if not proposed_foci:
            return
        by_skill = {
            str(pf.get("skill", "")).lower(): str(pf.get("style_note", "")).strip()
            for pf in proposed_foci
            if isinstance(pf, dict) and pf.get("skill")
        }
        for foc in self._nb.get("foci", []):
            note = by_skill.get(str(foc.get("skill", "")).lower())
            if note:
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
            self._upsert_experience(
                session_idx, skill.lower(), links, sr, category,
                style_note=str(foc.get("style_note", "")),
            )
            foc["last_recorded_sr"] = sr
            # protect the recorded target AND its consolidated links so rehearsal (§3.6) keeps them.
            protect = {skill.lower()} | {l.lower() for l in links if l}
            merged = set(self._nb.get("protected_set", [])) | protect
            self._nb["protected_set"] = sorted(merged)
            self.last_conquest = (
                f"{skill} (SR {sr}%, category={category}), chain=[{', '.join(links)}]"
            )

    def _upsert_experience(
        self, session_idx: int, target: str, links: list[str], sr: float, category: str,
        style_note: str = "",
    ) -> None:
        """Insert or, if ``target`` already recorded, UPDATE its verified_chains entry (dedup by target).

        ``style_note`` (§3.1) is the transferable attack know-how. On UPDATE it is overwritten only when
        the new note is non-empty, so a silent session keeps the prior know-how instead of erasing it."""
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
        self._merge_style_notes(proposed.get("foci") or [])

        # 2. §2① incremental success-experience recording (replaces the old conquered-retire gate).
        self._record_experience(session_idx, latest_profile)

        # 3. per-focus stall, then retire stalled foci (frees slots; §2⑤, never SR-threshold-driven).
        self._update_focus_stall(latest_profile)
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
        self._merge_style_notes(proposed.get("foci") or [])

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
                    f"stalled {foc.get('stall_sessions')} session(s))"
                )
                if foc.get("style_note"):
                    lines.append(f"      style-so-far: {foc['style_note']}")
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
                f"CONQUERED COMBAT MILESTONES ({len(milestones)}) — YOUR self-style, the transferable "
                f"wins to build on:"
            )
            for c in milestones[-6:]:
                lines.append(
                    f"    - {c.get('target')} via [{', '.join(c.get('links', []))}] "
                    f"(SR {c.get('last_recorded_sr')}%, s{c.get('last_recorded_session')})"
                )
                if c.get("style_note"):
                    lines.append(f"        style: {c['style_note']}")
        if enablers:
            lines.append(
                f"ENABLER GROUND HELD ({len(enablers)}) — gear/prereqs already in place (scaffolding, "
                f"NOT self-style):"
            )
            for c in enablers[-6:]:
                lines.append(
                    f"    - {c.get('target')} via [{', '.join(c.get('links', []))}] "
                    f"(SR {c.get('last_recorded_sr')}%, s{c.get('last_recorded_session')})"
                )
                if c.get("style_note"):
                    lines.append(f"        note: {c['style_note']}")
        if nb.get("protected_set"):
            lines.append(f"PROTECTED (rehearsal) skills: {', '.join(nb['protected_set'])}")
        return "\n".join(lines)
