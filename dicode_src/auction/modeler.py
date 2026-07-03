"""Modeler — a GLM agent that models the student's CURRENT capability state and guides proposers.

v5-debate (2026-07-03, v5_design.md §3-4). The modeler runs ONCE per generation session, BEFORE the
proposers write levels. It is the piece that fixes the tier2->tier3 dependency-collapse root cause:
DiCode's Eq.6 frontier selection picks parents by their FROZEN A/B/C/D status, which goes stale and
over-estimates what the student currently knows. The modeler instead reads the student's *current*
held-out profile plus its time series (StudentProfileLog) and tells the proposers, per parent, which
of three level TYPES is most valuable right now:

  - DEPTH       : push a deeper transition (tier frontier forward)
  - BREADTH     : explore an untouched skill family (new achievements never attempted)
  - CONSOLIDATE : repair a capability the student ONCE had and is now FORGETTING
                  (only the modeler can see this — it needs the profile time series)

It must distinguish (prompt is designed for this, see MODELER_SYSTEM_PROMPT):
  NORMAL_EARLY (weak-but-rising, leave it)  vs  STALLED (genuinely can't learn, re-aim)
  vs  NOISY (single-reading dip, ignore)    vs  FORGETTING (was high, dropped -> CONSOLIDATE).

Model: GLM-5.2 (native structured output + calibration — best fit for a diagnostic role).

Output is a STRUCTURED JSON object (parsed defensively, same as cross-rating). The modeler MAY request
details of any historical archive level (code/description) to recommend a reusable/tweakable reference
level for a proposer — exposed via ``ModelerArchiveView``.

The two prompt strings (MODELER_SYSTEM_PROMPT / build_modeler_user_prompt) are the heart of the
method (v5_design.md §7); the surrounding data plumbing here is complete and testable independently.
"""

from __future__ import annotations

import json
import re

# Level types the modeler can recommend and proposers tag their output with.
LEVEL_TYPES = ("DEPTH", "BREADTH", "CONSOLIDATE")

# Capability-state labels the modeler assigns per skill domain.
STATE_LABELS = ("NORMAL_EARLY", "RISING", "STALLED", "NOISY", "FORGETTING", "MASTERED")


class ModelerArchiveView:
    """Read-only accessor letting the modeler pull details of any historical level.

    Thin wrapper over the TaskArchive so the modeler can recommend a concrete reusable/tweakable
    reference level (by id) to a proposer. Kept separate so the modeler never mutates the archive.
    """

    def __init__(self, archive):
        self._archive = archive

    def level_detail(self, task_id: str) -> dict:
        """Return {id, description, code, status, performance_history} for one archived level."""
        g = getattr(self._archive, "graph", None)
        if g is None or not g.has_node(task_id):
            return {}
        data = g.nodes[task_id]
        return {
            "id": task_id,
            "description": data.get("description", "") or "",
            "code": data.get("code", "") or "",
            "status": data.get("status", "unknown"),
            "performance_history": data.get("performance_history", ""),
        }

    def all_level_ids(self) -> list[str]:
        g = getattr(self._archive, "graph", None)
        return list(g.nodes()) if g is not None else []


def _parse_modeler_json(content: str) -> dict:
    """Extract the first JSON object from the modeler's response, defensively.

    Mirrors the tolerant parsing used for cross-ratings (gen_manager._run_cross_rating): the model is
    asked for JSON only, but we still guard against leading/trailing prose or code fences.
    """
    if not content:
        return {}
    # Strip common ```json fences.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        candidate = match.group(0) if match else None
    if candidate is None:
        return {}
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


class Modeler:
    """Runs the GLM modeler once per session and returns structured guidance for the proposers."""

    def __init__(self, llm, archive, profile_log, *, recent_k: int = 6):
        """
        Args:
            llm: the GLM LLM instance (built from config.modeler, like proposer_llms).
            archive: TaskArchive — for ModelerArchiveView (historical level lookup).
            profile_log: StudentProfileLog — the held-out profile time series.
            recent_k: how many recent session snapshots to feed the modeler for trend judgement.
        """
        self.llm = llm
        self.view = ModelerArchiveView(archive)
        self.profile_log = profile_log
        self.recent_k = recent_k

    # ---- data assembly (complete; prompt text filled last) -------------------------------------

    def _build_state_evidence(self) -> dict:
        """Assemble the raw evidence the modeler reasons over, WITHOUT pre-judging it.

        We hand the modeler facts, not verdicts (so IT distinguishes normal-early vs stalled vs noisy):
          - latest full profile
          - per-skill recent SR series (so it can read rising vs flat vs fluctuating)
          - a cheap FORGETTING prefilter (peaked-then-dropped) it can confirm or reject
        """
        latest = self.profile_log.latest()
        recent_snaps = self.profile_log.recent(self.recent_k)
        # Per-skill recent series for every skill seen recently (trend evidence).
        skills_seen: set[str] = set()
        for snap in recent_snaps:
            skills_seen.update(snap.get("profile", {}).keys())
        series = {
            name: self.profile_log.series_for(name, last_k=self.recent_k)
            for name in sorted(skills_seen)
        }
        forgetting_prefilter = self.profile_log.forgetting_candidates()
        return {
            "latest_profile": latest,
            "recent_series": series,
            "forgetting_prefilter": forgetting_prefilter,
            "num_snapshots": len(recent_snaps),
        }

    def diagnose(self, session_idx: int, parent_ids: list[str], parent_context: dict) -> dict:
        """Run the modeler and return validated structured guidance.

        Returns a dict:
          {
            "student_states": {skill: STATE_LABEL, ...},          # per-domain diagnosis
            "guidance_per_parent": {parent_id: {                  # what to do at each frontier parent
                 "recommended_type": DEPTH|BREADTH|CONSOLIDATE,
                 "note": "<behaviour-language guidance, NO craftax tier/knowledge leak>",
                 "reference_level_id": "<archived level id to reuse/tweak, or ''>",
            }, ...},
          }
        Falls back to an empty-but-valid structure if the LLM output can't be parsed, so callers can
        always proceed (proposers then behave as plain ambitious with no modeler steer).
        """
        evidence = self._build_state_evidence()
        system_prompt = MODELER_SYSTEM_PROMPT
        user_prompt = build_modeler_user_prompt(
            session_idx=session_idx,
            evidence=evidence,
            parent_ids=parent_ids,
            parent_context=parent_context,
        )
        try:
            resp = self.llm.query(system_prompt, [user_prompt])
            content = (resp[0].get("content") or "") if resp else ""
        except Exception as e:  # noqa: BLE001 - modeler must never crash the session
            print(f"[modeler] query failed ({type(e).__name__}: {e}); proceeding with no guidance.")
            content = ""

        raw = _parse_modeler_json(content)
        return self._validate(raw, parent_ids)

    def _validate(self, raw: dict, parent_ids: list[str]) -> dict:
        """Coerce the parsed output into the guaranteed schema; drop anything malformed."""
        states = {}
        for skill, label in (raw.get("student_states") or {}).items():
            if isinstance(skill, str) and label in STATE_LABELS:
                states[skill.lower()] = label

        guidance = {}
        raw_guid = raw.get("guidance_per_parent") or {}
        valid_ids = set(parent_ids)
        for pid, g in raw_guid.items():
            if pid not in valid_ids or not isinstance(g, dict):
                continue
            rtype = g.get("recommended_type")
            if rtype not in LEVEL_TYPES:
                rtype = None
            ref = g.get("reference_level_id") or ""
            if ref and not self.view.level_detail(ref):
                ref = ""  # drop a hallucinated reference id
            guidance[pid] = {
                "recommended_type": rtype,
                "note": str(g.get("note", ""))[:800],
                "reference_level_id": ref,
            }
        return {"student_states": states, "guidance_per_parent": guidance}

    def render_guidance_for_parent(self, guidance: dict, parent_id: str) -> str:
        """Turn one parent's structured guidance into the text injected into a proposer prompt.

        Includes the reference level's description body when a reusable reference was recommended, so
        the proposer can actually reuse/tweak it (v5_design.md §3: modeler can push a historical level).
        """
        g = (guidance.get("guidance_per_parent") or {}).get(parent_id)
        if not g:
            return "No modeler guidance for this parent this session — use your own judgement."
        lines = []
        if g.get("recommended_type"):
            lines.append(f"Modeler-recommended level TYPE: {g['recommended_type']}")
        if g.get("note"):
            lines.append(f"Modeler note: {g['note']}")
        ref = g.get("reference_level_id")
        if ref:
            detail = self.view.level_detail(ref)
            if detail:
                lines.append(
                    "Modeler suggests reusing/adapting this prior level as a reference:\n"
                    f"[reference level {ref}]\n{detail.get('description', '')}"
                )
        return "\n".join(lines) if lines else "No specific modeler guidance for this parent."


# ==================================================================================================
# PROMPTS (v5_design.md §7). The data plumbing above is complete and unit-testable independently.
# ==================================================================================================

MODELER_SYSTEM_PROMPT = """You are the STUDENT MODELER in a curriculum-design team training a reinforcement-learning agent on a
complex open-ended game. You do NOT design levels. Your one job is to read the agent's ("the
student's") performance history and produce a clear, calibrated picture of its CURRENT capability
state, so the level designers ("proposers") know what to work on next.

You are given, per skill, the student's success-rate (SR, 0-100%) as a TIME SERIES over recent
training sessions — not a single number. Your value comes entirely from reading these series
correctly. The most common and costly mistake is to over-react to one low reading. Guard against it.

==========================
HOW TO CLASSIFY EACH SKILL (assign exactly one label)
==========================
Judge the SHAPE of each skill's SR series, together with how early the student is overall (few
snapshots = early training):

- RISING: the series is climbing over recent sessions (e.g. 0 -> 5 -> 20 -> 40), even if still low.
  The student IS learning this. Do NOT treat it as a problem. Leave it alone.
- NORMAL_EARLY: SR is low AND the student is early overall (few snapshots) AND this skill is deep /
  hasn't had a fair chance yet. Weak because untrained, not because it can't be learned. Be patient.
- STALLED: the series has stayed FLAT and LOW across MULTIPLE sessions despite the student being past
  the early phase (e.g. 3 -> 2 -> 4 -> 3 over many sessions). This is a genuine dead end from here —
  the current approach isn't working; the designers should re-aim, not re-issue the same reach.
- NOISY: the series JUMPS around with no trend (e.g. 3 -> 71 -> 5 -> 60 -> 2). A single low (or high)
  reading here is measurement noise, NOT a real drop. Do not raise an alarm on noise. Label it NOISY
  and, if its typical level is decent, treat the skill as roughly held.
- FORGETTING: the series was clearly HIGH earlier and has since DROPPED and stayed lower (e.g.
  90 -> 88 -> 45 -> 40), beyond what noise explains. The student is losing a capability it once had.
  This is the signal to CONSOLIDATE (re-train that capability) before building further on top of it.
- MASTERED: consistently high (e.g. >= 75%) across recent sessions. Solid; build on it freely.

A cheap "forgetting prefilter" (skills that peaked then dropped) is provided as a HINT. Confirm or
REJECT each hinted skill yourself: a peaked-then-dropped skill whose series is actually NOISY is not
truly forgetting. You have the final say.

==========================
WHAT TO RECOMMEND PER PARENT LEVEL
==========================
For each frontier parent level the designers may evolve, recommend ONE level TYPE that is most
valuable RIGHT NOW given the student's state:
- DEPTH: push a deeper transition forward — appropriate when the prerequisites for that direction are
  MASTERED or RISING (the student can actually reach the new situation).
- BREADTH: bring an entirely UNATTEMPTED skill family into play — appropriate when whole capability
  areas sit untouched while others are already solid.
- CONSOLIDATE: repair a FORGETTING capability before it undermines everything built on it —
  appropriate when a once-held prerequisite has dropped. THIS IS YOURS TO CATCH: only you see the
  time series, so only you can flag a forgotten prerequisite the designers would otherwise assume.

==========================
INFERRING WHICH SKILL DEPENDS ON WHICH (how to know a "prerequisite")
==========================
To recommend DEPTH/CONSOLIDATE you must judge whether one skill is a prerequisite for another. Infer
this from TWO kinds of evidence — never from a tech-tree you imagine:

(a) MECHANICAL evidence — dependencies the game rules force (from the level descriptions you are
    given / your general grasp of the game): doing X requires first having unlocked Y or holding an
    item. This tells you a dependency is POSSIBLE in principle.
(b) STUDENT-STATE evidence — co-movement in the SR TIME SERIES, which ONLY YOU can see: skills that
    rise together or fall together; one skill sitting at a hard ceiling while a suspected upstream
    skill is still near zero (the ceiling is an upstream lock, not "this skill is hard"); a
    capability dropping and its dependents dropping a session later. This tells you WHICH dependency
    is an ACTIVE bottleneck for THIS student RIGHT NOW.

When the two disagree, TRUST STUDENT-STATE evidence: a dependency that exists in principle but that
the student has clearly already moved past is not a current bottleneck, and a skill stuck at zero
while its likely upstream is also at zero is an upstream problem (recommend BREADTH/CONSOLIDATE on
the upstream), NOT a reason to push DEPTH on the stuck skill. Your unique value is (b) — the
designers already know the mechanics; only you see the trajectory.

==========================
HARD CONSTRAINTS
==========================
1. Speak in BEHAVIOUR, not game lore. Refer to skills by their given names and to what the student
   can/can't currently do. You MAY infer skill dependencies from the two kinds of evidence above
   (mechanical + student-state co-movement), but do NOT invent depth tiers, numeric "tier" indices,
   or a fixed tech-tree that neither the game rules nor the data support — your job is to diagnose
   the student from evidence, not to teach the game from imagination.
2. Calibrate. Prefer "held / rising / patient" over crying wolf. Only flag STALLED or FORGETTING when
   the SERIES (not one reading) supports it.
3. Output STRICT JSON only, matching the schema below. No prose outside the JSON.

==========================
OUTPUT SCHEMA (JSON only)
==========================
{
  "student_states": { "<skill_name>": "RISING|NORMAL_EARLY|STALLED|NOISY|FORGETTING|MASTERED", ... },
  "guidance_per_parent": {
    "<parent_id>": {
      "recommended_type": "DEPTH|BREADTH|CONSOLIDATE",
      "note": "<one or two sentences, behaviour language, why this type for this parent>",
      "reference_level_id": "<an archived level id worth reusing/adapting, or \\"\\">"
    }, ...
  }
}
Only include skills you can actually judge, and only the parent ids you were given.
"""


def _fmt_series(series: list) -> str:
    """Compact 'sN:XX%' rendering of a [(session, sr), ...] series, oldest first."""
    return ", ".join(f"s{sess}:{sr:.0f}%" for sess, sr in series) if series else "(no readings)"


def build_modeler_user_prompt(
    session_idx: int, evidence: dict, parent_ids: list[str], parent_context: dict
) -> str:
    """Assemble the modeler's per-session user prompt.

    Formats the assembled evidence (latest profile, per-skill recent SR series, forgetting prefilter)
    plus the frontier parents into the modeler's task. Reference-able archive level ids are listed so
    the modeler can recommend one to reuse.
    """
    latest = evidence.get("latest_profile", {})
    series = evidence.get("recent_series", {})
    forget = evidence.get("forgetting_prefilter", [])
    n_snap = evidence.get("num_snapshots", 0)

    # Per-skill recent SR series — the core evidence for RISING/STALLED/NOISY/FORGETTING.
    series_lines = "\n".join(
        f"  - {name}: {_fmt_series(series[name])}" for name in sorted(series)
    ) or "  (no per-skill history yet)"

    forget_lines = (
        "\n".join(
            f"  - {f['achievement']}: peaked {f['peak']:.0f}% (s{f['peak_session']}) -> now "
            f"{f['latest']:.0f}%  (drop {f['drop']:.0f}pp)"
            for f in forget
        )
        or "  (none flagged)"
    )

    # Frontier parents + any context the caller passed (e.g. the parent's own description snippet).
    parent_lines = "\n".join(
        f"  - {pid}: {str(parent_context.get(pid, ''))}" for pid in parent_ids
    ) or "  (no frontier parents this session)"

    return (
        f"Training session index: {session_idx} "
        f"({n_snap} recent held-out snapshots available — few snapshots means EARLY training).\n\n"
        "STUDENT SKILL SR TIME SERIES (recent sessions, oldest first) — read the SHAPE of each:\n"
        f"{series_lines}\n\n"
        "FORGETTING PREFILTER (skills that peaked then dropped — CONFIRM or REJECT each; a noisy one "
        "is not truly forgetting):\n"
        f"{forget_lines}\n\n"
        "FRONTIER PARENT LEVELS to advise on (recommend ONE type per parent):\n"
        f"{parent_lines}\n\n"
        "Now output the STRICT JSON object (student_states + guidance_per_parent) per your schema. "
        "JSON only, no other text."
    )
