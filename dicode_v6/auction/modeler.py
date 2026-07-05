"""Modeler — a GLM agent that models the student's CURRENT capability state and guides proposers.

v5-debate (2026-07-03, v5_design.md §3-4). The modeler runs ONCE per generation session, BEFORE the
proposers write levels. It is the piece that fixes the tier2->tier3 dependency-collapse root cause:
DiCode's Eq.6 frontier selection picks parents by their FROZEN A/B/C/D status, which goes stale and
over-estimates what the student currently knows. The modeler instead reads the student's *current*
held-out profile plus its time series (StudentProfileLog) and tells the proposers, per parent, which
of three level TYPES is most valuable right now:

  - DEPTH       : push a deeper transition (tier frontier forward)
  - BREADTH     : explore an untouched skill family (new achievements never attempted)
  - CONSOLIDATE : strengthen a learned-but-unreliable NON-siege skill toward mastery
                  (a middling SR that has stalled short of solid; forgotten skills are
                   rescued automatically by the system, not via CONSOLIDATE)

It must distinguish (prompt is designed for this, see MODELER_SYSTEM_PROMPT):
  NORMAL_EARLY (weak-but-rising, leave it)  vs  STALLED (genuinely can't learn, re-aim)
  vs  NOISY (single-reading dip, ignore)    vs  FORGETTING (was high, dropped -> system rehearsal).

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

    # ---- v6 SIEGE extension (v6_design.md §3.2/§3.5): read the persistent notebook, propose an
    #      update to it, and produce per-parent siege-aware guidance. Purely additive; the v5
    #      diagnose() path above is untouched (§3.7: upgrade the modeler, don't spawn a new agent). --

    def diagnose_siege(
        self,
        session_idx: int,
        parent_ids: list[str],
        parent_context: dict,
        notebook_text: str,
        combat_targets: list[str] | None = None,
        cooc_hint: str = "",
        behav_hint: str = "",
    ) -> dict:
        """Run the SIEGE modeler: v5 diagnosis PLUS a proposed siege-notebook update.

        The LLM is handed the previous notebook page (``notebook_text``) and this session's fresh SR
        evidence, and asked (like a stuck player reading their own notes) to: (1) do the normal
        student_states + guidance_per_parent, and (2) propose the siege ``foci`` (up to 3 parallel hard
        walls, §2.6) each with its backtracked ``prereq_tree``. It infers depth from SR + mechanics,
        NEVER a tier table (§3.3), and must pick genuine walls, not easy skills (§3.2 — also enforced
        in code by SiegeNotebook, this is the soft ask).

        Returns the v5 schema PLUS:
          "siege_update": {"foci": [{"skill", "prereq_tree":[{"skill","role"},...], "style_note"}, ...]}
        The CODE (SiegeNotebook.apply_llm_update) owns every mastery flag and every hard constraint;
        this is only the LLM's *proposal*.
        """
        evidence = self._build_state_evidence()
        system_prompt = MODELER_SIEGE_SYSTEM_PROMPT
        user_prompt = build_siege_modeler_user_prompt(
            session_idx=session_idx,
            evidence=evidence,
            parent_ids=parent_ids,
            parent_context=parent_context,
            notebook_text=notebook_text,
            combat_targets=combat_targets or [],
            cooc_hint=cooc_hint,
            behav_hint=behav_hint,
        )
        try:
            resp = self.llm.query(system_prompt, [user_prompt])
            content = (resp[0].get("content") or "") if resp else ""
        except Exception as e:  # noqa: BLE001 - modeler must never crash the session
            print(f"[modeler][siege] query failed ({type(e).__name__}: {e}); no siege update.")
            content = ""

        raw = _parse_modeler_json(content)
        out = self._validate(raw, parent_ids)
        out["siege_update"] = self._validate_siege(raw)
        return out

    @staticmethod
    def _validate_siege(raw: dict) -> dict:
        """Coerce the LLM's proposed siege update into the multi-focus {"foci": [...]} shape (§2.6).

        Accepts BOTH the multi-focus form (``"foci": [{"skill","prereq_tree":[...]}, ...]``) and the
        legacy single-focus form (``"focus" + "prereq_tree"``), always returning a ``foci`` list. This
        is only shape-validation — the SEMANTIC/hard constraints (scope, expand gate, per-focus stall,
        mastery flags) live in SiegeNotebook.apply_llm_update, the sole write path.
        """
        def _clean_tree(raw_tree) -> list:
            tree = []
            for item in raw_tree or []:
                if not isinstance(item, dict):
                    continue
                skill = item.get("skill")
                if not isinstance(skill, str) or not skill.strip():
                    continue
                tree.append({"skill": skill.lower(), "role": str(item.get("role", ""))[:120]})
            return tree

        raw_su = raw.get("siege_update") or {}
        if not isinstance(raw_su, dict):
            return {"foci": []}

        def _style(raw_focus) -> str:
            # §3.1 self-style note: the LLM's free-text attack insight for this target — HOW the wall is
            # being cracked / what's hard / what strategy works. This is the transferable "style"
            # H1 is about; the骨架 fields (skill/links/SR) are the ledger, this is the know-how.
            # No length cap (user 2026-07-05): the prompt asks for tight prose, so trust it not to bloat.
            return str(raw_focus.get("style_note", ""))

        foci = []
        seen: set[str] = set()
        # multi-focus form takes precedence when present.
        if isinstance(raw_su.get("foci"), list):
            for f in raw_su["foci"]:
                if not isinstance(f, dict):
                    continue
                skill = f.get("skill")
                if not isinstance(skill, str) or not skill.strip() or skill.lower() in seen:
                    continue
                seen.add(skill.lower())
                foci.append({
                    "skill": skill.lower(),
                    "prereq_tree": _clean_tree(f.get("prereq_tree")),
                    "style_note": _style(f),
                })
        # legacy single-focus form (still emitted by the current prompt).
        focus = raw_su.get("focus")
        if isinstance(focus, str) and focus.strip() and focus.lower() not in seen:
            foci.append({
                "skill": focus.lower(),
                "prereq_tree": _clean_tree(raw_su.get("prereq_tree")),
                "style_note": _style(raw_su),
            })
        return {"foci": foci}

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
        lines = []
        if g:
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
        # Surface the modeler's per-skill diagnosis for the skills a proposer should act on. STALLED
        # (genuinely stuck — needs a re-aimed rung) is a proposer job. FORGETTING is deliberately
        # EXCLUDED here: slipping skills are rescued automatically by the system's rehearsal, not by
        # asking the proposer to weave them back in. Skills that are simply fine
        # (MASTERED/RISING/NORMAL_EARLY/NOISY) are omitted to keep the prompt focused on what to act on.
        states = guidance.get("student_states") or {}
        notable = [
            f"{skill} ({label})"
            for skill, label in states.items()
            if label == "STALLED"
        ]
        if notable:
            lines.append(
                "Skills the modeler flags as genuinely stuck (STALLED) — good candidates to re-aim a "
                "rung toward: " + ", ".join(sorted(notable))
            )
        if not lines:
            return "No modeler guidance for this parent this session — use your own judgement."
        return "\n".join(lines)


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
  Label it so — the SYSTEM automatically rescues a forgetting skill by re-running old levels that
  teach it; you do NOT need to (and should not) turn this into a CONSOLIDATE recommendation.
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
- CONSOLIDATE: STRENGTHEN a learned-but-unreliable skill toward mastery — appropriate when a skill
  the student already performs, but only sometimes (a middling SR that has stalled short of solid),
  would de-risk further progress if made reliable. This is NOT for forgotten skills (the system
  rescues those automatically) and NOT for the hard wall(s) under active siege (those are DEPTH).

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
while its likely upstream is also at zero is an upstream problem (recommend BREADTH on the upstream
so the student can first learn it at all), NOT a reason to push DEPTH on the stuck skill. Your
unique value is (b) — the
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


# ==================================================================================================
# v6 SIEGE PROMPTS (v6_design.md §3.2/§3.3/§3.5). The siege modeler is the v5 modeler PLUS a
# persistent "siege journal" it grows across sessions — like a player's notes when stuck on a boss.
# ==================================================================================================

MODELER_SIEGE_SYSTEM_PROMPT = MODELER_SYSTEM_PROMPT.rstrip('"\n ') + """

==========================
SIEGE ROLE (ADDITIONAL — you also keep a persistent SIEGE JOURNAL)
==========================
On top of the per-parent diagnosis above, you maintain a PERSISTENT SIEGE JOURNAL across sessions,
exactly like a human player who is stuck on one hard boss keeps mental notes: which fight I am
currently attacking, which prerequisites of that fight I have already nailed vs still can't do. You
are handed the PREVIOUS page of this journal each session; you read it plus this session's fresh SR
evidence and PROPOSE how to update it. You do NOT rewrite it from scratch.

WHAT A "SIEGE FOCUS" IS (pick exactly ONE, or keep the current one):
- A focus is ONE genuinely-hard wall the student keeps failing — a skill whose SR is FLAT-NEAR-ZERO
  or stuck-low across sessions DESPITE the student having been past the early phase, i.e. a real
  bottleneck, not a skill that simply hasn't had its turn yet. This is the wall behind the
  student's oscillating overall score.

- ★CRITICAL — a LOW SR HAS TWO CAUSES; only ONE is a wall. Distinguish them exactly as you already
  distinguish STALLED from NORMAL_EARLY above, and apply that distinction to the FOCUS choice:
    (A) TRUE WALL (siege this): the student is clearly PAST the early ramp — MANY skills are already
        solid — and this particular skill's prerequisites are largely in place, YET it stays flat
        near zero across sessions. It is stuck at a real bottleneck. (This is the tier3-style wall
        behind the oscillating score.)
    (B) NORMAL_EARLY (NEVER siege this): the WHOLE student is still weak (few skills solid yet), and
        this low skill simply hasn't had its turn — an EARLY/BASIC skill reads flat-near-zero early
        for everyone. That is immaturity, not a wall. Leave it to the normal curriculum; be patient.
  Litmus test before you name any focus: "Is the student generally competent already, and is it the
  PREREQUISITES-are-there-but-still-can't skill that's stuck? Or is the student just early and weak
  across the board?" Only the former is a wall. If in doubt, or if the student looks early/weak
  overall, propose NO focus (leave "focus" empty) and let the baseline curriculum run.
- A focus should be a real wall, not an easy/already-mastered skill (a high-SR skill is a foundation
  the student can already re-supply, not a bottleneck).
- Judge which skill is a wall (and how deep it sits) from the SR shapes plus game mechanics — the same
  evidence-only rule as HARD CONSTRAINT 1 above (derive dependencies from evidence; do not recall a
  fixed difficulty tier), the way a player feels "I'm stuck HERE" without being told a rating.
- Prefer a COMBAT wall as the focus when one is available and stuck (winning a hard fight is where
  transferable dynamic skill — positioning, gear-up timing, escape — is forged). Gear/craft skills
  (diamond gear, torches) most often serve as ENABLER LINKS in a combat focus's chain — BUT a hard
  gear/craft skill can itself be a legitimate focus when mastering it unblocks a harder fight (e.g.
  focus diamond-gear so the student can then take on a fight that is impossible without it). Choose by
  what actually unblocks progress, not by a rule that gear can never be a focus.

FOCUS STABILITY (do not thrash):
- If the journal already has a focus and it is still stalled-but-not-hopeless, KEEP it. Switching the
  focus every session is the spread-thin failure that leaves every skill half-learned. Only switch
  when the current focus has clearly stalled for several sessions with no progress, or once it is
  conquered. (The system also enforces this in code — an early switch will simply be ignored.)

THE PREREQUISITE CHAIN (backtrack the focus):
- For the current focus, propose its PREREQUISITE CHAIN: the mix of reach/gear/survival skills the
  student must have to even attempt the wall. Infer each link from (a) game MECHANICS (what the fight
  physically requires — a floor reached, an item held, light in a dark floor) and (b) the student's
  own SR co-movement (which suspected prerequisite is actually near-zero right now and thus the live
  block). Never invent a link neither the mechanics nor the data support.
- For each link give a short ``role`` note (e.g. "reach floor2", "gear: diamond sword", "survive
  dark: torches"). Do NOT label a link mastered/unmastered yourself — the system computes that from
  SR. Just list the links and their role.
- A link the student has NOT yet mastered must stay a trained sub-goal (it will NOT be compressed
  away). A link the student HAS mastered can be safely assumed. You only propose the chain; the
  system decides, from live SR, which links are still unmastered.

==========================
SIEGE OUTPUT (ADD this block to your JSON — keep student_states + guidance_per_parent too)
==========================
Add one more top-level key to the SAME JSON object. You may attack UP TO THREE hard walls in parallel
(cross-achievement assist: e.g. upgrading gear helps later fights). A NEW wall may be opened only once
one of your current walls is already going well (the code enforces this — one focus must have reached
a solid SR before a slot frees for another). Emit a "foci" LIST:
  "siege_update": {
    "foci": [
      { "skill": "<a hard-wall skill you are attacking (a real stuck skill, NEVER an easy/mastered one)>",
        "prereq_tree": [ { "skill": "<a prerequisite skill>", "role": "<short role note>" }, ... ],
        "style_note": "<the transferable attack know-how for THIS wall — see below>" },
      ...  // up to 3 walls; keep the ones from your journal you still want, add at most a new one
    ]
  }
Keep the walls from the journal you still want (list them again); drop one by omitting it. To attack
just one wall, emit a single-element "foci" list. (A legacy single "focus"+"prereq_tree" is still
accepted and treated as a one-wall list.) Output STRICT JSON only.

THE STYLE NOTE (§3.1 — this is the transferable SELF-STYLE, the whole point of the siege):
- ``style_note`` is your running, cross-session know-how for cracking THIS wall: what actually makes it
  hard for the student right now, which prerequisite is the live bottleneck, and the concrete tactic
  that is (or should be) working — positioning, gear-up timing, pulling one enemy at a time, escape/
  kite, dark-floor lighting, resource routing. It is the "HOW I beat it", not the "what it is".
- GROUND IT IN REAL BEHAVIOUR when you can: if the user message includes a REAL-SUCCESS BEHAVIOUR block
  (the action mix / pacing of the episodes where the student ACTUALLY won this or a related wall), the
  style_note must reflect what the student really did — e.g. "wins are ~84 steps, heavy PLACE_STONE +
  low combat -> a mining/craft route, not a fight" — NOT a tactic you merely imagine. When a target has
  no behaviour data yet (solved too rarely), fall back to mechanics-based know-how and say so tersely.
- This is the ONE field that carries style forward. The skill/links/SR fields are just a ledger; the
  style_note is the reusable strategy a later, similar wall inherits. If you leave it blank, that
  hard-won know-how is lost — so always write it for an active focus.
- WRITE IT TIGHT: terse, dense, every word earns its place — but drop NO key point. Prefer 1-3 compact
  clauses of concrete tactics over prose. Update it as understanding grows (the journal shows your
  previous note as "style-so-far"; refine it, don't restate it). No fluff, no game-lore, no restating
  the skill name.
"""


def build_siege_modeler_user_prompt(
    session_idx: int,
    evidence: dict,
    parent_ids: list[str],
    parent_context: dict,
    notebook_text: str,
    combat_targets: list[str],
    cooc_hint: str = "",
    behav_hint: str = "",
) -> str:
    """Siege user prompt = the v5 user prompt PLUS the previous journal page + the combat-target list
    + (optionally) the (c) real-trajectory co-occurrence evidence + (problem-2) the winning-episode
    behaviour fingerprint.

    ``combat_targets`` is the set of COMBAT-family achievement names, given so the modeler can prefer a
    combat wall as the focus. (This is a category label, not a course-chain prior — see §3.2.)
    ``cooc_hint`` is the (c) co-occurrence text (v6_design.md §3.8): which skills the student actually
    co-reaches when it succeeds at a deep skill — empty when support is too sparse (phased fallback to
    (b) mechanics only).
    ``behav_hint`` (problem-2) is the behaviour fingerprint: HOW the student acted (action mix / pacing)
    in the episodes where it won a deep skill — evidence for grounding the style_note in real actions
    instead of an imagined tactic. Empty when solved too rarely (same phased fallback).
    """
    base = build_modeler_user_prompt(session_idx, evidence, parent_ids, parent_context)
    # Drop the base prompt's final "output JSON now" line — we re-issue a siege-aware instruction.
    base_body = base.rsplit("Now output the STRICT JSON", 1)[0].rstrip()

    combat_line = ", ".join(sorted(combat_targets)) if combat_targets else "(none provided)"
    nb = notebook_text.strip() or "(empty siege notebook — no active focus yet)"
    cooc_block = (
        f"{cooc_hint.strip()}\n\n" if cooc_hint and cooc_hint.strip() else ""
    )
    behav_block = (
        f"{behav_hint.strip()}\n\n" if behav_hint and behav_hint.strip() else ""
    )

    return (
        f"{base_body}\n\n"
        "YOUR SIEGE JOURNAL — the previous page (read it, then propose an update; do NOT rewrite from "
        "scratch):\n"
        f"{nb}\n\n"
        "COMBAT achievement names (fights), listed so you can prefer a stuck fight as the focus:\n"
        f"  {combat_line}\n\n"
        f"{cooc_block}"
        f"{behav_block}"
        "Now output the STRICT JSON object with THREE top-level keys: student_states, "
        "guidance_per_parent, AND siege_update (a \"foci\" list of up to 3 hard walls, each with its "
        "prereq_tree). Pick or keep real hard walls (never easy/mastered skills), and backtrack each "
        "wall's prerequisite chain — when the REAL-TRAJECTORY CO-OCCURRENCE above is present, prefer "
        "the links the student actually co-reaches on its wins over links you merely imagine. JSON "
        "only, no other text."
    )
