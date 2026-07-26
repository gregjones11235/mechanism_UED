"""Modeler — a GLM agent that models the student's CURRENT capability state and guides proposers.

v5-debate (2026-07-03, v5_design.md §3-4). The modeler runs ONCE per generation session, BEFORE the
proposers write levels. It is the piece that fixes the tier2->tier3 dependency-collapse root cause:
DiCode's Eq.6 frontier selection picks parents by their FROZEN A/B/C/D status, which goes stale and
over-estimates what the student currently knows. The modeler instead reads the student's *current*
held-out profile plus its time series (StudentProfileLog) and tells the proposers, per parent, which
of three level TYPES is most valuable right now:

  - DEPTH       : push a deeper transition (tier frontier forward)
  - BREADTH     : explore an untouched skill family (new achievements never attempted)
  - CONSOLIDATE : an ISOLATION DRILL — a stripped-down level that strips the unrelated
                  combat/survival distraction so the student repeats ONE target skill cleanly
                  (real execution chain kept). For a middling SR that has stalled short of
                  solid; MAY target the siege wall itself when it stalls because its craft
                  signal is drowned by fighting. (Forgotten skills are rescued automatically,
                   not via CONSOLIDATE.)

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
import os
import re

# Level types the modeler can recommend and proposers tag their output with.
LEVEL_TYPES = ("DEPTH", "BREADTH", "CONSOLIDATE")

# Capability-state labels the modeler assigns per skill domain.
STATE_LABELS = ("NORMAL_EARLY", "RISING", "STALLED", "NOISY", "FORGETTING", "MASTERED")

# v6fix9 P2 — the attribution gate. The LLM must commit to a CAUSE CLASS for each focus's failures,
# and the class is cross-checked against ChainOrderLog.forensics() (missing histogram / break shape
# / death timing / inventory gaps). This is what P1c's evidence_check could never be: a self-audit
# with data teeth — the armour case ("cannot craft under pressure" vs zero combat signal in 22
# sessions of fail_hist) is exactly the claim shape these checks reject.
ATTRIB_CLASSES = (
    "resource_shortfall", "interrupted_by_combat", "chain_unreached",
    "execution_failure", "unknown",
)
# An interruption claim is judged on DEATH TIMING alone: failures must die close to their chain
# frontier — long post-frontier survival refutes interruption outright. A separate died_frac floor
# was calibrated AWAY on job 3691755 data (2026-07-08): the eval horizon equals the env step cap
# (8192) while the mean episode is ~1234, so >=85% of finished episodes end by death for EVERY wall
# — an ambient-death floor is dead code that can only misfire; died_frac stays as rendered evidence.
# 50 steps ~ one combat encounter, vs the observed hundreds-to-thousands of post-frontier survival
# steps in resource-shaped failures — the threshold sits in a wide empty band, like _UNIVERSAL_SR.
_ATTRIB_INTERRUPT_MAX_STEPS = 50
# v6fix10 ⑥ (fix9 first-run calibration): the absolute 50-step bound can NEVER fire — observed
# ambient after-deepest survival is 134-468 across every wall (kobold/skeleton/pickaxe), so an
# interruption pattern is "well below this wall's OWN ambient", not "below a fixed count". The
# check becomes relative — max(floor, REL x the wall's historical ambient median) — with the old
# absolute constant as the fallback when no ambient history exists yet.
_ATTRIB_INTERRUPT_REL = 0.3
_ATTRIB_INTERRUPT_FLOOR = 30
# Reroll budget when claims contradict the data (same cap as the fix7 level validator): each retry
# is one LLM call; past that the claim is coerced to "unknown" + WARN — training never blocks.
_ATTRIB_REROLL_MAX = 2


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

    def __init__(self, llm, archive, profile_log, *, recent_k: int = 6, scientist_llm=None):
        """
        Args:
            llm: the GLM LLM instance (built from config.modeler, like proposer_llms).
            archive: TaskArchive — for ModelerArchiveView (historical level lookup).
            profile_log: StudentProfileLog — the held-out profile time series.
            recent_k: how many recent session snapshots to feed the modeler for trend judgement.
            scientist_llm: v7fix5.5 P2 — the SECOND, statically-configured LLM instance for the
                scientist pass (same model, think=True; the 2026-07-17 A/B pinned think per call
                type: the big bookkeeping call keeps think off, the small hypothesis call keeps
                it on — no runtime switching). None = the hypothesis loop stays dormant.
        """
        self.llm = llm
        self.scientist_llm = scientist_llm
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

    _LLM_DUMP_N = 0

    def _dump_llm_call(self, tag: str, content: str) -> None:
        """v7fix5.6: persist every raw LLM response to runtime_analysis/llm_calls/ (the
        2026-07-18 forensics gap — the modeler's behaviour at the s161-167 stall could only
        be reconstructed from side effects). Append-only, never read back, never fatal."""
        try:
            base = "."
            cfgs = getattr(self, "config", None)
            for key in ("output_dir", "run_dir", "save_dir", "log_dir"):
                v = getattr(cfgs, key, None) if cfgs is not None else None
                if v:
                    base = str(v)
                    break
            d = os.path.join(base, "runtime_analysis", "llm_calls")
            os.makedirs(d, exist_ok=True)
            Modeler._LLM_DUMP_N += 1
            p = os.path.join(d, f"{Modeler._LLM_DUMP_N:05d}_{tag}.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(content or "")
        except Exception:  # noqa: BLE001 - response logging must never break the session
            pass

    def diagnose_siege(
        self,
        session_idx: int,
        parent_ids: list[str],
        parent_context: dict,
        notebook_text: str,
        combat_targets: list[str] | None = None,
        cooc_hint: str = "",
        behav_hint: str = "",
        forensics: dict | None = None,
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
        # v6fix9 P2: query + validate, re-prompting (up to _ATTRIB_REROLL_MAX) when a
        # failure_attribution claim contradicts the measured forensics — the rejection feedback
        # quotes the exact violations so the LLM can fix them (or honestly say "unknown"). Both the
        # query and the parse stay inside guards: the modeler must never crash the session; on any
        # failure we degrade to empty guidance + empty siege_update exactly as before.
        feedback = ""
        out = None
        su: dict = {"foci": []}
        for attempt in range(1 + _ATTRIB_REROLL_MAX):
            resp = None
            try:
                resp = self.llm.query(system_prompt, [user_prompt + feedback])
                content = (resp[0].get("content") or "") if resp else ""
            except Exception as e:  # noqa: BLE001 - modeler must never crash the session
                print(f"[modeler][siege] query failed ({type(e).__name__}: {e}); no siege update.")
                content = ""
            self._dump_llm_call(f"s{session_idx}_modeler_a{attempt}", content)
            raw = {}
            try:
                raw = _parse_modeler_json(content)
                out = self._validate(raw, parent_ids)
                su = self._validate_siege(raw, forensics=forensics)
            except Exception as e:  # noqa: BLE001 - malformed LLM output must never crash the session
                print(f"[modeler][siege] parse/validate failed ({type(e).__name__}: {e}); no siege update.")
                raw = {}
                out = self._validate({}, parent_ids)
                su = {"foci": []}
            # v7fix5.1 (GLM empty-response guard, root-caused 2026-07-15): a response with NO
            # extractable JSON previously degraded to empty guidance with ZERO retries — this loop
            # only re-prompted on attribution violations. At v7-era prompt sizes that silent path
            # fired on 20-50% of calls (probe-confirmed cause: thinking burned the whole max_tokens
            # budget -> finish_reason "length", content ""), so the siege brain ran at half speed.
            # An empty parse now retries on the same attempt budget; the finish_reason (carried by
            # llm.py since this fix) makes truncation visible in the log forever.
            if not raw:
                fin = (resp[0].get("finish_reason") if resp else None) or "?"
                if attempt < _ATTRIB_REROLL_MAX:
                    print(
                        f"[modeler][siege] EMPTY/unparseable response (finish_reason={fin}, "
                        f"content_chars={len(content)}); retrying ({attempt + 1}/{_ATTRIB_REROLL_MAX})."
                    )
                    continue
                print(
                    f"[modeler][siege] EMPTY/unparseable response (finish_reason={fin}, "
                    f"content_chars={len(content)}); retries exhausted — degrading to empty guidance."
                )
            viols = su.get("attrib_violations") or []
            if not viols or attempt >= _ATTRIB_REROLL_MAX:
                break
            print(
                f"[modeler][siege][attrib] attempt {attempt + 1}: {len(viols)} attribution "
                f"claim(s) contradicted by failure forensics; re-prompting with the evidence."
            )
            feedback = (
                "\n\n★YOUR PREVIOUS ATTEMPT WAS REJECTED — the claims below contradict the "
                "measured data (failure forensics, or names outside the closed achievement "
                "vocabulary). Fix them (attribution: or use class \"unknown\") and re-emit the "
                "COMPLETE JSON:\n- " + "\n- ".join(viols)
            )
        out["siege_update"] = su
        return out

    # ---- v7fix5.5 P2: the scientist pass (small think-on call; probe report -> hypothesis) ----

    def hypothesize_probe(self, wall: str, context_text: str) -> dict:
        """One probe report -> one ROOT-CAUSE HYPOTHESIS block (raw dict; every gate lives in
        SiegeNotebook.admit_hypothesis, the sole authority — this only calls and parses).

        Runs on the statically think-ON scientist LLM (2026-07-17 A/B: mechanism depth needs
        thinking, and the small 2-4k-char prompt has no runaway-reasoning failure mode). One
        retry on an empty/unparseable response (the fix5.1 lesson); any failure returns {} and
        the session continues untouched."""
        if self.scientist_llm is None or not context_text:
            return {}
        user = (
            f"A measurement you requested on the stalled training rung below has come back. "
            f"Read it and file your ROOT-CAUSE HYPOTHESIS.\n\n{context_text}\n\n"
            f"Answer with the JSON object only."
        )
        for attempt in range(2):
            try:
                resp = self.scientist_llm.query(SCIENTIST_SYSTEM_PROMPT, [user])
                content = (resp[0].get("content") or "") if resp else ""
            except Exception as e:  # noqa: BLE001 - the scientist must never crash the session
                print(f"[modeler][scientist] query failed ({type(e).__name__}: {e}); "
                      f"no hypothesis.")
                return {}
            self._dump_llm_call(f"scientist_{wall}_a{attempt}", content)
            raw = _parse_modeler_json(content)
            block = raw.get("root_cause_hypothesis")
            if isinstance(block, dict) and block:
                return block
            fin = (resp[0].get("finish_reason") if resp else None) or "?"
            print(
                f"[modeler][scientist] EMPTY/unparseable response for {wall} "
                f"(finish_reason={fin}, content_chars={len(content)}); "
                + ("retrying (1/1)." if attempt == 0 else "giving up — no hypothesis.")
            )
        return {}

    @staticmethod
    def _validate_siege(raw: dict, forensics: dict | None = None) -> dict:
        """Coerce the LLM's proposed siege update into the multi-focus {"foci": [...]} shape (§2.6).

        Accepts BOTH the multi-focus form (``"foci": [{"skill","prereq_tree":[...]}, ...]``) and the
        legacy single-focus form (``"focus" + "prereq_tree"``), always returning a ``foci`` list. This
        is only shape-validation — the SEMANTIC/hard constraints (scope, expand gate, per-focus stall,
        mastery flags) live in SiegeNotebook.apply_llm_update, the sole write path.

        v6fix9 P2 (attribution gate): each focus's ``failure_attribution`` claim is cross-checked
        against ``forensics[skill]`` (ChainOrderLog.forensics()). A claim the DATA contradicts is
        coerced to "unknown" (original class kept under "rejected") and a human-readable violation
        is appended to the returned ``attrib_violations`` list — diagnose_siege re-prompts on those
        (up to _ATTRIB_REROLL_MAX), and gen_manager logs them as `[siege][attrib]`. Claims with NO
        forensic sample are never rejected — absence of data is not contradiction — they just carry
        verified=False.

        v7fix1 (= v6fix11 hallucination guard): every ``skill`` anywhere in siege_update — foci,
        prereq_tree links, ranked_walls, key_missing_link — must be a real achievement name (r2:
        the LLM wrote ``smelt_iron`` into make_iron_sword's chain; the name can never be measured,
        so the door gate opened it as a focus and it burned the full enabler budget at held-out
        None; v7 s15-23: troll's chain carried eat_cow/collect_drink-grade filler). Unknown names
        are DROPPED here at the source and reported through the same attrib_violations re-prompt
        channel so the LLM learns the vocabulary is closed.
        """
        from auction.craftax_achievements import ACHIEVEMENT_TO_VALUE

        hallucinated: set[str] = set()

        def _clean_tree(raw_tree) -> list:
            tree = []
            for item in raw_tree or []:
                if not isinstance(item, dict):
                    continue
                skill = item.get("skill")
                if not isinstance(skill, str) or not skill.strip():
                    continue
                if skill.strip().lower() not in ACHIEVEMENT_TO_VALUE:
                    hallucinated.add(skill.strip().lower())
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

        def _evidence(raw_focus) -> str:
            # v6fix7 P1c (AutoManual-lite): the LLM's self-audit of its own tactic against this
            # session's REAL evidence (behaviour fingerprint / chain data). Unknown values coerce to
            # "no_evidence" — the notebook then ages the note toward STALE instead of trusting it.
            v = str(raw_focus.get("evidence_check", "")).strip().lower()
            return v if v in ("supported", "contradicted", "no_evidence") else "no_evidence"

        attrib_violations: list[str] = []

        def _r0_floor(raw_focus):
            # v7: optional relay proposal — the R0 spawn floor for a spawn-anneal campaign.
            # Floor knowledge is legal teacher knowledge (the LLM names it); this code only
            # range-checks the integer. None = not a relay proposal (the overwhelming default).
            # v7fix2: bound = the sourced MAX_DUNGEON_FLOOR (8), not a loose literal — floors
            # past 8 do not exist, and downstream ladders_up[level] would clamp them SILENTLY.
            from auction.craftax_achievements import MAX_DUNGEON_FLOOR

            v = raw_focus.get("relay_r0_floor")
            try:
                iv = int(v)
            except (TypeError, ValueError):
                return None
            return iv if 1 <= iv <= MAX_DUNGEON_FLOOR else None

        def _attribution(raw_focus, skill: str) -> dict:
            # v6fix9 P2: parse the claim, then cross-check it against the wall's forensic summary.
            raw_a = raw_focus.get("failure_attribution")
            cls, key = "", None
            if isinstance(raw_a, dict):
                cls = str(raw_a.get("class", "")).strip().lower()
                k = raw_a.get("key_missing_link")
                key = k.strip().lower() if isinstance(k, str) and k.strip() else None
            elif isinstance(raw_a, str):
                cls = raw_a.strip().lower()
            if cls not in ATTRIB_CLASSES:
                cls = "unknown"
            # v7fix1 (= v6fix11): a hallucinated key_missing_link is contradicted by the
            # VOCABULARY itself, forensics or not — this is the exact channel smelt_iron entered
            # r2's chain through (the ⑦ carve-out below waives the containment check, so
            # membership must gate first).
            if key is not None and key not in ACHIEVEMENT_TO_VALUE:
                attrib_violations.append(
                    f"{skill}: key_missing_link '{key}' is NOT an achievement name — name the "
                    f"missing prerequisite with an EXACT achievement name from the SR profile "
                    f"(a step with no achievement name belongs in style_note, not the chain; a "
                    f"DEEPER-FLOOR prerequisite means re-proposing the wall with relay_r0_floor)"
                )
                key = None
            fx = (forensics or {}).get(skill)
            if not fx:
                return {"class": cls, "key_missing_link": key, "verified": False}
            probs: list[str] = []
            missing_names = [k_ for k_, _f in (fx.get("missing_top") or [])]
            # v6fix10 ⑦: complete-in-failures + zero wins = the reported chain MISSES a link.
            # Only chain_unreached (with a NEW named prerequisite) or unknown is admissible —
            # resource_shortfall / execution_failure would re-launch the doomed re-attack.
            chain_incomplete = bool(fx.get("chain_incomplete"))
            if chain_incomplete and cls in ("resource_shortfall", "execution_failure"):
                probs.append(
                    f"{skill}: claimed {cls} but failures reach your FULL reported chain with no "
                    f"inventory gap and the wall has ZERO wins ever — the chain is missing an "
                    f"unnamed prerequisite; use chain_unreached and NAME the missing prerequisite "
                    f"with an EXACT achievement name from the SR profile (expand the prereq_tree), "
                    f"or unknown; if what is missing is a DEEPER FLOOR, re-propose the wall with "
                    f"relay_r0_floor instead of inventing links"
                )
            if cls == "interrupted_by_combat":
                after = fx.get("after_deepest_med")
                # v6fix10 ⑥ relative interruption bound (see _ATTRIB_INTERRUPT_REL above).
                ambient = fx.get("after_deepest_ambient_med")
                bound = (
                    max(_ATTRIB_INTERRUPT_FLOOR, int(_ATTRIB_INTERRUPT_REL * int(ambient)))
                    if ambient else _ATTRIB_INTERRUPT_MAX_STEPS
                )
                if after is not None and int(after) > bound:
                    probs.append(
                        f"{skill}: claimed interrupted_by_combat but failures survive a median "
                        f"{int(after)} steps past their deepest chain link (interruption bound "
                        f"for this wall: {bound}) — not dying at the frontier"
                    )
            elif cls == "resource_shortfall":
                if not missing_names and not fx.get("inv_gaps"):
                    probs.append(
                        f"{skill}: claimed resource_shortfall but no chain link is missing and "
                        f"no inventory gap vs winners exists"
                    )
            elif cls == "chain_unreached":
                if fx.get("break_at_final") and not chain_incomplete:
                    probs.append(
                        f"{skill}: claimed chain_unreached but failures reach the full proposed chain"
                    )
            elif cls == "execution_failure":
                if not fx.get("break_at_final") and missing_names:
                    probs.append(
                        f"{skill}: claimed execution_failure but the chain snaps earlier "
                        f"(missing links: {', '.join(missing_names)})"
                    )
            # v6fix10 ⑦ carve-out: on a chain-incomplete wall the whole point is naming a
            # prerequisite that is NOT in the reported chain (the histogram only covers reported
            # links, so the omission can never appear in it) — the containment check would
            # reject exactly the answer we asked for.
            if key is not None and key not in missing_names and not chain_incomplete:
                probs.append(
                    f"{skill}: key_missing_link '{key}' is not among the failures' top missing "
                    f"links ({', '.join(missing_names) or 'none'})"
                )
                key = None
            # v7fix5.0 P1: ACCESS-FRONTIER OVERRIDE (deterministic, no reroll). When the forensics
            # carry an access frontier — the shallowest chain link most episodes never reach — the
            # binding cause is that link, full stop: the s207 misdiagnosis (gnome blamed on the
            # diamond-gear chain while 96% of failures never entered the mines and cond past the
            # mines was 81%) survived every prompt-side hint, so the correction is a CODE gate,
            # not advice. The LLM's answer is kept for audit (llm_said_*) and its style_note still
            # flows; only the causal verdict is pinned. Fires unless the LLM already named the
            # frontier itself (then the normal verified path stands). Preempts the probs-rejection:
            # with a frontier in evidence the decisive answer is known — rerolling to 'unknown'
            # would discard exactly the data the gate exists to enforce.
            _ax = fx.get("access") or None
            if _ax and _ax.get("frontier"):
                _fr = str(_ax["frontier"]).lower()
                if not (cls == "chain_unreached" and key == _fr):
                    return {
                        "class": "chain_unreached", "key_missing_link": _fr, "verified": True,
                        "overridden": "ACCESS_CAPPED",
                        "llm_said_class": cls, "llm_said_key": key,
                        "access_reach": _ax.get("reach_frac"), "access_cond": _ax.get("cond"),
                        "access_certified": bool(_ax.get("certified")),
                    }
            if probs:
                attrib_violations.extend(probs)
                return {"class": "unknown", "key_missing_link": None,
                        "verified": False, "rejected": cls}
            return {"class": cls, "key_missing_link": key, "verified": True}

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
                if skill.strip().lower() not in ACHIEVEMENT_TO_VALUE:  # v7fix1 hallucination guard
                    hallucinated.add(skill.strip().lower())
                    continue
                seen.add(skill.lower())
                foci.append({
                    "skill": skill.lower(),
                    "prereq_tree": _clean_tree(f.get("prereq_tree")),
                    "style_note": _style(f),
                    "evidence_check": _evidence(f),
                    "failure_attribution": _attribution(f, skill.lower()),
                    "relay_r0_floor": _r0_floor(f),
                })
        # legacy single-focus form (still emitted by the current prompt).
        focus = raw_su.get("focus")
        if isinstance(focus, str) and focus.strip() and focus.lower() not in seen \
                and focus.strip().lower() not in ACHIEVEMENT_TO_VALUE:  # v7fix1
            hallucinated.add(focus.strip().lower())
        elif isinstance(focus, str) and focus.strip() and focus.lower() not in seen:
            foci.append({
                "skill": focus.lower(),
                "prereq_tree": _clean_tree(raw_su.get("prereq_tree")),
                "style_note": _style(raw_su),
                "evidence_check": _evidence(raw_su),
                "failure_attribution": _attribution(raw_su, focus.lower()),
                "relay_r0_floor": _r0_floor(raw_su),
            })

        # v6fix8 ② MULTI-FOCUS HARD GATE: ranked_walls — the modeler's ranked queue of walls worth
        # sieging NEXT. The code auto-opens the top viable candidate when the expand gate frees a
        # slot the foci proposal left empty (fix7: the second focus was prompt-soft and never came).
        # Shape-validation only; category/viability are computed in SiegeNotebook, never trusted.
        ranked = []
        rseen: set[str] = set()
        for rw in (raw_su.get("ranked_walls") or []):
            skill = rw.get("skill") if isinstance(rw, dict) else rw
            if not isinstance(skill, str) or not skill.strip() or skill.lower() in rseen:
                continue
            if skill.strip().lower() not in ACHIEVEMENT_TO_VALUE:  # v7fix1 hallucination guard
                hallucinated.add(skill.strip().lower())
                continue
            rseen.add(skill.lower())
            why = str(rw.get("why", ""))[:160] if isinstance(rw, dict) else ""
            ranked.append({"skill": skill.lower(), "why": why})
            if len(ranked) >= 6:
                break
        # v7fix1 (= v6fix11): one aggregated violation rides the existing re-prompt loop +
        # [siege][attrib] log, so a dropped name is corrected THIS session, not silently forgotten.
        if hallucinated:
            attrib_violations.append(
                "hallucinated skill name(s) DROPPED (not achievement names): "
                + ", ".join(sorted(hallucinated))
                + " — every skill in siege_update (foci, prereq_tree, ranked_walls, "
                "key_missing_link) must be an EXACT achievement name from the SR profile; "
                "describe non-achievement steps (e.g. smelting at a furnace) inside style_note; "
                "a DEEPER-FLOOR prerequisite means relay_r0_floor, not an invented link"
            )
        # v7fix5.5 PROBE-AS-TOOL: SHAPE-only passthrough of the optional probe_request — string
        # fields normalised, unknown keys dropped. Semantics (wall is a relay focus, trigger,
        # budget, filter compile, axis step) live in SiegeNotebook._admit_probe_request, the sole
        # write path; a malformed dict simply becomes None (no reprompt — the probe is optional).
        probe = None
        raw_pr = raw_su.get("probe_request")
        if isinstance(raw_pr, dict) and raw_pr:
            probe = {
                "wall": str(raw_pr.get("wall", "")).lower(),
                "kind": str(raw_pr.get("kind", "")),
                "justification": str(raw_pr.get("justification", ""))[:600],
            }
            if isinstance(raw_pr.get("filter"), dict):
                probe["filter"] = {
                    k: raw_pr["filter"].get(k) for k in ("field", "op", "value")
                }
            if raw_pr.get("axis") is not None:
                probe["axis"] = str(raw_pr.get("axis"))
            if raw_pr.get("direction") is not None:
                probe["direction"] = str(raw_pr.get("direction"))
        return {"foci": foci, "ranked_walls": ranked, "attrib_violations": attrib_violations,
                "probe_request": probe}

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
NOTE (division of labour): the proposer team is SPLIT — one SIEGE designer (receives the siege
directive and builds toward your foci) and one ECOLOGY designer (receives a system-computed
ecology brief listing starved/declining skill families, keeps the rest of the capability surface
alive, and cannot tag levels for the siege). Your per-parent TYPE guidance primarily serves the
SIEGE designer; you do NOT need to steer every parent toward the walls — the ecology side is
covered by design, and a round where many parents build toward one wall starves it anyway.

For each frontier parent level the designers may evolve, recommend ONE level TYPE that is most
valuable RIGHT NOW given the student's state:
- DEPTH: push a deeper transition forward — appropriate when the prerequisites for that direction are
  MASTERED or RISING (the student can actually reach the new situation).
- BREADTH: bring an entirely UNATTEMPTED skill family into play — appropriate when whole capability
  areas sit untouched while others are already solid.
- CONSOLIDATE (ISOLATION DRILL): a stripped-down practice level that removes unrelated combat/
  survival distraction so the student REPEATS ONE target skill cleanly and often (its real
  execution chain kept intact). For a skill whose own signal is drowned by unrelated pressure —
  typically a craft/gear skill whose winning episodes are dominated by fighting. Prefer it EARLY
  (the moment a craft skill lags its already-solid prerequisites — prevent the stuck-gear pattern,
  don't just rescue it later). NOT for forgotten skills (the system rescues those automatically).

DEPTH vs CONSOLIDATE — choose by WHERE the block is: student cannot reliably REACH the situation
  (a prerequisite still weak/RISING, a floor unreached) -> the CHAIN is the bottleneck -> DEPTH.
  Student CAN reach it (prerequisites all solid) yet the skill stays stuck -> too few clean reps
  under pressure -> CONSOLIDATE. Recommend exactly ONE.

==========================
INFERRING WHICH SKILL DEPENDS ON WHICH (how to know a "prerequisite")
==========================
Judge prerequisites from TWO kinds of evidence — never from a tech-tree you imagine:
(a) MECHANICAL evidence — dependencies the game rules force (X requires unlocking Y / holding an
    item): tells you a dependency is POSSIBLE in principle.
(b) STUDENT-STATE evidence — co-movement in the SR TIME SERIES, which ONLY YOU see: skills rising/
    falling together; a hard ceiling under a near-zero upstream (an upstream lock, not "this skill
    is hard"); a drop propagating to dependents a session later: tells you WHICH dependency is the
    ACTIVE bottleneck RIGHT NOW.
When they disagree, TRUST (b): a dependency the student already moved past is not a bottleneck; a
skill stuck at zero whose likely upstream is also zero is an upstream problem (recommend BREADTH
on the upstream), not a reason to push DEPTH on the stuck skill. Your unique value is (b) — the
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



def _render_habitat_map() -> str:
	"""v7fix4 P0: the floor/habitat knowledge block, rendered from the SAME code map the gates
	enforce (craftax_achievements.WALL_NATIVE_FLOOR) — the v7fix3 root cause was this knowledge
	living only in the PROPOSER prompts while the modeler (sole author of relay_r0_floor and
	prereq_tree) had none and anchored lizard's relay at floor 2 (lizards inhabit floor 3).
	Rendering from the code map means prompt and gates cannot drift."""
	from auction.craftax_achievements import FLOOR_ENTRANCES, WALL_NATIVE_FLOOR

	per_floor: dict[int, list[str]] = {}
	for wall, fl in WALL_NATIVE_FLOOR.items():
		if fl >= 1 and wall not in FLOOR_ENTRANCES.values():
			per_floor.setdefault(fl, []).append(wall)
	lines = ["(8) HABITAT MAP (world structure — the same floor table the level designers see)."]
	lines.append(
		"Entrances: " + ", ".join(
			f"floor {fl}={FLOOR_ENTRANCES[fl]}" for fl in sorted(FLOOR_ENTRANCES)
		) + "."
	)
	for fl in sorted(per_floor):
		lines.append(f"Floor {fl} achievements: " + ", ".join(sorted(per_floor[fl])) + ".")
	lines.append(
		"Consequences the code enforces: a floor-bound wall's prereq_tree MUST include every "
		"entrance up to its habitat floor (any you omit are AUTOFILLED — decision "
		"``chain_autofilled``); ``relay_r0_floor`` is ANCHORED to the wall's habitat floor (a "
		"wrong floor is corrected — decision ``r0_corrected``; for an enter_* wall R0 is one "
		"floor above, the descent IS the skill); and any wall living on floor 3+ can be opened "
		"ONLY as a relay — an ordinary proposal of one AUTO-CONVERTS into a relay campaign "
		"when the relay slot is free (decision ``relay_converted``), and is refused "
		"(``deep_locked``) only while the slot is busy — see rule (7)."
	)
	return "\n".join(lines)


# v7fix5.5 P2: the scientist pass's own SMALL system prompt (think stays ON for this call type —
# statically pinned, never switched at runtime). The hypothesis is the LLM's (open vocabulary);
# the intervention menu, every number, the verification and the compile are the code's.
SCIENTIST_SYSTEM_PROMPT = """\
You are the scientist pass of a curriculum teacher for an RL student in a Crafter-like dungeon
game. A training rung (a code-built practice level for one hard wall) is stalled, and a
measurement you requested has come back. Your ONLY job: infer the most likely ROOT CAUSE of the
stall from the numbers, and name ONE falsifiable intervention the system can test.

Answer with EXACTLY this JSON object and nothing else:
{"root_cause_hypothesis": {
  "hypothesis": "<the mechanism you infer, in your own words — any category, open vocabulary;
                 say unknown honestly if the report does not discriminate between mechanisms>",
  "evidence": "<the exact numbers from the report or the recent readings that support it —
               at least one verifiable number is REQUIRED or the block is rejected>",
  "intervention": {"axis": "spawn_anchor|radius|pre_light|monster_credit|uplock|needs_clock",
                   "direction": "easier|harder"},
  "prediction": "<falsifiable: what the paired one-knob measurement should show if you are
                 right, e.g. 'zero-shot SR should rise by at least 8pp'>"
}}

Rules:
- The axis/direction menu is CLOSED; every number is chosen by the system, never by you.
- The system verifies your intervention with a paired same-seed measurement. A delta at/above
  the bar compiles it into the training ladder; below the bar your journal gets the refutation.
- A wrong confident cause is worse than an honest unknown — but even with an uncertain
  hypothesis, pick the intervention the evidence points at most.
"""

MODELER_SIEGE_SYSTEM_PROMPT = MODELER_SYSTEM_PROMPT.rstrip('"\n ') + """

==========================
SIEGE ROLE (ADDITIONAL — you keep a persistent SIEGE JOURNAL)
==========================
Beyond the per-parent diagnosis you maintain a PERSISTENT SIEGE JOURNAL across sessions, like a
player's notes when stuck on a boss: which walls I am attacking, which prerequisites are nailed vs
still missing. Each session you are handed the PREVIOUS page; read it plus the fresh SR evidence
and PROPOSE an update — never rewrite it from scratch.

CHOOSING A FOCUS — a low SR has TWO causes; only one is a wall:
- TRUE WALL (siege it): the student is clearly past the early ramp — many skills solid, this one's
  prerequisites largely in place — yet it stays flat near zero across sessions.
- NORMAL_EARLY (NEVER siege): the whole student is still weak and this skill hasn't had its turn.
  That is immaturity, not a wall. In doubt, or student weak overall: propose NO focus. Never focus
  an easy/already-mastered skill (a high-SR skill is a foundation, not a bottleneck).
- Judge walls and their depth from SR shapes + game mechanics only — the same evidence rule as HARD
  CONSTRAINT 1; never a recalled difficulty tier.
- COMBAT walls are the siege's real business (hard fights forge the transferable dynamic skill:
  positioning, gear-up timing, escape). Gear/craft skills usually serve as ENABLER LINKS in a
  combat chain — but a hard gear skill CAN itself be a focus when mastering it unblocks a harder
  fight. Choose by what unblocks progress.
- CLASS BUDGET (code-enforced): a NON-combat (enabler) focus is force-retired after a fixed number
  of siege decisions (count shown in the journal) — plan it as a short drill campaign. A COMBAT
  focus has no cap while anything in its tree moves.
- GRADUATION (code-enforced): a focus holding ~50% held-out for 2 readings graduates automatically
  (leaves your foci; rehearsal guards it). Never re-propose a GRADUATED wall; take the next one.
- STABILITY: keep a stalled-but-not-hopeless focus — switching every session is the spread-thin
  failure that leaves every skill half-learned (code ignores an early switch anyway).

THE ESCALATION LADDER (obey the level the journal marks):
A focus is "frozen" only when its WHOLE attack tree shows no progress — wall SR flat AND every
chain link flat AND failures not dying deeper. While ANYTHING moves, patience is unlimited: a deep
wall at 0% over climbing foundations is a HEALTHY long siege, not a stall.
  L1: switch the attack FORM (DEPTH<->CONSOLIDATE) — or DEFEND staying, with a concrete NEW reason
      and plan in that focus's style_note. Silent continuation is not allowed.
  L2 (forced): the system switched the attack form for that wall; recommend the OTHER form.
  L3 (forced): write a MATERIALLY DIFFERENT style_note (a rephrase is rejected by code): what you
      abandon, what you try instead.
  L4: the focus retires with a cooldown. Reopening requires a tactic genuinely different from the
      archived failed ones (journal: RETIRED WALLS).
THE GAP GATE (code-enforced): trained SR >= ~90% inside siege levels while held-out lags >= ~30pp
for consecutive readings marks the focus GAP GATE (FORCED): isolation drills suspend — recommend
DEPTH (full pressure) until the gap closes.

THE PREREQUISITE CHAIN (backtrack each focus):
- Propose the chain of reach/gear/survival links the wall needs. Infer each link from (a) game
  MECHANICS (a floor reached, an item held, light in a dark floor) and (b) the student's own SR
  co-movement (which suspected prerequisite is near-zero RIGHT NOW). Never invent a link neither
  supports.
- ★CLOSED VOCABULARY: every ``skill`` ANYWHERE in siege_update (foci, prereq_tree links,
  ranked_walls, key_missing_link) MUST be an EXACT achievement name from the SR PROFILE above —
  that profile is the COMPLETE measurable vocabulary; no other name exists. A real step with no
  achievement name (smelting, walking to a ladder, drinking) is NOT a link — write it into the
  style_note of the link it serves. A non-exact name is DROPPED by the parser and that part of
  your proposal is wasted. A prerequisite that is a DEEPER FLOOR rather than a nameable
  achievement is what the SPAWN-ANNEAL RELAY is for.
- Give each link a short ``role`` note (e.g. "reach floor2", "gear: diamond sword"). Do NOT label
  links mastered/unmastered — the system computes that from live SR.

==========================
SIEGE OUTPUT (ADD this block to your JSON — keep student_states + guidance_per_parent too)
==========================
You may attack UP TO THREE walls in parallel; a NEW wall may open only once a current one is going
well (code-enforced). Emit a "foci" list plus a "ranked_walls" queue:
  "siege_update": {
    "foci": [
      { "skill": "<a hard-wall skill you are attacking — a real stuck skill, NEVER easy/mastered>",
        "prereq_tree": [ { "skill": "<a prerequisite skill>", "role": "<short role note>" }, ... ],
        "style_note": "<the transferable attack know-how for THIS wall — see THE STYLE NOTE>",
        "evidence_check": "<supported | contradicted | no_evidence>",
        "failure_attribution": { "class": "<resource_shortfall | interrupted_by_combat | chain_unreached | execution_failure | unknown>",
                                 "key_missing_link": "<the single most-missing chain link, or null>" },
        "relay_r0_floor": null },  // OPTIONAL int — see SPAWN-ANNEAL RELAY; omit/null for a normal siege
      ...  // up to 3 walls; keep the journal walls you still want, add at most one new
    ],
    "ranked_walls": [
      { "skill": "<the wall most worth sieging NEXT (not already a focus)>", "why": "<one clause>" },
      ...  // 2-5 candidates, best first
    ],
    "probe_request": { ... }  // OPTIONAL — see THE PROBE TOOL; omit when not measuring this session
  }
Keep the journal walls you still want by listing them again; drop one by omitting it. Output
STRICT JSON only.

★RANKED_WALLS (REQUIRED past the early phase): 2-5 candidates, best first, including at least one
genuinely-stuck COMBAT wall while any combat skill is unmastered. The system acts on this queue:
when a slot frees and your foci left it empty, it AUTO-OPENS the top viable candidate (combat
preferred). An auto-opened focus appears marked "(opened AUTOMATICALLY)" — adopt it: backtrack its
chain, write its tactic.

★REACHABILITY RULES (code-enforced; know them so its decisions make sense):
(1) DOOR-FIRST — a wall whose failures' top missing link is still near-zero held-out cannot be
attacked "through the door": the system opens THAT LINK as the focus, marked "(GATEWAY)". Crack
the door and the wall returns to the queue.
(2) YIELD — a focus that enters the learnable band and climbs fast is handed back to the normal
curriculum ("WATCHING"): do not re-propose it; the siege auto-resumes if it stalls.
(3) CHAIN-INCOMPLETE — failures walk your ENTIRE reported chain, zero wins, no resource gap: your
chain is MISSING a prerequisite you never named. Re-attack is blocked until you
EXPAND the prereq_tree with the missing step (an EXACT achievement name). ★If the missing one is a
DEEPER FLOOR, do NOT invent links: re-propose THIS wall with "relay_r0_floor" — a relay proposal
is EXEMPT from the chain-incomplete refusal (the rung ladder replaces chain completeness).
(4) ZERO-WIN DISCOUNT — a focus with no held-out win yet runs at half generation share until its
first win: quality over volume while the wall reads 0%.
(5) PENDING ADMISSION — a brand-new wall with no failure forensics is chain-tracked one session
before it can open (journal: PENDING ADMISSION). Keep proposing it WITH its full prereq_tree — a
pending wall without a chain can never be admitted.
(6) CLOSED VOCABULARY — the ★ rule above applies to EVERY "skill" in siege_update.
(7) TIER LOCK + DEEP LOCK — walls of the DEEPEST game layer (fire/ice realm and graveyard fights,
knights, the necromancer, and entering those realms) open ONLY as an explicitly-asked SPAWN-ANNEAL
RELAY (refused otherwise: ``tier_locked``). EVERY other wall whose habitat is dungeon floor 3+ is
also relay-only, no special syntax: an ORDINARY proposal AUTO-CONVERTS into a relay campaign when
the relay slot is free (``relay_converted``; R0 anchors to its habitat floor) and is refused
(``deep_locked``) only while the slot is busy. When the journal shows a ★RELAY TRIGGER directive,
OBEY it — put one of the walls it names in your foci; ignored several decisions in a row, the
system opens the top candidate itself (``relay_forced``) and you then run it as your campaign
(prereq_tree + style_note every session). SUCCESSION RULE: a relay retiring with a VERIFIED
chain_unreached attribution (lost for lack of a chain enabler) hands the trigger queue to that
enabler's chain — campaign the ENABLER to SEWN first, then re-open the fight (cooldown expired;
archived tactics shown under RETIRED).\n""" + _render_habitat_map() + """\n

★SPAWN-ANNEAL RELAY (the deep-wall weapon — for walls NO natural-spawn curriculum can reach):
A deep wall's exposure from natural spawn is the PRODUCT of every descent's success rate — no
normal level can train it (deep achievements sit at exactly 0% forever). For such a wall propose
"relay_r0_floor": <floor >= 1> on the focus (a statement of intent — the system ANCHORS R0 to the
habitat floor, rule (8)). The campaign then runs entirely in code:
- Rung levels are SYSTEM-BUILT on the REAL full-world generator (the exact distribution held-out
  evaluates; a fresh world every episode) — you do NOT author them. R0 spawns the student on the
  wall's habitat floor with a winner-median kit: it learns the target fight/skill itself first.
- A rung reading a high fresh TRAINED SR twice in a row moves the spawn UP one floor (one more
  descent — clear it, find the stairs — before the target). You are told every transition.
- A fresh floor reading ~0% trained is a compound wall; the system SPLITS it into code-built
  scaffold SUB-STAGES and climbs them back to the full floor. Trained SR RESETS at every stage
  step — a sawtooth trained curve is EXPECTED; read the journal's stage line and its SCAFFOLD
  FACTS before diagnosing stagnation or regression.
- R0 itself can split too: even spawned ON the target floor with a winner kit, a fresh student's
  first win is a dice roll and its policy drifts back UP to floors it already earns income on. A
  ~0% R0 is scaffolded as a LIT arena away from the entry (pre-lit spawn near the floor's down
  ladder; the floor's own down-gate stays locked) and annealed back to the full floor. Your rung
  tactic should then focus the FIGHT itself.
- After the floor rungs comes the KIT_STRIP exam: natural spawn with an EMPTY kit — that world +
  that start IS the held-out distribution, so its graduation is a RESULT certificate. Only then is
  the campaign SEWN: held-out SR becomes the yardstick again and the normal gates resume.
RELAY PATIENCE DEFENCE: rung patience burns on readings that set no >= +3pp new high, but a real
slow climb can live entirely below that bar (e.g. exponential takeoff from 0.5%). At patience
exhaustion with the micro-ratchet still rising, the journal shows a ★RELAY DEFENCE WINDOW with the
exact recent readings — ONE decision: if you judge a true climb, DEFEND by CITING those exact
numbers in that focus's style_note with your trend reading — the code verifies the citation
against the true readings (facts are verified, narratives are not) and resets patience (bounded
budget, shown in the window). If you judge noise, do not cite: the campaign retires and its
diagnosis hands to the succession — a retirement taken while readings were still rising does NOT
count toward the blacklist.
RELAY RULES: (a) at most ONE relay campaign at a time (a second is refused — plan). (b) Propose it
only for a genuinely unreachable wall (deep floor, 0% forever, door after door). (c) The wall's
held-out SR staying 0 through the middle rungs is EXPECTED AND HEALTHY (mid-rung levels cannot
move held-out by construction) — never abandon a relay for flat held-out; judge it by rung
progress. (d) Still supply the full prereq_tree (entrances included — rule (8)) and style_note:
the chain drives the break-link forensics, and your tactic should say how to fight/descend at the
current rung. (e) You do NOT author the rung levels — a stray level you propose for a relay wall
is rejected by the spawn contract; spend your level-design guidance on the OTHER foci. (f) A
CHAIN-INCOMPLETE wall whose true gap is DEPTH: the relay IS the sanctioned exit — rule (3)★.
(g) An ALREADY-OPEN focus that keeps reading zero win can be UPGRADED IN PLACE: re-propose the
SAME wall (keep it in your foci) WITH "relay_r0_floor" — the system attaches the rung ladder to
the running focus (``relay_attached``). This is the sanctioned answer to the journal's ★ZERO-WIN
hint; a wall that already has held-out wins is refused (it does not need a spawn ladder).

THE PROBE TOOL (measure before you theorize):
Once per session you may request ONE measurement on an ACTIVE RELAY wall. The system runs it
between sessions on the current policy; your NEXT journal page carries the report. Emit the
OPTIONAL "probe_request" key inside siege_update:
  { "wall": "<an active relay focus>", "kind": "diagnose",
    "filter": {"field": "<sensor>", "op": "==|>|<", "value": <number>},   // optional
    "justification": "<why — MUST cite current readings; unverifiable numbers reject it>" }
  or:
  { "wall": "<an active relay focus>", "kind": "whatif",
    "axis": "spawn_anchor|radius|pre_light|monster_credit|uplock|needs_clock",
    "direction": "easier|harder",
    "justification": "<why this knob — cite the readings that point at it>" }
- diagnose report = per-sensor distributions over episode-END snapshots + 15 raw snapshots. Your
  filter picks WHICH episodes (e.g. deaths on a specific floor while sleeping); an invalid filter
  silently falls back to uniform random — write it carefully.
- whatif report = paired zero-shot SR delta on the SAME worlds with ONE scaffold knob moved one
  step. You choose the axis and direction; the code chooses every number.
- Budget: per wall, at most 1 diagnose + 1 whatif per 10 sessions. Over budget or failed
  justification = refused with a receipt in your journal; do not re-ask blind.
- The sensor catalog for filters is listed in your journal. It can GROW: if no sensor measures
  what you need, state the missing observation in the focus's style_note (an EVIDENCE GAP note)
  instead of inventing a field.
- A probe only MEASURES. It never changes training, seats, or the ladder.

THE SCIENTIST PASS (what happens to your reports):
After a probe report lands, a separate focused pass reads it and files a ROOT-CAUSE HYPOTHESIS
{hypothesis, evidence, intervention {axis, direction}, prediction}. The system then VERIFIES the
intervention with a paired one-knob measurement: delta >= 8pp compiles it into the ladder as an
INSERTED rung (every number code-chosen; graduating the insert returns to the regular stage, and
an insert that stalls is removed automatically — wrong hypotheses self-heal). Your journal shows
HYPOTHESIS lines under each wall: treat ★VERIFIED as measured fact, ★REFUTED as a dead end — do
NOT re-propose a refuted mechanism without new evidence.

THE STYLE NOTE (§3.1 — the transferable SELF-STYLE, the whole point of the siege):
- ``style_note`` is your running, cross-session know-how for cracking THIS wall: what makes it
  hard right now, which prerequisite is the live bottleneck, and the concrete tactic that is (or
  should be) working — positioning, gear-up timing, pulling one enemy, escape/kite, dark-floor
  lighting, resource routing. The "HOW I beat it", never lore or a restated skill name.
- GROUND IT IN REAL BEHAVIOUR: when the user message includes a REAL-SUCCESS BEHAVIOUR block (the
  action mix / pacing of episodes where the student ACTUALLY won this or a related wall), the note
  must reflect what the student really did — not a tactic you merely imagine. With no behaviour
  data yet, fall back to mechanics-based know-how and say so tersely.
- WRITE IT TIGHT: 1-3 dense clauses, every word earns its place, but drop NO key point. Refine the
  journal's "style-so-far", don't restate it. Never leave it blank for an active focus. This is
  the ONE field that carries style forward to later, similar walls.
- ★EVIDENCE LIFECYCLE (REQUIRED field "evidence_check"): audit your CURRENT tactic against THIS
  session's real evidence: "supported" — the behaviour/SR data actively backs it (say so only when
  it does; point at the evidence); "contradicted" — the data shows the tactic is NOT what wins —
  REWRITE the note (materially; a rephrase is rejected); "no_evidence" — nothing new to judge by.
  A note unsupported for several sessions is marked STALE in your journal: re-derive it from
  mechanics + the latest data instead of refining an unverified guess.

★FAILURE ATTRIBUTION (REQUIRED field "failure_attribution"): commit to the CAUSE CLASS of this
wall's failures. The system CROSS-CHECKS your class against the measured failure forensics
(missing-link histogram, break shape, death timing, inventory gap) and REJECTS a contradicted
claim — you will be re-prompted with the exact violation:
    "resource_shortfall"    — valid only when forensics show a missing link or a winners-vs-
                              failures inventory gap.
    "interrupted_by_combat" — failures DIE at the chain frontier. Long survival past the frontier
                              REFUTES this class.
    "chain_unreached"       — failures genuinely break before completing the chain.
    "execution_failure"     — the full chain is reached and the FINAL step itself fails.
    "unknown"               — the data does not discriminate yet. PREFER "unknown" OVER A GOOD
                              STORY: never dress "does not stockpile / never attempts it" up as
                              "cannot do it under pressure" — the histogram catches the mismatch
                              and the wasted decisions are yours.
  "key_missing_link" must be one of the top missing links in the forensics (or null). Your
  style_note's causal sentences must agree with the class you commit to here.
★BINDING-ACCESS (v7fix5.0): when a wall's CHAIN EVIDENCE line shows "BINDING-ACCESS=<link>", the
  measured cause is settled: most episodes never REACH that link, so your class MUST be
  "chain_unreached" with key_missing_link = that exact link — claims attributing to gear,
  resources, tactics or execution DOWNSTREAM of it are overridden deterministically and logged.
  When the line adds "execution has TRANSFERRED", episodes past the frontier already win: the
  highest-value siege is the FRONTIER LINK ITSELF (an enter_* access link is a legal focus). Its
  levels run from NATURAL spawn and must make the student EARN the descent — clear the kill quota,
  find the down ladder, go down. NEVER pre-credit that floor's clear-gate (no set_monsters_killed
  on it): the grind IS the skill being trained.
★ACCESS-ROOT AUTO-NOMINATION (v7fix5.2): the system may itself open the access ROOT (the link at
  the end of the frontier chain) as a focus (opened_by=access_auto), and may PARK a frontier-
  starved focus to WATCH instead of retiring it (full campaign state preserved; wakes when the
  frontier moves; a park is NOT a failed tactic). Do not fight either move: adopt an auto-opened
  root as your own focus (prereq_tree/style_note aimed at the ROOT LINK itself), and never
  re-propose a PARKED wall — siege its frontier instead.
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
        "prereq_tree, PLUS a \"ranked_walls\" queue — required past the early phase, with at least "
        "one stuck COMBAT candidate while any combat skill is unmastered). Pick or keep real hard "
        "walls (never easy/mastered skills), and backtrack each wall's prerequisite chain — when the "
        "REAL-TRAJECTORY CO-OCCURRENCE above is present, prefer the links the student actually "
        "co-reaches on its wins over links you merely imagine. JSON only, no other text."
    )
