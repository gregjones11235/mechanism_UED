"""C-0 scaffold gate (C-2-lite §3 static layer): AST fidelity check on generated task code.

Sits between compilation and preflight in the worker's validation pass. Zero LLM cost per
check; violators get their evidence formatted into a repair prompt and one/two targeted
regeneration attempts before being dropped (evolution_efficient hooks this in, flag-gated
by ``+skill_preflight.use_scaffold_gate=true``).

VIOLATION RULES (from the C2lite design doc §3, grounded in the 302-task audit):
  R1 premark_mastered    completed_achievements contains a skill the agent already masters
                         (pure noise: skips nothing the agent can't do, pollutes the audit).
  R2 focus_premarked     completed_achievements contains a RELEVANT achievement (pre-marking
                         a reward-relevant achievement short-circuits the task itself).
  R3 focus_prereq_scaffolded  a direct prerequisite of an unmastered relevant achievement is
                         scaffolded away — via pre-marking, inventory grants, or a starting
                         floor that skips the enter_* chain. This is the one-bare-step rule's
                         hard edge: the focus AND its immediate prerequisite stay bare.

The AST extraction mirrors the four-signature detector in the offline audit tool
(dicode_src/scaffold_audit.py). It is REIMPLEMENTED here rather than imported because the
root-level script is not on the import path of an installed run; test_scaffold_gate.py
asserts parity between the two on the task_19 replica so they cannot silently drift.

DELIBERATELY NOT VIOLATIONS (documented so nobody "fixes" them):
  * Near-spawn mobs (audit signature S3): placing the focus combat target near the player is
    exactly the practice the task exists to provide — the leak analysis showed terminal-skill
    repetition is where the mechanism arms' +12 points come from. bare_reverify keeps mobs too.
  * Scaffolding an UNMASTERED, non-focus, non-immediate prerequisite: that is the sanctioned
    use of scaffolding under the one-step contract.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Mapping

from auction.craftax_achievements import ALL_ACHIEVEMENTS
from dicode.skill_preflight.prereq_graph import (
    DIRECT_PREREQS,
    floor_grants,
    inventory_grants,
)
from dicode.skill_preflight.skill_scheduler import MASTERED_SR_CUT


@dataclass
class GateVerdict:
    ok: bool
    violations: list[str]          # rule tags, e.g. ["R1_premark_mastered", ...]
    evidence: str                  # human/LLM-readable indictment for the repair prompt
    parse_ok: bool = True


def _extract_scaffold_facts(code: str) -> dict:
    """AST pass: relevant achievements, pre-marked achievements, inventory items, start floor.

    Keep in lockstep with scaffold_audit.audit_code (parity-tested).
    """
    facts = {
        "relevant": [], "premarked": [], "inventory": {}, "floor": 0, "parse_ok": True,
    }
    try:
        tree = ast.parse(code)
    except SyntaxError:
        facts["parse_ok"] = False
        return facts

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                    and t.value.id == "self"):
                names = [n.attr for n in ast.walk(node.value)
                         if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                         and n.value.id == "Achievement"]
                if t.attr == "relevant_achievements":
                    facts["relevant"] = names
                elif t.attr == "completed_achievements":
                    facts["premarked"] = names
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            fn = node.func.attr
            if fn == "set_player_inventory":
                for a in node.args:
                    if isinstance(a, ast.Dict):
                        for k, v in zip(a.keys, a.values):
                            if isinstance(k, ast.Constant):
                                facts["inventory"][str(k.value)] = (
                                    v.value if isinstance(v, ast.Constant) else None
                                )
            elif fn == "set_starting_floor":
                if node.args and isinstance(node.args[0], ast.Constant):
                    try:
                        facts["floor"] = int(node.args[0].value)
                    except (TypeError, ValueError):
                        pass
    return facts


def _norm(names: list[str]) -> list[str]:
    """Achievement.FOO enum attrs -> lowercase canonical names, unknown names dropped."""
    out = []
    for n in names:
        low = n.lower()
        if low in ALL_ACHIEVEMENTS:
            out.append(low)
    return out


def check_code(
    code: str,
    sr_snapshot: Mapping[str, float],
    *,
    mastered_cut: float = MASTERED_SR_CUT,
    mastered_prereq_exemption: bool = False,
) -> GateVerdict:
    """Run the R1-R3 fidelity rules against one generated task's code.

    Args:
        code: the candidate task file (full text).
        sr_snapshot: per-achievement SR in [0,1] (SchedulerTarget.sr_snapshot, or derived
            from evaluation metrics via skill_scheduler._sr_map).
        mastered_cut: SR at/above which a skill counts as mastered (default 0.70, shared
            with the one-step prompt so the gate enforces exactly what the prompt asked).
        mastered_prereq_exemption: [v2, R3-exemption] when True, an immediate prerequisite
            that the agent already MASTERS (SR >= mastered_cut) MAY be scaffolded away
            (floor start / premark / inventory) without tripping R3. Rationale: leakage is
            defined as compressing UNMASTERED steps (bare re-verify collapse 0.65->0.01
            happened exclusively there); compressing a genuinely mastered prefix is the
            legitimate use of scaffolding and removes the "replay the mastered descent
            every episode" sampling tax that the 2e9 gap decomposition attributed the
            iron/gnomish consolidation deficit to. Default False -> v1 byte-identical.

    Returns:
        GateVerdict. ``ok=True`` also for unparseable code — a syntax failure is the
        compiler's jurisdiction, not the gate's (and compilation runs first anyway).
    """
    facts = _extract_scaffold_facts(code)
    if not facts["parse_ok"]:
        return GateVerdict(ok=True, violations=[], evidence="", parse_ok=False)

    sr = {a: float(sr_snapshot.get(a, 0.0)) for a in ALL_ACHIEVEMENTS}
    relevant = set(_norm(facts["relevant"]))
    premarked = set(_norm(facts["premarked"]))
    inv_grants = inventory_grants(facts["inventory"])
    flr_grants = floor_grants(facts["floor"])

    violations: list[str] = []
    lines: list[str] = []

    # R1: pre-marking mastered skills
    r1 = sorted(a for a in premarked if sr[a] >= mastered_cut)
    if r1:
        violations.append("R1_premark_mastered")
        lines.append(
            f"- completed_achievements pre-marks skills the agent ALREADY MASTERS "
            f"(SR>={mastered_cut:.0%}): {', '.join(r1)}. Remove them; the agent performs "
            f"mastered steps itself."
        )

    # R2: pre-marking a relevant (reward-defining) achievement
    r2 = sorted(premarked & relevant)
    if r2:
        violations.append("R2_focus_premarked")
        lines.append(
            f"- completed_achievements pre-marks RELEVANT achievements: {', '.join(r2)}. "
            f"A relevant achievement defines this task's reward and must be earned in-episode."
        )

    # R3: scaffolding away the immediate prerequisite of an unmastered focus skill
    focus = sorted(a for a in relevant if sr[a] < mastered_cut)
    scaffolded = premarked | inv_grants | flr_grants
    r3_pairs = []
    for f in focus:
        hit = sorted(
            p for p in DIRECT_PREREQS[f] & scaffolded
            if not (mastered_prereq_exemption and sr[p] >= mastered_cut)
        )
        if hit:
            r3_pairs.append((f, hit))
    if r3_pairs:
        violations.append("R3_focus_prereq_scaffolded")
        for f, hit in r3_pairs:
            how = []
            for h in hit:
                src = []
                if h in premarked:
                    src.append("pre-marked")
                if h in inv_grants:
                    src.append("granted via set_player_inventory")
                if h in flr_grants:
                    src.append("skipped via set_starting_floor")
                how.append(f"{h} ({'; '.join(src)})")
            lines.append(
                f"- focus skill '{f}' (SR {sr[f]:.0%}) has its immediate prerequisite(s) "
                f"scaffolded away: {', '.join(how)}. The focus and its direct prerequisites "
                f"must be performed bare."
            )

    if not violations:
        return GateVerdict(ok=True, violations=[], evidence="")

    evidence = (
        "[scaffold-gate] The generated code violates the scaffolding contract:\n"
        + "\n".join(lines)
        + "\nRegenerate the code with these fixes. Keep the task's objective, relevant "
          "achievements, world layout and mob placement; change ONLY the offending "
          "completed_achievements entries / inventory grants / starting floor."
    )
    return GateVerdict(ok=False, violations=violations, evidence=evidence)


def snapshot_from_metrics(evaluation_metrics: Mapping[str, float] | None) -> dict[str, float]:
    """Convenience: eval-style {'skill_x': 0..100} -> per-achievement SR in [0,1]."""
    from dicode.skill_preflight.skill_scheduler import _sr_map
    return _sr_map(evaluation_metrics)
