"""v6fix7 P0.2 — SiegeLevelValidator: code-enforced level rules with reroll feedback.

Why: the audit (fable_research_reports/idea代码实现风险审计结果.md) showed every siege-critical
constraint lived ONLY in the prompt — the proposer could gift the siege focus as a finished item
(Completed), compress the drill's real execution chain into the starting inventory, or hand back a
"drill" that still carries the parent's combat/survival goals. This module makes those rules code:

  R1_SHAPE   CONSOLIDATE (isolation drill): Relevant must contain ONLY the drilled skill + its own
             chain links (+ any still-unmastered siege links). Anything else = the drill is not
             isolated. Non-CONSOLIDATE keeps DiCode's lineage superset rule vs the parent.
  R2_GIFT    Any active siege focus (or the drill target) in Completed = gifting the finished item.
  R3_CHAIN   CONSOLIDATE: the drilled focus's prereq-tree links may NOT be compressed into
             Completed — the drill must actually perform the mine→smelt→craft sequence.
  R4_WORLD   (warn-only) after links are forced back into Relevant, the docstring should mention
             the matching resource — a crude token check to surface likely-unsolvable worlds.

Violations are first sent BACK to the proposer with explicit feedback (reroll, handled in
gen_manager); only after the reroll budget is exhausted do we fall back to mechanical fixes
(Completed→Relevant moves via enforce_completed_gate) plus a WARN. Chain data comes exclusively
from the SiegeNotebook's LLM-inferred prereq trees — no tech-tree prior is introduced.

Pure python (no jax/LLM), offline-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from auction.completed_gate import (
	_COMPLETED_RE,
	_RELEVANT_RE,
	_parse_names,
	enforce_completed_gate,
)

RULE_SHAPE = "R1_SHAPE"
RULE_SUPERSET = "R1_SUPERSET"
RULE_GIFT = "R2_GIFT"
RULE_CHAIN = "R3_CHAIN"
RULE_WORLD = "R4_WORLD"
RULE_FORM = "R5_FORM"  # v6fix7 P1a L2: ladder-forced attack form for a frozen wall

# Warn-only rules never trigger a reroll on their own.
_WARN_ONLY = {RULE_WORLD, RULE_SUPERSET}


@dataclass
class Violation:
	rule: str
	message: str
	# Names that a mechanical fallback can move Completed->Relevant (R2/R3). Empty for
	# reroll-or-warn rules (R1 shape/superset, R4).
	fixable_moves: list[str] = field(default_factory=list)


def _relevant_of(docstring: str) -> list[str]:
	m = _RELEVANT_RE.search(docstring or "")
	return _parse_names(m.group(2)) if m else []


def _completed_of(docstring: str) -> list[str]:
	m = _COMPLETED_RE.search(docstring or "")
	return _parse_names(m.group(2)) if m else []


def _focus_links(foci: list[dict]) -> dict[str, list[str]]:
	"""focus skill -> its prereq-tree link names (lowercase), from SiegeNotebook.foci."""
	out: dict[str, list[str]] = {}
	for foc in foci or []:
		skill = foc.get("skill")
		if not isinstance(skill, str):
			continue
		links = [
			l["skill"].lower()
			for l in foc.get("prereq_tree", []) or []
			if isinstance(l, dict) and isinstance(l.get("skill"), str)
		]
		out[skill.lower()] = links
	return out


def validate_level(
	description: str,
	level_meta: dict | None,
	foci: list[dict],
	unmastered: set[str] | None = None,
	parent_relevant: set[str] | None = None,
	required_forms: dict[str, str] | None = None,
) -> list[Violation]:
	"""Run all rules against one proposed level. Returns [] when clean (strict no-op path).

	Args:
		description: the proposer's docstring.
		level_meta: parsed <level_meta> (may be None — then only focus-gift R2 can fire).
		foci: SiegeNotebook.foci (active walls + their LLM-inferred prereq trees).
		unmastered: still-unmastered siege links (SiegeNotebook.unmastered_links(live_profile)).
		parent_relevant: the parent level's Relevant set (lineage superset rule, non-drills only).
	"""
	if not description or not foci:
		return []

	violations: list[Violation] = []
	relevant = _relevant_of(description)
	completed = _completed_of(description)
	links_by_focus = _focus_links(foci)
	active_focus_names = set(links_by_focus.keys())
	unmastered_l = {u.lower() for u in (unmastered or set())}

	meta_type = (level_meta or {}).get("type")
	drill_target = (level_meta or {}).get("drill_target")
	siege_wall = (level_meta or {}).get("siege_wall")

	# --- R5_FORM (ladder L2): a frozen wall's attack form is forced; a siege level for that wall
	#     built in the frozen form is rejected (reroll). ---
	if required_forms and meta_type:
		targeted = {t for t in (drill_target, siege_wall) if t}
		for wall in sorted(targeted):
			req = required_forms.get(wall)
			if req and meta_type != req:
				violations.append(
					Violation(
						rule=RULE_FORM,
						message=(
							f"The siege ladder has FORCED the attack form for {wall.upper()} to {req} "
							f"(the previous form froze the wall), but your level is {meta_type}. "
							f"Rebuild it as a {req} level, or retarget your level away from this wall."
						),
					)
				)

	# --- R2_GIFT: never hand the wall's finished item over in the initial state. ---
	gift_names = sorted((active_focus_names | ({drill_target} if drill_target else set())) & set(completed))
	if gift_names:
		violations.append(
			Violation(
				rule=RULE_GIFT,
				message=(
					f"You placed {', '.join(n.upper() for n in gift_names)} in Completed — that gifts "
					"the very skill under siege as a finished item, so the student never performs it. "
					"It MUST be in Relevant and actually executed in this level."
				),
				fixable_moves=gift_names,
			)
		)

	# --- CONSOLIDATE-only rules (need a usable drill target). ---
	if meta_type == "CONSOLIDATE" and drill_target:
		chain = links_by_focus.get(drill_target, [])

		# R3_CHAIN: the drill must keep the real execution chain in-level.
		chain_compressed = sorted(set(chain) & set(completed))
		if chain_compressed:
			violations.append(
				Violation(
					rule=RULE_CHAIN,
					message=(
						f"This is an isolation drill of {drill_target.upper()}, but you compressed its "
						f"own execution chain ({', '.join(n.upper() for n in chain_compressed)}) into "
						"Completed / starting inventory. The drill must perform the WHOLE real sequence "
						"(gather → process → craft); keep these links in Relevant and provide the raw "
						"resources in the world instead."
					),
					fixable_moves=chain_compressed,
				)
			)

		# R1_SHAPE: nothing unrelated may ride along in a drill's Relevant.
		allowed = {drill_target} | set(chain) | unmastered_l
		extras = sorted(set(relevant) - allowed)
		if extras:
			violations.append(
				Violation(
					rule=RULE_SHAPE,
					message=(
						f"An isolation drill's Relevant list may contain ONLY the drilled skill "
						f"({drill_target.upper()}) plus its own chain links — but yours also requires "
						f"{', '.join(n.upper() for n in extras)}. Strip these unrelated goals so the "
						"student repeats the target sequence cleanly (the superset rule is WAIVED for "
						"drills)."
					),
				)
			)

	# --- Lineage superset rule for non-drills (warn-only: auto-adding goals the world may not
	#     support would create unsolvable levels; the reroll feedback still asks for it). ---
	elif meta_type in ("DEPTH", "BREADTH") and parent_relevant:
		missing = sorted({p.lower() for p in parent_relevant} - set(relevant))
		if missing:
			violations.append(
				Violation(
					rule=RULE_SUPERSET,
					message=(
						f"Your Relevant list dropped {', '.join(n.upper() for n in missing)} from the "
						"trained task's — for DEPTH/BREADTH it must stay a superset (only CONSOLIDATE "
						"drills are exempt). Re-add them, or choose CONSOLIDATE if you meant a drill."
					),
				)
			)

	# --- R4_WORLD (warn-only): crude solvability token check for links we keep in Relevant. ---
	moved_or_kept = set()
	for v in violations:
		moved_or_kept.update(v.fixable_moves)
	if moved_or_kept:
		text = (description or "").lower()
		for name in sorted(moved_or_kept):
			token = name.split("_")[-1]
			if len(token) >= 4 and token not in text.replace(name, ""):
				violations.append(
					Violation(
						rule=RULE_WORLD,
						message=(
							f"{name.upper()} must now be trained in-level, but the World section never "
							f"mentions '{token}' — make sure the world actually provides it (otherwise "
							"the level is unsolvable)."
						),
					)
				)

	return violations


def reroll_worthy(violations: list[Violation]) -> bool:
	"""Warn-only rules never burn a reroll by themselves."""
	return any(v.rule not in _WARN_ONLY for v in violations)


def render_violation_feedback(violations: list[Violation]) -> str:
	"""The rejection text appended to the proposer's user prompt for the reroll."""
	lines = "\n".join(f"- [{v.rule}] {v.message}" for v in violations)
	return (
		"\n\n★★★ YOUR PREVIOUS RESPONSE WAS REJECTED BY THE SIEGE LEVEL VALIDATOR. Violations:\n"
		f"{lines}\n"
		"Regenerate your FULL response (reasoning + <level_meta> + docstring) fixing EVERY violation "
		"above. Keep everything else that was already correct. Remember: the World section must "
		"provide whatever the Relevant achievements need."
	)


def apply_fallback_fixes(description: str, violations: list[Violation]) -> tuple[str, list[str]]:
	"""Mechanical last resort after the reroll budget: move gift/chain names Completed->Relevant.

	Reuses enforce_completed_gate (same rewrite semantics). Shape/superset/world violations have no
	safe mechanical fix — they are logged by the caller and the level is accepted as-is.
	"""
	moves: set[str] = set()
	for v in violations:
		moves.update(v.fixable_moves)
	if not moves:
		return description, []
	return enforce_completed_gate(description, moves)
