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
RULE_SPAWN = "R6_SPAWN"  # v7: relay levels must spawn at the current rung's floor
# v7fix4 post-audit hardening: under system-built relay worldgen the directive PROMISES the FM
# that any level it authors for a relay wall "will be rejected by the spawn contract" — but plain
# R6 only rejects a MISMATCHED floor, and the directive itself prints the current rung floor, so
# a disobedient proposer could author a passing level and consume one of the wall's (only 2)
# discounted force-activation slots while the rung-reading quarantine keeps it evidence-blind
# anyway. The fix4 lesson applies to our own messages too: a promise the code does not enforce
# is a freedom the FM will eventually optimise into. This rule makes the promise literal.
RULE_SYS_RELAY = "R6_SYSTEM_RELAY"  # v7fix4: system-built relay walls take NO FM levels at all

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
	required_spawn_floors: dict[str, int] | None = None,
	breadth_frontier: int = 1,
	system_relay_walls: set[str] | None = None,
) -> list[Violation]:
	"""Run all rules against one proposed level. Returns [] when clean (strict no-op path).

	Args:
		description: the proposer's docstring.
		level_meta: parsed <level_meta> (may be None — then only focus-gift R2 can fire).
		foci: SiegeNotebook.foci (active walls + their LLM-inferred prereq trees).
		unmastered: still-unmastered siege links (SiegeNotebook.unmastered_links(live_profile)).
		parent_relevant: the parent level's Relevant set (lineage superset rule, non-drills only).
		required_spawn_floors: v7 — wall -> the spawn floor its relay's current rung demands
			(SiegeNotebook.required_spawn_floor). A level targeting a relay wall whose declared
			spawn_floor mismatches is rejected (R6_SPAWN, reroll).
		system_relay_walls: v7fix4 — walls whose rung levels are SYSTEM-BUILT (worldgen "base"):
			any FM level tagging one of them (siege_wall or drill_target) is rejected outright
			(R6_SYSTEM_RELAY, reroll) — even at the correct rung floor it could only consume the
			wall's discounted force-activation slots while staying quarantined from rung evidence.
			Empty/None (the "fm" ablation arm, siege off) restores plain R6 semantics byte-for-byte.
		breadth_frontier: v7fix3 P6 — deepest floor an untagged BREADTH level may spawn on
			(SiegeNotebook.breadth_frontier(); floor 1 is always unlocked). Any other non-relay
			deep spawn is rejected: deep spawn is either a relay rung contract or an in-frontier
			BREADTH ecology level, never a free scaffold for a siege target (the v5 lesson).

	v7fix3 P6: R6_SPAWN runs even with NO active foci — v7fix2's dormant phase (foci=[]) skipped
	validation entirely, which is how 33 unchecked deep-spawn levels slipped through; they turned
	out to be the run's best breadth mechanism, so the lane is now legal, BOUNDED and always-on
	instead of an accident of gate ordering. All other rules still need foci (unchanged).
	"""
	if not description:
		return []

	violations: list[Violation] = []
	relevant = _relevant_of(description)
	completed = _completed_of(description)
	links_by_focus = _focus_links(foci or [])
	active_focus_names = set(links_by_focus.keys())
	unmastered_l = {u.lower() for u in (unmastered or set())}

	meta_type = (level_meta or {}).get("type")
	drill_target = (level_meta or {}).get("drill_target")
	siege_wall = (level_meta or {}).get("siege_wall")

	# --- R6_SPAWN (v7 relay + v7fix3 P6 breadth frontier): spawn floor is a CONTRACT. ---
	if level_meta is not None:
		declared = 0
		try:
			declared = max(0, int(level_meta.get("spawn_floor") or 0))
		except (TypeError, ValueError):
			declared = 0
		targeted = {t for t in (drill_target, siege_wall) if t}
		# v7fix4 (checked FIRST — a superset of the floor-match rule for these walls): a
		# system-built relay wall takes no FM levels at all; the directive already told the
		# proposer to spend its fire elsewhere, this makes that contract code-enforced.
		sys_hits = sorted(t for t in targeted if t in (system_relay_walls or set()))
		if sys_hits:
			wall = sys_hits[0]
			violations.append(
				Violation(
					rule=RULE_SYS_RELAY,
					message=(
						f"{wall.upper()} is under a SPAWN-ANNEAL RELAY whose rung levels are "
						f"SYSTEM-BUILT on the real world generator — the system authors ALL "
						f"levels for this wall; yours cannot count as rung evidence. Remove "
						f"this level's siege_wall/drill_target tag for {wall.upper()} and "
						f"spend it on another focus or a still-unmastered link instead."
					),
				)
			)
		# sys_hits short-circuits BOTH remaining spawn checks: the one actionable fix is dropping
		# the tag, so stacking a floor/scaffold violation on top only muddies the reroll feedback.
		relay_hits = sorted(
			t for t in targeted if t in (required_spawn_floors or {})
		) if not sys_hits else []
		if sys_hits:
			pass
		elif relay_hits:
			wall = relay_hits[0]
			need = int((required_spawn_floors or {})[wall])
			if declared != need:
				violations.append(
					Violation(
						rule=RULE_SPAWN,
						message=(
							f"{wall.upper()} is under a SPAWN-ANNEAL RELAY whose current rung "
							f"requires spawn_floor={need}, but your level declares "
							f"spawn_floor={declared}. Set spawn_floor to {need} and build the "
							f"level so the student actually starts on floor {need} (with the "
							f"spawn kit) and fights its way to the target from there."
						),
					)
				)
		elif declared != 0:
			# v7fix3 P6: the legal non-relay deep spawn — a BREADTH ecology level, no siege
			# tags, within the unlocked frontier. Everything else is rejected with teaching.
			frontier = max(1, int(breadth_frontier or 1))
			is_breadth_lane = (
				meta_type == "BREADTH" and not siege_wall and not drill_target
			)
			if is_breadth_lane and declared <= frontier:
				pass  # in-frontier breadth spawn: legal by design (P6)
			elif is_breadth_lane:
				violations.append(
					Violation(
						rule=RULE_SPAWN,
						message=(
							f"Your BREADTH level declares spawn_floor={declared} but the breadth "
							f"spawn frontier currently ends at floor {frontier} (a deeper floor "
							f"unlocks once a floor-{frontier} breadth level is actually being "
							f"won in training). Set spawn_floor to at most {frontier}."
						),
					)
				)
			else:
				violations.append(
					Violation(
						rule=RULE_SPAWN,
						message=(
							f"Your level declares spawn_floor={declared} but a deep spawn is only "
							"legal in two lanes: (a) a level serving a SPAWN-ANNEAL RELAY wall, "
							"at exactly the rung floor the SIEGE DIRECTIVE states; or (b) a "
							f"BREADTH ecology level with NO siege_wall/drill_target tag, at most "
							f"the current breadth frontier (floor {frontier}). Deep spawn is "
							"never a free scaffold for a siege target — set spawn_floor to 0, or "
							"reshape the level into one of the two legal lanes."
						),
					)
				)

	# v7fix3 P6: with no active foci only the always-on spawn contract applies — every other rule
	# is a siege-shape rule and keeps its original "no focus, no ruling" semantics.
	if not foci:
		return violations

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
