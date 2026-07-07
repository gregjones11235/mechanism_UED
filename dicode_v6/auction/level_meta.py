"""v6fix7 P0.1 — machine-readable level metadata (<level_meta> block).

Why this exists: the proposer's chosen level TYPE (DEPTH / BREADTH / CONSOLIDATE) used to live only
inside <reasoning> free text, so no code could enforce any TYPE-dependent rule — the isolation-drill
shape check (superset waiver), the "never gift the siege focus" gate, and the siege quota in training
sampling all need code to KNOW what kind of level a candidate is. This module defines:

  - the spec text injected into the proposer user prompt via the {LEVEL_META_SPEC} placeholder
    (rendered ONLY when siege is on — when siege is off the placeholder is empty, the proposer is
    never asked for the block, and the parser tolerates its absence, so the v5y / baseline prompt
    and output shape are unchanged);
  - a tolerant parser for the block;
  - completeness validation (used by the retry loop: a siege-session response without a usable
    block is treated as a parse failure and re-queried).

Pure python, no jax/torch imports — offline-testable like the other auction modules.
"""

from __future__ import annotations

import json
import re

LEVEL_TYPES = ("DEPTH", "BREADTH", "CONSOLIDATE")

_BLOCK_RE = re.compile(r"<level_meta>\s*(\{.*?\})\s*</level_meta>", re.DOTALL)

# Injected via {LEVEL_META_SPEC}. Kept terse: one block, strict JSON, three keys.
LEVEL_META_SPEC_TEXT = """
★ MACHINE-READABLE LEVEL META (REQUIRED this session): immediately AFTER </reasoning> and BEFORE
<docstring>, output exactly one block:
<level_meta>{"type": "DEPTH", "drill_target": null, "siege_wall": null}</level_meta>
- "type": your chosen level TYPE — "DEPTH", "BREADTH" or "CONSOLIDATE". MUST match the TYPE you
  declared in reasoning point 0.
- "drill_target": ONLY for CONSOLIDATE — the single skill this isolation drill repeats (lowercase,
  e.g. "make_iron_pickaxe"); null for DEPTH/BREADTH.
- "siege_wall": the siege wall this level builds toward, if it targets a wall named in the SIEGE
  DIRECTIVE (lowercase skill name); null if the level is unrelated to the siege.
Strict JSON, exactly these three keys, no comments, no trailing text inside the block.
"""


def render_level_meta_spec(siege_active: bool = True) -> str:
	"""The {LEVEL_META_SPEC} placeholder value. Empty string when siege is off (baseline prompt
	byte-unchanged)."""
	return LEVEL_META_SPEC_TEXT if siege_active else ""


def _clean_skill(value) -> str | None:
	"""Normalise a skill-name field: lowercase snake, or None for null-ish values."""
	if value is None:
		return None
	if not isinstance(value, str):
		return None
	v = value.strip().strip("\"'").lower()
	if v in ("", "null", "none", "n/a", "-"):
		return None
	return v.replace(" ", "_")


def parse_level_meta(response_content: str | None) -> dict | None:
	"""Extract and normalise the <level_meta> block from a raw LLM response.

	Tolerant on purpose (this runs on every response, siege on or off):
	  - block absent -> None (siege-off responses never have one);
	  - malformed JSON -> one lenient repair pass (single->double quotes, python None/True/False),
	    then None if still unparseable;
	  - unknown "type" -> None (a block we cannot trust is no block);
	  - extra keys ignored; missing keys default to None.
	"""
	if not response_content:
		return None
	m = _BLOCK_RE.search(response_content)
	if not m:
		return None
	raw = m.group(1)
	data = None
	try:
		data = json.loads(raw)
	except ValueError:
		# Lenient repair: LLMs occasionally emit python-ish literals.
		repaired = (
			raw.replace("'", '"')
			.replace(": None", ": null")
			.replace(": True", ": true")
			.replace(": False", ": false")
		)
		try:
			data = json.loads(repaired)
		except ValueError:
			return None
	if not isinstance(data, dict):
		return None
	raw_type = data.get("type")
	if not isinstance(raw_type, str) or raw_type.strip().upper() not in LEVEL_TYPES:
		return None
	return {
		"type": raw_type.strip().upper(),
		"drill_target": _clean_skill(data.get("drill_target")),
		"siege_wall": _clean_skill(data.get("siege_wall")),
	}


def level_meta_complete(meta: dict | None) -> bool:
	"""Is this metadata usable for TYPE-dependent enforcement?

	CONSOLIDATE without a drill_target is unusable (the drill-shape rules cannot anchor), so it
	counts as incomplete -> the retry loop re-queries the proposer.
	"""
	if not isinstance(meta, dict):
		return False
	if meta.get("type") not in LEVEL_TYPES:
		return False
	if meta["type"] == "CONSOLIDATE" and not meta.get("drill_target"):
		return False
	return True
