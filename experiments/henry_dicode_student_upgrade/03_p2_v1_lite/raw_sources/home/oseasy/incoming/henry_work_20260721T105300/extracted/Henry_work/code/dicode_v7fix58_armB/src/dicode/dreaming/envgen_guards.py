"""Pure-python guards for FM-generated level code (v7fix4.1, 2026-07-11).

Born from the v7fix4 A/B double-crash: the env_generator FM wrote a level whose world
doubled the hostile-mob array capacities (melee 3->6, ranged 2->4). check_compilation
validates each task SOLO, so the non-standard world was internally consistent and
passed; training then compiled ALL tasks into one ``jax.lax.switch``, whose branches
must have identical output shapes — the whole job died (baseline arm crashed with
"switch branches must have equal output types", method arm hung). The original
pipeline only type-checked inventory dtypes; array SHAPES were never checked anywhere.

This module is deliberately jax-free (stdlib only): the shape diff and the teaching
messages live here so tests can load it by file path without jax/craftax installed
(same pattern as minicraftax/spawn_kit.py). gen_manager flattens the jax shape
structs into plain ``{path: (shape, dtype)}`` dicts before calling in.
"""

import re

# Under JIT, numpy.random / stdlib random execute once at trace time and freeze into
# constants: every env.reset() would silently rebuild the identical world (the level's
# DISTRIBUTION collapses into a single instance). Only the jax `rng` argument of
# generate_world threads fresh randomness through each reset. stdlib `random` is
# unusable without an import, so banning the imports covers all its call sites.
_BANNED_RANDOMNESS_PATTERNS = [
	(re.compile(r"\bnp\s*\.\s*random\b"), "np.random"),
	(re.compile(r"\bnumpy\s*\.\s*random\b"), "numpy.random"),
	(re.compile(r"^\s*import\s+random\b", re.MULTILINE), "import random"),
	(re.compile(r"^\s*from\s+random\s+import\b", re.MULTILINE), "from random import ..."),
	(re.compile(r"^\s*from\s+numpy\s+import\s+[^#\n]*\brandom\b", re.MULTILINE), "from numpy import random"),
]


def scan_banned_randomness(code: str) -> str:
	"""Returns a teaching error message if code uses a banned randomness source, else ''."""
	for pattern, label in _BANNED_RANDOMNESS_PATTERNS:
		if pattern.search(code):
			return (
				f"banned randomness source: {label}. Level code must draw ALL randomness "
				"from the `rng` argument of generate_world (split it with jax.random.split "
				"as needed). numpy.random and Python's `random` are frozen into constants "
				"under JIT, so every reset would silently produce the identical world."
			)
	return ""


def diff_world_specs(expected: dict, got: dict) -> list[str]:
	"""Compares two ``{path: (shape, dtype)}`` maps; returns human-readable mismatch lines.

	Walks EVERY leaf of the canonical EnvState mechanically — coverage does not depend
	on any hand-written list of fields, so there is nothing for the generator to route
	around (missing and extra paths are violations too).
	"""
	lines = []
	for path, (e_shape, e_dtype) in expected.items():
		if path not in got:
			lines.append(
				f"{path}: MISSING from the generated world "
				f"(expected shape {e_shape} dtype {e_dtype})"
			)
			continue
		g_shape, g_dtype = got[path]
		if g_shape != e_shape or g_dtype != e_dtype:
			lines.append(
				f"{path}: got shape {g_shape} dtype {g_dtype}, "
				f"expected shape {e_shape} dtype {e_dtype}"
			)
	for path in got:
		if path not in expected:
			g_shape, g_dtype = got[path]
			lines.append(f"{path}: UNEXPECTED extra field (shape {g_shape} dtype {g_dtype})")
	return lines


_MAX_MISMATCH_LINES = 12


def shape_mismatch_message(mismatches: list[str]) -> str:
	"""Formats world-shape contract violations into a reflection-loop teaching message."""
	shown = mismatches[:_MAX_MISMATCH_LINES]
	listing = "\n".join(f"  - {m}" for m in shown)
	more = len(mismatches) - len(shown)
	if more > 0:
		listing += f"\n  - ... and {more} more mismatched field(s)"
	return (
		"world-shape contract violation: generate_world must return an EnvState whose "
		"EVERY array matches the standard world exactly. All tasks are compiled into ONE "
		"jax.lax.switch, whose branches must have identical output shapes/dtypes — a "
		"non-standard world crashes the whole training system.\n"
		"Mismatched fields:\n"
		f"{listing}\n"
		"Most common cause: constructing your own StaticEnvParams(...) or EnvParams(...), "
		"or resizing/concatenating state arrays. Use the static_params/params passed to "
		"__init__ unchanged, populate only EXISTING array slots, and express extra "
		"difficulty through TaskParams in get_task_params (melee_spawn_multiplier, "
		"mob_health_multiplier, ...) — never through bigger arrays."
	)
