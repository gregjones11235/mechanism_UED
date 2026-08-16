"""C-1 API-hallucination lint + repair evidence (v2 layer; proposal_C Stage-2 landing).

Turns hallucinated-API compile failures (the #1 discard source: 115+ drops in the 2e9 run,
3.4x the scaffold gate's) from silent drops into evidence-carrying repairs. Error-driven
design: instead of maintaining static whitelists per symbol, we parse the compile error,
resolve the offending object's REAL members dynamically (from the very modules the task
code imports — the wrapper's enums, NOT upstream craftax; the BlockType.PATH case proved
the two differ), and hand the generator a precise indictment + valid options.

Covers the three hallucination classes in the 2e9 ledger:
  H1 enum member       AttributeError: type object 'BlockType' has no attribute 'LADDER_DOWN'
  H2 builder method    AttributeError: 'WorldBuilder' object has no attribute 'add_mobs_randomly'
  H3 ctor kwarg        TypeError: Inventory.__init__() got an unexpected keyword argument 'furnace'

Anything not matching these patterns returns None -> caller falls through to the old
drop path (syntax errors, logic errors etc. are not our jurisdiction).
"""
from __future__ import annotations

import ast
import dataclasses
import difflib
import importlib
import re

# error patterns ---------------------------------------------------------------------------
# NOTE: check_compilation returns f"Compilation error: {str(e)}" — str(e) carries NO
# exception-class prefix. The 3e8 ablation ran with prefix-required patterns and C-1 was
# silently inert (hallucination-class 0/45 while the log held 28 AttributeErrors). All
# prefixes are therefore OPTIONAL; the phrasings are distinctive enough without them.
_RE_TYPE_ATTR = re.compile(
    r"(?:AttributeError:\s*)?type object '(\w+)' has no attribute '(\w+)'")
_RE_OBJ_ATTR = re.compile(
    r"(?:AttributeError:\s*)?'(\w+)' object has no attribute '(\w+)'")
_RE_CTOR_KW = re.compile(
    r"(?:TypeError:\s*)?(\w+)\.__init__\(\) got an unexpected keyword argument '(\w+)'")
_RE_NAME_UNDEF = re.compile(
    r"(?:NameError:\s*)?name '(\w+)' is not defined")

_COMMON_IMPORTS = {
    "jnp": "import jax.numpy as jnp",
    "np": "import numpy as np",
    "jax": "import jax",
}

# fallback module map for symbols whose import the AST scan can't resolve (e.g. injected
# into the exec namespace rather than imported in the task file itself)
_FALLBACK_MODULES = {
    "WorldBuilder": ("minicraftax.world_builder", "WorldBuilder"),
    "Inventory": ("craftax.craftax.craftax_state", "Inventory"),
    "Achievement": ("craftax.craftax.constants", "Achievement"),
    "BlockType": ("craftax.craftax.constants", "BlockType"),
}


def _imported_source(code: str, symbol: str):
    """Find `from X import symbol` in the task code -> module path, else None."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if (alias.asname or alias.name) == symbol:
                    return node.module
    return None


def _resolve_members(code: str, symbol: str) -> list[str] | None:
    """The REAL member list of `symbol`, resolved from the module the code imports."""
    module_path = _imported_source(code, symbol)
    candidates = []
    if module_path:
        candidates.append((module_path, symbol))
    if symbol in _FALLBACK_MODULES:
        candidates.append(_FALLBACK_MODULES[symbol])
    for mod_path, attr in candidates:
        try:
            mod = importlib.import_module(mod_path)
            obj = getattr(mod, attr)
        except Exception:
            continue
        # dataclass ctor kwargs
        if dataclasses.is_dataclass(obj):
            return [f.name for f in dataclasses.fields(obj)]
        # enum / plain class attrs & methods
        members = [m for m in dir(obj) if not m.startswith("_")]
        if members:
            return members
    return None


def _format_options(bad: str, members: list[str]) -> str:
    close = difflib.get_close_matches(bad, members, n=6, cutoff=0.4)
    listed = close if close else members[:12]
    extra = "" if len(members) <= 12 or close else f" (and {len(members) - 12} more)"
    return ", ".join(listed) + extra


def diagnose(code: str, error_msg: str) -> str | None:
    """Hallucination indictment for the repair prompt, or None if not our class of error."""
    if not error_msg:
        return None

    m = _RE_TYPE_ATTR.search(error_msg) or _RE_OBJ_ATTR.search(error_msg)
    if m:
        cls, bad = m.group(1), m.group(2)
        members = _resolve_members(code, cls)
        if not members:
            return None
        return (
            f"[api-lint] `{cls}.{bad}` DOES NOT EXIST — it is a hallucinated name "
            f"(often borrowed from other games' APIs). Valid `{cls}` options closest to "
            f"what you wrote: {_format_options(bad, members)}. Replace the hallucinated "
            f"reference with a real one that fits the task's intent; change nothing else."
        )

    m = _RE_NAME_UNDEF.search(error_msg)
    if m and "object" not in error_msg and "attribute" not in error_msg:
        bad = m.group(1)
        hint = _COMMON_IMPORTS.get(bad, "add the appropriate import at module top")
        return (
            f"[api-lint] name `{bad}` is undefined — a missing import (H4). Add the "
            f"import at the top of the file ({hint}); change nothing else."
        )

    m = _RE_CTOR_KW.search(error_msg)
    if m:
        cls, bad = m.group(1), m.group(2)
        members = _resolve_members(code, cls)
        if not members:
            return None
        return (
            f"[api-lint] `{cls}` has no field `{bad}` — hallucinated constructor/keyword "
            f"argument. Valid `{cls}` fields: {_format_options(bad, members)}. Note the "
            f"levelled-gear convention: swords/pickaxes are LEVEL fields (e.g. `sword: 1` "
            f"means a wood sword), not per-item names. Fix the offending argument; change "
            f"nothing else."
        )

    return None
