"""Selectively-reused PURE core of skill-preflight-ued_Mason.

PROVENANCE: adapted (selectively, NOT a branch merge) from
``origin/skill-preflight-ued_Mason`` -> ``dicode_src/src/dicode/skill_preflight``:

  * ``preflight.py``     -> route / Decision / PreflightResult /
                            staged_preflight / infer_death_reason /
                            partial_progress_signals (all PURE, offline-testable);
  * ``prereq_graph.py``  -> a small self-contained DIRECT_PREREQS subset +
                            readiness test (the full 67-achievement graph needs
                            ``auction.craftax_achievements`` which is NOT a
                            dependency of this dry-run package).

Only the JAX-free pieces were carried over. The INTEGRATION piece
(``cold_preflight``, which lazy-imports ``make_evaluate`` + JAX) was NOT
copied; the real-Craftax seam in ``simulator_probe.CraftaxPreflightProbeRunner``
is the only place that may touch it, and it stays BLOCKED locally.

The routing semantics are unchanged so a later real-probe run reproduces the
same accept/reject decisions bit-for-bit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Mapping, Tuple


# ---------------------------------------------------------------------------
# partial-progress signals (from preflight.py — PURE; duck-typed on a
# Craftax-like final EnvState so it unit-tests with a mock state)
# ---------------------------------------------------------------------------

_BASIC_TOOLS = ("pickaxe", "sword")
_ORES = ("coal", "iron", "diamond", "ruby", "sapphire")
_TRACKED_INVENTORY = (
    "wood", "stone", "coal", "iron", "diamond", "ruby", "sapphire",
    "pickaxe", "sword", "bow", "arrows", "torches",
)


def _to_int(x: Any) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        try:
            return int(x.item())
        except Exception:
            return 0


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        try:
            return float(x.item())
        except Exception:
            return 0.0


def infer_death_reason(health: float, food: int, drink: int, energy: int) -> str:
    if health > 0:
        return "alive"
    if food <= 0:
        return "starved"
    if drink <= 0:
        return "dehydrated"
    if energy <= 0:
        return "exhausted"
    return "killed"


def partial_progress_signals(state: Any) -> dict:
    """Extract physical sub-signals from a single episode's FINAL state."""
    inv = getattr(state, "inventory", None)
    inventory = {name: _to_int(getattr(inv, name, 0))
                 for name in _TRACKED_INVENTORY}
    floor = _to_int(getattr(state, "player_level", 0))
    health = _to_float(getattr(state, "player_health", 0.0))
    food = _to_int(getattr(state, "player_food", 0))
    drink = _to_int(getattr(state, "player_drink", 0))
    energy = _to_int(getattr(state, "player_energy", 0))
    timestep = _to_int(getattr(state, "timestep", 0))
    ach = getattr(state, "achievements", None)
    try:
        n_achievements = _to_int(ach.sum()) if ach is not None else 0
    except Exception:
        n_achievements = (_to_int(sum(int(bool(a)) for a in ach))
                          if ach is not None else 0)
    has_basic_tools = any(inventory[t] >= 1 for t in _BASIC_TOOLS)
    got_ores = any(inventory[o] >= 1 for o in _ORES)
    return {
        "floor": floor,
        "reached_depth": floor > 0,
        "alive": health > 0,
        "death_reason": infer_death_reason(health, food, drink, energy),
        "timestep": timestep,
        "n_achievements": n_achievements,
        "inventory": inventory,
        "has_basic_tools": has_basic_tools,
        "got_ores": got_ores,
        "made_progress": (floor > 0) or has_basic_tools or got_ores
                         or (n_achievements > 0),
    }


# ---------------------------------------------------------------------------
# routing (from preflight.py — PURE)
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    action: str   # "accept" | "reject" | "hold"
    reason: str


def route(sr: float, any_partial_progress: bool, *,
          learnable_low: float = 0.05, too_easy: float = 0.85) -> Decision:
    """accept / reject a candidate from aggregated preflight signals.

    sr >= too_easy                         -> reject (too_easy)
    learnable_low <= sr < too_easy         -> accept (learnable_zone)
    sr < learnable_low & partial progress  -> accept (frontier)
    sr < learnable_low & no progress       -> reject (too_hard_no_progress)
    """
    if sr >= too_easy:
        return Decision("reject", "too_easy")
    if sr >= learnable_low:
        return Decision("accept", "learnable_zone")
    if any_partial_progress:
        return Decision("accept", "frontier")
    return Decision("reject", "too_hard_no_progress")


@dataclass
class PreflightResult:
    action: str                       # accept / reject / hold
    reason: str
    sr: float
    any_partial_progress: bool
    n_episodes: int
    extra: dict = field(default_factory=dict)


def staged_preflight(candidates: list, *,
                     static_check: Callable[[Any], Tuple[bool, str]],
                     run_short: Callable[[Any], PreflightResult],
                     run_full: Callable[[Any], PreflightResult],
                     dedup_key: Callable[[Any], Any] = lambda c: c) -> list:
    """Cheap-to-expensive funnel: L1 static -> L2 short -> L3 full."""
    results = []
    seen = set()
    for cand in candidates:
        ok, err = static_check(cand)
        if not ok:
            results.append((cand, None, "L1", f"static_fail:{err}"))
            continue
        key = dedup_key(cand)
        if key in seen:
            results.append((cand, None, "L1", "duplicate"))
            continue
        seen.add(key)
        r2 = run_short(cand)
        if r2.action == "reject":
            results.append((cand, r2, "L2", r2.reason))
            continue
        r3 = run_full(cand)
        results.append((cand, r3, "L3", r3.reason))
    return results


# ---------------------------------------------------------------------------
# self-contained prerequisite-graph subset (from prereq_graph.py — PURE DATA)
# ---------------------------------------------------------------------------
# A small, dependency-free slice of the DIRECT_PREREQS graph, enough to give
# the deterministic probe a principled "is the frontier target practicable"
# signal without importing auction.craftax_achievements. Edges are
# CONJUNCTIVE (every listed prerequisite is required).

DIRECT_PREREQS: dict = {
    "collect_wood": frozenset(),
    "place_table": frozenset({"collect_wood"}),
    "make_wood_pickaxe": frozenset({"collect_wood", "place_table"}),
    "collect_stone": frozenset({"make_wood_pickaxe"}),
    "place_furnace": frozenset({"collect_stone"}),
    "collect_coal": frozenset({"make_wood_pickaxe"}),
    "collect_iron": frozenset({"make_stone_pickaxe"}),
    "make_stone_pickaxe": frozenset({"collect_wood", "collect_stone",
                                     "place_table"}),
    "make_iron_pickaxe": frozenset({"collect_wood", "collect_stone",
                                    "collect_iron", "collect_coal",
                                    "place_table", "place_furnace"}),
}


def prereq_ready(skill: str, sr: Mapping[str, float], *,
                 prereq_threshold: float = 0.3) -> bool:
    """A skill is targetable iff EVERY direct prerequisite is individually at
    or above ``prereq_threshold`` (unlisted skills have no prerequisites)."""
    prereqs = DIRECT_PREREQS.get(skill, frozenset())
    return all(sr.get(p, 0.0) >= prereq_threshold for p in prereqs)


__all__ = [
    "Decision", "PreflightResult", "route", "staged_preflight",
    "partial_progress_signals", "infer_death_reason",
    "DIRECT_PREREQS", "prereq_ready",
]
