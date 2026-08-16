"""Preflight Gate (minimal core).

Before a freshly-generated candidate level enters the training archive, run the
CURRENT policy on it for a few episodes (no parameter updates) and decide whether
it is *learnable right now*. The judgement is grounded in the student's real
partial-progress signals — not a binary success rate and not the LLM's self-
assessment — so that we don't throw away high-value frontier levels the student
currently fails but could learn next.

This file has four pieces:

  1. partial_progress_signals(state)  -- PURE: extract physical sub-signals from a
     single episode's final EnvState (floor reached, tools/ores gathered, death
     reason, survival length, achievements unlocked). Offline-testable.
  2. route(sr, any_partial_progress)  -- PURE: accept / reject / hold decision.
     Offline-testable.
  3. cold_preflight(...)              -- INTEGRATION: run current policy on a
     candidate env via the existing make_evaluate, aggregate signals, route.
  4. staged_preflight(candidates,...) -- INTEGRATION: L1 static -> L2 short ->
     L3 full funnel (cheap-to-expensive), so most candidates are filtered by the
     cheap stages.

Pieces 1-2 are unit-tested in tests/test_preflight.py. Pieces 3-4 are wired
against the real rollout (make_evaluate) and are integration-tested on the pod
when hooked into run_dicode.py / evolution_efficient.py (Phase 3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


# --- key inventory items we treat as "prerequisite progress" ---
_BASIC_TOOLS = ("pickaxe", "sword")
_ORES = ("coal", "iron", "diamond", "ruby", "sapphire")
_TRACKED_INVENTORY = (
    "wood", "stone", "coal", "iron", "diamond", "ruby", "sapphire",
    "pickaxe", "sword", "bow", "arrows", "torches",
)


def _to_int(x: Any) -> int:
    """Cast a jnp/np scalar (or python number) to a plain int, robustly."""
    try:
        return int(x)
    except (TypeError, ValueError):
        try:
            return int(x.item())  # jnp/np scalar
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
    """Best-effort cause of episode end from the final needs.

    In Craftax, food/drink/energy at 0 drain health; health at 0 = dead.
    """
    if health > 0:
        return "alive"           # survived to timeout (or reached goal)
    if food <= 0:
        return "starved"
    if drink <= 0:
        return "dehydrated"
    if energy <= 0:
        return "exhausted"
    return "killed"              # health depleted by damage, needs were ok


def partial_progress_signals(state: Any) -> dict:
    """Extract physical sub-signals from a single episode's FINAL EnvState.

    Used to distinguish "too hard forever" from "SR~0 now but the student is
    clearly making progress toward it" (the learnable frontier). Duck-typed on
    the minicraftax EnvState fields so it can be unit-tested with a mock state.
    """
    inv = getattr(state, "inventory", None)
    inventory = {name: _to_int(getattr(inv, name, 0)) for name in _TRACKED_INVENTORY}

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
        n_achievements = _to_int(sum(int(bool(a)) for a in ach)) if ach is not None else 0

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
        # a single episode "shows progress" if it reached depth, built tools,
        # gathered ores, or unlocked any achievement.
        "made_progress": (floor > 0) or has_basic_tools or got_ores or (n_achievements > 0),
    }


@dataclass
class Decision:
    action: str   # "accept" | "reject" | "hold"
    reason: str


def route(
    sr: float,
    any_partial_progress: bool,
    *,
    learnable_low: float = 0.05,
    too_easy: float = 0.85,
) -> Decision:
    """Accept / reject / hold a candidate from its aggregated preflight signals.

    Args:
        sr: success rate on the target over the preflight episodes, in [0, 1].
        any_partial_progress: did ANY episode show partial progress
            (reached_depth / tools / ores / any achievement)?
        learnable_low: SR at/above which the level is in the learnable zone.
        too_easy: SR at/above which the level is already solved -> reject.

    Routing:
        sr >= too_easy                       -> reject  (too easy, no learning signal)
        learnable_low <= sr < too_easy       -> accept  (learnable zone)
        sr < learnable_low & partial progress-> accept  (frontier: fails now, reachable)
        sr < learnable_low & no progress     -> reject  (too hard: no signal at all)
    """
    if sr >= too_easy:
        return Decision("reject", "too_easy")
    if sr >= learnable_low:
        return Decision("accept", "learnable_zone")
    if any_partial_progress:
        return Decision("accept", "frontier")     # SR~0 now but student is reaching it
    return Decision("reject", "too_hard_no_progress")


@dataclass
class PreflightResult:
    action: str                       # accept / reject / hold
    reason: str
    sr: float
    any_partial_progress: bool
    n_episodes: int
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# INTEGRATION pieces (wired against the real rollout; tested on the pod in Phase 3)
# ---------------------------------------------------------------------------

def _target_hit_rate(eval_metrics: dict, target_achievements: Iterable[str]) -> float:
    """SR on the target achievements, from make_evaluate's per-achievement stats.

    make_evaluate / process_evaluation_metrics report per-achievement success as
    ``skill_<name>`` in 0..100. Average over the target achievements -> fraction.
    NOTE: verify the exact metric key format on the pod (skill_<name> vs <name>).
    """
    vals = []
    for name in target_achievements:
        for key in (f"skill_{name}", name, f"skill_{name.lower()}"):
            if key in eval_metrics:
                vals.append(_to_float(eval_metrics[key]) / 100.0)
                break
    if not vals:
        # fall back to overall mean_performance (0..100) as a coarse signal
        return _to_float(eval_metrics.get("mean_performance", 0.0)) / 100.0
    return sum(vals) / len(vals)


def cold_preflight(
    env: Any,
    env_params: Any,
    train_state: Any,
    rng: Any,
    config: Any,
    target_achievements: Iterable[str],
    *,
    learnable_low: float = 0.05,
    too_easy: float = 0.85,
) -> PreflightResult:
    """Run the CURRENT policy on a candidate env (no updates) and decide.

    Reuses the existing evaluation rollout (make_evaluate) to run the frozen
    policy on the candidate env and get success/return metrics, then routes.

    INTEGRATION: imported lazily so this module stays importable (and unit-
    testable) without JAX / the env stack. Wire + test on the pod in Phase 3.
    """
    from dicode.craftax_evaluation import make_evaluate  # lazy: heavy JAX import

    import jax
    evaluate = jax.jit(make_evaluate(config, env, env_params))
    metrics = evaluate(train_state, rng)

    sr = _target_hit_rate(metrics, target_achievements)
    mean_return = _to_float(metrics.get("mean_return", 0.0))
    # Coarse partial-progress proxy from aggregate metrics: any positive return, or
    # any non-target achievement unlocked, means the policy is doing *something*.
    any_partial_progress = mean_return > 0.0 or any(
        k.startswith("skill_") and _to_float(v) > 0.0 for k, v in metrics.items()
    )

    d = route(sr, any_partial_progress, learnable_low=learnable_low, too_easy=too_easy)
    return PreflightResult(
        action=d.action, reason=d.reason, sr=sr,
        any_partial_progress=any_partial_progress, n_episodes=-1,
        extra={"mean_return": mean_return},
    )


def staged_preflight(
    candidates: list,
    *,
    static_check,                       # callable(candidate) -> (ok: bool, err: str)
    run_short,                          # callable(candidate) -> PreflightResult  (L2, few episodes)
    run_full,                           # callable(candidate) -> PreflightResult  (L3, more episodes)
    dedup_key=lambda c: c,              # callable(candidate) -> hashable
) -> list:
    """Cheap-to-expensive funnel: L1 static -> L2 short rollout -> L3 full rollout.

    Each stage discards candidates so the expensive stages evaluate few. Returns
    a list of (candidate, PreflightResult|None, stage, decision_reason). Keeping
    the stage callables injected makes the funnel unit-testable with fakes and
    decouples it from the concrete env/policy wiring (supplied at hook time).
    """
    results = []
    seen = set()
    for cand in candidates:
        # L1: static (compile / dedup)
        ok, err = static_check(cand)
        if not ok:
            results.append((cand, None, "L1", f"static_fail:{err}"))
            continue
        key = dedup_key(cand)
        if key in seen:
            results.append((cand, None, "L1", "duplicate"))
            continue
        seen.add(key)

        # L2: very short rollout, coarse filter
        r2 = run_short(cand)
        if r2.action == "reject":
            results.append((cand, r2, "L2", r2.reason))
            continue

        # L3: full rollout, final decision
        r3 = run_full(cand)
        results.append((cand, r3, "L3", r3.reason))
    return results
