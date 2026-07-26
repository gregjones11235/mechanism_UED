"""v7fix4.8 guardrail layer — session-level training-health guard (pure python, no jax).

Companion to the value-target clip in ppo_tr.py. The clip removes the mathematical
possibility of the bootstrap runaway that killed the fast arm at banner s191 (value_loss
0.4 -> 1e6+, entropy -> 0.001, held-out pinned at -0.9); this guard bounds the blast
radius of ANY residual instability mode to a single session: a tripped session's weights,
archive updates, notebook/chain/profile accumulation and checkpoint are all discarded.

Calibration (2026-07-14, wandb train/* per update across arms —
see fable_research_reports/v7fix4.8value护栏与必死关准入制方案.md §1):
- healthy sessions:            v_loss mean 0.26-0.65 (spikes <= 2.3), entropy min >= 0.28
- SELF-RECOVERING flare-ups (must NOT trip — they recover on their own and tripping
  would discard real progress): fast arm 4 flares, v_loss mean 42-86, entropy >= 0.28
  (banner s162/167/170/172); BASELINE 2 flares, v_loss mean 171/196, entropy >= 0.54
  (2/145 blocks) — flares are a general property of this stack, not a deep-level artifact.
- the fatal runaway:           first block v_loss mean 13976, entropy min 0.16, unrecoverable.
Thresholds sit between the regimes with margins on both sides:
  v_loss mean > 1000 (5x the worst benign flare seen anywhere, 14x under the fatal block)
  OR entropy min < 0.15 (flares keep >= 0.28; the fatal block hit 0.16 on its way to 0.001).

The held-out check is DROP-based, not absolute, so a from-scratch run (collect_wood
legitimately < 50% in its first sessions) never trips: it requires a previously healthy
baseline before a crash-level reading counts as a trip.

Pure python on purpose: unit-testable locally without jax (spawn_kit.py precedent).
"""

from __future__ import annotations

import math

# --- module state (single-threaded training loop; reset per session) ---
_stats: list[tuple[float, float]] = []  # (value_loss, entropy) per update, fed by ppo_tr
_prev_heldout: dict = {"collect_wood": None, "mean_return": None}
_consecutive_reverts: int = 0


# ---------------------------------------------------------------- stats feed
def reset_session_stats() -> None:
    """Called by ppo_tr.run_training_session before each session's train loop."""
    _stats.clear()


def record_update(value_loss: float, entropy: float) -> None:
    """Called from ppo_tr's per-update wandb callback. Tolerates nan/inf (judged later)."""
    try:
        _stats.append((float(value_loss), float(entropy)))
    except (TypeError, ValueError):
        _stats.append((float("nan"), float("nan")))


def session_stats_snapshot() -> list[tuple[float, float]]:
    return list(_stats)


# ---------------------------------------------------------------- verdicts
def session_verdict(
    vloss_max_mean: float = 1000.0,
    entropy_min: float = 0.15,
) -> str | None:
    """None = healthy. A string = human-readable reason to revert the session."""
    if not _stats:
        return None  # nothing recorded (e.g. zero-update session) — nothing to judge
    v_losses = [v for v, _ in _stats]
    entropies = [e for _, e in _stats]
    if any(math.isnan(v) or math.isinf(v) for v in v_losses):
        return "nan/inf in train value_loss"
    v_mean = sum(v_losses) / len(v_losses)
    if v_mean > vloss_max_mean:
        return (
            f"session value_loss mean {v_mean:.1f} > {vloss_max_mean:.0f} "
            f"(benign self-recovering flares stay <= 196; runaway starts ~1e4)"
        )
    e_min = min(entropies)
    if e_min < entropy_min:
        return (
            f"session entropy min {e_min:.4f} < {entropy_min} "
            f"(flares keep >= 0.28; collapse goes 0.16 -> 0.001)"
        )
    return None


def heldout_verdict(
    evaluation_metrics: dict | None,
    collect_wood_floor: float = 50.0,
    collect_wood_prev_min: float = 60.0,
    mean_return_floor: float = 5.0,
    mean_return_prev_min: float = 20.0,
) -> str | None:
    """Drop-based held-out red line. Trips only if a PREVIOUSLY healthy reading crashes.

    fast-arm collapse: collect_wood 100 -> 32 -> 0, mean_return 46 -> 1.7 -> -0.9 (trips).
    from-scratch run: prev baseline is None until healthy, so early low readings never trip.
    """
    if not evaluation_metrics:
        return None
    cw = evaluation_metrics.get("skill_collect_wood")
    mr = evaluation_metrics.get("mean_return")
    prev_cw, prev_mr = _prev_heldout["collect_wood"], _prev_heldout["mean_return"]
    if cw is not None and prev_cw is not None and prev_cw >= collect_wood_prev_min:
        if cw < collect_wood_floor:
            return f"held-out collect_wood crashed {prev_cw:.1f} -> {cw:.1f}"
    if mr is not None and prev_mr is not None and prev_mr >= mean_return_prev_min:
        if mr < mean_return_floor:
            return f"held-out mean_return crashed {prev_mr:.2f} -> {mr:.2f}"
    return None


def note_heldout(evaluation_metrics: dict | None) -> None:
    """Record the accepted (non-reverted) session's held-out readings as the new baseline."""
    if not evaluation_metrics:
        return
    cw = evaluation_metrics.get("skill_collect_wood")
    mr = evaluation_metrics.get("mean_return")
    if cw is not None:
        _prev_heldout["collect_wood"] = float(cw)
    if mr is not None:
        _prev_heldout["mean_return"] = float(mr)


def register_verdict(tripped: bool) -> int:
    """Track consecutive reverts. Returns the current consecutive-revert count."""
    global _consecutive_reverts
    _consecutive_reverts = _consecutive_reverts + 1 if tripped else 0
    return _consecutive_reverts
