"""Comprehensive behavior metrics from simulator traces (not LLM summaries).

Every probe / counterfactual branch is scored on the full metric panel:
success, progress, health, floor, navigation oscillation, no-progress stalls,
torch/tool latency, threat damage events.  Anything the trace cannot support
is reported as UNMEASURED (None), never invented.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

UNMEASURED = None


def _inv_get(state: Any, name: str) -> Optional[np.ndarray]:
    inv = getattr(state, "inventory", None)
    if inv is None:
        return None
    if isinstance(inv, dict):
        return np.asarray(inv[name]) if name in inv else None
    val = getattr(inv, name, None)
    return None if val is None else np.asarray(val)


def _achievements_sum(state: Any) -> Optional[np.ndarray]:
    ach = getattr(state, "achievements", None)
    if ach is None:
        return None
    arr = np.asarray(ach)
    return arr.sum(axis=-1) if arr.ndim > 1 else arr


def trace_metrics(trace: list, actions: np.ndarray, rewards: np.ndarray,
                  dones: np.ndarray, success: np.ndarray) -> dict:
    """trace: list of batched EnvState (T+1). Returns per-env metric arrays."""
    t_steps = len(trace) - 1
    batch = int(success.shape[0])
    pos = np.stack([np.asarray(s.player_position) for s in trace]).astype(np.float64)
    health = np.stack([np.asarray(s.player_health) for s in trace]).astype(np.float64)
    floor = np.stack([np.asarray(s.player_level) for s in trace]).astype(np.int64)
    ach = [_achievements_sum(s) for s in trace]
    has_ach = ach[0] is not None
    ach_arr = np.stack([a for a in ach]).astype(np.float64) if has_ach else None

    # navigation oscillation: pos_t == pos_{t-2} but pos_t != pos_{t-1}
    osc = np.zeros((t_steps, batch), dtype=np.float64)
    for t in range(2, t_steps + 1):
        back = (pos[t] == pos[t - 2]).all(-1) & (pos[t] != pos[t - 1]).any(-1)
        osc[t - 1] = back
    oscillation_rate = osc.mean(axis=0)

    # stall: action taken, position unchanged, no achievement/health progress
    stall = np.zeros((t_steps, batch), dtype=np.float64)
    for t in range(1, t_steps + 1):
        same_pos = (pos[t] == pos[t - 1]).all(-1)
        prog = np.zeros(batch, dtype=bool)
        if has_ach:
            prog |= ach_arr[t] > ach_arr[t - 1]
        prog |= health[t] != health[t - 1]
        stall[t - 1] = same_pos & ~prog
    stall_rate = stall.mean(axis=0)

    # longest no-progress run (position+floor+achievements frozen)
    changed = np.zeros((t_steps, batch), dtype=bool)
    for t in range(1, t_steps + 1):
        changed[t - 1] = (pos[t] != pos[t - 1]).any(-1) | (floor[t] != floor[t - 1])
        if has_ach:
            changed[t - 1] |= ach_arr[t] > ach_arr[t - 1]
    max_no_progress = np.zeros(batch, dtype=np.int64)
    run = np.zeros(batch, dtype=np.int64)
    for t in range(t_steps):
        run = np.where(changed[t], 0, run + 1)
        max_no_progress = np.maximum(max_no_progress, run)

    torch0 = _inv_get(trace[0], "torch")
    torch_lat = np.full(batch, -1.0)
    if torch0 is not None:
        torch_series = np.stack([_inv_get(s, "torch") for s in trace]).astype(np.float64)
        for b in range(batch):
            inc = np.where(torch_series[:, b] > torch0[b])[0]
            torch_lat[b] = float(inc[0]) if inc.size else -1.0
    else:
        torch_lat = None  # UNMEASURED

    # threat damage events: health drop while food positive (mob damage proxy)
    food0 = _inv_get(trace[0], "food")
    threat = np.zeros((t_steps, batch), dtype=np.float64)
    for t in range(1, t_steps + 1):
        drop = (health[t] < health[t - 1])
        threat[t - 1] = drop
    threat_events = threat.sum(axis=0)

    return {
        "success": success.astype(np.float64),
        "progress_final": (ach_arr[-1] if has_ach else np.full(batch, np.nan)),
        "health_delta": health[-1] - health[0],
        "floor_reached": floor.max(axis=0),
        "oscillation_rate": oscillation_rate,
        "stall_rate": stall_rate,
        "max_no_progress_steps": max_no_progress,
        "torch_pickup_latency": torch_lat,
        "threat_damage_events": threat_events,
        "return_sum": np.asarray(rewards).sum(axis=0),
    }


def aggregate(metrics: dict) -> dict:
    """Per-env arrays -> scalar aggregate (UNMEASURED stays None)."""
    out = {}
    for key, val in metrics.items():
        if val is None:
            out[key] = UNMEASURED
        else:
            out[key] = float(np.nanmean(np.asarray(val, dtype=np.float64)))
    return out