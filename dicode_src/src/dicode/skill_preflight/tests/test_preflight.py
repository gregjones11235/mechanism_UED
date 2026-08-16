"""Offline tests for the Preflight Gate pure logic.

No RL / no env needed — mock EnvState objects and aggregated numbers exercise
partial_progress_signals, infer_death_reason, route, and the staged funnel.

Run on the pod:
    cd /workspace/mechanism_UED/dicode_src
    uv run pytest src/dicode/skill_preflight/tests/test_preflight.py -v
"""
from __future__ import annotations

from types import SimpleNamespace

from dicode.skill_preflight.preflight import (
    partial_progress_signals,
    infer_death_reason,
    route,
    staged_preflight,
    Decision,
    PreflightResult,
)


# ---- mock EnvState / Inventory --------------------------------------------

def make_state(*, floor=0, health=9.0, food=9, drink=9, energy=9, timestep=50,
               n_ach=0, **inv):
    inventory = SimpleNamespace(
        wood=inv.get("wood", 0), stone=inv.get("stone", 0), coal=inv.get("coal", 0),
        iron=inv.get("iron", 0), diamond=inv.get("diamond", 0),
        pickaxe=inv.get("pickaxe", 0), sword=inv.get("sword", 0),
        bow=inv.get("bow", 0), arrows=inv.get("arrows", 0), torches=inv.get("torches", 0),
        ruby=inv.get("ruby", 0), sapphire=inv.get("sapphire", 0),
    )
    # a tiny stand-in "achievements" vector with n_ach ones
    ach = [1] * n_ach + [0] * (67 - n_ach)
    return SimpleNamespace(
        inventory=inventory, player_level=floor, player_health=health,
        player_food=food, player_drink=drink, player_energy=energy,
        timestep=timestep, achievements=_FakeVec(ach),
    )


class _FakeVec(list):
    def sum(self):
        return sum(self)


# ---- infer_death_reason ----------------------------------------------------

def test_death_reason_alive():
    assert infer_death_reason(5.0, 5, 5, 5) == "alive"

def test_death_reason_starved():
    assert infer_death_reason(0.0, 0, 5, 5) == "starved"

def test_death_reason_dehydrated():
    assert infer_death_reason(0.0, 5, 0, 5) == "dehydrated"

def test_death_reason_killed():
    assert infer_death_reason(0.0, 5, 5, 5) == "killed"


# ---- partial_progress_signals ----------------------------------------------

def test_signals_no_progress():
    s = partial_progress_signals(make_state())  # empty inventory, floor 0, no ach
    assert s["made_progress"] is False
    assert s["reached_depth"] is False
    assert s["has_basic_tools"] is False
    assert s["n_achievements"] == 0

def test_signals_has_tools():
    s = partial_progress_signals(make_state(pickaxe=1))
    assert s["has_basic_tools"] is True
    assert s["made_progress"] is True

def test_signals_reached_depth():
    s = partial_progress_signals(make_state(floor=2))
    assert s["reached_depth"] is True
    assert s["made_progress"] is True

def test_signals_got_ores():
    s = partial_progress_signals(make_state(iron=3))
    assert s["got_ores"] is True
    assert s["made_progress"] is True

def test_signals_achievements_counted():
    s = partial_progress_signals(make_state(n_ach=4))
    assert s["n_achievements"] == 4
    assert s["made_progress"] is True

def test_signals_inventory_extracted():
    s = partial_progress_signals(make_state(wood=5, stone=3))
    assert s["inventory"]["wood"] == 5
    assert s["inventory"]["stone"] == 3


# ---- route -----------------------------------------------------------------

def test_route_too_easy():
    d = route(sr=0.95, any_partial_progress=True)
    assert d.action == "reject" and d.reason == "too_easy"

def test_route_learnable_zone():
    d = route(sr=0.30, any_partial_progress=True)
    assert d.action == "accept" and d.reason == "learnable_zone"

def test_route_frontier():
    # SR ~ 0 but the student is making partial progress -> keep it (frontier)
    d = route(sr=0.0, any_partial_progress=True)
    assert d.action == "accept" and d.reason == "frontier"

def test_route_too_hard():
    # SR ~ 0 AND no progress at all -> reject
    d = route(sr=0.0, any_partial_progress=False)
    assert d.action == "reject" and d.reason == "too_hard_no_progress"

def test_route_low_boundary_is_learnable():
    d = route(sr=0.05, any_partial_progress=False)
    assert d.action == "accept" and d.reason == "learnable_zone"


# ---- staged_preflight funnel ----------------------------------------------

def _acc(reason="learnable_zone"):
    return PreflightResult("accept", reason, sr=0.3, any_partial_progress=True, n_episodes=8)

def _rej(reason="too_hard_no_progress"):
    return PreflightResult("reject", reason, sr=0.0, any_partial_progress=False, n_episodes=8)

def test_funnel_static_fail_stops_at_L1():
    out = staged_preflight(
        ["bad"],
        static_check=lambda c: (False, "SyntaxError"),
        run_short=lambda c: _acc(), run_full=lambda c: _acc(),
    )
    assert out[0][2] == "L1" and out[0][1] is None

def test_funnel_dedup_stops_at_L1():
    out = staged_preflight(
        ["x", "x"],
        static_check=lambda c: (True, ""),
        run_short=lambda c: _acc(), run_full=lambda c: _acc(),
        dedup_key=lambda c: c,
    )
    assert out[1][2] == "L1" and out[1][3] == "duplicate"

def test_funnel_L2_reject_skips_L3():
    calls = {"full": 0}
    def full(c):
        calls["full"] += 1
        return _acc()
    out = staged_preflight(
        ["c"],
        static_check=lambda c: (True, ""),
        run_short=lambda c: _rej(), run_full=full,
    )
    assert out[0][2] == "L2" and calls["full"] == 0     # L3 never ran

def test_funnel_passes_to_L3():
    out = staged_preflight(
        ["c"],
        static_check=lambda c: (True, ""),
        run_short=lambda c: _acc(), run_full=lambda c: _acc("frontier"),
    )
    assert out[0][2] == "L3" and out[0][1].action == "accept"
