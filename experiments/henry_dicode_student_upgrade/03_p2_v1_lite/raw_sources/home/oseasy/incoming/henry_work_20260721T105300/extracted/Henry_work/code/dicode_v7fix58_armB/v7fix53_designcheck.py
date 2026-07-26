"""v7fix5.3 designcheck — descent-regime scaffold (uplock + needs-clock anneal + radius restart).

Root cause pinned (death-forensics probe 2026-07-16, armA ckpt-15500, jobs 4031672/4046511 —
per-episode PHYSICAL telemetry): the stage3->stage2 collapse was never "gnome exposure en
route" — 87% of stage-2 deaths happen ABOVE the rung floor (flee up the entry ladder into the
uncleared 3x-spawn floor, die of sleep-kill 40% / thirst 32%), while winners cross in ~26 steps
of a ~112-step speedrun. Paired what-if (512 shared worlds, zero training): base 14.1% /
needs0.3x 19.1% / uplock 21.3% / BOTH 25.0%; floor-1 pre-credit under the lock ≡ bit-identical
(unreachable) — not shipped. Wiring-level + functional assertions (no jax needed).
Run: python v7fix53_designcheck.py -> prints N/N PASS.
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "src"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def _read(*parts):
    with open(os.path.join(HERE, *parts), encoding="utf-8") as f:
        return f.read()


SNB = _read("auction", "siege_notebook.py")
GM = _read("src", "dicode", "dreaming", "gen_manager.py")
GMECH = _read("src", "minicraftax", "game_mechanics.py")
STATE = _read("src", "minicraftax", "craftax_state.py")
T53 = _read("auction", "tests", "test_siege_fix53_descent_regime.py")
T46 = _read("auction", "tests", "test_siege_fix46_cliff_split.py")
T47 = _read("auction", "tests", "test_siege_fix47_r0_scaffold_defend.py")

# ---- functional probes (pure-python notebook; mirrors the fix53 test helpers) ---------------------
from auction.siege_notebook import (  # noqa: E402
    MATURITY_MIN_MASTERED,
    MATURITY_MIN_SNAPSHOTS,
    MATURITY_SKILL_SR,
    RUNG_GRADUATE_SR,
    RUNG_LADDER_RADII,
    RUNG_NEEDS_MID,
    RUNG_NEEDS_SLOW,
    SiegeNotebook,
    SiegeThresholds,
)

_HI = RUNG_GRADUATE_SR + 5
_MAXN = 5 + len(RUNG_LADDER_RADII)


def _mk_nb(tmp, **kw):
    th = SiegeThresholds(**kw) if kw else None
    nb = SiegeNotebook(os.path.join(tmp, f"nb_{len(os.listdir(tmp))}.json"),
                       **({"thresholds": th} if th else {}))
    prof = {f"filler_{i}": MATURITY_SKILL_SR + 10 for i in range(MATURITY_MIN_MASTERED)}
    prof["defeat_kobold"] = 0.0
    nb.apply_llm_update(
        1, prof,
        {"foci": [{"skill": "defeat_kobold", "prereq_tree": [], "relay_r0_floor": 3}]},
        num_snapshots=MATURITY_MIN_SNAPSHOTS, forensics={},
    )
    return nb


def _table(nb, floor=2):
    r = nb.foci()[0]["relay"]
    r["spawn_floor"] = floor
    out = {}
    for st in range(1, 10):
        r["sub_stage"] = st
        sc = nb.relay_scaffold("defeat_kobold")
        out[st] = (
            sc["sub_stage"], sc["down_ladder_radius"], sc["monster_credit"],
            bool(sc.get("uplock")), float(sc.get("needs_multiplier", 1.0)),
        )
    return out


with tempfile.TemporaryDirectory() as _tmp:
    _nb_new = _mk_nb(_tmp)
    _tab_new = _table(_nb_new)
    _nb_old = _mk_nb(_tmp, rung_descent_regime=False)
    _tab_old = _table(_nb_old)
    # R0 graduate skip (new regime): cliff-split at R0, climb the radius leg, graduate 6 -> 0.
    # v7fix5.7-P2': judgments ride the last-3 window mean — 3 window-judged HI per stage.
    _nb_r0 = _mk_nb(_tmp)
    _nb_r0.note_rung_reading("defeat_kobold", 0.0, session_idx=3)
    _r0_split = _nb_r0.note_rung_reading("defeat_kobold", 0.0, session_idx=5)
    _r0_grad = ""
    _s53 = 7
    for _stage53 in (7, 6, 0):
        for _k53 in range(4):
            _r0_grad = _nb_r0.note_rung_reading("defeat_kobold", _HI, session_idx=_s53)
            _s53 += 2
            if "RUNG_SUBSTAGE_GRADUATED" in _r0_grad:
                break
    _r0_stage = _nb_r0.relay_sub_stage("defeat_kobold")

_r = list(RUNG_LADDER_RADII)

CHECKS = [
    # --- L.1-L.3: the stage table itself ---
    ("L.1 new 9-stage table: 8/7/6 = the old radius leg verbatim; 5 = r[-1]+LOCK+slow; "
     "4 = entry+LOCK+slow (the what-if D condition); 3 = entry+LOCK+mid; 2/1 = the pre-5.3 "
     "entry/half-credit stages verbatim",
     _tab_new[8] == (8, _r[0], 8, False, 1.0)
     and _tab_new[7] == (7, _r[1], 8, False, 1.0)
     and _tab_new[6] == (6, _r[2], 8, False, 1.0)
     and _tab_new[5] == (5, _r[2], 8, True, RUNG_NEEDS_SLOW)
     and _tab_new[4] == (4, None, 8, True, RUNG_NEEDS_SLOW)
     and _tab_new[3] == (3, None, 8, True, RUNG_NEEDS_MID)
     and _tab_new[2] == (2, None, 8, False, 1.0)
     and _tab_new[1] == (1, None, 4, False, 1.0)),
    ("L.2 the probe lesson is STRUCTURAL: every locked stage ships a slowed clock "
     "(uplock alone relocated wander-sleep-death to the rung floor, 55% sleeping)",
     all(needs < 1.0 for (_s, _rad, _c, lock, needs) in _tab_new.values() if lock)
     and RUNG_NEEDS_SLOW < RUNG_NEEDS_MID < 1.0),
    ("L.3 ablation renders the exact pre-5.3 table (uplock never, clocks never) and the "
     "fix46/fix47 suites are pinned on it",
     _tab_old[5] == (5, _r[0], 8, False, 1.0)
     and _tab_old[4] == (4, _r[1], 8, False, 1.0)
     and _tab_old[3] == (3, _r[2], 8, False, 1.0)
     and _tab_old[2] == (2, None, 8, False, 1.0)
     and _tab_old[1] == (1, None, 4, False, 1.0)
     and _tab_old[9][0] == 5  # over-max clamps to the OLD max under the ablation
     and 'kw.setdefault("rung_descent_regime", False)' in T46
     and 'kw.setdefault("rung_descent_regime", False)' in T47),
    # --- L.4-L.5: state-machine wiring ---
    ("L.4 R0 skips the descent-leg stages in BOTH directions (graduate 6 -> 0 skipping 5..1; "
     "regress FULL -> the radius leg), and the skip set widens with the regime switch",
     "RUNG_CLIFF_SPLIT" in _r0_split and _r0_stage == 0
     and "descent-leg stages 5..1" in _r0_grad
     # v7fix5.4 moved the shape formula into the single-source _ladder_shape helper — the
     # fix5.3 shape is its fallback branch, and note_rung_reading reads the skip set from it.
     and "return (5 if _regime else 2) + len(static_radii), (5 if _regime else 2), static_radii, False" in SNB
     and "1 <= new_stage <= _r0_skip_hi" in SNB
     and "new_stage = _r0_skip_hi + 1" in SNB),
    ("L.5 cliff-split enters the regime's easiest stage (max_stage rides the switch)",
     "max_stage, _r0_skip_hi, _ladder_radii54, _quant54 = self._ladder_shape(r)" in SNB
     and "_transition(\"rung_cliff_split\", _floor_now, max_stage)" in SNB
     and f"stage {_MAXN}" in _r0_split),
    # --- L.6-L.8: rendering (gen_manager) ---
    ("L.6 template: TaskParams({task_params_args}) + {build_tail} + guarded ItemType import; "
     "the UPLOCK block removes the up-ladder ITEM via ladders_up + ItemType.NONE",
     "return TaskParams({task_params_args})" in GM
     and "{credit_line}{kit_line}{build_tail}" in GM
     and "import Achievement{constants_import}" in GM
     and "builder.ladders_up[{floor}]" in GM and "ItemType.NONE.value" in GM
     and '_RELAY_BUILD_PLAIN = "        return builder.build(rng)"' in GM),
    ("L.7 knob consumption is guarded: needs only when < 1.0, uplock only on floors >= 1, "
     "and the docstring/stage line reports both to the level's readers",
     "if _needs < 1.0:" in GM
     and 'task_params_args = f"needs_depletion_multiplier={_needs}"' in GM
     and 'bool(scaffold.get("uplock")) and int(floor) >= 1' in GM
     and "up-ladder REMOVED (committed descent — no retreat)" in GM
     and "survival clocks at {_needs:.1f}x" in GM),
    ("L.8 graphml forensic attrs are default-skipped (spawn_uplock only when True, "
     "spawn_needs_multiplier only when < 1.0 — attribute-set parity for non-regime levels)",
     'if meta.get("spawn_uplock"):' in GM
     and 'node["spawn_uplock"] = True' in GM
     and "float(_nm53) < 1.0" in GM),
    # --- L.9: prompts report world-rule FACTS only — never grant authority, never leak ---
    ("L.9 modeler render: DESCENT REGIME clause = facts only ('code-set, not yours to change'; "
     "NO tactic dictation, NO researcher-probe numbers — knowledge-leak boundary, user "
     "2026-07-16); proposer directive mirrors it compactly",
     "DESCENT REGIME at this stage (code-set, not yours to " in SNB
     and "upward is impossible in this world" in SNB
     and "COMMITTED DESCENT" not in SNB
     and "probe-verified: winners cross" not in SNB
     and "water/sleep survival loops" not in SNB
     and "[descent regime: up-ladder removed, survival clocks at " in GM),
    # --- L.10: the engine primitives the design leans on actually exist ---
    ("L.10 engine: ASCEND requires standing on LADDER_UP (item removal = lock), the needs "
     "multiplier scales hunger/thirst/fatigue, and clearing credit already drops spawn 3x->1x "
     "(why no monster-density dial ships)",
     "ItemType.LADDER_UP.value" in GMECH
     and "hunger_add *= task.needs_depletion_multiplier" in GMECH
     and "thirst_add *= task.needs_depletion_multiplier" in GMECH
     and "task.needs_depletion_multiplier" in GMECH
     and "monsters_killed[state.player_level] < task.monsters_killed_to_clear_level" in GMECH
     and "needs_depletion_multiplier" in STATE),
    # --- L.11: config + tests shipped ---
    ("L.11 config keys registered (siege_rung_descent_regime / needs_slow / needs_mid) and the "
     "fix53 suite pins table/transitions/rendering/prompt",
     'g("siege_rung_descent_regime", RUNG_DESCENT_REGIME)' in SNB
     and 'g("siege_rung_needs_slow", RUNG_NEEDS_SLOW)' in SNB
     and 'g("siege_rung_needs_mid", RUNG_NEEDS_MID)' in SNB
     and "def test_stage_table_new_regime" in T53
     and "def test_full_climb_passes_through_the_regime_stages" in T53
     and "def test_r0_graduate_skips_descent_leg_stages" in T53
     and "def test_system_relay_levels_render_regime_knobs" in T53
     and "def test_modeler_render_reports_regime_without_granting_authority" in T53),
]

n_pass = 0
for name, ok in CHECKS:
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    n_pass += bool(ok)
print(f"\n{n_pass}/{len(CHECKS)} fix5.3 design points hold")
sys.exit(0 if n_pass == len(CHECKS) else 1)
