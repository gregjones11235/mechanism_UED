"""v7fix4 design checklist — habitat fidelity (source-level asserts, run at launch on Oscar).

Every check pins one line of the design (fable_research_reports/v7fix4真实世界接力与栖息地保真
方案.md + the 2026-07-11 revisions folded in during implementation: unconditional floor-3 deep
lock, r0 anchoring absorbed into code, C3 finding = the win ledger was ALWAYS held-out-sourced
so no ledger filtering is needed — the quarantine lives on the RUNG READING side instead).
Complements the pytest suite: pytest pins behaviour, this pins WIRING (the fix9 lesson —
features that exist but are not called).
"""

import inspect
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_n = 0


def ok(cond, label):
    global _n
    _n += 1
    if not cond:
        print(f"FAIL {_n:02d} {label}")
        sys.exit(1)
    print(f"PASS {_n:02d} {label}")


# ---- P1 substrate: the habitat map ---------------------------------------------------------------
from auction.craftax_achievements import (  # noqa: E402
    ALL_ACHIEVEMENTS,
    FLOOR_ENTRANCES,
    MAX_DUNGEON_FLOOR,
    WALL_NATIVE_FLOOR,
    native_floor_of,
)

ok(set(WALL_NATIVE_FLOOR) <= ALL_ACHIEVEMENTS, "P1.1 habitat map names are real achievements")
ok(native_floor_of("defeat_lizard") == 3 and native_floor_of("defeat_kobold") == 3,
   "P1.2 the v7fix3 walls are pinned to floor 3 (Sewers)")
ok(sorted(FLOOR_ENTRANCES) == list(range(1, MAX_DUNGEON_FLOOR + 1))
   and FLOOR_ENTRANCES[3] == "enter_sewers",
   "P1.3 entrance table covers floors 1..8, sewers pinned")
ok(all(ach in WALL_NATIVE_FLOOR for ach in ALL_ACHIEVEMENTS
       if ach.startswith(("defeat_", "enter_"))),
   "P1.4 every combat/entrance achievement has a habitat floor")

# ---- P1 gates in the notebook ---------------------------------------------------------------------
from auction.siege_notebook import (  # noqa: E402
    DEEP_WALL_RELAY_FLOOR,
    RELAY_KIT_STRIP,
    SANDBOX_MISMATCH_READINGS,
    WALL_FLOOR_ANCHOR,
    SiegeNotebook,
    SiegeThresholds,
)

ok(WALL_FLOOR_ANCHOR is True and DEEP_WALL_RELAY_FLOOR == 3,
   "P1.5 anchor on by default; deep lock line = floor 3 (gnome class stays free)")
src_rec = inspect.getsource(SiegeNotebook._reconcile_foci)
ok("deep_locked(" in src_rec, "P1.6 LLM ordinary proposal of a deep wall is deep_locked")
ok(src_rec.index("deep_locked(") < src_rec.index("chain_incomplete("),
   "P1.7 deep lock rules BEFORE ⑦ (relay is the only right answer for a deep wall)")
ok("native_floor_of(sl) >= self.th.deep_wall_relay_floor" in src_rec
   and src_rec.count("relay_refused(") >= 3,
   "P1.8 capacity fall-through re-checks the deep lock (the A1 lesson, habitat edition)")
src_auto = inspect.getsource(SiegeNotebook._auto_open_from_ranked)
ok("native_floor_of" in src_auto and "deep_wall_relay_floor" in src_auto,
   "P1.9 auto-open menu excludes deep walls (the v7fix3 hole: only the LLM path was locked)")
src_anchor = inspect.getsource(SiegeNotebook._anchored_r0)
ok("native_floor_of" in src_anchor and "enter_" in src_anchor,
   "P1.10 R0 anchored to habitat; enter_* walls one floor above")
ok("_anchored_r0" in inspect.getsource(SiegeNotebook._attach_relay),
   "P1.11 the attach path anchors too (the fix3 lizard attach was the actual bug site)")
src_autofill = inspect.getsource(SiegeNotebook._autofill_entrances)
ok("FLOOR_ENTRANCES" in src_autofill and "chain_autofilled" in src_autofill,
   "P1.12 entrance autofill exists and records its decision")
ok("_autofill_entrances" in inspect.getsource(SiegeNotebook._attach_prereq_trees),
   "P1.13 autofill wired into the per-session tree attach (stale chains repaired too)")

# ---- P3: KIT_STRIP -------------------------------------------------------------------------------
ok(RELAY_KIT_STRIP is True, "P3.1 kit-strip exam on by default")
src_rung = inspect.getsource(SiegeNotebook.note_rung_reading)
ok("KIT_STRIP" in src_rung and "rung_kit_strip" in src_rung,
   "P3.2 floor-0 graduation enters the kit-strip exam instead of sewing")
ok("kitless" in src_rung, "P3.3 SEWN message states the certificate semantics")
ok(src_rung.index("kit_strip") < src_rung.index("RUNG_REGRESSED"),
   "P3.4 kit-strip regress handled (back to floor 1 WITH kit)")
ok(callable(getattr(SiegeNotebook, "relay_kit_stripped", None)),
   "P3.5 kit-stage accessor for the level builder")

# ---- P4: sandbox_mismatch sentinel ----------------------------------------------------------------
ok(SANDBOX_MISMATCH_READINGS == 3, "P4.1 sentinel streak = 3 readings")
src_gap = inspect.getsource(SiegeNotebook.note_transfer_gap)
ok("sandbox_mismatch" in src_gap and "focus_retired_sandbox_mismatch" in src_gap,
   "P4.2 post-SEWN sentinel wired into the gap channel")
ok("focus_retired_sandbox_mismatch" in src_rec,
   "P4.3 a sandbox_mismatch retirement is NOT cooldown-waived for a relay re-proposal")
th = SiegeThresholds()
ok(th.wall_floor_anchor and th.relay_kit_strip and th.sandbox_mismatch_readings == 3
   and th.deep_wall_relay_floor == 3, "P4.4 thresholds carry all fix4 knobs")

# ---- P2: system-built relay levels ----------------------------------------------------------------
from dicode.dreaming.gen_manager import TaskArchive, TaskGenerator  # noqa: E402

ok(callable(getattr(TaskGenerator, "_system_relay_levels", None)),
   "P2.1 system relay level factory exists")
src_sys = inspect.getsource(TaskGenerator._system_relay_levels)
# v7fix5.5 batch-2: the kit read moved into the extracted _relay_level_build (one template,
# shared by the ladder and the probe executor; fix55 designcheck B2.7 pins the extraction as
# byte-identical). The P2.2 contract is unchanged — the factory's build CHAIN reads the rung's
# kit stage and the winner-median kit — it just spans the extraction now.
src_build = inspect.getsource(TaskGenerator._relay_level_build) \
    if hasattr(TaskGenerator, "_relay_level_build") else ""
ok("relay_kit_stripped" in src_sys and "_relay_level_build" in src_sys
   and "_relay_kit_dict" in src_build,
   "P2.2 the factory reads the rung's kit stage + the winner-median kit")
ok("system_built" in (src_sys + src_build) and "RELAY-BUILD" in src_sys,
   "P2.3 levels are flagged system_built and logged [siege][RELAY-BUILD]")
tpl = TaskGenerator._RELAY_LEVEL_CODE
ok("WorldBuilder" in tpl and "set_starting_floor" in tpl
   and "def generate_world(self, rng: jax.Array)" in tpl,
   "P2.4 template = the REAL world generator, fresh world per episode reset")
src_coop = inspect.getsource(TaskGenerator.evolve_mastered_coop)
ok("_system_relay_levels" in src_coop,
   "P2.5 factory wired into the coop session (after selection — no seat consumed)")
ok('"_system_code": parsed_data.get("_system_code")'
   in inspect.getsource(TaskGenerator._organize_data),
   "P2.6 shipped code rides through _organize_data")
ok("system_built" in inspect.getsource(TaskArchive.set_level_meta),
   "P2.7 the system_built flag persists onto the graphml node")
src_gaph = inspect.getsource(TaskGenerator._render_siege_gap_hint)
ok("_relay_sys_only" in src_gaph and "system_built" in src_gaph,
   "P2.8 rung readings accept ONLY system-built levels (telemetry quarantine)")
src_dir = inspect.getsource(TaskGenerator._render_siege_directive)
ok("SYSTEM-BUILT" in src_dir and "do NOT author" in src_dir,
   "P2.9 the directive tells proposer[0] the system builds these levels")
ok('siege_relay_worldgen' in src_sys and '!= "base"' in src_sys and "return []" in src_sys,
   "P2.10 the fm ablation knob short-circuits the factory (worldgen != base -> [])")

# ---- post-audit hardening (2026-07-11 review) ------------------------------------------------------
from auction.level_validator import RULE_SYS_RELAY, validate_level  # noqa: E402
import dicode.dreaming.gen_manager as _gm_mod  # noqa: E402

_gm_src = inspect.getsource(_gm_mod)
ok(RULE_SYS_RELAY == "R6_SYSTEM_RELAY",
   "H.1 hard rule exists: system-built relay walls take NO FM levels")
ok("system_relay_walls" in inspect.signature(validate_level).parameters,
   "H.2 validator accepts the system-relay wall set")
ok("system_relay_walls=system_relay_walls" in _gm_src,
   "H.3 gen_manager passes the wall set into every validate_level call")
ok("set(required_spawn_floors)" in _gm_src,
   "H.4 the wall set is gated on worldgen=base (fm ablation arm keeps plain R6)")
ok("RULE_SYS_RELAY for v in violations" in _gm_src and "DEMOTED" in _gm_src,
   "H.5 fallback DEMOTES a persistent violator (tag strip) instead of accepting the crowd-out")
ok("relay_kit_stage_" in _gm_src,
   "H.6 wandb kit-stage metric emitted (floor 0 alone cannot show the KIT_STRIP exam)")
ok("still-unmastered LINKS" in _gm_src and "do NOT tag levels for the wall itself" in _gm_src,
   "H.8 relay-wall chain header / tactic line rewritten (no 'build toward this wall' ambiguity)")
from auction.modeler import MODELER_SIEGE_SYSTEM_PROMPT as _P0  # noqa: E402

import re as _re  # noqa: E402

ok("prior run" not in _P0 and "40 sessions" not in _P0
   and not _re.search(r"wasted \d+ (siege )?decisions once", _P0)
   and not _re.search(r"cost a (whole )?(prior )?run", _P0),
   "H.7 no run-specific anecdotes in the modeler prompt (leakage hygiene, user review)")

# ---- P0: the modeler knows what the gates enforce --------------------------------------------------
from auction.modeler import MODELER_SIEGE_SYSTEM_PROMPT as P  # noqa: E402

ok("HABITAT MAP" in P and "enter_sewers" in P, "P0.1 habitat map rendered into the prompt")
ok("r0_corrected" in P and "chain_autofilled" in P and "deep_locked" in P,
   "P0.2 every new decision string is taught before it can fire")
ok("KIT_STRIP exam" in P and "SYSTEM-BUILT" in P,
   "P0.3 relay section states the fix4 semantics (system-built, kitless exam, SEWN=certificate)")
ok("floor 3" in P, "P0.4 the deep-lock line is named")

# ---- worker: codegen skipped for shipped code ------------------------------------------------------
import dicode.dreaming.gen_manager as _gm  # noqa: E402

_src_all = inspect.getsource(_gm)
ok('if not result.get("_system_code")' in _src_all,
   "W.1 system levels never reach the FM codegen call")
ok('gen_result.get("_system_code") or compilation_results.get(task_id)' in _src_all,
   "W.2 shipped code wins the merge")

# ---- G: v7fix4.1 env-generator guards (the 2026-07-11 A/B double-crash) -----------------------------
# A generated world with non-standard array shapes passes SOLO validation, then kills the whole
# job when training compiles all tasks into one lax.switch. These pin the WIRING of the guards.
_src_cc = inspect.getsource(_gm.EnvGenerator.check_compilation)

ok("scan_banned_randomness(code)" in _src_cc
   and _src_cc.index("scan_banned_randomness") < _src_cc.index("NamedTemporaryFile"),
   "G.1 randomness scan fires before the code is ever written/exec'd")
ok("_canonical_world_specs()" in _src_cc and "diff_world_specs(" in _src_cc
   and "shape_mismatch_message(" in _src_cc,
   "G.2 world-shape contract wired into check_compilation (not just defined)")
ok(_src_cc.index("_validate_on_cpu(key)") < _src_cc.index("diff_world_specs("),
   "G.3 shape check runs after the solo rollout (its errors keep the battle-tested messages)")

_specs = _gm._canonical_world_specs()
_melee = [v for k, v in _specs.items() if "melee_mobs" in k and "position" in k]
ok(len(_specs) > 30 and _melee and _melee[0][0] == "(9, 3, 2)",
   "G.4 canonical template covers every EnvState leaf; hostile capacity = the standard 3")

from dicode.dreaming.prompts.cl_ import gen_env as _ge  # noqa: E402

ok("HARD ENVIRONMENT CONTRACT" in _ge.system_prompt
   and "including any not explicitly named here" in _ge.system_prompt,
   "G.5 prompt contract present, universal rule first (enumeration is exemplary only)")
ok("melee_spawn_multiplier" in _ge.system_prompt,
   "G.6 the 'more monsters' motive is given a legal outlet (TaskParams), not just a wall")

ok("temp_task.task.is_success(state)" in _src_cc,
   "G.7 validation executes is_success (a multitask lax.switch branch the solo step never "
   "calls — job 3929744 died @s82 on a hallucinated attribute inside it)")
ok("temp_task.task.get_task_params()" in _src_cc,
   "G.8 validation executes get_task_params (multitask-init-only on the training path)")

# ---- T: v7fix4.2 deep-wall relay trigger (autoconvert + directive + K-session force) ----------------
# fix4-run s77: zero deep proposals ever — relay start was prompt-soft with no code path. These pin
# that the entry is now pinned: conversion wired IN the deep_locked branch, tick wired IN apply,
# directive survives the empty-notebook early return, and the state key survives _coerce.
from auction.siege_notebook import (  # noqa: E402
    DEEP_WALL_AUTOCONVERT,
    RELAY_TRIGGER_FORCE_SESSIONS,
    RELAY_TRIGGER_HINT,
    _empty_notebook,
)

ok(DEEP_WALL_AUTOCONVERT is True and RELAY_TRIGGER_HINT is True
   and RELAY_TRIGGER_FORCE_SESSIONS == 3,
   "T.1 trigger constants on by default (autoconvert + hint + K=3)")
_src_rec2 = inspect.getsource(SiegeNotebook._reconcile_foci)
ok('"relay_converted(' in _src_rec2
   and _src_rec2.index("deep_locked(") > _src_rec2.index('"relay_converted('),
   "T.2 conversion lives INSIDE the deep_locked branch (refusal is now the fallback, not the rule)")
ok("tier_of(sl) < 4" in _src_rec2,
   "T.3 tier-4 is NOT auto-convertible (the single relay slot never auto-commits to tier-4)")
_src_apply = inspect.getsource(SiegeNotebook.apply_llm_update)
ok("_relay_trigger_tick(" in _src_apply
   and _src_apply.index("_auto_open_from_ranked(") < _src_apply.index("_relay_trigger_tick("),
   "T.4 tick wired in apply AFTER auto-open (a slot auto-open just filled is seen)")
_src_render = inspect.getsource(SiegeNotebook.render_for_prompt)
ok("★RELAY TRIGGER" in _src_render and "relay_forced" in _src_render,
   "T.5 journal renders the directive incl. the escalation warning")
ok("_rt_armed" in _src_render
   and _src_render.index("_rt_armed") < _src_render.index("empty siege notebook"),
   "T.6 an armed trigger survives the empty-notebook early return")
ok("relay_trigger" in _empty_notebook(),
   "T.7 trigger state is in the notebook schema (_coerce would silently drop it otherwise)")
ok("relay_converted" in P and "relay_forced" in P and "★RELAY TRIGGER" in P,
   "T.8 modeler prompt teaches both new decision strings + the directive itself")

import yaml as _yaml  # noqa: E402

_cfg = _yaml.safe_load(open("conf/gen_manager/auction_c_v6siege.yaml", encoding="utf-8"))
ok(_cfg.get("siege_deep_wall_autoconvert") is True
   and _cfg.get("siege_relay_trigger_hint") is True
   and _cfg.get("siege_relay_trigger_force_sessions") == 3,
   "T.9 run config spells out the trigger knobs (sweepable, ablatable)")

# ---- C: v7fix4.4 — the random-cull dose bug (2026-07-12 rung-stall root cause) --------------------
# _process_worker_results random.sample'd ALL compiled designs down to num_generation_tasks=10
# out of ~20-22 per session — a ~50% random death tax with no system-level exemption: 9 of 14
# relay levels became code-less desc_generated husks, the rung reading channel went dry
# (no_fresh at s89/s91), and stall/patience counted the artifact. Source-level asserts read
# the FILES as text (run_dicode.py imports jax/hydra — never import it here).

_rd_src = open("experiments/training/run_dicode.py", encoding="utf-8").read()
ok('res.get("_system_code")' in _rd_src
   and "budget = max(0, limit - len(system_tasks))" in _rd_src
   and "system_tasks + random.sample(fm_tasks, budget)" in _rd_src,
   "C.1 system-built levels exempt from the random cull (FM sampled from the remainder only)")
ok("[cull] kept" in _rd_src and "dropped" in _rd_src,
   "C.2 the cull logs what it drops (no-silent-caps: it taxed 90+ sessions silently)")

from auction.siege_notebook import ZERO_WIN_FORCE_CAP  # noqa: E402

ok(int(_cfg.get("siege_relay_levels_per_session", 2))
   <= int(_cfg.get("siege_zero_win_force_cap", ZERO_WIN_FORCE_CAP)),
   "C.3 relay levels/session fit inside the zero-win force-activation cap "
   "(else the dose starves again one gate later, at activation)")

_gm_src = open("src/dicode/dreaming/gen_manager.py", encoding="utf-8").read()
ok("type(temp_task.task)(None, None)" in _gm_src
   and "_probe.relevant_achievements" in _gm_src and "_probe.label" in _gm_src
   and "Env(None, None) must be constructible" in _gm_src,
   "C.4 validation replicates the cls(None, None) conditioning-table probe AND teaches "
   "the fix through the reflection loop (baseline 3919416 @s96 died on None.replace)")

# ---- D: v7fix4.5 — attribution-driven succession + expand-gate relay exclusion (2026-07-13) ------
# fix4.4-run s114: the modeler's verified diagnosis (kobold lost for lack of enchant/fireball) had
# no road into action — equipment walls deep-locked behind the busy relay slot, ordinary expansion
# hostage to the relay focus's by-construction-zero held-out SR. These pins keep both roads open.

from auction.siege_notebook import RELAY_EXPAND_EXCLUDED, RELAY_SUCCESSION  # noqa: E402

ok(RELAY_SUCCESSION is True and RELAY_EXPAND_EXCLUDED is True,
   "D.1 succession + expand-exclusion on by default")
_src_arch = inspect.getsource(SiegeNotebook._archive_retirement)
ok("failure_attribution_at_retirement" in _src_arch,
   "D.2 the verified attribution survives retirement (succession's data source)")
_src_cand = inspect.getsource(SiegeNotebook._relay_trigger_candidates)
ok("_relay_succession" in _src_cand and "rank.get(s, 10**6)" in _src_cand,
   "D.3 succession outranks the fight-first order in the trigger queue (tree order)")
_src_tick2 = inspect.getsource(SiegeNotebook._relay_trigger_tick)
_src_render2 = inspect.getsource(SiegeNotebook.render_for_prompt)
ok("succession_from" in _src_tick2 and "SUCCESSION:" in _src_render2,
   "D.4 the directive EXPLAINS the succession (evidence shown, not silent reordering)")
_src_expand = inspect.getsource(SiegeNotebook._may_open_new_focus)
ok("relay_expand_excluded" in _src_expand and "_relay_active" in _src_expand,
   "D.5 expand test (b) runs over ORDINARY foci only (relay can't hold expansion hostage)")
ok("SUCCESSION RULE" in P,
   "D.6 modeler prompt teaches the succession contract (enabler first, re-fight after SEWN)")
ok(_cfg.get("siege_relay_succession") is True
   and _cfg.get("siege_relay_expand_excluded") is True,
   "D.7 run config spells out both fix4.5 knobs (sweepable, ablatable)")

# ---- E: v7fix4.6 — descent-wall cliff-split sub-rungs + liveness + succession widening -----------
# Double post-mortem 2026-07-13 (both arms): R0 graduated 73%, floor-2 rung 0% flat — one floor of
# descent is a compound gate (8-kill clear + dark traversal) = a zero-success cliff, no gradient.
# Plus two liveness holes: transitions None-reset the ratchet (oscillation never retired) and the
# succession rejected the run's own verified execution_failure diagnosis.

from auction.siege_notebook import (  # noqa: E402
    RELAY_MAX_REGRESSIONS,
    RELAY_SUCCESSION_CLASSES,
    RUNG_CLIFF_READINGS,
    RUNG_CLIFF_SPLIT,
    RUNG_CLIFF_SR,
    RUNG_LADDER_RADII,
)

ok(RUNG_CLIFF_SPLIT is True and len(RUNG_LADDER_RADII) == 3
   and RUNG_CLIFF_READINGS < 4 and RUNG_CLIFF_SR < 20.0,
   "E.1 cliff split on by default; 3 radius stages; fires below/before the stall boundary")
_src_rung = inspect.getsource(SiegeNotebook.note_rung_reading)
ok("RUNG_CLIFF_SPLIT" in _src_rung
   and _src_rung.index("RUNG_CLIFF_SPLIT") < _src_rung.index("---- REGRESS"),
   "E.2 the cliff branch exists and is checked BEFORE the regress branch (a cliff never regresses)")
ok('not r.get("kit_strip")' in _src_rung
   and '(not _at_r0 or self.th.rung_r0_scaffold)' in _src_rung
   and '_floor_now <= int(r.get("r0_floor", 0))' in _src_rung,
   "E.3 kit-strip never splits; R0 splits only behind rung_r0_scaffold (v7fix4.7 Q1 — the old "
   "'R0 never splits' pin moved to the ablation switch)")
ok("RUNG_SUBSTAGE_GRADUATED" in _src_rung and "rung_substage_graduate_x" in _src_rung
   and "RUNG_SUBSTAGE_REGRESSED" in _src_rung,
   "E.4 scaffold stages graduate on x1 toward FULL and step easier on a stall")
ok('best_by_rung' in _src_rung
   and 'r["best_rung_trained"] = (r.get("best_by_rung") or {}).get(' in _src_rung,
   "E.5 transitions RESTORE the per-rung best (the fake-new-high patience reset is closed)")
ok(_src_rung.count("_regress_budget(") >= 3 and RELAY_MAX_REGRESSIONS >= 2,
   "E.6 every regress-family path (kit-strip / sub-stage / floor) burns the bounded budget")
ok("focus_retired_relay_stalled" in inspect.getsource(SiegeNotebook.note_rung_reading),
   "E.7 the budget overflow retires through the normal machinery (succession-consumable)")
_src_succ = inspect.getsource(SiegeNotebook._relay_succession)
ok("RELAY_SUCCESSION_CLASSES" in _src_succ
   and "execution_failure" in RELAY_SUCCESSION_CLASSES
   and 'missing.startswith("enter_")' in _src_succ,
   "E.8 succession accepts verified execution_failure; an entrance key never re-enters via -1")
ok('relay_scaffold' in _gm_src and "set_monsters_killed" in _gm_src
   and "down_ladder_radius" in _gm_src and "int(floor) >= 1" in _gm_src,
   "E.9 gen_manager renders the scaffold knobs; the clear-gate credit is floor>=1 only "
   "(monsters_killed[0] inits 10 — a smaller write would LOCK the open overworld ladder)")
ok(_gm_src.count("spawn_sub_stage") >= 3,
   "E.10 the rung-reading filter keys off sub_stage too (built, persisted to graphml, filtered — "
   "a stale easier-stage level must not fake-graduate the harder stage)")
ok("rung_sub_stage_" in _gm_src and "rung_regress_count_" in _gm_src,
   "E.11 wandb telemetry carries the scaffold stage + the regress budget burn")
_wb_src = open("src/minicraftax/world_builder.py", encoding="utf-8").read()
ok("down_ladder_radius" in _wb_src and "_resolve_down_ladder_spawn" in _wb_src
   and "jnp.maximum(patch, TORCH_LIGHT_MAP" in _wb_src
   # v7fix5.7: the stamp loop is graded — default (ladder_only=False) still lights ladder AND
   # spawn (the E.12 contract); "ladder" mode drops only the spawn stamp (the anneal middle rung).
   and "((ladder,) if ladder_only else (ladder, self.player_position))" in _wb_src
   and "_stamp, _ladder_only = self._down_ladder_spawn_radius is not None, False" in _wb_src,
   "E.12 world_builder: radius spawn resolved in build(rng) + torch light at ladder AND spawn "
   "(dark floors: an unlit scaffold spawn is a blind start; v7fix5.7 grading keeps this default)")
ok(_cfg.get("siege_rung_cliff_split") is True
   and int(_cfg.get("siege_relay_max_regressions", 0)) == RELAY_MAX_REGRESSIONS
   and list(_cfg.get("siege_rung_ladder_radii") or []) == list(RUNG_LADDER_RADII),
   "E.13 run config spells out the fix4.6 knobs (sweepable, ablatable)")
_src_render46 = inspect.getsource(SiegeNotebook.render_for_prompt)
ok("scaffold sub-stage" in _src_render46 and "sawtooth" in _src_render46,
   "E.14 the journal renders the sub-stage (a sawtooth trained curve must not read as stagnation)")
from auction.modeler import MODELER_SIEGE_SYSTEM_PROMPT as _P46  # noqa: E402

ok("SUB-STAGES" in _P46 and "sawtooth" in _P46,
   "E.15 the modeler prompt teaches the scaffold interface (mechanism knowledge, zero course "
   "answers — the split trigger, dials and transitions all stay code-driven)")

# ---- v7fix4.7: R0 scaffold + DEFEND-driven relay patience + blacklist exemption --------------------
from auction.siege_notebook import (  # noqa: E402
    RELAY_DEFEND_BUDGET,
    RELAY_DEFEND_RISING_K,
    RUNG_R0_SCAFFOLD,
)

ok(RUNG_R0_SCAFFOLD is True and RELAY_DEFEND_BUDGET == 2 and RELAY_DEFEND_RISING_K == 2,
   "Q.1 fix4.7 defaults: R0 scaffold ON; 2 defence windows; ratchet must rise >= 2 of last 3")
ok('if floor_now == int(r.get("r0_floor", 0)) and 1 <= new_stage <= _r0_skip_hi:' in _src_rung
   and 'if _rg_at_r0 and 1 <= new_stage <= _r0_skip_hi:' in _src_rung
   and 'new_stage = _r0_skip_hi + 1' in _src_rung,
   "Q.2 R0 skips the descent-leg stages in BOTH directions (no descent leg on the target floor; "
   "v7fix5.3 generalized the skip set — 2/1 under the old table, 5..1 under the descent regime; "
   "the transition path 5->4->3->0 stays pinned in test_siege_fix47 and fix53 L.4)")
# v7fix5.5 batch-1: the stage->knob table (with the R0 credit pin) was extracted into
# _stage_knobs so the journal's scaffold-facts disclosure renders from the SAME table
# (relay_scaffold delegates; behavior stays pinned live in test_siege_fix47 + fix55 B3.9).
_src_scaf47 = inspect.getsource(SiegeNotebook.relay_scaffold) + (
    inspect.getsource(SiegeNotebook._stage_knobs)
    if hasattr(SiegeNotebook, "_stage_knobs") else ""
)
ok('int(r.get("spawn_floor", 0)) == int(r.get("r0_floor", -1))' in _src_scaf47
   and "credit = 0" in _src_scaf47,
   "Q.3 an R0 scaffold NEVER emits a monster credit (the target floor's down-gate stays LOCKED)")
ok(_src_rung.count("ratchet_log") >= 3 and 'r["ratchet_log"] = []' in _src_rung,
   "Q.4 the micro-ratchet log is maintained per reading and reset on every rung transition")
ok("RELAY_DEFENCE_WINDOW" in _src_rung and "RELAY_DEFENDED" in _src_rung
   and callable(getattr(SiegeNotebook, "_verify_relay_defence"))
   and callable(getattr(SiegeNotebook, "_relay_ratchet_rising")),
   "Q.5 patience exhaustion with a rising ratchet opens ONE defence window; the citation is "
   "verified against the true readings (facts, never narratives)")
ok(callable(getattr(SiegeNotebook, "_blacklist_count"))
   and "rising_retirements" in inspect.getsource(SiegeNotebook._archive_retirement)
   and "_blacklist_count" in inspect.getsource(SiegeNotebook._reconcile_foci),
   "Q.6 a retirement with the ratchet still rising archives + cooldowns but does NOT stack "
   "toward the 2-strikes blacklist")
_src_render47 = inspect.getsource(SiegeNotebook.render_for_prompt)
ok("RELAY DEFENCE WINDOW" in _src_render47 and "round(float(x), 1)" in _src_render47,
   "Q.7 the journal renders the defence window with the exact citable numbers, and low-SR rung "
   "readings render at 1dp (unreadable numbers cannot be cited)")
ok("RELAY PATIENCE DEFENCE" in _P46 and "R0 itself can split too" in _P46,
   "Q.8 the modeler prompt teaches the defence protocol + the R0 scaffold (mechanism knowledge)")
ok("_PRECREDITABLE_ACHIEVEMENTS" in _gm_src and "pre-credit" in _gm_src,
   "Q.9 gen_manager strips reset-time kit/floor pre-credit rows (~100% artifacts) from the "
   "task-performance prompt block (the 2026-07-14 double mis-diagnosis source)")
ok(_cfg.get("siege_rung_r0_scaffold") is True
   and int(_cfg.get("siege_relay_defend_budget", 0)) == RELAY_DEFEND_BUDGET
   and int(_cfg.get("siege_relay_defend_rising_k", 0)) == RELAY_DEFEND_RISING_K,
   "Q.10 run config spells out the fix4.7 knobs (sweepable, ablatable)")

print(f"\nv7fix4 designcheck: ALL {_n} CHECKS PASS")
