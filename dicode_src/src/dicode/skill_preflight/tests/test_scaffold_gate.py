"""Offline tests for the C-0 scaffold gate (C-2-lite §3).

Fixture = a replica of probe task_19 "MAKE_DIAMOND_SWORD" (the leak analysis's exhibit A:
pre-made pickaxe + 50 coal + 14-achievement pre-mark), per the design doc's §5.1 plan.

Run on the pod:
    cd /workspace/mechanism_UED/dicode_src
    uv run pytest src/dicode/skill_preflight/tests/test_scaffold_gate.py -v
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from dicode.skill_preflight.scaffold_gate import (
    GateVerdict,
    _extract_scaffold_facts,
    check_code,
)

# --- fixtures ------------------------------------------------------------------------------

# Probe-endpoint-like snapshot (fractions 0..1): resource line mastered, iron tools flat.
SNAPSHOT = {
    "collect_wood": 0.99, "place_table": 0.98, "make_wood_pickaxe": 0.95,
    "collect_stone": 0.96, "make_stone_pickaxe": 0.90, "collect_coal": 0.878,
    "collect_iron": 0.646, "place_furnace": 0.978,
    "make_iron_pickaxe": 0.027, "make_iron_sword": 0.026,
    "collect_diamond": 0.074, "make_diamond_sword": 0.006,
    "defeat_lizard": 0.0, "enter_sewers": 0.0, "defeat_zombie": 0.9,
}

TASK_19_REPLICA = '''
from craftax.craftax.constants import Achievement

class Env(BaseTask):
    """MAKE_DIAMOND_SWORD"""
    def __init__(self, static_params, params):
        super().__init__(static_params, params)
        self.relevant_achievements = [
            Achievement.MAKE_DIAMOND_SWORD, Achievement.COLLECT_DIAMOND,
        ]
        # 14-achievement pre-mark of the prerequisite chain (task_20 pattern)
        self.completed_achievements = [
            Achievement.COLLECT_WOOD, Achievement.PLACE_TABLE,
            Achievement.MAKE_WOOD_PICKAXE, Achievement.COLLECT_STONE,
            Achievement.MAKE_STONE_PICKAXE, Achievement.PLACE_FURNACE,
            Achievement.COLLECT_COAL, Achievement.COLLECT_IRON,
            Achievement.MAKE_IRON_PICKAXE, Achievement.MAKE_WOOD_SWORD,
            Achievement.MAKE_STONE_SWORD, Achievement.DEFEAT_ZOMBIE,
            Achievement.COLLECT_DRINK, Achievement.EAT_COW,
        ]
        self.label = "MAKE_DIAMOND_SWORD"

    def generate_world(self, rng, static_params, params):
        builder = WorldBuilder(static_params)
        # --- ADDED SCAFFOLDING ---
        builder.set_player_inventory({"pickaxe": 3, "coal": 50})
        builder.add_mobs_randomly_near(rng, mob_name="lizard", min_dist=4, max_dist=8, count=2)
        # --- END SCAFFOLDING ---
        return builder.build()
'''

CLEAN_ONE_STEP = '''
from craftax.craftax.constants import Achievement

class Env(BaseTask):
    """Train make_iron_pickaxe bare; scaffold nothing the agent masters."""
    def __init__(self, static_params, params):
        super().__init__(static_params, params)
        self.relevant_achievements = [Achievement.MAKE_IRON_PICKAXE]
        self.completed_achievements = []
        self.label = "MAKE_IRON_PICKAXE"

    def generate_world(self, rng, static_params, params):
        builder = WorldBuilder(static_params)
        builder.add_resource_randomly(rng, "iron_ore", count=6)
        return builder.build()
'''

# Sanctioned scaffolding under the one-step contract: focus = defeat_lizard (unmastered);
# its IMMEDIATE prereq is enter_sewers, which stays bare (the agent must descend). Starting
# on floor 2 grants only enter_dungeon + enter_gnomish_mines — DEEPER-in-the-chain,
# unmastered, non-immediate prerequisites: exactly the permitted scaffold.
# Semantics note: if enter_sewers were ALSO listed relevant, ITS immediate prereq
# (enter_gnomish_mines) would have to stay bare too and floor 2 would violate — declaring a
# prereq reward-relevant tightens the contract one hop. That stricter case is covered by
# test_floor_grant_hitting_immediate_prereq_is_r3.
SNAPSHOT_DEEP = dict(SNAPSHOT)
SNAPSHOT_DEEP.update({"enter_dungeon": 0.1, "enter_gnomish_mines": 0.05, "enter_sewers": 0.0})

SANCTIONED_FLOOR_SCAFFOLD = '''
from craftax.craftax.constants import Achievement

class Env(BaseTask):
    """Train the sewers descent + lizard fight; skip only the earlier unmastered floors."""
    def __init__(self, static_params, params):
        super().__init__(static_params, params)
        self.relevant_achievements = [Achievement.DEFEAT_LIZARD]
        self.completed_achievements = []
        self.label = "DEFEAT_LIZARD"

    def generate_world(self, rng, static_params, params):
        builder = WorldBuilder(static_params)
        builder.set_starting_floor(2)
        return builder.build()
'''


# --- tests ---------------------------------------------------------------------------------

def test_task19_replica_flags_r1_and_r3():
    v = check_code(TASK_19_REPLICA, SNAPSHOT)
    assert not v.ok
    assert "R1_premark_mastered" in v.violations   # collect_wood etc. are mastered
    assert "R3_focus_prereq_scaffolded" in v.violations
    # Evidence must carry the indictment specifics for the repair prompt.
    assert "make_iron_pickaxe" in v.evidence       # pickaxe:3 grants it; prereq of collect_diamond
    assert "set_player_inventory" in v.evidence
    assert "collect_wood" in v.evidence


def test_premarking_relevant_achievement_is_r2():
    code = TASK_19_REPLICA.replace(
        "Achievement.EAT_COW,", "Achievement.EAT_COW, Achievement.MAKE_DIAMOND_SWORD,"
    )
    v = check_code(code, SNAPSHOT)
    assert "R2_focus_premarked" in v.violations
    assert "make_diamond_sword" in v.evidence


def test_clean_one_step_task_passes():
    v = check_code(CLEAN_ONE_STEP, SNAPSHOT)
    assert v.ok, v.evidence


def test_sanctioned_deep_floor_scaffold_passes():
    """Skipping earlier UNMASTERED floors while the focus + immediate prereq stay bare is
    exactly what the one-step contract permits — the gate must not flag it."""
    v = check_code(SANCTIONED_FLOOR_SCAFFOLD, SNAPSHOT_DEEP)
    assert v.ok, v.evidence


def test_floor_grant_hitting_immediate_prereq_is_r3():
    """Starting ON the focus's floor (floor 3 grants enter_sewers, defeat_lizard's direct
    prereq) crosses the line."""
    code = SANCTIONED_FLOOR_SCAFFOLD.replace("set_starting_floor(2)", "set_starting_floor(3)")
    v = check_code(code, SNAPSHOT_DEEP)
    assert "R3_focus_prereq_scaffolded" in v.violations
    assert "set_starting_floor" in v.evidence


def test_near_mobs_alone_are_not_a_violation():
    """S3 (near-spawn mobs) is deliberately sanctioned — terminal-skill practice is the
    mechanism's payoff. A task whose ONLY scaffold signature is mobs must pass."""
    code = CLEAN_ONE_STEP.replace(
        'builder.add_resource_randomly(rng, "iron_ore", count=6)',
        'builder.add_mobs_randomly_near(rng, mob_name="zombie", min_dist=4, max_dist=8, count=2)',
    )
    v = check_code(code, SNAPSHOT)
    assert v.ok


def test_unparseable_code_is_not_the_gates_problem():
    v = check_code("def broken(:", SNAPSHOT)
    assert v.ok and not v.parse_ok


def test_parity_with_offline_audit_tool():
    """The gate's AST extraction must agree with scaffold_audit.audit_code (the offline
    302-task auditor) on the shared fields, so the two can't silently drift."""
    root = Path(__file__).resolve().parents[4]  # dicode_src/
    audit_path = root / "scaffold_audit.py"
    if not audit_path.exists():
        pytest.skip("scaffold_audit.py not in this checkout")
    spec = importlib.util.spec_from_file_location("scaffold_audit", audit_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["scaffold_audit"] = mod
    spec.loader.exec_module(mod)

    audit = mod.audit_code(TASK_19_REPLICA)
    facts = _extract_scaffold_facts(TASK_19_REPLICA)

    assert audit["parse_ok"] and facts["parse_ok"]
    assert set(audit["relevant"]) == {n.upper() for n in ("make_diamond_sword", "collect_diamond")}
    assert set(a.lower() for a in audit["relevant"]) == {a.lower() for a in facts["relevant"]}
    # S1 items agree
    assert audit["scaffolds"]["S1_inventory"]["items"].keys() == facts["inventory"].keys()
    # S2 premark count agrees
    assert audit["scaffolds"]["S2_premark"]["n_premarked"] == len(facts["premarked"]) == 14
