"""v2 tests: R3 mastered-prereq exemption + C-1 api_lint diagnosis.

Run on the pod:
    cd /workspace/mechanism_UED/dicode_src
    uv run pytest src/dicode/skill_preflight/tests/test_v2_exemption_apilint.py -v
"""
from __future__ import annotations

from dicode.skill_preflight.api_lint import diagnose
from dicode.skill_preflight.scaffold_gate import check_code
from dicode.skill_preflight.skill_scheduler import (
    format_scaffold_rules_for_coder,
    format_target_for_prompt_one_step,
)

# --- fixtures: the gnomish sampling-tax scenario from the 2e9 autopsy ----------------------
# enter_dungeon MASTERED (96%), focus = enter_gnomish_mines (16%), task starts on floor 1
# (grants enter_dungeon). v1: R3 violation. v2 exemption: sanctioned.
SNAPSHOT = {
    "enter_dungeon": 0.96, "enter_gnomish_mines": 0.16,
    "collect_wood": 0.99, "make_wood_sword": 0.9,
}

GNOMISH_FLOOR1_TASK = '''
from craftax.craftax.constants import Achievement

class Env(BaseTask):
    """Train the gnomish descent; skip the mastered floor-1 entry."""
    def __init__(self, static_params, params):
        super().__init__(static_params, params)
        self.relevant_achievements = [Achievement.ENTER_GNOMISH_MINES]
        self.completed_achievements = []
        self.label = "ENTER_GNOMISH_MINES"

    def generate_world(self, rng, static_params, params):
        builder = WorldBuilder(static_params)
        builder.set_starting_floor(1)
        return builder.build()
'''


def test_v1_default_still_flags_mastered_prereq_scaffold():
    """Default (no flag) must stay byte-identical to v1: floor-1 start trips R3."""
    v = check_code(GNOMISH_FLOOR1_TASK, SNAPSHOT)
    assert "R3_focus_prereq_scaffolded" in v.violations


def test_v2_exemption_sanctions_mastered_prereq_scaffold():
    v = check_code(GNOMISH_FLOOR1_TASK, SNAPSHOT, mastered_prereq_exemption=True)
    assert v.ok, v.evidence


def test_v2_exemption_still_blocks_unmastered_prereq_scaffold():
    """Exemption is mastery-conditional, not a blanket pass: with enter_dungeon UNMASTERED
    the same task must still trip R3 even with the flag on."""
    weak = dict(SNAPSHOT); weak["enter_dungeon"] = 0.10
    v = check_code(GNOMISH_FLOOR1_TASK, weak, mastered_prereq_exemption=True)
    assert "R3_focus_prereq_scaffolded" in v.violations


def test_prompt_wording_switches_with_flag():
    class T:  # minimal SchedulerTarget stand-in
        target_achievements = ["enter_gnomish_mines"]
        sr_snapshot = SNAPSHOT
    on = format_target_for_prompt_one_step(T(), mastered_exemption=True)
    off = format_target_for_prompt_one_step(T())
    assert "MAY be provided or skipped" in on and "MAY be provided or skipped" not in off
    con = format_scaffold_rules_for_coder(SNAPSHOT, mastered_exemption=True)
    coff = format_scaffold_rules_for_coder(SNAPSHOT)
    assert "EXCEPTION" in con and "EXCEPTION" not in coff


# --- api_lint: the three hallucination classes from the 2e9 ledger --------------------------

CODE_WITH_IMPORT = '''
from craftax.craftax.constants import Achievement, BlockType
class Env: pass
'''


def test_h1_full_path_hermetic():
    """Full H1 path (AST import scan -> importlib -> dir -> difflib) against a synthetic
    module, so the test cannot depend on craftax/jax being installed in the test env."""
    import sys, types
    fake = types.ModuleType("fake_game.constants")
    class BlockType:  # the "real" enum surface
        LADDER_UP = 1
        LADDER_DOWN_BROKEN = 2   # near-miss for the hallucinated LADDER_DOWN
        STONE = 3
        PATH = 4
    fake.BlockType = BlockType
    sys.modules["fake_game.constants"] = fake
    try:
        code = "from fake_game.constants import BlockType\nclass Env: pass\n"
        ev = diagnose(code,
                      "AttributeError: type object 'BlockType' has no attribute 'LADDER_DOWN'")
        assert ev and "LADDER_DOWN" in ev and "DOES NOT EXIST" in ev
        assert "LADDER_DOWN_BROKEN" in ev  # difflib surfaced the near-miss
    finally:
        del sys.modules["fake_game.constants"]


def test_h1_real_craftax_soft():
    """Real-module resolution: exercised for real on the pod (craftax+jax installed);
    soft-skips where craftax is unimportable."""
    ev = diagnose(CODE_WITH_IMPORT,
                  "AttributeError: type object 'BlockType' has no attribute 'LADDER_DOWN'")
    if ev is not None:
        assert "LADDER_DOWN" in ev and "DOES NOT EXIST" in ev


def test_h2_builder_method_hallucination():
    ev = diagnose("class Env: pass",
                  "AttributeError: 'WorldBuilder' object has no attribute 'add_mobs_randomly'")
    # WorldBuilder resolved via fallback map; if minicraftax importable, evidence carries
    # the real near-miss method name.
    if ev is not None:
        assert "add_mobs_randomly" in ev
        assert "add_mobs_randomly_near" in ev


def test_h3_ctor_kwarg_hallucination():
    ev = diagnose("class Env: pass",
                  "TypeError: Inventory.__init__() got an unexpected keyword argument 'wood_sword'")
    if ev is not None:
        assert "wood_sword" in ev and "sword" in ev
        assert "LEVEL fields" in ev


def test_real_world_error_format_no_prefix():
    """Pin the ACTUAL format check_compilation produces: 'Compilation error: ' + str(e),
    i.e. NO exception-class prefix. The 3e8 ablation ran inert because the patterns
    required prefixes — this test locks the fix."""
    import sys, types
    fake = types.ModuleType("fake_game.constants")
    class BlockType:
        LADDER_UP = 1; LADDER_DOWN_BROKEN = 2
    fake.BlockType = BlockType
    sys.modules["fake_game.constants"] = fake
    try:
        code = "from fake_game.constants import BlockType\nclass Env: pass\n"
        ev = diagnose(code, "Compilation error: type object 'BlockType' has no attribute 'LADDER_DOWN'")
        assert ev and "LADDER_DOWN" in ev
    finally:
        del sys.modules["fake_game.constants"]
    ev = diagnose("class E: pass",
                  "Compilation error: Inventory.__init__() got an unexpected keyword argument 'wood_sword'")
    if ev is not None:
        assert "wood_sword" in ev


def test_h4_missing_import():
    ev = diagnose("class E: pass", "Compilation error: name 'jnp' is not defined")
    assert ev and "jnp" in ev and "import jax.numpy as jnp" in ev
    assert diagnose("class E: pass", "Compilation error: name 'foo' is not defined") is not None


def test_non_hallucination_errors_pass_through():
    assert diagnose("x=", "SyntaxError: invalid syntax") is None
    assert diagnose("class E: pass", "") is None
