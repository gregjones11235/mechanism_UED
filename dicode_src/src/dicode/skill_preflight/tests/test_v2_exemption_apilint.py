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


def test_eval_detail_carry_threading_structural():
    """Pin the autopsy trackers into ALL carry positions of craftax_evaluation.
    Guards against the silent str.replace no-op that shipped an UnboundLocalError:
    each tracker must appear in init tuple + unpack + step-return + update (>=4 hits)."""
    import os
    src = open(os.path.join(os.path.dirname(__file__),
               "../../craftax_evaluation.py")).read()
    for name in ("floor_at_done", "health_at_done", "max_floor"):
        assert src.count(name) >= 4, f"{name} not threaded through eval carry"
    assert "final_carry[14]" in src and '"_details"' in src


def test_depth_potential_wrapper_math():
    """φ-shaping math on a fake env: r' = r + γ·c·lvl(s')·(1-done) − c·lvl(s).
    Runs on pod (jax present); skips hermetically where jax is unavailable."""
    import pytest
    jnp = pytest.importorskip("jax.numpy")
    from dicode.wrappers import DepthPotentialWrapper

    class _St:
        def __init__(self, lvl): self.player_level = jnp.array(lvl)

    class _FakeEnv:
        def step(self, rng, state, action, params=None):
            nxt = _St(state.player_level + 1)          # always descend one floor
            done = jnp.array(0.0)
            return None, nxt, jnp.array(1.0), done, {}
        def step_env(self, key, state, action, params=None, task_embeddings=None):
            return self.step(key, state, action, params)

    w = DepthPotentialWrapper(_FakeEnv(), c=0.5, gamma=0.99)
    _, nxt, r, d, _ = w.step(None, _St(1.0), None)
    _, _, r_env, _, _ = w.step_env(None, _St(1.0), None)
    assert abs(float(r_env) - float(r)) < 1e-6
    # r' = 1 + 0.99*0.5*2*(1-0) - 0.5*1 = 1 + 0.99 - 0.5 = 1.49
    assert abs(float(r) - 1.49) < 1e-5
    # terminal: φ(s') zeroed by (1-done)
    class _FakeEnvDone(_FakeEnv):
        def step(self, rng, state, action, params=None):
            return None, _St(state.player_level + 1), jnp.array(1.0), jnp.array(1.0), {}
    w2 = DepthPotentialWrapper(_FakeEnvDone(), c=0.5, gamma=0.99)
    _, _, r2, _, _ = w2.step(None, _St(2.0), None)
    assert abs(float(r2) - (1.0 - 1.0)) < 1e-5   # 1 + 0 - 0.5*2 = 0


def test_depth_potential_isolation_structural():
    """Shaping must NEVER touch the official eval or preflight-admission paths."""
    import os
    base = os.path.join(os.path.dirname(__file__), "../../..")
    for f in ("dicode/craftax_evaluation.py", "dicode/evaluation/online_evaluation.py"):
        assert "DepthPotential" not in open(os.path.join(base, f)).read(), f
    ppo = open(os.path.join(base, "dicode/ppo_tr.py")).read()
    assert ppo.count("DepthPotentialWrapper(base_env") == 1   # train path only, not AllTasks
    assert 'config.get("depth_potential_c", 0.0)' in ppo       # flag-gated, default off


def test_combat_bounty_wrapper_math():
    """Bounty fires only on 0->1 flips of selected achievement bits; latched bits pay nothing."""
    import pytest
    jnp = pytest.importorskip("jax.numpy")
    from dicode.wrappers import CombatBountyWrapper

    class _St:
        def __init__(self, ach): self.achievements = jnp.array(ach)

    class _FakeEnv:
        def __init__(self, before, after): self.b, self.a = before, after
        def step(self, rng, state, action, params=None):
            return None, _St(self.a), jnp.array(1.0), jnp.array(0.0), {}
        def step_env(self, key, state, action, params=None, task_embeddings=None):
            return self.step(key, state, action, params)

    # indices 2,3 selected; bit2 flips 0->1 (+bounty), bit3 already latched (no pay)
    env = _FakeEnv([0, 1, 0, 1], [0, 1, 1, 1])
    w = CombatBountyWrapper(env, bounty=2.0, indices=[2, 3])
    _, _, r, _, _ = w.step(None, _St([0, 1, 0, 1]), None)
    assert abs(float(r) - 3.0) < 1e-6      # 1 + 2.0*1
    _, _, r_env, _, _ = w.step_env(None, _St([0, 1, 0, 1]), None)
    assert abs(float(r_env) - 3.0) < 1e-6  # live path pays too
    # no flips -> base reward only
    env2 = _FakeEnv([0, 1, 1, 1], [0, 1, 1, 1])
    w2 = CombatBountyWrapper(env2, bounty=2.0, indices=[2, 3])
    _, _, r2, _, _ = w2.step(None, _St([0, 1, 1, 1]), None)
    assert abs(float(r2) - 1.0) < 1e-6


def test_shaping_wrappers_intercept_live_path():
    """The training chain calls step_env (wrappers_cl Distributed vmaps _env.step_env).
    A wrapper overriding only step() is silently bypassed via __getattr__ -- the bug
    that inertized three shaping runs. Pin: both wrappers define step_env themselves,
    and the live wrapper really does route through step_env."""
    import os
    base = os.path.join(os.path.dirname(__file__), "../../..")
    w = open(os.path.join(base, "dicode/wrappers.py")).read()
    for cls in ("DepthPotentialWrapper", "CombatBountyWrapper"):
        body = w.split(f"class {cls}")[1].split("\nclass ")[0]
        assert "def step_env(" in body, f"{cls} lacks step_env -- would be bypassed"
        assert "_env.step_env(" in body, f"{cls}.step_env must delegate to inner step_env"
    cl = open(os.path.join(base, "dicode/wrappers_cl.py")).read()
    assert "self._env.step_env" in cl


def test_combat_bounty_isolation_structural():
    import os
    base = os.path.join(os.path.dirname(__file__), "../../..")
    for f in ("dicode/craftax_evaluation.py", "dicode/evaluation/online_evaluation.py"):
        assert "CombatBounty" not in open(os.path.join(base, f)).read(), f
    ppo = open(os.path.join(base, "dicode/ppo_tr.py")).read()
    assert ppo.count("CombatBountyWrapper(base_env") == 2   # combat + placebo gates
    assert 'config.get("combat_bounty", 0.0)' in ppo


def test_shaping_survives_vmapped_stepenv_capture():
    """Replicates the EXACT capture semantics of wrappers_cl's Distributed wrapper:
    jax.vmap(wrapper.step_env, in_axes=(0,0,0,None,None)) over a batch.
    Shaping must survive this capture -- the liveness proof the ACTIVE print never was."""
    import pytest
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    from dicode.wrappers import CombatBountyWrapper, DepthPotentialWrapper

    class _St:
        def __init__(self, ach, lvl):
            self.achievements = ach; self.player_level = lvl
        def tree_flatten(self): return ((self.achievements, self.player_level), None)
        @classmethod
        def tree_unflatten(cls, aux, ch): return cls(*ch)
    jax.tree_util.register_pytree_node(_St, _St.tree_flatten, _St.tree_unflatten)

    class _Fake:
        def step_env(self, key, state, action, params=None, task_embeddings=None):
            nxt = _St(state.achievements.at[2].set(1), state.player_level + 1)
            return jnp.zeros(3), nxt, jnp.array(1.0), jnp.array(0.0), {}

    batch = _St(jnp.zeros((4, 5), dtype=jnp.int32), jnp.zeros((4,)))
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    acts = jnp.zeros((4,), dtype=jnp.int32)

    wb = CombatBountyWrapper(_Fake(), bounty=2.0, indices=[2, 3])
    _, _, r, _, _ = jax.vmap(wb.step_env, in_axes=(0, 0, 0, None, None))(keys, batch, acts, None, None)
    assert jnp.allclose(r, 3.0), f"bounty lost under vmapped step_env capture: {r}"

    wp = DepthPotentialWrapper(_Fake(), c=0.5, gamma=0.99)
    _, _, rp, _, _ = jax.vmap(wp.step_env, in_axes=(0, 0, 0, None, None))(keys, batch, acts, None, None)
    assert jnp.allclose(rp, 1.0 + 0.99 * 0.5 * 1.0), f"phi lost under capture: {rp}"


def test_rarity_bounty_placebo_structural():
    """Placebo flag: gated, mutually exclusive with combat_bounty, 8-name non-combat set,
    reuses the same (already liveness-proven) wrapper class."""
    import os
    ppo = open(os.path.join(os.path.dirname(__file__), "../../..", "dicode/ppo_tr.py")).read()
    assert 'config.get("rarity_bounty", 0.0)' in ppo
    assert "mutually exclusive" in ppo
    assert ppo.count("CombatBountyWrapper(base_env") == 2   # combat gate + placebo gate
    for n in ("COLLECT_DIAMOND", "ENCHANT_ARMOUR", "MAKE_DIAMOND_ARMOUR"):
        assert n in ppo


def test_online_eval_leak_fix_structural():
    """Structural test #3: the per-session evaluation/* channel must be a REAL held-out
    eval (run_session_evaluation), and the training-slot (shaped) numbers must live
    under evaluation_shaped/*. Guards the original_craftax-slot leak from recurring."""
    import os
    rd = open(os.path.join(os.path.dirname(__file__),
              "../../../../experiments/training/run_dicode.py")).read()
    assert rd.count("run_session_evaluation(") >= 2       # priming + in-loop
    assert 'f"evaluation_shaped/{key}"' in rd             # shaped channel renamed
    assert rd.count('f"evaluation/{key}"') == 0           # no direct evaluation/* from training slots
    assert "LEAK FIX" in rd


def test_crash_fix_arms_structural():
    """Both crash-fix arms flag-gated, default off = v1-identical math."""
    import os
    base = os.path.join(os.path.dirname(__file__), "../../..")
    net = open(os.path.join(base, "dicode/network.py")).read()
    assert 'self.config.get("critic_grad_firewall", False)' in net
    assert "stop_gradient(embedding)" in net and ")(critic_in)" in net
    assert ")(embedding)\n\t\tcritic = nn.relu" not in net   # critic head no longer reads trunk directly
    ppo = open(os.path.join(base, "dicode/ppo_tr.py")).read()
    assert 'config.get("adaptive_value_scale", False)' in ppo
    assert "jnp.maximum(jnp.std(targets_r), 1.0)" in ppo
    assert "_vscale = 1.0" in ppo                             # default path divides by exactly 1.0


def test_lr_schedule_clamped():
    """The 7x same-position crash root cause: unclamped anneal -> negative LR ->
    gradient ascent. Pin the clamp; in-horizon math is identity, so this is a pure
    bug fix requiring no flag."""
    import os
    ppo = open(os.path.join(os.path.dirname(__file__), "../../..",
               "dicode/ppo_tr.py")).read()
    seg = ppo.split("def linear_schedule")[1].split("return")[0]
    assert "jnp.maximum(" in seg and "0.0" in seg, "anneal must be clamped at zero"


# --- v2.1: rule 3 was ungated and defeated the rule-1 exemption ----------------------------

class _IronT:  # focus = collect_iron, its prereq make_stone_pickaxe MASTERED
    target_achievements = ["collect_iron"]
    sr_snapshot = dict(SNAPSHOT, make_stone_pickaxe=0.81, collect_iron=0.10)


def test_v1_rule3_contradicts_the_exemption():
    """Regression witness: with the exemption ON, rule 3 still forbids *provisioning*
    mastered skills unconditionally. Two prohibitions vs one parenthetical permission."""
    on = format_target_for_prompt_one_step(_IronT(), mastered_exemption=True)
    assert "MAY be provided or skipped" in on
    assert "Do NOT pre-mark or provision" in on


def test_r3_v2_splits_premark_from_provision():
    v2 = format_target_for_prompt_one_step(_IronT(), mastered_exemption=True, r3_v2=True)
    assert "Do NOT pre-mark or provision" not in v2
    assert "PROVIDING a mastered prerequisite" in v2
    cv2 = format_scaffold_rules_for_coder(_IronT.sr_snapshot,
                                          mastered_exemption=True, r3_v2=True)
    con = format_scaffold_rules_for_coder(_IronT.sr_snapshot, mastered_exemption=True)
    assert "TOOL TIER" in cv2 and "TOOL TIER" not in con


def test_r3_v2_default_off_is_byte_identical():
    """Branch discipline: passing nothing, or r3_v2=False, must be unchanged."""
    assert (format_target_for_prompt_one_step(_IronT(), mastered_exemption=True)
            == format_target_for_prompt_one_step(_IronT(), mastered_exemption=True,
                                                 r3_v2=False))
    assert (format_scaffold_rules_for_coder(SNAPSHOT)
            == format_scaffold_rules_for_coder(SNAPSHOT, r3_v2=False))


# --- v2.2: tier_cap guardrail against r3_v2 overshooting to a downstream tool -------------

def test_tier_cap_alone_is_a_noop():
    """Without r3_v2 there is no exception clause to cap, so the flag must change nothing."""
    assert (format_scaffold_rules_for_coder(_IronT.sr_snapshot, tier_cap=True)
            == format_scaffold_rules_for_coder(_IronT.sr_snapshot))
    assert (format_target_for_prompt_one_step(_IronT(), tier_cap=True)
            == format_target_for_prompt_one_step(_IronT()))


def test_tier_cap_forbids_downstream_tools_without_killing_r3_v2():
    on = format_scaffold_rules_for_coder(_IronT.sr_snapshot, mastered_exemption=True,
                                         r3_v2=True, tier_cap=True)
    off = format_scaffold_rules_for_coder(_IronT.sr_snapshot, mastered_exemption=True,
                                          r3_v2=True)
    assert "WRONG" in on and "WRONG" not in off
    assert "TOOL TIER" in on, "the cap must not remove r3_v2's permission"
    d = format_target_for_prompt_one_step(_IronT(), mastered_exemption=True,
                                          r3_v2=True, tier_cap=True)
    assert "DOWNSTREAM" in d and "PROVIDING a mastered prerequisite" in d


def test_tier_cap_default_off_is_byte_identical():
    """Branch discipline: a replication arm launched from this HEAD must match the old one."""
    for kw in ({}, {"mastered_exemption": True},
               {"mastered_exemption": True, "r3_v2": True}):
        assert (format_scaffold_rules_for_coder(_IronT.sr_snapshot, **kw)
                == format_scaffold_rules_for_coder(_IronT.sr_snapshot, tier_cap=False, **kw))
        assert (format_target_for_prompt_one_step(_IronT(), **kw)
                == format_target_for_prompt_one_step(_IronT(), tier_cap=False, **kw))
