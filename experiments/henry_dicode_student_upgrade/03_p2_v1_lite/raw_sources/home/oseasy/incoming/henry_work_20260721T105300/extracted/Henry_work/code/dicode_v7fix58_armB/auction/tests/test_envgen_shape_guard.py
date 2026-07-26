"""v7fix4.1 env-generator guards: world-shape contract + banned-randomness scan.

Root incident (2026-07-11, v7fix4 A/B double-crash): the env_generator FM wrote a level
whose world doubled the hostile-mob capacities (melee 3->6, ranged 2->4). Each task
validates SOLO in check_compilation, so the non-standard world passed; training then
compiled ALL tasks into one ``jax.lax.switch``, whose branches must have identical
output types -> baseline arm crashed, method arm hung. The guards close that gap at
validation time and teach the fix through the reflection loop:

  G1 gen_env system prompt carries the HARD ENVIRONMENT CONTRACT (universal rule first,
     named arrays only as examples — nothing to route around);
  G2 scan_banned_randomness rejects numpy.random / stdlib random pre-exec (frozen into
     constants under JIT: the world silently stops varying per reset);
  G3 diff_world_specs walks EVERY leaf mechanically (no hand-written field list) and
     flags shape/dtype drift, missing and extra paths alike;
  G4 (jax-gated) the canonical template comes from a blank WorldBuilder under default
     StaticEnvParams, and a WorldBuilder fed a custom StaticEnvParams diffs non-empty.

G1-G3 are pure python: envgen_guards.py / gen_env.py are loaded by file path so no
jax/craftax install is needed (same pattern as test_v7fix2_kit_contract.py).
"""

import importlib.util
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))


def _load_by_path(name, *rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_REPO, *rel))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guards = _load_by_path("envgen_guards", "src", "dicode", "dreaming", "envgen_guards.py")
gen_env = _load_by_path("gen_env_prompt", "src", "dicode", "dreaming", "prompts", "cl_", "gen_env.py")


# ---- G1: prompt contract ---------------------------------------------------------------------

def test_contract_block_present_in_system_prompt():
    sp = gen_env.system_prompt
    assert "HARD ENVIRONMENT CONTRACT" in sp
    assert "jax.lax.switch" in sp
    # Universal rule stated first — enumeration is only exemplary, so an FM cannot
    # "comply" by avoiding just the named arrays.
    assert "ANY field of `EnvState`" in sp
    assert "including any not explicitly named here" in sp


def test_contract_forbids_custom_params_and_teaches_the_legal_outlet():
    sp = gen_env.system_prompt
    assert "NEVER construct your own `StaticEnvParams(...)`" in sp
    # The "more monsters" motive gets a legal outlet, not just a wall:
    assert "melee_spawn_multiplier" in sp
    assert "get_task_params" in sp


def test_contract_bans_non_jax_randomness():
    sp = gen_env.system_prompt
    assert "numpy.random" in sp
    assert "jax.random.split" in sp


# ---- G2: banned-randomness scan --------------------------------------------------------------

@pytest.mark.parametrize(
    "snippet",
    [
        "pos = np.random.randint(0, 48, size=(2,))",
        "idx = numpy.random.choice(candidates)",
        "import random\nx = random.random()",
        "from random import randint",
        "from numpy import random as npr",
    ],
)
def test_scan_flags_banned_sources(snippet):
    msg = guards.scan_banned_randomness(snippet)
    assert msg
    assert "rng" in msg and "jax.random.split" in msg


@pytest.mark.parametrize(
    "snippet",
    [
        "rng, sub = jax.random.split(rng)\nx = jax.random.randint(sub, (), 0, 48)",
        "builder.place_randomly(rng, 0, BlockType.TREE, n=5)",
        "# randomize the arena layout across resets",
        "builder.add_mobs_randomly_near(rng, 0, MobType.MELEE, (24, 24), 2, 6, n=2)",
    ],
)
def test_scan_passes_legal_code(snippet):
    assert guards.scan_banned_randomness(snippet) == ""


# ---- G3: per-leaf spec diff ------------------------------------------------------------------

_CANON = {
    ".melee_mobs.position": ("(9, 3, 2)", "int32"),
    ".melee_mobs.health": ("(9, 3)", "float32"),
    ".ranged_mobs.position": ("(9, 2, 2)", "int32"),
}


def test_diff_empty_on_exact_match():
    assert guards.diff_world_specs(_CANON, dict(_CANON)) == []


def test_diff_flags_capacity_doubling():
    got = dict(_CANON)
    got[".melee_mobs.position"] = ("(9, 6, 2)", "int32")
    got[".ranged_mobs.position"] = ("(9, 4, 2)", "int32")
    lines = guards.diff_world_specs(_CANON, got)
    assert len(lines) == 2
    assert any("(9, 6, 2)" in l and "(9, 3, 2)" in l for l in lines)


def test_diff_flags_dtype_drift_and_missing_and_extra_paths():
    got = dict(_CANON)
    got[".melee_mobs.health"] = ("(9, 3)", "int32")  # dtype drift
    del got[".ranged_mobs.position"]  # missing
    got[".my_custom_field"] = ("(4,)", "int32")  # extra
    lines = guards.diff_world_specs(_CANON, got)
    assert len(lines) == 3
    assert any("MISSING" in l for l in lines)
    assert any("UNEXPECTED" in l for l in lines)


def test_mismatch_message_teaches_and_truncates():
    msg = guards.shape_mismatch_message([f".field_{i}: got X, expected Y" for i in range(20)])
    assert "lax.switch" in msg
    assert "StaticEnvParams" in msg
    assert "melee_spawn_multiplier" in msg
    assert "and 8 more mismatched field(s)" in msg


# ---- G4: canonical template + real WorldBuilder (jax-gated; runs in Oscar full sanity) -------

def test_canonical_template_and_custom_static_params_diff():
    jax = pytest.importorskip("jax")
    craftax_state = pytest.importorskip("craftax.craftax.craftax_state")
    from minicraftax.world_builder import WorldBuilder

    # Self-contained: build the spec map here rather than importing gen_manager's
    # _canonical_world_specs/_flatten_world_specs. In the FULL suite, an earlier test
    # installs a bare gen_manager double in sys.modules ("unknown location" ImportError),
    # so a top-level `from dicode.dreaming.gen_manager import ...` in a jax-gated test is
    # order-fragile. gen_manager's own helper is covered by the clean-process designcheck
    # (G.4); this test's job is the diff logic + the eval_shape round-trip.
    def _specs(static_params):
        def _world(rng):
            return WorldBuilder(rng, static_params, craftax_state.EnvParams()).build(rng)

        struct = jax.eval_shape(_world, jax.random.PRNGKey(0))
        leaves = jax.tree_util.tree_flatten_with_path(struct)[0]
        return {
            jax.tree_util.keystr(path): (str(tuple(leaf.shape)), str(leaf.dtype))
            for path, leaf in leaves
        }

    canon = _specs(craftax_state.StaticEnvParams())
    # The canonical template covers the whole EnvState, standard hostile capacities included.
    melee_pos = [v for k, v in canon.items() if "melee_mobs" in k and "position" in k]
    assert melee_pos and melee_pos[0][0] == "(9, 3, 2)"
    assert len(canon) > 30  # every leaf, not a hand-picked subset

    # A world built under a custom StaticEnvParams (the crash signature: capacity doubled)
    # must diff non-empty against the canon — and only on the mob arrays it inflated.
    got = _specs(craftax_state.StaticEnvParams(max_melee_mobs=6, max_ranged_mobs=4))
    lines = guards.diff_world_specs(canon, got)
    assert lines
    assert all(("melee_mobs" in l) or ("ranged_mobs" in l) for l in lines)
    assert any("(9, 6, 2)" in l for l in lines)

    # A second default-params world matches the canon exactly (no false positives).
    good = _specs(craftax_state.StaticEnvParams())
    assert guards.diff_world_specs(canon, good) == []
