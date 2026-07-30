#!/usr/bin/env python3
"""CC4 Tier3 — decomposed evaluator (deterministic; inference-only).

Runs ONE frozen evaluation contract across the three scenarios
(FULL_END_TO_END / TIER3_FRONT_HALF_SCAFFOLDED_L2 / TIER3_BACK_HALF_SCAFFOLDED_L2):
    action_mode = greedy_argmax   observation_schema = canonical_craftax_symbolic
    action_space = canonical_craftax_action_set   max_timesteps = 4096
identically for every arm. It validates each episode record (NEG19: an episode without
a valid_start flag is rejected), classifies the terminal label via the failure taxonomy
(NEG20 ambiguity fails closed), computes the frozen metrics, and asserts the Student
checkpoint params are UNCHANGED across the batch (NEG23, with the checkpoint adapter).

REAL ENVIRONMENT INTERFACE (JAX + craftax==1.4.5 host, all JAX-guarded):
  * make_canonical_env()          — exec the SHA-verified canonical S4 task source and
    build the canonical MultiTaskMiniCraftaxEnv (embedding conditioning, size 67);
    asserts observation shape / action count against the frozen audit bindings.
  * reset_from_bank_state()       — inject a materialized bank state, replicating the
    MultiTaskMiniCraftaxEnv.reset_env post-processing EXACTLY (task_params slice,
    calculate_inventory_achievements, floor-entry achievements, obs).
  * assert_front_reset_equivalence() — FRONT bank state + post-processing is
    leaf-identical (and obs-identical) to a canonical reset under the same rng.
  * rollout_episode()             — one episode under the frozen contract
    (greedy_argmax policy_fn supplied by the caller), recording the frozen boundary
    events evaluator-side from the full state (never entering the observation).
Without JAX+craftax every one of these FAILS CLOSED (BLOCKED_ENVIRONMENT) and
evaluate() keeps consuming already-produced episode records (synthetic in tests). It
NEVER produces a scaffold result that claims full-task success (enforced downstream
by the certificate, NEG25).
"""
from __future__ import annotations

import os
import sys

# Runnable-as-script AND importable-as-package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_source_audit as audit            # noqa: E402
import tier3_event_predicates as pred         # noqa: E402
import tier3_state_serializer as ser          # noqa: E402
import tier3_scaffold_builder as builder      # noqa: E402
import tier3_metrics as metrics               # noqa: E402
import tier3_failure_taxonomy as taxonomy     # noqa: E402
import tier3_checkpoint_adapter as ckpt       # noqa: E402

SCHEMA = "mechanism_UED.tier3_evaluation_result/v1"
RESULT_VERSION = "tier3_evaluation_result/v1"

FULL = metrics.FULL
FRONT = metrics.FRONT
BACK = metrics.BACK

ACTION_MODE = "greedy_argmax"
MAX_TIMESTEPS = 4096

REQUIRED_EPISODE_KEYS = [
    "episode_id", "scenario", "valid_start", "terminal_label",
    "front_floor_transition_reached", "corridor_exit_reached", "defeat_kobold", "timesteps",
]

# achievement name -> floor threshold, exactly as MultiTaskMiniCraftaxEnv.reset_env
# sets the ENTER_* floor-entry achievements (minicraftax.envs.base).
FLOOR_ENTRY_ACHIEVEMENTS = [
    ("ENTER_DUNGEON", 1), ("ENTER_GNOMISH_MINES", 2), ("ENTER_SEWERS", 3),
    ("ENTER_VAULT", 4), ("ENTER_TROLL_MINES", 5), ("ENTER_FIRE_REALM", 6),
    ("ENTER_ICE_REALM", 7), ("ENTER_GRAVEYARD", 8),
]


class FailClosed(Exception):
    """Hard stop on any evaluation-contract violation."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


# ---------------------------------------------------------------------------
# Episode record validation (NEG19)
# ---------------------------------------------------------------------------
def validate_episode_record(ep: dict):
    """An episode MUST carry an explicit valid_start flag and required keys (NEG19)."""
    require(isinstance(ep, dict),
            "FAIL CLOSED (NEG19): episode is not a dict")
    missing = [k for k in REQUIRED_EPISODE_KEYS if k not in ep]
    require(not missing,
            "FAIL CLOSED (NEG19): episode record missing required key(s): %s" % sorted(missing))
    require("valid_start" in ep,
            "FAIL CLOSED (NEG19): episode record has no valid_start flag")
    require(isinstance(ep["valid_start"], bool),
            "FAIL CLOSED (NEG19): valid_start must be a bool")
    require(ep["scenario"] in (FULL, FRONT, BACK),
            "FAIL CLOSED: episode scenario %r unknown" % ep["scenario"])
    return True


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def frozen_contract():
    return {
        "action_mode": ACTION_MODE,
        "observation_schema": "canonical_craftax_symbolic",
        "action_space": "canonical_craftax_action_set",
        "max_timesteps": MAX_TIMESTEPS,
        "identical_for_all_arms": True,
    }


def evaluate(scenario: str, episodes: list, checkpoint_record: dict = None,
             checkpoint_record_after: dict = None):
    """Validate + classify + measure one scenario's episodes under the frozen contract.

    If a checkpoint record is supplied, its params SHA must be unchanged after the run
    (NEG23). Real rollouts are BLOCKED_ENVIRONMENT here; this operates on episode
    records.
    """
    require(scenario in (FULL, FRONT, BACK),
            "FAIL CLOSED: unknown scenario %r" % scenario)
    # NEG23: params must not be updated by the evaluation.
    if checkpoint_record is not None:
        after = checkpoint_record_after if checkpoint_record_after is not None else checkpoint_record
        ckpt.assert_evaluation_does_not_update_params(checkpoint_record, after)

    classified = []
    for ep in episodes:
        validate_episode_record(ep)                       # NEG19
        require(ep["scenario"] == scenario,
                "FAIL CLOSED: episode %r scenario %r != evaluation scenario %r"
                % (ep.get("episode_id"), ep.get("scenario"), scenario))
        cls = taxonomy.classify_episode(ep)               # NEG20 ambiguity fails closed
        rec = dict(ep)
        rec["classified_label"] = cls["label"]
        rec["failure_rule_version"] = cls["failure_rule_version"]
        classified.append(rec)

    summary = metrics.summarize(scenario, classified)
    label_counts = {}
    for rec in classified:
        label_counts[rec["classified_label"]] = label_counts.get(rec["classified_label"], 0) + 1

    return {
        "schema": SCHEMA,
        "result_version": RESULT_VERSION,
        "scenario": scenario,
        "contract": frozen_contract(),
        "episode_count": len(classified),
        "valid_start_count": sum(1 for r in classified if r["valid_start"]),
        "terminal_label_counts": label_counts,
        "metrics": summary,
        "failure_rule_version": taxonomy.FAILURE_RULE_VERSION,
        "checkpoint_params_sha256": (checkpoint_record or {}).get("params_sha256"),
        "materialization_status": ser.environment_status(),
        "rollout_status": ("BLOCKED_ENVIRONMENT" if not ser.have_jax_craftax()
                           else "REAL_ENV_INTERFACE_READY"),
        "scaffolded_results_can_replace_full_task": False,
    }


# ---------------------------------------------------------------------------
# REAL environment interface (JAX + craftax host; FAILS CLOSED otherwise)
# ---------------------------------------------------------------------------
_ENV_CACHE = {}


def make_canonical_env():
    """Build (and cache) the canonical evaluation environment:

    the SHA-verified canonical S4 task source is exec'd (NEG03 re-verified at call
    time), wrapped in MultiTaskMiniCraftaxEnv with embedding conditioning (size 67),
    and the observation shape / action count are asserted against the frozen audit
    bindings. Returns an entry dict consumed by reset_from_bank_state / rollout_episode.
    """
    if "entry" in _ENV_CACHE:
        return _ENV_CACHE["entry"]
    require(ser.have_jax_craftax(),
            "FAIL CLOSED (BLOCKED_ENVIRONMENT): canonical env requires JAX+craftax "
            "(jax=%s, craftax=%s)" % (ser.have_jax(), ser.have_craftax()))
    import jax.numpy as jnp
    from craftax.craftax.constants import Achievement
    from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
    from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv

    task_ident = builder.verify_canonical_task_source()     # NEG03 at call time
    path = audit.resolve_source_path("canonical_s4_task")
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    require(audit.sha256_file(path) == task_ident["expected_sha256"],
            "FAIL CLOSED (NEG03): canonical task source changed on disk after verification")
    ns = {}
    exec(compile(src, str(path), "exec"), ns)               # canonical task class only
    require("Env" in ns, "FAIL CLOSED: canonical s4_task_code.py defines no Env")
    s4_cls = ns["Env"]

    n_ach = len(Achievement)
    dk = int(Achievement.DEFEAT_KOBOLD.value)
    require(n_ach == audit.CRAFTAX_RUNTIME_BINDINGS["achievement_count"],
            "FAIL CLOSED: len(Achievement)=%d != frozen audit binding %d"
            % (n_ach, audit.CRAFTAX_RUNTIME_BINDINGS["achievement_count"]))
    require(dk == audit.CRAFTAX_RUNTIME_BINDINGS["defeat_kobold_achievement_index"],
            "FAIL CLOSED: DEFEAT_KOBOLD index %d != frozen audit binding" % dk)
    # Equivalent to dicode get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD]).
    table = jnp.zeros((1, n_ach), dtype=jnp.float32).at[0, dk].set(1.0)
    ctor = EnvParams(max_timesteps=MAX_TIMESTEPS)
    static_params = StaticEnvParams()
    envns = MultiTaskMiniCraftaxEnv([s4_cls], static_params, ctor, True,
                                    conditioning_type="embedding", embedding_size=n_ach)
    action_count = int(envns.action_space(ctor).n)
    observation_shape = tuple(int(x) for x in envns.observation_space(ctor).shape)
    require(action_count == audit.resolve_action_count()
            == audit.CRAFTAX_RUNTIME_BINDINGS["action_count"],
            "FAIL CLOSED: action space size %d != canonical binding %d"
            % (action_count, audit.CRAFTAX_RUNTIME_BINDINGS["action_count"]))
    entry = {
        "envns": envns, "ctor": ctor, "static_params": static_params,
        "task_embeddings": table, "defeat_kobold_index": dk,
        "achievement_count": n_ach,
        "action_count": action_count, "observation_shape": observation_shape,
        "task_identity": task_ident,
    }
    _ENV_CACHE["entry"] = entry
    return entry


def reset_from_bank_state(entry, bank_state, task_id: int = 0):
    """Inject a materialized bank state, replicating MultiTaskMiniCraftaxEnv.reset_env
    post-processing EXACTLY (task_params slice -> calculate_inventory_achievements ->
    floor-entry achievements -> task_id/achievements replace -> get_obs). Returns
    (obs, state)."""
    require(ser.have_jax_craftax(),
            "FAIL CLOSED (BLOCKED_ENVIRONMENT): reset_from_bank_state requires JAX+craftax")
    import jax
    from craftax.craftax.constants import Achievement
    from craftax.craftax.game_logic import calculate_inventory_achievements
    envns, table = entry["envns"], entry["task_embeddings"]
    task_params = jax.tree.map(lambda x: x[task_id], envns.stacked_task_params)
    state = calculate_inventory_achievements(bank_state)
    a = state.achievements
    for name, lvl in FLOOR_ENTRY_ACHIEVEMENTS:
        a = a.at[int(getattr(Achievement, name).value)].set(state.player_level >= lvl)
    state = state.replace(task_id=task_id, task_params=task_params, achievements=a)
    obs = envns.get_obs(state, table)
    return obs, state


def assert_front_reset_equivalence(entry, rng_seed: int = 0) -> dict:
    """Prove the FRONT scaffold start is the canonical reset state: under the same rng,
    materialize_start(FRONT, world_rng=split(rng)[1]) + reset_from_bank_state must be
    leaf-identical AND obs-identical to env.reset_env(rng)."""
    require(ser.have_jax_craftax(),
            "FAIL CLOSED (BLOCKED_ENVIRONMENT): equivalence check requires JAX+craftax")
    import jax
    import numpy as np
    envns, table, ctor = entry["envns"], entry["task_embeddings"], entry["ctor"]
    rng = jax.random.PRNGKey(int(rng_seed))
    obs_c, state_c = envns.reset_env(rng, ctor, 0, table)
    world_rng = jax.random.split(rng)[1]            # canonical reset: world_gen(split(rng)[1])
    bank = builder.materialize_start(FRONT, world_rng=world_rng)
    obs_f, state_f = reset_from_bank_state(entry, bank)
    td_c, td_f = jax.tree_util.tree_structure(state_c), jax.tree_util.tree_structure(state_f)
    require(td_c == td_f, "FAIL CLOSED: FRONT bank treedef != canonical reset treedef")
    leaves_c = [np.asarray(x) for x in jax.tree_util.tree_leaves(state_c)]
    leaves_f = [np.asarray(x) for x in jax.tree_util.tree_leaves(state_f)]
    bad = [i for i, (x, y) in enumerate(zip(leaves_c, leaves_f))
           if x.shape != y.shape or not np.array_equal(x, y)]
    require(not bad, "FAIL CLOSED: FRONT bank state leaves != canonical reset: %s" % bad[:8])
    obs_equal = bool(np.array_equal(np.asarray(obs_c), np.asarray(obs_f)))
    require(obs_equal, "FAIL CLOSED: FRONT bank observation != canonical reset observation")
    return {"leaves_equal": True, "leaves_count": len(leaves_c), "obs_equal": True,
            "observation_shape": tuple(int(x) for x in np.asarray(obs_c).shape),
            "action_count": entry["action_count"], "rng_seed": int(rng_seed)}


def _front_walkable_grid(start_state, view):
    """Evaluator-only STATIC walkable mask for the front floor: map[FRONT_FLOOR] cells
    whose BlockType is in the resolved craftax land-creature walkable set (SOLID_BLOCK
    / WATER / LAVA excluded, exactly as game_logic.move_player collides). Dynamic mob
    obstruction is deliberately excluded — the frozen dense metric is graph distance
    over map topology.

    LADDER_TILE_TRANSIT rule (explicit, auditable — the ONLY exception to the static
    BlockType mask): the floor-2 down_ladder (the corridor exit) and up_ladder tile
    positions from the normalized view are OR-ed into the mask. Inter-floor transit
    tiles are positions the player legally occupies while changing floors; treating
    them as valid progress nodes keeps graph-distance progress well-defined at the
    exact transition tiles regardless of their underlying BlockType. A ladder
    position off the map grid fails closed (broken start state)."""
    import numpy as np
    walk_values = {int(v) for v in audit.resolve_walkable_blocktype_values()}
    m = np.asarray(start_state.map)[audit.FRONT_FLOOR]
    rows = len(m)
    cols = len(m[0]) if rows else 0
    grid = [[bool(int(b) in walk_values) for b in row] for row in m]
    for key in ("down_ladders", "up_ladders"):
        pos = (view.get(key) or {}).get(audit.FRONT_FLOOR)
        if pos is None:
            continue
        r, c = int(pos[0]), int(pos[1])
        require(0 <= r < rows and 0 <= c < cols,
                "FAIL CLOSED: FRONT %s transit tile (%d,%d) is off-grid (%dx%d) — "
                "broken scaffold start" % (key, r, c, rows, cols))
        grid[r][c] = True                       # LADDER_TILE_TRANSIT
    return grid


def rollout_episode(entry, start_state, scenario, policy_fn, episode_id, rng_seed,
                    kobold_type_id=None, max_steps=None):
    """Roll out ONE episode under the frozen Tier3 contract; return an episode record
    carrying the frozen boundary events (computed evaluator-side from the FULL state —
    none of them enter the Student observation).

    policy_fn(obs, state) -> int action; the frozen contract is greedy_argmax, which
    the caller's policy_fn implements. `max_steps` only SHORTENS the cap (chain smoke);
    the frozen contract cap is MAX_TIMESTEPS. An invalid start is returned with
    valid_start=False and zero steps (classified INVALID_START by the taxonomy).
    """
    require(ser.have_jax_craftax(),
            "FAIL CLOSED (BLOCKED_ENVIRONMENT): rollout requires JAX+craftax")
    require(scenario in (FULL, FRONT, BACK), "FAIL CLOSED: unknown scenario %r" % scenario)
    import jax
    import numpy as np
    envns, table, ctor = entry["envns"], entry["task_embeddings"], entry["ctor"]
    dk = entry["defeat_kobold_index"]
    steps_cap = MAX_TIMESTEPS if max_steps is None else int(max_steps)
    require(1 <= steps_cap <= MAX_TIMESTEPS,
            "FAIL CLOSED: max_steps %d outside [1, %d]" % (steps_cap, MAX_TIMESTEPS))

    view = builder.normalize_envstate(start_state)
    if scenario == FRONT:
        view["floor2_up_ladder_removed"] = True
        valid_start = pred.valid_front_scaffold_start(view)
        kb = None
    elif scenario == BACK:
        kb = builder.resolve_kobold_type_id(kobold_type_id)
        valid_start = pred.valid_back_scaffold_start(view, kb)
    else:
        valid_start = pred.valid_full_start(view)
        kb = None

    rec = {
        "episode_id": str(episode_id), "scenario": scenario,
        "valid_start": bool(valid_start), "terminal_label": "",
        "front_floor_transition_reached": False, "corridor_exit_reached": False,
        "defeat_kobold": False, "player_died": False, "timed_out": False,
        "kobold_engaged": False, "timesteps": 0, "graph_distance_progress": None,
    }
    if not valid_start:
        return rec                                    # INVALID_START; zero steps

    obs, state = reset_from_bank_state(entry, start_state)
    start_level = int(np.asarray(state.player_level))
    start_pos = (int(np.asarray(state.player_position)[0]),
                 int(np.asarray(state.player_position)[1]))
    kb_baseline = {}
    if scenario == BACK:
        for m in view["mobs"]:
            if (m["category"] == pred.KOBOLD_CATEGORY and int(m["type_id"]) == kb
                    and m["mask"] and float(m["health"]) > 0):
                kb_baseline[m["level"]] = max(kb_baseline.get(m["level"], 0.0),
                                              float(m["health"]))
    walkable = exit_pos = None
    if scenario == FRONT:
        walkable = _front_walkable_grid(start_state, view)
        exit_pos = view["down_ladders"].get(audit.FRONT_FLOOR)

    rng = jax.random.PRNGKey(int(rng_seed))
    max_level, max_progress, steps = start_level, 0.0, 0
    defeated = died = alias_seen = engaged = env_done = False
    for _ in range(steps_cap):
        action = int(policy_fn(obs, state))
        rng, sk = jax.random.split(rng)
        obs, state, _rew, done, _info = envns.step_env(sk, state, action, ctor, table)
        steps += 1
        lvl = int(np.asarray(state.player_level))
        max_level = max(max_level, lvl)
        if scenario == FRONT and lvl >= audit.CORRIDOR_EXIT_FLOOR:
            alias_seen = True
        if bool(np.asarray(state.achievements)[dk]):
            defeated = True
        if float(np.asarray(state.player_health)) <= 0.0:
            died = True
        if scenario == BACK and not engaged:
            rb = state.ranged_mobs
            fl = lvl
            h = np.asarray(rb.health)[fl]; msk = np.asarray(rb.mask)[fl]
            tid = np.asarray(rb.type_id)[fl]; cd = np.asarray(rb.attack_cooldown)[fl]
            base = kb_baseline.get(fl)
            for slot in range(int(h.shape[0])):
                if msk[slot] and int(tid[slot]) == kb:
                    if base is not None and float(h[slot]) < base:
                        engaged = True
                    if base is not None and float(h[slot]) <= 0.0:
                        engaged = True
                    if int(cd[slot]) > 0:
                        engaged = True
        if (scenario == FRONT and exit_pos is not None
                and lvl == audit.FRONT_FLOOR and max_level < audit.CORRIDOR_EXIT_FLOOR):
            # FAIL CLOSED (no swallowing): NEG18 start->exit unreachable, off-grid or
            # non-walkable player position raise pred.FailClosed, which propagates
            # out of the rollout and aborts the evaluation — never silently skipped.
            # Computed ONLY while the player is on the front floor (a floor-1
            # excursion carries floor-1 coordinates that are meaningless on the
            # floor-2 grid), and STOPS permanently once the player has reached the
            # exit floor (floor 3): floor-2 graph distance is undefined there, so the
            # frozen dense metric freezes at the transition.
            pos = (int(np.asarray(state.player_position)[0]),
                   int(np.asarray(state.player_position)[1]))
            p = pred.normalized_corridor_progress({"player_position": pos},
                                                  walkable, start_pos, exit_pos)
            max_progress = max(max_progress, p)
        env_done = bool(np.asarray(done))
        if defeated or died or env_done:
            break
    timed_out = not (defeated or died) and (steps >= steps_cap or env_done)
    transition = (scenario == FRONT
                  and pred.front_floor_transition_reached(start_level, max_level))
    rec.update({
        "front_floor_transition_reached": bool(transition),
        "corridor_exit_reached": bool(alias_seen),
        "defeat_kobold": bool(defeated),
        "player_died": bool(died),
        "timed_out": bool(timed_out),
        "kobold_engaged": bool(engaged),
        "timesteps": steps,
        "graph_distance_progress": float(max_progress) if scenario == FRONT else None,
    })
    return rec


# ---------------------------------------------------------------------------
# REAL CLI — evaluation bound to a real CC2 checkpoint + frozen contract
# ---------------------------------------------------------------------------
# Deterministic canonical-reset rng seeds for the FULL interface smoke (FULL starts
# are canonical S4 resets; FRONT/BACK starts come from the frozen bank schedule).
FULL_SMOKE_SEED_BASE = 42
SCENARIO_ALIASES = {"front_l2": FRONT, "back_l2": BACK, "full": FULL, "all": "all"}

# Run classes (task §三/§六): smoke is chain verification only; the provisional
# selection evaluation is single-seed, provisional, and authorizes no scientific
# claim. FORMAL_EVALUATION remains certificate-side only (multi-seed; not this round).
RUN_CLASS_SMOKE = "INTERFACE_SMOKE"
RUN_CLASS_PROVISIONAL = "PROVISIONAL_STRONG_STUDENT_SELECTION"

# Frozen held-out FULL start seeds for the provisional selection evaluation (task §七):
# 64 canonical reset seeds 200000..200063 — never used by training, never overlapping
# the smoke base; FRONT/BACK use the frozen 8-state banks, each state exactly once.
PERF_FULL_SEED_BASE = 200_000
PERF_FULL_N = 64
PERF_FULL_SEEDS = [PERF_FULL_SEED_BASE + i for i in range(PERF_FULL_N)]


def _sha256_lf_file(path: str) -> str:
    """LF-normalized SHA256 of a source file (EOL-independent source identity)."""
    import hashlib
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def performance_start_schedule() -> dict:
    """The FROZEN provisional-selection start schedule (task §七). Pure; depends on
    no arm / checkpoint / result, and BOTH arms MUST consume the identical return
    value: FULL = 64 held-out canonical reset seeds 200000..200063; FRONT_L2 /
    BACK_L2 = all 8 frozen bank states, each exactly once (frozen bank schedule)."""
    import tier3_state_bank_materializer as mat
    front_seeds = mat.fixed_seed_schedule(FRONT, mat.FROZEN_BANK_N,
                                          mat.FROZEN_SEED_BASE, mat.FROZEN_SEED_STRIDE)
    back_seeds = mat.fixed_seed_schedule(BACK, mat.FROZEN_BANK_N,
                                         mat.FROZEN_SEED_BASE, mat.FROZEN_SEED_STRIDE)
    return {
        FULL: {"kind": "canonical_reset_seeds_held_out", "base": PERF_FULL_SEED_BASE,
               "count": PERF_FULL_N, "seeds": list(PERF_FULL_SEEDS)},
        FRONT: {"kind": "frozen_bank_state_each_once", "seed_base": mat.FROZEN_SEED_BASE,
                "stride": mat.FROZEN_SEED_STRIDE, "count": mat.FROZEN_BANK_N,
                "seeds": list(front_seeds)},
        BACK: {"kind": "frozen_bank_state_each_once", "seed_base": mat.FROZEN_SEED_BASE,
               "stride": mat.FROZEN_SEED_STRIDE, "count": mat.FROZEN_BANK_N,
               "seeds": list(back_seeds)},
    }


def state_entry_ids_for(scenario: str, seeds: list) -> list:
    """Stable, scenario-qualified entry ids for one scenario's seed list."""
    if scenario == FULL:
        return ["full-seed%d" % int(s) for s in seeds]
    return ["%s-bank%d" % (scenario, i) for i in range(len(seeds))]


def assert_output_dir_fresh(out_dir: str):
    """Output-freshness gate (task §四): the final output directory must not exist or
    must be empty. Refuses rm -rf, auto-overwrite, appending, auto-renaming — the
    operator must clear a non-empty directory manually, then re-run."""
    if os.path.exists(out_dir):
        require(os.path.isdir(out_dir),
                "FAIL CLOSED (EVALUATION_OUTPUT_DIRECTORY_NOT_FRESH): %r exists and is "
                "not a directory" % out_dir)
        entries = os.listdir(out_dir)
        require(not entries,
                "FAIL CLOSED (EVALUATION_OUTPUT_DIRECTORY_NOT_FRESH): output directory "
                "%r is not empty (%d entries, e.g. %s). Refusing to overwrite / append "
                "/ rename: clear it manually, then re-run."
                % (out_dir, len(entries), sorted(entries)[:5]))
    return True


def _runtime_versions() -> dict:
    """Actual runtime versions (task §五): Python / JAX / jaxlib / NumPy / Flax /
    Craftax. Any failure to determine a version fails closed."""
    import platform
    import importlib
    out = {"python_version": platform.python_version()}
    for key, modname in (("jax_version", "jax"), ("jaxlib_version", "jaxlib"),
                         ("numpy_version", "numpy"), ("flax_version", "flax"),
                         ("craftax_version", "craftax")):
        v = None
        try:
            v = getattr(importlib.import_module(modname), "__version__", None)
        except Exception:
            v = None
        if not v:
            try:
                import importlib.metadata as md
                v = md.version(modname)
            except Exception:
                v = None
        require(v, "FAIL CLOSED: cannot determine %s (module %r unavailable)"
                % (key, modname))
        out[key] = str(v)
    return out


def _git_commit_head() -> str:
    """The evaluator repo HEAD (40-hex) at run time — binds the certificate to the
    exact evaluated code revision. Fails closed if git is unavailable."""
    import subprocess
    root = str(audit.repo_root())
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              capture_output=True, text=True, timeout=60)
    except Exception as exc:
        raise FailClosed("FAIL CLOSED: cannot determine evaluator_git_commit: %r" % exc)
    require(proc.returncode == 0,
            "FAIL CLOSED: git rev-parse HEAD failed (rc=%d): %s"
            % (proc.returncode, (proc.stderr or "").strip()[:200]))
    sha = (proc.stdout or "").strip()
    require(len(sha) == 40 and all(c in "0123456789abcdef" for c in sha),
            "FAIL CLOSED: git rev-parse HEAD returned %r (not a 40-hex commit)" % sha)
    return sha


def run_evaluation(checkpoint_path: str, cc2_snapshot_root: str, scenario: str,
                   out_dir: str, run_class: str, contract_path: str = None,
                   arm: str = None, episodes: int = 2, max_steps: int = None,
                   driver_source: str = None) -> dict:
    """The REAL evaluation engine (task §一/§五/§六/§七), fail-closed end to end.

    Two run classes share ONE frozen contract (greedy_argmax, max_timesteps<=4096,
    canonical obs 8335 / 43 actions):

      * INTERFACE_SMOKE            — chain verification only (short caps);
      * PROVISIONAL_STRONG_STUDENT_SELECTION — the frozen provisional selection
        evaluation: FULL = 64 held-out canonical reset seeds 200000..200063,
        FRONT_L2/BACK_L2 = all 8 frozen bank states each exactly once, every episode
        under the full 4096-step cap. Both arms consume the IDENTICAL schedule.

    Gates, in order (each fail closed, each before any binding):
      0. anti-pollution hook gate (handover §7; fires before any JAX import)
      1. output-directory freshness (task §四: non-empty -> rejected; no rm -rf /
         overwrite / append / rename)
      2. checkpoint contract loaded + self-checksum-verified; the LOADED checkpoint
         (file SHA, recomputed params SHA, every manifest field) verified against the
         declared arm -> FINAL_98304_CHECKPOINT_CONTRACT_MISMATCH on any drift
      3. CC2 policy source / driver Cfg / NEG21 / env interface / frozen banks
      4. rollouts + NEG23 (params unchanged)

    The engine writes episode_records.jsonl / evaluation_result.json /
    evaluation_certificate.json with ENGINE-STAGE certificates (no exit provenance —
    task §二: the engine cannot know its own literal exit code). SHA256SUMS and the
    runner provenance are added ONLY by tier3_evaluation_runner.py after wait()-ing
    on this process and reading its literal exit code.
    """
    # 0. ANTI-POLLUTION GATE (handover §7): RMT16_POSTJAX_BINDING_SELFTEST=1 makes the
    # CC2 driver exit rc=0 BEFORE training (false success). Refuse to bind ANY
    # checkpoint while it is set — fail closed; never silently pop-and-continue.
    hook = os.environ.get("RMT16_POSTJAX_BINDING_SELFTEST", "")
    require(hook.strip() in ("", "0"),
            "FAIL CLOSED (anti-pollution): RMT16_POSTJAX_BINDING_SELFTEST=%r is set; "
            "this hook makes the CC2 driver exit rc=0 before training (false success). "
            "Unset it before any real checkpoint binding." % hook)
    require(run_class in (RUN_CLASS_SMOKE, RUN_CLASS_PROVISIONAL),
            "FAIL CLOSED: run_class %r not in (%s, %s)"
            % (run_class, RUN_CLASS_SMOKE, RUN_CLASS_PROVISIONAL))
    is_perf = run_class == RUN_CLASS_PROVISIONAL

    # 1. Output freshness (task §四).
    assert_output_dir_fresh(out_dir)

    import json
    import hashlib
    import numpy as np
    import jax
    import tier3_state_bank_materializer as mat
    import tier3_boundary_schema as bnd
    import tier3_cc2_policy_adapter as policy_adapter
    import tier3_evaluation_certificate as certmod
    import tier3_checkpoint_contract as contractmod

    require(ser.have_jax_craftax(),
            "FAIL CLOSED (BLOCKED_ENVIRONMENT): the real CLI requires JAX+craftax "
            "(jax=%s, craftax=%s)" % (ser.have_jax(), ser.have_craftax()))

    # 2. Frozen start schedule (task §七). Performance: the frozen held-out schedule;
    #    smoke: a short prefix of the same deterministic schedules.
    if is_perf:
        scenarios = [FULL, FRONT, BACK]
        schedule = performance_start_schedule()
        max_steps = MAX_TIMESTEPS
    else:
        require(scenario in SCENARIO_ALIASES,
                "FAIL CLOSED: --scenario %r not in %s"
                % (scenario, sorted(SCENARIO_ALIASES)))
        scenarios = [FULL, FRONT, BACK] if SCENARIO_ALIASES[scenario] == "all" \
            else [SCENARIO_ALIASES[scenario]]
        require(1 <= int(episodes) <= mat.FROZEN_BANK_N,
                "FAIL CLOSED: --episodes %r outside [1, %d]"
                % (episodes, mat.FROZEN_BANK_N))
        episodes = int(episodes)
        if max_steps is None:
            max_steps = 32
        require(1 <= int(max_steps) <= MAX_TIMESTEPS,
                "FAIL CLOSED: --max-steps %r outside [1, %d]" % (max_steps, MAX_TIMESTEPS))
        max_steps = int(max_steps)
        schedule = {}
        for sc in scenarios:
            seeds = ([FULL_SMOKE_SEED_BASE + i for i in range(episodes)] if sc == FULL
                     else mat.fixed_seed_schedule(sc, mat.FROZEN_BANK_N,
                                                  mat.FROZEN_SEED_BASE,
                                                  mat.FROZEN_SEED_STRIDE)[:episodes])
            schedule[sc] = {
                "kind": ("canonical_reset_seeds_smoke" if sc == FULL
                         else "frozen_bank_state_smoke"),
                "count": len(seeds), "seeds": [int(s) for s in seeds],
            }
    os.makedirs(out_dir, exist_ok=True)

    # 3. CC2 policy source identity (byte-bound) + real module import from the root.
    modules, src_id = policy_adapter.load_cc2_policy_modules(cc2_snapshot_root)
    # 3b. Frozen network hyperparameters from the SHA-bound driver SOURCE — real
    #     manifests carry config={} BY DESIGN (Cfg is a class-attributes class). The
    #     driver is AST-literal-parsed only: never executed, never guessed, never
    #     defaulted; its LF-SHA must equal the frozen value (fail closed).
    if driver_source is None:
        driver_source = policy_adapter.DEFAULT_DRIVER_SOURCE
    driver_cfg, driver_sha = policy_adapter.load_cfg_from_driver_source(
        driver_source, policy_adapter.FROZEN_DRIVER_FILE_SHA256)

    # 4. CC2 checkpoint (REAL full_state.pkl format; NEG21 verified inside) +
    #    frozen final-checkpoint CONTRACT verification of the LOADED bytes/manifest
    #    (task §一 — verify, not copy).
    require(contract_path is not None and arm is not None,
            "FAIL CLOSED: the evaluation CLI requires --checkpoint-contract <PATH> and "
            "--arm {persistent|reset128} (frozen final-98304 checkpoint contract)")
    contract = contractmod.load_contract(contract_path)
    params, params_sha, manifest, file_sha = ckpt.load_full_params_readonly(checkpoint_path)
    contract_verification = contractmod.verify_checkpoint_against_contract(
        arm, file_sha, params_sha, manifest, driver_sha,
        src_id["cc2_policy_source_sha256"], contract)

    # 5. Canonical evaluation environment + frozen interface assertion.
    entry = make_canonical_env()
    require(tuple(entry["observation_shape"]) == (8335,),
            "FAIL CLOSED: observation shape %s != frozen (8335,)"
            % (entry["observation_shape"],))
    require(int(entry["action_count"]) == 43,
            "FAIL CLOSED: action count %d != frozen 43" % entry["action_count"])
    # 6. Network + greedy policy built from the SHA-bound driver Cfg, cross-checked
    #    against the manifest (carry_mode is READ from the manifest, never chosen
    #    here; params stay read-only).
    network, rmt_cfg, carry_mode = policy_adapter.build_network_from_manifest(
        modules, manifest, entry["action_count"], driver_cfg)
    require(carry_mode == arm,
            "FAIL CLOSED (FINAL_98304_CHECKPOINT_CONTRACT_MISMATCH): loaded carry_mode "
            "%r != contract arm %r" % (carry_mode, arm))
    policy = policy_adapter.CC2RMT16Policy(
        modules, network, params, rmt_cfg, carry_mode,
        driver_cfg["window_mem"], driver_cfg["num_heads"],
        driver_cfg["num_layers"], driver_cfg["embed_size"])
    # 7. Re-verify BOTH frozen bank identities before any real evaluation.
    frozen_bindings = {sc: mat.verify_frozen_bank_identity(sc) for sc in (FRONT, BACK)}
    rec_before = ckpt.make_cc2_checkpoint_record(
        params, manifest, file_sha, entry["observation_shape"],
        "canonical_craftax_action_set",
        checkpoint_ref=checkpoint_path,      # invocation record; SHA is the identity
        driver_source_sha256=driver_sha)

    # 8. Rollouts (params read-only; fresh real RMT+GTrXL state per episode; the
    #    frozen schedule is consumed exactly — a scaffold state is never repeated to
    #    fake more samples).
    records_by_scenario, results_by_scenario = {}, {}
    for sc in scenarios:
        seeds = [int(s) for s in schedule[sc]["seeds"]]
        entry_ids = state_entry_ids_for(sc, seeds)
        eps = []
        for i, seed in enumerate(seeds):
            policy.reset()
            if sc == FULL:
                _obs0, start_state = entry["envns"].reset_env(
                    jax.random.PRNGKey(int(seed)), entry["ctor"], 0,
                    entry["task_embeddings"])
            else:
                start_state = builder.materialize_start(sc, int(seed))
            rec = rollout_episode(entry, start_state, sc, policy, entry_ids[i],
                                  int(seed), max_steps=max_steps)
            eps.append(rec)
            print("  [%s %d/%d %s seed=%d] steps=%d defeat=%s died=%s "
                  "transition=%s engaged=%s"
                  % (sc, i + 1, len(seeds), entry_ids[i], seed, rec["timesteps"],
                     rec["defeat_kobold"], rec["player_died"],
                     rec["front_floor_transition_reached"], rec["kobold_engaged"]),
                  flush=True)
        lines = [json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                 for r in eps]
        records_by_scenario[sc] = {
            "seeds": seeds,
            "episode_records": eps,
            "episode_records_sha256": hashlib.sha256(
                ("\n".join(lines) + "\n").encode("utf-8")).hexdigest(),
        }
        results_by_scenario[sc] = evaluate(sc, eps)

    # 9. NEG23: params byte-identical after every rollout.
    rec_after = dict(rec_before)
    rec_after["params_sha256"] = ckpt.cc2_params_sha256(params)
    ckpt.assert_evaluation_does_not_update_params(rec_before, rec_after)

    # 10. ENGINE-STAGE certificates with REAL value bindings (NEG27). NO exit
    #     provenance here (task §二): the parent runner finalizes these after
    #     wait()-ing on this child and reading its literal exit code.
    evaluator_sha = _sha256_lf_file(os.path.abspath(__file__))
    psha = bnd.predicate_code_sha256()
    require(psha == mat.FROZEN_PREDICATE_CODE_SHA256,
            "FAIL CLOSED: predicate code SHA %s != frozen %s"
            % (psha[:16], mat.FROZEN_PREDICATE_CODE_SHA256[:16]))
    versions = _runtime_versions()
    git_commit = _git_commit_head()
    entry_ids_by_scenario = {sc: state_entry_ids_for(
        sc, [int(s) for s in schedule[sc]["seeds"]]) for sc in scenarios}
    mode_label = "performance_evaluation" if is_perf else "interface_smoke"
    student_state = {
        "student_checkpoint_loaded": True,
        "student_policy_rollout_executed": True,
        "performance_evaluation_executed": bool(is_perf),
        "scientific_claim_authorized": False,
    }
    claims = (["PROVISIONAL_SELECTION_ONLY", "SINGLE_TRAINING_SEED",
               "NO_SCIENTIFIC_SUPERIORITY_CLAIM"] if is_perf
              else ["INTERFACE_SMOKE_ONLY"])
    certs = {}
    for sc in scenarios:
        scaffolded = sc in (FRONT, BACK)
        binding = {
            "bank_kind": "FROZEN_SCAFFOLD_BANK" if scaffolded else "CANONICAL_RESET_SEEDS",
            "state_bank_hash": (frozen_bindings[sc]["state_bank_hash"] if scaffolded
                                else frozen_bindings[FRONT]["canonical_task_sha256"]),
            "state_payload_hashes": (frozen_bindings[sc]["ordered_payload_hashes"]
                                     if scaffolded
                                     else [frozen_bindings[FRONT]["canonical_task_sha256"]]),
            "checkpoint_file_sha256": file_sha,
            "cc2_params_sha256": params_sha,
            "checkpoint_step": manifest.get("step"),
            "carry_mode": carry_mode,
            "run_class": run_class,
            "episode_records_sha256": records_by_scenario[sc]["episode_records_sha256"],
            "cc2_policy_source_sha256": src_id["cc2_policy_source_sha256"],
            "evaluator_source_sha256": evaluator_sha,
            "predicate_code_sha256": psha,
            "driver_source_sha256": driver_sha,
            "checkpoint_contract_sha256": contract["checkpoint_contract_sha256"],
            "checkpoint_contract_arm": arm,
            "action_mode": ACTION_MODE,
            "max_timesteps": int(max_steps),
            "evaluation_seed_schedule": schedule,
            "state_entry_ids": entry_ids_by_scenario,
            "python_version": versions["python_version"],
            "jax_version": versions["jax_version"],
            "jaxlib_version": versions["jaxlib_version"],
            "numpy_version": versions["numpy_version"],
            "flax_version": versions["flax_version"],
            "craftax_version": versions["craftax_version"],
            "evaluator_git_commit": git_commit,
            "scientific_claim_authorized": False,
            "single_training_seed": True,
            "provisional_selection_only": True,
            "observation_shape": list(entry["observation_shape"]),
            "action_dim": int(entry["action_count"]),
            "params_unchanged": True,
            "performance_claim_authorized": False,
        }
        label = ("FRONT_SCAFFOLD_STATE_BANK_HASH" if sc == FRONT
                 else "BACK_SCAFFOLD_STATE_BANK_HASH" if sc == BACK
                 else "CANONICAL_TASK_RESET_CONTRACT")
        certs[sc] = certmod.build_certificate(
            results_by_scenario[sc], state_bank_hash_label=label,
            claims=claims, has_real_rollout=True, student_state=student_state,
            mode=mode_label, eval_binding=binding, finalized=False)

    # 11. Deterministic artifacts. SHA256SUMS is written ONLY by the runner after it
    #     finalizes the certificates with the literal exit provenance (task §二).
    jl = os.path.join(out_dir, "episode_records.jsonl")
    with open(jl, "w", encoding="utf-8", newline="\n") as fh:
        for sc in scenarios:
            for r in records_by_scenario[sc]["episode_records"]:
                fh.write(json.dumps(r, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
    result_doc = {
        "schema": SCHEMA, "run_class": run_class,
        "performance_claim_authorized": False,
        "scientific_claim_authorized": False,
        "single_training_seed": True,
        "provisional_selection_only": True,
        "max_steps": max_steps,
        "evaluation_seed_schedule": schedule,
        "state_entry_ids": entry_ids_by_scenario,
        "checkpoint_contract": {
            "checkpoint_contract_sha256": contract["checkpoint_contract_sha256"],
            "arm": arm,
            "verified": contract_verification["verified"],
        },
        "runtime_versions": versions,
        "evaluator_git_commit": git_commit,
        "checkpoint": rec_before,
        "cc2_policy_source": src_id,
        "frozen_bank_bindings": frozen_bindings,
        "results": results_by_scenario,
        "episode_records_by_scenario": {
            sc: {"seeds": records_by_scenario[sc]["seeds"],
                 "episode_records_sha256": records_by_scenario[sc]["episode_records_sha256"]}
            for sc in scenarios},
    }
    with open(os.path.join(out_dir, "evaluation_result.json"), "w",
              encoding="utf-8", newline="\n") as fh:
        json.dump(result_doc, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    with open(os.path.join(out_dir, "evaluation_certificate.json"), "w",
              encoding="utf-8", newline="\n") as fh:
        json.dump(certs, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")

    for sc in scenarios:
        res = results_by_scenario[sc]
        print("  [%s] episodes=%d valid_start=%d/%d labels=%s"
              % (sc, res["episode_count"], res["valid_start_count"],
                 res["episode_count"], res["terminal_label_counts"]))
    return {"run_class": run_class, "arm": arm, "scenarios": scenarios,
            "out_dir": os.path.abspath(out_dir),
            "checkpoint_contract_sha256": contract["checkpoint_contract_sha256"],
            "params_unchanged": True, "performance_claim_authorized": False,
            "scientific_claim_authorized": False}


def run_interface_smoke(checkpoint_path: str, cc2_snapshot_root: str, scenario: str,
                        out_dir: str, episodes: int = 2, max_steps: int = 32,
                        driver_source: str = None, contract_path: str = None,
                        arm: str = None) -> dict:
    """REAL interface smoke (run_class=INTERFACE_SMOKE, chain verification only;
    performance_claim_authorized=False ALWAYS). Thin wrapper over run_evaluation."""
    return run_evaluation(checkpoint_path, cc2_snapshot_root, scenario, out_dir,
                          RUN_CLASS_SMOKE, contract_path=contract_path, arm=arm,
                          episodes=episodes, max_steps=max_steps,
                          driver_source=driver_source)


def run_performance_evaluation(checkpoint_path: str, cc2_snapshot_root: str,
                               out_dir: str, contract_path: str, arm: str,
                               driver_source: str = None) -> dict:
    """The frozen PROVISIONAL_STRONG_STUDENT_SELECTION evaluation (task §六): greedy
    argmax, max_timesteps=4096, FULL 64 held-out seeds 200000..200063 + FRONT/BACK
    all 8 frozen bank states each once, performance_evaluation_executed=true,
    scientific_claim_authorized=false, provisional_selection_only=true."""
    return run_evaluation(checkpoint_path, cc2_snapshot_root, "all", out_dir,
                          RUN_CLASS_PROVISIONAL, contract_path=contract_path, arm=arm,
                          driver_source=driver_source)


# ---------------------------------------------------------------------------
# Self-test (synthetic episodes; runs on this host).
# ---------------------------------------------------------------------------
def _ep(scenario, eid, valid_start, **flags):
    e = {"episode_id": eid, "scenario": scenario, "valid_start": valid_start,
         "terminal_label": "", "front_floor_transition_reached": False,
         "corridor_exit_reached": False, "defeat_kobold": False,
         "player_died": False, "timed_out": False, "timesteps": 10,
         "kobold_engaged": False, "graph_distance_progress": None}
    e.update(flags)
    return e


def self_test() -> int:
    problems = []

    def check(name, cond):
        if not cond:
            problems.append(name)

    # FULL evaluation over synthetic episodes.
    full_eps = [
        _ep(FULL, "f0", True, defeat_kobold=True, timesteps=900),
        _ep(FULL, "f1", True, player_died=True, timesteps=300),
        _ep(FULL, "f2", True, timed_out=True, timesteps=MAX_TIMESTEPS),
        _ep(FULL, "f3", False, timed_out=True, timesteps=MAX_TIMESTEPS),  # invalid start
    ]
    res = evaluate(FULL, full_eps)
    check("full_episode_count", res["episode_count"] == 4)
    check("full_valid_start_count", res["valid_start_count"] == 3)
    check("full_primary_value", abs(res["metrics"]["primary"]["value"] - 1 / 3) < 1e-9)
    check("full_labels_classified",
          res["terminal_label_counts"].get("SUCCESS_DEFEAT_KOBOLD") == 1
          and res["terminal_label_counts"].get("INVALID_START") == 1)

    # FRONT evaluation with dense progress (primary = floor transition 2 -> 3).
    front_eps = [
        _ep(FRONT, "r0", True, front_floor_transition_reached=True,
            corridor_exit_reached=True, timesteps=500, graph_distance_progress=1.0),
        _ep(FRONT, "r1", True, player_died=True, timesteps=200,
            graph_distance_progress=0.4),
    ]
    fres = evaluate(FRONT, front_eps)
    check("front_primary", fres["metrics"]["primary"]["value"] == 0.5)

    # NEG19: episode missing valid_start rejected.
    bad = _ep(FULL, "x", True)
    del bad["valid_start"]
    try:
        evaluate(FULL, [bad])
        check("NEG19_missing_valid_start_rejected", False)
    except (FailClosed, taxonomy.FailClosed, metrics.FailClosed):
        check("NEG19_missing_valid_start_rejected", True)

    # NEG19: episode missing a required key rejected.
    bad2 = _ep(FULL, "y", True, defeat_kobold=True)
    del bad2["episode_id"]
    try:
        validate_episode_record(bad2)
        check("NEG19_missing_required_key_rejected", False)
    except FailClosed:
        check("NEG19_missing_required_key_rejected", True)

    # NEG23: evaluation with unchanged checkpoint accepted; changed rejected.
    rec = ckpt.make_checkpoint_record({"w": [1, 2, 3]}, (67, 7, 7), "canonical_craftax_action_set")
    check("NEG23_unchanged_params_ok",
          evaluate(FULL, full_eps, checkpoint_record=rec)["checkpoint_params_sha256"]
          == rec["params_sha256"])
    mutated = dict(rec)
    mutated["params_sha256"] = "0" * 64
    try:
        evaluate(FULL, full_eps, checkpoint_record=rec, checkpoint_record_after=mutated)
        check("NEG23_changed_params_rejected", False)
    except (ckpt.FailClosed, FailClosed):
        check("NEG23_changed_params_rejected", True)

    # task §七 pure gates (any host): the frozen provisional start schedule is
    # IDENTICAL for both arms and reproduces the held-out seeds exactly; entry ids
    # derive deterministically from (scenario, seed).
    sched = performance_start_schedule()
    check("perf_schedule_full_frozen",
          sched[FULL]["seeds"] == [200000 + i for i in range(64)]
          and sched[FULL]["count"] == 64
          and sched[FULL]["kind"] == "canonical_reset_seeds_held_out")
    check("perf_schedule_front_frozen",
          sched[FRONT]["seeds"] == [10000 + i for i in range(8)]
          and sched[FRONT]["count"] == 8)
    check("perf_schedule_back_frozen",
          sched[BACK]["seeds"] == [1010000 + i for i in range(8)]
          and sched[BACK]["count"] == 8)
    check("perf_schedule_arm_identical",
          performance_start_schedule() == sched)      # one schedule, both arms
    check("entry_ids_full",
          state_entry_ids_for(FULL, [200000, 200001])
          == ["full-seed200000", "full-seed200001"])
    check("entry_ids_bank",
          state_entry_ids_for(FRONT, [10000]) == ["front_l2-bank0"])

    # task §四 pure gates (any host): output freshness — a missing or empty dir is
    # fresh; a non-empty dir or a file path is rejected (no rm -rf / overwrite).
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        fresh = os.path.join(td, "new_dir")
        assert_output_dir_fresh(fresh)                 # not-yet-existing -> fresh
        os.makedirs(fresh)
        assert_output_dir_fresh(fresh)                 # existing + empty -> fresh
        with open(os.path.join(fresh, "x.json"), "w") as fh:
            fh.write("{}")
        try:
            assert_output_dir_fresh(fresh)
            check("NEG39_nonempty_dir_rejected", False)
        except FailClosed:
            check("NEG39_nonempty_dir_rejected", True)
        fpath = os.path.join(td, "afile")
        with open(fpath, "w") as fh:
            fh.write("x")
        try:
            assert_output_dir_fresh(fpath)
            check("NEG39_path_is_file_rejected", False)
        except FailClosed:
            check("NEG39_path_is_file_rejected", True)

    # run_class gate (pure): only the two frozen run classes are accepted.
    try:
        run_evaluation("<none>", "<none>", "all", "<none>", "SMOKE_BUT_PERFORMANCE")
        check("bad_run_class_rejected", False)
    except FailClosed:
        check("bad_run_class_rejected", True)

    # ANTI-POLLUTION GATE (handover §7; pure, any host): run_interface_smoke must FAIL
    # CLOSED while RMT16_POSTJAX_BINDING_SELFTEST is set (the hook makes the CC2 driver
    # exit rc=0 before training = false success). The gate fires before any JAX import.
    _hook = "RMT16_POSTJAX_BINDING_SELFTEST"
    _prev = os.environ.get(_hook)
    os.environ[_hook] = "1"
    try:
        try:
            run_interface_smoke("<none>", "<none>", "all", "<none>")
            check("selftest_hook_gate_fails_closed", False)
        except FailClosed:
            check("selftest_hook_gate_fails_closed", True)
    finally:
        if _prev is None:
            os.environ.pop(_hook, None)
        else:
            os.environ[_hook] = _prev

    # REAL interface chain (JAX host only): canonical env + FRONT reset equivalence +
    # short NOOP rollouts through the record/classify/metrics chain (CHAIN check only).
    if ser.have_jax_craftax():
        from craftax.craftax.constants import Action
        entry = make_canonical_env()
        check("env_observation_shape_frozen",
              entry["observation_shape"] == (8335,))
        check("env_action_count_frozen",
              entry["action_count"] == audit.CRAFTAX_RUNTIME_BINDINGS["action_count"])
        eq = assert_front_reset_equivalence(entry, rng_seed=0)
        check("front_reset_equivalence", eq["leaves_equal"] and eq["obs_equal"])
        noop = int(Action.NOOP.value)
        policy = lambda obs, state: noop                      # noqa: E731
        front_bank = builder.materialize_start(FRONT, 0)
        rec_f = rollout_episode(entry, front_bank, FRONT, policy, "smoke-front", 7,
                                max_steps=16)
        check("front_rollout_valid_start", rec_f["valid_start"] is True)
        check("front_rollout_stepped", 1 <= rec_f["timesteps"] <= 16)
        back_bank = builder.materialize_start(BACK, 0)
        rec_b = rollout_episode(entry, back_bank, BACK, policy, "smoke-back", 7,
                                max_steps=16)
        check("back_rollout_valid_start", rec_b["valid_start"] is True)
        check("back_rollout_stepped", 1 <= rec_b["timesteps"] <= 16)
        # the chain: real records flow through validate -> classify -> metrics.
        chained = evaluate(BACK, [rec_b])
        check("back_rollout_classified",
              chained["terminal_label_counts"] != {} and chained["episode_count"] == 1)

    if problems:
        print("TIER3_EVALUATOR_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_EVALUATOR_SELF_TEST_PASS (contract frozen; NEG19/NEG23 guards live; rollout=%s)"
          % ("BLOCKED_ENVIRONMENT" if not ser.have_jax_craftax()
             else "REAL_ENV_INTERFACE_READY"))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # The real interface imports minicraftax (<repo>/dicode_src/src, audited relpaths).
    _src = str(audit.repo_root() / "dicode_src" / "src")
    if os.path.isdir(_src) and _src not in sys.path:
        sys.path.insert(0, _src)
    if "--self-test" in argv:
        return self_test()

    def _opt(flag, default=None):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                return argv[i + 1]
        return default

    # task §六: the two entry points are MUTUALLY EXCLUSIVE; neither / both -> usage.
    smoke = "--interface-smoke" in argv
    perf = "--performance-evaluation" in argv
    if smoke == perf:
        print("usage: tier3_evaluator.py --self-test\n"
              "       tier3_evaluator.py --interface-smoke --checkpoint <full_state.pkl> "
              "--checkpoint-contract <PATH> --arm {persistent|reset128} "
              "[--cc2_snapshot_root <PATH>] [--cc2_driver_source <PATH>] "
              "[--scenario {front_l2,back_l2,full,all}] --out <DIR> "
              "[--episodes N] [--max-steps M]\n"
              "       tier3_evaluator.py --performance-evaluation "
              "--checkpoint <full_state.pkl> --checkpoint-contract <PATH> "
              "--arm {persistent|reset128} [--cc2_snapshot_root <PATH>] "
              "[--cc2_driver_source <PATH>] --out <DIR>\n"
              "(--interface-smoke and --performance-evaluation are mutually exclusive)")
        return 3
    checkpoint = _opt("--checkpoint")
    contract = _opt("--checkpoint-contract")
    arm = _opt("--arm")
    out = _opt("--out")
    if not checkpoint or not contract or not arm or not out:
        print("FAIL CLOSED (usage): --checkpoint, --checkpoint-contract, --arm and "
              "--out are required for every real evaluation")
        return 3
    import tier3_cc2_policy_adapter as policy_adapter
    root = _opt("--cc2_snapshot_root", policy_adapter._default_snapshot_root())
    driver_src = _opt("--cc2_driver_source", policy_adapter.DEFAULT_DRIVER_SOURCE)
    if perf:
        summary = run_performance_evaluation(checkpoint, root, out, contract, arm,
                                             driver_source=driver_src)
        print("TIER3_PERFORMANCE_EVALUATION_DONE "
              "(run_class=PROVISIONAL_STRONG_STUDENT_SELECTION; arm=%s; "
              "checkpoint_contract_sha256=%s; action_mode=greedy_argmax; "
              "max_timesteps=4096; scientific_claim_authorized=false; "
              "provisional_selection_only=true; params_unchanged=%s; out=%s)"
              % (arm, summary["checkpoint_contract_sha256"],
                 summary["params_unchanged"], summary["out_dir"]))
        return 0
    summary = run_interface_smoke(
        checkpoint, root, _opt("--scenario", "all"), out,
        episodes=int(_opt("--episodes", "2")),
        max_steps=int(_opt("--max-steps", "32")),
        driver_source=driver_src, contract_path=contract, arm=arm)
    print("TIER3_INTERFACE_SMOKE_DONE (run_class=INTERFACE_SMOKE; arm=%s; "
          "performance_claim_authorized=false; params_unchanged=%s; out=%s)"
          % (arm, summary["params_unchanged"], summary["out_dir"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
