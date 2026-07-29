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


def _front_walkable_grid(start_state):
    """Evaluator-only STATIC walkable mask for the front floor: map[FRONT_FLOOR] cells
    whose BlockType is in the resolved craftax land-creature walkable set (SOLID_BLOCK
    / WATER / LAVA excluded, exactly as game_logic.move_player collides). Dynamic mob
    obstruction is deliberately excluded — the frozen dense metric is graph distance
    over map topology."""
    import numpy as np
    walk_values = {int(v) for v in audit.resolve_walkable_blocktype_values()}
    m = np.asarray(start_state.map)[audit.FRONT_FLOOR]
    return [[bool(int(b) in walk_values) for b in row] for row in m]


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
        walkable = _front_walkable_grid(start_state)
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
        if scenario == FRONT and exit_pos is not None:
            pos = (int(np.asarray(state.player_position)[0]),
                   int(np.asarray(state.player_position)[1]))
            try:
                p = pred.normalized_corridor_progress({"player_position": pos},
                                                      walkable, start_pos, exit_pos)
                max_progress = max(max_progress, p)
            except pred.FailClosed:
                pass      # transiently off the static walkable set (e.g. on a ladder)
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
    print("usage: tier3_evaluator.py --self-test")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
