"""Real-Craftax restore/parity checks (R4a env side, gates G0-G8).

Every craftax/minicraftax import happens inside functions so this module can
be imported even where craftax is missing; the driver then reports BLOCKED
with the real import error instead of a fake PASS.

Scope honesty: passing all checks here proves ONLY the R4a env-side restore
and dynamics parity.  It is not the R4c combined fresh-process proof.
"""

from __future__ import annotations

import base64
import copy
import dataclasses
import hashlib
import inspect
import json
import os
import platform
import sys
import time
from importlib.metadata import version as pkg_version
from typing import Any, Mapping

import numpy as np

from .dynamics_parity import compare_env_states, compare_flat_states, run_parity_rollout
from .env_restore import (
    build_template,
    encode_env_state,
    flatten_env_state,
    restore_env_state,
    slice_env_state,
    stack_env_states,
    unflatten_env_state,
)
from .errors import SchemaMismatchError
from .state_codec import StateCodec
from .terminal_events import TerminalEventAdapter


RESET_SEED = 20260803
RUNNER_SEED = 777
ACTION_SEED = 0


def _canonical_payload_hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _action_dim() -> int:
    from craftax.craftax.constants import Action
    return len(Action)


# ---------------------------------------------------------------------------
# G0 bootstrap
# ---------------------------------------------------------------------------

def bootstrap_environment() -> dict:
    """Version/device/upstream-semantics evidence bundle (G0)."""
    import jax
    import jaxlib
    import numpy
    import flax
    from craftax.craftax.constants import Action
    from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
    from craftax.craftax.envs.craftax_symbolic_env import (
        CraftaxSymbolicEnv,
        CraftaxSymbolicEnvNoAutoReset,
    )
    from craftax.environment_base.environment_bases import EnvironmentNoAutoReset
    from minicraftax.envs.base import MiniCraftaxTrain

    out: dict[str, Any] = {"pass": False}
    out["versions"] = {
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": numpy.__version__,
        "flax": flax.__version__,
        "craftax": pkg_version("craftax"),
        "gymnax": pkg_version("gymnax"),
        "platform": platform.platform(),
    }
    out["devices"] = [str(d) for d in jax.devices("cpu")]
    out["x64_enabled"] = bool(jax.config.x64_enabled)
    out["action_dim"] = int(len(Action))
    env_param_fields = sorted(f.name for f in dataclasses.fields(EnvParams))
    out["env_param_fields"] = env_param_fields
    out["env_params_has_max_timesteps"] = "max_timesteps" in env_param_fields
    out["env_params_has_fractal_noise_angles"] = "fractal_noise_angles" in env_param_fields
    out["minicraftax_train_is_noautoreset_subclass"] = bool(
        issubclass(MiniCraftaxTrain, CraftaxSymbolicEnvNoAutoReset))
    out["noautoreset_is_environment_noautoreset_subclass"] = bool(
        issubclass(CraftaxSymbolicEnvNoAutoReset, EnvironmentNoAutoReset))
    # Upstream semantics evidence (offline-proof of what step/reset actually do).
    out["upstream_source_evidence"] = {
        "EnvironmentNoAutoReset.step": inspect.getsource(EnvironmentNoAutoReset.step),
        "EnvironmentNoAutoReset.reset": inspect.getsource(EnvironmentNoAutoReset.reset),
        "MiniCraftaxTrain.step_env": inspect.getsource(MiniCraftaxTrain.step_env),
        "MiniCraftaxTrain.reset_env": inspect.getsource(MiniCraftaxTrain.reset_env),
    }
    # AutoResetEnvWrapper jits with params static: params must be hashable.
    probe = EnvParams(max_timesteps=8)
    out["env_params_hashable"] = True
    try:
        out["env_params_hash_sample"] = hash(probe)
    except TypeError as exc:
        out["env_params_hashable"] = False
        out["env_params_hash_error"] = str(exc)
    out["pass"] = bool(
        not out["x64_enabled"]
        and out["env_params_has_max_timesteps"]
        and out["env_params_has_fractal_noise_angles"]
        and out["minicraftax_train_is_noautoreset_subclass"]
        and out["noautoreset_is_environment_noautoreset_subclass"]
        and out["env_params_hashable"]
        and out["action_dim"] > 0
    )
    return out


# ---------------------------------------------------------------------------
# setup builder
# ---------------------------------------------------------------------------

def build_core_setup(*, max_timesteps: int, reset_seed: int = RESET_SEED) -> dict:
    """Build the real MiniCraftaxTrain(survive.Env) setup.

    The SAME EnvParams object is passed to task construction and stepping:
    BaseTask.is_terminal reads the construction-time params.max_timesteps.
    """
    import jax
    from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
    from minicraftax.envs.base import MiniCraftaxTrain
    from minicraftax.tasks.seed_tasks import survive

    params = EnvParams(max_timesteps=max_timesteps)
    static_params = StaticEnvParams()
    task = survive.Env(static_params, params)
    env = MiniCraftaxTrain(task, static_env_params=static_params)
    obs0, state0 = env.reset_env(jax.random.PRNGKey(reset_seed), params)
    step_fn = jax.jit(env.step_env)
    return {"env": env, "task": task, "params": params, "static_params": static_params,
            "obs0": obs0, "state0": state0, "step_fn": step_fn,
            "max_timesteps": max_timesteps, "reset_seed": reset_seed}


def _step_loop(setup: dict, state: Any, actions, key):
    import jax
    step_fn = setup["step_fn"]
    params = setup["params"]
    prev = None
    for action in actions:
        key, step_key = jax.random.split(key)
        prev = state
        _, state, _, _, _ = step_fn(step_key, state, int(action), params)
    return state, prev, key


# ---------------------------------------------------------------------------
# G1 restore round-trip
# ---------------------------------------------------------------------------

def check_restore_roundtrip(setup: dict) -> dict:
    import jax
    env, params = setup["env"], setup["params"]
    state0 = setup["state0"]
    # Template from a DIFFERENT reset of the same lineage (values differ).
    _, state_ref = env.reset_env(jax.random.PRNGKey(setup["reset_seed"] + 1), params)
    template = build_template(state_ref)
    encoded, bundle = encode_env_state(state0, next_step_key=jax.random.PRNGKey(RUNNER_SEED),
                                       previous_action=0, previous_reward=0.0)
    restored = restore_env_state(encoded, template)
    cmp = compare_env_states(state0, restored.env_state)
    encoded_again, _ = encode_env_state(state0, next_step_key=jax.random.PRNGKey(RUNNER_SEED),
                                        previous_action=0, previous_reward=0.0)
    hash_stable = encoded_again.payload_hash == encoded.payload_hash
    none_paths = [p for p in template.leaf_paths if p.startswith("fractal_noise_angles")]
    none_ok = len(none_paths) == 4 and restored.env_state.fractal_noise_angles == (None, None, None, None)
    return {"pass": bool(cmp["ok"] and hash_stable and none_ok),
            "n_leaves": cmp["n_leaves"], "payload_hash_stable": bool(hash_stable),
            "none_leaf_paths": none_paths, "none_preserved": bool(none_ok),
            "template_env_state_type": template.env_state_type,
            "leaves": cmp["leaves"]}


# ---------------------------------------------------------------------------
# G2 dynamics parity
# ---------------------------------------------------------------------------

def check_dynamics_parity(setup: dict, *, k_steps: int = 40, restore_at: int = 13,
                          seed: int = RUNNER_SEED) -> dict:
    import jax
    act_dim = _action_dim()
    actions = np.random.default_rng(ACTION_SEED).integers(0, act_dim, size=restore_at + k_steps)
    key = jax.random.PRNGKey(seed)
    state, prev_state, key = _step_loop(setup, setup["state0"], actions[:restore_at], key)
    if prev_state is None:
        raise SchemaMismatchError("parity setup needs restore_at >= 1")
    template = build_template(prev_state)  # same jit-stepped lineage, different values
    captured = state
    encoded, _ = encode_env_state(captured, next_step_key=key,
                                  previous_action=int(actions[restore_at - 1]), previous_reward=0.0)
    restored = restore_env_state(encoded, template)
    cmp_at_capture = compare_env_states(captured, restored.env_state)

    params = setup["params"]
    step_fn = setup["step_fn"]

    def parity_step(k, s, a):
        return step_fn(k, s, a, params)

    report, _, _ = run_parity_rollout(parity_step, captured, restored.env_state,
                                      actions=list(actions[restore_at:]), key=key)
    return {"pass": bool(cmp_at_capture["ok"] and report["ok"]),
            "k_steps": k_steps, "restore_at": restore_at,
            "capture_restore_ok": bool(cmp_at_capture["ok"]),
            "first_divergence": report["first_divergence"],
            "steps": report["steps"]}


# ---------------------------------------------------------------------------
# G3 terminal restore
# ---------------------------------------------------------------------------

def check_terminal_restore(*, seed: int = RESET_SEED, max_timesteps: int = 12) -> dict:
    import jax
    setup = build_core_setup(max_timesteps=max_timesteps, reset_seed=seed)
    act_dim = _action_dim()
    actions = np.random.default_rng(ACTION_SEED).integers(0, act_dim, size=max_timesteps)
    key = jax.random.PRNGKey(RUNNER_SEED)
    step_fn = setup["step_fn"]
    params = setup["params"]
    state = setup["state0"]
    prev_state = None
    terminal_done = False
    terminal_step = None
    terminal_state = None
    for i in range(max_timesteps):
        key, step_key = jax.random.split(key)
        prev_state = state
        _, state, _, done, _ = step_fn(step_key, state, int(actions[i]), params)
        if bool(done):
            terminal_done = True
            terminal_step = i
            terminal_state = state
            break
    if not terminal_done:
        return {"pass": False, "terminal_done": False, "reason": "no terminal reached"}
    if prev_state is None:
        return {"pass": False, "terminal_done": True, "reason": "terminal at step 0; no same-lineage template reference"}
    template = build_template(prev_state)  # same jit-stepped lineage
    encoded, _ = encode_env_state(terminal_state, next_step_key=key,
                                  previous_action=int(actions[terminal_step]), previous_reward=0.0)
    restored = restore_env_state(encoded, template)
    terminal_cmp = compare_env_states(terminal_state, restored.env_state)
    # One post-terminal step from both tracks with the SAME step key (parity
    # convention: identical key stream, never re-split an already-consumed key).
    key, k1 = jax.random.split(key)
    _, next_a, _, done_a, _ = step_fn(k1, terminal_state, 0, params)
    _, next_b, _, done_b, _ = step_fn(k1, restored.env_state, 0, params)
    post_cmp = compare_env_states(next_a, next_b)
    return {"pass": bool(terminal_cmp["ok"] and post_cmp["ok"] and done_a is not None
                         and bool(np.asarray(done_a)) == bool(np.asarray(done_b))),
            "terminal_done": True, "terminal_step": int(terminal_step),
            "max_timesteps": max_timesteps,
            "terminal_restore_ok": bool(terminal_cmp["ok"]),
            "post_terminal_parity": bool(post_cmp["ok"]),
            "post_terminal_done_equal": bool(bool(np.asarray(done_a)) == bool(np.asarray(done_b)))}


# ---------------------------------------------------------------------------
# G4 autoreset evidence chain
# ---------------------------------------------------------------------------

def check_autoreset_evidence(*, seed: int = RESET_SEED, max_timesteps: int = 8) -> dict:
    import jax
    from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
    from dicode.wrappers import AutoResetEnvWrapper
    from minicraftax.envs.base import MiniCraftaxTrain
    from minicraftax.tasks.seed_tasks import survive

    params = EnvParams(max_timesteps=max_timesteps)
    static_params = StaticEnvParams()
    task = survive.Env(static_params, params)
    env = MiniCraftaxTrain(task, static_env_params=static_params)
    wrapped = AutoResetEnvWrapper(env)

    obs, state = wrapped.reset(jax.random.PRNGKey(seed), params)
    rng = jax.random.PRNGKey(seed + 1)
    captured = None
    for _ in range(max_timesteps + 2):
        rng, rng_t = jax.random.split(rng)
        obs_ret, state_ret, reward, done, info = wrapped.step(rng_t, state, 0, params)
        if bool(done):
            captured = (rng_t, state, obs_ret, state_ret, reward)
            break
        state = state_ret
    if captured is None:
        return {"pass": False, "done_detected": False, "reason": "no done within budget"}
    rng_t, state_before, obs_ret, state_ret, reward = captured

    # Deterministic replay of the wrapper's internal key tree.  The replay goes
    # through jit exactly like the wrapper internals do, so leaf lineages
    # (arrays vs python scalars) match the wrapper's jit outputs.
    replay_step = jax.jit(env.step_env)
    replay_reset = jax.jit(env.reset_env)
    c0, c1 = jax.random.split(rng_t)
    obs_st, state_st, r_replay, d_replay, _ = replay_step(c1, state_before, 0, params)
    g0, g1 = jax.random.split(c0)
    obs_re, state_re = replay_reset(g1, params)
    source_state = state_re if bool(d_replay) else state_st
    source_obs = obs_re if bool(d_replay) else obs_st
    replay_faithful = bool(
        compare_env_states(state_ret, source_state)["ok"]
        and np.array_equal(np.asarray(obs_ret), np.asarray(source_obs))
        and float(np.asarray(r_replay)) == float(np.asarray(reward)))
    returned_differs_from_terminal = not compare_env_states(state_ret, state_st)["ok"]

    adapter = TerminalEventAdapter()
    flat_before = flatten_env_state(state_before)
    flat_ret = flatten_env_state(state_ret)
    flat_st = flatten_env_state(state_st)
    flat_re = flatten_env_state(state_re)
    negative_raised = False
    try:
        adapter.adapt(previous_state=flat_before, action_metadata=0, reward=float(np.asarray(reward)),
                      done=True, returned_state=flat_ret, reset_state=flat_re, terminal_state=None, info={})
    except ValueError:
        negative_raised = True
    transition = adapter.adapt(previous_state=flat_before, action_metadata=0,
                               reward=float(np.asarray(reward)), done=True,
                               returned_state=flat_ret, reset_state=flat_re,
                               terminal_state=flat_st, info={})
    goal = adapter.goal_state(transition)
    goal_is_terminal = bool(compare_flat_states(goal, flat_st)["ok"])

    # Terminal evidence must itself be restorable through the codec.
    _, ref_state = env.reset_env(jax.random.PRNGKey(seed + 2), params)
    ref_key = jax.random.PRNGKey(seed + 3)
    ref_key, rk = jax.random.split(ref_key)
    _, ref_stepped, _, _, _ = replay_step(rk, ref_state, 0, params)
    template = build_template(ref_stepped)  # jit/stepped lineage, same as state_st
    encoded, _ = encode_env_state(state_st, next_step_key=rng, previous_action=0, previous_reward=0.0)
    restored = restore_env_state(encoded, template)
    terminal_restorable = bool(compare_env_states(state_st, restored.env_state)["ok"])

    return {"pass": bool(replay_faithful and returned_differs_from_terminal and negative_raised
                         and goal_is_terminal and terminal_restorable),
            "done_detected": True, "replay_faithful": replay_faithful,
            "returned_differs_from_terminal": bool(returned_differs_from_terminal),
            "adapter_negative_raised": bool(negative_raised),
            "adapter_positive_ok": True,
            "goal_state_is_terminal": goal_is_terminal,
            "terminal_restorable": terminal_restorable}


# ---------------------------------------------------------------------------
# G5 corrupted payload
# ---------------------------------------------------------------------------

def _encoded_leaf_items(payload: Mapping[str, Any]) -> dict:
    env_enc = payload["env_state"]
    top = {k: v for k, v in env_enc["items"]}
    leaves_enc = top["leaves"]
    return {k: v for k, v in leaves_enc["items"]}


def _flip_b64_changes_bytes(data: str) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    original = base64.b64decode(data, validate=True)
    chars = list(data)
    for idx in range(len(chars)):
        for alt in alphabet:
            if alt == chars[idx]:
                continue
            cand = chars.copy()
            cand[idx] = alt
            candidate = "".join(cand)
            try:
                if base64.b64decode(candidate, validate=True) != original:
                    return candidate
            except Exception:
                continue
    raise SchemaMismatchError("could not construct a byte-changing base64 flip")


def check_corrupted_payload(setup: dict) -> dict:
    import jax
    state0 = setup["state0"]
    _, state_ref = setup["env"].reset_env(jax.random.PRNGKey(setup["reset_seed"] + 5), setup["params"])
    template = build_template(state_ref)
    encoded, _ = encode_env_state(state0, next_step_key=jax.random.PRNGKey(RUNNER_SEED),
                                  previous_action=0, previous_reward=0.0)
    codec = StateCodec()
    cases = []

    def expect_schema_error(case: str, fn) -> None:
        try:
            fn()
            cases.append({"case": case, "raised": None, "expected": "SchemaMismatchError", "ok": False})
        except SchemaMismatchError:
            cases.append({"case": case, "raised": "SchemaMismatchError",
                          "expected": "SchemaMismatchError", "ok": True})
        except Exception as exc:  # honest record of unexpected exception types
            cases.append({"case": case, "raised": type(exc).__name__,
                          "expected": "SchemaMismatchError", "ok": False})

    # c1: hash tamper without payload change
    expect_schema_error("c1_hash_tamper", lambda: codec.decode(
        dataclasses.replace(encoded, payload_hash="0" * 64)))

    # pick an array leaf with >= 2 elements for c2-c4
    leaf_items = _encoded_leaf_items(encoded.payload)
    target_path = None
    for path, enc in leaf_items.items():
        if enc.get("kind") == "array" and int(np.prod(enc["shape"], dtype=np.int64)) >= 2:
            target_path = path
            break
    if target_path is None:
        cases.append({"case": "c2_array_leaf_available", "raised": None,
                      "expected": "array leaf with >=2 elements", "ok": False})
    else:
        # c2: base64 bit-flip WITHOUT hash recompute -> hash gate catches it
        payload_c2 = copy.deepcopy(encoded.payload)
        items_c2 = _encoded_leaf_items(payload_c2)
        items_c2[target_path]["data"] = _flip_b64_changes_bytes(items_c2[target_path]["data"])
        expect_schema_error("c2_bitflip_no_rehash", lambda: codec.decode(
            dataclasses.replace(encoded, payload=payload_c2)))

        # c3: bit-flip WITH hash recompute -> decode passes; leaf parity must catch it
        payload_c3 = copy.deepcopy(encoded.payload)
        items_c3 = _encoded_leaf_items(payload_c3)
        items_c3[target_path]["data"] = _flip_b64_changes_bytes(items_c3[target_path]["data"])
        encoded_c3 = dataclasses.replace(encoded, payload=payload_c3,
                                         payload_hash=_canonical_payload_hash(payload_c3))
        bundle_c3 = codec.decode(encoded_c3)
        cmp_c3 = compare_flat_states(bundle_c3.env_state, flatten_env_state(state0))
        cases.append({"case": "c3_bitflip_with_rehash_detected_by_parity",
                      "raised": None, "expected": "parity detects value corruption",
                      "ok": bool(not cmp_c3["ok"] and target_path in cmp_c3["mismatched"])})

        # c4: truncate array payload (keep dtype multiple) WITH rehash -> size mismatch
        payload_c4 = copy.deepcopy(encoded.payload)
        items_c4 = _encoded_leaf_items(payload_c4)
        enc4 = items_c4[target_path]
        raw = base64.b64decode(enc4["data"], validate=True)
        itemsize = int(np.dtype(enc4["dtype"]).itemsize)
        enc4["data"] = base64.b64encode(raw[: len(raw) - itemsize]).decode("ascii")
        encoded_c4 = dataclasses.replace(encoded, payload=payload_c4,
                                         payload_hash=_canonical_payload_hash(payload_c4))
        expect_schema_error("c4_truncated_payload_with_rehash", lambda: codec.decode(encoded_c4))

    # c5: flat-dict tamper before unflatten (clean decode first)
    bundle_clean = codec.decode(encoded)
    flat0 = bundle_clean.env_state

    def tamper(mutator) -> Any:
        flat = copy.deepcopy(flat0)
        mutator(flat)
        return unflatten_env_state(flat, template)

    array_path = None
    for path, value in flat0["leaves"].items():
        if isinstance(value, np.ndarray) and value.size >= 2:
            array_path = path
            break
    none_path = next(p for p in flat0["leaf_paths"] if p.startswith("fractal_noise_angles"))

    expect_schema_error("c5a_drop_leaf", lambda: tamper(
        lambda f: (f["leaves"].pop(array_path), f["leaf_paths"].remove(array_path))))
    expect_schema_error("c5b_add_leaf", lambda: tamper(
        lambda f: (f["leaves"].__setitem__("zz_injected_leaf", 1), f["leaf_paths"].append("zz_injected_leaf"))))
    expect_schema_error("c5c_wrong_shape", lambda: tamper(
        lambda f: f["leaves"].__setitem__(array_path, np.zeros(tuple(np.asarray(f["leaves"][array_path]).shape + (2,)),
                                                               dtype=np.asarray(f["leaves"][array_path]).dtype))))
    def _other_dtype(arr: np.ndarray) -> str:
        return "int64" if arr.dtype != np.dtype("int64") else "float64"

    expect_schema_error("c5d_wrong_dtype", lambda: tamper(
        lambda f: f["leaves"].__setitem__(
            array_path, np.asarray(f["leaves"][array_path]).astype(_other_dtype(np.asarray(f["leaves"][array_path]))))))
    expect_schema_error("c5e_none_replaced_by_array", lambda: tamper(
        lambda f: f["leaves"].__setitem__(none_path, np.zeros(1, dtype="int32"))))

    return {"pass": bool(all(c["ok"] for c in cases)), "cases": cases}


# ---------------------------------------------------------------------------
# G6 version mismatch
# ---------------------------------------------------------------------------

def check_version_mismatch(setup: dict) -> dict:
    import jax
    state0 = setup["state0"]
    _, state_ref = setup["env"].reset_env(jax.random.PRNGKey(setup["reset_seed"] + 7), setup["params"])
    template = build_template(state_ref)
    encoded, _ = encode_env_state(state0, next_step_key=jax.random.PRNGKey(RUNNER_SEED),
                                  previous_action=0, previous_reward=0.0)
    codec = StateCodec()
    cases = []

    def expect_schema_error(case: str, fn) -> None:
        try:
            fn()
            cases.append({"case": case, "raised": None, "expected": "SchemaMismatchError", "ok": False})
        except SchemaMismatchError:
            cases.append({"case": case, "raised": "SchemaMismatchError",
                          "expected": "SchemaMismatchError", "ok": True})
        except Exception as exc:
            cases.append({"case": case, "raised": type(exc).__name__,
                          "expected": "SchemaMismatchError", "ok": False})

    # v1: schema_version swap
    expect_schema_error("v1_schema_version_swap", lambda: codec.decode(
        dataclasses.replace(encoded, schema_version="simulator_frontier.state/v0")))

    # v2: field-set tamper (drop history_reference everywhere) + rehash
    payload_v2 = copy.deepcopy(encoded.payload)
    payload_v2.pop("history_reference", None)
    meta_v2 = copy.deepcopy(dict(encoded.scalar_metadata))
    meta_v2["field_names"] = [n for n in meta_v2["field_names"] if n != "history_reference"]
    expect_schema_error("v2_field_set_tamper", lambda: codec.decode(
        dataclasses.replace(encoded, payload=payload_v2, scalar_metadata=meta_v2,
                            payload_hash=_canonical_payload_hash(payload_v2))))

    def rewrite_flat_field(payload: Mapping[str, Any], key: str, value: Any) -> Any:
        env_enc = payload["env_state"]
        for pair in env_enc["items"]:
            if pair[0] == key:
                pair[1]["value"] = value
                return payload
        raise SchemaMismatchError(f"flat field {key} not found in encoded payload")

    # v3: treedef fingerprint drift (self-consistent re-encode)
    payload_v3 = copy.deepcopy(encoded.payload)
    rewrite_flat_field(payload_v3, "treedef_fingerprint", "f" * 64)
    encoded_v3 = dataclasses.replace(encoded, payload=payload_v3,
                                     payload_hash=_canonical_payload_hash(payload_v3))
    expect_schema_error("v3_treedef_fingerprint_drift",
                        lambda: restore_env_state(encoded_v3, template, codec=codec))

    # v4: env_state_type drift
    payload_v4 = copy.deepcopy(encoded.payload)
    rewrite_flat_field(payload_v4, "env_state_type", "other_module.OldEnvState")
    encoded_v4 = dataclasses.replace(encoded, payload=payload_v4,
                                     payload_hash=_canonical_payload_hash(payload_v4))
    expect_schema_error("v4_env_state_type_drift",
                        lambda: restore_env_state(encoded_v4, template, codec=codec))

    note = ("codec_version is not validated by the v1 codec by design; structural drift "
            "defense lives in treedef_fingerprint. Recorded as fact, not a case.")
    return {"pass": bool(all(c["ok"] for c in cases)), "cases": cases, "note": note}


# ---------------------------------------------------------------------------
# G7 batch slice/restore/stack parity
# ---------------------------------------------------------------------------

def check_batch_parity(setup: dict, *, batch: int = 2, slice_index: int = 1,
                       steps: int = 12, seed: int = RUNNER_SEED) -> dict:
    import jax
    import jax.numpy as jnp
    if batch != 2:
        raise SchemaMismatchError("batch parity check is specified for batch=2")
    env, params = setup["env"], setup["params"]
    act_dim = _action_dim()

    def reset_with(s: int):
        return env.reset_env(jax.random.PRNGKey(s), params)[1]

    st_a = reset_with(seed)
    st_b = reset_with(seed + 101)
    st_c = reset_with(seed + 303)
    st_d = reset_with(seed + 404)
    companion = reset_with(seed + 900)

    vmap_step = jax.jit(jax.vmap(env.step_env, in_axes=(0, 0, 0, None)))

    def advance(batch_state, actions, key_seed: int):
        stream = jax.random.PRNGKey(key_seed)
        s = batch_state
        for action in actions:
            stream, k = jax.random.split(stream)
            keys = jax.random.split(k, batch)
            # vmap in_axes=(0,0,0,None): actions are per-env, same value here
            _, s, _, _, _ = vmap_step(keys, s, jnp.full((batch,), int(action)), params)
        return s

    warmup_actions = np.random.default_rng(ACTION_SEED).integers(0, act_dim, size=3)
    batch_orig = advance(stack_env_states([st_a, st_b]), warmup_actions, seed + 7000)
    batch_ref = advance(stack_env_states([st_c, st_d]), warmup_actions, seed + 7000)

    captured = slice_env_state(batch_orig, slice_index)
    template = build_template(slice_env_state(batch_ref, slice_index))  # sliced lineage
    stream = jax.random.PRNGKey(seed + 7500)
    encoded, _ = encode_env_state(captured, next_step_key=stream, previous_action=0, previous_reward=0.0)
    restored = restore_env_state(encoded, template).env_state
    capture_cmp = compare_env_states(captured, restored)

    # Two independent tracks; original and restored never share a batch.
    track_a = stack_env_states([companion, captured]) if slice_index == 1 else stack_env_states([captured, companion])
    track_b = stack_env_states([companion, restored]) if slice_index == 1 else stack_env_states([restored, companion])
    actions = np.random.default_rng(ACTION_SEED + 1).integers(0, act_dim, size=steps)
    s_a, s_b = track_a, track_b
    step_records = []
    first_divergence = None
    for t in range(steps):
        keys = jax.random.split(jax.random.PRNGKey(seed + 8000 + t), batch)
        action = int(actions[t])
        action_vec = jnp.full((batch,), action)
        obs_a, s_a, r_a, d_a, _ = vmap_step(keys, s_a, action_vec, params)
        obs_b, s_b, r_b, d_b, _ = vmap_step(keys, s_b, action_vec, params)
        obs_equal = bool(np.array_equal(np.asarray(obs_a), np.asarray(obs_b)))
        reward_equal = bool(np.array_equal(np.asarray(r_a), np.asarray(r_b)))
        done_equal = bool(np.array_equal(np.asarray(d_a), np.asarray(d_b)))
        slice_cmp = compare_env_states(slice_env_state(s_a, slice_index),
                                       slice_env_state(s_b, slice_index))
        step_ok = obs_equal and reward_equal and done_equal and slice_cmp["ok"]
        step_records.append({"step": t, "action": action, "obs_equal": obs_equal,
                             "reward_equal": reward_equal, "done_equal": done_equal,
                             "sliced_state_ok": slice_cmp["ok"]})
        if not step_ok and first_divergence is None:
            first_divergence = {"step": t, "kind": "batch_step"}
    return {"pass": bool(capture_cmp["ok"] and first_divergence is None),
            "batch": batch, "slice_index": slice_index, "steps": steps,
            "capture_restore_ok": bool(capture_cmp["ok"]),
            "first_divergence": first_divergence, "steps_detail": step_records}


# ---------------------------------------------------------------------------
# secondary: multitask task_id=0
# ---------------------------------------------------------------------------

def check_multitask_secondary(*, seed: int = RESET_SEED, steps: int = 8,
                              max_timesteps: int = 64) -> dict:
    import jax
    from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
    from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
    from minicraftax.tasks.seed_tasks import survive

    params = EnvParams(max_timesteps=max_timesteps)
    static_params = StaticEnvParams()
    env = MultiTaskMiniCraftaxEnv(task_classes=[survive.Env], static_env_params=static_params,
                                  params=params, condition_on_task=False)
    obs0, state0 = env.reset_env(jax.random.PRNGKey(seed), params, 0)
    act_dim = _action_dim()
    actions = np.random.default_rng(ACTION_SEED).integers(0, act_dim, size=steps)
    step_fn = jax.jit(env.step_env)
    state = state0
    prev = None
    key = jax.random.PRNGKey(RUNNER_SEED)
    for action in actions:
        key, step_key = jax.random.split(key)
        prev = state
        _, state, _, _, _ = step_fn(step_key, state, int(action), params)
    template = build_template(prev)
    encoded, _ = encode_env_state(state, next_step_key=key, previous_action=int(actions[-1]),
                                  previous_reward=0.0)
    restored = restore_env_state(encoded, template)
    cmp = compare_env_states(state, restored.env_state)
    return {"pass": bool(cmp["ok"]), "steps": steps, "task_id": 0,
            "n_leaves": cmp["n_leaves"], "restore_ok": bool(cmp["ok"])}


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def run_all(out_dir: str) -> dict:
    started = time.time()
    report: dict[str, Any] = {
        "report": "simulator_frontier.cc1.phase1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pin_set": "proven",
        "key_convention": {
            "runner_split": "r, step_key = jax.random.split(r); step_env(step_key, ...)",
            "env_rng_meaning": "StateBundle.env_rng is the runner key r that continues stepping",
            "state_rng_note": "EnvState.state_rng is overwritten by the engine every step and never consumed by step_env; restored byte-for-byte as an ordinary leaf, never used as a step key",
        },
        "seeds": {"reset": RESET_SEED, "runner": RUNNER_SEED, "action": ACTION_SEED},
        "scope_note": "PASS here proves R4a (env-side restore/parity) ONLY; it is not the R4c combined fresh-process proof.",
    }
    checks: dict[str, Any] = {}
    report["checks"] = checks
    verdict = "PASS"
    blocking = None

    try:
        boot = bootstrap_environment()
    except Exception as exc:  # missing craftax/jax etc. -> BLOCKED, never fake PASS
        report["verdict"] = "BLOCKED"
        report["blocking"] = {"blocked_at": "G0", "reason": f"{type(exc).__name__}: {exc}",
                              "checks_run": [], "next_action": "install craftax-capable venv (see Stage 0 pins)"}
        _write_reports(out_dir, report)
        return report

    checks["bootstrap"] = boot
    if not boot["pass"]:
        verdict = "FAIL"
        blocking = {"blocked_at": "G0", "reason": "bootstrap assertions failed",
                    "checks_run": ["bootstrap"], "next_action": "inspect bootstrap evidence"}
    else:
        report["env"] = {"core": "MiniCraftaxTrain(survive.Env)", "act_dim": _action_dim(),
                         "max_timesteps": {"parity": 64, "terminal": 12, "autoreset": 8, "multitask": 64}}
        setup = build_core_setup(max_timesteps=64)
        checks["restore_roundtrip"] = check_restore_roundtrip(setup)
        checks["dynamics_parity"] = check_dynamics_parity(setup)
        checks["terminal_restore"] = check_terminal_restore()
        checks["autoreset_evidence"] = check_autoreset_evidence()
        checks["corrupted_payload"] = check_corrupted_payload(setup)
        checks["version_mismatch"] = check_version_mismatch(setup)
        checks["batch_parity"] = check_batch_parity(setup)
        checks["multitask_secondary"] = check_multitask_secondary()
        failing = [name for name, c in checks.items() if not c.get("pass", False)]
        if failing:
            verdict = "FAIL"
            blocking = {"blocked_at": failing[0], "reason": "one or more checks failed",
                        "checks_run": list(checks), "next_action": "inspect phase1_parity_report.json"}

    report["verdict"] = verdict
    report["blocking"] = blocking
    report["wall_seconds"] = round(time.time() - started, 2)
    _write_reports(out_dir, report)
    return report


def _write_reports(out_dir: str, report: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    blob = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if len(blob.encode("utf-8")) >= 512 * 1024:
        raise SchemaMismatchError("phase1 report exceeds 512KB budget; trim evidence")
    with open(os.path.join(out_dir, "phase1_parity_report.json"), "w", encoding="utf-8") as f:
        f.write(blob)
    with open(os.path.join(out_dir, "PHASE1_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(_render_markdown(report))


def _render_markdown(report: dict) -> str:
    checks = report.get("checks", {})
    lines = [
        "# Phase 1 真实 Craftax EnvState 恢复/动力学平价报告（R4a env 侧）",
        "",
        f"- verdict：**{report.get('verdict')}**",
        f"- pin 集：{report.get('pin_set')}（主机已验证组合）",
        f"- 核心环境：{report.get('env', {}).get('core')}，动作维度 {report.get('env', {}).get('act_dim')}",
        f"- seeds：{report.get('seeds')}",
        f"- key 约定：{report.get('key_convention', {}).get('runner_split')}",
        f"- state_rng 说明：{report.get('key_convention', {}).get('state_rng_note')}",
        "",
        "## 门禁结果",
        "",
        "| 检查 | 结果 |",
        "|---|---|",
    ]
    for name, check in checks.items():
        lines.append(f"| {name} | {'PASS' if check.get('pass') else 'FAIL'} |")
    lines += [
        "",
        "## 范围声明",
        "",
        report.get("scope_note", ""),
        "",
        "本报告仅为 R4a（env 侧 restore/parity）证据；不构成 R4c 联合 fresh-process 证明，",
        "也不构成任何性能评估。",
        "",
    ]
    if report.get("blocking"):
        lines += ["## BLOCKED/FAIL 详情", "", json.dumps(report["blocking"], ensure_ascii=False), ""]
    return "\n".join(lines)
