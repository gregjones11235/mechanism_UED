#!/usr/bin/env python3
"""CC4 Tier3 — COMMON CANDIDATE RUNTIME ABI (closing contract §5).

The public application boundary between ANY candidate architecture and the common
evaluator. A candidate runtime exposes exactly these operations:

    load_candidate(checkpoint_contract)            -> CandidateRuntime
    runtime.init_memory(batch_size)                -> memory_state
    runtime.policy_step(observation, memory_state, done_mask=None)
                                                   -> {"action": int, "memory_state": ...}
    runtime.reset_memory(memory_state, reset_mask=None) -> memory_state
    runtime.candidate_metadata()                   -> dict

BOUNDARY RULES (fail closed on any violation):

  * The runner dispatches on ``runtime_family``; ``rmt16_gtrxl_cc2`` is the ONLY
    family registered this round. An unknown / missing family fails closed — the
    runner is NOT hardcoded to RMT16, but CC4 does not implement other families
    (Base GTrXL / Control / SlowGRU / Teacher are their owners' deliverables).
  * A candidate runtime NEVER defines scientific predicates: state-bank selection,
    terminal labels, FRONT transition / graph-distance progress, BACK/FULL
    DEFEAT_KOBOLD judgement, metric aggregation and certificates belong ONLY to the
    common evaluator (closing contract §3). candidate_metadata() reports
    scientific_predicates_defined_here=false.
  * Memory semantics are REUSED from CC2's real dynamics (tier3_cc2_policy_adapter
    drives CC2's own rmt_step_forward) — never reimplemented. A memory_state is an
    OPAQUE snapshot of (memories, mem_mask, mem_idx, rmt_st); the evaluator treats
    it as a black box and only passes it back.
  * RMT16 is batch-1 (CC2's policy runs N=1): init_memory(batch_size) fails closed
    unless batch_size == 1.
  * A Tier3 episode STOPS at terminal — no step follows a done — so CC2's
    terminal-reset branch is never entered mid-episode: policy_step fails closed
    unless done_mask is None or all-False.
  * load_candidate verifies the checkpoint against the frozen final-98304 contract
    (exact file SHA / params SHA / manifest / driver-source SHA / policy-source SHA)
    EXACTLY as tier3_evaluator does — one verification path, no second gate.

PURE at import time (runs on the base interpreter); JAX is only needed when a
runtime is actually loaded.
"""
from __future__ import annotations

import os
import sys

# Runnable-as-script AND importable-as-package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_source_audit as audit        # noqa: E402
import tier3_state_serializer as ser      # noqa: E402

SCHEMA = "mechanism_UED.candidate_runtime_abi/v1"
ABI_VERSION = "candidate_runtime_abi/v1"

# Registered runtime families. The runner dispatches on runtime_family; UNKNOWN
# families fail closed (the runner must not hardcode RMT16 — other architectures
# register here when their owners deliver them).
RUNTIME_FAMILIES = ("rmt16_gtrxl_cc2",)

# The ABI surface every registered runtime must expose.
ABI_METHODS = ("load_candidate", "init_memory", "policy_step", "reset_memory",
               "candidate_metadata")

# Opaque RMT16 memory snapshot fields (CC2 driver convention; captured/restored
# verbatim — never reinterpreted by the evaluator).
MEMORY_FIELDS = ("memories", "mem_mask", "mem_idx", "rmt_st")

FROZEN_OBSERVATION_SHAPE = (8335,)
FROZEN_ACTION_DIM = 43
FROZEN_ACTION_MODE = "greedy_argmax"
ARMS = ("persistent", "reset128")


class FailClosed(Exception):
    """Hard stop on any ABI / family / memory-contract violation."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


# ---------------------------------------------------------------------------
# Family-independent argument gates
# ---------------------------------------------------------------------------
def check_batch_size(batch_size: int, supported: int = 1):
    """Batch contract: the RMT16 family is batch-1 (CC2 policy N=1). A batched
    rollout would silently change CC2's per-step dynamics — fail closed instead."""
    require(isinstance(batch_size, int) and not isinstance(batch_size, bool),
            "FAIL CLOSED: init_memory batch_size %r is not an int" % (batch_size,))
    require(batch_size == supported,
            "FAIL CLOSED: init_memory batch_size %d != %d (rmt16_gtrxl_cc2 runs "
            "batch-1; CC2's rmt_step_forward pads 1->2 internally per CC2 — a "
            "different batch would change the dynamics, not just the throughput)"
            % (batch_size, supported))


def check_done_mask(done_mask):
    """A Tier3 episode STOPS at terminal; no step ever follows a done, so CC2's
    terminal-reset branch must never be entered mid-episode. done_mask must be
    None or all-False — anything else fails closed."""
    if done_mask is None:
        return
    import numpy as np
    arr = np.asarray(done_mask)
    require(arr.dtype.kind == "b" or set(np.unique(arr).tolist()) <= {0, 1, False},
            "FAIL CLOSED: done_mask %r is not a boolean mask" % (done_mask,))
    require(bool(np.all(arr == False)),  # noqa: E712
            "FAIL CLOSED: done_mask carries a True entry — a Tier3 episode STOPS at "
            "terminal, so no policy step may observe a done (CC2's terminal-reset "
            "branch is never entered mid-episode)")


def check_reset_mask(reset_mask):
    """batch-1 reset mask: None / False / [False] -> keep state; True / [True] ->
    re-initialize. Anything with more than one element fails closed."""
    if reset_mask is None:
        return False
    import numpy as np
    arr = np.asarray(reset_mask, dtype=bool)
    require(arr.size == 1,
            "FAIL CLOSED: reset_memory reset_mask %r has size %d != 1 (batch-1 "
            "runtime)" % (reset_mask, arr.size))
    return bool(arr.reshape(-1)[0])


# ---------------------------------------------------------------------------
# The runtime object
# ---------------------------------------------------------------------------
class CandidateRuntime:
    """A loaded candidate behind the common ABI. Holds a stateful CC2 policy (for
    the rmt16_gtrxl_cc2 family) and its verified identity; exposes init_memory /
    policy_step / reset_memory / candidate_metadata. params are READ-ONLY and
    captured by reference — nothing here trains or mutates."""

    def __init__(self, family: str, arm: str, policy, identity: dict,
                 observation_shape=FROZEN_OBSERVATION_SHAPE,
                 action_dim: int = FROZEN_ACTION_DIM):
        require(family in RUNTIME_FAMILIES,
                "FAIL CLOSED: runtime_family %r not in registered families %s"
                % (family, RUNTIME_FAMILIES))
        require(arm in ARMS, "FAIL CLOSED: arm %r not in %s" % (arm, ARMS))
        self.family = family
        self.arm = arm
        self._policy = policy
        self._identity = dict(identity)
        self.observation_shape = tuple(int(x) for x in observation_shape)
        self.action_dim = int(action_dim)
        require(self.observation_shape == FROZEN_OBSERVATION_SHAPE,
                "FAIL CLOSED: observation_shape %r != frozen %r"
                % (self.observation_shape, FROZEN_OBSERVATION_SHAPE))
        require(self.action_dim == FROZEN_ACTION_DIM,
                "FAIL CLOSED: action_dim %d != frozen %d"
                % (self.action_dim, FROZEN_ACTION_DIM))

    # -- opaque memory capture / restore (reuse, never reimplement) ----------
    # Snapshots are REFERENCE captures over the four state fields. This is safe
    # because CC2's rmt_step_forward (the only per-step transition, audited source)
    # is purely functional: it returns fresh arrays / a fresh rmt_st dict and never
    # mutates or donates its inputs, so an earlier snapshot stays bit-valid no
    # matter how many steps the policy takes afterwards.
    def _capture(self) -> dict:
        return {"memories": self._policy.memories,
                "mem_mask": self._policy.mem_mask,
                "mem_idx": self._policy.mem_idx,
                "rmt_st": self._policy.rmt_st}

    def _restore(self, memory_state: dict):
        require(isinstance(memory_state, dict)
                and set(memory_state.keys()) == set(MEMORY_FIELDS),
                "FAIL CLOSED: memory_state keys %r != the opaque RMT16 snapshot "
                "fields %s (the evaluator must pass back exactly what init_memory / "
                "policy_step returned)" % (sorted(memory_state or {}), MEMORY_FIELDS))
        self._policy.memories = memory_state["memories"]
        self._policy.mem_mask = memory_state["mem_mask"]
        self._policy.mem_idx = memory_state["mem_idx"]
        self._policy.rmt_st = memory_state["rmt_st"]

    # -- the ABI --------------------------------------------------------------
    def init_memory(self, batch_size: int) -> dict:
        check_batch_size(batch_size)
        self._policy.reset()
        return self._capture()

    def policy_step(self, observation, memory_state: dict, done_mask=None) -> dict:
        check_done_mask(done_mask)
        self._restore(memory_state)
        action = int(self._policy(observation, None))
        require(0 <= action < self.action_dim,
                "FAIL CLOSED: greedy action %d outside [0, %d)"
                % (action, self.action_dim))
        return {"action": action, "memory_state": self._capture()}

    def reset_memory(self, memory_state: dict, reset_mask=None) -> dict:
        if check_reset_mask(reset_mask):
            self._policy.reset()
            return self._capture()
        # No reset requested: validate + pass the state through untouched.
        self._restore(memory_state)
        return self._capture()

    def candidate_metadata(self) -> dict:
        meta = {"schema": SCHEMA, "abi_version": ABI_VERSION,
                "runtime_family": self.family, "arm": self.arm,
                "abi_methods": list(ABI_METHODS),
                "action_mode": FROZEN_ACTION_MODE,
                "action_dim": self.action_dim,
                "observation_shape": list(self.observation_shape),
                "trainable": False,
                "batch_size_supported": 1,
                "memory_fields": list(MEMORY_FIELDS),
                "scientific_predicates_defined_here": False,
                "boundary_note": "scientific predicates / state bank / terminal "
                                 "labels / metrics / certificates are computed ONLY "
                                 "by the common evaluator (closing contract §3)"}
        meta.update(self._identity)
        return meta


# ---------------------------------------------------------------------------
# load_candidate: the ONLY public construction path
# ---------------------------------------------------------------------------
def load_candidate(checkpoint_contract: dict) -> CandidateRuntime:
    """Load + verify a candidate behind the ABI and return a CandidateRuntime.

    ``checkpoint_contract`` keys:
      runtime_family          REQUIRED, must be a registered family
      arm                     REQUIRED, "persistent" | "reset128"
      checkpoint_path         REQUIRED, path to full_state.pkl
      checkpoint_contract_path  optional (default: frozen contract in configs/)
      cc2_snapshot_root       optional (default: audited in-repo CC2 snapshot)
      driver_source_path      optional (default: SHA-bound driver source)
      observation_shape       optional (must equal (8335,) if given)
      action_dim              optional (must equal 43 if given)

    An unknown runtime_family fails closed — this function never assumes RMT16."""
    require(isinstance(checkpoint_contract, dict),
            "FAIL CLOSED: load_candidate needs a checkpoint_contract dict, got %r"
            % type(checkpoint_contract))
    family = checkpoint_contract.get("runtime_family")
    require(family in RUNTIME_FAMILIES,
            "FAIL CLOSED: runtime_family %r is not registered (registered: %s). The "
            "common runner supports multiple architectures by registration — CC4 "
            "only ships rmt16_gtrxl_cc2 this round." % (family, RUNTIME_FAMILIES))
    arm = checkpoint_contract.get("arm")
    require(arm in ARMS,
            "FAIL CLOSED: arm %r not in %s" % (arm, ARMS))
    require(checkpoint_contract.get("checkpoint_path"),
            "FAIL CLOSED: checkpoint_contract carries no checkpoint_path")
    if family == "rmt16_gtrxl_cc2":
        return _load_rmt16_gtrxl_cc2(checkpoint_contract)
    raise FailClosed("FAIL CLOSED: no loader registered for family %r" % family)


def _load_rmt16_gtrxl_cc2(spec: dict) -> CandidateRuntime:
    """Verify the checkpoint against the frozen final-98304 contract EXACTLY as
    tier3_evaluator does (one verification path), rebuild CC2's real network from
    the SHA-bound driver source, and wrap CC2's stateful greedy policy."""
    import tier3_cc2_policy_adapter as pa
    import tier3_checkpoint_adapter as ckpt
    import tier3_checkpoint_contract as contractmod

    # CC2's network_rmt16 imports dicode.transformer.transformerXL from the audited
    # source tree (<repo>/dicode_src/src) — the same setup every evaluator entry
    # point performs (tier3_evaluator.main / self_test).
    _src = str(audit.repo_root() / "dicode_src" / "src")
    if os.path.isdir(_src) and _src not in sys.path:
        sys.path.insert(0, _src)

    checkpoint_path = spec["checkpoint_path"]
    contract_path = spec.get("checkpoint_contract_path",
                             contractmod.DEFAULT_CONTRACT_PATH)
    snapshot_root = spec.get("cc2_snapshot_root", pa._default_snapshot_root())
    driver_source = spec.get("driver_source_path", pa.DEFAULT_DRIVER_SOURCE)
    arm = spec["arm"]
    action_dim = int(spec.get("action_dim", FROZEN_ACTION_DIM))
    observation_shape = tuple(int(x) for x in
                              spec.get("observation_shape", FROZEN_OBSERVATION_SHAPE))
    require(action_dim == FROZEN_ACTION_DIM,
            "FAIL CLOSED: action_dim %d != frozen %d" % (action_dim, FROZEN_ACTION_DIM))
    require(observation_shape == FROZEN_OBSERVATION_SHAPE,
            "FAIL CLOSED: observation_shape %r != frozen %r"
            % (observation_shape, FROZEN_OBSERVATION_SHAPE))

    # The evaluator's exact verification sequence (tier3_evaluator lines 781-822):
    modules, src_id = pa.load_cc2_policy_modules(snapshot_root)
    driver_cfg, driver_sha = pa.load_cfg_from_driver_source(
        driver_source, pa.FROZEN_DRIVER_FILE_SHA256)
    contract = contractmod.load_contract(contract_path)
    params, params_sha, manifest, file_sha = ckpt.load_full_params_readonly(
        checkpoint_path)
    contractmod.verify_checkpoint_against_contract(
        arm, file_sha, params_sha, manifest, driver_sha,
        src_id["cc2_policy_source_sha256"], contract)
    network, rmt_cfg, carry_mode = pa.build_network_from_manifest(
        modules, manifest, action_dim, driver_cfg)
    require(carry_mode == arm,
            "FAIL CLOSED: checkpoint carry_mode %r != requested arm %r"
            % (carry_mode, arm))
    policy = pa.CC2RMT16Policy(modules, network, params, rmt_cfg, carry_mode,
                               driver_cfg["window_mem"], driver_cfg["num_heads"],
                               driver_cfg["num_layers"], driver_cfg["embed_size"])
    identity = {"carry_mode": carry_mode,
                "checkpoint_step": int(manifest.get("step")),
                "checkpoint_file_sha256": file_sha,
                "params_sha256": params_sha,
                "base_checkpoint_params_sha256":
                    contract["common"]["base_checkpoint_params_sha256"],
                "driver_source_sha256": driver_sha,
                "cc2_policy_source_sha256": src_id["cc2_policy_source_sha256"],
                "checkpoint_contract_sha256":
                    contract.get("checkpoint_contract_sha256"),
                "checkpoint_path": os.path.abspath(checkpoint_path)}
    return CandidateRuntime("rmt16_gtrxl_cc2", arm, policy, identity,
                            observation_shape, action_dim)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _synthetic_runtime(carry_mode: str = "persistent"):
    """Build a CandidateRuntime over RANDOM-INIT params via CC2's own modules —
    TEST ONLY (no checkpoint, no contract). Mirrors the adapter self-test
    construction; exercises the ABI without touching real identities."""
    import numpy as np                      # noqa: F401
    import jax
    import jax.numpy as jnp
    import tier3_cc2_policy_adapter as pa

    # Audited dicode source tree (see _load_rmt16_gtrxl_cc2).
    _src = str(audit.repo_root() / "dicode_src" / "src")
    if os.path.isdir(_src) and _src not in sys.path:
        sys.path.insert(0, _src)

    root = pa._default_snapshot_root()
    modules, _sid = pa.load_cc2_policy_modules(root)
    cfg, _dsha = pa.load_cfg_from_driver_source(pa.DEFAULT_DRIVER_SOURCE,
                                                pa.FROZEN_DRIVER_FILE_SHA256)
    manifest = pa._synthetic_manifest()
    manifest["carry_mode"] = carry_mode
    network, rmt_cfg, carry = pa.build_network_from_manifest(modules, manifest, 43, cfg)
    full = network.init(
        jax.random.PRNGKey(0),
        jnp.zeros((2, cfg["window_mem"], cfg["num_layers"], cfg["embed_size"])),
        jnp.zeros((2, 8335)),
        jnp.zeros((2, cfg["num_heads"], 1, cfg["window_mem"] + 1), jnp.bool_),
        mem_tokens=jnp.zeros((2, cfg["rmt_num_tokens"], cfg["embed_size"])),
        seg_buf=jnp.zeros((2, cfg["num_steps"], cfg["embed_size"])),
        method=network.init_all)
    params = jax.tree_util.tree_map(jnp.asarray, full["params"])
    policy = pa.CC2RMT16Policy(modules, network, params, rmt_cfg, carry,
                               cfg["window_mem"], cfg["num_heads"],
                               cfg["num_layers"], cfg["embed_size"])
    identity = {"carry_mode": carry, "checkpoint_step": -1,
                "checkpoint_file_sha256": "0" * 64, "params_sha256": "0" * 64,
                "base_checkpoint_params_sha256": "0" * 64,
                "driver_source_sha256": pa.FROZEN_DRIVER_FILE_SHA256,
                "cc2_policy_source_sha256":
                    _sid["cc2_policy_source_sha256"],
                "checkpoint_contract_sha256": "0" * 64,
                "checkpoint_path": "SYNTHETIC_SELF_TEST"}
    return CandidateRuntime("rmt16_gtrxl_cc2", carry, policy, identity)


def self_test() -> int:
    problems = []

    def check(name, cond):
        if not cond:
            problems.append(name)

    # ---- PURE gates (any interpreter, no JAX) --------------------------------
    try:
        load_candidate({"runtime_family": "gpt9000_ultra", "arm": "persistent",
                        "checkpoint_path": "x.pkl"})
        check("unknown_family_rejected", False)
    except FailClosed:
        check("unknown_family_rejected", True)
    try:
        load_candidate({"arm": "persistent", "checkpoint_path": "x.pkl"})
        check("missing_family_rejected", False)
    except FailClosed:
        check("missing_family_rejected", True)
    try:
        load_candidate({"runtime_family": "rmt16_gtrxl_cc2", "arm": "sideways",
                        "checkpoint_path": "x.pkl"})
        check("bad_arm_rejected", False)
    except FailClosed:
        check("bad_arm_rejected", True)
    try:
        load_candidate({"runtime_family": "rmt16_gtrxl_cc2", "arm": "persistent"})
        check("missing_checkpoint_path_rejected", False)
    except FailClosed:
        check("missing_checkpoint_path_rejected", True)
    try:
        load_candidate("not-a-dict")
        check("non_dict_contract_rejected", False)
    except FailClosed:
        check("non_dict_contract_rejected", True)

    try:
        check_batch_size(2)
        check("batch_gt1_rejected", False)
    except FailClosed:
        check("batch_gt1_rejected", True)
    try:
        check_batch_size(True)
        check("batch_bool_rejected", False)
    except FailClosed:
        check("batch_bool_rejected", True)
    check_batch_size(1)
    check("batch_1_accepted", True)

    import numpy as np
    check_done_mask(None)
    check_done_mask(np.zeros((1,), bool))
    check_done_mask([False])
    check("done_mask_none_or_false_accepted", True)
    try:
        check_done_mask(np.ones((1,), bool))
        check("done_mask_true_rejected", False)
    except FailClosed:
        check("done_mask_true_rejected", True)
    try:
        check_done_mask([0, 1])
        check("done_mask_any_true_rejected", False)
    except FailClosed:
        check("done_mask_any_true_rejected", True)

    check("reset_mask_none_keeps", check_reset_mask(None) is False)
    check("reset_mask_false_keeps", check_reset_mask(False) is False)
    check("reset_mask_true_resets", check_reset_mask(True) is True)
    check("reset_mask_list_true_resets", check_reset_mask([True]) is True)
    try:
        check_reset_mask([True, False])
        check("reset_mask_size2_rejected", False)
    except FailClosed:
        check("reset_mask_size2_rejected", True)

    # ABI surface completeness.
    check("abi_methods_frozen",
          ABI_METHODS == ("load_candidate", "init_memory", "policy_step",
                          "reset_memory", "candidate_metadata"))
    check("rmt16_registered_only", RUNTIME_FAMILIES == ("rmt16_gtrxl_cc2",))

    # ---- JAX-HOST gates (venv only): real ABI over CC2's own dynamics --------
    if ser.have_jax():
        rt = _synthetic_runtime("persistent")
        m0 = rt.init_memory(1)
        check("init_memory_fields", set(m0.keys()) == set(MEMORY_FIELDS))
        try:
            rt.init_memory(2)
            check("runtime_batch2_rejected", False)
        except FailClosed:
            check("runtime_batch2_rejected", True)

        rng = np.random.default_rng(20260730)
        obs_seq = [rng.normal(size=8335).astype(np.float32) for _ in range(6)]

        # Deterministic: two fresh runtimes, same params, same obs -> same actions.
        rt2 = _synthetic_runtime("persistent")
        a_seq, m = [], m0
        for obs in obs_seq:
            out = rt.policy_step(obs, m)
            a_seq.append(out["action"])
            m = out["memory_state"]
        a_seq2, m2 = [], rt2.init_memory(1)
        for obs in obs_seq:
            out = rt2.policy_step(obs, m2)
            a_seq2.append(out["action"])
            m2 = out["memory_state"]
        check("abi_deterministic_across_runtimes", a_seq == a_seq2)
        check("actions_in_range", all(0 <= a < 43 for a in a_seq))

        # Capture/restore: stepping from a restored mid-episode snapshot reproduces
        # the continuation of the original run (state threaded through, exactly as
        # the evaluator's episode loop does).
        m_mid = m0
        for obs in obs_seq[:3]:
            m_mid = rt.policy_step(obs, m_mid)["memory_state"]
        cont, mm = [], m_mid
        for obs in obs_seq[3:]:
            out = rt.policy_step(obs, mm)
            cont.append(out["action"])
            mm = out["memory_state"]
        check("memory_snapshot_restore_continues", cont == a_seq[3:])

        # reset_memory: mask=True returns the FRESH state (equal to init); mask=False
        # passes the advanced state through untouched. (Compared leaf-wise: rmt_st
        # is a pytree DICT of arrays, not a single array.)
        import jax as _jax

        def _states_equal(s1, s2):
            l1 = [np.asarray(x) for x in _jax.tree_util.tree_leaves(s1)]
            l2 = [np.asarray(x) for x in _jax.tree_util.tree_leaves(s2)]
            return (len(l1) == len(l2)
                    and all(np.array_equal(a, b) for a, b in zip(l1, l2)))

        fresh = rt.reset_memory(m, True)
        init_fresh = rt.init_memory(1)
        check("reset_true_restores_init", _states_equal(fresh, init_fresh))
        kept = rt.reset_memory(m, False)
        check("reset_false_passes_through", _states_equal(kept, m))

        # A done_mask=True policy step fails closed even on a live runtime.
        try:
            rt.policy_step(obs_seq[0], m0, done_mask=np.ones((1,), bool))
            check("runtime_done_true_rejected", False)
        except FailClosed:
            check("runtime_done_true_rejected", True)

        # A foreign memory_state (wrong keys) fails closed.
        try:
            rt.policy_step(obs_seq[0], {"memories": m0["memories"]})
            check("foreign_memory_state_rejected", False)
        except FailClosed:
            check("foreign_memory_state_rejected", True)

        # params READ-ONLY: leaf values identical after a full rollout (NEG23 lite).
        leaves_before = [np.asarray(x).copy() for x in
                         __import__("jax").tree_util.tree_leaves(rt._policy.params)]
        m = rt.init_memory(1)
        for obs in obs_seq:
            m = rt.policy_step(obs, m)["memory_state"]
        leaves_after = [np.asarray(x) for x in
                        __import__("jax").tree_util.tree_leaves(rt._policy.params)]
        check("params_readonly",
              all(np.array_equal(b, a) for b, a in zip(leaves_before, leaves_after)))

        meta = rt.candidate_metadata()
        check("metadata_complete",
              meta["runtime_family"] == "rmt16_gtrxl_cc2"
              and meta["action_mode"] == "greedy_argmax"
              and meta["action_dim"] == 43
              and meta["observation_shape"] == [8335]
              and meta["trainable"] is False
              and meta["batch_size_supported"] == 1
              and meta["scientific_predicates_defined_here"] is False
              and set(meta["abi_methods"]) == set(ABI_METHODS)
              and len(meta["params_sha256"]) == 64
              and len(meta["cc2_policy_source_sha256"]) == 64)

        # carry_mode dispatch: both arms load behind the same ABI.
        rt_r = _synthetic_runtime("reset128")
        check("reset128_family_loads",
              rt_r.candidate_metadata()["carry_mode"] == "reset128"
              and rt_r.init_memory(1) is not None)
        print("  jax_host: ABI determinism / capture-restore / reset / readonly "
              "params / dual carry_mode all verified")
    else:
        print("  jax_host: SKIPPED (no JAX on this host; pure gates above still bind)")

    # ---- REAL checkpoint path (server only; opt-in via env) -------------------
    real_pkl = os.environ.get("CC4_REAL_PKL_PERSISTENT")
    if real_pkl and os.path.isfile(real_pkl):
        rt = load_candidate({"runtime_family": "rmt16_gtrxl_cc2",
                             "arm": "persistent", "checkpoint_path": real_pkl})
        meta = rt.candidate_metadata()
        ok = (meta["checkpoint_file_sha256"]
              == "2866b5defc356b57345ca47b2f4f44f19f63e618aa62bc8e03c8f751f005c723"
              and meta["params_sha256"]
              == "aa6ba44040a0742bd709ebe6299acde6242e4faf748159dd0765ee2428addd0d"
              and meta["checkpoint_step"] == 98304)
        m = rt.init_memory(1)
        obs = np.zeros(8335, np.float32)
        out = rt.policy_step(obs, m)
        check("real_persistent_load_and_step",
              ok and 0 <= out["action"] < 43)
        print("  real_pkl: persistent contract verification + ABI step verified")
    else:
        print("  real_pkl: SKIPPED (set CC4_REAL_PKL_PERSISTENT to a full_state.pkl "
              "to exercise the real contract path; server-side runs do this)")

    if problems:
        print("TIER3_CANDIDATE_RUNTIME_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_CANDIDATE_RUNTIME_SELF_TEST_PASS "
          "(ABI frozen; unknown families rejected; memory contract enforced)")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return self_test()
    print("usage: tier3_candidate_runtime.py --self-test")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
