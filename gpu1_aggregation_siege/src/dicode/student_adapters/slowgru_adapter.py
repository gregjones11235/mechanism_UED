"""Read-only SlowGRU StudentAdapter (E3 canonical frontier mount).

Mounts a CC3-trained SlowGRU Student strictly READ-ONLY against the shared
StudentAdapter protocol, via the CC3 shared slowgru_runtime:

  * params come from the real CC3 ``full_state.pkl`` (bakeoff-format packed
    params, loaded by slowgru_runtime.load_candidate with file-SHA + params-SHA
    recomputation fail-closed);
  * the network is the SHA-bound slowgru_network.py (ActorCriticSlowGRU) from
    the CC3 arm_src directory — never reimplemented, never guessed;
  * the forward path is slowgru_runtime.policy_step (verbatim trainer _env_step
    memory mechanics);
  * persistent memory: slow longstate + fast window memories carry across
    segment boundaries; cleared only on true episode done/reset;
  * ZERO parameter updates. save_full_state / restore_full_state raise
    NotImplementedError this round.

The adapter imports jax/slowgru_runtime lazily (function-level) so that
importing it at module level never crashes a jax-less interpreter.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from typing import Any, Mapping

import numpy as np

from .checkpoint_codec import FORMAT_BAKEOFF_PKL
from .identity import validate_identity
from .protocol import ActionSpec, CheckpointSpec, MemoryFieldSpec, ObsSpec
from .registry import StudentProfile


class SlowGRUMountError(RuntimeError):
    """Raised on any SlowGRU mount/identity violation (fail closed)."""


# Memory field keys are FLAT dotted names, matching the profile yamls.
MEMORY_FIELD_KEYS = (
    "memories", "memories_mask", "memories_mask_idx",
    "longstate.h", "longstate.buf", "longstate.count",
)

# Internal bookkeeping fields (not exposed in memory_spec).
_INTERNAL_KEYS = frozenset({"true_done", "step_idx"})

# Known network limitation: the slowgru_network's transformerXL.forward_eval
# uses x.squeeze() which collapses the batch dimension when batch_size=1,
# causing a jax concatenation error.  The adapter pads batch_size=1 to 2
# internally and strips the extra output.  This is a workaround for the
# network's x.squeeze() behaviour; the network is NOT modified.
_MIN_BATCH_FOR_NETWORK = 2


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SlowGRUMountError(msg)


def _craftax_action_count() -> int:
    """Cross-validation anchor for action_count (fail closed if unavailable)."""
    try:
        from craftax.craftax.constants import Action
    except ImportError:
        try:
            from craftax.constants import Action
        except Exception as exc:
            raise SlowGRUMountError(
                f"BLOCKED_ENVIRONMENT: craftax required for action_count "
                f"cross-validation: {exc}") from exc
    return len(Action)


def _stable_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _validate_runtime_params(params: Any, reference: Any, cache: set[tuple] | None = None) -> None:
    """Fail closed before binding caller-owned params to a runtime forward.

    The SlowGRU source checkpoint is used only to construct and authenticate the
    network runtime.  Resumed E3 RunState parameters are allowed to differ in
    value, but never in pytree structure, leaf shape/dtype, or finiteness.
    """
    _require(params is not None, "policy_step params must not be None")
    try:
        import jax
        leaves = jax.tree_util.tree_leaves(params)
        ref_leaves = jax.tree_util.tree_leaves(reference)
        _require(
            jax.tree_util.tree_structure(params)
            == jax.tree_util.tree_structure(reference),
            "policy_step params pytree structure differs from mounted SlowGRU params",
        )
    except SlowGRUMountError:
        raise
    except Exception as exc:
        raise SlowGRUMountError(
            f"cannot inspect policy_step params pytree: {type(exc).__name__}: {exc}"
        ) from exc
    token = (id(params), tuple((id(x), getattr(x, "shape", None), str(getattr(x, "dtype", ""))) for x in leaves))
    if cache is not None and token in cache:
        return
    _require(len(leaves) == len(ref_leaves),
             "policy_step params leaf count differs from mounted SlowGRU params")
    for index, (leaf, ref_leaf) in enumerate(zip(leaves, ref_leaves)):
        try:
            arr = np.asarray(jax.device_get(leaf))
            ref_arr = np.asarray(jax.device_get(ref_leaf))
        except Exception as exc:
            raise SlowGRUMountError(
                f"cannot materialize policy_step params leaf #{index}: {exc}"
            ) from exc
        _require(arr.shape == ref_arr.shape and arr.dtype == ref_arr.dtype,
                 f"policy_step params leaf #{index} shape/dtype "
                 f"{arr.shape}/{arr.dtype} != mounted {ref_arr.shape}/{ref_arr.dtype}")
        _require(not np.issubdtype(arr.dtype, np.number)
                 or bool(np.isfinite(arr).all()),
                 f"policy_step params leaf #{index} contains NaN/Inf")
    if cache is not None:
        cache.add(token)


def _pad_memory_batch(memory_nested: dict, target_batch: int) -> dict:
    """Duplicate the first batch element to reach target_batch size."""
    import numpy as np
    padded = {}
    for key, value in memory_nested.items():
        if key == "longstate" and isinstance(value, dict):
            padded[key] = {}
            for sk, sv in value.items():
                arr = np.asarray(sv)
                if arr.shape[0] == 1 and target_batch > 1:
                    padded[key][sk] = np.repeat(arr, target_batch, axis=0)
                else:
                    padded[key][sk] = sv
        elif key in ("step_idx",):
            padded[key] = value
        else:
            arr = np.asarray(value)
            if arr.shape[0] == 1 and target_batch > 1:
                padded[key] = np.repeat(arr, target_batch, axis=0)
            else:
                padded[key] = value
    return padded


def _strip_padded_output(output_nested: dict, original_batch: int) -> dict:
    """Strip padded batch elements back to original_batch size."""
    stripped = {}
    for key, value in output_nested.items():
        if key == "longstate" and isinstance(value, dict):
            stripped[key] = {}
            for sk, sv in value.items():
                arr = np.asarray(sv)
                stripped[key][sk] = arr[:original_batch]
        elif key in ("step_idx",):
            stripped[key] = value
        else:
            arr = np.asarray(value)
            stripped[key] = arr[:original_batch]
    return stripped


def _flat_to_nested(memory_flat: dict) -> dict:
    """Convert flat dotted-key memory to nested dict for slowgru_runtime."""
    nested: dict = {}
    longstate: dict = {}
    for key, value in memory_flat.items():
        if key.startswith("longstate."):
            subkey = key.split(".", 1)[1]
            longstate[subkey] = value
        elif key not in _INTERNAL_KEYS:
            nested[key] = value
    if longstate:
        nested["longstate"] = longstate
    # Carry internal bookkeeping if present
    for ik in _INTERNAL_KEYS:
        if ik in memory_flat:
            nested[ik] = memory_flat[ik]
    return nested


def _nested_to_flat(memory_nested: dict) -> dict:
    """Convert nested dict (slowgru_runtime format) to flat dotted keys."""
    flat: dict = {}
    for key, value in memory_nested.items():
        if key == "longstate":
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat[f"longstate.{subkey}"] = subvalue
        elif key not in _INTERNAL_KEYS:
            flat[key] = value
        elif key in _INTERNAL_KEYS:
            flat[key] = value  # preserve internal keys for round-trip
    return flat


class SlowGRUStudentAdapter:
    """Read-only mount of the CC3 SlowGRU Student family (PERSISTENT/RESET128).

    Constructor arguments
    ---------------------
    profile : StudentProfile
        Logical identity + interface parameters (from the shared registry).
    slowgru_runtime_path : str
        Server path to the slowgru_runtime directory (sys.path injected at load
        time).  Must contain ``slowgru_runtime.py``.
    checkpoint_contract_path : str
        Server path to ``checkpoint_contract.json`` in the candidate capsule.
    expected_network_src_sha256 : str
        SHA256 of the network source (slowgru_network.py) as declared in the
        contract — verified against the on-disk file at load time.
    expected_trainer_src_sha256 : str
        SHA256 of the canonical trainer source as declared in the contract —
        verified against the pkl-embedded frozen code_sha256 at load time.
    """

    def __init__(self, profile: StudentProfile, *,
                 slowgru_runtime_path: str | None = None,
                 checkpoint_contract_path: str | None = None,
                 expected_network_src_sha256: str | None = None,
                 expected_trainer_src_sha256: str | None = None) -> None:
        _require(isinstance(profile, StudentProfile),
                 f"SlowGRUStudentAdapter needs a StudentProfile, got {type(profile)!r}")
        _require(profile.architecture_family == "SLOWGRU",
                 f"architecture_family must be SLOWGRU, got {profile.architecture_family!r}")
        _require(profile.checkpoint_format == FORMAT_BAKEOFF_PKL,
                 f"SlowGRUStudentAdapter mounts BAKEOFF_PKL only, got {profile.checkpoint_format!r}")
        _require(profile.memory_mode == "PERSISTENT",
                 f"SlowGRUStudentAdapter requires PERSISTENT memory_mode, got {profile.memory_mode!r}")
        carry = profile.notes.get("carry_mode")
        _require(carry == "persistent",
                 f"profile notes carry_mode must be 'persistent', got {carry!r}")

        self._profile = profile
        self._slowgru_runtime_path = slowgru_runtime_path
        self._checkpoint_contract_path = checkpoint_contract_path
        self._expected_network_src_sha256 = expected_network_src_sha256
        self._expected_trainer_src_sha256 = expected_trainer_src_sha256
        # set by load_full_state
        self._handle: dict | None = None
        self._loaded: dict | None = None

    # ------------------------------------------------------------------ specs

    def identity(self):
        return self._profile.expected_identity()

    def observation_spec(self) -> ObsSpec:
        return ObsSpec(shape=tuple(self._profile.observation_shape))

    def action_spec(self) -> ActionSpec:
        return ActionSpec(count=int(self._profile.action_count))

    def memory_spec(self):
        return self._profile.memory_spec()

    def checkpoint_spec(self) -> CheckpointSpec:
        return CheckpointSpec(
            format=FORMAT_BAKEOFF_PKL,
            params_sha256=self._profile.params_sha256,
            source_commit=self._profile.source_commit,
            contains_optimizer=False,
            contains_rng=False,
            contains_memory=False,
            notes={
                "checkpoint_file_sha256": str(self._profile.notes.get("checkpoint_file_sha256", "")),
                "network_src_sha256": str(self._profile.notes.get("network_src_sha256", "")),
                "r4c_joint_proof_status": (
                    "unavailable: CC3 bakeoff pkl carries params only "
                    "(no optimizer/rng/policy-memory)"),
                "formal_ranking": str(self._profile.notes.get("formal_ranking", "")),
                "provisional_rank": int(self._profile.notes.get("provisional_rank", -1)),
            },
        )

    # ------------------------------------------------------------- load gates

    def configure_runtime_paths(self, slowgru_runtime_path: str,
                                checkpoint_contract_path: str,
                                expected_network_src_sha256: str,
                                expected_trainer_src_sha256: str) -> None:
        """Supply runtime paths post-construction (fail closed at load time if
        never supplied)."""
        _require(bool(slowgru_runtime_path), "slowgru_runtime_path is empty")
        _require(bool(checkpoint_contract_path), "checkpoint_contract_path is empty")
        _require(bool(expected_network_src_sha256), "expected_network_src_sha256 is empty")
        _require(bool(expected_trainer_src_sha256), "expected_trainer_src_sha256 is empty")
        self._slowgru_runtime_path = str(slowgru_runtime_path)
        self._checkpoint_contract_path = str(checkpoint_contract_path)
        self._expected_network_src_sha256 = str(expected_network_src_sha256)
        self._expected_trainer_src_sha256 = str(expected_trainer_src_sha256)

    def load_full_state(self, checkpoint_path: str, expected_identity) -> Mapping[str, Any]:
        """REAL_CHECKPOINT_LOADED gate chain for SlowGRU (every step fail-closed).

        G0 identity: expected_identity validates and equals the profile identity.
        G1 contract: checkpoint_contract.json exists, candidate_id/carry_mode match.
        G2 runtime: slowgru_runtime.py accessible, sys.path injected.
        G3 network: on-disk slowgru_network.py SHA matches expected.
        G4 load: slowgru_runtime.load_candidate (file SHA + params SHA + network
           SHA + params finiteness all fail-closed inside the runtime).
        G5 trainer: pkl-embedded frozen trainer SHA matches expected.
        G6 memory: profile memory shapes match the runtime constants.
        """
        gates: dict[str, Any] = {}

        # G0 identity completeness + agreement
        expected_identity = validate_identity(expected_identity)
        profile_identity = self._profile.expected_identity()
        _require(expected_identity.identity_hash() == profile_identity.identity_hash(),
                 f"identity mismatch: expected_identity hash {expected_identity.identity_hash()[:16]}... "
                 f"!= profile identity hash {profile_identity.identity_hash()[:16]}... "
                 "(candidate/step/params-sha/memory-spec must all agree; never guess)")
        gates["G0_identity"] = {"identity_hash": profile_identity.identity_hash(),
                                "candidate_id": profile_identity.candidate_id}

        # G1 runtime paths configured
        _require(bool(self._slowgru_runtime_path) and bool(self._checkpoint_contract_path),
                 "FAIL CLOSED: slowgru_runtime_path/checkpoint_contract_path not configured — "
                 "cannot load the SlowGRU checkpoint without the runtime (no guessing, no defaults)")
        _require(bool(self._expected_network_src_sha256) and bool(self._expected_trainer_src_sha256),
                 "FAIL CLOSED: expected_network_src_sha256/expected_trainer_src_sha256 not configured")

        # G1: read checkpoint_contract.json
        contract_path = self._checkpoint_contract_path
        if not os.path.isfile(contract_path):
            raise SlowGRUMountError(f"checkpoint_contract.json not found: {contract_path!r}")
        with open(contract_path, "r", encoding="utf-8") as fh:
            contract = json.load(fh)
        _require(isinstance(contract, dict),
                 f"checkpoint_contract.json is not a dict: {contract_path!r}")
        _require(contract.get("candidate_id") == self._profile.candidate_id,
                 f"contract candidate_id {contract.get('candidate_id')!r} != profile "
                 f"{self._profile.candidate_id!r}")
        _require(contract.get("carry_mode") == "PERSISTENT",
                 f"contract carry_mode {contract.get('carry_mode')!r} != PERSISTENT")
        _require(contract.get("checkpoint_path") == checkpoint_path,
                 f"contract checkpoint_path {contract.get('checkpoint_path')!r} != "
                 f"supplied {checkpoint_path!r}")
        gates["G1_contract"] = {
            "candidate_id": contract["candidate_id"],
            "carry_mode": contract["carry_mode"],
            "contract_checkpoint_path": contract["checkpoint_path"],
        }

        # G2: sys.path inject slowgru_runtime_path
        rt_path = self._slowgru_runtime_path
        if not os.path.isdir(rt_path):
            raise SlowGRUMountError(f"slowgru_runtime_path not a directory: {rt_path!r}")
        rt_file = os.path.join(rt_path, "slowgru_runtime.py")
        if not os.path.isfile(rt_file):
            raise SlowGRUMountError(f"slowgru_runtime.py not found in {rt_path!r}")
        if rt_path not in sys.path:
            sys.path.insert(0, rt_path)
        gates["G2_runtime_path"] = {"slowgru_runtime_path": rt_path}

        # G3: verify network source SHA on disk (before slowgru_runtime does it too)
        import hashlib
        arm_src = contract.get("arm_src", "")
        if not arm_src or not os.path.isdir(arm_src):
            raise SlowGRUMountError(f"contract arm_src not a directory: {arm_src!r}")
        net_module = contract.get("network_module", "slowgru_network")
        net_path = os.path.join(arm_src, net_module + ".py")
        if not os.path.isfile(net_path):
            raise SlowGRUMountError(f"network source not found: {net_path!r}")
        net_sha = hashlib.sha256(open(net_path, "rb").read()).hexdigest()
        _require(net_sha == self._expected_network_src_sha256,
                 f"network_src_sha256 mismatch: disk={net_sha[:16]}... != "
                 f"expected={self._expected_network_src_sha256[:16]}...")
        gates["G3_network_src_sha256"] = net_sha

        # G4: load via slowgru_runtime (internal file-SHA + params-SHA + finiteness gates)
        try:
            import slowgru_runtime as sr
        except ImportError as exc:
            raise SlowGRUMountError(
                f"BLOCKED_ENVIRONMENT: cannot import slowgru_runtime from {rt_path!r}: {exc}") from exc
        try:
            handle = sr.load_candidate(contract)
        except Exception as exc:
            raise SlowGRUMountError(
                f"slowgru_runtime.load_candidate failed: {type(exc).__name__}: {exc}") from exc
        _require(isinstance(handle, dict) and handle.get("params") is not None,
                 "slowgru_runtime.load_candidate returned an invalid handle")
        _require(handle.get("params_sha256") == self._profile.params_sha256,
                 f"runtime params_sha256 {handle.get('params_sha256')!r} != profile "
                 f"{self._profile.params_sha256!r}")
        _require(handle.get("file_sha256") == self._profile.notes.get("checkpoint_file_sha256"),
                 f"runtime file_sha256 {handle.get('file_sha256')!r} != profile "
                 f"checkpoint_file_sha256 {self._profile.notes.get('checkpoint_file_sha256')!r}")
        gates["G4_load"] = {
            "params_sha256": handle["params_sha256"],
            "file_sha256": handle["file_sha256"],
            "runtime": handle.get("runtime", ""),
            "carry_mode": handle.get("carry_mode", ""),
        }

        # G5: verify trainer source SHA (pkl-embedded frozen code_sha256)
        contract_trainer = contract.get("canonical_trainer_src_sha256", "")
        _require(contract_trainer == self._expected_trainer_src_sha256,
                 f"contract canonical_trainer_src_sha256 {contract_trainer!r} != "
                 f"expected {self._expected_trainer_src_sha256!r}")
        gates["G5_trainer_src_sha256"] = contract_trainer

        # G6: memory shape consistency against profile
        expected_shapes = {
            "memories": ((128, 2, 256), "float32"),
            "memories_mask": ((8, 1, 129), "bool"),
            "memories_mask_idx": ((), "int32"),
            "longstate.h": ((256,), "float32"),
            "longstate.buf": ((32, 256), "float32"),
            "longstate.count": ((), "int32"),
        }
        spec = self._profile.memory_spec()
        for name, (trailing, dtype) in expected_shapes.items():
            _require(name in spec.fields,
                     f"profile memory spec missing field {name!r} required by SlowGRU contract")
            fspec: MemoryFieldSpec = spec.fields[name]
            _require(tuple(fspec.shape[1:]) == tuple(trailing),
                     f"profile memory field {name!r} shape {fspec.shape} != expected "
                     f"(None, {trailing})")
            _require(fspec.dtype == dtype,
                     f"profile memory field {name!r} dtype {fspec.dtype!r} != {dtype!r}")
        gates["G6_memory_contract"] = {
            "fields_checked": sorted(expected_shapes),
            "mode": "PERSISTENT",
        }

        # G7: validate action_count against craftax
        craftax_actions = _craftax_action_count()
        _require(craftax_actions == int(self._profile.action_count),
                 f"craftax len(Action) {craftax_actions} != profile action_count "
                 f"{self._profile.action_count}")
        gates["G7_action_count"] = {"craftax_len_action": craftax_actions,
                                    "profile_action_count": int(self._profile.action_count)}

        # Seed the policy rng
        sr.seed_policy_rng(handle, 42)

        self._handle = handle
        self._loaded = {
            "params_sha256": handle["params_sha256"],
            "file_sha256": handle["file_sha256"],
            "global_step": self._profile.global_step,
        }
        return {
            "params": handle["params"],
            "params_sha256": handle["params_sha256"],
            "file_sha256": handle["file_sha256"],
            "driver_source_sha256": contract_trainer,
            "network_src_sha256": net_sha,
            "global_step": self._profile.global_step,
            "contains_optimizer": False,
            "contains_rng": False,
            "contains_policy_memory": False,
            "gates": gates,
            "handle": handle,
            "memory_shapes": {k: v[0] for k, v in expected_shapes.items()},
            "r4c_joint_proof_status": {
                "params": "restored+hash-bound (slowgru_runtime)",
                "optimizer": "ABSENT_IN_CHECKPOINT",
                "train_rng": "ABSENT_IN_CHECKPOINT",
                "global_step": "from profile (no optimizer state to pair)",
                "policy_memory": "re-derived from init_memory",
                "env_state": "R4a-side proof (separate); R4c combined run NOT executed",
            },
        }

    # ----------------------------------------------------------------- memory

    def _require_loaded(self) -> None:
        _require(self._handle is not None and self._loaded is not None,
                 "adapter not loaded: load_full_state must run first")

    def initial_memory(self, batch_size: int) -> dict:
        self._require_loaded()
        batch_size = int(batch_size)
        _require(batch_size >= 1, f"batch_size must be >= 1, got {batch_size}")
        import slowgru_runtime as sr
        nested = sr.init_memory(self._handle, batch_size)
        return _nested_to_flat(nested)

    def validate_memory(self, memory: Any, batch_size: int) -> Mapping[str, Any]:
        reasons: list[str] = []
        if self._handle is None:
            return {"ok": False,
                    "reasons": ["adapter not loaded: load_full_state must run first"]}
        if not isinstance(memory, Mapping):
            return {"ok": False, "reasons": [f"memory is not a Mapping (got {type(memory)!r})"]}

        expected = {
            "memories": ((128, 2, 256), np.float32),
            "memories_mask": ((8, 1, 129), np.bool_),
            "memories_mask_idx": ((), np.int32),
            "longstate.h": ((256,), np.float32),
            "longstate.buf": ((32, 256), np.float32),
            "longstate.count": ((), np.int32),
        }
        keys = set(memory.keys())
        # Allow internal keys to be present
        public_keys = {k for k in keys if k not in _INTERNAL_KEYS}
        missing = sorted(set(expected) - public_keys)
        extra = sorted(public_keys - set(expected))
        if missing:
            reasons.append(f"missing memory fields {missing}")
        if extra:
            reasons.append(f"unexpected memory fields {extra} (exact contract required)")
        for name, (trailing, dtype) in expected.items():
            if name not in memory:
                continue
            arr = np.asarray(memory[name])
            want_shape = (int(batch_size),) + tuple(trailing)
            if arr.shape != want_shape:
                reasons.append(f"field {name!r} shape {arr.shape} != expected {want_shape}")
            if arr.dtype != dtype:
                reasons.append(f"field {name!r} dtype {arr.dtype} != expected {dtype}")
        if not reasons:
            # Check finite values
            for name in ("memories", "longstate.h", "longstate.buf"):
                if name in memory:
                    f = np.asarray(memory[name])
                    if f.size and not np.isfinite(f).all():
                        reasons.append(f"field {name!r} carries non-finite values")
            # Check mem_idx range
            if "memories_mask_idx" in memory:
                idx = np.asarray(memory["memories_mask_idx"])
                if idx.size and (idx.min() < 0 or idx.max() > 129):
                    reasons.append("memories_mask_idx values outside [0, 129]")
        return {"ok": not reasons, "reasons": reasons}

    # ----------------------------------------------------------------- forward

    def policy_step(self, params: Any, observation: Any, memory: Any,
                    previous_action: Any, previous_reward: Any, rng: Any,
                    deterministic: bool) -> Mapping[str, Any]:
        """One READ-ONLY forward through slowgru_runtime.policy_step.

        ``previous_action``/``previous_reward`` are accepted for protocol
        compatibility; the SlowGRU architecture does not consume them.
        ``done_mask`` is held False for every stepped batch entry.

        Known limitation: the slowgru_network's transformerXL.forward_eval
        uses x.squeeze() which collapses the batch dimension at batch_size=1.
        When batch_size=1, the adapter pads to batch_size=2 internally and
        strips the extra output.  The network code is NOT modified.
        """
        self._require_loaded()
        import slowgru_runtime as sr

        # The production runtime stores the authenticated source-checkpoint
        # params in its handle and policy_step reads handle["params"].  E3
        # continuation, however, supplies newer params restored from its
        # canonical RunState.  Bind those caller-owned params to an ephemeral
        # shallow handle: no mutation of self._handle, no fallback to stale
        # source params, and no network reimplementation.
        # Validate on every call.  Callers may mutate a pytree in place, so
        # object-identity caching cannot safely prove structure/finiteness.
        cache = getattr(self, "_validated_param_tokens", None)
        if cache is None:
            cache = set()
            self._validated_param_tokens = cache
        _validate_runtime_params(params, self._handle.get("params"), cache)
        runtime_handle = copy.copy(self._handle)
        runtime_handle["params"] = params

        obs = np.asarray(observation, dtype=np.float32)
        single = obs.ndim == 1
        if single:
            obs = obs[None, :]
        obs_dim = int(self._profile.observation_shape[0])
        _require(obs.ndim == 2 and obs.shape[1] == obs_dim,
                 f"observation shape {obs.shape} incompatible with obs_dim {obs_dim}")
        if not np.isfinite(obs).all():
            raise SlowGRUMountError("observation carries non-finite values (fail closed)")
        batch = int(obs.shape[0])

        check = self.validate_memory(memory, batch)
        if not check["ok"]:
            raise SlowGRUMountError(f"memory contract violation: {check['reasons']}")

        # Convert flat memory to nested format for slowgru_runtime
        memory_nested = _flat_to_nested(dict(memory))

        # Pad batch_size=1 to 2 (workaround for x.squeeze() in transformerXL)
        pad = batch < _MIN_BATCH_FOR_NETWORK
        if pad:
            obs = np.repeat(obs, _MIN_BATCH_FOR_NETWORK, axis=0)
            memory_nested = _pad_memory_batch(memory_nested, _MIN_BATCH_FOR_NETWORK)
            eff_batch = _MIN_BATCH_FOR_NETWORK
        else:
            eff_batch = batch

        done_mask = np.zeros((eff_batch,), dtype=np.bool_)

        try:
            action, ms_new, extras = sr.policy_step(
                runtime_handle, obs, memory_nested, done_mask, true_done=None)
        except Exception as exc:
            raise SlowGRUMountError(
                f"slowgru_runtime.policy_step failed: {type(exc).__name__}: {exc}") from exc

        # Strip padding
        if pad:
            ms_new = _strip_padded_output(ms_new, batch)
            extras["logits"] = np.asarray(extras["logits"])[:batch]
            extras["value"] = np.asarray(extras["value"])[:batch]
            action = action[:batch] if hasattr(action, "__getitem__") else action

        new_memory = _nested_to_flat(ms_new)
        logits_np = np.asarray(extras["logits"])
        action_count = int(self._profile.action_count)

        if deterministic:
            action_np = np.argmax(logits_np, axis=-1)
        else:
            _require(rng is not None,
                     "stochastic policy_step requires a numpy rng (Generator)")
            probs = _stable_softmax(logits_np)
            action_np = np.array(
                [int(rng.choice(action_count, p=probs[i])) for i in range(batch)],
                dtype=np.int64)

        out: dict[str, Any] = {
            "action": action_np,
            "logits": logits_np,
            "value": np.asarray(extras["value"]),
            "memory": new_memory,
        }
        if single:
            out["action"] = int(action_np[0])
            out["logits"] = logits_np[0]
            out["value"] = np.asarray(extras["value"])[0]
        return out

    # --------------------------------------------------- training surface (R9)

    def save_full_state(self, output_path: str, train_state: Any,
                        metadata: Mapping[str, Any]) -> str:
        raise NotImplementedError(
            "PENDING(R9 training surface): SlowGRUStudentAdapter is READ-ONLY this round; "
            "full-state save (params+optimizer+step+rng+memory) is not implemented. "
            "Zero training updates are authorized for this mount.")

    def restore_full_state(self, checkpoint_path: str) -> Mapping[str, Any]:
        raise NotImplementedError(
            "PENDING(R9 training surface): SlowGRUStudentAdapter is READ-ONLY this round; "
            "full-state restore (inverse of save_full_state) is not implemented. "
            "Read-only CC3 loading goes through load_full_state.")
