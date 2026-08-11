"""Read-only RMT16 StudentAdapter (Stage 3 / R4b checkpoint-side mount).

Mounts a CC2-trained ``ActorCriticTransformerRMT16`` Student strictly
READ-ONLY against the shared Stage-2 ``StudentAdapter`` protocol:

  * params come from a real CC2 ``full_state.pkl`` via ``checkpoint_codec``
    (file-SHA gate + params-tree SHA recomputed at load time, never trusted
    blindly);
  * network hyperparameters come from the SHA-bound CC2 driver source via
    AST-literal parsing (never executed, never guessed, never defaulted) —
    real CC2 manifests carry ``config == {}`` by design;
  * the forward path is CC2's own vendored ``rmt_step_forward`` (the single
    shared per-step transition) — this module NEVER reimplements RMT/GTrXL
    state dynamics;
  * ZERO parameter updates.  ``save_full_state`` / ``restore_full_state``
    (the training surface, R9) raise ``NotImplementedError`` this round.

R4b scope statement: a PASS here is the checkpoint-side restore proof only.
Combined with the Stage-1 R4a env-side PASS it still does NOT constitute the
R4c joint fresh-process proof (``COMBINED_FRESH_PROCESS_RESTORE`` stays
false until that combined run is executed).

This module imports jax/flax lazily (function-level) so that importing it at
module level never crashes a jax-less interpreter; the vendored architecture
modules (which do import jax) are only touched when a network is actually
built.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .architectures.rmt16_provenance import (
    CARRY_MODES,
    FROZEN_RMT16_CFG,
    REQUIRED_CFG_FIELDS,
    DriverSourceError,
    load_rmt16_cfg_from_driver_source,
    verify_frozen_cfg,
)
from .checkpoint_codec import FORMAT_CC2_PKL, file_sha256, load_cc2_pkl
from .identity import validate_identity
from .protocol import ActionSpec, CheckpointSpec, MemoryFieldSpec, ObsSpec
from .registry import StudentProfile


class RMT16MountError(RuntimeError):
    """Raised on any RMT16 mount/identity violation (fail closed)."""


# memory_mode (profile contract) -> CC2 manifest carry_mode spelling.
CARRY_MODE_BY_MEMORY_MODE = {"PERSISTENT": "persistent", "RESET128": "reset128"}

# Memory field keys are FLAT dotted names, exactly as recorded in the
# student profile yamls (so MemorySpec.spec_hash agrees).
MEMORY_FIELD_KEYS = (
    "memories", "mem_mask", "mem_idx",
    "rmt.mem_tokens", "rmt.seg_buf", "rmt.seg_count",
)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RMT16MountError(msg)


def _craftax_action_count() -> int:
    """Cross-validation anchor for action_count (fail closed if unavailable)."""
    try:
        from craftax.craftax.constants import Action  # craftax 1.4.5 layout
    except ImportError:
        try:
            from craftax.constants import Action  # defensive: other layouts
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RMT16MountError(
                f"BLOCKED_ENVIRONMENT: craftax required for action_count "
                f"cross-validation: {exc}") from exc
    return len(Action)


def _stable_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


class RMT16StudentAdapter:
    """Read-only mount of the CC2 RMT16 Student family (PERSISTENT/RESET128).

    Constructor arguments
    ---------------------
    profile : StudentProfile
        Logical identity + interface parameters (from the shared registry).
    driver_source_path / expected_driver_sha256 :
        The SHA-bound CC2 driver source used to recover the frozen network
        hyperparameters.  ``load_full_state`` fails closed if they are unset
        (never guesses, never defaults).
    """

    def __init__(self, profile: StudentProfile, *,
                 driver_source_path: str | None = None,
                 expected_driver_sha256: str | None = None) -> None:
        _require(isinstance(profile, StudentProfile),
                 f"RMT16StudentAdapter needs a StudentProfile, got {type(profile)!r}")
        _require(profile.architecture_family == "RMT16",
                 f"architecture_family must be RMT16, got {profile.architecture_family!r}")
        _require(profile.checkpoint_format == FORMAT_CC2_PKL,
                 f"RMT16StudentAdapter mounts CC2_PKL only, got {profile.checkpoint_format!r}")
        carry_expected = CARRY_MODE_BY_MEMORY_MODE.get(profile.memory_mode)
        _require(carry_expected in CARRY_MODES,
                 f"profile memory_mode {profile.memory_mode!r} has no carry_mode mapping "
                 f"(known: {sorted(CARRY_MODE_BY_MEMORY_MODE)})")
        carry = profile.notes.get("carry_mode")
        _require(carry in CARRY_MODES,
                 f"profile notes must carry 'carry_mode' in {list(CARRY_MODES)}, got {carry!r}")
        _require(carry == carry_expected,
                 f"profile carry_mode {carry!r} contradicts memory_mode-derived "
                 f"{carry_expected!r} (fail closed)")
        self._profile = profile
        self._carry_mode = str(carry)
        self._driver_source_path = driver_source_path
        self._expected_driver_sha256 = expected_driver_sha256
        # set by load_full_state
        self._cfg: dict | None = None
        self._network = None
        self._rmt_cfg = None
        self._apply_eval_rmt = None
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
            format=FORMAT_CC2_PKL,
            params_sha256=self._profile.params_sha256,
            source_commit=self._profile.source_commit,
            contains_optimizer=False,
            contains_rng=False,
            contains_memory=False,
            notes={
                "checkpoint_file_sha256": str(self._profile.notes.get("checkpoint_file_sha256", "")),
                "r4c_joint_proof_status": (
                    "unavailable: CC2 pkl carries params+manifest only "
                    "(no optimizer/rng/policy-memory)"),
            },
        )

    # ------------------------------------------------------------- load gates

    def configure_driver_source(self, driver_source_path: str,
                                expected_driver_sha256: str) -> None:
        """Supply the SHA-bound driver source post-construction (fail closed
        at load time if never supplied)."""
        _require(bool(driver_source_path), "driver_source_path is empty")
        _require(bool(expected_driver_sha256), "expected_driver_sha256 is empty")
        self._driver_source_path = str(driver_source_path)
        self._expected_driver_sha256 = str(expected_driver_sha256)

    def load_full_state(self, checkpoint_path: str, expected_identity) -> Mapping[str, Any]:
        """REAL_CHECKPOINT_LOADED gate chain (every step fail-closed).

        G0 identity: expected_identity validates and equals the profile identity.
        G1 driver cfg: SHA-bound driver AST parse == frozen expectation.
        G2 file SHA: bytes on disk == profile-recorded checkpoint_file_sha256.
        G3 codec: CC2 structure + params-tree sha recomputed == expected.
        G4 manifest: carry_mode/step/config/phase4a_v2 gates.
        G5 structure: params tree == network.init reference tree (shapes/dtypes),
           plus obs/action dims read back from the REAL params leaves and
           cross-validated against profile + craftax len(Action).
        G6 honesty: optimizer/rng/policy-memory absence recorded (CC2 pkl has none).
        """
        gates: dict[str, Any] = {}

        # G0 identity completeness + agreement
        expected_identity = validate_identity(expected_identity)
        profile_identity = self._profile.expected_identity()
        _require(expected_identity.identity_hash() == profile_identity.identity_hash(),
                 f"identity mismatch: expected_identity hash {expected_identity.identity_hash()[:16]}… "
                 f"!= profile identity hash {profile_identity.identity_hash()[:16]}… "
                 "(candidate/step/params-sha/memory-spec must all agree; never guess)")
        gates["G0_identity"] = {"identity_hash": profile_identity.identity_hash(),
                                "candidate_id": profile_identity.candidate_id}

        # G1 frozen hyperparameters from the SHA-bound driver source
        _require(bool(self._driver_source_path) and bool(self._expected_driver_sha256),
                 "FAIL CLOSED: driver_source_path/expected_driver_sha256 not configured — "
                 "cannot rebuild the RMT16 network without the SHA-bound driver Cfg "
                 "(no guessing, no defaults)")
        try:
            cfg, driver_sha = load_rmt16_cfg_from_driver_source(
                self._driver_source_path, self._expected_driver_sha256)
            verify_frozen_cfg(cfg)
        except DriverSourceError as exc:
            raise RMT16MountError(str(exc)) from exc
        gates["G1_driver_cfg"] = {"driver_sha256": driver_sha,
                                  "cfg_fields": {k: cfg[k] for k in REQUIRED_CFG_FIELDS}}

        # G2 file SHA
        expected_file_sha = str(self._profile.notes.get("checkpoint_file_sha256", ""))
        _require(len(expected_file_sha) == 64,
                 "profile notes must carry a 64-hex checkpoint_file_sha256 (fail closed)")
        fsha = file_sha256(checkpoint_path)
        _require(fsha == expected_file_sha,
                 f"checkpoint file SHA256 {fsha[:16]}… != profile-recorded {expected_file_sha[:16]}… "
                 f"for {checkpoint_path!r} (foreign/tampered/moved file; never guess)")
        gates["G2_file_sha256"] = fsha

        # G3 codec load with params-tree sha recomputation
        loaded = load_cc2_pkl(checkpoint_path,
                              expected_params_sha256=expected_identity.params_sha256)
        gates["G3_params_sha256"] = loaded.params_sha256

        # G4 manifest gates
        manifest = loaded.manifest
        _require(manifest.get("carry_mode") == self._carry_mode,
                 f"manifest carry_mode {manifest.get('carry_mode')!r} != profile "
                 f"{self._carry_mode!r} (checkpoint does not belong to this candidate)")
        _require(int(manifest.get("step", -1)) == expected_identity.global_step,
                 f"manifest step {manifest.get('step')!r} != identity global_step "
                 f"{expected_identity.global_step}")
        mcfg = manifest.get("config")
        if mcfg:
            _require(isinstance(mcfg, dict), "manifest['config'] is present but not a dict")
            clashes = {k: (mcfg[k], cfg[k]) for k in mcfg if k in cfg and mcfg[k] != cfg[k]}
            _require(not clashes,
                     f"manifest config disagrees with the SHA-bound driver Cfg "
                     f"(key: (manifest, driver)) {clashes} — foreign/tampered checkpoint")
        p4 = manifest.get("phase4a_v2")
        if isinstance(p4, dict) and p4.get("segment_len") is not None:
            _require(int(p4["segment_len"]) == int(cfg["num_steps"]),
                     f"manifest phase4a_v2.segment_len {p4['segment_len']!r} != driver cfg "
                     f"num_steps {cfg['num_steps']!r} (RMT segment geometry mismatch)")
        gates["G4_manifest"] = {"carry_mode": manifest.get("carry_mode"),
                                "step": manifest.get("step"),
                                "arm": manifest.get("arm", "")}

        # G5 build network from the frozen cfg; structure-gate the params tree
        import jax
        import jax.numpy as jnp
        from flax.core import unfreeze
        from .architectures import rmt16_memory, rmt16_network

        network = rmt16_network.ActorCriticTransformerRMT16(
            action_dim=int(self._profile.action_count),
            activation=cfg["activation"],
            encoder_size=cfg["embed_size"],
            hidden_layers=cfg["hidden_layers"],
            num_heads=cfg["num_heads"],
            qkv_features=cfg["qkv_features"],
            num_layers=cfg["num_layers"],
            gating=cfg["gating"],
            gating_bias=cfg["gating_bias"],
            rmt_num_tokens=cfg["rmt_num_tokens"])
        rmt_cfg = rmt16_memory.RMT16Config(
            num_tokens=cfg["rmt_num_tokens"], segment_len=cfg["num_steps"],
            encoder_size=cfg["embed_size"])

        params = jax.tree_util.tree_map(jnp.asarray, loaded.params)

        # profile memory-spec consistency against the frozen cfg (fail closed)
        expected_shapes = self._cfg_memory_shapes(cfg)
        spec = self._profile.memory_spec()
        for name, (trailing, dtype) in expected_shapes.items():
            _require(name in spec.fields,
                     f"profile memory spec missing field {name!r} required by the RMT16 contract")
            fspec: MemoryFieldSpec = spec.fields[name]
            _require(tuple(fspec.shape[1:]) == tuple(trailing),
                     f"profile memory field {name!r} shape {fspec.shape} != cfg-derived "
                     f"(None, {trailing})")
            _require(fspec.dtype == dtype,
                     f"profile memory field {name!r} dtype {fspec.dtype!r} != {dtype!r}")

        # reference init for the tree-structure gate (PRNGKey(0) is a
        # deterministic structure probe only; never used for policy sampling)
        obs_dim = int(self._profile.observation_shape[0])
        ref = network.init(
            jax.random.PRNGKey(0),
            jnp.zeros((2, cfg["window_mem"], cfg["num_layers"], cfg["embed_size"])),
            jnp.zeros((2, obs_dim)),
            jnp.zeros((2, cfg["num_heads"], 1, cfg["window_mem"] + 1), jnp.bool_),
            mem_tokens=jnp.zeros((2, cfg["rmt_num_tokens"], cfg["embed_size"])),
            seg_buf=jnp.zeros((2, cfg["num_steps"], cfg["embed_size"])),
            method=network.init_all)
        # unfreeze: the pickle carries plain dicts; flax init returns FrozenDict
        # (a distinct pytree node type) — compare like with like.
        ref_params = unfreeze(ref["params"])
        _require(jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(ref_params),
                 "checkpoint params tree structure != network.init reference structure "
                 "(checkpoint does not match ActorCriticTransformerRMT16 with the frozen cfg)")
        leaves, ref_leaves = (jax.tree_util.tree_leaves(params),
                              jax.tree_util.tree_leaves(ref_params))
        for i, (leaf, ref_leaf) in enumerate(zip(leaves, ref_leaves)):
            _require(np.shape(leaf) == np.shape(ref_leaf) and np.asarray(leaf).dtype == np.asarray(ref_leaf).dtype,
                     f"params leaf #{i} shape/dtype {np.shape(leaf)}/{np.asarray(leaf).dtype} != "
                     f"reference {np.shape(ref_leaf)}/{np.asarray(ref_leaf).dtype}")

        # obs/action dims read back from the REAL params leaves
        enc_kernel = np.asarray(params["transformer"]["encoder"]["kernel"])
        actor_kernel = np.asarray(params["actor_out"]["kernel"])
        _require(enc_kernel.shape[0] == obs_dim,
                 f"checkpoint encoder input dim {enc_kernel.shape[0]} != profile observation "
                 f"dim {obs_dim} (obs 8335 = 8268 MiniCraftaxTrain + 67 multitask embedding)")
        _require(actor_kernel.shape[1] == int(self._profile.action_count),
                 f"checkpoint actor output dim {actor_kernel.shape[1]} != profile action_count "
                 f"{self._profile.action_count}")
        craftax_actions = _craftax_action_count()
        _require(craftax_actions == int(self._profile.action_count),
                 f"craftax len(Action) {craftax_actions} != profile action_count "
                 f"{self._profile.action_count}")
        gates["G5_structure"] = {
            "encoder_kernel_shape": list(enc_kernel.shape),
            "actor_out_kernel_shape": list(actor_kernel.shape),
            "observation_dim": obs_dim,
            "action_count": int(self._profile.action_count),
            "craftax_len_action": craftax_actions,
            "num_params_leaves": len(leaves),
        }

        # G6 honesty of absence (CC2 pkl carries params+manifest ONLY)
        gates["G6_absent_components"] = {
            "optimizer": "ABSENT_IN_CHECKPOINT",
            "train_rng": "ABSENT_IN_CHECKPOINT",
            "policy_memory": "ABSENT_IN_CHECKPOINT (memory re-derives from the CC2 reset "
                             "convention via initial_memory)",
        }

        # mount is now live for read-only forward
        from .architectures import rmt16_anchor
        self._cfg = cfg
        self._network = network
        self._rmt_cfg = rmt_cfg
        self._apply_eval_rmt = rmt16_anchor.make_apply_eval_rmt(network)
        self._loaded = {
            "params_sha256": loaded.params_sha256,
            "file_sha256": fsha,
            "driver_sha256": driver_sha,
            "global_step": loaded.global_step,
        }
        return {
            "params": params,
            "manifest": dict(manifest),
            "params_sha256": loaded.params_sha256,
            "file_sha256": fsha,
            "driver_source_sha256": driver_sha,
            "cfg": {k: cfg[k] for k in sorted(cfg)},
            "carry_mode": self._carry_mode,
            "global_step": loaded.global_step,
            "contains_optimizer": False,
            "contains_rng": False,
            "contains_policy_memory": False,
            "gates": gates,
            "r4c_joint_proof_status": {
                "params": "restored+hash-bound (R4b)",
                "optimizer": "ABSENT_IN_CHECKPOINT -> joint proof unavailable this round",
                "train_rng": "ABSENT_IN_CHECKPOINT -> joint proof unavailable this round",
                "global_step": "restored from manifest['step'] (no optimizer state to pair)",
                "policy_memory": "ABSENT_IN_CHECKPOINT -> re-derived from reset convention",
                "env_state": "R4a-side proof (separate); R4c combined run NOT executed",
            },
        }

    # ----------------------------------------------------------------- memory

    @staticmethod
    def _cfg_memory_shapes(cfg: Mapping[str, Any]) -> dict:
        """Memory field -> (trailing shape, dtype) derived from the frozen cfg."""
        wm, nl, es = int(cfg["window_mem"]), int(cfg["num_layers"]), int(cfg["embed_size"])
        heads, nt, seg = int(cfg["num_heads"]), int(cfg["rmt_num_tokens"]), int(cfg["num_steps"])
        return {
            "memories": ((wm, nl, es), "float32"),
            "mem_mask": ((heads, 1, wm + 1), "bool"),
            "mem_idx": ((), "int32"),
            "rmt.mem_tokens": ((nt, es), "float32"),
            "rmt.seg_buf": ((seg, es), "float32"),
            "rmt.seg_count": ((), "int32"),
        }

    def _require_loaded(self) -> None:
        _require(self._loaded is not None and self._cfg is not None,
                 "adapter not loaded: load_full_state must run first (memory shapes derive "
                 "from the SHA-bound driver cfg, never from defaults)")

    def initial_memory(self, batch_size: int) -> dict:
        self._require_loaded()
        batch_size = int(batch_size)
        _require(batch_size >= 1, f"batch_size must be >= 1, got {batch_size}")
        cfg = self._cfg
        wm, nl, es = int(cfg["window_mem"]), int(cfg["num_layers"]), int(cfg["embed_size"])
        heads, nt, seg = int(cfg["num_heads"]), int(cfg["rmt_num_tokens"]), int(cfg["num_steps"])
        return {
            # CC2 driver reset convention (tier3 CC2RMT16Policy.reset):
            # zeros + mem_idx = window_mem (P2 convention).
            "memories": np.zeros((batch_size, wm, nl, es), dtype=np.float32),
            "mem_mask": np.zeros((batch_size, heads, 1, wm + 1), dtype=np.bool_),
            "mem_idx": np.full((batch_size,), wm, dtype=np.int32),
            "rmt.mem_tokens": np.zeros((batch_size, nt, es), dtype=np.float32),
            "rmt.seg_buf": np.zeros((batch_size, seg, es), dtype=np.float32),
            "rmt.seg_count": np.zeros((batch_size,), dtype=np.int32),
        }

    def validate_memory(self, memory: Any, batch_size: int) -> Mapping[str, Any]:
        reasons: list[str] = []
        if self._loaded is None or self._cfg is None:
            return {"ok": False,
                    "reasons": ["adapter not loaded: load_full_state must run first "
                                "(shapes derive from the SHA-bound driver cfg)"]}
        if not isinstance(memory, Mapping):
            return {"ok": False, "reasons": [f"memory is not a Mapping (got {type(memory)!r})"]}
        expected = self._cfg_memory_shapes(self._cfg)
        keys = set(memory.keys())
        missing = sorted(set(expected) - keys)
        extra = sorted(keys - set(expected))
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
            if arr.dtype != np.dtype(dtype):
                reasons.append(f"field {name!r} dtype {arr.dtype} != expected {dtype}")
        if not reasons:
            wm = int(self._cfg["window_mem"])
            seg = int(self._cfg["num_steps"])
            idx = np.asarray(memory["mem_idx"])
            cnt = np.asarray(memory["rmt.seg_count"])
            if idx.size and (idx.min() < 0 or idx.max() > wm):
                reasons.append(f"mem_idx values outside [0, window_mem={wm}]")
            if cnt.size and (cnt.min() < 0 or cnt.max() > seg):
                reasons.append(f"rmt.seg_count values outside [0, segment_len={seg}]")
            for name in ("memories", "rmt.mem_tokens", "rmt.seg_buf"):
                f = np.asarray(memory[name])
                if f.size and not np.isfinite(f).all():
                    reasons.append(f"field {name!r} carries non-finite values")
        return {"ok": not reasons, "reasons": reasons}

    # ----------------------------------------------------------------- forward

    def policy_step(self, params: Any, observation: Any, memory: Any,
                    previous_action: Any, previous_reward: Any, rng: Any,
                    deterministic: bool) -> Mapping[str, Any]:
        """One READ-ONLY forward through CC2's vendored rmt_step_forward.

        ``previous_action``/``previous_reward`` are accepted for protocol
        compatibility; the RMT16 architecture does not consume them.
        ``done`` is held False for every stepped batch entry: episode
        boundaries are the CALLER's responsibility (fresh initial_memory),
        mirroring the tier3 convention of never stepping past a terminal.
        """
        self._require_loaded()
        import jax
        import jax.numpy as jnp
        from .architectures import rmt16_anchor

        obs = np.asarray(observation, dtype=np.float32)
        single = obs.ndim == 1
        if single:
            obs = obs[None, :]
        obs_dim = int(self._profile.observation_shape[0])
        _require(obs.ndim == 2 and obs.shape[1] == obs_dim,
                 f"observation shape {obs.shape} incompatible with obs_dim {obs_dim}")
        if not np.isfinite(obs).all():
            raise RMT16MountError("observation carries non-finite values (fail closed)")
        batch = int(obs.shape[0])

        check = self.validate_memory(memory, batch)
        if not check["ok"]:
            raise RMT16MountError(f"memory contract violation: {check['reasons']}")

        cfg = self._cfg
        params_j = jax.tree_util.tree_map(jnp.asarray, params)
        memories = jnp.asarray(memory["memories"])
        mem_mask = jnp.asarray(memory["mem_mask"])
        mem_idx = jnp.asarray(memory["mem_idx"])
        rmt_st = {"mem_tokens": jnp.asarray(memory["rmt.mem_tokens"]),
                  "seg_buf": jnp.asarray(memory["rmt.seg_buf"]),
                  "seg_count": jnp.asarray(memory["rmt.seg_count"])}
        done_in = jnp.zeros((batch,), jnp.bool_)
        update_fn = rmt16_anchor.make_update_fn(self._network, params_j)
        (post_memories, new_mask, new_idx, new_rmt_st,
         logits, value, _mem_pre, _entering) = rmt16_anchor.rmt_step_forward(
            self._apply_eval_rmt, params_j, memories, mem_mask, mem_idx, rmt_st,
            jnp.asarray(obs), done_in,
            int(cfg["window_mem"]), int(cfg["num_heads"]), self._rmt_cfg,
            self._carry_mode, update_fn)

        new_memory = {
            "memories": post_memories,
            "mem_mask": new_mask,
            "mem_idx": new_idx,
            "rmt.mem_tokens": new_rmt_st["mem_tokens"],
            "rmt.seg_buf": new_rmt_st["seg_buf"],
            "rmt.seg_count": new_rmt_st["seg_count"],
        }
        logits_np = np.asarray(logits)
        action_count = int(self._profile.action_count)
        if deterministic:
            action = np.argmax(logits_np, axis=-1)
        else:
            _require(rng is not None,
                     "stochastic policy_step requires a numpy rng (Generator)")
            probs = _stable_softmax(logits_np)
            action = np.array(
                [int(rng.choice(action_count, p=probs[i])) for i in range(batch)],
                dtype=np.int64)
        out: dict[str, Any] = {
            "action": action,
            "logits": logits_np,
            "value": np.asarray(value),
            "memory": new_memory,
        }
        if single:
            out["action"] = int(action[0])
            out["logits"] = logits_np[0]
            out["value"] = np.asarray(value)[0]
        return out

    # --------------------------------------------------- training surface (R9)

    def save_full_state(self, output_path: str, train_state: Any,
                        metadata: Mapping[str, Any]) -> str:
        raise NotImplementedError(
            "PENDING(R9 training surface): RMT16StudentAdapter is READ-ONLY this round; "
            "full-state save (params+optimizer+step+rng+memory) is not implemented. "
            "Zero training updates are authorized for this mount.")

    def restore_full_state(self, checkpoint_path: str) -> Mapping[str, Any]:
        raise NotImplementedError(
            "PENDING(R9 training surface): RMT16StudentAdapter is READ-ONLY this round; "
            "full-state restore (inverse of save_full_state) is not implemented. "
            "Read-only CC2 loading goes through load_full_state.")
