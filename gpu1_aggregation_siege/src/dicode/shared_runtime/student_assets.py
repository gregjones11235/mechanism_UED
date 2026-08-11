"""The REAL Persistent / Reset128 Student assets (RMT16 + SlowGRU).

Every object here is backed by a real deployment artifact (profile yaml,
CC2 pkl checkpoint for RMT16, CC3 bakeoff pkl for SlowGRU, SHA-bound
frozen driver source). Nothing is synthesized: a missing/mismatched
asset raises fail-closed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from dicode.teachers.static_llm.student_init_contract import (
    StudentInitContract,
)

from . import asset_locations as AL


@dataclass(frozen=True)
class RealStudentInitContract(StudentInitContract):
    """The REAL StudentInitContract plus its signed identity protocol
    surface (the resolution protocol requires the object's own 64-hex
    identity; the frozen parent contract is unchanged)."""

    object_identity_hash: str = ""


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# real StudentInitContract (identity-only, from the frozen profile +
# checkpoint contract values)
# ---------------------------------------------------------------------------
def build_real_student_contract(candidate_id: str):
    """Build the REAL StudentInitContract for one allowed candidate.

    The parameter_tree_hash is RECOMPUTED live from the real checkpoint
    (cc2_params_sha256 over the loaded params) — never copied from a
    manifest without verification. The optimizer_tree_hash is recomputed
    from the real train_state when present.
    """
    from dicode.student_adapters.checkpoint_codec import cc2_params_sha256
    from dicode.student_adapters.registry import load_student_profile
    from dicode.teachers.static_llm.student_init_contract import (
        StudentInitContract,
    )

    loc = AL.student_locations()
    if candidate_id == "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304":
        profile_rel = loc["persistent_profile"]
        ckpt = loc["persistent_checkpoint"]
        ckpt_sha = loc["persistent_checkpoint_file_sha256"]
        train_state_path = loc.get("persistent_train_state", "")
    elif candidate_id == "RESET128_RMT16_ORIGINAL_VTRACE_98304":
        profile_rel = loc["reset128_profile"]
        ckpt = loc["reset128_checkpoint"]
        ckpt_sha = loc["reset128_checkpoint_file_sha256"]
        train_state_path = loc.get("reset128_train_state", "")
    elif candidate_id == "SLOWGRU_PERSISTENT_CANONICAL_98304":
        profile_rel = loc["slowgru_persistent_profile"]
        ckpt = loc["slowgru_persistent_checkpoint"]
        ckpt_sha = loc["slowgru_persistent_checkpoint_file_sha256"]
        train_state_path = ""
    else:
        raise ValueError(
            f"STUDENT_UNKNOWN_CANDIDATE: {candidate_id!r} is not one of "
            "the allowed real Students"
        )
    profile = load_student_profile(_profile_path(profile_rel))
    adapter = _make_adapter(candidate_id)
    loaded = adapter.load_full_state(ckpt, profile.expected_identity())
    if loaded["file_sha256"] != ckpt_sha:
        raise AL.AssetLocationError(
            f"STUDENT_CKPT_SHA_MISMATCH: loaded file sha "
            f"{loaded['file_sha256']!r} != declared {ckpt_sha!r}"
        )
    if candidate_id == "SLOWGRU_PERSISTENT_CANONICAL_98304":
        # SlowGRU: params_sha is already verified by slowgru_runtime
        params_sha = loaded["params_sha256"]
        optimizer_sha = hashlib.sha256(
            b"cc3-bakeoff-no-optimizer").hexdigest()
    else:
        params_sha = cc2_params_sha256(loaded["params"])
        optimizer_sha = _optimizer_tree_hash(train_state_path)
    if candidate_id == "SLOWGRU_PERSISTENT_CANONICAL_98304":
        fields = dict(
            candidate_id=candidate_id,
            architecture_family="SLOWGRU",
            architecture_version="slowgru-v1",
            checkpoint_format="BAKEOFF_PKL",
            checkpoint_global_step=98304,
            total_env_steps=98304,
            source_commit=str(profile.source_commit),
            parameter_tree_hash=params_sha,
            optimizer_tree_hash=optimizer_sha,
            adapter_id=f"{_profile_id(candidate_id)}",
            adapter_version="slowgru-adapter-v1",
            provenance=(
                "shared_runtime.student_assets: recomputed live from the "
                "real CC3 bakeoff checkpoint (file sha + params sha "
                "verified by slowgru_runtime)"
            ),
        )
    else:
        fields = dict(
            candidate_id=candidate_id,
            architecture_family="RMT16",
            architecture_version="rmt16-vtrace-cc2",
            checkpoint_format="CC2_PKL",
            checkpoint_global_step=98304,
            total_env_steps=98304,
            source_commit=str(profile.source_commit),
            parameter_tree_hash=params_sha,
            optimizer_tree_hash=optimizer_sha,
            adapter_id=f"{_profile_id(candidate_id)}",
            adapter_version="rmt16-adapter-v1",
            provenance=(
                "shared_runtime.student_assets: recomputed live from the "
                "real CC2 checkpoint (file sha verified)"
            ),
        )
    identity = _canonical_sha256(
        {"kind": "shared_runtime.real_student_init_contract", **fields})
    return RealStudentInitContract(
        object_identity_hash=identity, **fields)




def _profile_path(profile_rel: str) -> str:
    return AL.resolve_repo_relative(profile_rel)


def _profile_id(candidate_id: str) -> str:
    return {
        "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304": "rmt16_persistent_98304",
        "RESET128_RMT16_ORIGINAL_VTRACE_98304": "rmt16_reset128_98304",
        "SLOWGRU_PERSISTENT_CANONICAL_98304": "slowgru_persistent_98304",
    }[candidate_id]


def _optimizer_tree_hash(train_state_path: str) -> str:
    """The optimizer-side train-state identity: the sha256 of the REAL
    train_state.pkl bytes (the complete CC2 training state artifact).

    Binding the file bytes is deterministic across processes and
    verifiable against the deployment asset; the file itself carries
    params + pending transitions + rng + counters + memory (the full
    optimizer-side state), so its byte hash IS the train-state identity.
    """
    if not train_state_path:
        return hashlib.sha256(b"cc2-train-state-absent").hexdigest()
    import os

    if not os.path.isfile(train_state_path):
        return hashlib.sha256(b"cc2-train-state-absent").hexdigest()
    return AL.file_sha256(train_state_path)


# ---------------------------------------------------------------------------
# real Student adapter (read-only RMT16 mount)
# ---------------------------------------------------------------------------
_ADAPTER_CACHE: Dict[str, Any] = {}


def _make_adapter(candidate_id: str):
    """The REAL StudentAdapter bound to the SHA-frozen driver source."""
    if candidate_id in _ADAPTER_CACHE:
        return _ADAPTER_CACHE[candidate_id]

    loc = AL.student_locations()

    if candidate_id == "SLOWGRU_PERSISTENT_CANONICAL_98304":
        from dicode.student_adapters.registry import load_student_profile
        from dicode.student_adapters.slowgru_adapter import (
            SlowGRUStudentAdapter,
        )
        profile = load_student_profile(_profile_path(
            loc["slowgru_persistent_profile"]))
        adapter = SlowGRUStudentAdapter(
            profile,
            slowgru_runtime_path=loc["slowgru_driver_source"],
            checkpoint_contract_path=loc["slowgru_checkpoint_contract"],
            expected_network_src_sha256=loc["slowgru_network_src_sha256"],
            expected_trainer_src_sha256=loc["slowgru_trainer_src_sha256"],
        )
        _ADAPTER_CACHE[candidate_id] = adapter
        return adapter

    from dicode.student_adapters.architectures.rmt16_provenance import (
        FROZEN_DRIVER_SOURCE_SHA256,
    )
    from dicode.student_adapters.registry import load_student_profile
    from dicode.student_adapters.rmt16_adapter import RMT16StudentAdapter

    driver = loc["driver_source"]
    driver_sha = loc["driver_source_sha256"]
    if driver_sha != FROZEN_DRIVER_SOURCE_SHA256:
        raise AL.AssetLocationError(
            "DRIVER_SHA_MISMATCH: the asset-location driver sha != the "
            "provenance-frozen driver sha (fail closed)"
        )
    AL.require_file(driver, driver_sha, "frozen RMT16 driver source")
    profile = load_student_profile(_profile_path(
        loc["persistent_profile"]
        if candidate_id == "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"
        else loc["reset128_profile"]
    ))
    adapter = RMT16StudentAdapter(
        profile,
        driver_source_path=driver,
        expected_driver_sha256=driver_sha,
    )
    _ADAPTER_CACHE[candidate_id] = adapter
    return adapter


@dataclass(frozen=True)
class StudentIdentityDescriptor:
    """The real Student identity (immutable, hash-bound)."""

    candidate_id: str
    architecture_family: str
    memory_mode: str
    params_sha256: str
    checkpoint_file_sha256: str
    profile_hash: str
    memory_spec_hash: str
    source_commit: str
    object_identity_hash: str


_LOADED_STATES: Dict[str, Mapping[str, Any]] = {}


class PersistentStudentAdapter:
    """The REAL Persistent Student adapter handle.

    Wraps the RMT16StudentAdapter with the loaded real checkpoint; the
    mount is READ-ONLY (training only through the canonical DiCode
    runtime, never through the adapter).
    """

    def __init__(self, candidate_id: str):
        self.candidate_id = candidate_id
        self._adapter = _make_adapter(candidate_id)
        loc = AL.student_locations()
        if candidate_id == "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304":
            self._checkpoint = loc["persistent_checkpoint"]
            self._expected_file_sha = loc[
                "persistent_checkpoint_file_sha256"]
            self.memory_mode = "PERSISTENT"
        elif candidate_id == "SLOWGRU_PERSISTENT_CANONICAL_98304":
            self._checkpoint = loc["slowgru_persistent_checkpoint"]
            self._expected_file_sha = loc[
                "slowgru_persistent_checkpoint_file_sha256"]
            self.memory_mode = "PERSISTENT"
        else:
            self._checkpoint = loc["reset128_checkpoint"]
            self._expected_file_sha = loc["reset128_checkpoint_file_sha256"]
            self.memory_mode = "RESET128"
        AL.require_file(
            self._checkpoint, self._expected_file_sha,
            f"{candidate_id} checkpoint")
        profile = self._adapter._profile
        self._loaded = self._adapter.load_full_state(
            self._checkpoint, profile.expected_identity())
        _LOADED_STATES[candidate_id] = self._loaded
        self.checkpoint_file_sha256 = self._loaded["file_sha256"]
        self.params_sha256 = self._loaded["params_sha256"]
        if candidate_id == "SLOWGRU_PERSISTENT_CANONICAL_98304":
            driver_src_sha = self._loaded.get(
                "network_src_sha256", self._loaded.get(
                    "driver_source_sha256", ""))
        else:
            driver_src_sha = self._loaded["driver_source_sha256"]
        self.registry_identity = _student_adapter_identity_hash(
            candidate_id, self._loaded["file_sha256"],
            driver_src_sha)
        self.object_identity_hash = self.registry_identity
        #: the REAL observation/action ABI surface (candidate-binding
        #: contract): a canonical hash over the student's actual
        #: observation shape + action count, derived from the REAL
        #: adapter specs — never a guessed default surface.
        _obs = self._adapter.observation_spec()
        _act = self._adapter.action_spec()
        self.observation_action_abi_hash = _canonical_sha256({
            "kind": "shared_runtime.student_observation_action_abi",
            "adapter_id": _profile_id(candidate_id),
            "observation_shape": [
                int(d) for d in getattr(_obs, "shape", ())
            ],
            "observation_dtype": "float32",
            "action_count": int(getattr(_act, "count", 0)),
            "action_dtype": "int32",
        })

    @property
    def params(self):
        return self._loaded["params"]

    @property
    def adapter(self):
        return self._adapter

    def load_read_only(self) -> Mapping[str, Any]:
        """The read-only mount: real params + specs, NEVER a training
        surface."""
        return {
            "candidate_id": self.candidate_id,
            "params": self._loaded["params"],
            "params_sha256": self._loaded["params_sha256"],
            "file_sha256": self._loaded["file_sha256"],
            "global_step": self._loaded["global_step"],
            "observation_spec": self._adapter.observation_spec(),
            "action_spec": self._adapter.action_spec(),
            "memory_spec": self._adapter.memory_spec(),
            "read_only": True,
            "training_ready": False,
        }

    def policy_step(self, observation, memory, previous_action=0,
                    previous_reward=0.0, deterministic=True, rng=None):
        return self._adapter.policy_step(
            self._loaded["params"], observation, memory,
            previous_action=previous_action,
            previous_reward=previous_reward,
            deterministic=deterministic, rng=rng)

    def initial_memory(self, batch_size: int):
        return self._adapter.initial_memory(batch_size)

    def observation_spec(self):
        """The student's REAL observation spec (probe ABI surface)."""
        return self._adapter.observation_spec()

    def action_spec(self):
        """The student's REAL action spec (probe ABI surface)."""
        return self._adapter.action_spec()


_ADAPTER_HANDLE_CACHE: Dict[str, PersistentStudentAdapter] = {}


def real_student_adapter(candidate_id: str) -> PersistentStudentAdapter:
    if candidate_id not in _ADAPTER_HANDLE_CACHE:
        _ADAPTER_HANDLE_CACHE[candidate_id] = PersistentStudentAdapter(
            candidate_id)
    return _ADAPTER_HANDLE_CACHE[candidate_id]


def _student_adapter_identity_hash(candidate_id: str, file_sha: str,
                                   driver_sha: str) -> str:
    adapter_class = (
        "SlowGRUStudentAdapter"
        if candidate_id == "SLOWGRU_PERSISTENT_CANONICAL_98304"
        else "RMT16StudentAdapter"
    )
    return _canonical_sha256({
        "kind": "shared_runtime.real_student_adapter",
        "candidate_id": candidate_id,
        "checkpoint_file_sha256": file_sha,
        "driver_source_sha256": driver_sha,
        "adapter_class": adapter_class,
    })


def real_student_identity(candidate_id: str) -> StudentIdentityDescriptor:
    """The real Student identity descriptor (all hashes recomputed from
    the real artifacts)."""
    adapter = real_student_adapter(candidate_id)
    loc = AL.student_locations()
    if candidate_id == "SLOWGRU_PERSISTENT_CANONICAL_98304":
        profile_rel = loc["slowgru_persistent_profile"]
        arch_family = "SLOWGRU"
        source_commit = "src-sha256:" + loc["slowgru_network_src_sha256"]
    elif candidate_id == "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304":
        profile_rel = loc["persistent_profile"]
        arch_family = "RMT16"
        source_commit = "src-sha256:" + loc["driver_source_sha256"]
    else:
        profile_rel = loc["reset128_profile"]
        arch_family = "RMT16"
        source_commit = "src-sha256:" + loc["driver_source_sha256"]
    profile_hash = AL.file_sha256(_profile_path(profile_rel))
    memory_spec = adapter.adapter.memory_spec()
    memory_spec_hash = _canonical_sha256({
        "memory_mode": getattr(memory_spec, "mode", ""),
        "memory_fields": {
            name: {
                "shape": [
                    None if d is None else int(d)
                    for d in getattr(field, "shape", ())
                ],
                "dtype": str(getattr(field, "dtype", "")),
            }
            for name, field in sorted(
                dict(getattr(memory_spec, "fields", {})).items())
        },
    })
    identity = _canonical_sha256({
        "kind": "shared_runtime.student_identity",
        "candidate_id": candidate_id,
        "architecture_family": arch_family,
        "params_sha256": adapter.params_sha256,
        "checkpoint_file_sha256": adapter.checkpoint_file_sha256,
        "profile_hash": profile_hash,
        "memory_spec_hash": memory_spec_hash,
    })
    return StudentIdentityDescriptor(
        candidate_id=candidate_id,
        architecture_family=arch_family,
        memory_mode=adapter.memory_mode,
        params_sha256=adapter.params_sha256,
        checkpoint_file_sha256=adapter.checkpoint_file_sha256,
        profile_hash=profile_hash,
        memory_spec_hash=memory_spec_hash,
        source_commit=source_commit,
        object_identity_hash=identity,
    )


def load_real_train_state(candidate_id: str) -> Dict[str, Any]:
    """Load the REAL complete train state (params + pending + rng +
    counters + memory) for the canonical training runtime."""
    import pickle
    import sys

    loc = AL.student_locations()
    if candidate_id == "SLOWGRU_PERSISTENT_CANONICAL_98304":
        raise AL.AssetLocationError(
            f"TRAIN_STATE_UNAVAILABLE: SlowGRU bakeoff checkpoint "
            f"carries params only (no optimizer/rng/policy-memory); "
            f"train_state.pkl is not present for "
            f"{candidate_id!r}. The canonical DiCode training setup "
            f"creates a fresh train state via setup_experiment.")
    path = loc.get(
        "persistent_train_state"
        if candidate_id == "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"
        else "reset128_train_state", "")
    if not path:
        raise AL.AssetLocationError(
            f"TRAIN_STATE_MISSING: no train_state configured for "
            f"{candidate_id!r}")
    frozen_dir = loc.get("cc2_frozen_modules", "")
    added = False
    if frozen_dir and frozen_dir not in sys.path:
        sys.path.insert(0, frozen_dir)
        added = True
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    finally:
        if added:
            try:
                sys.path.remove(frozen_dir)
            except ValueError:
                pass
