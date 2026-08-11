"""CC2 follow-up P0-12: full-state round-trip attestation.

A checkpoint that LOADS non-empty but restores only part of the state
proves nothing. The full-state round-trip attestation binds the WHOLE
state list and the actual restore evidence::

    identity = build_full_state_checkpoint_identity(...)   # state list
    attested = attest_full_state_round_trip(identity, ...) # restore proof
    verify_full_state_round_trip(attested, identity, ctx)

The full state list (``FULL_STATE_FIELDS``) covers params, optimizer
state, global/update/optimizer steps, training + env RNG, EnvState,
wrapper state, previous action/reward, policy memory/history, the
Student identity, the anchor manifest hash, the formal asset registry
hash, the E1 window / selection / verified batch hashes and the
source commit.

Mechanical invariants on consumption:

* the restore must be a FRESH subprocess restore
  (``fresh_process_restored=True``; a same-process self-report fails
  closed);
* the leaf-by-leaf comparison must match the restored state identity
  (``leaf_comparison_hash``);
* the next-policy-step replay must be IDENTICAL to the pre-save
  replay (``replay_identical=True``).

This round performs NO real checkpointing: the production surface is
impossible (empty whitelist); the TEST_ONLY contract exercises the
attestation shape with conspicuously-marked synthetic evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical import canonical_sha256
from .schemas import E1SchemaError

#: the FULL state list (order is meaningful — identity of the whole
#: checkpoint, not a summary)
FULL_STATE_FIELDS = (
    "params_hash",
    "optimizer_state_hash",
    "global_env_steps",
    "update_step",
    "optimizer_step",
    "training_rng_hash",
    "env_rng_hash",
    "env_state_hash",
    "wrapper_state_hash",
    "prev_action_reward_hash",
    "policy_memory_history_hash",
    "student_identity_hash",
    "anchor_manifest_hash",
    "formal_asset_registry_hash",
    "window_hash",
    "selection_hash",
    "verified_batch_hash",
    "source_commit",
)

#: synthetic TEST_ONLY signer (greppable)
SYNTHETIC_TEST_ONLY_ROUNDTRIP_SIGNER = (
    "SYNTHETIC_TEST_ONLY_ROUNDTRIP_SIGNER"
)

#: supervisor-owned production signer whitelist — EMPTY this round
AUTHORIZED_ROUNDTRIP_SIGNERS: tuple = ()

#: attestation version
ROUNDTRIP_ATTESTATION_VERSION = "e1-full-state-roundtrip-v1"

# fail-closed codes (greppable)
ROUNDTRIP_BAD_TYPE = "ROUNDTRIP_BAD_TYPE"
ROUNDTRIP_SIGNER_UNAUTHORIZED = "ROUNDTRIP_SIGNER_UNAUTHORIZED"
ROUNDTRIP_TEST_ONLY_REJECTED = "ROUNDTRIP_TEST_ONLY_REJECTED"
ROUNDTRIP_HASH_MISMATCH = "ROUNDTRIP_HASH_MISMATCH"
ROUNDTRIP_NO_FRESH_RESTORE = "ROUNDTRIP_NO_FRESH_RESTORE"
ROUNDTRIP_REPLAY_MISMATCH = "ROUNDTRIP_REPLAY_MISMATCH"
ROUNDTRIP_LEAF_MISMATCH = "ROUNDTRIP_LEAF_MISMATCH"


class RoundtripAttestationError(E1SchemaError):
    """Fail-closed round-trip violation; ``code`` is greppable."""


def _require_sha64(value: Any, name: str, ctx: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RoundtripAttestationError(
            ROUNDTRIP_BAD_TYPE,
            f"{ctx}: {name} must be a 64-hex hash, got {value!r}",
        )
    return value


@dataclass(frozen=True)
class FullStateCheckpointIdentity:
    """The full-state checkpoint description (immutable, hash-bound)."""

    params_hash: str
    optimizer_state_hash: str
    global_env_steps: int
    update_step: int
    optimizer_step: int
    training_rng_hash: str
    env_rng_hash: str
    env_state_hash: str
    wrapper_state_hash: str
    prev_action_reward_hash: str
    policy_memory_history_hash: str
    student_identity_hash: str
    anchor_manifest_hash: str
    formal_asset_registry_hash: str
    window_hash: str
    selection_hash: str
    verified_batch_hash: str
    source_commit: str
    checkpoint_hash: str


def compute_checkpoint_identity_hash(
    *,
    params_hash: str,
    optimizer_state_hash: str,
    global_env_steps: int,
    update_step: int,
    optimizer_step: int,
    training_rng_hash: str,
    env_rng_hash: str,
    env_state_hash: str,
    wrapper_state_hash: str,
    prev_action_reward_hash: str,
    policy_memory_history_hash: str,
    student_identity_hash: str,
    anchor_manifest_hash: str,
    formal_asset_registry_hash: str,
    window_hash: str,
    selection_hash: str,
    verified_batch_hash: str,
    source_commit: str,
) -> str:
    return canonical_sha256(
        {
            "roundtrip_version": ROUNDTRIP_ATTESTATION_VERSION,
            "params_hash": params_hash,
            "optimizer_state_hash": optimizer_state_hash,
            "global_env_steps": global_env_steps,
            "update_step": update_step,
            "optimizer_step": optimizer_step,
            "training_rng_hash": training_rng_hash,
            "env_rng_hash": env_rng_hash,
            "env_state_hash": env_state_hash,
            "wrapper_state_hash": wrapper_state_hash,
            "prev_action_reward_hash": prev_action_reward_hash,
            "policy_memory_history_hash": policy_memory_history_hash,
            "student_identity_hash": student_identity_hash,
            "anchor_manifest_hash": anchor_manifest_hash,
            "formal_asset_registry_hash": formal_asset_registry_hash,
            "window_hash": window_hash,
            "selection_hash": selection_hash,
            "verified_batch_hash": verified_batch_hash,
            "source_commit": source_commit,
        }
    )


def build_full_state_checkpoint_identity(
    *,
    params_hash: str,
    optimizer_state_hash: str,
    global_env_steps: int,
    update_step: int,
    optimizer_step: int,
    training_rng_hash: str,
    env_rng_hash: str,
    env_state_hash: str,
    wrapper_state_hash: str,
    prev_action_reward_hash: str,
    policy_memory_history_hash: str,
    student_identity_hash: str,
    anchor_manifest_hash: str,
    formal_asset_registry_hash: str,
    window_hash: str,
    selection_hash: str,
    verified_batch_hash: str,
    source_commit: str,
) -> FullStateCheckpointIdentity:
    """Build the full-state identity fail-closed on every field."""
    ctx = "roundtrip.build_identity"
    fields = {}
    for name in (
        "params_hash",
        "optimizer_state_hash",
        "training_rng_hash",
        "env_rng_hash",
        "env_state_hash",
        "wrapper_state_hash",
        "prev_action_reward_hash",
        "policy_memory_history_hash",
        "student_identity_hash",
        "anchor_manifest_hash",
        "formal_asset_registry_hash",
        "window_hash",
        "selection_hash",
        "verified_batch_hash",
    ):
        fields[name] = _require_sha64(
            locals()[name], name, ctx
        )
    for name in ("global_env_steps", "update_step", "optimizer_step"):
        value = locals()[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RoundtripAttestationError(
                ROUNDTRIP_BAD_TYPE,
                f"{ctx}: {name} must be a non-negative int, got {value!r}",
            )
        fields[name] = value
    if not isinstance(source_commit, str) or not source_commit.strip():
        raise RoundtripAttestationError(
            ROUNDTRIP_BAD_TYPE,
            f"{ctx}: source_commit must be a non-empty str, got "
            f"{source_commit!r}",
        )
    fields["source_commit"] = source_commit.strip()
    checkpoint_hash = compute_checkpoint_identity_hash(
        params_hash=fields["params_hash"],
        optimizer_state_hash=fields["optimizer_state_hash"],
        global_env_steps=fields["global_env_steps"],
        update_step=fields["update_step"],
        optimizer_step=fields["optimizer_step"],
        training_rng_hash=fields["training_rng_hash"],
        env_rng_hash=fields["env_rng_hash"],
        env_state_hash=fields["env_state_hash"],
        wrapper_state_hash=fields["wrapper_state_hash"],
        prev_action_reward_hash=fields["prev_action_reward_hash"],
        policy_memory_history_hash=fields["policy_memory_history_hash"],
        student_identity_hash=fields["student_identity_hash"],
        anchor_manifest_hash=fields["anchor_manifest_hash"],
        formal_asset_registry_hash=fields["formal_asset_registry_hash"],
        window_hash=fields["window_hash"],
        selection_hash=fields["selection_hash"],
        verified_batch_hash=fields["verified_batch_hash"],
        source_commit=fields["source_commit"],
    )
    return FullStateCheckpointIdentity(
        params_hash=fields["params_hash"],
        optimizer_state_hash=fields["optimizer_state_hash"],
        global_env_steps=fields["global_env_steps"],
        update_step=fields["update_step"],
        optimizer_step=fields["optimizer_step"],
        training_rng_hash=fields["training_rng_hash"],
        env_rng_hash=fields["env_rng_hash"],
        env_state_hash=fields["env_state_hash"],
        wrapper_state_hash=fields["wrapper_state_hash"],
        prev_action_reward_hash=fields["prev_action_reward_hash"],
        policy_memory_history_hash=fields["policy_memory_history_hash"],
        student_identity_hash=fields["student_identity_hash"],
        anchor_manifest_hash=fields["anchor_manifest_hash"],
        formal_asset_registry_hash=fields["formal_asset_registry_hash"],
        window_hash=fields["window_hash"],
        selection_hash=fields["selection_hash"],
        verified_batch_hash=fields["verified_batch_hash"],
        source_commit=fields["source_commit"],
        checkpoint_hash=checkpoint_hash,
    )


@dataclass(frozen=True)
class FullStateRoundTripAttestation:
    """The restore evidence for ONE full-state checkpoint (immutable)."""

    checkpoint_hash: str
    restored_state_hash: str
    leaf_comparison_hash: str
    next_policy_step_hash: str
    fresh_process_restored: bool
    replay_identical: bool
    signer_id: str
    verifier_hash: str
    attestation_hash: str
    test_only: bool


def attest_full_state_round_trip(
    identity: Any,
    *,
    restored_state_hash: str,
    leaf_comparison_hash: str,
    next_policy_step_hash: str,
    fresh_process_restored: bool,
    replay_identical: bool,
    signer_id: str,
    test_only: bool = False,
    ctx: str = "roundtrip.attest",
) -> FullStateRoundTripAttestation:
    """Attest the full-state restore evidence fail-closed."""
    if not isinstance(identity, FullStateCheckpointIdentity):
        raise RoundtripAttestationError(
            ROUNDTRIP_BAD_TYPE,
            f"{ctx}: identity must be a FullStateCheckpointIdentity, "
            f"got {type(identity).__name__}",
        )
    if test_only:
        if signer_id != SYNTHETIC_TEST_ONLY_ROUNDTRIP_SIGNER:
            raise RoundtripAttestationError(
                ROUNDTRIP_TEST_ONLY_REJECTED,
                f"{ctx}: TEST_ONLY round-trips must be signed by "
                f"{SYNTHETIC_TEST_ONLY_ROUNDTRIP_SIGNER!r}, got "
                f"{signer_id!r}",
            )
    elif signer_id not in AUTHORIZED_ROUNDTRIP_SIGNERS:
        raise RoundtripAttestationError(
            ROUNDTRIP_SIGNER_UNAUTHORIZED,
            f"{ctx}: signer {signer_id!r} is not on the supervisor-"
            "owned round-trip whitelist (EMPTY this round)",
        )
    restored = _require_sha64(
        restored_state_hash, "restored_state_hash", ctx
    )
    leaf = _require_sha64(leaf_comparison_hash, "leaf_comparison_hash", ctx)
    _require_sha64(next_policy_step_hash, "next_policy_step_hash", ctx)
    if not isinstance(fresh_process_restored, bool):
        raise RoundtripAttestationError(
            ROUNDTRIP_BAD_TYPE,
            f"{ctx}: fresh_process_restored must be bool",
        )
    if not fresh_process_restored:
        raise RoundtripAttestationError(
            ROUNDTRIP_NO_FRESH_RESTORE,
            f"{ctx}: the state was NOT restored in a fresh subprocess; "
            "a same-process self-report never attests a round-trip",
        )
    if not isinstance(replay_identical, bool):
        raise RoundtripAttestationError(
            ROUNDTRIP_BAD_TYPE,
            f"{ctx}: replay_identical must be bool",
        )
    if not replay_identical:
        raise RoundtripAttestationError(
            ROUNDTRIP_REPLAY_MISMATCH,
            f"{ctx}: the next-policy-step replay differs from the "
            "pre-save replay; the restore changed the behavior",
        )
    if leaf != restored:
        raise RoundtripAttestationError(
            ROUNDTRIP_LEAF_MISMATCH,
            f"{ctx}: the leaf comparison {leaf!r} != the restored "
            f"state {restored!r}; the state did not survive restore "
            "leaf-for-leaf",
        )
    verifier_hash = canonical_sha256(
        {"verifier": ROUNDTRIP_ATTESTATION_VERSION}
    )
    attestation_hash = canonical_sha256(
        {
            "checkpoint_hash": identity.checkpoint_hash,
            "restored_state_hash": restored,
            "leaf_comparison_hash": leaf,
            "next_policy_step_hash": next_policy_step_hash,
            "fresh_process_restored": fresh_process_restored,
            "replay_identical": replay_identical,
            "signer_id": signer_id,
            "verifier_hash": verifier_hash,
            "test_only": test_only,
        }
    )
    return FullStateRoundTripAttestation(
        checkpoint_hash=identity.checkpoint_hash,
        restored_state_hash=restored,
        leaf_comparison_hash=leaf,
        next_policy_step_hash=next_policy_step_hash,
        fresh_process_restored=fresh_process_restored,
        replay_identical=replay_identical,
        signer_id=signer_id,
        verifier_hash=verifier_hash,
        attestation_hash=attestation_hash,
        test_only=test_only,
    )


def verify_full_state_round_trip(
    attested: Any,
    identity: Any,
    ctx: str = "roundtrip.verify",
) -> None:
    """Re-derive the attestation against the identity fail-closed."""
    if not isinstance(attested, FullStateRoundTripAttestation):
        raise RoundtripAttestationError(
            ROUNDTRIP_BAD_TYPE,
            f"{ctx}: expected a FullStateRoundTripAttestation, got "
            f"{type(attested).__name__}",
        )
    if not isinstance(identity, FullStateCheckpointIdentity):
        raise RoundtripAttestationError(
            ROUNDTRIP_BAD_TYPE,
            f"{ctx}: identity must be a FullStateCheckpointIdentity, "
            f"got {type(identity).__name__}",
        )
    if attested.checkpoint_hash != identity.checkpoint_hash:
        raise RoundtripAttestationError(
            ROUNDTRIP_HASH_MISMATCH,
            f"{ctx}: attested checkpoint {attested.checkpoint_hash!r} != "
            f"identity {identity.checkpoint_hash!r}",
        )
    # tamper check FIRST: a recomputed-hash mismatch is always reported
    # as ROUNDTRIP_HASH_MISMATCH, never masked by an invariant failure
    recomputed = canonical_sha256(
        {
            "checkpoint_hash": identity.checkpoint_hash,
            "restored_state_hash": attested.restored_state_hash,
            "leaf_comparison_hash": attested.leaf_comparison_hash,
            "next_policy_step_hash": attested.next_policy_step_hash,
            "fresh_process_restored": attested.fresh_process_restored,
            "replay_identical": attested.replay_identical,
            "signer_id": attested.signer_id,
            "verifier_hash": attested.verifier_hash,
            "test_only": attested.test_only,
        }
    )
    if recomputed != attested.attestation_hash:
        raise RoundtripAttestationError(
            ROUNDTRIP_HASH_MISMATCH,
            f"{ctx}: attestation_hash {attested.attestation_hash!r} != "
            f"recomputed {recomputed!r} (tampered)",
        )
    if attested.leaf_comparison_hash != attested.restored_state_hash:
        raise RoundtripAttestationError(
            ROUNDTRIP_LEAF_MISMATCH,
            f"{ctx}: leaf comparison {attested.leaf_comparison_hash!r} "
            f"!= restored {attested.restored_state_hash!r}",
        )
    if not attested.fresh_process_restored:
        raise RoundtripAttestationError(
            ROUNDTRIP_NO_FRESH_RESTORE,
            f"{ctx}: no fresh subprocess restore is attested",
        )
    if not attested.replay_identical:
        raise RoundtripAttestationError(
            ROUNDTRIP_REPLAY_MISMATCH,
            f"{ctx}: next-policy-step replay differs from the pre-save "
            "replay",
        )
