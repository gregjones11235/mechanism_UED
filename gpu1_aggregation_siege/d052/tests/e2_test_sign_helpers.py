"""TEST_ONLY signing helpers (director smoke handoff round, §六).

Every SIGNING helper that mints attestations / manifests for fixtures
lives HERE — in the tests directory — so that no production module can
call it. The production path consumes the director's signed artifacts
(verify only); it never signs.

* :func:`sign_director_runtime_bundle` — mints a director Runtime Bundle
  manifest (the DIRECTOR's signature is emulated with a TEST_ONLY hash);
* :func:`sign_full_state_round_trip` — mints a FullStateRoundTripResult
  (TEST_ONLY; the production seam rejects locally-signed round-trips and
  consumes only the director-runtime's DirectorVerifiedRunStateRoundTrip).

All fixtures here are TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

from d052.bagr_ued.hashing import canonical_sha256, text_sha256
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.anchor_manifest import SharedAnchorManifest
from d052.feedback_llm_ued.director_runtime_bundle import (
    DIRECTOR_RUNTIME_BUNDLE_VERSION,
    DirectorRuntimeBundleManifest,
)
from d052.feedback_llm_ued.student_binding import (
    FullStateRoundTripResult,
)


def valid_director_bundle_payload(*, anchors=None,
                                  original_task_id="DICODE_ORIGINAL_TASK_V1",
                                  candidate_id=C.STRONG_STUDENT_CANDIDATE_ID):
    """A TEST_ONLY, fully-consistent director Runtime Bundle payload: all
    12 assets present, the smoke update contract, the DiCode 15+1 batch
    binding and the formal-start gate. The anchor hash is the recomputed
    one (build_shared_bundle re-verifies it). The Student's memory/carry
    follow the candidate's legal profile mapping."""
    anchor_manifest = SharedAnchorManifest(
        manifest_id="TEST_ONLY_SHARED_ANCHOR_MANIFEST",
        anchors=list(anchors if anchors is not None
                     else C.GLOBAL_CANONICAL_ANCHOR_IDS),
        frozen=True)
    memory_mode, carry_mode = C.STUDENT_PROFILE_MEMORY_MAP[candidate_id]
    return dict(
        registry_identity=text_sha256("TEST_ONLY_RUNTIME_BUNDLE"),
        formal_asset_registry=text_sha256("TEST_ONLY_FORMAL_ASSET_REGISTRY"),
        student_init_contract=dict(
            candidate_id=candidate_id,
            architecture_family="RMT16",
            memory_family="RMT16_ORIGINAL",
            carry_mode=carry_mode,
            parameter_tree_hash=text_sha256("TEST_ONLY_STUDENT_PARAM_TREE"),
            checkpoint_global_step=0,
            profile_hash=text_sha256("TEST_ONLY_PROFILE"),
            memory_mode=memory_mode,
            memory_spec_hash=text_sha256("TEST_ONLY_MEMORY_SPEC"),
            adapter_identity_hash=text_sha256("TEST_ONLY_ADAPTER"),
            runtime_bundle_hash=text_sha256("TEST_ONLY_RUNTIME_BUNDLE")),
        student_identity=text_sha256("TEST_ONLY_STUDENT_IDENTITY"),
        reference_identity=dict(
            candidate_id="TEST_ONLY_REFERENCE_CANDIDATE",
            parameter_tree_hash=text_sha256("TEST_ONLY_REFERENCE_PARAM_TREE"),
            checkpoint_global_step=0,
            identity_hash=text_sha256("TEST_ONLY_REFERENCE_IDENTITY")),
        candidate_probe_runner=text_sha256(
            "TEST_ONLY_CANDIDATE_PROBE_RUNNER"),
        shared_anchor_manifest=dict(
            manifest_id=anchor_manifest.manifest_id,
            anchors=list(anchor_manifest.anchors),
            frozen=True,
            manifest_hash=anchor_manifest.manifest_hash),
        canonical_dicode_one_update_runtime=text_sha256(
            "TEST_ONLY_DICODE_ONE_UPDATE_RUNTIME"),
        canonical_dicode_run_state_checkpoint=text_sha256(
            "TEST_ONLY_DICODE_RUN_STATE_CHECKPOINT"),
        authorized_six_role_llm_runtime=text_sha256(
            "TEST_ONLY_AUTHORIZED_SIX_ROLE_LLM_RUNTIME"),
        backend_model_identity=dict(backend_id="test.backend.v1",
                                    model_id="test-model.v1"),
        transport_closure=text_sha256("TEST_ONLY_TRANSPORT_CLOSURE"),
        auxiliary_compute_ledger=text_sha256(
            "TEST_ONLY_AUXILIARY_COMPUTE_LEDGER"),
        smoke_semantics=dict(window0_update_delta=0,
                             window1_update_delta=1, total_updates=1),
        batch_binding=dict(
            dynamic_task_count=C.DICODE_CURRICULUM_DYNAMIC,
            non_target_anchor_count=C.DICODE_CURRICULUM_NON_TARGET_ANCHORS,
            curriculum_task_count=C.DICODE_CURRICULUM_TASK_COUNT,
            non_target_anchor_ids=list(C.GLOBAL_CANONICAL_ANCHOR_IDS[:3]),
            original_task_id=original_task_id,
            original_task_proportion=C.DICODE_ORIGINAL_TASK_PROPORTION,
            total_task_count=C.DICODE_BATCH_TOTAL_TASKS),
        formal_start_gate=dict(smoke_only_origin=True,
                               formal_start_requires_human=True))


def valid_director_bundle(*, anchors=None,
                          candidate_id=C.STRONG_STUDENT_CANDIDATE_ID
                          ) -> DirectorRuntimeBundleManifest:
    """A signed, fully-valid TEST_ONLY director Runtime Bundle."""
    return sign_director_runtime_bundle(
        valid_director_bundle_payload(anchors=anchors,
                                      candidate_id=candidate_id))


def sign_director_runtime_bundle(
        payload: Dict[str, Any]) -> DirectorRuntimeBundleManifest:
    """TEST_ONLY: mint a director Runtime Bundle (the DIRECTOR's side is
    emulated with a canonical hash — direction two never signs a bundle
    in production; it only verifies the director's). The nested model
    fields are normalized through their own model dumps so the signature
    covers exactly what ``model_dump()`` reproduces (including the nested
    ``protocol_version`` fields)."""
    from d052.feedback_llm_ued.director_runtime_bundle import (
        AnchorManifestData,
        DiCodeBatchBindingData,
        ReferenceIdentityData,
        SmokeSemanticsData,
        StudentInitContractData,
    )
    body = dict(payload)
    body.pop("bundle_hash", None)
    body.setdefault(
        "protocol_version",
        DirectorRuntimeBundleManifest.model_fields["protocol_version"]
        .default)
    body.setdefault("bundle_version", DIRECTOR_RUNTIME_BUNDLE_VERSION)
    body["student_init_contract"] = StudentInitContractData(
        **body["student_init_contract"]).model_dump()
    body["reference_identity"] = ReferenceIdentityData(
        **body["reference_identity"]).model_dump()
    body["shared_anchor_manifest"] = AnchorManifestData(
        **body["shared_anchor_manifest"]).model_dump()
    body["smoke_semantics"] = SmokeSemanticsData(
        **body["smoke_semantics"]).model_dump()
    body["batch_binding"] = DiCodeBatchBindingData(
        **body["batch_binding"]).model_dump()
    signature = canonical_sha256(body)
    return DirectorRuntimeBundleManifest(**body, bundle_hash=signature)


def sign_full_state_round_trip(payload: Mapping[str, object]
                               ) -> FullStateRoundTripResult:
    """TEST_ONLY: mint a locally-signed FullStateRoundTripResult. The
    production seam REFUSES such local self-signatures (it consumes only
    the director-runtime's DirectorVerifiedRunStateRoundTrip); this helper
    exists solely to build fixtures for the rejection tests."""
    body = dict(payload)
    body.pop("round_trip_hash", None)
    body.setdefault(
        "protocol_version",
        FullStateRoundTripResult.model_fields["protocol_version"].default)
    signature = canonical_sha256(body)
    return FullStateRoundTripResult(**body, round_trip_hash=signature)


def sign_director_verified_round_trip(payload: Mapping[str, object]
                                      ) -> "DirectorVerifiedRunStateRoundTrip":
    """TEST_ONLY: emulate the DIRECTOR-runtime's unforgeable
    DirectorVerifiedRunStateRoundTrip attestation. Production never signs
    this (there is no local signer); the fixture emulates the director
    with a canonical hash so the seam's consumption + verification can be
    exercised."""
    from d052.feedback_llm_ued.student_binding import (
        DirectorVerifiedRunStateRoundTrip,
    )
    body = dict(payload)
    body.pop("attestation_hash", None)
    body.setdefault(
        "protocol_version",
        DirectorVerifiedRunStateRoundTrip.model_fields["protocol_version"]
        .default)
    signature = canonical_sha256(body)
    return DirectorVerifiedRunStateRoundTrip(**body,
                                             attestation_hash=signature)


def director_round_trip_payload(window, checkpoint_hash,
                                runtime_bundle_hash, *,
                                verifier_id=None,
                                next_policy_step_equivalent=True,
                                verified=True):
    """A valid TEST_ONLY DirectorVerifiedRunStateRoundTrip payload."""
    return dict(
        window=window, checkpoint_hash=checkpoint_hash,
        verifier_id=(verifier_id or text_sha256("TEST_ONLY_DIRECTOR_VERIFIER")),
        verifier_implementation_hash=text_sha256(
            "TEST_ONLY_VERIFIER_IMPL"),
        runtime_bundle_hash=runtime_bundle_hash,
        student_checkpoint_hash=text_sha256("TEST_ONLY_STUDENT_CKPT"),
        optimizer_state_hash=text_sha256("TEST_ONLY_OPTIMIZER_STATE"),
        global_update_step=1, global_env_steps=128,
        rng_state_hash=text_sha256("TEST_ONLY_RNG_STATE"),
        controller_store_hash=text_sha256("TEST_ONLY_CONTROLLER_STORE"),
        next_policy_step_equivalent=next_policy_step_equivalent,
        verified=verified)


def student_contract(candidate_id=C.STRONG_STUDENT_CANDIDATE_ID,
                     *, checkpoint_global_step=98304,
                     runtime_bundle_hash=None):
    """A valid TEST_ONLY StudentInitContract for the given allowed
    candidate (memory/carry follow the legal profile mapping)."""
    from types import SimpleNamespace
    memory_mode, carry_mode = C.STUDENT_PROFILE_MEMORY_MAP[candidate_id]
    return SimpleNamespace(
        candidate_id=candidate_id, architecture_family="RMT16",
        memory_family="RMT16_ORIGINAL", carry_mode=carry_mode,
        parameter_tree_hash=text_sha256("TEST_ONLY_STUDENT_PARAM_TREE"),
        checkpoint_global_step=checkpoint_global_step,
        profile_hash=text_sha256("TEST_ONLY_PROFILE"),
        memory_mode=memory_mode,
        memory_spec_hash=text_sha256("TEST_ONLY_MEMORY_SPEC"),
        adapter_identity_hash=text_sha256("TEST_ONLY_ADAPTER"),
        runtime_bundle_hash=(runtime_bundle_hash
                             or text_sha256("TEST_ONLY_RUNTIME_BUNDLE")))


__all__ = ["sign_director_runtime_bundle", "sign_full_state_round_trip",
           "sign_director_verified_round_trip",
           "director_round_trip_payload", "student_contract",
           "valid_director_bundle_payload", "valid_director_bundle"]
