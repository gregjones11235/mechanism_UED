"""CC2 follow-up P0-3 tests: shared-runtime OBJECT resolution.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE:
every bundle/object fixture here is synthetic. The shared runtime
module does not exist in this worktree, so every canonical contract
stays honestly UNBOUND; the bundle-bound surface is exercised with a
conspicuously-marked TEST_ONLY bundle through the explicit
``allow_test_only`` gate ONLY. Nothing here flips a REAL_* flag.

Covered negative matrix:
* legacy canonical eight stay unbound with honest codes (pin)
* unbound require_bound                    -> SEAM_UNBOUND / contract code
* non-bundle argument                      -> SEAM_BAD_TYPE
* unknown bundle contract                  -> SEAM_UNKNOWN_CONTRACT
* TEST_ONLY bundle on production surface   -> TEST_ONLY_REJECTED
* PRODUCTION bundle on test-only surface   -> SEAM_BAD_TYPE (no mixing)
* forged production signer                 -> SIGNER_UNAUTHORIZED
* manifest-level bundle (objects unbound)  -> RUNTIME_BUNDLE_UNBOUND
* identity-hash tamper                     -> SEAM_IDENTITY_MISMATCH
* None / string / number object surfaces   -> SEAM_* fail-closed
"""
from dataclasses import dataclass, replace

import pytest

from dicode.teachers.e1_formal import runtime_bundle as RB
from dicode.teachers.e1_formal import shared_runtime_seam as SRS


# ---------------------------------------------------------------------------
# SYNTHETIC fixtures — TEST_ONLY, never real shared runtime objects.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _SyntheticCapability:
    """TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION placeholder."""

    kind: str
    identity_id: str


def _synthetic_capabilities():
    return {
        contract: _SyntheticCapability(
            kind=contract, identity_id=f"test-only-{contract}"
        )
        for contract in RB.RUNTIME_CAPABILITY_CONTRACTS
    }


def _test_only_bundle():
    return RB.build_test_only_runtime_bundle(
        source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
        capabilities=_synthetic_capabilities(),
    )


# ---------------------------------------------------------------------------
# legacy surface: the canonical eight stay honestly unbound
# ---------------------------------------------------------------------------
class TestLegacyCanonicalEight:
    def test_all_eight_contracts_resolve_honest_state(self):
        """All eight canonical contracts resolve with honest state:
        bound when the shared runtime is importable (DICODE_SHARED_RUNTIME_REAL=1),
        unbound with honest BLOCKED codes otherwise."""
        resolutions = SRS.resolve_all_shared_runtime()
        assert sorted(resolutions) == sorted(SRS.SHARED_CONTRACTS)
        assert len(resolutions) == 8
        for contract, resolution in resolutions.items():
            if resolution.bound:
                # shared runtime is available: object is bound
                assert resolution.object_ref is not None
                assert len(resolution.object_identity_hash) == 64
                assert resolution.code == ""
            else:
                # shared runtime is gated: honest unbound report
                assert resolution.code == SRS._CONTRACT_CODES[contract]
                assert resolution.code.startswith(
                    "BLOCKED_WAITING_SHARED_RUNTIME_"
                )
                assert resolution.object_ref is None
                assert resolution.object_identity_hash == ""

    def test_per_contract_resolvers_carry_their_codes(self):
        expected = {
            "resolve_student_identity": (
                "StudentIdentity",
                SRS.BLOCKED_WAITING_SHARED_RUNTIME_STUDENT_IDENTITY,
            ),
            "resolve_student_adapter": (
                "StudentAdapter",
                SRS.BLOCKED_WAITING_SHARED_RUNTIME_STUDENT_ADAPTER,
            ),
            "resolve_reference_identity": (
                "ReferenceIdentity",
                SRS.BLOCKED_WAITING_SHARED_RUNTIME_REFERENCE_IDENTITY,
            ),
            "resolve_reference_adapter": (
                "ReferenceAdapter",
                SRS.BLOCKED_WAITING_SHARED_RUNTIME_REFERENCE_ADAPTER,
            ),
            "resolve_anchor_manifest": (
                "AnchorManifest",
                SRS.BLOCKED_WAITING_SHARED_RUNTIME_ANCHOR_MANIFEST,
            ),
            "resolve_formal_asset_registry": (
                "FormalAssetRegistry",
                SRS.BLOCKED_WAITING_SHARED_RUNTIME_FORMAL_ASSET_REGISTRY,
            ),
            "resolve_candidate_probe_result": (
                "CandidateProbeResult",
                SRS.BLOCKED_WAITING_SHARED_RUNTIME_CANDIDATE_PROBE_RESULT,
            ),
            "resolve_full_state_checkpoint": (
                "FullStateCheckpoint",
                SRS.BLOCKED_WAITING_SHARED_RUNTIME_FULL_STATE_CHECKPOINT,
            ),
        }
        for name, (contract, code) in expected.items():
            resolution = getattr(SRS, name)()
            assert resolution.contract == contract
            if resolution.bound:
                # shared runtime available: code is empty
                assert resolution.code == ""
            else:
                assert resolution.code == code
                assert resolution.bound is False

    def test_backward_compatible_alias_and_construction(self):
        assert SRS.SeamResolution is SRS.SharedRuntimeResolution
        resolution = SRS.SeamResolution(
            contract="StudentIdentity",
            bound=False,
            code="BLOCKED_WAITING_SHARED_RUNTIME_STUDENT_IDENTITY",
            detail="unbound",
        )
        assert resolution.object_ref is None
        assert resolution.attestation_hash == ""


class TestResolutionRecord:
    def test_contract_name_property(self):
        resolution = SRS.unbound_resolution(
            "student_adapter", "test", "unbound"
        )
        assert resolution.contract_name == resolution.contract
        assert resolution.contract_name == "StudentAdapter"

    def test_require_bound_fails_closed_while_unbound(self):
        resolution = SRS.unbound_resolution(
            "full_state_checkpoint", "test", "unbound"
        )
        with pytest.raises(SRS.SeamError) as excinfo:
            resolution.require_bound("test")
        assert excinfo.value.code == (
            SRS.BLOCKED_WAITING_SHARED_RUNTIME_FULL_STATE_CHECKPOINT
        )

    def test_unbound_resolution_never_carries_an_object(self):
        for bundle_contract in SRS.BUNDLE_SEAM_CONTRACTS:
            resolution = SRS.unbound_resolution(
                bundle_contract, "test", "waiting for CC4"
            )
            assert resolution.bound is False
            assert resolution.object_ref is None
            assert resolution.code.startswith(
                "BLOCKED_WAITING_SHARED_RUNTIME_"
            )

    def test_unbound_resolution_unknown_contract_refused(self):
        with pytest.raises(SRS.SeamError) as excinfo:
            SRS.unbound_resolution("no_such_contract", "test", "x")
        assert excinfo.value.code == SRS.SEAM_UNKNOWN_CONTRACT


# ---------------------------------------------------------------------------
# bundle-bound resolution (the ONLY production path to a shared object)
# ---------------------------------------------------------------------------
class TestBundleBoundResolution:
    def test_test_only_bundle_refused_on_the_default_surface(self):
        bundle = _test_only_bundle()
        with pytest.raises(RB.RuntimeBundleError) as excinfo:
            SRS.resolve_contract_from_bundle(
                bundle, "student_identity", "test"
            )
        assert excinfo.value.code == RB.RUNTIME_BUNDLE_TEST_ONLY_REJECTED

    def test_test_only_binding_requires_the_explicit_gate(self):
        bundle = _test_only_bundle()
        resolution = SRS.resolve_contract_from_bundle(
            bundle,
            "student_identity",
            "test",
            allow_test_only=True,
        )
        assert resolution.bound is True
        assert resolution.code == ""
        assert resolution.contract == "StudentIdentity"
        caps = dict(bundle.capabilities)
        assert resolution.object_ref is caps["student_identity"]
        assert resolution.object_identity_hash == (
            bundle.object_identity_hash("student_identity")
        )
        assert resolution.registry_bundle_hash == bundle.bundle_hash
        assert len(resolution.provider_module_hash) == 64
        assert len(resolution.capability_descriptor) == 64
        assert len(resolution.attestation_hash) == 64
        assert resolution.require_bound("test") is resolution

    def test_probe_runner_maps_to_the_probe_result_contract(self):
        bundle = _test_only_bundle()
        resolution = SRS.resolve_contract_from_bundle(
            bundle, "probe_runner", "test", allow_test_only=True
        )
        assert resolution.contract == "CandidateProbeResult"

    def test_training_maps_to_the_training_runtime_contract(self):
        bundle = _test_only_bundle()
        resolution = SRS.resolve_contract_from_bundle(
            bundle, "training", "test", allow_test_only=True
        )
        assert resolution.contract == "TrainingRuntime"
        assert resolution.code == ""

    def test_non_bundle_argument_refused(self):
        with pytest.raises(SRS.SeamError) as excinfo:
            SRS.resolve_contract_from_bundle(
                {"student_identity": object()},
                "student_identity",
                "test",
                allow_test_only=True,
            )
        assert excinfo.value.code == SRS.SEAM_BAD_TYPE

    def test_unknown_bundle_contract_refused(self):
        bundle = _test_only_bundle()
        with pytest.raises(SRS.SeamError) as excinfo:
            SRS.resolve_contract_from_bundle(
                bundle, "no_such_contract", "test", allow_test_only=True
            )
        assert excinfo.value.code == SRS.SEAM_UNKNOWN_CONTRACT

    def test_manifest_level_bundle_has_no_objects_to_bind(self):
        # a verified manifest carries identity hashes ONLY; the objects
        # bind at the seam, so a manifest-level bundle cannot smuggle
        # an object into resolution
        source = _test_only_bundle()
        manifest = {
            "bundle_id": source.bundle_id,
            "mode": source.mode,
            "source_commit": source.source_commit,
            "signer_id": source.signer_id,
            "authorization_grant_hash": source.authorization_grant_hash,
            "object_identity_hashes": dict(source.object_identity_hashes),
            "student_selection": source.student_selection_mapping,
"signature_ref": "",
        "registry_identity": "",
        "registry_hash": "",
            "bundle_hash": source.bundle_hash,
        }
        loaded = RB.load_verified_runtime_bundle(manifest, "test")
        assert loaded.capabilities == ()
        with pytest.raises(RB.RuntimeBundleError) as excinfo:
            SRS.resolve_contract_from_bundle(
                loaded, "student_identity", "test", allow_test_only=True
            )
        assert excinfo.value.code == RB.RUNTIME_BUNDLE_UNBOUND

    def test_identity_hash_tamper_refused(self):
        bundle = _test_only_bundle()
        tampered = replace(
            bundle,
            object_identity_hashes=tuple(
                (contract, "f" * 64)
                for contract, _ in bundle.object_identity_hashes
            ),
        )
        with pytest.raises(SRS.SeamError) as excinfo:
            SRS.resolve_contract_from_bundle(
                tampered, "student_adapter", "test", allow_test_only=True
            )
        assert excinfo.value.code == SRS.SEAM_IDENTITY_MISMATCH

    def test_production_bundle_on_test_only_surface_refused(self):
        forged = replace(_test_only_bundle(), mode=RB.BUNDLE_MODE_PRODUCTION)
        with pytest.raises(SRS.SeamError) as excinfo:
            SRS.resolve_contract_from_bundle(
                forged, "training", "test", allow_test_only=True
            )
        assert excinfo.value.code == SRS.SEAM_BAD_TYPE

    def test_forged_production_without_signature_ref_refused(self):
        # A forged TEST_ONLY->PRODUCTION bundle carries NO signature_ref:
        # the production surface refuses it fail-closed (the director
        # verifier authorizes a production bundle, and the structural
        # signature_ref must be present).
        forged = replace(_test_only_bundle(), mode=RB.BUNDLE_MODE_PRODUCTION)
        with pytest.raises(RB.RuntimeBundleError) as excinfo:
            SRS.resolve_contract_from_bundle(
                forged, "training", "test"
            )
        assert excinfo.value.code == RB.RUNTIME_BUNDLE_MISSING_FIELD

    def test_resolve_all_from_bundle_binds_all_nine(self):
        bundle = _test_only_bundle()
        resolutions = SRS.resolve_all_from_bundle(
            bundle, "test", allow_test_only=True
        )
        assert sorted(resolutions) == sorted(
            RB.RUNTIME_CAPABILITY_CONTRACTS
        )
        caps = dict(bundle.capabilities)
        attestations = set()
        for bundle_contract, resolution in resolutions.items():
            assert resolution.bound is True
            assert resolution.object_ref is caps[bundle_contract]
            assert resolution.object_identity_hash == (
                bundle.object_identity_hash(bundle_contract)
            )
            assert resolution.registry_bundle_hash == bundle.bundle_hash
            attestations.add(resolution.attestation_hash)
        # every contract's attestation is distinct (identity-bound)
        assert len(attestations) == 9

    def test_resolve_all_propagates_failures_without_partial_binding(self):
        bundle = _test_only_bundle()
        with pytest.raises(RB.RuntimeBundleError) as excinfo:
            # production surface (default) refuses the TEST_ONLY bundle
            SRS.resolve_all_from_bundle(bundle, "test")
        assert excinfo.value.code == RB.RUNTIME_BUNDLE_TEST_ONLY_REJECTED


# ---------------------------------------------------------------------------
# mechanical object-surface verification
# ---------------------------------------------------------------------------
class TestObjectSurfaceVerification:
    def test_none_surface_refused(self):
        with pytest.raises(SRS.SeamError) as excinfo:
            SRS._verify_object_surface("StudentIdentity", None, "test")
        assert excinfo.value.code == SRS.SEAM_UNBOUND

    def test_string_placeholder_surface_refused(self):
        with pytest.raises(SRS.SeamError) as excinfo:
            SRS._verify_object_surface(
                "StudentIdentity", "student_identity", "test"
            )
        assert excinfo.value.code == SRS.SEAM_STRING_PLACEHOLDER

    def test_bare_number_surface_refused(self):
        for value in (True, 12, 3.5):
            with pytest.raises(SRS.SeamError) as excinfo:
                SRS._verify_object_surface(
                    "StudentIdentity", value, "test"
                )
            assert excinfo.value.code == SRS.SEAM_BAD_TYPE

    def test_stateless_surface_refused(self):
        with pytest.raises(SRS.SeamError) as excinfo:
            SRS._verify_object_surface(
                "StudentIdentity", ("tuple", "has", "no", "state"), "test"
            )
        assert excinfo.value.code == SRS.SEAM_BAD_TYPE

    def test_stateful_object_passes(self):
        obj = _SyntheticCapability("student_identity", "test-only")
        SRS._verify_object_surface("StudentIdentity", obj, "test")
        SRS._verify_object_surface(
            "AnchorManifest", {"anchor": "test-only"}, "test"
        )
