"""CC2 follow-up P0-3 tests: the signed E1 runtime bundle.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE:
every capability object in this file is a conspicuously-marked
synthetic fixture. These tests prove the bundle CONTRACT (shape,
hash binding, signer gating, TEST_ONLY/PRODUCTION separation) — they
never mint real evidence, never flip a REAL_* flag and never stand
in for a supervisor-signed production bundle.

Covered negative matrix:
* string placeholder capability          -> STRING_PLACEHOLDER
* None capability                        -> UNBOUND
* missing capability contract            -> MISSING_FIELD
* unknown capability contract            -> UNKNOWN_FIELD
* manifest hash tamper                   -> HASH_MISMATCH
* unknown manifest field                 -> UNKNOWN_FIELD
* bad mode / non-mapping manifest        -> BAD_TYPE
* production signer (whitelist EMPTY)    -> SIGNER_UNAUTHORIZED
* TEST_ONLY signer impersonation         -> SIGNER_UNAUTHORIZED
* TEST_ONLY bundle on production surface -> TEST_ONLY_REJECTED
"""
from dataclasses import dataclass

import pytest

from dicode.teachers.e1_formal import runtime_bundle as RB


# ---------------------------------------------------------------------------
# SYNTHETIC capability fixtures — TEST_ONLY, never real shared objects.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _SyntheticCapability:
    """TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION placeholder standing
    where a real shared runtime object would bind."""

    kind: str
    identity_id: str


@dataclass(frozen=True)
class _SelfIdentifyingCapability:
    """TEST_ONLY capability carrying its own signed identity hash (the
    shared runtime signs its own identities; the bundle must prefer
    that surface over any re-derivation)."""

    kind: str
    object_identity_hash: str


def _synthetic_capabilities():
    """Nine distinct TEST_ONLY capability objects (one per contract)."""
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


def _manifest_from_bundle(bundle) -> dict:
    """The manifest (signed description) of one assembled bundle."""
    return {
        "bundle_id": bundle.bundle_id,
        "mode": bundle.mode,
        "source_commit": bundle.source_commit,
        "signer_id": bundle.signer_id,
        "authorization_grant_hash": bundle.authorization_grant_hash,
        "object_identity_hashes": dict(bundle.object_identity_hashes),
        "student_selection": bundle.student_selection_mapping,
        "bundle_hash": bundle.bundle_hash,
    }


# ---------------------------------------------------------------------------
# assembly + identity binding
# ---------------------------------------------------------------------------
class TestTestOnlyBundleAssembly:
    def test_assembles_with_all_nine_capabilities(self):
        bundle = _test_only_bundle()
        assert bundle.mode == RB.BUNDLE_MODE_TEST_ONLY
        assert bundle.signer_id == RB.SYNTHETIC_TEST_ONLY_SIGNER
        assert len(bundle.bundle_hash) == 64
        assert len(bundle.capabilities) == 9
        assert len(bundle.object_identity_hashes) == 9
        for contract in RB.RUNTIME_CAPABILITY_CONTRACTS:
            obj = bundle.capability(contract)
            assert isinstance(obj, _SyntheticCapability)
            assert obj.kind == contract
            assert len(bundle.object_identity_hash(contract)) == 64

    def test_bundle_hash_binds_every_identity_field(self):
        bundle = _test_only_bundle()
        base = dict(bundle.object_identity_hashes)
        sel_hash = bundle.student_selection.descriptor_hash
        same = RB.compute_bundle_hash(
            bundle_id=bundle.bundle_id,
            mode=bundle.mode,
            source_commit=bundle.source_commit,
            signer_id=bundle.signer_id,
            authorization_grant_hash=bundle.authorization_grant_hash,
            object_identity_hashes=base,
            student_selection_hash=sel_hash,
        )
        assert same == bundle.bundle_hash
        for field, value in (
            ("bundle_id", "other-bundle"),
            ("source_commit", "OTHER_COMMIT"),
            ("signer_id", "other-signer"),
            ("authorization_grant_hash", "other-grant"),
        ):
            kwargs = dict(
                bundle_id=bundle.bundle_id,
                mode=bundle.mode,
                source_commit=bundle.source_commit,
                signer_id=bundle.signer_id,
                authorization_grant_hash=bundle.authorization_grant_hash,
                object_identity_hashes=base,
                student_selection_hash=sel_hash,
            )
            kwargs[field] = value
            assert RB.compute_bundle_hash(**kwargs) != bundle.bundle_hash
        tampered = dict(base)
        tampered["training"] = "f" * 64
        assert (
            RB.compute_bundle_hash(
                bundle_id=bundle.bundle_id,
                mode=bundle.mode,
                source_commit=bundle.source_commit,
                signer_id=bundle.signer_id,
                authorization_grant_hash=bundle.authorization_grant_hash,
                object_identity_hashes=tampered,
                student_selection_hash=sel_hash,
            )
            != bundle.bundle_hash
        )
        # CC2-Student repair: a different Student selection yields a
        # different bundle hash (student_selection is in the identity)
        assert (
            RB.compute_bundle_hash(
                bundle_id=bundle.bundle_id,
                mode=bundle.mode,
                source_commit=bundle.source_commit,
                signer_id=bundle.signer_id,
                authorization_grant_hash=bundle.authorization_grant_hash,
                object_identity_hashes=base,
                student_selection_hash="f" * 64,
            )
            != bundle.bundle_hash
        )

    def test_convenience_properties_return_the_same_objects(self):
        bundle = _test_only_bundle()
        caps = dict(bundle.capabilities)
        assert bundle.student_identity is caps["student_identity"]
        assert bundle.reference_identity is caps["reference_identity"]
        assert bundle.student_adapter is caps["student_adapter"]
        assert bundle.reference_adapter is caps["reference_adapter"]
        assert bundle.anchor_manifest is caps["anchor_manifest"]
        assert bundle.formal_asset_registry is caps["formal_asset_registry"]
        assert bundle.probe_runner is caps["probe_runner"]
        assert bundle.training is caps["training"]
        assert bundle.full_state_checkpoint is caps["full_state_checkpoint"]

    def test_missing_contract_lookup_fails_closed(self):
        bundle = _test_only_bundle()
        with pytest.raises(RB.RuntimeBundleError) as excinfo:
            bundle.capability("no_such_contract")
        assert excinfo.value.code == RB.RUNTIME_BUNDLE_UNBOUND


class TestCapabilityValidation:
    def test_string_placeholder_capability_refused(self):
        caps = _synthetic_capabilities()
        caps["probe_runner"] = "probe_runner"  # a contract NAME, not an object
        with pytest.raises(RB.RuntimeBundleError) as excinfo:
            RB.build_test_only_runtime_bundle(
                source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
                capabilities=caps,
            )
        assert excinfo.value.code == RB.RUNTIME_BUNDLE_STRING_PLACEHOLDER

    def test_none_capability_refused(self):
        caps = _synthetic_capabilities()
        caps["training"] = None
        with pytest.raises(RB.RuntimeBundleError) as excinfo:
            RB.build_test_only_runtime_bundle(
                source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
                capabilities=caps,
            )
        assert excinfo.value.code == RB.RUNTIME_BUNDLE_UNBOUND

    def test_missing_capability_contract_refused(self):
        caps = _synthetic_capabilities()
        del caps["full_state_checkpoint"]
        with pytest.raises(RB.RuntimeBundleError) as excinfo:
            RB.build_test_only_runtime_bundle(
                source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
                capabilities=caps,
            )
        assert excinfo.value.code == RB.RUNTIME_BUNDLE_MISSING_FIELD

    def test_unknown_capability_contract_refused(self):
        caps = _synthetic_capabilities()
        caps["extra_contract"] = _SyntheticCapability("extra", "x")
        with pytest.raises(RB.RuntimeBundleError) as excinfo:
            RB.build_test_only_runtime_bundle(
                source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
                capabilities=caps,
            )
        assert excinfo.value.code == RB.RUNTIME_BUNDLE_UNKNOWN_FIELD

    def test_non_mapping_capabilities_refused(self):
        with pytest.raises(RB.RuntimeBundleError) as excinfo:
            RB.build_test_only_runtime_bundle(
                source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
                capabilities=(),  # not a mapping
            )
        assert excinfo.value.code == RB.RUNTIME_BUNDLE_BAD_TYPE

    def test_empty_source_commit_refused(self):
        with pytest.raises(RB.RuntimeBundleError) as excinfo:
            RB.build_test_only_runtime_bundle(
                source_commit="   ",
                capabilities=_synthetic_capabilities(),
            )
        assert excinfo.value.code == RB.RUNTIME_BUNDLE_BAD_TYPE


# ---------------------------------------------------------------------------
# object identity hashing
# ---------------------------------------------------------------------------
class TestObjectIdentityHash:
    def test_self_signed_identity_hash_wins(self):
        digest = "a" * 64
        obj = _SelfIdentifyingCapability("training", digest)
        assert RB.object_identity_hash(obj) == digest

    def test_mapping_identity_is_canonical(self):
        from dicode.teachers.e1_formal.canonical import canonical_sha256

        mapping = {"kind": "anchor_manifest", "id": "test-only"}
        assert RB.object_identity_hash(mapping) == canonical_sha256(
            dict(mapping)
        )

    def test_plain_object_identity_uses_state(self):
        obj = _SyntheticCapability("probe_runner", "id-1")
        digest = RB.object_identity_hash(obj)
        assert len(digest) == 64
        other = _SyntheticCapability("probe_runner", "id-2")
        assert RB.object_identity_hash(other) != digest

    def test_identity_hash_attribute_also_honored(self):
        class _WithIdentityHashAttr:
            identity_hash = "b" * 64

        assert (
            RB.object_identity_hash(_WithIdentityHashAttr()) == "b" * 64
        )


# ---------------------------------------------------------------------------
# manifest verification (shape -> hash -> signer)
# ---------------------------------------------------------------------------
class TestManifestVerification:
    def test_test_only_manifest_roundtrip_verifies(self):
        bundle = _test_only_bundle()
        loaded = RB.load_verified_runtime_bundle(
            _manifest_from_bundle(bundle), "test"
        )
        assert loaded.bundle_id == bundle.bundle_id
        assert loaded.mode == RB.BUNDLE_MODE_TEST_ONLY
        assert loaded.signer_id == RB.SYNTHETIC_TEST_ONLY_SIGNER
        assert loaded.bundle_hash == bundle.bundle_hash
        assert loaded.object_identity_hashes == bundle.object_identity_hashes
        # objects bind at the seam, NEVER inside the manifest
        assert loaded.capabilities == ()

    def test_manifest_hash_tamper_refused(self):
        mapping = _manifest_from_bundle(_test_only_bundle())
        mapping["object_identity_hashes"]["student_identity"] = "f" * 64
        with pytest.raises(RB.RuntimeBundleError) as excinfo:
            RB.load_verified_runtime_bundle(mapping, "test")
        assert excinfo.value.code == RB.RUNTIME_BUNDLE_HASH_MISMATCH

    def test_manifest_bundle_hash_tamper_refused(self):
        mapping = _manifest_from_bundle(_test_only_bundle())
        mapping["bundle_hash"] = "e" * 64
        with pytest.raises(RB.RuntimeBundleError) as excinfo:
            RB.load_verified_runtime_bundle(mapping, "test")
        assert excinfo.value.code == RB.RUNTIME_BUNDLE_HASH_MISMATCH

    def test_manifest_unknown_field_refused(self):
        mapping = _manifest_from_bundle(_test_only_bundle())
        mapping["extra_field"] = True
        with pytest.raises(RB.RuntimeBundleError) as excinfo:
            RB.load_verified_runtime_bundle(mapping, "test")
        assert excinfo.value.code == RB.RUNTIME_BUNDLE_UNKNOWN_FIELD

    def test_manifest_missing_identity_hash_refused(self):
        mapping = _manifest_from_bundle(_test_only_bundle())
        del mapping["object_identity_hashes"]["training"]
        with pytest.raises(RB.RuntimeBundleError) as excinfo:
            RB.load_verified_runtime_bundle(mapping, "test")
        assert excinfo.value.code == RB.RUNTIME_BUNDLE_MISSING_FIELD

    def test_manifest_short_identity_hash_refused(self):
        mapping = _manifest_from_bundle(_test_only_bundle())
        mapping["object_identity_hashes"]["training"] = "abc"
        with pytest.raises(RB.RuntimeBundleError) as excinfo:
            RB.load_verified_runtime_bundle(mapping, "test")
        assert excinfo.value.code == RB.RUNTIME_BUNDLE_MISSING_FIELD

    def test_non_mapping_manifest_refused(self):
        with pytest.raises(RB.RuntimeBundleError) as excinfo:
            RB.load_verified_runtime_bundle(["not", "a", "mapping"], "test")
        assert excinfo.value.code == RB.RUNTIME_BUNDLE_BAD_TYPE

    def test_manifest_bad_mode_refused(self):
        mapping = _manifest_from_bundle(_test_only_bundle())
        mapping["mode"] = "REAL"
        with pytest.raises(RB.RuntimeBundleError) as excinfo:
            RB.load_verified_runtime_bundle(mapping, "test")
        assert excinfo.value.code == RB.RUNTIME_BUNDLE_BAD_TYPE


class TestSignerGating:
    def test_production_signer_whitelist_is_empty_this_round(self):
        # the supervisor has authorized NO production signer yet — any
        # production bundle must fail closed
        assert RB.AUTHORIZED_BUNDLE_SIGNERS == ()

    def test_production_manifest_signer_unauthorized(self):
        bundle = _test_only_bundle()
        mapping = _manifest_from_bundle(bundle)
        mapping["mode"] = RB.BUNDLE_MODE_PRODUCTION
        mapping["signer_id"] = "would-be-production-signer"
        mapping["bundle_hash"] = RB.compute_bundle_hash(
            bundle_id=mapping["bundle_id"],
            mode=mapping["mode"],
            source_commit=mapping["source_commit"],
            signer_id=mapping["signer_id"],
            authorization_grant_hash=mapping["authorization_grant_hash"],
            object_identity_hashes=mapping["object_identity_hashes"],
            student_selection_hash=bundle.student_selection.descriptor_hash,
        )
        with pytest.raises(RB.RuntimeBundleError) as excinfo:
            RB.load_verified_runtime_bundle(mapping, "test")
        assert excinfo.value.code == RB.RUNTIME_BUNDLE_SIGNER_UNAUTHORIZED

    def test_test_only_manifest_requires_the_synthetic_signer(self):
        bundle = _test_only_bundle()
        mapping = _manifest_from_bundle(bundle)
        mapping["signer_id"] = "attacker-claiming-test-only"
        mapping["bundle_hash"] = RB.compute_bundle_hash(
            bundle_id=mapping["bundle_id"],
            mode=mapping["mode"],
            source_commit=mapping["source_commit"],
            signer_id=mapping["signer_id"],
            authorization_grant_hash=mapping["authorization_grant_hash"],
            object_identity_hashes=mapping["object_identity_hashes"],
            student_selection_hash=bundle.student_selection.descriptor_hash,
        )
        with pytest.raises(RB.RuntimeBundleError) as excinfo:
            RB.load_verified_runtime_bundle(mapping, "test")
        assert excinfo.value.code == RB.RUNTIME_BUNDLE_SIGNER_UNAUTHORIZED


class TestProductionAdmissibility:
    def test_test_only_bundle_refused_on_production_surface(self):
        bundle = _test_only_bundle()
        with pytest.raises(RB.RuntimeBundleError) as excinfo:
            RB.require_bundle_admissible_for_production(bundle, "test")
        assert excinfo.value.code == RB.RUNTIME_BUNDLE_TEST_ONLY_REJECTED

    def test_non_bundle_refused_on_production_surface(self):
        with pytest.raises(RB.RuntimeBundleError) as excinfo:
            RB.require_bundle_admissible_for_production(
                {"mode": "PRODUCTION"}, "test"
            )
        assert excinfo.value.code == RB.RUNTIME_BUNDLE_BAD_TYPE
