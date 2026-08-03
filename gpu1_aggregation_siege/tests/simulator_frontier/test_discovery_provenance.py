"""Registry-bound TRAINING_DISCOVERY capture provenance tests (condition 2).

Hardened after the controller's PASS_WITH_BLOCKER review: marker-string
denylists alone are bypassable, so isolation is now two-layer —
(1) allowlist binding of every discovery input to a
``DiscoveryProvenanceRegistry`` record, (2) a case-insensitive forbidden
formal asset identity sweep (canonical id AND sha256) over every textual
field, including nested notes.

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT: every registry below is a test
fixture.  The REAL frozen formal asset identity set has not been injected by
the controller this round, so these tests prove CONTRACT_READY only — real
isolation stays BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY (rule 5).
"""

from __future__ import annotations

import pytest

from dicode.simulator_frontier.discovery_provenance import (
    BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY,
    DISCOVERY_FORMAL_PROVENANCE_ISOLATED,
    DISCOVERY_PROVENANCE_CONTRACT_READY,
    FROZEN_FORMAL_ASSET_REGISTRY_BOUND,
    REGISTRY_USAGE_PRODUCTION,
    REGISTRY_USAGE_TEST_ONLY,
    AssetKind,
    CaptureProvenance,
    DiscoveryAssetRecord,
    DiscoveryProvenance,
    DiscoveryProvenanceRegistry,
    FormalAssetIdentity,
    assert_not_formal,
    clear_injected_production_registry,
    discovery_source_for,
    inject_frozen_formal_asset_registry,
    production_registry_bound,
    registry_hash_of,
    registry_status,
    validate_capture_provenance,
    validate_capture_provenance_production,
    validate_discovery_registry,
)
from dicode.simulator_frontier.errors import ProvenanceViolationError
from dicode.simulator_frontier.provenance import DataSource

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True  # labeling discipline

# Fixed synthetic forbidden formal asset identities (never real content).
FORMAL_BANK_SHA = "f" * 64
FORMAL_WORLD_SHA = "e" * 64
NEUTRAL_BANK_SHA = "a1" * 32  # 64 hex: a formal bank with a NEUTRAL canonical id
REGISTERED_WORLD_HASH = "ab0c" * 16  # valid 64-hex


def _forbidden() -> tuple[FormalAssetIdentity, ...]:
    return (
        FormalAssetIdentity(AssetKind.BANK, "FORMAL_BANK_01", FORMAL_BANK_SHA),
        FormalAssetIdentity(AssetKind.WORLD_SET, "FORMAL_WORLD_ICE", FORMAL_WORLD_SHA),
        # the bypass case: a formal asset whose canonical id carries no
        # FORMAL_* marker at all — only the registry can catch it
        FormalAssetIdentity(AssetKind.BANK, "evaluation_holdout_alpha", NEUTRAL_BANK_SHA),
    )


def _allowed() -> tuple[DiscoveryAssetRecord, ...]:
    return (
        DiscoveryAssetRecord("training_capture_bank", AssetKind.BANK),
        DiscoveryAssetRecord("discovery_worlds_v1", AssetKind.WORLD_SET,
                             world_set_hash=REGISTERED_WORLD_HASH),
    )


def _registry(**overrides) -> DiscoveryProvenanceRegistry:
    forbidden = overrides.pop("forbidden_formal_identities", _forbidden())
    allowed = overrides.pop("allowed_discovery_assets", _allowed())
    registry_id = overrides.pop("registry_id", "SYNTHETIC_DISCOVERY_REGISTRY")
    signature = overrides.pop("controller_signature_ref",
                              "SYNTHETIC_CONTROLLER_SIGNATURE_REF")
    usage = overrides.pop("usage", REGISTRY_USAGE_TEST_ONLY)
    registry_hash = overrides.pop("registry_hash", None) or registry_hash_of(
        registry_id, signature, forbidden, allowed, usage=usage)
    return DiscoveryProvenanceRegistry(
        registry_id=registry_id, controller_signature_ref=signature,
        frozen=overrides.pop("frozen", True),
        forbidden_formal_identities=forbidden,
        allowed_discovery_assets=allowed, registry_hash=registry_hash,
        usage=usage)


def _cap(**overrides) -> CaptureProvenance:
    base = dict(
        provenance=DiscoveryProvenance.TRAINING_DISCOVERY,
        rollout_protocol_id="standard-reset-v1",
        world_set_hash=REGISTERED_WORLD_HASH,
    )
    base.update(overrides)
    return CaptureProvenance(**base)


class TestRegistryValidation:
    def test_valid_synthetic_registry_validates(self):
        validate_discovery_registry(_registry())  # raises nothing

    def test_missing_controller_signature_raises(self):
        with pytest.raises(ProvenanceViolationError):
            validate_discovery_registry(_registry(controller_signature_ref=""))

    def test_unfrozen_registry_raises(self):
        with pytest.raises(ProvenanceViolationError):
            validate_discovery_registry(_registry(frozen=False))

    def test_empty_forbidden_set_raises(self):
        with pytest.raises(ProvenanceViolationError):
            validate_discovery_registry(_registry(forbidden_formal_identities=()))

    def test_forbidden_set_must_cover_bank_and_world(self):
        only_bank = (FormalAssetIdentity(AssetKind.BANK, "B", FORMAL_BANK_SHA),)
        with pytest.raises(ProvenanceViolationError):
            validate_discovery_registry(_registry(forbidden_formal_identities=only_bank))

    def test_empty_allowlist_raises(self):
        with pytest.raises(ProvenanceViolationError):
            validate_discovery_registry(_registry(allowed_discovery_assets=()))

    def test_duplicate_asset_ids_raise(self):
        dup = (DiscoveryAssetRecord("training_capture_bank", AssetKind.BANK),
               DiscoveryAssetRecord("TRAINING_CAPTURE_BANK", AssetKind.BANK))
        with pytest.raises(ProvenanceViolationError):
            validate_discovery_registry(_registry(allowed_discovery_assets=dup))

    def test_allowed_id_colliding_with_forbidden_raises(self):
        colliding = (DiscoveryAssetRecord("FORMAL_BANK_01", AssetKind.BANK),)
        with pytest.raises(ProvenanceViolationError):
            validate_discovery_registry(_registry(allowed_discovery_assets=colliding))

    def test_allowed_hash_colliding_with_forbidden_sha_raises(self):
        colliding = (DiscoveryAssetRecord("worlds_x", AssetKind.WORLD_SET,
                                          world_set_hash=FORMAL_WORLD_SHA),)
        with pytest.raises(ProvenanceViolationError):
            validate_discovery_registry(_registry(allowed_discovery_assets=colliding))

    def test_allowed_id_embedding_forbidden_id_raises(self):
        colliding = (DiscoveryAssetRecord("xFORMAL_BANK_01y", AssetKind.BANK),)
        with pytest.raises(ProvenanceViolationError):
            validate_discovery_registry(_registry(allowed_discovery_assets=colliding))

    def test_hash_mismatch_raises(self):
        with pytest.raises(ProvenanceViolationError):
            validate_discovery_registry(_registry(registry_hash="0" * 64))

    @pytest.mark.parametrize("bad", ["", "UNKNOWN", "PENDING_MANIFEST", "N/A"])
    def test_placeholder_registry_ids_raise(self, bad):
        with pytest.raises(ProvenanceViolationError):
            validate_discovery_registry(_registry(registry_id=bad))

    def test_formal_identity_requires_real_sha(self):
        with pytest.raises(ProvenanceViolationError):
            FormalAssetIdentity(AssetKind.BANK, "FORMAL_BANK_01", "not-a-sha")

    def test_world_record_requires_world_hash(self):
        with pytest.raises(ProvenanceViolationError):
            DiscoveryAssetRecord("worlds_x", AssetKind.WORLD_SET)

    def test_bank_record_rejects_world_hash(self):
        with pytest.raises(ProvenanceViolationError):
            DiscoveryAssetRecord("bank_x", AssetKind.BANK,
                                 world_set_hash=REGISTERED_WORLD_HASH)


class TestMissingRegistry:
    def test_none_registry_fails_closed(self):
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(_cap(), registry=None)

    def test_registry_argument_is_mandatory(self):
        # the old no-registry call shape must not even type-check: isolation
        # cannot be skipped by forgetting the argument
        with pytest.raises(TypeError):
            validate_capture_provenance(_cap())  # noqa: missing required kwarg

    def test_invalid_registry_fails_closed_before_capture_checks(self):
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(_cap(), registry=_registry(frozen=False))


class TestLegalDiscoveryCapture:
    def test_registered_discovery_capture_passes(self):
        validate_capture_provenance(
            _cap(bank_refs=("training_capture_bank",),
                 world_set_id="discovery_worlds_v1",
                 notes={"purpose": "frontier collection"}),
            registry=_registry())  # must not raise

    def test_string_provenance_is_normalized(self):
        cap = _cap(provenance="TRAINING_DISCOVERY")
        assert cap.provenance is DiscoveryProvenance.TRAINING_DISCOVERY
        validate_capture_provenance(cap, registry=_registry())

    def test_synthetic_fixture_requires_explicit_opt_in(self):
        cap = _cap(provenance=DiscoveryProvenance.SYNTHETIC_FIXTURE)
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(cap, registry=_registry())
        validate_capture_provenance(cap, registry=_registry(),
                                    allow_synthetic_fixture=True)

    def test_missing_protocol_or_world_hash_raises(self):
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(_cap(rollout_protocol_id=""),
                                        registry=_registry())
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(_cap(world_set_hash=""),
                                        registry=_registry())

    def test_non_hash_world_set_hash_raises(self):
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(_cap(world_set_hash="not-a-sha"),
                                        registry=_registry())


class TestWorldSetHashRegistration:
    def test_unregistered_world_set_hash_raises(self):
        # well-formed 64-hex but NOT in the discovery allowlist
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(_cap(world_set_hash="b" * 64),
                                        registry=_registry())

    def test_world_hash_of_another_registry_raises(self):
        other_hash = "c" * 64
        registry_a = _registry(allowed_discovery_assets=(
            DiscoveryAssetRecord("worlds_a", AssetKind.WORLD_SET,
                                 world_set_hash=other_hash),))
        # valid under registry A ...
        validate_capture_provenance(_cap(world_set_hash=other_hash),
                                    registry=registry_a)
        # ... but rejected by the default registry (wrong-registry defence)
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(_cap(world_set_hash=other_hash),
                                        registry=_registry())


class TestBypassClosures:
    """The exact bypasses named in the PASS_WITH_BLOCKER review."""

    def test_neutral_alias_bank_ref_is_rejected(self):
        # no FORMAL_* marker anywhere: the old marker denylist passed this;
        # allowlist binding rejects it because it is not a registered asset
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(_cap(bank_refs=("eval_holdout_bank",)),
                                        registry=_registry())

    def test_neutral_canonical_id_from_forbidden_set_is_rejected(self):
        # 'evaluation_holdout_alpha' is a formal bank with a neutral name —
        # registered nowhere, and its identity is in the forbidden set
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(
                _cap(bank_refs=("evaluation_holdout_alpha",)), registry=_registry())

    def test_bare_formal_sha_as_bank_ref_is_rejected(self):
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(_cap(bank_refs=(FORMAL_BANK_SHA,)),
                                        registry=_registry())

    def test_bare_formal_world_sha_in_notes_is_rejected(self):
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(
                _cap(notes={"seed_source": FORMAL_WORLD_SHA}), registry=_registry())

    @pytest.mark.parametrize("variant", [
        "formal_bank_01", "Formal_Bank_01", "fOrMaL_bAnK_01"])
    def test_case_variants_of_forbidden_id_are_rejected(self, variant):
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(_cap(notes={"ref": variant}),
                                        registry=_registry())

    def test_forbidden_id_nested_inside_note_text_is_rejected(self):
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(
                _cap(notes={"layer1": "see ref:FORMAL_BANK_01 inside layer one"}),
                registry=_registry())

    def test_forbidden_id_in_note_key_is_rejected(self):
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(
                _cap(notes={"FORMAL_WORLD_ICE_source": "x"}), registry=_registry())

    def test_forbidden_id_in_world_set_id_is_rejected(self):
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(_cap(world_set_id="FORMAL_WORLD_ICE"),
                                        registry=_registry())

    def test_unregistered_world_set_id_is_rejected(self):
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(_cap(world_set_id="some_other_worlds"),
                                        registry=_registry())

    def test_empty_bank_ref_is_rejected(self):
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(_cap(bank_refs=("",)), registry=_registry())

    def test_legacy_formal_markers_still_rejected(self):
        # tertiary layer: even if a marker string ever became 'registered',
        # the marker sweep and the leakage guard still block it
        for marker in ("FORMAL_FRONT", "FORMAL_BACK_BANK_B", "FORMAL_FULL",
                       "FORMAL_EVAL_WORLD_3"):
            with pytest.raises(ProvenanceViolationError):
                validate_capture_provenance(_cap(notes={"src": marker}),
                                            registry=_registry())

    def test_exact_formal_datasource_ref_raises_via_guard(self):
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(
                _cap(bank_refs=(DataSource.FORMAL_FRONT.value,)), registry=_registry())


class TestRoundStatusHonesty:
    def test_flags_are_downgraded_per_rule_5(self):
        assert DISCOVERY_PROVENANCE_CONTRACT_READY is True
        assert FROZEN_FORMAL_ASSET_REGISTRY_BOUND is False
        assert DISCOVERY_FORMAL_PROVENANCE_ISOLATED is False

    def test_registry_status_reports_blocked(self):
        status = registry_status()
        assert status["bound"] is False
        assert status["status"] == BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY
        assert status["contract_ready"] is True
        assert status["real_isolation_proven"] is False


class TestSourceMapping:
    def test_discovery_source_mapping(self):
        assert discovery_source_for(DiscoveryProvenance.TRAINING_DISCOVERY) \
            is DataSource.TRAINING_FRONTIER_CAPTURE
        assert discovery_source_for(DiscoveryProvenance.SYNTHETIC_FIXTURE) \
            is DataSource.SYNTHETIC_TEST

    def test_assert_not_formal_bridge(self):
        assert_not_formal(DataSource.TRAINING_FRONTIER_CAPTURE, consumer="FrontierArchive")
        with pytest.raises(ProvenanceViolationError):
            assert_not_formal(DataSource.FORMAL_FULL, consumer="FrontierArchive")
        with pytest.raises(ProvenanceViolationError):
            assert_not_formal("FORMAL_FRONT", consumer="curriculum")


class TestProductionPath:
    """CC4/E3: production capture paths require the controller-injected
    frozen formal asset registry; synthetic registries stay test-only.

    Every registry used here is still a SYNTHETIC fixture — these tests
    prove the mechanical enforcement contract only (real isolation stays
    BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY until the controller
    injects the real manifest).
    """

    @pytest.fixture(autouse=True)
    def _clean_production_slot(self):
        clear_injected_production_registry()
        yield
        clear_injected_production_registry()

    def test_production_path_without_injection_fails_closed(self):
        assert production_registry_bound() is False
        with pytest.raises(ProvenanceViolationError) as excinfo:
            validate_capture_provenance_production(_cap())
        assert BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY in str(excinfo.value)

    def test_test_only_registry_cannot_enter_production_slot(self):
        with pytest.raises(ProvenanceViolationError):
            inject_frozen_formal_asset_registry(_registry())  # usage=TEST_ONLY
        assert production_registry_bound() is False
        # production entry point stays blocked after the rejected injection
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance_production(_cap())

    def test_invalid_production_registry_cannot_enter_slot(self):
        # hash computed correctly for PRODUCTION usage, but frozen=False:
        # injection validates BEFORE binding and must fail closed
        bad = _registry(usage=REGISTRY_USAGE_PRODUCTION, frozen=False)
        with pytest.raises(ProvenanceViolationError):
            inject_frozen_formal_asset_registry(bad)
        assert production_registry_bound() is False

    def test_valid_production_injection_passes_legal_capture(self):
        inject_frozen_formal_asset_registry(_registry(usage=REGISTRY_USAGE_PRODUCTION))
        assert production_registry_bound() is True
        validate_capture_provenance_production(
            _cap(bank_refs=("training_capture_bank",),
                 world_set_id="discovery_worlds_v1",
                 notes={"purpose": "frontier collection"}))  # must not raise

    def test_valid_production_injection_rejects_bypass_captures(self):
        inject_frozen_formal_asset_registry(_registry(usage=REGISTRY_USAGE_PRODUCTION))
        # neutral-alias bank ref (the core PASS_WITH_BLOCKER bypass)
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance_production(
                _cap(bank_refs=("eval_holdout_bank",)))
        # neutral canonical id from the forbidden identity set
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance_production(
                _cap(bank_refs=("evaluation_holdout_alpha",)))
        # forbidden formal SHA embedded in notes
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance_production(
                _cap(notes={"seed_source": FORMAL_WORLD_SHA}))
        # unregistered world set hash
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance_production(_cap(world_set_hash="b" * 64))

    def test_production_path_never_accepts_synthetic_fixture(self):
        inject_frozen_formal_asset_registry(_registry(usage=REGISTRY_USAGE_PRODUCTION))
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance_production(
                _cap(provenance=DiscoveryProvenance.SYNTHETIC_FIXTURE))

    def test_double_injection_is_rejected(self):
        inject_frozen_formal_asset_registry(_registry(usage=REGISTRY_USAGE_PRODUCTION))
        with pytest.raises(ProvenanceViolationError):
            inject_frozen_formal_asset_registry(_registry(usage=REGISTRY_USAGE_PRODUCTION))

    def test_uninjected_production_registry_direct_use_is_rejected(self):
        # a PRODUCTION-labelled registry that bypasses the injection slot
        # cannot be handed to validate_capture_provenance directly
        rogue = _registry(usage=REGISTRY_USAGE_PRODUCTION)
        with pytest.raises(ProvenanceViolationError):
            validate_capture_provenance(_cap(), registry=rogue)
        assert production_registry_bound() is False

    def test_registry_hash_is_usage_sensitive(self):
        forbidden, allowed = _forbidden(), _allowed()
        test_hash = registry_hash_of("RID", "SIG", forbidden, allowed)
        prod_hash = registry_hash_of("RID", "SIG", forbidden, allowed,
                                     usage=REGISTRY_USAGE_PRODUCTION)
        assert test_hash != prod_hash
        # a PRODUCTION registry stamped with its TEST_ONLY hash fails closed
        forged = DiscoveryProvenanceRegistry(
            registry_id="RID", controller_signature_ref="SIG", frozen=True,
            forbidden_formal_identities=forbidden,
            allowed_discovery_assets=allowed, registry_hash=test_hash,
            usage=REGISTRY_USAGE_PRODUCTION)
        with pytest.raises(ProvenanceViolationError):
            validate_discovery_registry(forged)
        with pytest.raises(ProvenanceViolationError):
            inject_frozen_formal_asset_registry(forged)
        assert production_registry_bound() is False

    def test_registry_status_reflects_production_slot(self):
        status = registry_status()
        assert status["production_entrypoint_enforced"] is True
        assert status["production_registry_bound"] is False
        inject_frozen_formal_asset_registry(_registry(usage=REGISTRY_USAGE_PRODUCTION))
        assert registry_status()["production_registry_bound"] is True

    def test_clear_restores_blocked_state(self):
        inject_frozen_formal_asset_registry(_registry(usage=REGISTRY_USAGE_PRODUCTION))
        assert production_registry_bound() is True
        clear_injected_production_registry()
        assert production_registry_bound() is False
        with pytest.raises(ProvenanceViolationError) as excinfo:
            validate_capture_provenance_production(_cap())
        assert BLOCKED_WAITING_FROZEN_FORMAL_ASSET_REGISTRY in str(excinfo.value)
