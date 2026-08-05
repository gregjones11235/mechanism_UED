# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

CC4 follow-up (P0-1): the production archive write path resolves the frozen
formal asset registry ONLY from the controller injection slot.  A TEST_ONLY
registry can never enter the production slot, the slot refuses to re-inject
without an explicit clear, and the production reader fails closed when
nothing is bound.  These tests prove the injection channel is unbypassable.
"""

import pytest

from dicode.simulator_frontier.discovery_provenance import (
    REGISTRY_USAGE_PRODUCTION,
    REGISTRY_USAGE_TEST_ONLY,
    AssetKind,
    DiscoveryAssetRecord,
    DiscoveryProvenanceRegistry,
    FormalAssetIdentity,
    ProvenanceViolationError,
    clear_injected_production_registry,
    inject_frozen_formal_asset_registry,
    production_registry,
    production_registry_bound,
    registry_hash_of,
)

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True


def _forbidden() -> tuple[FormalAssetIdentity, ...]:
    return (
        FormalAssetIdentity(asset_kind=AssetKind.BANK,
                            canonical_id="formal_bank_fixture_98304",
                            sha256="b" * 64),
        FormalAssetIdentity(asset_kind=AssetKind.WORLD_SET,
                            canonical_id="formal_world_fixture_98304",
                            sha256="c" * 64),
    )


def _allowed() -> tuple[DiscoveryAssetRecord, ...]:
    return (
        DiscoveryAssetRecord(asset_id="discovery_state_a",
                             asset_kind=AssetKind.WORLD_SET,
                             world_set_hash="d" * 64),
        DiscoveryAssetRecord(asset_id="discovery_state_b",
                             asset_kind=AssetKind.BANK,
                             content_sha256="e" * 64),
    )


def _registry(usage: str) -> DiscoveryProvenanceRegistry:
    forbidden = _forbidden()
    allowed = _allowed()
    registry_hash = registry_hash_of("reg-fixture", "controller-signature/reg",
                                     forbidden, allowed, usage=usage)
    return DiscoveryProvenanceRegistry(
        registry_id="reg-fixture",
        controller_signature_ref="controller-signature/reg",
        frozen=True,
        forbidden_formal_identities=forbidden,
        allowed_discovery_assets=allowed,
        registry_hash=registry_hash,
        usage=usage,
    )


@pytest.fixture(autouse=True)
def _clean_slot():
    clear_injected_production_registry()
    yield
    clear_injected_production_registry()


def test_test_only_registry_can_never_enter_production_slot():
    test_only = _registry(REGISTRY_USAGE_TEST_ONLY)
    assert test_only.usage == REGISTRY_USAGE_TEST_ONLY
    with pytest.raises(ProvenanceViolationError):
        inject_frozen_formal_asset_registry(test_only)
    assert not production_registry_bound()


def test_production_registry_enters_only_via_injection_slot():
    prod = _registry(REGISTRY_USAGE_PRODUCTION)
    inject_frozen_formal_asset_registry(prod)
    assert production_registry_bound()
    assert production_registry() is prod


def test_production_reader_fails_closed_when_unbound():
    with pytest.raises(ProvenanceViolationError):
        production_registry()


def test_reinjection_without_clear_is_a_violation():
    prod = _registry(REGISTRY_USAGE_PRODUCTION)
    inject_frozen_formal_asset_registry(prod)
    with pytest.raises(ProvenanceViolationError):
        inject_frozen_formal_asset_registry(prod)


def test_clear_then_reinject_is_allowed():
    prod = _registry(REGISTRY_USAGE_PRODUCTION)
    inject_frozen_formal_asset_registry(prod)
    clear_injected_production_registry()
    assert not production_registry_bound()
    inject_frozen_formal_asset_registry(prod)
    assert production_registry_bound()


def test_fake_mapping_registry_is_rejected_fail_closed():
    # A caller-supplied Mapping can never be a registry: the injection slot
    # type-checks and the production reader only returns the injected object.
    with pytest.raises(ProvenanceViolationError):
        inject_frozen_formal_asset_registry({"registry_id": "spoofed"})
