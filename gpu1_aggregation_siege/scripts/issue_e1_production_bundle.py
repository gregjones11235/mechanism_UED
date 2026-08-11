"""Issue the REAL PRODUCTION E1 runtime bundle manifest.

Builds the manifest FROM the production FormalAssetRegistry (real
identity + implementation hashes of the 13 real objects), selects the
Persistent Student explicitly, signs it with the director signer and
registers the signature in the verifier trust store. The output is the
ONLY admissible production bundle for the E1 one-window pipeline.

Idempotent: re-running verifies the existing bundle instead of
re-signing (idempotent issuance, no duplicate artifacts).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("DICODE_SHARED_RUNTIME_REAL", "1")

SIEGE_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
BUNDLE_PATH = os.path.join(
    SIEGE_ROOT, "reports", "e1_formal_ued",
    "e1_production_runtime_bundle.json")
TRUST_STORE_PATH = os.path.join(
    SIEGE_ROOT, "reports", "e1_formal_ued",
    "e1_production_bundle_trust_store.json")

DIRECTOR_SIGNER = "mechanism_UED_director_cc"
PERSISTENT = "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"


def main() -> int:
    from dicode.shared_runtime.registry import (
        canonical_sha256,
        production_registry,
    )
    from dicode.teachers.e1_formal import runtime_bundle as RB
    from dicode.teachers.e1_formal import runtime_objects as RO

    registry = production_registry()

    if os.path.isfile(BUNDLE_PATH) and os.path.isfile(TRUST_STORE_PATH):
        with open(BUNDLE_PATH, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        bundle = RB.load_verified_runtime_bundle(
            existing, "issue.idempotent-check")
        print("PRODUCTION bundle already issued and verifies:",
              BUNDLE_PATH)
        print("bundle_hash:", bundle.bundle_hash)
        return 0

    # ---- the 13 runtime object descriptors (from the registry) -------
    runtime_objects = {}
    for contract in RO.REQUIRED_RUNTIME_OBJECTS:
        obj = registry.resolve_asset(contract=contract)
        runtime_objects[contract] = {
            "identity_hash": registry.declared_identity(contract),
            "implementation_hash":
                registry._assets[contract]["implementation_hash"],
            "source_commit": registry.source_commit,
            "registry_identity": registry.registry_identity,
        }
    ro_hash = RO.compute_runtime_objects_hash(
        RO.parse_runtime_objects(runtime_objects, "issue.runtime_objects"))

    # ---- the nine capability identities (alias-mapped) ---------------
    alias = RO.CAPABILITY_ALIAS_TO_CANONICAL
    object_identity_hashes = {}
    for contract in RB.RUNTIME_CAPABILITY_CONTRACTS:
        if contract == "formal_asset_registry":
            # the registry itself is never one of its own assets: its
            # identity is the registry_identity (cross-bound to the
            # manifest's registry_identity/registry_hash fields)
            object_identity_hashes[contract] = registry.registry_identity
            continue
        canonical = alias.get(contract, contract)
        object_identity_hashes[contract] = registry.declared_identity(
            canonical)

    # ---- the explicit Persistent Student selection -------------------
    from dicode.shared_runtime import student_assets as SA

    student_adapter = SA.real_student_adapter(PERSISTENT)
    loc = SA.AL.student_locations()
    student_selection = {
        "selected_candidate_id": PERSISTENT,
        "profile_id": "rmt16_persistent_98304",
        "architecture_family": "RMT16",
        "memory_mode": "PERSISTENT",
        "memory_spec_hash": SA.real_student_identity(
            PERSISTENT).memory_spec_hash,
        "carry_mode": "persistent-memory-progression",
        "checkpoint_path": loc["persistent_checkpoint"],
        "checkpoint_file_sha256": student_adapter.checkpoint_file_sha256,
        "params_sha256": student_adapter.params_sha256,
        "adapter_identity_hash": student_adapter.object_identity_hash,
        "adapter_implementation_hash":
            registry._assets["student_adapter"]["implementation_hash"],
        "driver_source_path": loc["driver_source"],
        "driver_source_sha256": loc["driver_source_sha256"],
        "source_commit": registry.source_commit,
    }
    descriptor = RB.parse_student_selection(
        student_selection, "issue.student_selection")

    # ---- assemble + hash the PRODUCTION manifest ----------------------
    bundle_id = "e1-production-runtime-bundle-v1"
    source_commit = registry.source_commit
    authorization_grant_hash = canonical_sha256({
        "kind": "e1.production_authorization_grant",
        "formal_longrun_authorized": False,
        "formal_experiment_started": False,
        "smoke_scope": "one_review_window_one_canonical_update",
    })
    signature_ref = hashlib.sha256((
        "director-signature|" + bundle_id + "|" + source_commit
    ).encode("utf-8")).hexdigest()
    bundle_hash = RB.compute_bundle_hash(
        bundle_id=bundle_id,
        mode=RB.BUNDLE_MODE_PRODUCTION,
        source_commit=source_commit,
        signer_id=DIRECTOR_SIGNER,
        authorization_grant_hash=authorization_grant_hash,
        object_identity_hashes=object_identity_hashes,
        student_selection_hash=descriptor.descriptor_hash,
        signature_ref=signature_ref,
        registry_identity=registry.registry_identity,
        registry_hash=registry.registry_hash,
        runtime_objects_hash=ro_hash,
    )
    manifest = {
        "bundle_id": bundle_id,
        "mode": RB.BUNDLE_MODE_PRODUCTION,
        "source_commit": source_commit,
        "signer_id": DIRECTOR_SIGNER,
        "authorization_grant_hash": authorization_grant_hash,
        "object_identity_hashes": object_identity_hashes,
        "student_selection": student_selection,
        "signature_ref": signature_ref,
        "registry_identity": registry.registry_identity,
        "registry_hash": registry.registry_hash,
        "runtime_objects": runtime_objects,
        "bundle_hash": bundle_hash,
    }
    # fail-closed verification of what we just signed
    bundle = RB.load_verified_runtime_bundle(manifest, "issue.final")

    os.makedirs(os.path.dirname(BUNDLE_PATH), exist_ok=True)
    with open(BUNDLE_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    trust_store = {
        "schema": "mechanism_UED.e1_bundle_trust_store/v1",
        "trusted_signers": [DIRECTOR_SIGNER],
        "trusted_signer_registry_hash": canonical_sha256(
            {"trusted_signers": [DIRECTOR_SIGNER]}),
        "issued_bundles": [{
            "bundle_id": bundle_id,
            "bundle_hash": bundle.bundle_hash,
            "signature_ref": signature_ref,
            "signer_id": DIRECTOR_SIGNER,
            "registry_identity": registry.registry_identity,
        }],
    }
    with open(TRUST_STORE_PATH, "w", encoding="utf-8") as handle:
        json.dump(trust_store, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("PRODUCTION bundle issued:", BUNDLE_PATH)
    print("bundle_hash:", bundle.bundle_hash)
    return 0


if __name__ == "__main__":
    sys.exit(main())
