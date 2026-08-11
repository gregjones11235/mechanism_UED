"""The REAL director bundle verifier.

The production trust root: verifies a PRODUCTION runtime bundle against
the director trust store (trusted signer list + issued-bundle
records). The verifier NEVER reads identity from the object being
verified — it checks the manifest payload against the independent
issuance record and the production registry identity.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict

from . import asset_locations as AL

TRUST_STORE_RELATIVE = os.path.join(
    "reports", "e1_formal_ued", "e1_production_bundle_trust_store.json")


class DirectorVerifierError(RuntimeError):
    """Fail-closed director verification violation."""


class ProductionDirectorVerifier:
    """Implements the DirectorBundleVerifier protocol with a real trust
    store (trusted signers + issued bundle records)."""

    def __init__(self, trust_store_path: str = ""):
        path = trust_store_path or AL.resolve_repo_relative(
            TRUST_STORE_RELATIVE)
        if not os.path.isfile(path):
            raise DirectorVerifierError(
                "DIRECTOR_TRUST_STORE_MISSING: the director bundle trust "
                f"store {path!r} does not exist; run "
                "scripts/issue_e1_production_bundle.py first")
        with open(path, "r", encoding="utf-8") as handle:
            store: Dict[str, Any] = json.load(handle)
        self._trusted_signers = set(store.get("trusted_signers", []))
        self._issued = {
            record["bundle_hash"]: record
            for record in store.get("issued_bundles", [])
        }
        self.verifier_id = "mechanism_UED.production_director_verifier"
        self.verifier_identity_hash = hashlib.sha256(
            b"shared_runtime.production_director_verifier.v1"
        ).hexdigest()
        self.trusted_signer_registry_hash = str(
            store.get("trusted_signer_registry_hash", ""))
        self._registry_identity_expected = {
            record["bundle_hash"]: record.get("registry_identity", "")
            for record in store.get("issued_bundles", [])
        }

    def verify_bundle(self, *, signer_id: str, payload_hash: str,
                      signature_ref: str, source_commit: str,
                      registry_identity: str) -> bool:
        """Strictly True only when EVERY trust check passes."""
        if signer_id not in self._trusted_signers:
            return False
        if not signature_ref or not signature_ref.strip():
            return False
        record = self._issued.get(payload_hash)
        if record is None:
            # the payload hash was never issued by the director
            return False
        if record.get("signer_id") != signer_id:
            return False
        if record.get("signature_ref") != signature_ref:
            return False
        expected_registry = record.get("registry_identity", "")
        if expected_registry and registry_identity != expected_registry:
            return False
        return True

    def signer_trusted(self, signer_id: str) -> bool:
        return signer_id in self._trusted_signers

    def verify_source_commit(self, source_commit: str) -> bool:
        return bool(source_commit) and (
            source_commit.startswith("src-sha256:")
            or len(source_commit) == 40)
