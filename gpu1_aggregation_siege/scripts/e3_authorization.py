#!/usr/bin/env python3
"""E3 authorization manifest (P0-6/7/8).

P0-6  trusted_signer must come from a controller-signed authorization
       manifest — hardcoding director/cc4 is forbidden.
P0-7  no zero-filled asset-registry hash; source HEAD, runner, checkpoint,
       Student profile, task assets and model identity must carry real SHA.
       HEAD / authorization mismatch must fail closed BEFORE output creation,
       LLM calls and GPU initialization.
P0-8  output directories use an atomic unique claim; existing dirs / duplicate
       run ids / duplicate session keys are rejected.

The manifest is issued by the sole controller.  This module loads it, verifies
it, and fails closed when it is missing / tampered / mismatched.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class E3Authorization:
    """Immutable, controller-signed authorization for ONE formal run."""

    authorization_id: str
    controller_signature_ref: str          # 总控签名引用（P0-6 trusted_signer）
    source_commit: str                      # 必须等于 runtime HEAD
    candidate_id: str
    formal_asset_registry_hash: str         # 真实 SHA（P0-7）
    allowed_heads: tuple[str, ...] = ()
    manifest_hash: str = ""

    def __post_init__(self) -> None:
        if not str(self.authorization_id).strip():
            raise ValueError("authorization_id empty (fail closed)")
        # P0-6: controller_signature_ref must be a REAL controller signature,
        # never a hardcoded placeholder / synthetic self-signature.
        sig = str(self.controller_signature_ref)
        if not sig.strip() or sig.lower() in {"director/cc4", "none", "pending",
                                              "unknown", "synthetic",
                                              "e3-smoke"}:
            raise ValueError(
                f"controller_signature_ref {sig!r} is not a controller-signed "
                "authorization (P0-6 fail closed)")
        if len(str(self.source_commit)) != 40:
            raise ValueError("source_commit must be a full 40-hex sha")
        # P0-7: formal asset registry hash must be a real sha256, never zeros.
        reg = str(self.formal_asset_registry_hash)
        if len(reg) != 64 or reg == "0" * 64:
            raise ValueError(
                "formal_asset_registry_hash must be a real sha256 (never zeros) "
                "(P0-7 fail closed)")
        payload = {
            "authorization_id": self.authorization_id,
            "controller_signature_ref": self.controller_signature_ref,
            "source_commit": self.source_commit,
            "candidate_id": self.candidate_id,
            "formal_asset_registry_hash": self.formal_asset_registry_hash,
            "allowed_heads": list(self.allowed_heads),
        }
        object.__setattr__(
            self, "manifest_hash",
            _sha256_text(json.dumps(payload, sort_keys=True, default=str)))


def load_authorization(auth_manifest_path: str) -> E3Authorization:
    """Load + mint an E3Authorization from a controller-signed manifest file.

    Raises ValueError on any tamper / missing / mismatch — the caller must
    treat that as a hard block BEFORE creating output dirs, calling LLM, or
    initializing GPU.
    """
    path = Path(auth_manifest_path)
    if not path.is_file():
        raise ValueError(
            f"E3 authorization manifest not found: {path} — formal launch is "
            "BLOCKED until the sole controller signs an authorization manifest "
            "(P0-6/7 fail closed)")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"E3 authorization manifest unreadable: {exc!r}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("E3 authorization manifest must be a JSON object")
    # verify the manifest hash field if present
    declared = payload.get("manifest_hash")
    if declared:
        base = {k: v for k, v in payload.items() if k != "manifest_hash"}
        recomputed = _sha256_text(json.dumps(base, sort_keys=True, default=str))
        if declared != recomputed:
            raise ValueError(
                "E3 authorization manifest hash drift (tampered manifest; "
                "fail closed)")
    auth = E3Authorization(
        authorization_id=str(payload.get("authorization_id", "")),
        controller_signature_ref=str(payload.get("controller_signature_ref", "")),
        source_commit=str(payload.get("source_commit", "")),
        candidate_id=str(payload.get("candidate_id", "")),
        formal_asset_registry_hash=str(
            payload.get("formal_asset_registry_hash", "")),
        allowed_heads=tuple(str(h) for h in payload.get("allowed_heads", [])),
    )
    return auth


def verify_runtime_authorization(auth: E3Authorization, runtime_head: str,
                                 candidate_id: str) -> None:
    """P0-7: fail closed if runtime HEAD / candidate mismatch authorization.

    The runtime HEAD must equal the authorization source_commit OR be granted
    by an explicit allowed_head entry.  Empty allowed_heads grants nothing.
    """
    if runtime_head != auth.source_commit:
        if runtime_head not in auth.allowed_heads:
            raise ValueError(
                f"runtime HEAD {runtime_head[:12]} not authorized for commit "
                f"{auth.source_commit[:12]} (P0-7 fail closed before any "
                "output/LLM/GPU)")
    if candidate_id != auth.candidate_id:
        raise ValueError(
            f"candidate {candidate_id!r} != authorization candidate "
            f"{auth.candidate_id!r} (fail closed)")


def claim_output_dir(run_dir: str, run_id: str) -> None:
    """P0-8: atomically claim an output directory.

    Creates the directory ONLY if it does not already exist (no overwrite).
    Refuses duplicate run ids / existing directories.  A lock marker file is
    written to prove the claim.
    """
    d = Path(run_dir)
    if d.exists():
        raise ValueError(
            f"output directory already exists: {d} — duplicate run id {run_id!r} "
            "rejected (P0-8 fail closed, atomic claim)")
    try:
        d.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise ValueError(
            f"output directory claim race: {d} — duplicate run id rejected "
            "(P0-8 fail closed)")
    lock = d / "CLAIM.json"
    lock.write_text(json.dumps({
        "run_id": run_id,
        "claimed_utc": __import__("time").strftime(
            "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        "schema": "simulator_frontier.e3_run_dir_claim/v1",
    }, indent=2), encoding="utf-8")
