#!/usr/bin/env python3
"""E3 authorization manifest — REAL Ed25519 signatures (audit-hardened).

Sole-controller directive 2026-08-10:

  * ``controller_signature_ref`` is gone.  The manifest carries a REAL
    Ed25519 signature produced by the controlled local controller's private
    key.  The runner holds ONLY the controller's public key (verification
    material) — it can verify but never forge.
  * The manifest BINDS: source commit, candidate, runner SHA (the formal
    controller script), checkpoint SHA, Student profile SHA, task-asset
    manifest SHA and the formal_asset_registry_hash.
  * formal_asset_registry_hash is RECOMPUTED by the runner from the actual
    registry file (auth/formal_asset_registry.json) and every asset's file
    SHA256 is re-verified against the registry — not merely checked for
    "64 hex chars and non-zero".
  * Anything missing / tampered / mismatched FAILS CLOSED before any output
    dir, LLM call or GPU initialization.

The signature covers the canonical payload (all binding fields EXCEPT
``signature`` and ``manifest_hash``), JSON-serialised with sort_keys — the
same bytes the local controller signed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from dicode.simulator_frontier.ed25519_pure import verify_bytes


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_payload(payload: Mapping[str, Any]) -> str:
    """The EXACT bytes the local controller signs: payload minus
    ``signature`` / ``manifest_hash``, JSON sort_keys + separators."""
    base = {k: v for k, v in payload.items()
            if k not in ("signature", "manifest_hash")}
    return json.dumps(base, sort_keys=True, default=str, ensure_ascii=False,
                      separators=(",", ":"))


def recompute_registry_hash(registry_path: str) -> str:
    """SHA256 of the actual registry FILE's canonical bytes."""
    with open(registry_path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def verify_registry_assets(registry_path: str) -> dict:
    """Re-verify EVERY asset's file SHA256 against the registry file.

    Returns {asset_id: {"kind":..., "sha256_ok": bool}}.  Raises on any
    missing file / missing entry / mismatch (fail closed).
    """
    with open(registry_path, "r", encoding="utf-8") as fh:
        registry = json.load(fh)
    if registry.get("schema") != "e3_formal_asset_registry/v2":
        raise ValueError(
            "formal asset registry schema mismatch (fail closed)")
    assets = registry.get("assets", {})
    if not isinstance(assets, Mapping) or not assets:
        raise ValueError(
            "formal asset registry has no assets (fail closed)")
    verified = {}
    for aid, meta in assets.items():
        path = meta.get("path")
        want = meta.get("sha256")
        kind = meta.get("kind")
        if not path or not os.path.isfile(path):
            raise ValueError(
                f"registry asset {aid!r} file missing: {path} (fail closed)")
        if not want or len(want) != 64:
            raise ValueError(
                f"registry asset {aid!r} sha256 malformed (fail closed)")
        got = _sha256_file(path)
        if got != want:
            raise ValueError(
                f"registry asset {aid!r} SHA mismatch: file={got} "
                f"registry={want} (fail closed)")
        verified[aid] = {"kind": kind, "sha256_ok": True}
    return verified


@dataclass(frozen=True)
class E3Authorization:
    """Controller-signed, independently verifiable authorization."""

    authorization_id: str
    source_commit: str
    candidate_id: str
    runner_sha256: str
    checkpoint_sha256: str
    student_profile_sha256: str
    task_asset_manifest_sha256: str
    formal_asset_registry_hash: str
    allowed_heads: tuple[str, ...]
    issued_at_utc: str
    scope: str
    signer_public_key_fingerprint: str
    signature: str
    manifest_hash: str

    @property
    def canonical(self) -> str:
        return canonical_payload(self.__dict__)

    def __post_init__(self) -> None:
        for name, val in (
            ("authorization_id", self.authorization_id),
            ("source_commit", self.source_commit),
            ("candidate_id", self.candidate_id),
            ("runner_sha256", self.runner_sha256),
            ("checkpoint_sha256", self.checkpoint_sha256),
            ("student_profile_sha256", self.student_profile_sha256),
            ("task_asset_manifest_sha256", self.task_asset_manifest_sha256),
            ("formal_asset_registry_hash", self.formal_asset_registry_hash),
            ("signer_public_key_fingerprint",
             self.signer_public_key_fingerprint),
        ):
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{name} empty (fail closed)")
        # source_commit is a git commit (40-hex); all *_sha256 fields are 64-hex.
        if len(self.source_commit) != 40:
            raise ValueError(
                f"source_commit must be a full 40-hex git sha, got len "
                f"{len(self.source_commit)} (fail closed)")
        for name, val in (
            ("runner_sha256", self.runner_sha256),
            ("checkpoint_sha256", self.checkpoint_sha256),
            ("student_profile_sha256", self.student_profile_sha256),
            ("task_asset_manifest_sha256", self.task_asset_manifest_sha256),
            ("formal_asset_registry_hash", self.formal_asset_registry_hash),
            ("signer_public_key_fingerprint",
             self.signer_public_key_fingerprint),
        ):
            if len(val) != 64:
                raise ValueError(
                    f"{name} must be a full 64-hex sha256, got len {len(val)} "
                    "(fail closed)")


def _load_public_key(public_key_path: str) -> bytes:
    if not os.path.isfile(public_key_path):
        raise ValueError(
            f"controller public key not found: {public_key_path} (fail closed)")
    with open(public_key_path, "rb") as fh:
        # Raw 32-byte Ed25519 public key.  NEVER strip: a key whose final byte
        # is an ASCII whitespace byte would be silently truncated by strip().
        pk = fh.read()
    if len(pk) != 32:
        raise ValueError(
            f"controller public key must be 32 bytes, got {len(pk)} "
            "(fail closed)")
    return pk


def verify_signature(auth: E3Authorization, public_key_path: str) -> None:
    """Verify the Ed25519 signature over the canonical payload.  The runner
    holds only the public key — it can verify but never forge."""
    pk = _load_public_key(public_key_path)
    got_fp = hashlib.sha256(pk).hexdigest()
    if got_fp != auth.signer_public_key_fingerprint:
        raise ValueError(
            f"public key fingerprint mismatch: runner key={got_fp} "
            f"manifest={auth.signer_public_key_fingerprint} (fail closed)")
    try:
        sig = base64.b64decode(auth.signature)
    except Exception as exc:
        raise ValueError(f"manifest signature not base64: {exc!r} (fail closed)") from exc
    if len(sig) != 64:
        raise ValueError("manifest signature must be 64 bytes (fail closed)")
    try:
        verify_bytes(auth.canonical.encode("utf-8"), sig, pk)
    except ValueError as exc:
        raise ValueError(
            f"manifest Ed25519 signature INVALID: {exc} (fail closed)") from exc


def load_authorization(auth_manifest_path: str, *,
                       public_key_path: str,
                       registry_path: str) -> E3Authorization:
    """Load + verify a controller-signed authorization manifest.

    Verifies: manifest_hash integrity, Ed25519 signature, registry hash
    recomputation and per-asset SHA verification.  Any failure raises
    ValueError — the caller MUST treat it as a hard block BEFORE output dirs,
    LLM calls or GPU init.
    """
    path = Path(auth_manifest_path)
    if not path.is_file():
        raise ValueError(
            f"E3 authorization manifest not found: {path} (fail closed)")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"E3 authorization manifest unreadable: {exc!r} (fail closed)") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("E3 authorization manifest must be a JSON object")

    # 1. manifest_hash integrity (payload minus manifest_hash).
    declared_hash = payload.get("manifest_hash")
    if not declared_hash:
        raise ValueError("manifest_hash missing (fail closed)")
    recomputed = _sha256_text(canonical_payload(payload))
    if declared_hash != recomputed:
        raise ValueError(
            "manifest_hash drift (tampered manifest; fail closed)")

    auth = E3Authorization(
        authorization_id=str(payload.get("authorization_id", "")),
        source_commit=str(payload.get("source_commit", "")),
        candidate_id=str(payload.get("candidate_id", "")),
        runner_sha256=str(payload.get("runner_sha256", "")),
        checkpoint_sha256=str(payload.get("checkpoint_sha256", "")),
        student_profile_sha256=str(payload.get("student_profile_sha256", "")),
        task_asset_manifest_sha256=str(
            payload.get("task_asset_manifest_sha256", "")),
        formal_asset_registry_hash=str(
            payload.get("formal_asset_registry_hash", "")),
        allowed_heads=tuple(str(h) for h in payload.get("allowed_heads", [])),
        issued_at_utc=str(payload.get("issued_at_utc", "")),
        scope=str(payload.get("scope", "")),
        signer_public_key_fingerprint=str(
            payload.get("signer_public_key_fingerprint", "")),
        signature=str(payload.get("signature", "")),
        manifest_hash=declared_hash,
    )

    # 2. Ed25519 signature (runner holds only the public key).
    verify_signature(auth, public_key_path)

    # 3. formal_asset_registry_hash recomputed from the ACTUAL registry file.
    actual_reg_hash = recompute_registry_hash(registry_path)
    if actual_reg_hash != auth.formal_asset_registry_hash:
        raise ValueError(
            f"formal_asset_registry_hash mismatch: actual registry file="
            f"{actual_reg_hash} manifest={auth.formal_asset_registry_hash} "
            "(fail closed)")

    # 4. per-asset SHA verification against the actual files.
    verify_registry_assets(registry_path)

    return auth


def verify_runtime_authorization(auth: E3Authorization, runtime_head: str,
                                 candidate_id: str,
                                 runner_sha256: str,
                                 checkpoint_sha256: str,
                                 student_profile_sha256: str,
                                 registry_path: str) -> dict:
    """P0-7/audit: bind the RUNNING artifacts to the signed manifest.

    - runtime HEAD must equal auth.source_commit OR be an explicit
      allowed_head (empty allowed_heads grants nothing).
    - candidate / runner SHA / checkpoint SHA / Student profile SHA must all
      match the signed manifest.
    - registry hash + per-asset re-verified (independent of manifest).
    Returns the per-asset verification map.  Any mismatch raises (fail
    closed, before output/LLM/GPU).
    """
    if runtime_head != auth.source_commit:
        if runtime_head not in auth.allowed_heads:
            raise ValueError(
                f"runtime HEAD {runtime_head[:12]} not authorized for signed "
                f"commit {auth.source_commit[:12]} (fail closed)")
    if candidate_id != auth.candidate_id:
        raise ValueError(
            f"candidate {candidate_id!r} != signed {auth.candidate_id!r} "
            "(fail closed)")
    if runner_sha256 != auth.runner_sha256:
        raise ValueError(
            f"runner SHA mismatch: actual={runner_sha256[:16]} signed="
            f"{auth.runner_sha256[:16]} (fail closed)")
    if checkpoint_sha256 != auth.checkpoint_sha256:
        raise ValueError(
            f"checkpoint SHA mismatch: actual={checkpoint_sha256[:16]} signed="
            f"{auth.checkpoint_sha256[:16]} (fail closed)")
    if student_profile_sha256 != auth.student_profile_sha256:
        raise ValueError(
            f"student profile SHA mismatch: actual={student_profile_sha256[:16]} "
            f"signed={auth.student_profile_sha256[:16]} (fail closed)")
    verified = verify_registry_assets(registry_path)
    actual_reg_hash = recompute_registry_hash(registry_path)
    if actual_reg_hash != auth.formal_asset_registry_hash:
        raise ValueError(
            f"registry hash drifted at runtime: {actual_reg_hash[:16]} != "
            f"{auth.formal_asset_registry_hash[:16]} (fail closed)")
    return verified


def claim_output_dir(run_dir: str, run_id: str) -> None:
    """P0-8: atomically claim an output directory (unchanged)."""
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
