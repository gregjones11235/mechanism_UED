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
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

# Do not import ``dicode.simulator_frontier`` here: importing a package runs
# its __init__, whose production surface imports JAX.  Authorization is the
# pre-GPU gate, so load the dependency-free verifier directly from its file.
_ED25519_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "src", "dicode", "simulator_frontier",
    "ed25519_pure.py"))
_ED_SPEC = importlib.util.spec_from_file_location(
    "_e3_ed25519_pure_preflight", _ED25519_PATH)
if _ED_SPEC is None or _ED_SPEC.loader is None:
    raise ImportError("cannot load dependency-free Ed25519 verifier")
_ED_MODULE = importlib.util.module_from_spec(_ED_SPEC)
_ED_SPEC.loader.exec_module(_ED_MODULE)
verify_bytes = _ED_MODULE.verify_bytes


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


@dataclass(frozen=True)
class E3FormalBudget:
    """Immutable, fail-closed E3 budget contract."""

    candidate: str
    sessions: int
    updates_per_session: int
    env_steps_per_update: int
    resolved_env_steps: int
    budget_semantics: str
    scope: str


FORMAL_BUDGET_CANDIDATE = "SLOWGRU_PERSISTENT_CANONICAL_98304"
FORMAL_BUDGET_SESSIONS = 151
FORMAL_BUDGET_UPDATES_PER_SESSION = 100
FORMAL_BUDGET_ENV_STEPS_PER_UPDATE = 131072
FORMAL_BUDGET_RESOLVED_ENV_STEPS = 1_979_187_200
FORMAL_BUDGET_SEMANTICS = "ADDITIONAL_FROM_PRETRAINED_CHECKPOINT"


def resolve_e3_budget(*, candidate: str, sessions: int, scope: str = "formal") -> E3FormalBudget:
    """Resolve only the canonical formal budget or a small verification scope."""
    if isinstance(sessions, bool) or not isinstance(sessions, int) or sessions <= 0:
        raise ValueError("sessions must be a positive integer (fail closed)")
    if scope == "formal":
        if candidate != FORMAL_BUDGET_CANDIDATE or sessions != FORMAL_BUDGET_SESSIONS:
            raise ValueError("formal scope requires canonical SlowGRU 151-session budget")
    elif scope != "verification":
        raise ValueError(f"unknown E3 budget scope {scope!r}")
    if scope == "verification" and sessions > 3:
        raise ValueError("verification scope is capped at 3 sessions")
    updates = sessions * FORMAL_BUDGET_UPDATES_PER_SESSION
    steps = updates * FORMAL_BUDGET_ENV_STEPS_PER_UPDATE
    return E3FormalBudget(
        candidate=candidate, sessions=sessions,
        updates_per_session=FORMAL_BUDGET_UPDATES_PER_SESSION,
        env_steps_per_update=FORMAL_BUDGET_ENV_STEPS_PER_UPDATE,
        resolved_env_steps=(FORMAL_BUDGET_RESOLVED_ENV_STEPS if scope == "formal" else steps),
        budget_semantics=FORMAL_BUDGET_SEMANTICS,
        scope=scope,
    )


def recompute_registry_hash(registry_path: str) -> str:
    """SHA256 of the actual registry FILE's canonical bytes."""
    with open(registry_path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _resolve_registry_asset_path(registry_path: str, raw_path: str) -> str:
    """Resolve a registry path without allowing relative assets to escape.

    Repo-owned assets are stored relative to the registry so a clean deploy
    may relocate the whole worktree.  Absolute paths remain reserved for
    external immutable assets (for example the student checkpoint).
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("registry asset path empty (fail closed)")
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return str(candidate)
    root = Path(registry_path).resolve().parents[1]
    resolved = (Path(registry_path).resolve().parent / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"relative registry asset escapes repository: {raw_path!r}") from exc
    return str(resolved)


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
        path = _resolve_registry_asset_path(registry_path, meta.get("path", ""))
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
        # A task manifest is itself a registry asset.  Re-verify every source
        # file it references so a relocated deployment cannot silently use a
        # different task implementation.
        if kind == "task_asset_manifest":
            with open(path, "r", encoding="utf-8") as nested_fh:
                nested = json.load(nested_fh)
            task_assets = nested.get("task_assets", {})
            if not isinstance(task_assets, Mapping):
                raise ValueError("task manifest task_assets malformed")
            for task_id, task_meta in task_assets.items():
                if not isinstance(task_meta, Mapping):
                    raise ValueError(f"task asset {task_id!r} malformed")
                nested_path = _resolve_registry_asset_path(path, task_meta.get("path", ""))
                nested_sha = str(task_meta.get("sha256", ""))
                if len(nested_sha) != 64 or not os.path.isfile(nested_path):
                    raise ValueError(f"task asset {task_id!r} missing or malformed")
                if _sha256_file(nested_path) != nested_sha:
                    raise ValueError(f"task asset {task_id!r} SHA mismatch")
    return verified


def resolve_candidate_static_assets(registry_path: str,
                                    candidate_id: str) -> dict[str, str]:
    """Resolve authorization inputs without importing JAX or mounting a model.

    A formal candidate must have exactly one checkpoint asset.  A future
    immutable Student-profile file is additionally returned when present, but
    is not required by the current registry: its authoritative params identity
    is already signed in ``E3Authorization.student_profile_sha256`` and is
    checked against the real mount after this pre-GPU gate.  The task manifest
    remains mandatory.
    """
    with open(registry_path, "r", encoding="utf-8") as fh:
        registry = json.load(fh)
    assets = registry.get("assets", {})
    if not isinstance(assets, Mapping):
        raise ValueError("formal asset registry assets malformed (fail closed)")

    def _one(kind: str, *, candidate_required: bool) -> Mapping[str, Any]:
        matches = []
        for meta in assets.values():
            if not isinstance(meta, Mapping) or meta.get("kind") != kind:
                continue
            if candidate_required and meta.get("candidate") != candidate_id:
                continue
            matches.append(meta)
        if len(matches) != 1:
            raise ValueError(
                f"registry requires exactly one {kind!r} asset for "
                f"candidate {candidate_id!r}, got {len(matches)} (fail closed)")
        return matches[0]

    checkpoint = _one("student_checkpoint", candidate_required=True)
    profiles = [meta for meta in assets.values()
                if isinstance(meta, Mapping)
                and meta.get("kind") == "student_profile"
                and meta.get("candidate") == candidate_id]
    if len(profiles) > 1:
        raise ValueError(
            f"registry has ambiguous student_profile assets for candidate "
            f"{candidate_id!r} (fail closed)")
    profile = profiles[0] if profiles else None
    task_manifest = _one("task_asset_manifest", candidate_required=False)
    anchor_matches = [meta for meta in assets.values()
                      if isinstance(meta, Mapping)
                      and meta.get("kind") == "executable_anchor_manifest"]
    if len(anchor_matches) > 1:
        raise ValueError("registry has multiple executable_anchor_manifest assets")
    anchor_manifest = anchor_matches[0] if anchor_matches else None
    return {
        "checkpoint_path": _resolve_registry_asset_path(registry_path, str(checkpoint["path"])),
        "checkpoint_sha256": str(checkpoint["sha256"]),
        "student_profile_path": (_resolve_registry_asset_path(registry_path, str(profile["path"]))
                                  if profile else ""),
        "student_profile_file_sha256": (
            str(profile["sha256"]) if profile else ""),
        "task_asset_manifest_path": _resolve_registry_asset_path(registry_path, str(task_manifest["path"])),
        "task_asset_manifest_sha256": str(task_manifest["sha256"]),
        "executable_anchor_manifest_path": (_resolve_registry_asset_path(registry_path, str(anchor_manifest["path"]))
                                             if anchor_manifest else ""),
        "executable_anchor_manifest_sha256": str(anchor_manifest["sha256"]) if anchor_manifest else "",
    }


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
    executable_anchor_manifest_sha256: str | None = None
    preseed_journal_sha256: str | None = None
    continuation_manifest_sha256: str | None = None
    budget_candidate: str | None = None
    budget_sessions: int | None = None
    budget_updates_per_session: int | None = None
    budget_env_steps_per_update: int | None = None
    budget_resolved_env_steps: int | None = None
    budget_semantics: str | None = None
    provider: str | None = None
    requested_model: str | None = None
    max_logical_calls: int | None = None
    max_output_tokens_per_call: int | None = None
    max_total_tokens_per_call: int | None = None
    retry_cap: int | None = None
    client_factory_implementation_hash: str | None = None
    candidate: str | None = None
    sessions: int | None = None
    updates_per_session: int | None = None
    env_steps_per_update: int | None = None
    resolved_env_steps: int | None = None
    expected_physical_gpu_uuid: str | None = None

    @property
    def canonical(self) -> str:
        # Keep legacy verification manifests byte-compatible: newly added
        # formal binding fields are omitted when absent, while any present
        # field is cryptographically bound.
        return canonical_payload({k: v for k, v in self.__dict__.items()
                                  if v is not None})

    def formal_binding(self) -> dict[str, Any]:
        names = ("budget_candidate", "budget_sessions", "budget_updates_per_session",
                 "budget_env_steps_per_update", "budget_resolved_env_steps",
                 "budget_semantics", "provider", "requested_model",
                 "max_logical_calls", "max_output_tokens_per_call",
                 "max_total_tokens_per_call", "retry_cap",
                 "client_factory_implementation_hash", "candidate", "sessions",
                 "updates_per_session", "env_steps_per_update", "resolved_env_steps",
                 "executable_anchor_manifest_sha256", "expected_physical_gpu_uuid",
                 "preseed_journal_sha256")
        out = {name: getattr(self, name) for name in names}
        if any(value is None for value in out.values()):
            raise ValueError("formal authorization budget/LLM binding is incomplete")
        return out

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
        budget_candidate=payload.get("budget_candidate"),
        budget_sessions=payload.get("budget_sessions"),
        budget_updates_per_session=payload.get("budget_updates_per_session"),
        budget_env_steps_per_update=payload.get("budget_env_steps_per_update"),
        budget_resolved_env_steps=payload.get("budget_resolved_env_steps"),
        budget_semantics=payload.get("budget_semantics"),
        provider=payload.get("provider"),
        requested_model=payload.get("requested_model"),
        max_logical_calls=payload.get("max_logical_calls"),
        max_output_tokens_per_call=payload.get("max_output_tokens_per_call"),
        max_total_tokens_per_call=payload.get("max_total_tokens_per_call"),
        retry_cap=payload.get("retry_cap"),
        client_factory_implementation_hash=payload.get("client_factory_implementation_hash"),
        candidate=payload.get("candidate"),
        sessions=payload.get("sessions"),
        updates_per_session=payload.get("updates_per_session"),
        env_steps_per_update=payload.get("env_steps_per_update"),
        resolved_env_steps=payload.get("resolved_env_steps"),
        executable_anchor_manifest_sha256=payload.get("executable_anchor_manifest_sha256"),
        preseed_journal_sha256=payload.get("preseed_journal_sha256"),
        continuation_manifest_sha256=payload.get("continuation_manifest_sha256"),
        expected_physical_gpu_uuid=payload.get("expected_physical_gpu_uuid"),
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
                                 registry_path: str,
                                 task_asset_manifest_sha256: str | None = None,
                                 executable_anchor_manifest_sha256: str | None = None) -> dict:
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
    if not task_asset_manifest_sha256:
        raise ValueError(
            "task asset manifest SHA missing from runtime binding "
            "(fail closed)")
    if task_asset_manifest_sha256 != auth.task_asset_manifest_sha256:
        raise ValueError(
            f"task asset manifest SHA mismatch: actual="
            f"{task_asset_manifest_sha256[:16]} signed="
            f"{auth.task_asset_manifest_sha256[:16]} (fail closed)")
    if auth.scope == "formal":
        if not auth.executable_anchor_manifest_sha256:
            raise ValueError("formal authorization missing executable anchor manifest SHA")
        if executable_anchor_manifest_sha256 != auth.executable_anchor_manifest_sha256:
            raise ValueError("executable anchor manifest SHA mismatch (fail closed)")
        if auth.expected_physical_gpu_uuid != "GPU-3c7a2864-755b-7045-b293-6f80e748283f":
            raise ValueError("formal authorization missing GPU1 physical UUID binding")
    verified = verify_registry_assets(registry_path)
    actual_reg_hash = recompute_registry_hash(registry_path)
    if actual_reg_hash != auth.formal_asset_registry_hash:
        raise ValueError(
            f"registry hash drifted at runtime: {actual_reg_hash[:16]} != "
            f"{auth.formal_asset_registry_hash[:16]} (fail closed)")
    return verified


def verify_formal_authorization_budget(auth: E3Authorization, *, candidate: str,
                                       sessions: int, provider: str,
                                       requested_model: str,
                                       client_factory_hash: str) -> E3FormalBudget:
    """Verify the signed canonical formal budget and LLM cost contract."""
    budget = resolve_e3_budget(candidate=candidate, sessions=sessions, scope="formal")
    binding = auth.formal_binding()
    expected = {
        "budget_candidate": budget.candidate,
        "budget_sessions": budget.sessions,
        "budget_updates_per_session": budget.updates_per_session,
        "budget_env_steps_per_update": budget.env_steps_per_update,
        "budget_resolved_env_steps": budget.resolved_env_steps,
        "budget_semantics": budget.budget_semantics,
        "provider": provider,
        "requested_model": requested_model,
        "max_logical_calls": 302,
        "max_output_tokens_per_call": 4096,
        "max_total_tokens_per_call": 20000,
        "retry_cap": 0,
        "client_factory_implementation_hash": client_factory_hash,
        "candidate": budget.candidate,
        "sessions": budget.sessions,
        "updates_per_session": budget.updates_per_session,
        "env_steps_per_update": budget.env_steps_per_update,
        "resolved_env_steps": budget.resolved_env_steps,
        "preseed_journal_sha256": auth.preseed_journal_sha256,
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            raise ValueError(f"formal authorization binding mismatch for {key}: "
                             f"signed={binding.get(key)!r} expected={value!r}")
    return budget


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
