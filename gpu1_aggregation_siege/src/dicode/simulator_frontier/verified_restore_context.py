"""Mint-only verified restore context binding fresh-process evidence to search.

Closes the CC4 audit finding (2026-08-04, P0-2) that ``run_actual_n`` accepted
a PLAIN MAPPING restore context that could self-report
``production_joint_pass: True`` — any caller could forge the dict and start a
"production" branch search from main-process state.

This module replaces that surface with an immutable, MINT-ONLY
``VerifiedRestoreContext``:

* the constructor is NOT a public minting surface: ``production_joint_pass``
  and ``context_hash`` are ``init=False`` fields — they cannot be supplied by
  any caller.  They are set exactly once, inside
  ``mint_verified_restore_context``, AFTER the mechanical gates
  (``verify_fresh_process_evidence`` already green inside the driver outcome
  and ``production_joint_pass`` recomputed on the supplied verdict/evidence)
  pass.  A context therefore cannot exist with a self-reported joint pass.
* the context carries the full evidence binding required downstream:
  restore-request hash, ProductionRegistryBundle hash, controller signature
  reference, real child PID/PPID, process-evidence hash, verdict hash, the
  next-policy-step replay digest, the nine component leaf digests
  (params/optimizer/global_step/train_rng/env_state/env_rng/wrapper_state/
  policy_memory/history), Student identity hash, checkpoint manifest hash,
  anchor manifest hash, formal asset registry hash, captured state id/hash,
  archive hash, schema version, verifier id/hash and the capture entry's
  checkpoint id / memory spec hash.
* ``verify_verified_restore_context`` re-derives the context hash from the
  fields and rejects plain Mappings, wrong schemas, fixture-labelled evidence
  and non-green joint passes.  ``branch_search_runner.run_actual_n`` accepts
  ONLY objects that pass this verification and additionally re-binds the
  context to the run: state id/hash, checkpoint id, memory spec hash and the
  search Student identity must all agree.

The formal asset registry hash is read INTERNALLY from the controller
injection slot (same discipline as the production archive write path) — the
minter accepts no caller-supplied registry.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from .combined_restore_contract import (
    CROSS_CHECKS,
    REQUIRED_COMPONENTS,
    CombinedRestoreVerdict,
)
from .discovery_provenance import (
    REGISTRY_USAGE_PRODUCTION,
    production_registry,
)
from .errors import InvalidEvidenceError
from .fresh_process_restore import (
    EVIDENCE_SCHEMA,
    SYNTHETIC_SIGNATURE_PREFIX,
    FreshProcessRestoreOutcome,
    FreshProcessRestoreRequest,
    ProcessEvidence,
    production_joint_pass,
)

VERIFIED_RESTORE_CONTEXT_SCHEMA = "simulator_frontier.verified_restore_context/v1"

# The ONLY restore driver whose evidence may back a production context.
RESTORE_CONTEXT_DRIVER = "fresh_process_restore.run_fresh_process_restore_production"

VERIFIER_ID = (
    "simulator_frontier.fresh_process_evidence_verifier/v1 "
    "(verify_fresh_process_evidence + production_joint_pass)")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_VERIFIER_DESCRIPTOR = {
    "verifier_id": VERIFIER_ID,
    "evidence_schema": EVIDENCE_SCHEMA,
    "joint_gate": "production_joint_pass/v1",
    "context_schema": VERIFIED_RESTORE_CONTEXT_SCHEMA,
}
VERIFIER_HASH = hashlib.sha256(
    json.dumps(_VERIFIER_DESCRIPTOR, sort_keys=True, separators=(",", ":"),
               ensure_ascii=False).encode("utf-8")).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or not _SHA256_RE.match(value):
        raise InvalidEvidenceError(f"{name} must be a 64-hex sha256, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Deterministic hashes of the underlying evidence objects.
# ---------------------------------------------------------------------------

def restore_request_hash_of(request: FreshProcessRestoreRequest) -> str:
    """Canonical hash of the immutable fresh-process restore request payload."""
    if not isinstance(request, FreshProcessRestoreRequest):
        raise InvalidEvidenceError(
            "restore_request_hash_of requires FreshProcessRestoreRequest")
    return _canonical_sha256(request.to_payload())


def process_evidence_hash_of(evidence: ProcessEvidence) -> str:
    """Canonical hash over the mechanically verified process evidence."""
    if not isinstance(evidence, ProcessEvidence):
        raise InvalidEvidenceError(
            "process_evidence_hash_of requires mechanically verified ProcessEvidence")
    projection = {
        "schema": evidence.schema,
        "fixture_label": evidence.fixture_label,
        "child_pid": int(evidence.child_pid),
        "parent_pid": int(evidence.parent_pid),
        "child_argv": [str(part) for part in evidence.child_argv],
        "started_at": evidence.started_at,
        "ended_at": evidence.ended_at,
        "exit_code": int(evidence.exit_code),
        "worker_module": evidence.worker_module,
        "components": [
            {"component": comp.component, "status": comp.status,
             "origin": comp.origin, "source_path": comp.source_path,
             "pid": int(comp.pid), "leaf_count": int(comp.leaf_count),
             "leaves_digest": comp.leaves_digest}
            for comp in evidence.components],
        "cross_checks": [
            {"name": row.name, "status": row.status, "digest": row.digest,
             "pid": int(row.pid)}
            for row in evidence.cross_checks],
        "error": evidence.error,
    }
    return _canonical_sha256(projection)


def verdict_hash_of(verdict: CombinedRestoreVerdict) -> str:
    """Canonical hash of a CombinedRestoreVerdict (statuses + bound digests)."""
    if not isinstance(verdict, CombinedRestoreVerdict):
        raise InvalidEvidenceError("verdict_hash_of requires CombinedRestoreVerdict")
    projection = {
        "components": [
            {"component": name, "status": result.status.value,
             "bound_digest": result.bound_digest}
            for name, result in sorted(verdict.components.items())],
        "cross_checks": [
            {"check": name, "status": result.status.value,
             "bound_digest": result.bound_digest}
            for name, result in sorted(verdict.cross_checks.items())],
        "combined_pass": bool(verdict.combined_pass),
        "env_only_pass": bool(verdict.env_only_pass),
        "checkpoint_only_pass": bool(verdict.checkpoint_only_pass),
    }
    return _canonical_sha256(projection)


# ---------------------------------------------------------------------------
# The immutable context.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class VerifiedRestoreContext:
    """Immutable proof that ONE verified fresh-process joint restore happened.

    ``production_joint_pass`` and ``context_hash`` are ``init=False``: no
    caller can construct a context with a self-reported joint pass — the only
    way to obtain a context is ``mint_verified_restore_context``, which
    recomputes the joint pass from mechanically verified evidence.
    """

    schema_version: str
    restore_driver: str
    restore_request_hash: str
    registry_bundle_hash: str
    controller_signature_ref: str
    child_pid: int
    child_ppid: int
    process_evidence_hash: str
    verdict_hash: str
    next_policy_step_replay_digest: str
    component_digests: Mapping[str, str]
    student_identity_hash: str
    checkpoint_manifest_hash: str
    anchor_manifest_hash: str
    formal_asset_registry_hash: str
    state_id: str
    state_hash: str
    archive_hash: str
    source_checkpoint_id: str
    source_memory_spec_hash: str
    verifier_id: str
    verifier_hash: str
    production_joint_pass: bool = field(init=False)
    context_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != VERIFIED_RESTORE_CONTEXT_SCHEMA:
            raise InvalidEvidenceError(
                f"VerifiedRestoreContext schema must be {VERIFIED_RESTORE_CONTEXT_SCHEMA}, "
                f"got {self.schema_version!r}")
        if self.restore_driver != RESTORE_CONTEXT_DRIVER:
            raise InvalidEvidenceError(
                f"restore_driver must be {RESTORE_CONTEXT_DRIVER}, got "
                f"{self.restore_driver!r} (no other driver is production evidence)")
        for name in ("restore_request_hash", "registry_bundle_hash",
                     "process_evidence_hash", "verdict_hash",
                     "next_policy_step_replay_digest", "student_identity_hash",
                     "checkpoint_manifest_hash", "anchor_manifest_hash",
                     "formal_asset_registry_hash", "state_hash", "archive_hash",
                     "source_memory_spec_hash", "verifier_hash"):
            _require_sha256(f"VerifiedRestoreContext.{name}", getattr(self, name))
        if not isinstance(self.controller_signature_ref, str) \
                or not self.controller_signature_ref.strip():
            raise InvalidEvidenceError(
                "VerifiedRestoreContext.controller_signature_ref is required")
        if self.controller_signature_ref.startswith(SYNTHETIC_SIGNATURE_PREFIX):
            raise InvalidEvidenceError(
                "synthetic controller signatures can never back a verified restore "
                f"context ({self.controller_signature_ref!r})")
        for name in ("child_pid", "child_ppid"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise InvalidEvidenceError(
                    f"VerifiedRestoreContext.{name} must be a positive int, got {value!r}")
        digests = dict(self.component_digests or {})
        if set(digests) != set(REQUIRED_COMPONENTS):
            raise InvalidEvidenceError(
                "component_digests must cover EXACTLY the nine required components "
                f"{tuple(REQUIRED_COMPONENTS)}; got {sorted(digests)}")
        for comp, digest in digests.items():
            _require_sha256(f"component digest {comp!r}", digest)
        object.__setattr__(self, "component_digests", dict(sorted(digests.items())))
        if not isinstance(self.state_id, str) or not self.state_id.strip():
            raise InvalidEvidenceError("VerifiedRestoreContext.state_id is required")
        if not isinstance(self.source_checkpoint_id, str) \
                or not self.source_checkpoint_id.strip():
            raise InvalidEvidenceError(
                "VerifiedRestoreContext.source_checkpoint_id is required")
        if self.verifier_id != VERIFIER_ID or self.verifier_hash != VERIFIER_HASH:
            raise InvalidEvidenceError(
                "VerifiedRestoreContext verifier id/hash do not match the frozen "
                f"verifier descriptor ({VERIFIER_ID})")

    def to_payload(self) -> dict:
        """Full serializable projection (reports/step records)."""
        return {
            "schema_version": self.schema_version,
            "restore_driver": self.restore_driver,
            "restore_request_hash": self.restore_request_hash,
            "registry_bundle_hash": self.registry_bundle_hash,
            "controller_signature_ref": self.controller_signature_ref,
            "child_pid": int(self.child_pid),
            "child_ppid": int(self.child_ppid),
            "process_evidence_hash": self.process_evidence_hash,
            "verdict_hash": self.verdict_hash,
            "next_policy_step_replay_digest": self.next_policy_step_replay_digest,
            "component_digests": dict(self.component_digests),
            "student_identity_hash": self.student_identity_hash,
            "checkpoint_manifest_hash": self.checkpoint_manifest_hash,
            "anchor_manifest_hash": self.anchor_manifest_hash,
            "formal_asset_registry_hash": self.formal_asset_registry_hash,
            "state_id": self.state_id,
            "state_hash": self.state_hash,
            "archive_hash": self.archive_hash,
            "source_checkpoint_id": self.source_checkpoint_id,
            "source_memory_spec_hash": self.source_memory_spec_hash,
            "verifier_id": self.verifier_id,
            "verifier_hash": self.verifier_hash,
            "production_joint_pass": bool(self.production_joint_pass),
            "context_hash": self.context_hash,
        }


def compute_context_hash(context: VerifiedRestoreContext) -> str:
    """Canonical hash over every field EXCEPT context_hash itself.

    The payload is built from the dataclass fields directly (NOT via
    ``to_payload``): ``to_payload`` reads ``context_hash``, which does not
    exist yet while the minter is computing it — hashing the fields
    themselves keeps the mint path functional and yields the same digest for
    an already-minted context (``to_payload`` excludes ``context_hash``
    anyway).
    """
    if not isinstance(context, VerifiedRestoreContext):
        raise InvalidEvidenceError("compute_context_hash requires VerifiedRestoreContext")
    payload = {
        f.name: getattr(context, f.name)
        for f in fields(context)
        if f.name != "context_hash"
    }
    payload["component_digests"] = dict(payload["component_digests"])
    return _canonical_sha256(payload)


# ---------------------------------------------------------------------------
# The ONLY minting surface.
# ---------------------------------------------------------------------------

def mint_verified_restore_context(*,
                                  restore_request: FreshProcessRestoreRequest,
                                  outcome: FreshProcessRestoreOutcome,
                                  verdict: CombinedRestoreVerdict,
                                  student_identity_hash: str,
                                  anchor_manifest_hash: str,
                                  state_id: str,
                                  state_hash: str,
                                  archive_hash: str,
                                  source_checkpoint_id: str,
                                  source_memory_spec_hash: str
                                  ) -> VerifiedRestoreContext:
    """Mint the immutable context AFTER the mechanical gates pass (fail closed).

    Requirements (all raise on violation — a context is never minted from
    self-reported state):

    * ``outcome`` is an ACCEPTED ``FreshProcessRestoreOutcome`` whose evidence
      is non-fixture ``ProcessEvidence`` (synthetic evidence can never back a
      production context);
    * ``verdict`` is a ``CombinedRestoreVerdict`` and
      ``production_joint_pass(verdict, evidence)`` recomputed HERE is True;
    * the evidence child PID equals the outcome child PID;
    * the formal asset registry hash is read from the controller injection
      slot (usage PRODUCTION) — never caller-supplied.
    """
    if not isinstance(restore_request, FreshProcessRestoreRequest):
        raise InvalidEvidenceError(
            "mint_verified_restore_context requires FreshProcessRestoreRequest")
    if not isinstance(outcome, FreshProcessRestoreOutcome):
        raise InvalidEvidenceError(
            "mint_verified_restore_context requires FreshProcessRestoreOutcome")
    if not outcome.accepted or outcome.evidence is None:
        raise InvalidEvidenceError(
            "cannot mint a verified restore context from a rejected/missing "
            "fresh-process outcome (fail closed)")
    evidence = outcome.evidence
    if not isinstance(evidence, ProcessEvidence):
        raise InvalidEvidenceError(
            "outcome evidence must be mechanically verified ProcessEvidence")
    if evidence.fixture_label:
        raise InvalidEvidenceError(
            "synthetic fixture evidence can never mint a production restore "
            f"context (fixture_label={evidence.fixture_label!r})")
    if not isinstance(verdict, CombinedRestoreVerdict):
        raise InvalidEvidenceError(
            "mint_verified_restore_context requires CombinedRestoreVerdict")

    # THE gate: recompute the joint pass; never trust a supplied boolean.
    if not production_joint_pass(verdict, evidence):
        raise InvalidEvidenceError(
            "production_joint_pass is not green for the supplied verdict/evidence; "
            "no verified restore context can be minted (fail closed)")
    if outcome.child_pid is None or int(outcome.child_pid) != int(evidence.child_pid):
        raise InvalidEvidenceError(
            "outcome child_pid does not equal the evidence child_pid (split-process "
            "composition rejected)")

    # Formal asset registry hash: injection slot only (P0-1 discipline).
    registry = production_registry()
    if registry.usage != REGISTRY_USAGE_PRODUCTION:
        raise InvalidEvidenceError(
            "verified restore context minting rejects a TEST_ONLY registry in the "
            "production slot (fail closed)")

    signature = restore_request.registry_bundle.controller_signature_ref
    if signature.startswith(SYNTHETIC_SIGNATURE_PREFIX):
        raise InvalidEvidenceError(
            f"synthetic controller signatures can never mint a production context "
            f"({signature!r})")

    replay_rows = [row for row in evidence.cross_checks if row.name == CROSS_CHECKS[0]]
    if not replay_rows:
        raise InvalidEvidenceError(
            f"evidence carries no {CROSS_CHECKS[0]} cross-check (fail closed)")
    component_digests = {comp.component: comp.leaves_digest
                         for comp in evidence.components}
    if set(component_digests) != set(REQUIRED_COMPONENTS):
        raise InvalidEvidenceError(
            "evidence must cover EXACTLY the nine required components; got "
            f"{sorted(component_digests)}")

    context = VerifiedRestoreContext(
        schema_version=VERIFIED_RESTORE_CONTEXT_SCHEMA,
        restore_driver=RESTORE_CONTEXT_DRIVER,
        restore_request_hash=restore_request_hash_of(restore_request),
        registry_bundle_hash=restore_request.registry_bundle.bundle_sha256(),
        controller_signature_ref=signature,
        child_pid=int(evidence.child_pid),
        child_ppid=int(evidence.parent_pid),
        process_evidence_hash=process_evidence_hash_of(evidence),
        verdict_hash=verdict_hash_of(verdict),
        next_policy_step_replay_digest=replay_rows[0].digest,
        component_digests=component_digests,
        student_identity_hash=_require_sha256("student_identity_hash", student_identity_hash),
        checkpoint_manifest_hash=_require_sha256(
            "checkpoint_manifest_hash", restore_request.manifest_hash),
        anchor_manifest_hash=_require_sha256("anchor_manifest_hash", anchor_manifest_hash),
        formal_asset_registry_hash=registry.registry_hash,
        state_id=state_id,
        state_hash=_require_sha256("state_hash", state_hash),
        archive_hash=_require_sha256("archive_hash", archive_hash),
        source_checkpoint_id=source_checkpoint_id,
        source_memory_spec_hash=_require_sha256(
            "source_memory_spec_hash", source_memory_spec_hash),
        verifier_id=VERIFIER_ID,
        verifier_hash=VERIFIER_HASH,
    )
    # Mint-only fields: set exactly once, here, after every gate passed.
    object.__setattr__(context, "production_joint_pass", True)
    object.__setattr__(context, "context_hash", compute_context_hash(context))
    return context


def verify_verified_restore_context(context: Any) -> VerifiedRestoreContext:
    """Fail-closed verification of a context before any production consumption.

    Rejects plain Mappings (self-reported contexts), any non-context object,
    tampered field sets (context hash recomputation) and contexts whose joint
    pass flag is not True.
    """
    if isinstance(context, Mapping):
        raise InvalidEvidenceError(
            "plain Mapping restore contexts are rejected: a verified restore context "
            "must be minted from mechanically verified fresh-process evidence "
            "(self-reported production_joint_pass is never accepted)")
    if not isinstance(context, VerifiedRestoreContext):
        raise InvalidEvidenceError(
            f"expected a minted VerifiedRestoreContext, got {type(context).__name__}")
    if not bool(context.production_joint_pass):
        raise InvalidEvidenceError(
            "verified restore context production_joint_pass is not True (fail closed)")
    recomputed = compute_context_hash(context)
    if context.context_hash != recomputed:
        raise InvalidEvidenceError(
            "verified restore context hash mismatch: fields were altered after "
            f"minting (carries {str(context.context_hash)[:16]}…, recomputes to "
            f"{recomputed[:16]}…)")
    return context
