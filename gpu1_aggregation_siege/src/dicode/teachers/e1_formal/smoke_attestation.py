"""CC2 follow-up P0-13: the signed E1 real-smoke attestation.

Readiness used to trust a plain local JSON report (forgeable). The
single source of readiness truth is now the signed
``E1RealSmokeAttestation``::

    attestation = issue_e1_real_smoke_attestation(run_id=..., ...)
    verify_e1_real_smoke_attestation(attested, expected={...}, ctx)

It binds the run id, branch, git SHA, runtime bundle hash, Student /
Reference identity + checkpoint hashes, the Board journal hash, the
EnvCoder artifact pool hash, the probe pool hash, the selection
attestation hash, the 12+4 verified batch hash, the update
attestation hash, the round-trip attestation hash, the formal asset
registry hash, the anchor manifest hash, the status, and the
signer/verifier identity — all folded into one attestation hash.

Discipline:

* ``status`` must be EXECUTED (a smoke attestation never certifies a
  non-executed run);
* verification RE-VERIFIES every bound hash against the expected
  live values (a claimed hash that does not match the current code /
  runtime state fails closed);
* plain JSON reports are human-readable ONLY: without a valid signed
  attestation inside them they are parse-level evidence at best and
  never grant readiness;
* the production signer whitelist is EMPTY this round, so only the
  conspicuously-marked TEST_ONLY surface can mint (contract tests);
  production readiness stays fail-closed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from .canonical import canonical_sha256
from .schemas import E1SchemaError

#: the only status a smoke attestation may carry
SMOKE_STATUS_EXECUTED = "EXECUTED"

#: synthetic TEST_ONLY signer (greppable)
SYNTHETIC_TEST_ONLY_SMOKE_SIGNER = "SYNTHETIC_TEST_ONLY_SMOKE_SIGNER"

#: supervisor-owned production signer whitelist — EMPTY this round
AUTHORIZED_SMOKE_SIGNERS: Tuple[str, ...] = ()

#: binding version
SMOKE_ATTESTATION_VERSION = "e1-real-smoke-attestation-v1"

#: every bound hash field (re-verified against expected live values)
SMOKE_BOUND_HASH_FIELDS = (
    "runtime_bundle_hash",
    "student_identity_hash",
    "student_checkpoint_hash",
    "reference_identity_hash",
    "reference_checkpoint_hash",
    "board_journal_hash",
    "envcoder_artifact_pool_hash",
    "probe_pool_hash",
    "selection_attestation_hash",
    "verified_batch_hash",
    "update_attestation_hash",
    "roundtrip_attestation_hash",
    "formal_asset_registry_hash",
    "anchor_manifest_hash",
)

# fail-closed codes (greppable)
SMOKE_BAD_TYPE = "SMOKE_BAD_TYPE"
SMOKE_SIGNER_UNAUTHORIZED = "SMOKE_SIGNER_UNAUTHORIZED"
SMOKE_TEST_ONLY_REJECTED = "SMOKE_TEST_ONLY_REJECTED"
SMOKE_STATUS = "SMOKE_STATUS"
SMOKE_HASH_MISMATCH = "SMOKE_HASH_MISMATCH"
SMOKE_BOUND_HASH_MISMATCH = "SMOKE_BOUND_HASH_MISMATCH"


class SmokeAttestationError(E1SchemaError):
    """Fail-closed smoke-attestation violation; ``code`` is
    greppable."""


def _require_sha64(value: Any, name: str, ctx: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise SmokeAttestationError(
            SMOKE_BAD_TYPE,
            f"{ctx}: {name} must be a 64-hex hash, got {value!r}",
        )
    return value


def _require_non_empty_str(value: Any, name: str, ctx: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SmokeAttestationError(
            SMOKE_BAD_TYPE,
            f"{ctx}: {name} must be a non-empty str, got {value!r}",
        )
    return value.strip()


@dataclass(frozen=True)
class E1RealSmokeAttestation:
    """The signed record of one REAL executed one-window smoke."""

    run_id: str
    branch: str
    git_sha: str
    runtime_bundle_hash: str
    student_identity_hash: str
    student_checkpoint_hash: str
    reference_identity_hash: str
    reference_checkpoint_hash: str
    board_journal_hash: str
    envcoder_artifact_pool_hash: str
    probe_pool_hash: str
    selection_attestation_hash: str
    verified_batch_hash: str
    update_attestation_hash: str
    roundtrip_attestation_hash: str
    formal_asset_registry_hash: str
    anchor_manifest_hash: str
    status: str
    signer_id: str
    verifier_id: str
    attestation_hash: str
    test_only: bool


def compute_smoke_attestation_hash(
    *,
    run_id: str,
    branch: str,
    git_sha: str,
    runtime_bundle_hash: str,
    student_identity_hash: str,
    student_checkpoint_hash: str,
    reference_identity_hash: str,
    reference_checkpoint_hash: str,
    board_journal_hash: str,
    envcoder_artifact_pool_hash: str,
    probe_pool_hash: str,
    selection_attestation_hash: str,
    verified_batch_hash: str,
    update_attestation_hash: str,
    roundtrip_attestation_hash: str,
    formal_asset_registry_hash: str,
    anchor_manifest_hash: str,
    status: str,
    signer_id: str,
    verifier_id: str,
    test_only: bool,
) -> str:
    return canonical_sha256(
        {
            "smoke_version": SMOKE_ATTESTATION_VERSION,
            "run_id": run_id,
            "branch": branch,
            "git_sha": git_sha,
            "runtime_bundle_hash": runtime_bundle_hash,
            "student_identity_hash": student_identity_hash,
            "student_checkpoint_hash": student_checkpoint_hash,
            "reference_identity_hash": reference_identity_hash,
            "reference_checkpoint_hash": reference_checkpoint_hash,
            "board_journal_hash": board_journal_hash,
            "envcoder_artifact_pool_hash": envcoder_artifact_pool_hash,
            "probe_pool_hash": probe_pool_hash,
            "selection_attestation_hash": selection_attestation_hash,
            "verified_batch_hash": verified_batch_hash,
            "update_attestation_hash": update_attestation_hash,
            "roundtrip_attestation_hash": roundtrip_attestation_hash,
            "formal_asset_registry_hash": formal_asset_registry_hash,
            "anchor_manifest_hash": anchor_manifest_hash,
            "status": status,
            "signer_id": signer_id,
            "verifier_id": verifier_id,
            "test_only": test_only,
        }
    )


def issue_e1_real_smoke_attestation(
    *,
    run_id: str,
    branch: str,
    git_sha: str,
    runtime_bundle_hash: str,
    student_identity_hash: str,
    student_checkpoint_hash: str,
    reference_identity_hash: str,
    reference_checkpoint_hash: str,
    board_journal_hash: str,
    envcoder_artifact_pool_hash: str,
    probe_pool_hash: str,
    selection_attestation_hash: str,
    verified_batch_hash: str,
    update_attestation_hash: str,
    roundtrip_attestation_hash: str,
    formal_asset_registry_hash: str,
    anchor_manifest_hash: str,
    status: str,
    signer_id: str,
    test_only: bool = False,
    ctx: str = "smoke_attestation.issue",
) -> E1RealSmokeAttestation:
    """Mint the smoke attestation fail-closed on EVERY field."""
    run_id = _require_non_empty_str(run_id, "run_id", ctx)
    branch = _require_non_empty_str(branch, "branch", ctx)
    git_sha = _require_sha64(git_sha, "git_sha", ctx)
    status = _require_non_empty_str(status, "status", ctx)
    if status != SMOKE_STATUS_EXECUTED:
        raise SmokeAttestationError(
            SMOKE_STATUS,
            f"{ctx}: a smoke attestation may only certify "
            f"{SMOKE_STATUS_EXECUTED!r}, got {status!r}",
        )
    hashes = {}
    for field in SMOKE_BOUND_HASH_FIELDS:
        hashes[field] = _require_sha64(locals()[field], field, ctx)
    if test_only:
        if signer_id != SYNTHETIC_TEST_ONLY_SMOKE_SIGNER:
            raise SmokeAttestationError(
                SMOKE_TEST_ONLY_REJECTED,
                f"{ctx}: TEST_ONLY smoke attestations must be signed by "
                f"{SYNTHETIC_TEST_ONLY_SMOKE_SIGNER!r}, got "
                f"{signer_id!r}",
            )
    elif signer_id not in AUTHORIZED_SMOKE_SIGNERS:
        raise SmokeAttestationError(
            SMOKE_SIGNER_UNAUTHORIZED,
            f"{ctx}: signer {signer_id!r} is not on the supervisor-"
            "owned smoke whitelist (EMPTY this round)",
        )
    verifier_id = canonical_sha256(
        {"verifier": SMOKE_ATTESTATION_VERSION}
    )
    attestation_hash = compute_smoke_attestation_hash(
        run_id=run_id,
        branch=branch,
        git_sha=git_sha,
        runtime_bundle_hash=hashes["runtime_bundle_hash"],
        student_identity_hash=hashes["student_identity_hash"],
        student_checkpoint_hash=hashes["student_checkpoint_hash"],
        reference_identity_hash=hashes["reference_identity_hash"],
        reference_checkpoint_hash=hashes["reference_checkpoint_hash"],
        board_journal_hash=hashes["board_journal_hash"],
        envcoder_artifact_pool_hash=hashes["envcoder_artifact_pool_hash"],
        probe_pool_hash=hashes["probe_pool_hash"],
        selection_attestation_hash=hashes["selection_attestation_hash"],
        verified_batch_hash=hashes["verified_batch_hash"],
        update_attestation_hash=hashes["update_attestation_hash"],
        roundtrip_attestation_hash=hashes["roundtrip_attestation_hash"],
        formal_asset_registry_hash=hashes["formal_asset_registry_hash"],
        anchor_manifest_hash=hashes["anchor_manifest_hash"],
        status=status,
        signer_id=signer_id,
        verifier_id=verifier_id,
        test_only=test_only,
    )
    return E1RealSmokeAttestation(
        run_id=run_id,
        branch=branch,
        git_sha=git_sha,
        runtime_bundle_hash=hashes["runtime_bundle_hash"],
        student_identity_hash=hashes["student_identity_hash"],
        student_checkpoint_hash=hashes["student_checkpoint_hash"],
        reference_identity_hash=hashes["reference_identity_hash"],
        reference_checkpoint_hash=hashes["reference_checkpoint_hash"],
        board_journal_hash=hashes["board_journal_hash"],
        envcoder_artifact_pool_hash=hashes["envcoder_artifact_pool_hash"],
        probe_pool_hash=hashes["probe_pool_hash"],
        selection_attestation_hash=hashes["selection_attestation_hash"],
        verified_batch_hash=hashes["verified_batch_hash"],
        update_attestation_hash=hashes["update_attestation_hash"],
        roundtrip_attestation_hash=hashes["roundtrip_attestation_hash"],
        formal_asset_registry_hash=hashes["formal_asset_registry_hash"],
        anchor_manifest_hash=hashes["anchor_manifest_hash"],
        status=status,
        signer_id=signer_id,
        verifier_id=verifier_id,
        attestation_hash=attestation_hash,
        test_only=test_only,
    )


def consume_smoke_attestation_mapping(
    mapping: Any, ctx: str
) -> E1RealSmokeAttestation:
    """Parse a JSON-embedded attestation + verify its hash + signer.

    The plain JSON around it is human-readable only; this is where
    the root-of-trust verification happens.
    """
    if not isinstance(mapping, Mapping):
        raise SmokeAttestationError(
            SMOKE_BAD_TYPE,
            f"{ctx}: the attestation block must be a mapping, got "
            f"{type(mapping).__name__}",
        )
    expected_fields = {
        "run_id",
        "branch",
        "git_sha",
        "status",
        "signer_id",
        "test_only",
        "attestation_hash",
        "verifier_id",
    } | set(SMOKE_BOUND_HASH_FIELDS)
    unknown = sorted(k for k in mapping if k not in expected_fields)
    if unknown:
        raise SmokeAttestationError(
            SMOKE_BAD_TYPE,
            f"{ctx}: unknown attestation field(s) {unknown}",
        )
    try:
        return issue_e1_real_smoke_attestation(
            run_id=mapping.get("run_id", ""),
            branch=mapping.get("branch", ""),
            git_sha=mapping.get("git_sha", ""),
            runtime_bundle_hash=mapping.get("runtime_bundle_hash", ""),
            student_identity_hash=mapping.get("student_identity_hash", ""),
            student_checkpoint_hash=mapping.get(
                "student_checkpoint_hash", ""
            ),
            reference_identity_hash=mapping.get(
                "reference_identity_hash", ""
            ),
            reference_checkpoint_hash=mapping.get(
                "reference_checkpoint_hash", ""
            ),
            board_journal_hash=mapping.get("board_journal_hash", ""),
            envcoder_artifact_pool_hash=mapping.get(
                "envcoder_artifact_pool_hash", ""
            ),
            probe_pool_hash=mapping.get("probe_pool_hash", ""),
            selection_attestation_hash=mapping.get(
                "selection_attestation_hash", ""
            ),
            verified_batch_hash=mapping.get("verified_batch_hash", ""),
            update_attestation_hash=mapping.get(
                "update_attestation_hash", ""
            ),
            roundtrip_attestation_hash=mapping.get(
                "roundtrip_attestation_hash", ""
            ),
            formal_asset_registry_hash=mapping.get(
                "formal_asset_registry_hash", ""
            ),
            anchor_manifest_hash=mapping.get("anchor_manifest_hash", ""),
            status=mapping.get("status", ""),
            signer_id=mapping.get("signer_id", ""),
            test_only=bool(mapping.get("test_only")),
            ctx=ctx,
        )
    except SmokeAttestationError as e:
        if e.code == SMOKE_TEST_ONLY_REJECTED:
            raise SmokeAttestationError(
                SMOKE_SIGNER_UNAUTHORIZED,
                f"{ctx}: {e}",
            )
        raise
    # hash + signer re-verification
    recomputed = compute_smoke_attestation_hash(
        run_id=(
            mapping["run_id"]
            if isinstance(mapping.get("run_id"), str)
            else ""
        ),
        branch=(
            mapping["branch"]
            if isinstance(mapping.get("branch"), str)
            else ""
        ),
        git_sha=(
            mapping["git_sha"]
            if isinstance(mapping.get("git_sha"), str)
            else ""
        ),
        runtime_bundle_hash=mapping.get("runtime_bundle_hash", ""),
        student_identity_hash=mapping.get("student_identity_hash", ""),
        student_checkpoint_hash=mapping.get(
            "student_checkpoint_hash", ""
        ),
        reference_identity_hash=mapping.get(
            "reference_identity_hash", ""
        ),
        reference_checkpoint_hash=mapping.get(
            "reference_checkpoint_hash", ""
        ),
        board_journal_hash=mapping.get("board_journal_hash", ""),
        envcoder_artifact_pool_hash=mapping.get(
            "envcoder_artifact_pool_hash", ""
        ),
        probe_pool_hash=mapping.get("probe_pool_hash", ""),
        selection_attestation_hash=mapping.get(
            "selection_attestation_hash", ""
        ),
        verified_batch_hash=mapping.get("verified_batch_hash", ""),
        update_attestation_hash=mapping.get(
            "update_attestation_hash", ""
        ),
        roundtrip_attestation_hash=mapping.get(
            "roundtrip_attestation_hash", ""
        ),
        formal_asset_registry_hash=mapping.get(
            "formal_asset_registry_hash", ""
        ),
        anchor_manifest_hash=mapping.get("anchor_manifest_hash", ""),
        status=mapping.get("status", ""),
        signer_id=mapping.get("signer_id", ""),
        verifier_id=mapping.get("verifier_id", ""),
        test_only=bool(mapping.get("test_only")),
    )
    if recomputed != mapping.get("attestation_hash"):
        raise SmokeAttestationError(
            SMOKE_HASH_MISMATCH,
            f"{ctx}: attestation_hash {mapping.get('attestation_hash')!r} "
            f"!= recomputed {recomputed!r} (tampered or stale)",
        )
    return issue_e1_real_smoke_attestation(
        run_id=mapping.get("run_id", ""),
        branch=mapping.get("branch", ""),
        git_sha=mapping.get("git_sha", ""),
        runtime_bundle_hash=mapping.get("runtime_bundle_hash", ""),
        student_identity_hash=mapping.get("student_identity_hash", ""),
        student_checkpoint_hash=mapping.get(
            "student_checkpoint_hash", ""
        ),
        reference_identity_hash=mapping.get(
            "reference_identity_hash", ""
        ),
        reference_checkpoint_hash=mapping.get(
            "reference_checkpoint_hash", ""
        ),
        board_journal_hash=mapping.get("board_journal_hash", ""),
        envcoder_artifact_pool_hash=mapping.get(
            "envcoder_artifact_pool_hash", ""
        ),
        probe_pool_hash=mapping.get("probe_pool_hash", ""),
        selection_attestation_hash=mapping.get(
            "selection_attestation_hash", ""
        ),
        verified_batch_hash=mapping.get("verified_batch_hash", ""),
        update_attestation_hash=mapping.get(
            "update_attestation_hash", ""
        ),
        roundtrip_attestation_hash=mapping.get(
            "roundtrip_attestation_hash", ""
        ),
        formal_asset_registry_hash=mapping.get(
            "formal_asset_registry_hash", ""
        ),
        anchor_manifest_hash=mapping.get("anchor_manifest_hash", ""),
        status=mapping.get("status", ""),
        signer_id=mapping.get("signer_id", ""),
        test_only=bool(mapping.get("test_only")),
        ctx=ctx,
    )


def verify_e1_real_smoke_attestation(
    attested: Any,
    *,
    expected: Mapping[str, str],
    ctx: str = "smoke_attestation.verify",
) -> None:
    """Re-verify the attestation + EVERY bound hash against the live
    expected values (fail-closed on any mismatch)."""
    if not isinstance(attested, E1RealSmokeAttestation):
        raise SmokeAttestationError(
            SMOKE_BAD_TYPE,
            f"{ctx}: expected an E1RealSmokeAttestation, got "
            f"{type(attested).__name__}",
        )
    if not isinstance(expected, Mapping):
        raise SmokeAttestationError(
            SMOKE_BAD_TYPE,
            f"{ctx}: expected must be a mapping of field -> hash, got "
            f"{type(expected).__name__}",
        )
    recomputed = compute_smoke_attestation_hash(
        run_id=attested.run_id,
        branch=attested.branch,
        git_sha=attested.git_sha,
        runtime_bundle_hash=attested.runtime_bundle_hash,
        student_identity_hash=attested.student_identity_hash,
        student_checkpoint_hash=attested.student_checkpoint_hash,
        reference_identity_hash=attested.reference_identity_hash,
        reference_checkpoint_hash=attested.reference_checkpoint_hash,
        board_journal_hash=attested.board_journal_hash,
        envcoder_artifact_pool_hash=attested.envcoder_artifact_pool_hash,
        probe_pool_hash=attested.probe_pool_hash,
        selection_attestation_hash=attested.selection_attestation_hash,
        verified_batch_hash=attested.verified_batch_hash,
        update_attestation_hash=attested.update_attestation_hash,
        roundtrip_attestation_hash=attested.roundtrip_attestation_hash,
        formal_asset_registry_hash=attested.formal_asset_registry_hash,
        anchor_manifest_hash=attested.anchor_manifest_hash,
        status=attested.status,
        signer_id=attested.signer_id,
        verifier_id=attested.verifier_id,
        test_only=attested.test_only,
    )
    if recomputed != attested.attestation_hash:
        raise SmokeAttestationError(
            SMOKE_HASH_MISMATCH,
            f"{ctx}: attestation_hash {attested.attestation_hash!r} != "
            f"recomputed {recomputed!r} (tampered)",
        )
    for field in ("run_id", "branch", "git_sha") + SMOKE_BOUND_HASH_FIELDS:
        actual = getattr(attested, field)
        want = expected.get(field, "")
        if want and actual != want:
            raise SmokeAttestationError(
                SMOKE_BOUND_HASH_MISMATCH,
                f"{ctx}: attested {field} {actual!r} != live expected "
                f"{want!r}",
            )
        if not want and actual:
            raise SmokeAttestationError(
                SMOKE_BOUND_HASH_MISMATCH,
                f"{ctx}: attested {field} {actual!r} has no live "
                "counterpart this round (expected empty); claimed "
                "evidence with no verifiable source fails closed",
            )
