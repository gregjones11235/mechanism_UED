"""Mechanical optimizer-update attestation (CC4 follow-up, P0-13).

Before this contract existed, STEP10 trusted whatever the injected update
callable CLAIMED: it read ``update_count`` and ``grad_norm`` straight out of
the callable's return value.  A callable could report "1 update, grad_norm
0.5" while doing anything at all — the numbers were self-reported and
therefore worthless as evidence.

``OptimizerUpdateAttestation`` closes that gap.  The attestation is MINTED by
the pipeline from evidence it measures ITSELF:

* ``params_sha256_before`` / ``params_sha256_after`` — recomputed by the
  pipeline over the parameter trees, never supplied by the callable;
* ``optimizer_step_after == optimizer_step_before + 1`` — the baseline step
  comes from the loaded full state (``loaded_state["global_step"]``), so the
  increment is mechanical, not claimed;
* ``params_changed`` / ``params_finite_after`` — measured structurally (a
  bit-identical or non-finite result is never attestable);
* ``batch_digest`` — sha256 over the exact batch arrays actually fed to the
  update, computed inside the minter;
* ``attestation_hash`` — mint-only (init=False), recomputed from the fields
  above, so a self-reported or tampered attestation is structurally
  impossible.

Self-reported ``update_count`` / ``grad_norm`` values are NEVER read on the
production path: only the ``"params"`` key of the update output is consumed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from .errors import InvalidEvidenceError, ProductionBlockedError

OPTIMIZER_ATTESTATION_VERSION = "optimizer-update-attestation/v1"

BLOCKED_OPTIMIZER_STEP_BASELINE_UNMEASURED = (
    "BLOCKED_OPTIMIZER_STEP_BASELINE_UNMEASURED")

_BATCH_KEYS = ("observations", "actions", "rewards", "dones")


def _require_sha256(label: str, digest: Any) -> str:
    text = str(digest)
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise ProductionBlockedError(
            f"{label} is not a lowercase sha256 hex digest: {text[:24]!r}… "
            "(fail closed)")
    return text


def _batch_digest(batch: Mapping[str, Any]) -> tuple[str, int]:
    """sha256 binding the attestation to the exact batch arrays (computed,
    never caller-supplied).  Returns (digest, transitions)."""
    import numpy as np
    parts: list[dict[str, Any]] = []
    transitions = None
    for key in _BATCH_KEYS:
        if key not in batch:
            raise ProductionBlockedError(
                f"optimizer update batch is missing {key!r} — the attestation "
                "cannot bind an incomplete batch (fail closed)")
        arr = np.ascontiguousarray(np.asarray(batch[key]))
        if key == "actions":
            transitions = int(arr.reshape(-1).shape[0])
        parts.append({
            "key": key,
            "dtype": str(arr.dtype),
            "shape": [int(x) for x in arr.shape],
            "bytes_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
        })
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest(), int(transitions or 0)


def _params_finite(params_after: Any) -> bool:
    """Structural finiteness sweep over the updated parameter tree."""
    import jax
    import numpy as np
    for leaf in jax.tree_util.tree_leaves(params_after):
        arr = np.asarray(leaf)
        if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
            continue
        if not bool(np.isfinite(arr.astype(np.float64)).all()):
            return False
    return True


@dataclass(frozen=True)
class OptimizerUpdateAttestation:
    """One immutable, mechanically derived update attestation (mint-only).

    ``attestation_hash`` is NOT a constructor argument: it is computed in
    ``__post_init__`` from the measured fields only.  Structural invariants
    (step increment, params changed, params finite) are enforced HERE, so no
    attestation for an invalid update can even be constructed.
    """

    params_sha256_before: str
    params_sha256_after: str
    optimizer_step_before: int
    optimizer_step_after: int
    batch_digest: str
    transitions: int
    params_changed: bool
    params_finite_after: bool
    attestation_hash: str = field(init=False)
    attestation_version: str = OPTIMIZER_ATTESTATION_VERSION

    def __post_init__(self) -> None:
        for label, digest in (("params_sha256_before", self.params_sha256_before),
                              ("params_sha256_after", self.params_sha256_after),
                              ("batch_digest", self.batch_digest)):
            text = str(digest)
            if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
                raise InvalidEvidenceError(
                    f"OptimizerUpdateAttestation.{label} is not a lowercase "
                    f"sha256 hex digest: {text[:24]!r}…")
        for label, value in (("optimizer_step_before", self.optimizer_step_before),
                             ("optimizer_step_after", self.optimizer_step_after),
                             ("transitions", self.transitions)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InvalidEvidenceError(
                    f"OptimizerUpdateAttestation.{label} must be a non-negative "
                    f"int, got {value!r}")
        if self.optimizer_step_after != self.optimizer_step_before + 1:
            raise InvalidEvidenceError(
                "OptimizerUpdateAttestation invariant violated: optimizer_step_after "
                f"must equal optimizer_step_before + 1, got "
                f"{self.optimizer_step_after} != {self.optimizer_step_before} + 1")
        if self.transitions == 0:
            raise InvalidEvidenceError(
                "OptimizerUpdateAttestation.transitions must be positive — an "
                "update on an empty batch is never attested")
        if not bool(self.params_changed):
            raise InvalidEvidenceError(
                "OptimizerUpdateAttestation invariant violated: params_changed must "
                "be True (a bit-identical 'update' is never attested)")
        if not bool(self.params_finite_after):
            raise InvalidEvidenceError(
                "OptimizerUpdateAttestation invariant violated: params_finite_after "
                "must be True (non-finite params are never attested)")
        payload = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "attestation_hash"
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, default=str)
        object.__setattr__(
            self, "attestation_hash",
            hashlib.sha256(blob.encode("utf-8")).hexdigest())


def mint_optimizer_update_attestation(*, params_sha256_before: Any,
                                      params_sha256_after: Any,
                                      params_after: Any,
                                      optimizer_step_before: Any,
                                      batch: Any) -> OptimizerUpdateAttestation:
    """Mint the attestation from PIPELINE-MEASURED evidence only.

    Every input is validated fail-closed; the batch digest is computed here
    (never accepted from the caller), and an update that left the params
    bit-identical, non-finite or with an unmeasurable step baseline is never
    attestable.
    """
    before = _require_sha256("params_sha256_before", params_sha256_before)
    after = _require_sha256("params_sha256_after", params_sha256_after)
    if isinstance(optimizer_step_before, bool) \
            or not isinstance(optimizer_step_before, int) \
            or optimizer_step_before < 0:
        raise ProductionBlockedError(
            f"{BLOCKED_OPTIMIZER_STEP_BASELINE_UNMEASURED}: optimizer_step_before "
            f"must be a non-negative int, got {optimizer_step_before!r} (the step "
            "increment is never self-reported)")
    if not isinstance(batch, Mapping):
        raise ProductionBlockedError(
            f"optimizer update batch must be a mapping, got "
            f"{type(batch).__name__} (fail closed)")
    if params_after is None:
        raise ProductionBlockedError(
            "optimizer update returned no params tree — nothing to attest "
            "(fail closed)")
    if before == after:
        raise ProductionBlockedError(
            "optimizer update left params bit-identical (no real update "
            "happened; a no-op is never attested)")
    if not _params_finite(params_after):
        raise ProductionBlockedError(
            "optimizer update produced non-finite params (NaN/Inf) — never "
            "attested")
    digest, transitions = _batch_digest(batch)
    if transitions <= 0:
        raise ProductionBlockedError(
            "optimizer update batch carries zero transitions — an empty batch "
            "is never attested (fail closed)")
    return OptimizerUpdateAttestation(
        params_sha256_before=before,
        params_sha256_after=after,
        optimizer_step_before=int(optimizer_step_before),
        optimizer_step_after=int(optimizer_step_before) + 1,
        batch_digest=digest,
        transitions=transitions,
        params_changed=True,
        params_finite_after=True,
    )


def verify_optimizer_update_attestation(attestation: Any) -> None:
    """Recompute the attestation hash + invariants; reject fakes and tamper."""
    if isinstance(attestation, Mapping):
        raise InvalidEvidenceError(
            "verify_optimizer_update_attestation requires a minted "
            "OptimizerUpdateAttestation, not a mapping")
    if not isinstance(attestation, OptimizerUpdateAttestation):
        raise InvalidEvidenceError(
            f"verify_optimizer_update_attestation requires a minted "
            f"OptimizerUpdateAttestation, got {type(attestation).__name__}")
    if attestation.optimizer_step_after != attestation.optimizer_step_before + 1:
        raise InvalidEvidenceError(
            "attestation invariant violated: optimizer_step_after != "
            "optimizer_step_before + 1")
    if not (attestation.params_changed and attestation.params_finite_after):
        raise InvalidEvidenceError(
            "attestation invariant violated: an attestation must record a real, "
            "finite parameter change")
    if attestation.params_sha256_before == attestation.params_sha256_after:
        raise InvalidEvidenceError(
            "attestation invariant violated: params unchanged (before == after)")
    payload = {
        f.name: getattr(attestation, f.name)
        for f in fields(attestation)
        if f.name != "attestation_hash"
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    expected = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    if expected != attestation.attestation_hash:
        raise InvalidEvidenceError(
            "attestation_hash mismatch: the OptimizerUpdateAttestation was "
            "tampered with or self-reported (fail closed)")


def attestation_fields(attestation: Any) -> dict[str, Any]:
    """JSON-safe copy of every attestation field (for pipeline records)."""
    if not isinstance(attestation, OptimizerUpdateAttestation):
        raise InvalidEvidenceError(
            f"attestation_fields requires a minted OptimizerUpdateAttestation, "
            f"got {type(attestation).__name__}")
    return {f.name: getattr(attestation, f.name) for f in fields(attestation)}
