"""Labelled SYNTHETIC loader/replay entry points for the child process.

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT — everything in this module is a
contract-test fixture.  It exists ONLY so the single-fresh-process driver's
bundle-driven child resolution can be exercised end-to-end by labelled
contract tests: the ``ProductionRegistryBundle`` carried by the request names
these functions as explicit entry points, and the CHILD process imports them
fresh from disk through ``importlib`` — never through parent-process state.

The production path can never reach this module:
  * ``run_fresh_process_restore_production`` rejects fixture-labelled
    requests, and unlabelled (production) bundles must not carry a
    ``SYNTHETIC_SIGNATURE_*`` controller signature reference;
  * the child fails closed (``verify_controller_signature``) on any
    unlabelled bundle while no controller verification material is bound.

Loader protocol (child side): ``loader(context) -> component tree`` where
``context`` is a plain mapping with keys ``component``, ``spec``
(path/sha256/expected_leaves_digest), and the request identity fields.
Replay protocol: ``replay(context) -> 64-hex digest`` where ``context``
carries ``component_digests`` plus the identity echo.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .errors import InvalidEvidenceError
from .fresh_process_restore import ARTIFACT_SCHEMA, synthetic_replay_digest


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_synthetic_component(context: Mapping[str, Any]) -> Any:
    """Labelled synthetic loader: restore one component tree from its
    hash-bound artifact file (file hash re-verified here even though the
    worker already verified it — fixture loaders must be self-contained)."""
    spec = context["spec"]
    path = spec["path"]
    try:
        with open(path, "rb") as handle:
            blob = handle.read()
    except OSError as exc:
        raise InvalidEvidenceError(
            f"synthetic fixture loader: cannot read artifact {path!r}: {exc}")
    if _sha256_hex(blob) != spec["sha256"]:
        raise InvalidEvidenceError(
            f"synthetic fixture loader: artifact file sha256 mismatch for {path!r} "
            "(authoritative source hash violated)")
    try:
        payload = json.loads(blob.decode("utf-8"))
    except Exception as exc:
        raise InvalidEvidenceError(
            f"synthetic fixture loader: artifact {path!r} not parseable: {exc}")
    if not isinstance(payload, dict) or payload.get("schema") != ARTIFACT_SCHEMA:
        raise InvalidEvidenceError(
            f"synthetic fixture loader: artifact schema must be {ARTIFACT_SCHEMA}")
    return payload["tree"]


def synthetic_fixture_replay(context: Mapping[str, Any]) -> str:
    """Labelled synthetic next-step replay: a pure function of the JOINTLY
    restored component digests (never a real network replay)."""
    return synthetic_replay_digest(context["component_digests"])
