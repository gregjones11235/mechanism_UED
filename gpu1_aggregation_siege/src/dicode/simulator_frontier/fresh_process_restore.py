"""Single-fresh-process joint restore: production driver + mechanical evidence.

Closes the independent audit design gap (2026-08-04): the contract-level
``combined_restore_contract.run_combined_restore`` executes caller-supplied
callbacks in the CURRENT process — freshness there is only a caller
obligation, ``ComponentResult`` RESTORED statuses and detail strings are
self-asserted, and ``evaluate_verdict`` can compose ``combined_pass=True``
without any child PID / argv / timestamps or authoritative leaf evidence.

This module is the PRODUCTION path.  It mechanically enforces the R4c
requirement instead of documenting it:

  * ``run_fresh_process_restore_production`` spawns EXACTLY ONE new Python
    process (``restore_worker``) which jointly restores all nine required
    components — params, optimizer, global_step, train_rng, env_state,
    env_rng, wrapper_state, policy_memory, history — from authoritative
    artifact files, then replays the next policy step on the JOINTLY
    restored state.
  * The child emits ATOMIC evidence (tmp file + ``os.replace``): real child
    PID/PPID/argv/start/end/exit code, the exact source checkpoint / Student
    ABI / registry / manifest hashes it ran against, and per-component
    authoritative path / treedef / leaf count + order + shape + dtype +
    value hashes.
  * ``verify_fresh_process_evidence`` re-checks everything in the parent:
    the evidence child PID must be a NEW process (parent-process execution
    rejected) and the evidence parent PID must match either the driver PID
    or the PID this call launched (covers the Windows venv launcher, where
    ``Popen.pid`` is an intermediate process; any other parent PID is a
    split/forged chain and is rejected), every component must be
    present with RESTORED statuses from the frozen enum only (self-asserted
    strings rejected), leaf records must be exactly ordered 0..n-1 with a
    recomputed digest matching both the recorded digest and the request
    expectation (missing / reordered / shape / dtype / value tampering
    rejected), the optimizer must carry CHECKPOINT_LEAVES origin and the
    checkpoint digest (tx.init substitution rejected), all components and
    the replay cross-check must come from the SAME child PID (split-process
    composition rejected), and exit code must be 0 with readable, untorn
    evidence (crash / torn report rejected).
  * Synthetic callbacks are rejected in production: the production entry
    point accepts NO callbacks at all (structurally) and refuses
    fixture-labelled requests / synthetic controller signatures.

Audit follow-up (2026-08-04, round 2) — child resolution is bundle-driven:
an earlier design kept ``_PRODUCTION_LOADERS`` / ``_PRODUCTION_REPLAY`` as
PARENT-process globals, but the spawned child imports a fresh module where
those registries are always empty — no parent state survives ``Popen``, so
the parent globals were dead surface that could only be bypassed by hidden
import side effects.  That surface is REMOVED entirely.  The child now
receives ONLY the immutable, controller-signed ``ProductionRegistryBundle``
nested in (and hash-bound to) the request, and resolves every component
loader and the replay ONLY through the bundle's explicit entry points,
imported fresh in the child.  The request is hash-bound to the bundle
(``registry_hash == bundle_sha256``), the bundle is cross-bound to the
Student ABI / manifest hashes, the child RE-VERIFIES all of these bindings
on its side, and the evidence echoes the bundle hash back to the parent.
``production_joint_pass`` additionally refuses any ``CombinedRestoreVerdict``
whose component statuses/bound digests do not correspond to the SAME
verified ``ProcessEvidence`` (canonical builder: ``verdict_from_evidence``).

Honest status (this round): the audited CC2 pkl carries params + manifest
only, and NO controller-signed registry bundle exists yet
(``CONTROLLER_SIGNED_REGISTRY_BUNDLE_BOUND`` is False), so no REAL joint run
can execute yet — ``COMBINED_FRESH_PROCESS_RESTORE_EXECUTED`` stays False
and the only green path is the labelled synthetic subprocess contract test
(SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT), which proves the enforcement
mechanism, not a real restore.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .combined_restore_contract import (
    CROSS_CHECKS,
    REQUIRED_COMPONENTS,
    RESTORED_STATUSES,
    CombinedRestoreVerdict,
    ComponentResult,
    ComponentStatus,
    evaluate_verdict,
)
from .errors import InvalidEvidenceError

EVIDENCE_SCHEMA = "simulator_frontier.fresh_process_restore_evidence/v1"
REQUEST_SCHEMA = "simulator_frontier.fresh_process_restore_request/v1"
ARTIFACT_SCHEMA = "simulator_frontier.restore_artifact/v1"
BUNDLE_SCHEMA = "simulator_frontier.production_registry_bundle/v1"
WORKER_MODULE = "dicode.simulator_frontier.restore_worker"
REPLAY_CHECK = CROSS_CHECKS[0]  # "policy_step_next_replay"

SYNTHETIC_FIXTURE_LABEL = "SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT"
# Controller-signature discipline: labelled synthetic bundles must carry a
# signature reference marked synthetic; an unlabelled (production-intent)
# bundle must NOT carry one, and is admitted only once real controller
# signature verification material is bound (not this round).
SYNTHETIC_SIGNATURE_PREFIX = "SYNTHETIC_SIGNATURE_"
BLOCKED_WAITING_CONTROLLER_SIGNED_REGISTRY_BUNDLE = (
    "BLOCKED_WAITING_CONTROLLER_SIGNED_REGISTRY_BUNDLE")

# Optimizer origin discipline: the optimizer must be restored from the
# checkpoint's own leaves.  A fresh ``tx.init`` re-initialization is a
# substitution attack and is rejected mechanically.
OPTIMIZER_ORIGIN_CHECKPOINT = "CHECKPOINT_LEAVES"
OPTIMIZER_ORIGIN_TX_INIT = "TX_INIT_SUBSTITUTION"
ORIGIN_SOURCE_ARTIFACT = "SOURCE_ARTIFACT"

# ---------------------------------------------------------------------------
# Honest round status: the enforcement mechanism is contract-ready (green via
# the labelled synthetic subprocess test), but NO real artifact run has
# executed -> the combined flag stays false.  The bundle-driven child
# resolution contract is likewise ready, but NO controller-signed registry
# bundle exists yet -> production runs stay blocked and honest.
# ---------------------------------------------------------------------------
FRESH_PROCESS_DRIVER_CONTRACT_READY = True
COMBINED_FRESH_PROCESS_RESTORE_EXECUTED = False
PRODUCTION_REGISTRY_BUNDLE_CONTRACT_READY = True
CONTROLLER_SIGNED_REGISTRY_BUNDLE_BOUND = False

_STATUS_DISCLAIMER = (
    "env-only restore PASS /\\ checkpoint-only restore PASS != combined "
    "fresh-process joint proof; COMBINED_FRESH_PROCESS_RESTORE stays false "
    "until run_fresh_process_restore_production executes on real artifacts "
    "with verified single-child-process evidence"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Request types (fail-closed construction).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComponentArtifactSpec:
    """Authoritative source artifact for one restore component."""

    path: str
    sha256: str
    expected_leaves_digest: str = ""

    def __post_init__(self) -> None:
        if not self.path:
            raise InvalidEvidenceError("ComponentArtifactSpec.path is required")
        if not _SHA256_RE.match(self.sha256):
            raise InvalidEvidenceError(
                f"ComponentArtifactSpec.sha256 must be 64-hex, got {self.sha256!r}")
        if self.expected_leaves_digest and not _SHA256_RE.match(self.expected_leaves_digest):
            raise InvalidEvidenceError(
                "ComponentArtifactSpec.expected_leaves_digest must be 64-hex when present")


@dataclass(frozen=True)
class LoaderEntryPoint:
    """Explicit child-side entry point named by the registry bundle.

    The child imports ``entry_module`` FRESH and resolves ``entry_attr`` on
    it — no parent-process object, global registry or import side effect is
    ever consulted.  Fail closed on anything but a plain importable target.
    """

    entry_module: str
    entry_attr: str

    def __post_init__(self) -> None:
        if not isinstance(self.entry_module, str) or not self.entry_module.strip():
            raise InvalidEvidenceError("LoaderEntryPoint.entry_module must be a non-empty str")
        if not isinstance(self.entry_attr, str) or not self.entry_attr.isidentifier():
            raise InvalidEvidenceError(
                f"LoaderEntryPoint.entry_attr must be a valid identifier, got {self.entry_attr!r}")
        root = self.entry_module.split(".")[0]
        if root in FORBIDDEN_ENTRY_MODULE_ROOTS:
            raise InvalidEvidenceError(
                f"entry point module {self.entry_module!r} rejected: {root!r} is a "
                "forbidden entry-point namespace (fail closed)")

    def to_payload(self) -> dict:
        return {"entry_module": self.entry_module, "entry_attr": self.entry_attr}

    @classmethod
    def from_payload(cls, payload: Any) -> "LoaderEntryPoint":
        if not isinstance(payload, Mapping):
            raise InvalidEvidenceError("loader entry point must be a mapping")
        return cls(entry_module=str(payload.get("entry_module", "")),
                   entry_attr=str(payload.get("entry_attr", "")))


@dataclass(frozen=True)
class ProductionRegistryBundle:
    """Immutable controller-signed registry/spec bundle.

    This is the ONLY thing the fresh child process receives to resolve
    loaders/replay — there is deliberately NO parent-process loader registry
    anywhere in this module (the earlier ``_PRODUCTION_LOADERS`` /
    ``_PRODUCTION_REPLAY`` globals were removed: a spawned child imports a
    fresh module where parent globals never exist, so any design relying on
    them could only work through hidden import side effects).

    The bundle is hash-bound into the request (``registry_hash`` must equal
    ``bundle_sha256()``) and cross-bound to the Student ABI identity hash
    and the manifest hash; the child re-verifies every one of these
    bindings before resolving a single entry point.
    """

    registry_id: str
    controller_signature_ref: str
    student_abi_identity_hash: str
    manifest_hash: str
    loader_entry_points: Mapping[str, LoaderEntryPoint]
    replay_entry_point: LoaderEntryPoint

    def __post_init__(self) -> None:
        if not isinstance(self.registry_id, str) or not self.registry_id.strip():
            raise InvalidEvidenceError("ProductionRegistryBundle.registry_id is required")
        if not isinstance(self.controller_signature_ref, str) \
                or not self.controller_signature_ref.strip():
            raise InvalidEvidenceError(
                "ProductionRegistryBundle.controller_signature_ref is required "
                "(controller-signed bundles only; fail closed)")
        for name in ("student_abi_identity_hash", "manifest_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SHA256_RE.match(value):
                raise InvalidEvidenceError(
                    f"ProductionRegistryBundle.{name} must be a 64-hex sha256, got {value!r}")
        points = dict(self.loader_entry_points or {})
        missing = [c for c in REQUIRED_COMPONENTS if c not in points]
        if missing:
            raise InvalidEvidenceError(
                f"registry bundle must name a loader entry point for every required "
                f"component; missing {missing}")
        unknown = [c for c in points if c not in REQUIRED_COMPONENTS]
        if unknown:
            raise InvalidEvidenceError(f"registry bundle names unknown components {unknown}")
        for name, point in points.items():
            if not isinstance(point, LoaderEntryPoint):
                raise InvalidEvidenceError(
                    f"loader_entry_points[{name!r}] must be a LoaderEntryPoint, "
                    f"got {type(point).__name__}")
        if not isinstance(self.replay_entry_point, LoaderEntryPoint):
            raise InvalidEvidenceError("replay_entry_point must be a LoaderEntryPoint")
        object.__setattr__(self, "loader_entry_points", points)

    def to_payload(self) -> dict:
        return {
            "schema": BUNDLE_SCHEMA,
            "registry_id": self.registry_id,
            "controller_signature_ref": self.controller_signature_ref,
            "student_abi_identity_hash": self.student_abi_identity_hash,
            "manifest_hash": self.manifest_hash,
            "loader_entry_points": {
                name: point.to_payload()
                for name, point in sorted(self.loader_entry_points.items())},
            "replay_entry_point": self.replay_entry_point.to_payload(),
        }

    def bundle_sha256(self) -> str:
        """Deterministic hash of the immutable bundle payload."""
        return _sha256_hex(_canonical_json(self.to_payload()).encode("utf-8"))

    @classmethod
    def from_payload(cls, payload: Any) -> "ProductionRegistryBundle":
        if not isinstance(payload, Mapping):
            raise InvalidEvidenceError("registry bundle must be a JSON object")
        if payload.get("schema") != BUNDLE_SCHEMA:
            raise InvalidEvidenceError(
                f"registry bundle schema must be {BUNDLE_SCHEMA!r}, got {payload.get('schema')!r}")
        raw_points = payload.get("loader_entry_points")
        if not isinstance(raw_points, Mapping):
            raise InvalidEvidenceError("registry bundle loader_entry_points must be a mapping")
        points = {str(name): LoaderEntryPoint.from_payload(raw)
                  for name, raw in raw_points.items()}
        return cls(registry_id=str(payload.get("registry_id", "")),
                   controller_signature_ref=str(payload.get("controller_signature_ref", "")),
                   student_abi_identity_hash=str(payload.get("student_abi_identity_hash", "")),
                   manifest_hash=str(payload.get("manifest_hash", "")),
                   loader_entry_points=points,
                   replay_entry_point=LoaderEntryPoint.from_payload(
                       payload.get("replay_entry_point")))


def verify_controller_signature(bundle: ProductionRegistryBundle) -> None:
    """Controller-signature verification for production-intent bundles.

    HONEST STATE THIS ROUND: no controller signature verification material
    is bound (``CONTROLLER_SIGNED_REGISTRY_BUNDLE_BOUND`` is False), so every
    unlabelled production bundle FAILS CLOSED here with the blocked status.
    Synthetic-signature references can never masquerade as controller
    signatures.  Upgrading this gate requires the controller to bind real
    verification material — a code-level change, never a runtime knob.
    """
    if not isinstance(bundle, ProductionRegistryBundle):
        raise InvalidEvidenceError("verify_controller_signature requires ProductionRegistryBundle")
    if bundle.controller_signature_ref.startswith(SYNTHETIC_SIGNATURE_PREFIX):
        raise InvalidEvidenceError(
            f"synthetic controller signature {bundle.controller_signature_ref!r} can never "
            "be admitted on the production path")
    raise InvalidEvidenceError(
        f"{BLOCKED_WAITING_CONTROLLER_SIGNED_REGISTRY_BUNDLE}: controller signature "
        f"verification material is NOT bound (CONTROLLER_SIGNED_REGISTRY_BUNDLE_BOUND="
        f"{CONTROLLER_SIGNED_REGISTRY_BUNDLE_BOUND}); production registry bundle "
        f"{bundle.registry_id!r} cannot be admitted this round (fail closed)")


@dataclass(frozen=True)
class FreshProcessRestoreRequest:
    """Everything the single fresh child process needs, hash-bound.

    Every identity field is FAIL-CLOSED required: the source checkpoint file
    hash, the Student ABI identity hash, the registry hash and the manifest
    hash must all be exact 64-hex sha256 values, and every required component
    must have an authoritative artifact spec.  ``optimizer_source`` is pinned
    to "checkpoint" — tx.init substitution is forbidden by construction.

    The request is HASH-BOUND to its ``registry_bundle``: ``registry_hash``
    must equal the bundle's canonical sha256, and the bundle's Student ABI /
    manifest hashes must equal the request's.  The bundle is the child's
    ONLY loader/replay resolution surface.
    """

    checkpoint_path: str
    checkpoint_sha256: str
    student_abi_identity_hash: str
    registry_hash: str
    manifest_hash: str
    expected_global_step: int
    expected_next_step_digest: str
    component_artifacts: Mapping[str, ComponentArtifactSpec]
    registry_bundle: ProductionRegistryBundle
    optimizer_source: str = "checkpoint"
    fixture_label: str = ""
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.checkpoint_path:
            raise InvalidEvidenceError("FreshProcessRestoreRequest.checkpoint_path is required")
        for name in ("checkpoint_sha256", "student_abi_identity_hash",
                     "registry_hash", "manifest_hash", "expected_next_step_digest"):
            value = getattr(self, name)
            if not value or not _SHA256_RE.match(value):
                raise InvalidEvidenceError(
                    f"FreshProcessRestoreRequest.{name} must be a 64-hex sha256, got {value!r}")
        if int(self.expected_global_step) < 0:
            raise InvalidEvidenceError("expected_global_step must be >= 0")
        if self.optimizer_source != "checkpoint":
            raise InvalidEvidenceError(
                "optimizer must restore checkpoint leaves; optimizer_source="
                f"{self.optimizer_source!r} (e.g. tx.init substitution) is rejected fail closed")
        artifacts = dict(self.component_artifacts or {})
        missing = [c for c in REQUIRED_COMPONENTS if c not in artifacts]
        if missing:
            raise InvalidEvidenceError(
                f"component_artifacts must cover every required component; missing {missing}")
        for name, spec in artifacts.items():
            if not isinstance(spec, ComponentArtifactSpec):
                raise InvalidEvidenceError(
                    f"component_artifacts[{name!r}] must be ComponentArtifactSpec, "
                    f"got {type(spec).__name__}")
        if self.fixture_label not in ("", SYNTHETIC_FIXTURE_LABEL):
            raise InvalidEvidenceError(
                f"fixture_label must be empty or exactly {SYNTHETIC_FIXTURE_LABEL!r}, "
                f"got {self.fixture_label!r}")
        if not isinstance(self.registry_bundle, ProductionRegistryBundle):
            raise InvalidEvidenceError(
                "registry_bundle must be a ProductionRegistryBundle — the child's only "
                f"loader/replay resolution surface; got {type(self.registry_bundle).__name__}")
        bundle_hash = self.registry_bundle.bundle_sha256()
        if self.registry_hash != bundle_hash:
            raise InvalidEvidenceError(
                f"request must be hash-bound to the registry bundle: registry_hash "
                f"{self.registry_hash!r} != bundle sha256 {bundle_hash}")
        if self.registry_bundle.student_abi_identity_hash != self.student_abi_identity_hash:
            raise InvalidEvidenceError(
                "registry bundle student_abi_identity_hash "
                f"{self.registry_bundle.student_abi_identity_hash!r} != request "
                f"student_abi_identity_hash {self.student_abi_identity_hash!r} "
                "(explicit child binding requires EXACT hashes)")
        if self.registry_bundle.manifest_hash != self.manifest_hash:
            raise InvalidEvidenceError(
                f"registry bundle manifest_hash {self.registry_bundle.manifest_hash!r} != "
                f"request manifest_hash {self.manifest_hash!r} "
                "(explicit child binding requires EXACT hashes)")
        signature = self.registry_bundle.controller_signature_ref
        if self.fixture_label == SYNTHETIC_FIXTURE_LABEL:
            if not signature.startswith(SYNTHETIC_SIGNATURE_PREFIX):
                raise InvalidEvidenceError(
                    "labelled synthetic requests must carry a "
                    f"{SYNTHETIC_SIGNATURE_PREFIX!r}-prefixed bundle signature, got "
                    f"{signature!r} (signature/label cross-binding)")
        elif signature.startswith(SYNTHETIC_SIGNATURE_PREFIX):
            raise InvalidEvidenceError(
                f"production-intent requests reject synthetic bundle signatures "
                f"({signature!r}); a controller-signed bundle is required")
        object.__setattr__(self, "component_artifacts", artifacts)

    def to_payload(self) -> dict:
        return {
            "schema": REQUEST_SCHEMA,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "student_abi_identity_hash": self.student_abi_identity_hash,
            "registry_hash": self.registry_hash,
            "manifest_hash": self.manifest_hash,
            "expected_global_step": int(self.expected_global_step),
            "expected_next_step_digest": self.expected_next_step_digest,
            "optimizer_source": self.optimizer_source,
            "fixture_label": self.fixture_label,
            "registry_bundle": self.registry_bundle.to_payload(),
            "component_artifacts": {
                name: {"path": spec.path, "sha256": spec.sha256,
                       "expected_leaves_digest": spec.expected_leaves_digest}
                for name, spec in sorted(self.component_artifacts.items())},
        }


# ---------------------------------------------------------------------------
# Canonical leaf evidence (deterministic order; values only as sha256).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LeafRecord:
    order: int
    shape: tuple[int, ...]
    dtype: str
    value_sha256: str

    def to_row(self) -> list:
        return [int(self.order), [int(d) for d in self.shape], self.dtype, self.value_sha256]


def treedef_of(tree: Any) -> str:
    """Deterministic structural description (no values, container nesting only)."""
    if isinstance(tree, dict):
        inner = ",".join(f"{k}:{treedef_of(tree[k])}" for k in sorted(tree.keys(), key=str))
        return "dict{" + inner + "}"
    if isinstance(tree, (list, tuple)):
        kind = "list" if isinstance(tree, list) else "tuple"
        return kind + "[" + ",".join(treedef_of(item) for item in tree) + "]"
    if hasattr(tree, "shape") and hasattr(tree, "dtype") and not isinstance(tree, (str, bytes)):
        arr = _as_array(tree)
        if arr is not None:
            return f"leaf<ndarray:{arr.dtype}({','.join(str(d) for d in arr.shape)})>"
    return "leaf<scalar>"


def _as_array(node: Any):
    """Lazy numpy conversion for array-like leaves (None when unavailable)."""
    try:
        import numpy as np
    except Exception:
        return None
    try:
        return np.asarray(node)
    except Exception:
        return None


def _leaf_record(node: Any, order: int) -> LeafRecord:
    if hasattr(node, "shape") and hasattr(node, "dtype") and not isinstance(
            node, (str, bytes, bool, int, float, type(None))):
        arr = _as_array(node)
        if arr is None:
            raise InvalidEvidenceError(
                f"array-like leaf at order {order} cannot be canonicalized without numpy")
        return LeafRecord(order=order, shape=tuple(int(d) for d in arr.shape),
                          dtype=f"ndarray:{arr.dtype}",
                          value_sha256=_sha256_hex(b"ndarray:" + arr.tobytes()))
    canonical = "scalar:" + _canonical_json(node)
    return LeafRecord(order=order, shape=(), dtype="json-scalar",
                      value_sha256=_sha256_hex(canonical.encode("utf-8")))


def tree_leaf_records(tree: Any) -> tuple[LeafRecord, ...]:
    """Deterministic leaf traversal: dict keys sorted, lists/tuples in order."""
    records: list[LeafRecord] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in sorted(node.keys(), key=str):
                walk(node[key])
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)
        else:
            records.append(_leaf_record(node, len(records)))

    walk(tree)
    if not records:
        raise InvalidEvidenceError("component tree has no leaves (fail closed)")
    return tuple(records)


def leaves_digest_of(records: tuple[LeafRecord, ...] | list) -> str:
    """Digest binding leaf count + order + shape + dtype + value hashes."""
    rows = [rec.to_row() for rec in records]
    return _sha256_hex(_canonical_json(rows).encode("utf-8"))


def synthetic_replay_digest(digest_map: Mapping[str, str]) -> str:
    """Labelled synthetic next-step replay: a pure function of the JOINTLY
    restored component digests.  Real network replay arrives only through
    the controller-signed registry bundle's replay entry point (fail closed
    until a controller-signed bundle is bound)."""
    payload = {"replay": "SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT",
               "components": {k: digest_map[k] for k in sorted(digest_map)}}
    return _sha256_hex(_canonical_json(payload).encode("utf-8"))


# ---------------------------------------------------------------------------
# Child-side entry-point resolution (bundle-driven ONLY).
#
# There is deliberately NO parent-process loader/replay registry in this
# module.  The earlier ``_PRODUCTION_LOADERS`` / ``_PRODUCTION_REPLAY``
# globals were removed (audit follow-up 2026-08-04 round 2): a spawned child
# imports a fresh module where parent globals never exist, so they were dead
# surface that could only ever be bypassed through hidden import side
# effects.  The child resolves every loader and the replay exclusively
# through the explicit entry points named by the request's hash-bound
# ``ProductionRegistryBundle``.
# ---------------------------------------------------------------------------

# Defense in depth: bundle entry points must never name process/system
# manipulation namespaces, even though the bundle is controller-signed.
FORBIDDEN_ENTRY_MODULE_ROOTS = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "signal", "ctypes",
    "builtins", "multiprocessing", "threading", "_thread", "importlib",
    "webbrowser",
})


def resolve_bundle_entry_point(point: LoaderEntryPoint, purpose: str) -> Any:
    """Import ``point.entry_module`` FRESH (in the child) and return the
    callable ``point.entry_attr``.  Fail closed on any resolution problem."""
    root = point.entry_module.split(".")[0]
    if root in FORBIDDEN_ENTRY_MODULE_ROOTS:
        raise InvalidEvidenceError(
            f"entry point for {purpose!r} rejected: module {point.entry_module!r} is a "
            "forbidden namespace")
    try:
        module = importlib.import_module(point.entry_module)
    except Exception as exc:
        raise InvalidEvidenceError(
            f"entry point for {purpose!r}: cannot import module {point.entry_module!r} "
            f"in the child process: {exc}")
    target = getattr(module, point.entry_attr, None)
    if not callable(target):
        raise InvalidEvidenceError(
            f"entry point for {purpose!r}: {point.entry_module}.{point.entry_attr} is not "
            "callable in the child process")
    return target


# ---------------------------------------------------------------------------
# Child worker entry point (runs in the ONE spawned fresh process).
# ---------------------------------------------------------------------------

class _WorkerFailure(Exception):
    """Child-side fail-closed failure (exit code 4, error evidence written)."""


def _atomic_write_json(path: str | Path, payload: dict) -> None:
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(_canonical_json(payload))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, str(path))


def _read_json_file(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _worker_component_evidence(name: str, spec: Mapping[str, Any], tree: Any,
                               pid: int, origin: str) -> tuple[dict, str]:
    records = tree_leaf_records(tree)
    digest = leaves_digest_of(records)
    expected = spec.get("expected_leaves_digest", "")
    if expected and digest != expected:
        raise _WorkerFailure(
            f"component {name!r}: restored leaves digest {digest} != expected {expected}")
    evidence = {
        "component": name,
        "status": "RESTORED_HASH_BOUND",
        "origin": origin,
        "source_path": spec["path"],
        "pid": pid,
        "treedef": treedef_of(tree),
        "leaf_count": len(records),
        "leaves": [rec.to_row() for rec in records],
        "leaves_digest": digest,
    }
    return evidence, digest


def restore_worker_main(argv: list[str] | None = None) -> int:
    """Child process entry: jointly restore all components, emit atomic
    evidence, exit 0 — or fail closed with a non-zero exit code."""
    argv = list(sys.argv if argv is None else argv)
    started_at = _utc_iso_now()
    pid, ppid = os.getpid(), os.getppid()

    def process_block(exit_code: int) -> dict:
        return {"child_pid": pid, "parent_pid": ppid, "argv": argv,
                "started_at": started_at, "ended_at": _utc_iso_now(),
                "exit_code": exit_code, "worker_module": WORKER_MODULE}

    try:
        req_path = argv[argv.index("--request") + 1]
        ev_path = argv[argv.index("--evidence") + 1]
    except (ValueError, IndexError):
        sys.stderr.write("restore_worker requires --request PATH --evidence PATH\n")
        return 2

    try:
        request = _read_json_file(req_path)
        if not isinstance(request, dict) or request.get("schema") != REQUEST_SCHEMA:
            raise _WorkerFailure(f"request schema must be {REQUEST_SCHEMA}")
        for name in ("checkpoint_path", "checkpoint_sha256", "student_abi_identity_hash",
                     "registry_hash", "manifest_hash", "expected_next_step_digest"):
            value = request.get(name, "")
            if not value or (name != "checkpoint_path" and not _SHA256_RE.match(value)):
                raise _WorkerFailure(f"request field {name!r} missing or not 64-hex")
        artifacts = request.get("component_artifacts", {})
        missing = [c for c in REQUIRED_COMPONENTS if c not in artifacts]
        if missing:
            raise _WorkerFailure(f"request component_artifacts missing {missing}")
        fixture = request.get("fixture_label", "")
        if fixture not in ("", SYNTHETIC_FIXTURE_LABEL):
            raise _WorkerFailure(
                f"fixture_label must be empty or {SYNTHETIC_FIXTURE_LABEL!r}")
        optimizer_source = request.get("optimizer_source", "")
        if optimizer_source != "checkpoint":
            raise _WorkerFailure(
                f"optimizer_source {optimizer_source!r} rejected: optimizer must restore "
                "checkpoint leaves (tx.init substitution forbidden)")

        # --- CHILD-SIDE BUNDLE RE-VERIFICATION (trust nothing) -------------
        # The bundle is the ONLY loader/replay resolution surface this child
        # ever sees — there is no parent registry to fall back to.
        bundle = ProductionRegistryBundle.from_payload(request.get("registry_bundle"))
        bundle_hash = bundle.bundle_sha256()
        if bundle_hash != request["registry_hash"]:
            raise _WorkerFailure(
                f"registry bundle hash {bundle_hash} != request registry_hash "
                f"{request['registry_hash']} (request must be hash-bound to the bundle)")
        if bundle.student_abi_identity_hash != request["student_abi_identity_hash"]:
            raise _WorkerFailure(
                "registry bundle student_abi_identity_hash != request identity hash "
                "(explicit child binding requires EXACT hashes)")
        if bundle.manifest_hash != request["manifest_hash"]:
            raise _WorkerFailure(
                "registry bundle manifest_hash != request manifest_hash "
                "(explicit child binding requires EXACT hashes)")
        if fixture == SYNTHETIC_FIXTURE_LABEL:
            if not bundle.controller_signature_ref.startswith(SYNTHETIC_SIGNATURE_PREFIX):
                raise _WorkerFailure(
                    "labelled synthetic requests must carry a "
                    f"{SYNTHETIC_SIGNATURE_PREFIX!r}-prefixed bundle signature")
        else:
            # PRODUCTION path: fail closed until the controller binds real
            # signature verification material (not this round).
            verify_controller_signature(bundle)

        identity_echo = {
            "checkpoint_path": request["checkpoint_path"],
            "checkpoint_sha256": request["checkpoint_sha256"],
            "student_abi_identity_hash": request["student_abi_identity_hash"],
            "manifest_hash": request["manifest_hash"],
            "registry_bundle_sha256": bundle_hash,
            "expected_global_step": request["expected_global_step"],
        }

        # --- resolve loaders/replay THROUGH THE BUNDLE ONLY ----------------
        loaders = {name: resolve_bundle_entry_point(bundle.loader_entry_points[name],
                                                    f"component loader {name!r}")
                   for name in REQUIRED_COMPONENTS}
        replay = resolve_bundle_entry_point(bundle.replay_entry_point,
                                            f"replay {REPLAY_CHECK!r}")

        components: list[dict] = []
        digests: dict[str, str] = {}
        for name in REQUIRED_COMPONENTS:
            spec = artifacts[name]
            # Authoritative source hash is verified by the worker itself,
            # before any loader runs (defense in depth: loaders re-verify).
            try:
                with open(spec["path"], "rb") as handle:
                    blob = handle.read()
            except OSError as exc:
                raise _WorkerFailure(f"component {name!r}: cannot read artifact: {exc}")
            if _sha256_hex(blob) != spec["sha256"]:
                raise _WorkerFailure(
                    f"component {name!r}: artifact file sha256 mismatch "
                    "(authoritative source hash violated)")
            context = {"component": name, "spec": dict(spec), **identity_echo}
            tree = loaders[name](context)
            origin = (OPTIMIZER_ORIGIN_CHECKPOINT if name == "optimizer"
                      else ORIGIN_SOURCE_ARTIFACT)
            evidence_row, digest = _worker_component_evidence(
                name, spec, tree, pid, origin)
            components.append(evidence_row)
            digests[name] = digest

        replay_context = {
            "component_digests": digests,
            "registry_hash": request["registry_hash"],
            **identity_echo,
        }
        replay_digest = replay(replay_context)
        if not isinstance(replay_digest, str) or not _SHA256_RE.match(replay_digest):
            raise _WorkerFailure(
                f"replay entry point must return a 64-hex digest, got {replay_digest!r}")

        if replay_digest != request["expected_next_step_digest"]:
            raise _WorkerFailure(
                "policy_step_next_replay diverged on the jointly restored state: "
                f"{replay_digest} != {request['expected_next_step_digest']}")

        evidence = {
            "schema": EVIDENCE_SCHEMA,
            "fixture_label": fixture,
            "error": "",
            "process": process_block(0),
            "request_echo": {
                "checkpoint_path": request["checkpoint_path"],
                "checkpoint_sha256": request["checkpoint_sha256"],
                "student_abi_identity_hash": request["student_abi_identity_hash"],
                "registry_hash": request["registry_hash"],
                "manifest_hash": request["manifest_hash"],
                "expected_global_step": request["expected_global_step"],
                "optimizer_source": request["optimizer_source"],
                "registry_bundle_sha256": bundle_hash,
            },
            "components": components,
            "cross_checks": [{"name": REPLAY_CHECK, "status": "RESTORED_CROSS_VERIFIED",
                              "digest": replay_digest, "pid": pid}],
        }
        _atomic_write_json(ev_path, evidence)
        return 0
    except InvalidEvidenceError as exc:
        # Fail-closed contract violations (bundle parsing, entry-point
        # resolution, controller signature block) — never exit 0.
        try:
            _atomic_write_json(ev_path, {
                "schema": EVIDENCE_SCHEMA, "fixture_label": "", "error": str(exc),
                "process": process_block(4), "request_echo": {},
                "components": [], "cross_checks": []})
        except Exception:
            pass
        sys.stderr.write(f"restore_worker FAILED: {exc}\n")
        return 4
    except _WorkerFailure as exc:
        try:
            _atomic_write_json(ev_path, {
                "schema": EVIDENCE_SCHEMA, "fixture_label": "", "error": str(exc),
                "process": process_block(4), "request_echo": {},
                "components": [], "cross_checks": []})
        except Exception:
            pass
        sys.stderr.write(f"restore_worker FAILED: {exc}\n")
        return 4
    except Exception as exc:  # never exit 0 on an unexpected failure
        sys.stderr.write(f"restore_worker CRASHED: {exc!r}\n")
        return 5


# ---------------------------------------------------------------------------
# Evidence parsing + mechanical verification (parent side).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComponentLeafEvidence:
    component: str
    status: str
    origin: str
    source_path: str
    pid: int
    treedef: str
    leaf_count: int
    leaves: tuple[LeafRecord, ...]
    leaves_digest: str


@dataclass(frozen=True)
class ReplayEvidence:
    name: str
    status: str
    digest: str
    pid: int


@dataclass(frozen=True)
class ProcessEvidence:
    schema: str
    fixture_label: str
    child_pid: int
    parent_pid: int
    child_argv: tuple[str, ...]
    started_at: str
    ended_at: str
    exit_code: int
    worker_module: str
    request_echo: Mapping[str, Any]
    components: tuple[ComponentLeafEvidence, ...]
    cross_checks: tuple[ReplayEvidence, ...]
    error: str = ""

    def component_map(self) -> dict[str, ComponentLeafEvidence]:
        return {comp.component: comp for comp in self.components}


def load_evidence_payload(path: str | Path) -> dict:
    """Read the atomic evidence file; missing/torn files fail closed."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        raise InvalidEvidenceError(
            f"fresh-process evidence file missing or unreadable ({exc}) -> crash/torn "
            "report rejected fail closed")
    try:
        payload = json.loads(text)
    except Exception as exc:
        raise InvalidEvidenceError(
            f"fresh-process evidence is torn/unparseable ({exc}) -> rejected fail closed")
    if not isinstance(payload, dict):
        raise InvalidEvidenceError("fresh-process evidence must be a JSON object")
    return payload


def _parse_iso(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise InvalidEvidenceError(f"timestamp must be a non-empty ISO string, got {value!r}")
    try:
        return datetime.fromisoformat(value)
    except Exception as exc:
        raise InvalidEvidenceError(f"unparseable timestamp {value!r}: {exc}")


def verify_fresh_process_evidence(payload: Mapping[str, Any], *,
                                  launched_pid: int,
                                  expected_parent_pid: int,
                                  request: FreshProcessRestoreRequest,
                                  allow_synthetic_fixture: bool) -> ProcessEvidence:
    """Mechanical verification of the child's atomic evidence.

    Every check is fail-closed; violations are collected and raised together
    so the rejection reason is precise.  This function is the enforcement
    heart of the R4c production gate.

    Process-chain binding: the evidence child PID must be a NEW process
    (never the driver PID), and the evidence parent PID must match EITHER
    the driver PID (direct spawn) OR the PID this call actually launched.
    The second case covers the Windows venv launcher, where ``Popen.pid`` is
    an intermediate launcher process and the real interpreter is its child —
    the chain is still anchored to this invocation because the launched PID
    is fresh and unique to it.  Any other parent PID is a split/forged
    process and is rejected.
    """
    if not isinstance(request, FreshProcessRestoreRequest):
        raise InvalidEvidenceError("verify_fresh_process_evidence requires "
                                   "FreshProcessRestoreRequest")
    if not isinstance(payload, Mapping):
        raise InvalidEvidenceError("evidence payload must be a mapping")

    violations: list[str] = []

    if payload.get("schema") != EVIDENCE_SCHEMA:
        violations.append(f"evidence schema must be {EVIDENCE_SCHEMA!r}")

    fixture = payload.get("fixture_label", "")
    if not allow_synthetic_fixture:
        if fixture:
            violations.append(
                "production path rejects synthetic fixture evidence "
                f"(fixture_label={fixture!r})")
    elif fixture not in ("", SYNTHETIC_FIXTURE_LABEL):
        violations.append(f"fixture_label must be empty or {SYNTHETIC_FIXTURE_LABEL!r}")

    # --- process identity: exactly one NEW child process -------------------
    proc = payload.get("process", {}) if isinstance(payload.get("process"), Mapping) else {}
    child_pid = proc.get("child_pid")
    parent_pid = proc.get("parent_pid")
    argv = proc.get("argv")
    started_at = proc.get("started_at", "")
    ended_at = proc.get("ended_at", "")
    exit_code = proc.get("exit_code")
    worker_module = proc.get("worker_module", "")

    if not isinstance(child_pid, int) or child_pid <= 0:
        violations.append(f"evidence child_pid must be a positive int, got {child_pid!r}")
    elif child_pid == int(expected_parent_pid):
        violations.append(
            "parent-process execution rejected: evidence child_pid equals the parent "
            "PID (the joint restore must run in a NEW process)")
    if not isinstance(parent_pid, int) or parent_pid not in (
            int(expected_parent_pid), int(launched_pid)):
        violations.append(
            f"evidence parent_pid {parent_pid!r} matches neither the driver PID "
            f"{int(expected_parent_pid)} nor the launched process PID {int(launched_pid)} "
            "(split/forged process chain rejected)")
    if not isinstance(argv, (list, tuple)) or not argv or not any(
            "restore_worker" in str(part) for part in argv):
        violations.append("evidence argv must be non-empty and reference restore_worker")
    if worker_module != WORKER_MODULE:
        violations.append(f"worker_module must be {WORKER_MODULE!r}, got {worker_module!r}")
    start_ok = end_ok = False
    try:
        start_dt = _parse_iso(started_at)
        start_ok = True
    except InvalidEvidenceError as exc:
        violations.append(f"started_at invalid: {exc}")
    try:
        end_dt = _parse_iso(ended_at)
        end_ok = True
    except InvalidEvidenceError as exc:
        violations.append(f"ended_at invalid: {exc}")
    if start_ok and end_ok and end_dt < start_dt:
        violations.append("ended_at precedes started_at")
    if exit_code != 0:
        violations.append(f"child exit_code must be 0, got {exit_code!r} "
                          "(crash/failure rejected fail closed)")

    # --- request echo: exact source checkpoint / ABI / registry / manifest -
    # plus the registry-bundle hash the child actually resolved through.
    echo = payload.get("request_echo", {}) if isinstance(payload.get("request_echo"), Mapping) else {}
    expected_echo = {
        "checkpoint_path": request.checkpoint_path,
        "checkpoint_sha256": request.checkpoint_sha256,
        "student_abi_identity_hash": request.student_abi_identity_hash,
        "registry_hash": request.registry_hash,
        "manifest_hash": request.manifest_hash,
        "expected_global_step": int(request.expected_global_step),
        "optimizer_source": request.optimizer_source,
        "registry_bundle_sha256": request.registry_bundle.bundle_sha256(),
    }
    for key, expected in expected_echo.items():
        if echo.get(key) != expected:
            violations.append(f"request_echo[{key!r}] {echo.get(key)!r} != required {expected!r}")

    # --- components: all nine, single PID, frozen RESTORED statuses --------
    raw_components = payload.get("components")
    if not isinstance(raw_components, list):
        violations.append("evidence components must be a list")
        raw_components = []
    by_name: dict[str, ComponentLeafEvidence] = {}
    for raw in raw_components:
        if not isinstance(raw, Mapping):
            violations.append("component evidence rows must be mappings")
            continue
        name = raw.get("component", "")
        if name not in REQUIRED_COMPONENTS:
            violations.append(f"unexpected/unknown component {name!r} in evidence")
            continue
        if name in by_name:
            violations.append(f"duplicate component evidence for {name!r}")
            continue
        status = raw.get("status", "")
        if status not in RESTORED_STATUSES:
            violations.append(
                f"component {name!r}: self-asserted status {status!r} rejected; only the "
                f"frozen statuses {tuple(RESTORED_STATUSES)} are admissible")
        comp_pid = raw.get("pid")
        if isinstance(child_pid, int) and comp_pid != child_pid:
            violations.append(
                f"component {name!r}: evidence pid {comp_pid!r} != child_pid {child_pid} "
                "(split-process composition rejected: ONE fresh process is mandatory)")
        leaves_raw = raw.get("leaves")
        if not isinstance(leaves_raw, list) or not leaves_raw:
            violations.append(f"component {name!r}: leaf evidence missing (fake RESTORED rejected)")
            continue
        records: list[LeafRecord] = []
        leaf_ok = True
        for index, row in enumerate(leaves_raw):
            if not (isinstance(row, (list, tuple)) and len(row) == 4):
                violations.append(f"component {name!r}: malformed leaf row {index}")
                leaf_ok = False
                break
            order, shape, dtype, value_sha = row
            if order != index:
                violations.append(
                    f"component {name!r}: leaf order gap/reorder at index {index} "
                    f"(order={order!r}; leaves must be exactly 0..n-1 in evidence order)")
                leaf_ok = False
                break
            if not (isinstance(shape, list) and all(isinstance(d, int) and d >= 0 for d in shape)):
                violations.append(f"component {name!r}: leaf {index} shape must be a list of ints")
                leaf_ok = False
                break
            if not isinstance(dtype, str) or not dtype:
                violations.append(f"component {name!r}: leaf {index} dtype must be a non-empty str")
                leaf_ok = False
                break
            if not isinstance(value_sha, str) or not _SHA256_RE.match(value_sha):
                violations.append(f"component {name!r}: leaf {index} value_sha256 must be 64-hex")
                leaf_ok = False
                break
            records.append(LeafRecord(order=index, shape=tuple(shape),
                                      dtype=dtype, value_sha256=value_sha))
        if not leaf_ok or not records:
            continue
        leaf_count = raw.get("leaf_count")
        if leaf_count != len(records):
            violations.append(
                f"component {name!r}: leaf_count {leaf_count!r} != actual leaf rows "
                f"{len(records)} (missing/extra leaves rejected)")
        treedef = raw.get("treedef", "")
        if not isinstance(treedef, str) or not treedef:
            violations.append(f"component {name!r}: treedef must be a non-empty string")
        recorded_digest = raw.get("leaves_digest", "")
        recomputed = leaves_digest_of(tuple(records))
        if recorded_digest != recomputed:
            violations.append(
                f"component {name!r}: leaves_digest {recorded_digest!r} != recomputed "
                f"{recomputed} (leaf value/order/shape/dtype tampering rejected)")
        spec = request.component_artifacts[name]
        if spec.expected_leaves_digest and recorded_digest != spec.expected_leaves_digest:
            violations.append(
                f"component {name!r}: leaves digest does not match the request expectation "
                "(checkpoint-bound digest violated)")
        if raw.get("source_path") != spec.path:
            violations.append(
                f"component {name!r}: source_path {raw.get('source_path')!r} != authoritative "
                f"artifact path {spec.path!r}")
        origin = raw.get("origin", "")
        if name == "optimizer":
            if origin != OPTIMIZER_ORIGIN_CHECKPOINT:
                violations.append(
                    f"optimizer origin {origin!r} rejected: optimizer must restore checkpoint "
                    f"leaves (origin must be {OPTIMIZER_ORIGIN_CHECKPOINT}; tx.init "
                    "substitution is forbidden)")
        elif origin != ORIGIN_SOURCE_ARTIFACT:
            violations.append(f"component {name!r}: unknown origin {origin!r}")
        by_name[name] = ComponentLeafEvidence(
            component=name, status=status, origin=origin,
            source_path=raw.get("source_path", ""), pid=comp_pid if isinstance(comp_pid, int) else -1,
            treedef=treedef, leaf_count=len(records), leaves=tuple(records),
            leaves_digest=recorded_digest)
    missing_components = [c for c in REQUIRED_COMPONENTS if c not in by_name]
    if missing_components:
        violations.append(
            f"evidence is missing required components {missing_components} "
            "(nine-component joint restore is mandatory)")

    # --- cross-check: next policy step replay on the JOINT state -----------
    raw_cross = payload.get("cross_checks")
    if not isinstance(raw_cross, list):
        violations.append("evidence cross_checks must be a list")
        raw_cross = []
    cross_rows: list[ReplayEvidence] = []
    for raw in raw_cross:
        if not isinstance(raw, Mapping):
            violations.append("cross-check rows must be mappings")
            continue
        if raw.get("name") != REPLAY_CHECK:
            violations.append(f"unexpected cross-check {raw.get('name')!r}")
            continue
        if raw.get("status") != "RESTORED_CROSS_VERIFIED":
            violations.append(
                f"cross-check {REPLAY_CHECK}: status {raw.get('status')!r} rejected; "
                "RESTORED_CROSS_VERIFIED required")
        cross_pid = raw.get("pid")
        if isinstance(child_pid, int) and cross_pid != child_pid:
            violations.append(
                f"cross-check {REPLAY_CHECK}: pid {cross_pid!r} != child_pid {child_pid} "
                "(split-process composition rejected)")
        digest = raw.get("digest", "")
        if digest != request.expected_next_step_digest:
            violations.append(
                f"cross-check {REPLAY_CHECK}: digest {digest!r} != expected next-step digest "
                f"{request.expected_next_step_digest} (replay diverged on the joint state)")
        cross_rows.append(ReplayEvidence(name=REPLAY_CHECK, status=raw.get("status", ""),
                                         digest=digest,
                                         pid=cross_pid if isinstance(cross_pid, int) else -1))
    if not any(row.name == REPLAY_CHECK for row in cross_rows):
        violations.append(f"cross-check {REPLAY_CHECK} missing from evidence")

    if violations:
        raise InvalidEvidenceError(
            "fresh-process evidence rejected fail closed: " + "; ".join(violations))

    proc_block = payload["process"]
    return ProcessEvidence(
        schema=EVIDENCE_SCHEMA, fixture_label=fixture,
        child_pid=int(proc_block["child_pid"]), parent_pid=int(proc_block["parent_pid"]),
        child_argv=tuple(str(part) for part in proc_block["argv"]),
        started_at=proc_block["started_at"], ended_at=proc_block["ended_at"],
        exit_code=int(proc_block["exit_code"]), worker_module=proc_block["worker_module"],
        request_echo=dict(payload.get("request_echo", {})),
        components=tuple(by_name[name] for name in REQUIRED_COMPONENTS),
        cross_checks=tuple(cross_rows), error=str(payload.get("error", "")))


# ---------------------------------------------------------------------------
# Parent driver: spawn EXACTLY ONE child, verify its atomic evidence.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FreshProcessRestoreOutcome:
    accepted: bool
    joint_proof_status: str
    violations: tuple[str, ...] = ()
    child_pid: int | None = None
    child_returncode: int | None = None
    evidence: ProcessEvidence | None = None


def _rejected(violations: tuple[str, ...] | list, child_pid: int | None = None,
              child_returncode: int | None = None,
              evidence: ProcessEvidence | None = None) -> FreshProcessRestoreOutcome:
    return FreshProcessRestoreOutcome(
        accepted=False,
        joint_proof_status="COMBINED_FRESH_PROCESS_RESTORE=false (" + _STATUS_DISCLAIMER
                           + "; rejected: " + "; ".join(violations) + ")",
        violations=tuple(violations), child_pid=child_pid,
        child_returncode=child_returncode, evidence=evidence)


def run_fresh_process_restore(request: FreshProcessRestoreRequest, *,
                              allow_synthetic_fixture: bool,
                              scratch_dir: str | Path,
                              timeout_s: float = 120.0) -> FreshProcessRestoreOutcome:
    """Spawn exactly one fresh child process and mechanically verify its
    atomic joint-restore evidence.  NO callback surface exists here: the
    child restores from authoritative hash-bound artifacts only, resolving
    loaders/replay exclusively through the request's hash-bound registry
    bundle entry points (there is no parent-process loader registry)."""
    if not isinstance(request, FreshProcessRestoreRequest):
        raise InvalidEvidenceError("run_fresh_process_restore requires FreshProcessRestoreRequest")
    if not allow_synthetic_fixture and request.fixture_label:
        raise InvalidEvidenceError(
            "production path rejects synthetic fixture requests "
            f"(fixture_label={request.fixture_label!r}); only the controller-injected real "
            "artifact path is admissible")
    scratch = Path(scratch_dir)
    if not scratch.is_dir():
        raise InvalidEvidenceError(f"scratch_dir must exist: {scratch}")

    request_path = scratch / "fresh_process_request.json"
    evidence_path = scratch / "fresh_process_evidence.json"
    _atomic_write_json(request_path, request.to_payload())

    argv = [sys.executable, "-m", WORKER_MODULE,
            "--request", str(request_path), "--evidence", str(evidence_path)]
    # EXACTLY ONE spawn: this single Popen call is the only process creation.
    try:
        with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as child:
            child_pid = child.pid
            try:
                _stdout, stderr = child.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                child.kill()
                child.communicate()
                return _rejected((f"child process timed out after {timeout_s}s (killed)",),
                                 child_pid=child_pid, child_returncode=None)
            returncode = child.returncode
    except OSError as exc:
        return _rejected((f"failed to spawn the fresh child process: {exc}",))

    if returncode != 0:
        tail = (stderr or b"").decode("utf-8", "replace")[-400:]
        return _rejected(
            (f"child exited {returncode} (crash/BLOCKED/failure rejected fail closed): {tail}",),
            child_pid=child_pid, child_returncode=returncode)

    try:
        payload = load_evidence_payload(evidence_path)
    except InvalidEvidenceError as exc:
        return _rejected((str(exc),), child_pid=child_pid, child_returncode=returncode)

    try:
        evidence = verify_fresh_process_evidence(
            payload, launched_pid=child_pid, expected_parent_pid=os.getpid(),
            request=request, allow_synthetic_fixture=allow_synthetic_fixture)
    except InvalidEvidenceError as exc:
        return _rejected((str(exc),), child_pid=child_pid, child_returncode=returncode)

    if evidence.fixture_label == SYNTHETIC_FIXTURE_LABEL:
        status = ("FRESH_PROCESS_DRIVER_CONTRACT_PASS ("
                  "SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT; single-subprocess evidence "
                  "verified mechanically; NOT a joint proof on real artifacts; "
                  "COMBINED_FRESH_PROCESS_RESTORE=false)")
    else:
        # Reachable only when a controller-registered production path has run
        # on real artifacts inside the child — never happened this round.
        status = ("COMBINED_FRESH_PROCESS_RESTORE=true (verified single-fresh-process "
                  "atomic evidence on the production path)")
    return FreshProcessRestoreOutcome(
        accepted=True, joint_proof_status=status, violations=(),
        child_pid=child_pid, child_returncode=returncode, evidence=evidence)


def run_fresh_process_restore_production(request: FreshProcessRestoreRequest, *,
                                         scratch_dir: str | Path,
                                         timeout_s: float = 120.0) -> FreshProcessRestoreOutcome:
    """PRODUCTION entry point (R4c).  Structurally callback-free: no
    restorers/cross_checkers parameters exist; synthetic fixtures and
    synthetic controller signatures are rejected; the child resolves
    loaders/replay ONLY through the request's hash-bound, controller-signed
    registry bundle and fails closed while no controller signature
    verification material is bound (this round).
    """
    if request.fixture_label:
        raise InvalidEvidenceError(
            "run_fresh_process_restore_production rejects synthetic fixture requests "
            f"(fixture_label={request.fixture_label!r})")
    signature = request.registry_bundle.controller_signature_ref
    if signature.startswith(SYNTHETIC_SIGNATURE_PREFIX):
        raise InvalidEvidenceError(
            f"run_fresh_process_restore_production rejects synthetic controller "
            f"signatures ({signature!r}); a controller-signed registry bundle is "
            "required on the production path")
    return run_fresh_process_restore(request, allow_synthetic_fixture=False,
                                     scratch_dir=scratch_dir, timeout_s=timeout_s)


def verdict_from_evidence(evidence: ProcessEvidence) -> CombinedRestoreVerdict:
    """The CANONICAL builder for a CombinedRestoreVerdict grounded in
    mechanically verified fresh-process evidence: every component result is
    bound to that component's authoritative leaves digest from the SAME
    evidence, and the replay cross-check is bound to the evidence replay
    digest.  ``production_joint_pass`` accepts verdicts corresponding to the
    evidence; this builder is how such a verdict is produced honestly."""
    if not isinstance(evidence, ProcessEvidence):
        raise InvalidEvidenceError(
            "verdict_from_evidence requires mechanically verified ProcessEvidence")
    components = {
        comp.component: ComponentResult(
            comp.component, ComponentStatus.RESTORED_HASH_BOUND,
            "fresh-process authoritative leaf evidence", bound_digest=comp.leaves_digest)
        for comp in evidence.components}
    cross = {
        row.name: ComponentResult(
            row.name, ComponentStatus.RESTORED_CROSS_VERIFIED,
            "policy-step replay on the jointly restored state", bound_digest=row.digest)
        for row in evidence.cross_checks}
    return evaluate_verdict(components, cross)


def production_joint_pass(verdict: CombinedRestoreVerdict,
                          evidence: ProcessEvidence) -> bool:
    """The ONLY composition that may upgrade COMBINED_FRESH_PROCESS_RESTORE.

    Requirements (ALL fail closed):
      * mechanically verified, non-fixture ProcessEvidence with exit code 0;
      * ``verdict.combined_pass`` true;
      * the verdict CORRESPONDS TO THE SAME evidence — not an independently
        fabricated CombinedRestoreVerdict: exact component coverage, every
        verdict component status in the frozen RESTORED set, every verdict
        ``bound_digest`` EQUAL to that component's evidence leaves digest,
        and the replay cross-check bound to the evidence replay digest.
    Self-asserted verdicts without digest binding can never compose.
    """
    if not isinstance(verdict, CombinedRestoreVerdict):
        raise InvalidEvidenceError("production_joint_pass requires CombinedRestoreVerdict")
    if not isinstance(evidence, ProcessEvidence):
        raise InvalidEvidenceError(
            "production_joint_pass requires mechanically verified ProcessEvidence; "
            "self-asserted component statuses alone can never compose a joint proof")
    if not (bool(verdict.combined_pass) and evidence.exit_code == 0
            and evidence.fixture_label == ""):
        return False

    # --- verdict <-> same-evidence correspondence --------------------------
    evidence_map = evidence.component_map()
    if set(verdict.components) != set(REQUIRED_COMPONENTS) \
            or set(evidence_map) != set(REQUIRED_COMPONENTS):
        return False
    for name in REQUIRED_COMPONENTS:
        result = verdict.components[name]
        if result.status.value not in RESTORED_STATUSES:
            return False
        if result.bound_digest != evidence_map[name].leaves_digest:
            return False  # unbound/fabricated digest: not the same evidence
        if evidence_map[name].pid != evidence.child_pid:
            return False  # split-process evidence can never compose
    evidence_cross = {row.name: row for row in evidence.cross_checks}
    for check in CROSS_CHECKS:
        if check not in verdict.cross_checks or check not in evidence_cross:
            return False
        result = verdict.cross_checks[check]
        if result.status.value not in RESTORED_STATUSES:
            return False
        if result.bound_digest != evidence_cross[check].digest:
            return False
        if evidence_cross[check].pid != evidence.child_pid:
            return False
    return True
