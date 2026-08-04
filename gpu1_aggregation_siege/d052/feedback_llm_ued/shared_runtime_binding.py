"""P0 shared-runtime bindings — direction two is CONSUME-ONLY.

Master-directive rule: the P0 shared infrastructure (unique StudentIdentity,
ReferenceIdentity + ReferenceAdapter with output-leak guard, frozen
four-anchor manifest, formal asset registry, full-state checkpoint +
production registry bundle, shared CandidateProbeRunner) has ONE owner; the
three directions only consume it. This module is the consume-only binding
layer: a set of slots that accept EXPLICITLY injected shared assets and
nothing else — no loader, no registry, no codec, no second implementation.

Verified local state (this worktree): the shared StudentAdapter exists
only in the mechanism_UED_sim_foundation worktree; the ReferenceAdapter,
the shared CandidateProbeRunner, the frozen AnchorManifest, the formal
asset registry and the signed full-state checkpoint are ALL ABSENT. Every
slot therefore stays EMPTY with status ``BLOCKED_WAITING_SHARED_RUNTIME``
and :func:`resolve_shared_runtime` fails closed — which is exactly the
honest posture the two-window real entrypoint must report.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Protocol, Tuple, \
    runtime_checkable

from d052.bagr_ued.hashing import canonical_sha256
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.anchor_manifest import (
    SCAFFOLD_PLACEHOLDER_NOT_SHARED,
    SHARED_MANIFEST_BOUND_LABEL,
    AnchorManifestBlocked,
    AnchorManifestSource,
    SharedAnchorManifest,
)
from d052.feedback_llm_ued.student_binding import (
    StudentBindingIdentity,
    StudentInitContract,
    resolve_student_binding,
)
from d052.schemas.common import is_sha256_hex

#: slot statuses
STATUS_BOUND = "BOUND"
STATUS_EMPTY = C.BLOCKED_WAITING_SHARED_RUNTIME


class SharedRuntimeBlocked(RuntimeError):
    """Fail-closed refusal: required shared runtime assets are absent."""


class SharedBindingRejected(RuntimeError):
    """An injected asset failed identity/integrity verification."""


# ---------------------------------------------------------------------------
# Reference identity (the shared ReferenceAdapter's contract, consume-only)
# ---------------------------------------------------------------------------
@runtime_checkable
class ReferenceInitContract(Protocol):
    """Read-only shape of the shared Reference contract this direction
    consumes (unique ReferenceIdentity; output-leak guard enforced by the
    owner, asserted here at the boundary)."""

    candidate_id: str
    parameter_tree_hash: str
    checkpoint_global_step: int


@dataclass(frozen=True)
class ReferenceBindingIdentity:
    """Identity stamped onto feedback records alongside the Student's."""

    candidate_id: str
    parameter_tree_hash: str
    checkpoint_global_step: int
    provenance_label: str

    @property
    def identity_hash(self) -> str:
        return canonical_sha256(dict(
            candidate_id=self.candidate_id,
            parameter_tree_hash=self.parameter_tree_hash,
            checkpoint_global_step=self.checkpoint_global_step,
            provenance_label=self.provenance_label))


def resolve_reference_binding(contract: Optional[ReferenceInitContract]
                              ) -> ReferenceBindingIdentity:
    """Validate an explicitly injected shared Reference contract.

    Fail-closed ladder:
      * no contract           -> REFERENCE_INIT_CONTRACT_MISSING
      * incomplete identity   -> REFERENCE_IDENTITY_INCOMPLETE
    """
    if contract is None:
        raise SharedBindingRejected(
            "REFERENCE_INIT_CONTRACT_MISSING: the shared ReferenceAdapter / "
            "ReferenceIdentity is not present in this worktree; direction "
            "two consumes it only")
    candidate_id = getattr(contract, "candidate_id", None)
    if not isinstance(candidate_id, str) or not candidate_id:
        raise SharedBindingRejected(
            f"REFERENCE_IDENTITY_INCOMPLETE: candidate_id must be a "
            f"non-empty string, got {candidate_id!r}")
    param_hash = getattr(contract, "parameter_tree_hash", None) or ""
    if not is_sha256_hex(param_hash):
        raise SharedBindingRejected(
            f"REFERENCE_IDENTITY_INCOMPLETE: parameter_tree_hash must be a "
            f"sha256 hex string, got {param_hash!r}")
    step = getattr(contract, "checkpoint_global_step", None)
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise SharedBindingRejected(
            f"REFERENCE_IDENTITY_INCOMPLETE: checkpoint_global_step must be "
            f"a non-negative int, got {step!r}")
    return ReferenceBindingIdentity(
        candidate_id=candidate_id,
        parameter_tree_hash=param_hash,
        checkpoint_global_step=step,
        provenance_label="SHARED_REFERENCE_INIT_CONTRACT")


# ---------------------------------------------------------------------------
# shared training / checkpoint contract (consume-only)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainingUpdateResult:
    """What exactly one real optimizer update must report (audit-grade)."""

    window: int
    optimizer_steps: int
    env_steps: int
    checkpoint_hash_before: str
    checkpoint_hash_after: str


@runtime_checkable
class SharedTrainingContract(Protocol):
    """The shared runtime's training/checkpoint surface this loop consumes.

    Exactly ONE optimizer update per revision window; checkpoint save/load
    round-trip around it. The owner guarantees full-state semantics; this
    direction never re-implements an optimizer or a checkpoint codec.
    """

    def run_one_optimizer_update(self, *, window: int,
                                 batch_candidate_ids: List[str]
                                 ) -> TrainingUpdateResult: ...

    def save_checkpoint(self, *, tag: str) -> str: ...

    def load_checkpoint(self, *, checkpoint_hash: str) -> None: ...


# ---------------------------------------------------------------------------
# the consume-only slots
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SharedStudentSlot:
    status: str = STATUS_EMPTY
    detail: str = "shared StudentAdapter absent from this worktree"
    binding: Optional[StudentBindingIdentity] = None

    def bind(self, contract: StudentInitContract) -> "SharedStudentSlot":
        #: resolve_student_binding is the existing fail-closed ladder
        #: (STUDENT_INIT_CONTRACT_MISSING / STUDENT_IDENTITY_MISMATCH /
        #: STUDENT_IDENTITY_INCOMPLETE) — identity MUST be the fixed
        #: PERSISTENT_RMT16_ORIGINAL_VTRACE_98304 candidate.
        binding = resolve_student_binding(contract)
        return SharedStudentSlot(status=STATUS_BOUND,
                                 detail=binding.provenance_label,
                                 binding=binding)


@dataclass(frozen=True)
class SharedReferenceSlot:
    status: str = STATUS_EMPTY
    detail: str = "shared ReferenceAdapter absent from this worktree"
    binding: Optional[ReferenceBindingIdentity] = None

    def bind(self, contract: ReferenceInitContract) -> "SharedReferenceSlot":
        binding = resolve_reference_binding(contract)
        return SharedReferenceSlot(status=STATUS_BOUND,
                                   detail=binding.provenance_label,
                                   binding=binding)


@dataclass(frozen=True)
class SharedProbeRunnerSlot:
    status: str = STATUS_EMPTY
    detail: str = "shared CandidateProbeRunner absent from this worktree"
    runner: Optional[object] = None

    def bind(self, runner: object) -> "SharedProbeRunnerSlot":
        if getattr(runner, "real_simulator", None) is not True:
            raise SharedBindingRejected(
                "PROBE_RUNNER_NOT_REAL: the shared CandidateProbeRunner "
                "must report real_simulator=True (production paths may not "
                "consume candidate-hash-derived symbolic metrics)")
        runner_id = getattr(runner, "runner_id", "")
        if not isinstance(runner_id, str) or not runner_id:
            raise SharedBindingRejected(
                "PROBE_RUNNER_ID_MISSING: the shared runner must expose a "
                "non-empty runner_id")
        return SharedProbeRunnerSlot(status=STATUS_BOUND,
                                     detail=runner_id,
                                     runner=runner)


@dataclass(frozen=True)
class SharedAnchorManifestSlot:
    status: str = STATUS_EMPTY
    detail: str = ("no cross-direction shared frozen anchor manifest exists "
                   "in this worktree")
    anchor_ids: Tuple[str, ...] = ()
    manifest_hash: str = ""
    binding_label: str = SCAFFOLD_PLACEHOLDER_NOT_SHARED

    def bind(self, manifest: object) -> "SharedAnchorManifestSlot":
        #: wraps the EXISTING AnchorManifestSource seam (manifest_hash
        #: recomputed and compared; anything unfrozen fails closed)
        if isinstance(manifest, Mapping):
            manifest = SharedAnchorManifest(**manifest)
        source = AnchorManifestSource(manifest=manifest)
        try:
            anchor_ids = source.resolve()
        except AnchorManifestBlocked as exc:
            raise SharedBindingRejected(
                f"ANCHOR_MANIFEST_BINDING_REJECTED: {exc}") from exc
        return SharedAnchorManifestSlot(
            status=STATUS_BOUND,
            detail="shared frozen anchor manifest bound",
            anchor_ids=tuple(anchor_ids),
            manifest_hash=str(getattr(manifest, "manifest_hash", "")),
            binding_label=SHARED_MANIFEST_BOUND_LABEL)


@dataclass(frozen=True)
class SharedTrainingSlot:
    status: str = STATUS_EMPTY
    detail: str = ("shared full-state checkpoint + optimizer surface absent "
                   "from this worktree")
    contract: Optional[SharedTrainingContract] = None

    def bind(self, contract: SharedTrainingContract) -> "SharedTrainingSlot":
        for method in ("run_one_optimizer_update", "save_checkpoint",
                       "load_checkpoint"):
            if not callable(getattr(contract, method, None)):
                raise SharedBindingRejected(
                    f"SHARED_TRAINING_CONTRACT_INCOMPLETE: missing callable "
                    f"{method!r}")
        return SharedTrainingSlot(status=STATUS_BOUND,
                                  detail="shared training contract bound",
                                  contract=contract)


# ---------------------------------------------------------------------------
# the bundle
# ---------------------------------------------------------------------------
#: slot name -> human-readable asset description (for blocker reports)
SLOT_ASSET_DESCRIPTIONS = {
    "student": "shared StudentAdapter (unique StudentIdentity)",
    "reference": "shared ReferenceAdapter (unique ReferenceIdentity)",
    "probe_runner": "shared CandidateProbeRunner (real reset/rollout/"
                    "transition accounting)",
    "anchor_manifest": "cross-direction shared frozen four-anchor manifest",
    "training": "shared full-state checkpoint + optimizer surface",
}


@dataclass(frozen=True)
class SharedRuntimeBundle:
    """All five consume-only slots; every slot EMPTY by default."""

    student: SharedStudentSlot = SharedStudentSlot()
    reference: SharedReferenceSlot = SharedReferenceSlot()
    probe_runner: SharedProbeRunnerSlot = SharedProbeRunnerSlot()
    anchor_manifest: SharedAnchorManifestSlot = SharedAnchorManifestSlot()
    training: SharedTrainingSlot = SharedTrainingSlot()

    def status_report(self) -> Dict[str, Dict[str, str]]:
        report: Dict[str, Dict[str, str]] = {}
        for name, slot in (("student", self.student),
                           ("reference", self.reference),
                           ("probe_runner", self.probe_runner),
                           ("anchor_manifest", self.anchor_manifest),
                           ("training", self.training)):
            report[name] = dict(asset=SLOT_ASSET_DESCRIPTIONS[name],
                                status=slot.status, detail=slot.detail)
        return report

    def missing_assets(self) -> List[str]:
        return [SLOT_ASSET_DESCRIPTIONS[name]
                for name, slot in (("student", self.student),
                                   ("reference", self.reference),
                                   ("probe_runner", self.probe_runner),
                                   ("anchor_manifest", self.anchor_manifest),
                                   ("training", self.training))
                if slot.status != STATUS_BOUND]

    def bindings_hash(self) -> str:
        return canonical_sha256(self.status_report())


def resolve_shared_runtime(bundle: Optional[SharedRuntimeBundle] = None
                           ) -> SharedRuntimeBundle:
    """Return a COMPLETE bundle or fail closed with the full missing list.

    A missing asset is never downgraded to a local stand-in on the
    production path — the entrypoint must report
    ``BLOCKED_WAITING_SHARED_RUNTIME`` and stop.
    """
    bundle = bundle or SharedRuntimeBundle()
    missing = bundle.missing_assets()
    if missing:
        raise SharedRuntimeBlocked(
            f"{C.BLOCKED_WAITING_SHARED_RUNTIME}: missing shared assets: "
            f"{missing}")
    return bundle


__all__ = [
    "STATUS_BOUND", "STATUS_EMPTY", "SharedRuntimeBlocked",
    "SharedBindingRejected", "ReferenceInitContract",
    "ReferenceBindingIdentity", "resolve_reference_binding",
    "TrainingUpdateResult", "SharedTrainingContract", "SharedStudentSlot",
    "SharedReferenceSlot", "SharedProbeRunnerSlot",
    "SharedAnchorManifestSlot", "SharedTrainingSlot",
    "SLOT_ASSET_DESCRIPTIONS", "SharedRuntimeBundle",
    "resolve_shared_runtime",
]
