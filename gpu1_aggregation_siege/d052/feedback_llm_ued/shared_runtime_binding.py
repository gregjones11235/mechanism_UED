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

P0-7 (CC3 follow-up audit): ``bindings_hash`` folds the REAL ASSET
IDENTITIES — the registry-issued Student / Reference / ProbeRunner /
AnchorManifest / Training identities plus the formal asset registry's own
identity — NOT status strings. Two bundles bound to different real assets
can therefore never collide on the same bindings hash, and a missing asset
folds the explicit ``ABSENT_NOT_REGISTRY_ISSUED`` sentinel (never a silent
stand-in). The ProbeRunner and Training slots additionally refuse assets
that do not expose a registry-issued sha256 identity; absence still keeps
every slot EMPTY and ``resolve_shared_runtime`` fail-closed on
``BLOCKED_WAITING_SHARED_RUNTIME``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
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
    FullStateRoundTripResult,
    StudentBindingIdentity,
    StudentInitContract,
    resolve_student_binding,
)
from d052.schemas.common import is_sha256_hex

#: slot statuses
STATUS_BOUND = "BOUND"
STATUS_EMPTY = C.BLOCKED_WAITING_SHARED_RUNTIME

#: P0-7: folded into ``bindings_hash`` for every asset that is NOT bound.
#: The sentinel is explicit and unmistakable — an absent asset can never
#: hash-collide with a bound one, and its absence is auditable in the
#: identity map itself.
SLOT_ABSENT_IDENTITY = "ABSENT_NOT_REGISTRY_ISSUED"


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
    """Identity stamped onto feedback records alongside the Student's.

    P0-16: ``declared_identity_hash`` is the DIRECTOR-issued canonical
    Reference identity (consume-only — authenticated by the signed
    Runtime Bundle). When declared it is authoritative; otherwise the
    identity is recomputed from the fields (historical behavior).
    """

    candidate_id: str
    parameter_tree_hash: str
    checkpoint_global_step: int
    provenance_label: str
    declared_identity_hash: str = ""

    @property
    def identity_hash(self) -> str:
        if self.declared_identity_hash:
            return self.declared_identity_hash
        return canonical_sha256(dict(
            candidate_id=self.candidate_id,
            parameter_tree_hash=self.parameter_tree_hash,
            checkpoint_global_step=self.checkpoint_global_step,
            provenance_label=self.provenance_label))


def resolve_reference_binding(contract: Optional[ReferenceInitContract],
                              *,
                              declared_identity_hash: str = ""
                              ) -> ReferenceBindingIdentity:
    """Validate an explicitly injected shared Reference contract.

    Fail-closed ladder:
      * no contract           -> REFERENCE_INIT_CONTRACT_MISSING
      * incomplete identity   -> REFERENCE_IDENTITY_INCOMPLETE

    P0-16: ``declared_identity_hash`` is the DIRECTOR-issued canonical
    Reference identity (consume-only, authenticated by the signed Runtime
    Bundle); when declared it is authoritative.
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
    if declared_identity_hash and not is_sha256_hex(declared_identity_hash):
        raise SharedBindingRejected(
            f"REFERENCE_IDENTITY_INCOMPLETE: declared_identity_hash must be "
            f"a sha256 hex string, got {declared_identity_hash!r}")
    return ReferenceBindingIdentity(
        candidate_id=candidate_id,
        parameter_tree_hash=param_hash,
        checkpoint_global_step=step,
        provenance_label="SHARED_REFERENCE_INIT_CONTRACT",
        declared_identity_hash=declared_identity_hash)


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
    round-trip around it, PROVEN by the director-verifier's immutable
    :class:`FullStateRoundTripResult` attestation (P0-11: "save hash
    differs + load called" is NOT a round-trip). The owner guarantees
    full-state semantics; this direction never re-implements an optimizer
    or a checkpoint codec.
    """

    def run_one_optimizer_update(self, *, window: int,
                                 batch_candidate_ids: List[str]
                                 ) -> TrainingUpdateResult: ...

    def save_checkpoint(self, *, tag: str) -> str: ...

    def load_checkpoint(self, *, checkpoint_hash: str) -> None: ...

    def verify_full_state_round_trip(self, *, window: int,
                                     checkpoint_hash: str
                                     ) -> FullStateRoundTripResult: ...


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

    def slot_identity(self) -> str:
        """P0-7: the REAL asset identity folded into ``bindings_hash`` —
        the resolved Student binding's content identity hash (registry-
        issued candidate identity), never a status string."""
        if self.status != STATUS_BOUND:
            return SLOT_ABSENT_IDENTITY
        identity = self.binding.identity_hash if self.binding else ""
        if not identity:
            raise SharedBindingRejected(
                "SLOT_BOUND_WITHOUT_REGISTRY_IDENTITY: the student slot is "
                "BOUND but carries no resolved binding identity — a bound "
                "slot without an identity is a smuggled stand-in and is "
                "refused fail-closed")
        return identity


@dataclass(frozen=True)
class SharedReferenceSlot:
    status: str = STATUS_EMPTY
    detail: str = "shared ReferenceAdapter absent from this worktree"
    binding: Optional[ReferenceBindingIdentity] = None

    def bind(self, contract: ReferenceInitContract, *,
             declared_identity_hash: str = "") -> "SharedReferenceSlot":
        binding = resolve_reference_binding(
            contract, declared_identity_hash=declared_identity_hash)
        return SharedReferenceSlot(status=STATUS_BOUND,
                                   detail=binding.provenance_label,
                                   binding=binding)

    def slot_identity(self) -> str:
        """P0-7: the REAL asset identity folded into ``bindings_hash`` —
        the resolved Reference binding's content identity hash (registry-
        issued candidate identity), never a status string."""
        if self.status != STATUS_BOUND:
            return SLOT_ABSENT_IDENTITY
        identity = self.binding.identity_hash if self.binding else ""
        if not identity:
            raise SharedBindingRejected(
                "SLOT_BOUND_WITHOUT_REGISTRY_IDENTITY: the reference slot "
                "is BOUND but carries no resolved binding identity — a "
                "bound slot without an identity is a smuggled stand-in and "
                "is refused fail-closed")
        return identity


@dataclass(frozen=True)
class SharedProbeRunnerSlot:
    status: str = STATUS_EMPTY
    detail: str = "shared CandidateProbeRunner absent from this worktree"
    runner: Optional[object] = None
    #: P0-7: the registry-issued identity of the bound runner (sha256 hex,
    #: issued by the formal asset registry — never derived locally)
    registry_identity: str = ""

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
        #: P0-7: registry-issued identity only — a runner that does not
        #: declare the sha256 identity the formal asset registry issued
        #: for it cannot be bound (direction two never derives one)
        registry_identity = getattr(runner, "registry_identity", "")
        if (not isinstance(registry_identity, str)
                or not is_sha256_hex(registry_identity)):
            raise SharedBindingRejected(
                "PROBE_RUNNER_REGISTRY_IDENTITY_MISSING: the shared runner "
                "must expose ``registry_identity`` — the sha256 hex "
                "identity issued for it by the formal asset registry, got "
                f"{registry_identity!r}")
        return SharedProbeRunnerSlot(status=STATUS_BOUND,
                                     detail=runner_id,
                                     runner=runner,
                                     registry_identity=registry_identity)

    def bind_director_declared(self, *, registry_identity: str,
                               detail: str) -> "SharedProbeRunnerSlot":
        """P0-16: bind a DIRECTOR-DECLARED runner identity (the director's
        Runtime Bundle records the registry identity; the runner OBJECT is
        injected by the director at smoke time). Direction two records the
        identity — it never fabricates a runner object."""
        if not is_sha256_hex(registry_identity):
            raise SharedBindingRejected(
                "PROBE_RUNNER_DIRECTOR_DECLARED_IDENTITY_INVALID: "
                f"{registry_identity!r}")
        return SharedProbeRunnerSlot(status=STATUS_BOUND, detail=detail,
                                     runner=None,
                                     registry_identity=registry_identity)

    def slot_identity(self) -> str:
        """P0-7: the REAL asset identity folded into ``bindings_hash`` —
        the runner's registry-issued identity, never a status string."""
        if self.status != STATUS_BOUND:
            return SLOT_ABSENT_IDENTITY
        if not self.registry_identity:
            raise SharedBindingRejected(
                "SLOT_BOUND_WITHOUT_REGISTRY_IDENTITY: the probe-runner "
                "slot is BOUND but carries no registry identity — a bound "
                "slot without an identity is a smuggled stand-in and is "
                "refused fail-closed")
        return self.registry_identity


@dataclass(frozen=True)
class SharedAnchorManifestSlot:
    status: str = STATUS_EMPTY
    detail: str = ("no cross-direction shared frozen anchor manifest exists "
                   "in this worktree")
    anchor_ids: Tuple[str, ...] = ()
    manifest_hash: str = ""
    binding_label: str = SCAFFOLD_PLACEHOLDER_NOT_SHARED
    #: the bound manifest object itself, kept so a production entrypoint can
    #: hand the SAME verified manifest to the controller's anchor seam (no
    #: re-derivation, no second parse)
    manifest: Optional[object] = None

    def bind(self, manifest: object) -> "SharedAnchorManifestSlot":
        #: wraps the EXISTING AnchorManifestSource seam (manifest_hash
        #: recomputed and compared; anything unfrozen fails closed). Both
        #: refusal shapes — AnchorManifestBlocked (absent/unfrozen) and
        #: ValueError (ANCHOR_MANIFEST_HASH_MISMATCH tamper) — are wrapped
        #: into the typed SharedBindingRejected ladder.
        if isinstance(manifest, Mapping):
            manifest = SharedAnchorManifest(**manifest)
        source = AnchorManifestSource(manifest=manifest)
        try:
            anchor_ids = source.resolve()
        except (AnchorManifestBlocked, ValueError) as exc:
            raise SharedBindingRejected(
                f"ANCHOR_MANIFEST_BINDING_REJECTED: {exc}") from exc
        manifest_hash = str(getattr(manifest, "manifest_hash", ""))
        if not is_sha256_hex(manifest_hash):
            raise SharedBindingRejected(
                "ANCHOR_MANIFEST_REGISTRY_IDENTITY_MISSING: the bound "
                "manifest carries no legal sha256 manifest_hash identity "
                f"({manifest_hash!r}) — refused fail-closed")
        return SharedAnchorManifestSlot(
            status=STATUS_BOUND,
            detail="shared frozen anchor manifest bound",
            anchor_ids=tuple(anchor_ids),
            manifest_hash=manifest_hash,
            binding_label=SHARED_MANIFEST_BOUND_LABEL,
            manifest=manifest)

    def slot_identity(self) -> str:
        """P0-7: the REAL asset identity folded into ``bindings_hash`` —
        the recomputed-and-verified frozen manifest hash (the registry
        identity of the anchor set), never a status string."""
        if self.status != STATUS_BOUND:
            return SLOT_ABSENT_IDENTITY
        if not self.manifest_hash:
            raise SharedBindingRejected(
                "SLOT_BOUND_WITHOUT_REGISTRY_IDENTITY: the anchor-manifest "
                "slot is BOUND but carries no manifest identity — a bound "
                "slot without an identity is a smuggled stand-in and is "
                "refused fail-closed")
        return self.manifest_hash


@dataclass(frozen=True)
class SharedTrainingSlot:
    status: str = STATUS_EMPTY
    detail: str = ("shared full-state checkpoint + optimizer surface absent "
                   "from this worktree")
    contract: Optional[SharedTrainingContract] = None
    #: P0-7: the registry-issued identity of the bound training/checkpoint
    #: surface (sha256 hex, issued by the formal asset registry)
    registry_identity: str = ""

    def bind(self, contract: SharedTrainingContract) -> "SharedTrainingSlot":
        #: P0-11: the director-verifier attestation surface is part of the
        #: contract — a training surface without it cannot prove a
        #: checkpoint round-trip and is refused
        for method in ("run_one_optimizer_update", "save_checkpoint",
                       "load_checkpoint", "verify_full_state_round_trip"):
            if not callable(getattr(contract, method, None)):
                raise SharedBindingRejected(
                    f"SHARED_TRAINING_CONTRACT_INCOMPLETE: missing callable "
                    f"{method!r}")
        #: P0-7: registry-issued identity only — a training surface that
        #: does not declare the sha256 identity the formal asset registry
        #: issued for it cannot be bound (direction two never derives one)
        registry_identity = getattr(contract, "registry_identity", "")
        if (not isinstance(registry_identity, str)
                or not is_sha256_hex(registry_identity)):
            raise SharedBindingRejected(
                "SHARED_TRAINING_REGISTRY_IDENTITY_MISSING: the shared "
                "training contract must expose ``registry_identity`` — the "
                "sha256 hex identity issued for it by the formal asset "
                f"registry, got {registry_identity!r}")
        return SharedTrainingSlot(status=STATUS_BOUND,
                                  detail="shared training contract bound",
                                  contract=contract,
                                  registry_identity=registry_identity)

    def bind_director_declared(self, *, registry_identity: str,
                               detail: str) -> "SharedTrainingSlot":
        """P0-16: bind a DIRECTOR-DECLARED training runtime identity (the
        director's Runtime Bundle records the CanonicalDiCodeOneUpdateRuntime
        registry identity; the object is injected by the director at smoke
        time). Direction two never implements an optimizer."""
        if not is_sha256_hex(registry_identity):
            raise SharedBindingRejected(
                "SHARED_TRAINING_DIRECTOR_DECLARED_IDENTITY_INVALID: "
                f"{registry_identity!r}")
        return SharedTrainingSlot(status=STATUS_BOUND, detail=detail,
                                  contract=None,
                                  registry_identity=registry_identity)

    def slot_identity(self) -> str:
        """P0-7: the REAL asset identity folded into ``bindings_hash`` —
        the training surface's registry-issued identity, never a status
        string."""
        if self.status != STATUS_BOUND:
            return SLOT_ABSENT_IDENTITY
        if not self.registry_identity:
            raise SharedBindingRejected(
                "SLOT_BOUND_WITHOUT_REGISTRY_IDENTITY: the training slot "
                "is BOUND but carries no registry identity — a bound slot "
                "without an identity is a smuggled stand-in and is refused "
                "fail-closed")
        return self.registry_identity


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
    """All five consume-only slots; every slot EMPTY by default.

    P0-7: ``bindings_hash`` folds ``asset_identities()`` — the REAL
    registry-issued asset identities — NOT the status strings of
    ``status_report()``. The formal asset registry's own identity
    (``formal_registry_identity``) is folded too; in this worktree the
    registry is absent, so it stays empty and folds the explicit ABSENT
    sentinel.
    """

    student: SharedStudentSlot = SharedStudentSlot()
    reference: SharedReferenceSlot = SharedReferenceSlot()
    probe_runner: SharedProbeRunnerSlot = SharedProbeRunnerSlot()
    anchor_manifest: SharedAnchorManifestSlot = SharedAnchorManifestSlot()
    training: SharedTrainingSlot = SharedTrainingSlot()
    #: P0-7: the formal asset registry's own registry-issued identity
    #: (sha256 hex). Empty while the registry is absent from this worktree
    #: — absence is folded into the bindings hash as the explicit ABSENT
    #: sentinel, never silently omitted.
    formal_registry_identity: str = ""

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

    def asset_identities(self) -> Dict[str, str]:
        """P0-7: the REAL asset identity map folded into ``bindings_hash``.

        Every entry is a REGISTRY-ISSUED asset identity (or the resolved
        content identity hash of one), never a status string: two bundles
        bound to DIFFERENT real assets can never collide, and an absent
        asset is folded as the explicit ``SLOT_ABSENT_IDENTITY`` sentinel
        — absence is auditable in the map itself. The six entries: the
        five consume-only slots plus the formal asset registry's own
        identity (empty here — the registry is absent from this worktree).
        """
        return dict(
            student=self.student.slot_identity(),
            reference=self.reference.slot_identity(),
            probe_runner=self.probe_runner.slot_identity(),
            anchor_manifest=self.anchor_manifest.slot_identity(),
            training=self.training.slot_identity(),
            formal_registry=(self.formal_registry_identity
                             or SLOT_ABSENT_IDENTITY))

    def with_formal_registry_identity(self, identity: str
                                      ) -> "SharedRuntimeBundle":
        """Bind the formal asset registry's own registry-issued identity
        (P0-7). Fail closed: only a sha256 hex identity is accepted — the
        registry identity is issued by the registry itself and is never
        derived locally."""
        if not isinstance(identity, str) or not is_sha256_hex(identity):
            raise SharedBindingRejected(
                "FORMAL_REGISTRY_IDENTITY_INVALID: the formal asset "
                "registry identity must be a registry-issued sha256 hex "
                f"string, got {identity!r}")
        return replace(self, formal_registry_identity=identity)

    def bindings_hash(self) -> str:
        """P0-7: folds the REAL ASSET IDENTITIES (``asset_identities()``),
        NOT the status strings of ``status_report()`` — a hash over
        statuses could not distinguish bundles bound to different real
        assets."""
        return canonical_sha256(self.asset_identities())


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
    "STATUS_BOUND", "STATUS_EMPTY", "SLOT_ABSENT_IDENTITY",
    "SharedRuntimeBlocked", "SharedBindingRejected", "ReferenceInitContract",
    "ReferenceBindingIdentity", "resolve_reference_binding",
    "TrainingUpdateResult", "SharedTrainingContract", "SharedStudentSlot",
    "SharedReferenceSlot", "SharedProbeRunnerSlot",
    "SharedAnchorManifestSlot", "SharedTrainingSlot",
    "SLOT_ASSET_DESCRIPTIONS", "SharedRuntimeBundle",
    "resolve_shared_runtime",
]
