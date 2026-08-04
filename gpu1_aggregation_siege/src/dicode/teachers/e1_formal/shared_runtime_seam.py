"""Round-3 P0-5: shared-runtime seam (resolution ONLY, never invention).

E1 consumes the future CC4 shared runtime exclusively through this
seam. Eight shared contracts are resolved LAZILY, one resolver per
contract::

    StudentIdentity, StudentAdapter, ReferenceIdentity,
    ReferenceAdapter, AnchorManifest, FormalAssetRegistry,
    CandidateProbeResult, FullStateCheckpoint

While a contract's shared module/attribute is absent (this whole
round), its resolution is ``SeamResolution(bound=False,
code=BLOCKED_WAITING_SHARED_RUNTIME_<CONTRACT>)`` — honest, greppable,
never guessed.

Discipline (hard constraints):
* the seam ONLY resolves; it NEVER constructs, mints, loads or
  disguises a shared object; a fixture identity can never pose as a
  real one through this surface;
* NO second checkpoint loader / Student registry / Reference registry /
  anchor manifest / formal asset registry / probe minting is created
  here or anywhere in E1;
* the existing E1-side contract CONSUMERS stay exactly as they are:
  ``StudentInitContract`` (pinned candidate id),
  ``ReferenceIdentityContract`` (supervisor-frozen identity),
  the shared anchor manifest consumer (DRAFT_UNFROZEN),
  ``CandidateEvalAdapterRegistry`` + ``DualProbeResult`` (the C15
  REUSE-certification mint path). When the shared
  CandidateProbeResult contract lands, ``DualProbeResult`` becomes an
  E1-internal evidence object — documented, not re-invented.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

#: base block code (the generic status) and per-contract codes
BLOCKED_WAITING_SHARED_RUNTIME = "BLOCKED_WAITING_SHARED_RUNTIME"
BLOCKED_WAITING_SHARED_RUNTIME_STUDENT_IDENTITY = (
    "BLOCKED_WAITING_SHARED_RUNTIME_STUDENT_IDENTITY"
)
BLOCKED_WAITING_SHARED_RUNTIME_STUDENT_ADAPTER = (
    "BLOCKED_WAITING_SHARED_RUNTIME_STUDENT_ADAPTER"
)
BLOCKED_WAITING_SHARED_RUNTIME_REFERENCE_IDENTITY = (
    "BLOCKED_WAITING_SHARED_RUNTIME_REFERENCE_IDENTITY"
)
BLOCKED_WAITING_SHARED_RUNTIME_REFERENCE_ADAPTER = (
    "BLOCKED_WAITING_SHARED_RUNTIME_REFERENCE_ADAPTER"
)
BLOCKED_WAITING_SHARED_RUNTIME_ANCHOR_MANIFEST = (
    "BLOCKED_WAITING_SHARED_RUNTIME_ANCHOR_MANIFEST"
)
BLOCKED_WAITING_SHARED_RUNTIME_FORMAL_ASSET_REGISTRY = (
    "BLOCKED_WAITING_SHARED_RUNTIME_FORMAL_ASSET_REGISTRY"
)
BLOCKED_WAITING_SHARED_RUNTIME_CANDIDATE_PROBE_RESULT = (
    "BLOCKED_WAITING_SHARED_RUNTIME_CANDIDATE_PROBE_RESULT"
)
BLOCKED_WAITING_SHARED_RUNTIME_FULL_STATE_CHECKPOINT = (
    "BLOCKED_WAITING_SHARED_RUNTIME_FULL_STATE_CHECKPOINT"
)

#: the eight shared contracts, in canonical order
SHARED_CONTRACTS = (
    "StudentIdentity",
    "StudentAdapter",
    "ReferenceIdentity",
    "ReferenceAdapter",
    "AnchorManifest",
    "FormalAssetRegistry",
    "CandidateProbeResult",
    "FullStateCheckpoint",
)

#: the future home of the shared runtime (CC4). The module does NOT
#: exist in this worktree; resolvers attempt the lazy import and
#: report honestly. E1 never vendors a substitute.
SHARED_RUNTIME_MODULE = "dicode.shared_runtime"

#: contract -> attribute name on the future shared runtime module
_CONTRACT_ATTRIBUTES = {
    "StudentIdentity": "student_identity",
    "StudentAdapter": "student_adapter",
    "ReferenceIdentity": "reference_identity",
    "ReferenceAdapter": "reference_adapter",
    "AnchorManifest": "anchor_manifest",
    "FormalAssetRegistry": "formal_asset_registry",
    "CandidateProbeResult": "candidate_probe_result",
    "FullStateCheckpoint": "full_state_checkpoint",
}

_CONTRACT_CODES = {
    "StudentIdentity": BLOCKED_WAITING_SHARED_RUNTIME_STUDENT_IDENTITY,
    "StudentAdapter": BLOCKED_WAITING_SHARED_RUNTIME_STUDENT_ADAPTER,
    "ReferenceIdentity": BLOCKED_WAITING_SHARED_RUNTIME_REFERENCE_IDENTITY,
    "ReferenceAdapter": BLOCKED_WAITING_SHARED_RUNTIME_REFERENCE_ADAPTER,
    "AnchorManifest": BLOCKED_WAITING_SHARED_RUNTIME_ANCHOR_MANIFEST,
    "FormalAssetRegistry": BLOCKED_WAITING_SHARED_RUNTIME_FORMAL_ASSET_REGISTRY,
    "CandidateProbeResult": BLOCKED_WAITING_SHARED_RUNTIME_CANDIDATE_PROBE_RESULT,
    "FullStateCheckpoint": BLOCKED_WAITING_SHARED_RUNTIME_FULL_STATE_CHECKPOINT,
}


@dataclass(frozen=True)
class SeamResolution:
    """One contract's resolution state (bound/unbound + honest code).

    Deliberately carries NO object reference: the seam answers
    WHETHER the shared runtime provides the contract; obtaining the
    real object is the shared runtime's own surface. This keeps the
    seam incapable of minting or disguising shared identities.
    """

    contract: str
    bound: bool
    code: str  # "" when bound, else BLOCKED_WAITING_SHARED_RUNTIME_<CONTRACT>
    detail: str


def _resolve(contract: str) -> SeamResolution:
    code = _CONTRACT_CODES[contract]
    attribute = _CONTRACT_ATTRIBUTES[contract]
    try:
        module = importlib.import_module(SHARED_RUNTIME_MODULE)
    except ImportError:
        return SeamResolution(
            contract=contract,
            bound=False,
            code=code,
            detail=(
                f"shared runtime module {SHARED_RUNTIME_MODULE!r} is not "
                "importable in this worktree; waiting for the CC4 shared "
                "runtime (E1 never constructs a substitute)"
            ),
        )
    if getattr(module, attribute, None) is None:
        return SeamResolution(
            contract=contract,
            bound=False,
            code=code,
            detail=(
                f"shared runtime module {SHARED_RUNTIME_MODULE!r} exists "
                f"but provides no {attribute!r}; waiting for the CC4 "
                "shared runtime contract"
            ),
        )
    return SeamResolution(
        contract=contract,
        bound=True,
        code="",
        detail=(
            f"{SHARED_RUNTIME_MODULE}.{attribute} is present; the real "
            "shared object is obtained from the shared runtime itself"
        ),
    )


def resolve_student_identity() -> SeamResolution:
    return _resolve("StudentIdentity")


def resolve_student_adapter() -> SeamResolution:
    return _resolve("StudentAdapter")


def resolve_reference_identity() -> SeamResolution:
    return _resolve("ReferenceIdentity")


def resolve_reference_adapter() -> SeamResolution:
    return _resolve("ReferenceAdapter")


def resolve_anchor_manifest() -> SeamResolution:
    return _resolve("AnchorManifest")


def resolve_formal_asset_registry() -> SeamResolution:
    return _resolve("FormalAssetRegistry")


def resolve_candidate_probe_result() -> SeamResolution:
    return _resolve("CandidateProbeResult")


def resolve_full_state_checkpoint() -> SeamResolution:
    return _resolve("FullStateCheckpoint")


_RESOLVERS: Tuple[Tuple[str, Callable[[], SeamResolution]], ...] = (
    ("StudentIdentity", resolve_student_identity),
    ("StudentAdapter", resolve_student_adapter),
    ("ReferenceIdentity", resolve_reference_identity),
    ("ReferenceAdapter", resolve_reference_adapter),
    ("AnchorManifest", resolve_anchor_manifest),
    ("FormalAssetRegistry", resolve_formal_asset_registry),
    ("CandidateProbeResult", resolve_candidate_probe_result),
    ("FullStateCheckpoint", resolve_full_state_checkpoint),
)


def resolve_all_shared_runtime() -> Dict[str, SeamResolution]:
    """Resolve ALL eight contracts (canonical order preserved).

    Pure resolution: this performs no construction, no minting, no
    checkpoint loading and no I/O beyond the lazy import attempt.
    """
    return {contract: resolver() for contract, resolver in _RESOLVERS}
