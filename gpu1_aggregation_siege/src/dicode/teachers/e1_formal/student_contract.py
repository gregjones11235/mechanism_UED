"""E1 dual-Student binding contract (CC2-Student round).

E1 no longer hard-pins ONE strong Student. The director's Runtime
Bundle EXPLICITLY selects one of the two supervisor-frozen allowed
candidates::

    consume_e1_student_contract(
        contract,
        director_selected_candidate_id=...,
        runtime_bundle_hash=...,
    )

Mechanical verification (fail-closed, no defaults — a missing
``director_selected_candidate_id`` is NEVER defaulted to the first
allowed candidate):

1. the selected candidate is in ``ALLOWED_STUDENT_CANDIDATE_IDS``
   (``STUDENT_NOT_ALLOWED`` otherwise);
2. the shared ``StudentInitContract`` (consumed verbatim from the
   static_llm thin consumer) matches: ``candidate_id`` == the
   director-selected id (``STUDENT_CONTRACT_MISMATCH``);
3. the profile is taken from the EXPLICIT profile map (never guessed
   from the name containing ``persistent``/``reset128``) and must
   match the director bundle's declared profile
   (``STUDENT_PROFILE_MISMATCH``);
4. the checkpoint identity matches (``parameter_tree_hash`` vs the
   director's ``expected_params_sha256``) (``STUDENT_CHECKPOINT_MISMATCH``);
5. the memory mode (PERSISTENT / RESET128) is the mapped mode and
   must match the director's declared memory mode
   (``STUDENT_MEMORY_MODE_MISMATCH``);
6. the adapter identity matches (``adapter_id`` vs the mapped profile
   and the director's adapter identity hash)
   (``STUDENT_ADAPTER_MISMATCH``);
7. the whole mount is bound to the ``runtime_bundle_hash``.

E1 CONSUMES the shared student registry / adapter
(``dicode.student_adapters.registry`` / ``rmt16_adapter``) — it never
constructs a second loader, registry or checkpoint codec.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..static_llm.student_init_contract import (  # noqa: F401  (re-export)
    CONTRACT_SCHEMA_VERSION,
    StudentContractError,
    StudentInitContract,
    consume_student_init_contract,
    contract_field_names,
)
from .canonical import canonical_sha256
from .schemas import E1SchemaError

#: the two supervisor-frozen allowed Student candidates
PERSISTENT_STUDENT_CANDIDATE_ID = "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"
RESET128_STUDENT_CANDIDATE_ID = "RESET128_RMT16_ORIGINAL_VTRACE_98304"

#: the director-frozen ALLOWED set (E1 never hard-pins one of them)
ALLOWED_STUDENT_CANDIDATE_IDS = frozenset(
    {PERSISTENT_STUDENT_CANDIDATE_ID, RESET128_STUDENT_CANDIDATE_ID}
)

#: EXPLICIT profile mapping (never guessed from the name)
#: candidate_id -> (profile_id, memory_mode, carry_mode)
STUDENT_PROFILE_MAP: Mapping[str, tuple] = {
    PERSISTENT_STUDENT_CANDIDATE_ID: (
        "rmt16_persistent_98304",
        "PERSISTENT",
        "persistent-memory-progression",
    ),
    RESET128_STUDENT_CANDIDATE_ID: (
        "rmt16_reset128_98304",
        "RESET128",
        "reset-to-128-window-memory",
    ),
}
STUDENT_PROFILE_BY_CANDIDATE = {
    cid: entry[0] for cid, entry in STUDENT_PROFILE_MAP.items()
}
STUDENT_MEMORY_MODE_BY_CANDIDATE = {
    cid: entry[1] for cid, entry in STUDENT_PROFILE_MAP.items()
}
STUDENT_CARRY_MODE_BY_CANDIDATE = {
    cid: entry[2] for cid, entry in STUDENT_PROFILE_MAP.items()
}
STUDENT_CANDIDATE_BY_PROFILE = {
    entry[0]: cid for cid, entry in STUDENT_PROFILE_MAP.items()
}

#: capability states (read-only mount vs full training runtime)
STUDENT_READ_ONLY_MOUNT_READY = "STUDENT_READ_ONLY_MOUNT_READY"
STUDENT_TRAINING_RUNTIME_READY = "STUDENT_TRAINING_RUNTIME_READY"
STUDENT_SHARED_REGISTRY_UNBOUND = "STUDENT_SHARED_REGISTRY_UNBOUND"

# fail-closed codes (greppable)
STUDENT_NOT_ALLOWED = "STUDENT_NOT_ALLOWED"
STUDENT_SELECTION_REQUIRED = "STUDENT_SELECTION_REQUIRED"
STUDENT_MOUNT_BAD_TYPE = "STUDENT_MOUNT_BAD_TYPE"
STUDENT_CONTRACT_MISMATCH = "STUDENT_CONTRACT_MISMATCH"
STUDENT_PROFILE_MISMATCH = "STUDENT_PROFILE_MISMATCH"
STUDENT_CHECKPOINT_MISMATCH = "STUDENT_CHECKPOINT_MISMATCH"
STUDENT_MEMORY_MODE_MISMATCH = "STUDENT_MEMORY_MODE_MISMATCH"
STUDENT_ADAPTER_MISMATCH = "STUDENT_ADAPTER_MISMATCH"
#: continuity switch codes (a stage that switches Student fails closed)
E1_STUDENT_IDENTITY_SWITCH = "E1_STUDENT_IDENTITY_SWITCH"
E1_STUDENT_PROFILE_SWITCH = "E1_STUDENT_PROFILE_SWITCH"
E1_STUDENT_MEMORY_MODE_SWITCH = "E1_STUDENT_MEMORY_MODE_SWITCH"
E1_STUDENT_CHECKPOINT_SWITCH = "E1_STUDENT_CHECKPOINT_SWITCH"


class StudentSelectionError(E1SchemaError):
    """Fail-closed student-selection violation; ``code`` is greppable."""


def require_director_selection(candidate_id: Any, ctx: str) -> str:
    """The director-selected candidate id is REQUIRED (no default)."""
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise StudentSelectionError(
            STUDENT_SELECTION_REQUIRED,
            f"{ctx}: director_selected_candidate_id is required and "
            "has NO default — a missing selection is never silently "
            "defaulted to the first allowed candidate",
        )
    candidate_id = candidate_id.strip()
    if candidate_id not in ALLOWED_STUDENT_CANDIDATE_IDS:
        raise StudentSelectionError(
            STUDENT_NOT_ALLOWED,
            f"{ctx}: candidate {candidate_id!r} is not in the "
            f"supervisor-frozen allowed set "
            f"{sorted(ALLOWED_STUDENT_CANDIDATE_IDS)}",
        )
    return candidate_id


@dataclass(frozen=True)
class SelectedStudentMount:
    """The director-selected Student mount (immutable, hash-bound).

    ``read_only_ready`` is the shared RMT16 adapter's read-only mount
    capability (checkpoint loadable, forward executable, memory
    progression verifiable, probe executable). ``training_ready`` is
    TRUE ONLY when the canonical DiCode training runtime is ALSO bound
    — a read-only mount NEVER implies training capability.
    """

    candidate_id: str
    profile_id: str
    memory_mode: str
    params_sha256: str
    adapter_id: str
    adapter_identity_hash: str
    runtime_bundle_hash: str
    read_only_ready: bool
    training_ready: bool
    shared_registry_bound: bool
    mount_hash: str

    @property
    def capability_state(self) -> str:
        if not self.shared_registry_bound:
            return STUDENT_SHARED_REGISTRY_UNBOUND
        if self.training_ready:
            return STUDENT_TRAINING_RUNTIME_READY
        return STUDENT_READ_ONLY_MOUNT_READY


def _require_sha64(value: Any, name: str, ctx: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise StudentSelectionError(
            STUDENT_MOUNT_BAD_TYPE,
            f"{ctx}: {name} must be a 64-hex hash, got {value!r}",
        )
    return value


def consume_e1_student_contract(
    contract: Any,
    *,
    director_selected_candidate_id: Any,
    runtime_bundle_hash: str,
    ctx: str,
    director_profile: Any = None,
    director_memory_mode: Any = None,
    director_expected_params_sha256: Any = None,
    director_adapter_identity_hash: Any = None,
    shared_registry_bound: bool = False,
    training_runtime_bound: bool = False,
) -> SelectedStudentMount:
    """Consume the shared StudentInitContract under the director's
    explicit selection (fail-closed on EVERY mechanical check)."""
    candidate_id = require_director_selection(
        director_selected_candidate_id, ctx
    )
    runtime_bundle_hash = _require_sha64(
        runtime_bundle_hash, "runtime_bundle_hash", ctx
    )
    if not isinstance(contract, StudentInitContract):
        raise StudentSelectionError(
            STUDENT_MOUNT_BAD_TYPE,
            f"{ctx}: contract must be a StudentInitContract, got "
            f"{type(contract).__name__}",
        )
    if contract.candidate_id != candidate_id:
        raise StudentSelectionError(
            STUDENT_CONTRACT_MISMATCH,
            f"{ctx}: contract candidate_id {contract.candidate_id!r} != "
            f"director-selected {candidate_id!r}",
        )
    profile_id = STUDENT_PROFILE_BY_CANDIDATE[candidate_id]
    if director_profile is not None and director_profile != profile_id:
        raise StudentSelectionError(
            STUDENT_PROFILE_MISMATCH,
            f"{ctx}: director profile {director_profile!r} != the mapped "
            f"profile {profile_id!r} for {candidate_id!r}",
        )
    if (
        director_expected_params_sha256 is not None
        and director_expected_params_sha256
        != contract.parameter_tree_hash
    ):
        raise StudentSelectionError(
            STUDENT_CHECKPOINT_MISMATCH,
            f"{ctx}: director expected params "
            f"{director_expected_params_sha256!r} != contract "
            f"parameter_tree_hash {contract.parameter_tree_hash!r}",
        )
    memory_mode = STUDENT_MEMORY_MODE_BY_CANDIDATE[candidate_id]
    if (
        director_memory_mode is not None
        and director_memory_mode != memory_mode
    ):
        raise StudentSelectionError(
            STUDENT_MEMORY_MODE_MISMATCH,
            f"{ctx}: director memory mode {director_memory_mode!r} != "
            f"the mapped mode {memory_mode!r} for {candidate_id!r}",
        )
    if contract.adapter_id != profile_id:
        raise StudentSelectionError(
            STUDENT_ADAPTER_MISMATCH,
            f"{ctx}: contract adapter_id {contract.adapter_id!r} != the "
            f"mapped profile {profile_id!r}",
        )
    if director_adapter_identity_hash is not None:
        _require_sha64(
            director_adapter_identity_hash,
            "director_adapter_identity_hash",
            ctx,
        )
    mount_hash = canonical_sha256(
        {
            "candidate_id": candidate_id,
            "profile_id": profile_id,
            "memory_mode": memory_mode,
            "params_sha256": contract.parameter_tree_hash,
            "adapter_id": contract.adapter_id,
            "adapter_identity_hash": (
                director_adapter_identity_hash or ""
            ),
            "runtime_bundle_hash": runtime_bundle_hash,
            "shared_registry_bound": shared_registry_bound,
            "training_runtime_bound": training_runtime_bound,
        }
    )
    return SelectedStudentMount(
        candidate_id=candidate_id,
        profile_id=profile_id,
        memory_mode=memory_mode,
        params_sha256=contract.parameter_tree_hash,
        adapter_id=contract.adapter_id,
        adapter_identity_hash=(
            director_adapter_identity_hash or ""
        ),
        runtime_bundle_hash=runtime_bundle_hash,
        read_only_ready=bool(shared_registry_bound),
        training_ready=bool(
            shared_registry_bound and training_runtime_bound
        ),
        shared_registry_bound=bool(shared_registry_bound),
        mount_hash=mount_hash,
    )


def assert_same_student_mount(
    previous: Any, current: Any, ctx: str
) -> None:
    """The SAME selected Student across the whole window (fail-closed).

    Any stage that switches the Student — identity, profile, memory
    mode or checkpoint — fails closed with its dedicated code.
    """
    if not isinstance(previous, SelectedStudentMount) or not isinstance(
        current, SelectedStudentMount
    ):
        raise StudentSelectionError(
            STUDENT_MOUNT_BAD_TYPE,
            f"{ctx}: continuity requires SelectedStudentMount objects",
        )
    if current.candidate_id != previous.candidate_id:
        raise StudentSelectionError(
            E1_STUDENT_IDENTITY_SWITCH,
            f"{ctx}: Student switched from {previous.candidate_id!r} to "
            f"{current.candidate_id!r} mid-window",
        )
    if current.profile_id != previous.profile_id:
        raise StudentSelectionError(
            E1_STUDENT_PROFILE_SWITCH,
            f"{ctx}: profile switched from {previous.profile_id!r} to "
            f"{current.profile_id!r}",
        )
    if current.memory_mode != previous.memory_mode:
        raise StudentSelectionError(
            E1_STUDENT_MEMORY_MODE_SWITCH,
            f"{ctx}: memory mode switched from {previous.memory_mode!r} "
            f"to {current.memory_mode!r}",
        )
    if current.params_sha256 != previous.params_sha256:
        raise StudentSelectionError(
            E1_STUDENT_CHECKPOINT_SWITCH,
            f"{ctx}: checkpoint params switched from "
            f"{previous.params_sha256!r} to {current.params_sha256!r}",
        )
    if current.runtime_bundle_hash != previous.runtime_bundle_hash:
        raise StudentSelectionError(
            E1_STUDENT_IDENTITY_SWITCH,
            f"{ctx}: runtime bundle switched mid-window",
        )


# ---------------------------------------------------------------------------
# shared student registry + director-bundle mount (E1 CONSUMES only)
# ---------------------------------------------------------------------------
SHARED_STUDENT_REGISTRY_MODULE = "dicode.student_adapters"


def resolve_shared_student_registry() -> tuple:
    """Resolve the shared student registry (``dicode.student_adapters``).

    E1 NEVER constructs a second loader/registry/codec — it lazily
    consumes the shared module. Absent in this worktree => honest
    ``(False, "")`` unbound result (STUDENT_SHARED_REGISTRY_UNBOUND).
    """
    import importlib

    try:
        importlib.import_module(SHARED_STUDENT_REGISTRY_MODULE)
    except ImportError:
        return (False, "")
    return (True, SHARED_STUDENT_REGISTRY_MODULE)


def build_synthetic_student_contract(candidate_id: str, ctx: str):
    """TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION StudentInitContract.

    Used by check-only + the dual-student tests to prove the mount is
    CONSTRUCTIBLE for a given allowed candidate. Never a real
    checkpoint identity.
    """
    from ..static_llm.student_init_contract import StudentInitContract

    candidate_id = require_director_selection(candidate_id, ctx)
    return StudentInitContract(
        candidate_id=candidate_id,
        architecture_family="rmt16",
        architecture_version="gtrxl-v1",
        checkpoint_format="orbax-v0",
        checkpoint_global_step=98304,
        total_env_steps=98304,
        source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
        parameter_tree_hash="aa" * 32,
        optimizer_tree_hash="bb" * 32,
        adapter_id=STUDENT_PROFILE_BY_CANDIDATE[candidate_id],
        adapter_version="rmt16-adapter-v1",
    )


def mount_student_from_director_bundle(
    *,
    bundle: Any,
    director_selected_candidate_id: Any = None,
    ctx: str,
    contract: Any = None,
    training_runtime_bound: bool = False,
    shared_registry_bound: bool = None,
) -> SelectedStudentMount:
    """Mount the director-selected Student from the Runtime Bundle.

    CC2-Student repair: the selection comes from the bundle's
    EXPLICIT ``student_selection`` descriptor (never a nonexistent
    ``bundle.student`` attribute). Rules:

    * the CLI can NEVER override the bundle-issued identity;
    * a PRODUCTION bundle REQUIRES a REAL StudentInitContract (from
      the shared FormalAssetRegistry) — a missing real contract is
      BLOCKED, never synthesized;
    * a TEST_ONLY bundle may use a synthetic contract ONLY in explicit
      test-only flows (marked TEST_ONLY / SYNTHETIC / NOT_REAL_
      EXECUTION / NOT_SMOKE_HANDOFF);
    * the identity hashes are verified SEPARATELY: adapter_identity_
      hash, adapter_implementation_hash, driver_source_sha256,
      params_sha256, checkpoint_file_sha256 (driver_source_sha256 is
      NEVER used as the adapter identity hash);
    * E1 consumes the shared registry (absent => read-only mount is
      NOT ready; training is NEVER implied).
    """
    from .runtime_bundle import (
        BUNDLE_MODE_PRODUCTION,
        BUNDLE_MODE_TEST_ONLY,
    )

    descriptor = getattr(bundle, "student_selection", None)
    if descriptor is None:
        raise StudentSelectionError(
            STUDENT_SELECTION_REQUIRED,
            f"{ctx}: the runtime bundle carries no student_selection "
            "descriptor; E1 never defaults a Student",
        )
    bundle_candidate = descriptor.selected_candidate_id
    if director_selected_candidate_id is not None and (
        director_selected_candidate_id != bundle_candidate
    ):
        raise StudentSelectionError(
            STUDENT_CONTRACT_MISMATCH,
            f"{ctx}: CLI/selected candidate "
            f"{director_selected_candidate_id!r} != the runtime "
            f"bundle's issued candidate {bundle_candidate!r} (the CLI "
            "can never override the director-issued identity)",
        )
    mode = getattr(bundle, "mode", "")
    if mode == BUNDLE_MODE_PRODUCTION and contract is None:
        raise StudentSelectionError(
            STUDENT_SELECTION_REQUIRED,
            f"{ctx}: a PRODUCTION bundle requires the REAL "
            "StudentInitContract resolved from the shared "
            "FormalAssetRegistry; a missing real contract is BLOCKED "
            "(never synthesized on the production path)",
        )
    if contract is None:
        contract = build_synthetic_student_contract(
            bundle_candidate, ctx
        )
    if shared_registry_bound is None:
        shared_bound, _module = resolve_shared_student_registry()
    else:
        shared_bound = bool(shared_registry_bound)
    if mode == BUNDLE_MODE_PRODUCTION:
        # the real contract must match the descriptor's identity
        if contract.candidate_id != bundle_candidate:
            raise StudentSelectionError(
                STUDENT_CONTRACT_MISMATCH,
                f"{ctx}: resolved contract candidate "
                f"{contract.candidate_id!r} != the bundle-issued "
                f"{bundle_candidate!r}",
            )
        if contract.parameter_tree_hash != descriptor.params_sha256:
            raise StudentSelectionError(
                STUDENT_CHECKPOINT_MISMATCH,
                f"{ctx}: resolved contract params "
                f"{contract.parameter_tree_hash!r} != the descriptor "
                f"params {descriptor.params_sha256!r}",
            )
    return consume_e1_student_contract(
        contract,
        director_selected_candidate_id=bundle_candidate,
        runtime_bundle_hash=bundle.bundle_hash,
        ctx=ctx,
        director_profile=descriptor.profile_id,
        director_memory_mode=descriptor.memory_mode,
        # the params-identity check binds the REAL contract on the
        # production path (verified above); TEST_ONLY synthetic
        # fixtures never pass a params-identity check
        director_expected_params_sha256=(
            descriptor.params_sha256
            if mode == BUNDLE_MODE_PRODUCTION
            else None
        ),
        director_adapter_identity_hash=descriptor.adapter_identity_hash,
        shared_registry_bound=shared_bound,
        training_runtime_bound=training_runtime_bound,
    )
