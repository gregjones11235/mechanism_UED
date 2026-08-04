"""R4c combined fresh-process restore contract (review condition 1).

The R4 final gate requires that ONE fresh process jointly restores and
cross-verifies:

    Student params + optimizer + global step + train RNG,
    EnvState, environment RNG, wrapper state,
    policy memory / history,

and then reproduces the reference next policy step on the jointly restored
components.  An env-only restore PASS together with a checkpoint-only restore
PASS is NOT a joint proof — this module exists to make that distinction
mechanically enforceable.

This round delivers the CONTRACT ONLY: types, validation and the driver
signature ``run_combined_restore``.  The actual combined execution is pending
(policy-memory/history restore is Phase 2; the audited CC2 pkl carries
params+manifest only, so optimizer/train-rng/policy-memory are
ABSENT_IN_CHECKPOINT this round).  Until a real combined run has executed:
``COMBINED_FRESH_PROCESS_RESTORE`` MUST stay false.

Audit closure (2026-08-04): ``run_combined_restore`` executes caller-supplied
callbacks in the CURRENT process — freshness there is only a caller
obligation and component statuses are self-asserted, so it is frozen as
CONTRACT-LEVEL ONLY (``CALLBACK_DRIVER_IS_CONTRACT_ONLY``).  The mechanically
enforced production proof lives in ``fresh_process_restore``: exactly one
spawned child process, atomic PID/argv/timestamp evidence, authoritative
per-component leaf hashes, checkpoint-leaf optimizer binding and
``production_joint_pass`` as the only gate that may ever upgrade
``COMBINED_FRESH_PROCESS_RESTORE``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

from .errors import InvalidEvidenceError

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

JOINT_PROOF_DISCLAIMER = (
    "env-only restore PASS /\\ checkpoint-only restore PASS != combined "
    "fresh-process joint proof; COMBINED_FRESH_PROCESS_RESTORE stays false "
    "until run_combined_restore executes in one fresh process and every "
    "required component reports RESTORED with cross-checks green"
)

# Every component the joint proof must restore and verify.
REQUIRED_COMPONENTS = (
    "params",
    "optimizer",
    "global_step",
    "train_rng",
    "env_state",
    "env_rng",
    "wrapper_state",
    "policy_memory",
    "history",
)

# Cross-verification performed on the JOINTLY restored components.
CROSS_CHECKS = ("policy_step_next_replay",)

RESTORED_STATUSES = ("RESTORED_HASH_BOUND", "RESTORED_CROSS_VERIFIED")

# Frozen after the independent audit: the callback driver below can NEVER be
# the production proof (it runs callbacks in the current process and trusts
# self-asserted ComponentResult statuses).  Production joint proof requires
# fresh_process_restore.run_fresh_process_restore_production with verified
# ProcessEvidence; production_joint_pass is the only composition gate.
CALLBACK_DRIVER_IS_CONTRACT_ONLY = True


class ComponentStatus(str, Enum):
    RESTORED_HASH_BOUND = "RESTORED_HASH_BOUND"
    RESTORED_CROSS_VERIFIED = "RESTORED_CROSS_VERIFIED"
    ABSENT_IN_CHECKPOINT = "ABSENT_IN_CHECKPOINT"
    NOT_EXECUTED = "NOT_EXECUTED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ComponentResult:
    component: str
    status: ComponentStatus
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.component:
            raise InvalidEvidenceError("component name must be non-empty")
        if not isinstance(self.status, ComponentStatus):
            raise InvalidEvidenceError(f"component status must be ComponentStatus, "
                                       f"got {self.status!r}")


@dataclass(frozen=True)
class CombinedRestoreRequest:
    """Everything one fresh process needs to attempt the joint restore.

    All expected-* fields are FAIL-CLOSED requirements: a restore that cannot
    be hash-bound to these expectations must be reported FAILED, never
    accepted as "close enough".
    """

    encoded_bundle_ref: str
    checkpoint_path: str
    expected_candidate_id: str
    expected_params_sha256: str
    expected_file_sha256: str
    expected_env_payload_hash: str
    expected_global_step: int
    expected_memory_spec_hash: str
    # Optional components: None = not expected to exist in the artifact;
    # when the artifact claims them anyway the restorer must FAIL the
    # component (unexpected presence is also a contract violation).
    expected_optimizer_sha256: str | None = None
    expected_train_rng_replay_ref: str | None = None
    expected_wrapper_state_hash: str | None = None
    history_reference: str | None = None
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("encoded_bundle_ref", "checkpoint_path", "expected_candidate_id",
                     "expected_params_sha256", "expected_file_sha256",
                     "expected_env_payload_hash", "expected_memory_spec_hash"):
            if not getattr(self, name):
                raise InvalidEvidenceError(f"CombinedRestoreRequest.{name} is required")
        for name in ("expected_params_sha256", "expected_file_sha256"):
            if not _SHA256_RE.match(getattr(self, name)):
                raise InvalidEvidenceError(f"CombinedRestoreRequest.{name} must be a "
                                           f"64-hex sha256, got {getattr(self, name)!r}")
        for name in ("expected_optimizer_sha256",):
            value = getattr(self, name)
            if value is not None and not _SHA256_RE.match(value):
                raise InvalidEvidenceError(f"CombinedRestoreRequest.{name} must be a "
                                           f"64-hex sha256 when present")
        if int(self.expected_global_step) < 0:
            raise InvalidEvidenceError("expected_global_step must be >= 0")


@dataclass(frozen=True)
class CombinedRestoreVerdict:
    components: Mapping[str, ComponentResult]
    cross_checks: Mapping[str, ComponentResult]
    combined_pass: bool
    env_only_pass: bool
    checkpoint_only_pass: bool
    joint_proof_status: str

    def component_matrix(self) -> dict[str, str]:
        return {name: result.status.value for name, result in self.components.items()}


_ENV_SIDE = ("env_state", "env_rng", "wrapper_state")
_CHECKPOINT_SIDE = ("params", "optimizer", "global_step", "train_rng")


def _passed(result: ComponentResult | None) -> bool:
    return result is not None and result.status.value in RESTORED_STATUSES


def evaluate_verdict(components: Mapping[str, ComponentResult],
                     cross_checks: Mapping[str, ComponentResult]) -> CombinedRestoreVerdict:
    """Pure COMPONENT-COMPOSITION rule of the R4c gate.

    combined_pass requires EVERY required component and EVERY cross-check to
    be restored/verified.  env_only_pass / checkpoint_only_pass are reported
    alongside so the honest gap (one-sided passes) is always visible.

    IMPORTANT (audit closure): combined_pass here composes SELF-ASSERTED
    ComponentResult statuses only; it is NEVER a production proof by itself.
    The production joint proof additionally requires mechanically verified
    fresh-process evidence — see ``fresh_process_restore.production_joint_pass``.
    """
    missing = [c for c in REQUIRED_COMPONENTS if c not in components]
    if missing:
        raise InvalidEvidenceError(f"verdict is missing required components {missing}")
    missing_x = [c for c in CROSS_CHECKS if c not in cross_checks]
    if missing_x:
        raise InvalidEvidenceError(f"verdict is missing cross-checks {missing_x}")

    env_only = all(_passed(components[c]) for c in _ENV_SIDE)
    ckpt_only = all(_passed(components[c]) for c in _CHECKPOINT_SIDE)
    combined = (all(_passed(components[c]) for c in REQUIRED_COMPONENTS)
                and all(_passed(cross_checks[c]) for c in CROSS_CHECKS))
    if combined:
        status = "COMBINED_FRESH_PROCESS_RESTORE=true"
    elif env_only and ckpt_only:
        # env side and checkpoint side both green, but policy_memory/history
        # or a cross-check failed: one-sided passes still do NOT compose.
        status = ("COMBINED_FRESH_PROCESS_RESTORE=false (one-sided passes do NOT "
                  "compose: " + JOINT_PROOF_DISCLAIMER + ")")
    else:
        status = "COMBINED_FRESH_PROCESS_RESTORE=false (" + JOINT_PROOF_DISCLAIMER + ")"
    return CombinedRestoreVerdict(
        components=dict(components), cross_checks=dict(cross_checks),
        combined_pass=bool(combined), env_only_pass=bool(env_only),
        checkpoint_only_pass=bool(ckpt_only), joint_proof_status=status)


def run_combined_restore(
    request: CombinedRestoreRequest,
    *,
    restorers: Mapping[str, Callable[[CombinedRestoreRequest], ComponentResult]],
    cross_checkers: Mapping[str, Callable[[CombinedRestoreRequest, Mapping[str, ComponentResult]],
                                          ComponentResult]] | None = None,
) -> CombinedRestoreVerdict:
    """R4c driver signature: the JOINT restore must happen in ONE fresh process.

    ``restorers[component](request)`` must return the ComponentResult for each
    required component; ``cross_checkers[check](request, components)`` runs
    AFTER all components are restored (it receives the component map so it can
    replay policy_step on the jointly restored state).  A restorer raising is
    recorded as FAILED (fail closed); a missing restorer is NOT_EXECUTED —
    both keep combined_pass false.  This function itself performs no I/O and
    no network: the fresh-process discipline is the caller's obligation and is
    part of the contract documented here.
    """
    if not isinstance(request, CombinedRestoreRequest):
        raise InvalidEvidenceError("run_combined_restore requires CombinedRestoreRequest")
    restorers = dict(restorers or {})
    cross_checkers = dict(cross_checkers or {})

    components: dict[str, ComponentResult] = {}
    for name in REQUIRED_COMPONENTS:
        fn = restorers.get(name)
        if fn is None:
            components[name] = ComponentResult(name, ComponentStatus.NOT_EXECUTED,
                                               "no restorer registered (fail closed)")
            continue
        try:
            result = fn(request)
        except Exception as exc:  # fail closed, keep the evidence honest
            components[name] = ComponentResult(name, ComponentStatus.FAILED, repr(exc))
            continue
        if not isinstance(result, ComponentResult) or result.component != name:
            components[name] = ComponentResult(
                name, ComponentStatus.FAILED,
                f"restorer returned an invalid result: {result!r}")
            continue
        components[name] = result

    cross: dict[str, ComponentResult] = {}
    for name in CROSS_CHECKS:
        fn = cross_checkers.get(name)
        if fn is None:
            cross[name] = ComponentResult(name, ComponentStatus.NOT_EXECUTED,
                                          "no cross-checker registered (fail closed)")
            continue
        try:
            result = fn(request, dict(components))
        except Exception as exc:
            cross[name] = ComponentResult(name, ComponentStatus.FAILED, repr(exc))
            continue
        if not isinstance(result, ComponentResult) or result.component != name:
            cross[name] = ComponentResult(name, ComponentStatus.FAILED,
                                          f"cross-checker returned an invalid result: {result!r}")
            continue
        cross[name] = result

    return evaluate_verdict(components, cross)
