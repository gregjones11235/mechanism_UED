"""Runtime authorization grants — the ONLY channel that may enable real
capabilities on the direction-two production path (TWO_REAL_WINDOWS_READY
_FOR_AUDIT round).

Design rule: the round constants in ``constants.py`` (``REAL_LLM_CALLS_
AUTHORIZED`` etc.) stay False and are re-asserted False by the controller's
authorization-posture check; they are never hand-flipped. A production
entrypoint that wants REAL execution therefore carries an EXPLICIT
:class:`RealRuntimeAuthorization` object — constructed deliberately, never
derived from environment variables or config files — and the gate combines
it with the declared execution mode.

The grants are strictly LAYERED (a fail-closed capability hierarchy): real
training presumes real probing, real probing presumes a real EnvCoder, and
a real EnvCoder presumes a real LLM backend. Any inconsistent grant set is
refused at construction.

Even a complete grant set does NOT make the loop real by itself: the shared
runtime assets (Student/Reference adapters, shared CandidateProbeRunner,
frozen anchor manifest, checkpoint ABI, a real Craftax interpreter) must be
physically present and injected — see ``shared_runtime_binding``. This
module's :func:`assert_real_mode_servicable` is the single place that turns
an absent real LLM transport into ``REAL_MODE_BLOCKED_NO_LLM_BACKEND``
instead of a silent mock fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from d052.feedback_llm_ued import constants as C


class RuntimeAuthorizationBlocked(RuntimeError):
    """Fail-closed refusal of the runtime authorization channel."""


@dataclass(frozen=True)
class RealRuntimeAuthorization:
    """Explicit per-capability grants for a REAL production run.

    Every grant defaults to False; the hierarchy is enforced in
    ``__post_init__`` so a partially granted set can never reach the gate.
    """

    real_llm_backend: bool = False
    real_envcoder: bool = False
    real_probe: bool = False
    real_training: bool = False

    def __post_init__(self) -> None:
        if self.real_envcoder and not self.real_llm_backend:
            raise RuntimeAuthorizationBlocked(
                "INCONSISTENT_RUNTIME_GRANTS: real_envcoder requires "
                "real_llm_backend (the EnvCoder is an LLM call)")
        if self.real_probe and not (self.real_llm_backend
                                    and self.real_envcoder):
            raise RuntimeAuthorizationBlocked(
                "INCONSISTENT_RUNTIME_GRANTS: real_probe requires "
                "real_llm_backend and real_envcoder (probes consume "
                "real-coded candidate environments)")
        if self.real_training and not (self.real_llm_backend
                                       and self.real_envcoder
                                       and self.real_probe):
            raise RuntimeAuthorizationBlocked(
                "INCONSISTENT_RUNTIME_GRANTS: real_training requires every "
                "lower capability (training batches derive from real "
                "probe-selected environments)")

    def any_grant(self) -> bool:
        return (self.real_llm_backend or self.real_envcoder
                or self.real_probe or self.real_training)

    def describe(self) -> str:
        return ("RealRuntimeAuthorization(real_llm_backend="
                f"{self.real_llm_backend}, real_envcoder="
                f"{self.real_envcoder}, real_probe={self.real_probe}, "
                f"real_training={self.real_training})")


def empty_authorization() -> RealRuntimeAuthorization:
    """The default posture this round: no real capability granted."""
    return RealRuntimeAuthorization()


def assert_real_mode_servicable(*,
                                authorization: RealRuntimeAuthorization,
                                llm_transport: Optional[object],
                                missing_assets: Sequence[str] = ()) -> None:
    """Fail closed when a requested REAL run cannot actually be real.

    * real LLM granted but no transport injected ->
      ``REAL_MODE_BLOCKED_NO_LLM_BACKEND``. A silent fallback to the mock
      backend would violate NO_SILENT_FALLBACK; refusing is the only honest
      behavior.
    * shared runtime assets missing -> ``BLOCKED_WAITING_SHARED_RUNTIME``
      with the full missing list (the caller assembles it from
      ``shared_runtime_binding``).
    """
    if authorization.real_llm_backend and llm_transport is None:
        raise RuntimeAuthorizationBlocked(
            f"{C.REAL_MODE_BLOCKED_NO_LLM_BACKEND}: real LLM execution was "
            "requested (runtime grant real_llm_backend=true) but no real "
            "transport is available in this environment; the loop must NOT "
            "fall back to the mock backend and claim to be real")
    if authorization.any_grant() and missing_assets:
        raise RuntimeAuthorizationBlocked(
            f"{C.BLOCKED_WAITING_SHARED_RUNTIME}: real execution was "
            f"requested but shared runtime assets are missing: "
            f"{sorted(missing_assets)}")


__all__ = [
    "RuntimeAuthorizationBlocked",
    "RealRuntimeAuthorization",
    "empty_authorization",
    "assert_real_mode_servicable",
]
