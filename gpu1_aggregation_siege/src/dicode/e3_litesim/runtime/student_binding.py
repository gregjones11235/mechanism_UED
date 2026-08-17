"""Student identity / params binding: prove RunState == probe == PPO params.

Every probe / data generation / PPO update must pass this guard first.  The
guard records a :class:`StudentIdentity` (student_id, student_version,
architecture_family, params_hash, checkpoint_step) and verifies BOTH

  * params-consistency (probe_params_hash == ppo_input_params_hash == current
    student params hash) — catches stale probe / params mismatch / checkpoint
    mismatch, and
  * identity-consistency (the recorded identity params_hash matches the actual
    params, and the student_id/architecture_family are present) — catches
    identity mismatch / stale identity.

Fail-closed: any violation raises before frontier selection, data generation,
LLM, or PPO can proceed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional, Tuple

from .hashing import hash_payload, hash_pytree


class BindingError(RuntimeError):
    """Student binding violation (fail closed)."""


@dataclass(frozen=True)
class StudentIdentity:
    student_id: str
    student_version: int
    architecture_family: str
    params_hash: str
    checkpoint_step: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BindingRecord:
    session_idx: int
    global_step: int
    identity: StudentIdentity
    runstate_params_hash: str
    checkpoint_params_hash: Optional[str]
    probe_params_hash: str
    ppo_params_hash: str
    binding_verified: bool
    failure_reasons: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        body = asdict(self)
        body["identity"] = self.identity.to_dict()
        body["failure_reasons"] = list(self.failure_reasons)
        return body


class StudentBindingGuard:
    """Every probe / data generation / PPO update must pass this guard first."""

    def bind(self, *, session_idx: int, global_step: int,
             identity: StudentIdentity, runstate_params: Any,
             probe_params: Any, ppo_params: Any,
             checkpoint_params: Any = None) -> BindingRecord:
        run_hash = hash_pytree(runstate_params)
        probe_hash = hash_pytree(probe_params)
        ppo_hash = hash_pytree(ppo_params)
        ckpt_hash = (hash_pytree(checkpoint_params)
                     if checkpoint_params is not None else None)

        reasons: list = []
        if run_hash != probe_hash:
            reasons.append("stale_probe")
        if run_hash != ppo_hash:
            reasons.append("params_mismatch")
        if ckpt_hash is not None and ckpt_hash != run_hash:
            reasons.append("checkpoint_mismatch")
        if not identity.student_id:
            reasons.append("missing_student_id")
        if not identity.architecture_family:
            reasons.append("missing_architecture_family")
        if identity.params_hash != run_hash:
            reasons.append("identity_params_mismatch")
        verified = not reasons
        return BindingRecord(int(session_idx), int(global_step), identity,
                             run_hash, ckpt_hash, probe_hash, ppo_hash,
                             bool(verified), tuple(reasons))

    def verify(self, record: BindingRecord) -> BindingRecord:
        if not record.binding_verified:
            raise BindingError(
                "PROBE_INVALID: student binding violation "
                f"({', '.join(record.failure_reasons) or 'unspecified'}); "
                f"student_id={record.identity.student_id!r} "
                f"version={record.identity.student_version} "
                f"runstate={record.runstate_params_hash[:12]} "
                f"probe={record.probe_params_hash[:12]} "
                f"ppo={record.ppo_params_hash[:12]}; refusing frontier "
                "selection / data generation / LLM / PPO")
        return record

    def report(self, record: BindingRecord) -> dict:
        body = record.to_dict()
        return {"schema": "e3_litesim.student_binding/v1", **body,
                "payload_hash": hash_payload(body)}
