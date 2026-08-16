"""P1 StudentBindingGuard: prove RunState == probe == PPO params, fail closed."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional

from .hashing import hash_payload, hash_pytree


class BindingError(RuntimeError):
    """PROBE_INVALID: student binding violation (fail closed)."""


@dataclass(frozen=True)
class BindingRecord:
    session_idx: int
    global_step: int
    student_version: str
    runstate_params_hash: str
    checkpoint_params_hash: Optional[str]
    probe_params_hash: str
    ppo_params_hash: str
    binding_verified: bool

    def to_dict(self) -> dict:
        return asdict(self)


class StudentBindingGuard:
    """Every probe / data generation / PPO update must pass this guard first."""

    def bind(self, *, session_idx: int, global_step: int, student_version: str,
             runstate_params: Any, probe_params: Any, ppo_params: Any,
             checkpoint_params: Any = None) -> BindingRecord:
        run_hash = hash_pytree(runstate_params)
        probe_hash = hash_pytree(probe_params)
        ppo_hash = hash_pytree(ppo_params)
        ckpt_hash = hash_pytree(checkpoint_params) if checkpoint_params is not None else None
        verified = (run_hash == probe_hash == ppo_hash) and (
            ckpt_hash is None or ckpt_hash == run_hash)
        return BindingRecord(int(session_idx), int(global_step), str(student_version),
                             run_hash, ckpt_hash, probe_hash, ppo_hash, bool(verified))

    def verify(self, record: BindingRecord) -> BindingRecord:
        if not record.binding_verified:
            raise BindingError(
                "PROBE_INVALID: params hash mismatch "
                f"(runstate={record.runstate_params_hash[:12]} "
                f"probe={record.probe_params_hash[:12]} "
                f"ppo={record.ppo_params_hash[:12]}); refusing frontier "
                "selection / data generation / LLM / PPO")
        return record

    def report(self, record: BindingRecord) -> dict:
        body = record.to_dict()
        return {"schema": "e3_litesim.student_binding/v1", **body,
                "payload_hash": hash_payload(body)}