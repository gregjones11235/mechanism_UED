"""HO (history-observation) reinjection contract for the Memory Study.

Frozen design anchors (source-grounded; see docs/memory_study/MEMORY_STUDY_CODE_MAP.md):

* Student observation dimensionality: 8335 = 8268 base symbolic + 67 achievement
  multi-hot. Source of truth: gpu1_aggregation_siege/d052/legacy/canonical_constants.py
  (STUDENT_OBS_DIM, frozen label D052_STUDENT_OBS_DIM_8335=PASS). Imported
  file-based when importable; otherwise the identical constant is used with the
  source recorded here.
* Serialization discipline mirrors
  tools/tier3_scaffolded_evaluation/tier3_state_serializer.py: canonical JSON
  (sorted keys, compact separators) + sha256 hex.
* Isolation assertions are MECHANICAL: any violated G2 check raises FailClosed.
  A receipt is never issued for a violated isolation contract.

This module is JAX-free by design (local SYNTHETIC mode and protocol tests run
without jax/craftax; only the server REAL path touches jax arrays, via
adapters).
"""
from __future__ import annotations

import dataclasses
import enum
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Optional, Tuple

SCHEMA_ID_CONTRACT = "mechanism_UED.memory_study_ho_contract/v1"

#: Frozen canonical student observation dimensionality.
#: Source: gpu1_aggregation_siege/d052/legacy/canonical_constants.py
#: STUDENT_OBS_DIM = BASE_OBS_DIM 8268 + NUM_ACHIEVEMENTS 67 = 8335.
_FALLBACK_STUDENT_OBS_DIM = 8335


class FailClosed(Exception):
    """Hard stop: a Memory Study isolation/provenance contract was violated."""


class HOMode(enum.Enum):
    """History-observation burn-in modes compared by the Floor2->Floor3 probe."""

    BASE = "base"        # no burn-in; memory as initialized (control)
    HO_ZERO = "ho_zero"  # burn-in with all-zero segment (structural control)
    HO_REAL = "ho_real"  # burn-in with a result-blind captured segment


def canonical_obs_dim() -> int:
    """STUDENT_OBS_DIM from the frozen canonical constants when importable, else
    the identical fallback constant. Drift between the two fails closed."""
    path = (Path(__file__).resolve().parents[3]
            / "d052" / "legacy" / "canonical_constants.py")
    try:
        spec = importlib.util.spec_from_file_location(
            "memory_study_canonical_constants", str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        val = int(getattr(mod, "STUDENT_OBS_DIM"))
    except FailClosed:
        raise
    except Exception:
        return _FALLBACK_STUDENT_OBS_DIM
    if val != _FALLBACK_STUDENT_OBS_DIM:
        raise FailClosed(
            "CANONICAL_OBS_DIM_DRIFT: canonical_constants STUDENT_OBS_DIM=%r "
            "!= frozen %r" % (val, _FALLBACK_STUDENT_OBS_DIM))
    return val


def canonical_json_bytes(obj: Any) -> bytes:
    """Canonical JSON bytes (sorted keys, compact separators, UTF-8). Mirrors the
    tier3_state_serializer discipline so hashes cross-check across tooling."""
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FailClosed("NON_CANONICAL_PAYLOAD: %s" % exc) from exc


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def structural_form(obj: Any) -> Any:
    """Deterministic structural form for pytree-like objects (dict/list/tuple/
    scalars; numpy/jax leaves handled via .tolist())."""
    if isinstance(obj, dict):
        return {"__dict__": [[str(k), structural_form(v)]
                             for k, v in sorted(obj.items(),
                                                key=lambda kv: str(kv[0]))]}
    if isinstance(obj, (list, tuple)):
        return {"__seq__": [structural_form(v) for v in obj]}
    if isinstance(obj, bytes):
        return {"__bytes__": obj.hex()}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return {"__leaf__": [type(obj).__name__, obj]}
    if hasattr(obj, "tolist"):
        try:
            return {"__leaf_tolist__": structural_form(obj.tolist())}
        except Exception as exc:
            raise FailClosed("UNHASHABLE_LEAF: %s" % exc) from exc
    raise FailClosed("UNHASHABLE_NODE: %r" % type(obj))


def hash_pytree(obj: Any) -> str:
    """Deterministic sha256 over the structural form of a pytree-like object."""
    return sha256_hex(canonical_json_bytes(structural_form(obj)))


@dataclasses.dataclass(frozen=True)
class HistoryCapture:
    """One result-blind captured observation segment (T, obs_dim) + provenance."""

    capture_id: str
    obs_segment: Tuple[Tuple[float, ...], ...]
    source_seed: int
    capture_policy_id: str
    bank_hash: str
    payload_sha256: str
    floor_context: int = 2
    timestep_start: int = 0

    def payload_form(self) -> dict:
        return {
            "capture_id": self.capture_id,
            "obs_segment": [list(row) for row in self.obs_segment],
            "source_seed": int(self.source_seed),
            "capture_policy_id": self.capture_policy_id,
            "floor_context": int(self.floor_context),
            "timestep_start": int(self.timestep_start),
        }

    def validate(self) -> None:
        if not self.obs_segment:
            raise FailClosed("EMPTY_OBS_SEGMENT: %s" % self.capture_id)
        width = len(self.obs_segment[0])
        if width <= 0:
            raise FailClosed("EMPTY_OBS_ROW: %s" % self.capture_id)
        for i, row in enumerate(self.obs_segment):
            if len(row) != width:
                raise FailClosed(
                    "RAGGED_OBS_SEGMENT row %d: %s" % (i, self.capture_id))
        actual = sha256_hex(canonical_json_bytes(self.payload_form()))
        if actual != self.payload_sha256:
            raise FailClosed(
                "CAPTURE_PAYLOAD_HASH_MISMATCH: %s claimed=%s actual=%s"
                % (self.capture_id, self.payload_sha256, actual))


@dataclasses.dataclass(frozen=True)
class IsolationContext:
    """Caller-supplied G2 context snapshot taken BEFORE burn-in.

    env_state_payload_hash MUST be None: burn-in is structurally isolated from
    environment state (burnin_history accepts no env parameter). Supplying a
    hash here is itself a contract violation and fails closed."""

    params_sha_before: str
    env_state_payload_hash: Optional[str]
    rng_stream_id: str
    task_embedding_hash: str
    timestep: int
    inventory_hash: str
    position_hash: str
    entities_hash: str


@dataclasses.dataclass(frozen=True)
class IsolationReceipt:
    """Mechanical G2 receipt. Issuing raises FailClosed on ANY failed check."""

    ho_mode: str
    params_sha_before: str
    params_sha_after: str
    env_state_hash_before: Optional[str]
    env_state_hash_after: Optional[str]
    rng_stream_id: str
    task_embedding_hash: str
    timestep: int
    inventory_hash: str
    position_hash: str
    entities_hash: str
    burnin_steps: int
    checks: Tuple[Tuple[str, bool], ...]
    verdict: str

    @staticmethod
    def issue(ho_mode: str, params_sha_before: str, params_sha_after: str,
              ctx: IsolationContext, burnin_steps: int) -> "IsolationReceipt":
        checks = (
            ("params_invariant", params_sha_before == params_sha_after),
            ("env_state_structurally_absent",
             ctx.env_state_payload_hash is None),
            ("rng_stream_declared", bool(ctx.rng_stream_id)),
            ("task_embedding_hash_present", bool(ctx.task_embedding_hash)),
            ("inventory_hash_present", bool(ctx.inventory_hash)),
            ("position_hash_present", bool(ctx.position_hash)),
            ("entities_hash_present", bool(ctx.entities_hash)),
        )
        failed = [name for name, ok in checks if not ok]
        if failed:
            raise FailClosed("ISOLATION_VIOLATION: " + ",".join(failed))
        return IsolationReceipt(
            ho_mode=ho_mode,
            params_sha_before=params_sha_before,
            params_sha_after=params_sha_after,
            env_state_hash_before=None,
            env_state_hash_after=None,
            rng_stream_id=ctx.rng_stream_id,
            task_embedding_hash=ctx.task_embedding_hash,
            timestep=int(ctx.timestep),
            inventory_hash=ctx.inventory_hash,
            position_hash=ctx.position_hash,
            entities_hash=ctx.entities_hash,
            burnin_steps=int(burnin_steps),
            checks=checks,
            verdict="PASS")
