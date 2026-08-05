"""CC2 follow-up P0-10/P0-12: ONE Student checkpoint probe->update.

The probes run against the window's Student checkpoint, and the
exactly-one optimizer update must consume that SAME checkpoint as its
input — never a different Student, never a re-init claim::

    binding = open_student_binding(
        student_identity_hash, student_checkpoint_hash, window_hash)
    assert_probe_student_binding(binding, probe_pool, ctx)   # probes
    bound = bind_update_input(binding, update_input_checkpoint_hash)
    assert_update_output_differs(bound.input, update_output_hash)

Fail-closed guarantees:

* every probe in the window ran against the SAME Student identity and
  checkpoint (E1_STUDENT_CHECKPOINT_SWAPPED otherwise);
* the update input checkpoint EQUALS the probe checkpoint
  (E1_UPDATE_STUDENT_MISMATCH otherwise — a probe on one Student and
  an update on another can never bind);
* the update OUTPUT differs from its input (E1_UPDATE_NO_PROGRESS);
* the input checkpoint cannot claim a re-init (zero global env steps
  as a fresh start is a fabricated reinit — rejected with
  E1_STUDENT_REINIT_CLAIM).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .canonical import canonical_sha256
from .schemas import E1SchemaError

# fail-closed codes (greppable)
E1_STUDENT_CONTINUITY_BAD = "E1_STUDENT_CONTINUITY_BAD"
E1_STUDENT_CHECKPOINT_SWAPPED = "E1_STUDENT_CHECKPOINT_SWAPPED"
E1_STUDENT_IDENTITY_SWAPPED = "E1_STUDENT_IDENTITY_SWAPPED"
E1_UPDATE_STUDENT_MISMATCH = "E1_UPDATE_STUDENT_MISMATCH"
E1_UPDATE_NO_PROGRESS = "E1_UPDATE_NO_PROGRESS"
E1_STUDENT_REINIT_CLAIM = "E1_STUDENT_REINIT_CLAIM"


class StudentContinuityError(E1SchemaError):
    """Fail-closed student-continuity violation; ``code`` is
    greppable."""


@dataclass(frozen=True)
class StudentBinding:
    """The window's Student binding (probe <-> update link)."""

    student_identity_hash: str
    student_checkpoint_hash: str
    input_global_env_steps: int
    window_hash: str
    binding_hash: str


@dataclass(frozen=True)
class BoundUpdateInput:
    """The update input checkpoint, bound to the probe checkpoint."""

    student_identity_hash: str
    input_checkpoint_hash: str
    input_global_env_steps: int
    window_hash: str
    binding_hash: str


def _require_sha64(value: Any, name: str, ctx: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise StudentContinuityError(
            E1_STUDENT_CONTINUITY_BAD,
            f"{ctx}: {name} must be a 64-hex hash, got {value!r}",
        )
    return value


def open_student_binding(
    *,
    student_identity_hash: str,
    student_checkpoint_hash: str,
    input_global_env_steps: int,
    window_hash: str,
) -> StudentBinding:
    """Open the window's Student binding (fail-closed on every field)."""
    ctx = "student_continuity.open"
    identity = _require_sha64(
        student_identity_hash, "student_identity_hash", ctx
    )
    checkpoint = _require_sha64(
        student_checkpoint_hash, "student_checkpoint_hash", ctx
    )
    if isinstance(input_global_env_steps, bool) or not isinstance(
        input_global_env_steps, int
    ):
        raise StudentContinuityError(
            E1_STUDENT_CONTINUITY_BAD,
            f"{ctx}: input_global_env_steps must be an int, got "
            f"{input_global_env_steps!r}",
        )
    if input_global_env_steps <= 0:
        raise StudentContinuityError(
            E1_STUDENT_REINIT_CLAIM,
            f"{ctx}: the update input claims {input_global_env_steps} "
            "global env steps; a real update input can never be a "
            "re-init (global_step=0 is a fabricated reinit)",
        )
    window = _require_sha64(window_hash, "window_hash", ctx)
    binding_hash = canonical_sha256(
        {
            "student_identity_hash": identity,
            "student_checkpoint_hash": checkpoint,
            "input_global_env_steps": input_global_env_steps,
            "window_hash": window,
        }
    )
    return StudentBinding(
        student_identity_hash=identity,
        student_checkpoint_hash=checkpoint,
        input_global_env_steps=input_global_env_steps,
        window_hash=window,
        binding_hash=binding_hash,
    )


def assert_probe_student_binding(
    binding: Any, probe_pool: Sequence[Any], ctx: str
) -> None:
    """Every probe in the window ran against the SAME Student identity
    + checkpoint."""
    if not isinstance(binding, StudentBinding):
        raise StudentContinuityError(
            E1_STUDENT_CONTINUITY_BAD,
            f"{ctx}: expected a StudentBinding, got "
            f"{type(binding).__name__}",
        )
    if not isinstance(probe_pool, (tuple, list)) or len(probe_pool) == 0:
        raise StudentContinuityError(
            E1_STUDENT_CONTINUITY_BAD,
            f"{ctx}: the probe pool is empty or not a sequence",
        )
    for probe in probe_pool:
        if getattr(probe, "student_checkpoint_hash", "") != (
            binding.student_checkpoint_hash
        ):
            raise StudentContinuityError(
                E1_STUDENT_CHECKPOINT_SWAPPED,
                f"{ctx}: probe {getattr(probe, 'result_id', '?')!r} ran "
                f"against Student checkpoint "
                f"{getattr(probe, 'student_checkpoint_hash', '')!r} != "
                f"the window's {binding.student_checkpoint_hash!r}",
            )
        if getattr(probe, "student_identity_hash", "") != (
            binding.student_identity_hash
        ):
            raise StudentContinuityError(
                E1_STUDENT_IDENTITY_SWAPPED,
                f"{ctx}: probe {getattr(probe, 'result_id', '?')!r} ran "
                f"against Student identity "
                f"{getattr(probe, 'student_identity_hash', '')!r} != "
                f"the window's {binding.student_identity_hash!r}",
            )


def bind_update_input(
    binding: Any, *, update_input_checkpoint_hash: str, ctx: str
) -> BoundUpdateInput:
    """Bind the exactly-one update's INPUT to the probe checkpoint."""
    if not isinstance(binding, StudentBinding):
        raise StudentContinuityError(
            E1_STUDENT_CONTINUITY_BAD,
            f"{ctx}: expected a StudentBinding, got "
            f"{type(binding).__name__}",
        )
    update_input = _require_sha64(
        update_input_checkpoint_hash, "update_input_checkpoint_hash", ctx
    )
    if update_input != binding.student_checkpoint_hash:
        raise StudentContinuityError(
            E1_UPDATE_STUDENT_MISMATCH,
            f"{ctx}: the update input checkpoint {update_input!r} != "
            f"the probe checkpoint {binding.student_checkpoint_hash!r}; "
            "the probes and the update must run on the SAME Student "
            "checkpoint",
        )
    return BoundUpdateInput(
        student_identity_hash=binding.student_identity_hash,
        input_checkpoint_hash=update_input,
        input_global_env_steps=binding.input_global_env_steps,
        window_hash=binding.window_hash,
        binding_hash=binding.binding_hash,
    )


def assert_update_output_differs(
    bound: Any, *, update_output_checkpoint_hash: str, ctx: str
) -> None:
    """The update OUTPUT must differ from its input checkpoint."""
    if not isinstance(bound, BoundUpdateInput):
        raise StudentContinuityError(
            E1_STUDENT_CONTINUITY_BAD,
            f"{ctx}: expected a BoundUpdateInput, got "
            f"{type(bound).__name__}",
        )
    update_output = _require_sha64(
        update_output_checkpoint_hash, "update_output_checkpoint_hash", ctx
    )
    if update_output == bound.input_checkpoint_hash:
        raise StudentContinuityError(
            E1_UPDATE_NO_PROGRESS,
            f"{ctx}: the update output checkpoint {update_output!r} "
            "equals its input — an optimizer update that changes "
            "nothing never counts as an update",
        )
