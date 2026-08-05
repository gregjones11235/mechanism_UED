"""CC2 follow-up P0-10: ONE GenManager across the whole window.

Every stage of the one-window pipeline must run against the SAME
teacher instance (one GenManager => one ledger, one archive view, one
bookkeeping state). ``OneWindowContinuity`` is the mechanical record
of that identity, opened with the window and re-checked on every
later stage — a swapped teacher instance (fresh state, replayed
counters, second loader) fails closed::

    session = begin_one_window_session(teacher, runtime)
    ... later stage ...
    assert_one_window_continuity(session, teacher, runtime, ctx)

The check is object identity (``is``), never structural equality: a
second GenManager rebuilt from the same config is a DIFFERENT teacher
and must be refused.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical import canonical_sha256
from .schemas import E1SchemaError

# fail-closed codes (greppable)
E1_TEACHER_SWAPPED = "E1_TEACHER_SWAPPED"
E1_TEACHER_RUNTIME_SWAPPED = "E1_TEACHER_RUNTIME_SWAPPED"
E1_TEACHER_CONTINUITY_BAD = "E1_TEACHER_CONTINUITY_BAD"


class TeacherContinuityError(E1SchemaError):
    """Fail-closed continuity violation; ``code`` is greppable."""


@dataclass(frozen=True)
class OneWindowContinuity:
    """The window's teacher-identity record (immutable)."""

    teacher_id: int
    teacher_type: str
    runtime_bundle_hash: str
    cycles_run_at_open: int
    consecutive_reuses_at_open: int
    session_hash: str


def begin_one_window_session(teacher: Any, runtime: Any) -> OneWindowContinuity:
    """Open the continuity record for ONE window."""
    ctx = "teacher_continuity.begin"
    if teacher is None:
        raise TeacherContinuityError(
            E1_TEACHER_CONTINUITY_BAD,
            f"{ctx}: teacher is None — a window needs its ONE real "
            "GenManager",
        )
    if getattr(runtime, "bundle_hash", "") == "":
        raise TeacherContinuityError(
            E1_TEACHER_CONTINUITY_BAD,
            f"{ctx}: runtime carries no bundle_hash — continuity binds "
            "the signed runtime bundle",
        )
    cycles_run = int(getattr(teacher, "cycles_run", 0))
    consecutive_reuses = int(getattr(teacher, "consecutive_reuses", 0))
    session_hash = canonical_sha256(
        {
            "teacher_id": id(teacher),
            "teacher_type": type(teacher).__name__,
            "runtime_bundle_hash": runtime.bundle_hash,
            "cycles_run_at_open": cycles_run,
            "consecutive_reuses_at_open": consecutive_reuses,
        }
    )
    return OneWindowContinuity(
        teacher_id=id(teacher),
        teacher_type=type(teacher).__name__,
        runtime_bundle_hash=runtime.bundle_hash,
        cycles_run_at_open=cycles_run,
        consecutive_reuses_at_open=consecutive_reuses,
        session_hash=session_hash,
    )


def assert_one_window_continuity(
    session: Any, teacher: Any, runtime: Any, ctx: str
) -> None:
    """Fail closed unless this stage runs on the SAME teacher + bundle.

    Object identity ONLY: a structurally identical second GenManager
    (fresh ledger, fresh archive, replayed counters) is a swapped
    teacher and is refused.
    """
    if not isinstance(session, OneWindowContinuity):
        raise TeacherContinuityError(
            E1_TEACHER_CONTINUITY_BAD,
            f"{ctx}: expected a OneWindowContinuity session, got "
            f"{type(session).__name__}",
        )
    if teacher is None or id(teacher) != session.teacher_id:
        raise TeacherContinuityError(
            E1_TEACHER_SWAPPED,
            f"{ctx}: stage teacher (id {id(teacher)}) != the window's "
            f"teacher (id {session.teacher_id}); ONE window runs on "
            "ONE GenManager instance — a second loader/teacher is "
            "never a substitute",
        )
    if type(teacher).__name__ != session.teacher_type:
        raise TeacherContinuityError(
            E1_TEACHER_SWAPPED,
            f"{ctx}: stage teacher type {type(teacher).__name__!r} != "
            f"window teacher type {session.teacher_type!r}",
        )
    if getattr(runtime, "bundle_hash", "") != session.runtime_bundle_hash:
        raise TeacherContinuityError(
            E1_TEACHER_RUNTIME_SWAPPED,
            f"{ctx}: stage runtime bundle "
            f"{getattr(runtime, 'bundle_hash', '')!r} != window bundle "
            f"{session.runtime_bundle_hash!r}; the window binds ONE "
            "signed runtime bundle",
        )
