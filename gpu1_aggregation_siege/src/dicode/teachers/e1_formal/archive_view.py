"""Provenance-admissible archive view (E1 edge; feeds ``evidence.py``).

This is the ONLY surface through which archive-derived facts may enter
the E1 teacher. It consumes a plain JSON-shaped snapshot (extraction
from any live archive structure happens OUTSIDE this module, at the
caller) and validates fail-closed:

* every task record carries an explicit provenance label that must be
  LLM-role admissible (TRAINING / NORMAL_TRAINING_FEEDBACK); FORMAL_*
  and CANDIDATE_EVALUATION labels are rejected here;
* every history entry is exactly ``{session_idx: int>=0,
  success_rate: float in [0, 1]}`` (bools rejected, no coercion);
* no tier/verdict/eval field exists in the schema, so none can leak.

``evidence_items()`` emits raw items for
``evidence.build_evidence_snapshot`` (source
``archive.performance_history``), where provenance is re-verified a
second time — never trusted from here. Pure standard library.

The view also exposes the minimal duck surface the training loop reads
on ``gen_manager.archive`` (``graph`` / ``save_graph`` / ``_lock``);
this round there is no persistent teacher-owned archive, which the
docstrings state honestly instead of pretending otherwise.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

from .schemas import E1SchemaError, assert_llm_role_admissible

# fail-closed codes
ARCHIVE_VIEW_BAD_TYPE = "ARCHIVE_VIEW_BAD_TYPE"
ARCHIVE_VIEW_MISSING_FIELD = "ARCHIVE_VIEW_MISSING_FIELD"
ARCHIVE_VIEW_UNKNOWN_FIELD = "ARCHIVE_VIEW_UNKNOWN_FIELD"
ARCHIVE_VIEW_OUT_OF_RANGE = "ARCHIVE_VIEW_OUT_OF_RANGE"

#: evidence source label emitted by this view (known to evidence.py)
SOURCE_PERFORMANCE_HISTORY = "archive.performance_history"

_TASK_FIELDS = frozenset({"task_id", "provenance", "performance_history"})
_HISTORY_FIELDS = frozenset({"session_idx", "success_rate"})


class ArchiveViewError(E1SchemaError):
    """Fail-closed archive-view violation; ``code`` is greppable."""


@dataclass(frozen=True)
class ArchiveTaskView:
    """One admissible task record (facts only, no verdicts)."""

    task_id: str
    provenance: str
    history: Tuple[Tuple[int, float], ...]  # (session_idx, success_rate)


@dataclass(frozen=True)
class ArchiveView:
    """Immutable, provenance-checked snapshot of admissible facts."""

    tasks: Tuple[ArchiveTaskView, ...]

    @property
    def task_ids(self) -> Tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks)

    # ------------------------------------------------------------------
    # Evidence emission (consumed by evidence.build_evidence_snapshot,
    # which RE-VERIFIES provenance and guard-scans every fact).
    # ------------------------------------------------------------------
    def evidence_items(self) -> List[Dict[str, Any]]:
        """Raw evidence items, one per task with a non-empty history."""
        items: List[Dict[str, Any]] = []
        for task in self.tasks:
            if len(task.history) == 0:
                continue  # no facts -> nothing admissible to say
            latest_session = max(session for session, _ in task.history)
            items.append(
                {
                    "source": SOURCE_PERFORMANCE_HISTORY,
                    "session_idx": latest_session,
                    "provenance": task.provenance,
                    "facts": {
                        "task_id": task.task_id,
                        "history": [list(entry) for entry in task.history],
                    },
                }
            )
        return items

    # ------------------------------------------------------------------
    # Duck surface for the training loop (gen_manager.archive.*)
    # ------------------------------------------------------------------
    @property
    def graph(self) -> Dict[str, Any]:
        """Deterministic read-only snapshot (NOT a live mutable graph)."""
        return {
            task.task_id: {
                "provenance": task.provenance,
                "performance_history": [
                    {"session_idx": session, "success_rate": rate}
                    for session, rate in task.history
                ],
            }
            for task in self.tasks
        }

    def save_graph(self) -> None:
        """Honest no-op: this round the teacher owns no persistent
        archive; the snapshot is immutable by construction. The legacy
        call site keeps working; nothing is silently written."""

    @property
    def _lock(self) -> threading.RLock:
        """A real re-entrant lock so legacy ``with archive._lock:``
        blocks behave correctly even though the view is immutable."""
        return self._shared_lock


# a single lock is shared per view instance via __post_init__ below is
# unnecessary: dataclass(frozen) + attribute => create lazily per view.
def _attach_lock(view: ArchiveView) -> ArchiveView:
    object.__setattr__(view, "_shared_lock", threading.RLock())
    return view


# ---------------------------------------------------------------------------
# Fail-closed consumption
# ---------------------------------------------------------------------------
def _consume_history(raw: Any, ctx: str) -> Tuple[Tuple[int, float], ...]:
    if not isinstance(raw, (list, tuple)):
        raise ArchiveViewError(
            ARCHIVE_VIEW_BAD_TYPE,
            f"{ctx}: performance_history must be a sequence, got "
            f"{type(raw).__name__}",
        )
    entries: List[Tuple[int, float]] = []
    for i, entry in enumerate(raw):
        entry_ctx = f"{ctx}.history[{i}]"
        if not isinstance(entry, Mapping):
            raise ArchiveViewError(
                ARCHIVE_VIEW_BAD_TYPE,
                f"{entry_ctx}: entry must be a mapping, got "
                f"{type(entry).__name__}",
            )
        unknown = sorted(k for k in entry if k not in _HISTORY_FIELDS)
        if unknown:
            raise ArchiveViewError(
                ARCHIVE_VIEW_UNKNOWN_FIELD,
                f"{entry_ctx}: unknown history field(s) {unknown}",
            )
        for name in ("session_idx", "success_rate"):
            if name not in entry:
                raise ArchiveViewError(
                    ARCHIVE_VIEW_MISSING_FIELD,
                    f"{entry_ctx}: missing {name!r}",
                )
        session_idx = entry["session_idx"]
        if isinstance(session_idx, bool) or not isinstance(session_idx, int):
            raise ArchiveViewError(
                ARCHIVE_VIEW_BAD_TYPE,
                f"{entry_ctx}: session_idx must be int, got "
                f"{type(session_idx).__name__}",
            )
        if session_idx < 0:
            raise ArchiveViewError(
                ARCHIVE_VIEW_OUT_OF_RANGE,
                f"{entry_ctx}: session_idx must be >= 0, got {session_idx}",
            )
        rate = entry["success_rate"]
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            raise ArchiveViewError(
                ARCHIVE_VIEW_BAD_TYPE,
                f"{entry_ctx}: success_rate must be a number, got "
                f"{type(rate).__name__}",
            )
        rate = float(rate)
        if not (0.0 <= rate <= 1.0):
            raise ArchiveViewError(
                ARCHIVE_VIEW_OUT_OF_RANGE,
                f"{entry_ctx}: success_rate must be in [0, 1], got {rate}",
            )
        entries.append((session_idx, rate))
    return tuple(entries)


def consume_archive_snapshot(mapping: Any, context: str) -> ArchiveView:
    """Parse an archive snapshot fail-closed (no defaults, no guesses).

    Expected shape::

        {"tasks": [
            {"task_id": str, "provenance": str,
             "performance_history": [
                 {"session_idx": int, "success_rate": float}, ...]},
            ...
        ]}
    """
    if not isinstance(mapping, Mapping):
        raise ArchiveViewError(
            ARCHIVE_VIEW_BAD_TYPE,
            f"{context}: archive snapshot must be a mapping, got "
            f"{type(mapping).__name__}",
        )
    unknown = sorted(k for k in mapping if k != "tasks")
    if unknown:
        raise ArchiveViewError(
            ARCHIVE_VIEW_UNKNOWN_FIELD,
            f"{context}: unknown snapshot field(s) {unknown} (only "
            "'tasks' is admissible; no tier/verdict fields)",
        )
    if "tasks" not in mapping:
        raise ArchiveViewError(
            ARCHIVE_VIEW_MISSING_FIELD, f"{context}: missing 'tasks'"
        )
    raw_tasks = mapping["tasks"]
    if not isinstance(raw_tasks, (list, tuple)):
        raise ArchiveViewError(
            ARCHIVE_VIEW_BAD_TYPE,
            f"{context}: 'tasks' must be a sequence, got "
            f"{type(raw_tasks).__name__}",
        )
    tasks: List[ArchiveTaskView] = []
    seen_ids = set()
    for i, raw in enumerate(raw_tasks):
        task_ctx = f"{context}.tasks[{i}]"
        if not isinstance(raw, Mapping):
            raise ArchiveViewError(
                ARCHIVE_VIEW_BAD_TYPE,
                f"{task_ctx}: task must be a mapping, got "
                f"{type(raw).__name__}",
            )
        unknown = sorted(k for k in raw if k not in _TASK_FIELDS)
        if unknown:
            raise ArchiveViewError(
                ARCHIVE_VIEW_UNKNOWN_FIELD,
                f"{task_ctx}: unknown task field(s) {unknown} (admissible: "
                f"{sorted(_TASK_FIELDS)})",
            )
        for name in ("task_id", "provenance", "performance_history"):
            if name not in raw:
                raise ArchiveViewError(
                    ARCHIVE_VIEW_MISSING_FIELD,
                    f"{task_ctx}: missing {name!r}",
                )
        task_id = raw["task_id"]
        if not isinstance(task_id, str) or not task_id.strip():
            raise ArchiveViewError(
                ARCHIVE_VIEW_BAD_TYPE,
                f"{task_ctx}: task_id must be a non-empty str",
            )
        task_id = task_id.strip()
        if task_id in seen_ids:
            raise ArchiveViewError(
                ARCHIVE_VIEW_OUT_OF_RANGE,
                f"{task_ctx}: duplicate task_id {task_id!r}",
            )
        seen_ids.add(task_id)
        provenance = assert_llm_role_admissible(raw["provenance"], task_ctx)
        history = _consume_history(
            raw["performance_history"], task_ctx
        )
        tasks.append(
            ArchiveTaskView(
                task_id=task_id, provenance=provenance, history=history
            )
        )
    return _attach_lock(ArchiveView(tasks=tuple(tasks)))


def empty_archive_view() -> ArchiveView:
    """The honest initial state: NO admissible archive facts exist."""
    return _attach_lock(ArchiveView(tasks=()))
