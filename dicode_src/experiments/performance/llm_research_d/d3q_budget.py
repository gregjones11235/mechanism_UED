"""D3Q shared POST budget state machine.

D3Q freezes the LLM request budget so that no arm can compensate for failed
generations with an unbounded number of POSTs:

* ``SlotBudget``: each slot may issue at most 3 POSTs in total. The kinds
  ``initial``, ``transport_retry`` and ``semantic_repair`` all consume the same
  budget; the 4th reserve raises :class:`BudgetExceededError` carrying the
  ``slot_id``.
* ``ProviderBudget``: each provider (``ollama`` / ``deepseek_official``) may
  issue at most 108 POSTs in total across all slots.
* ``D3QLedger``: records every reserve event as a JSONL line (``ts_utc``,
  ``slot_id``, ``model``, ``provider``, ``kind``, ``attempt_index``,
  ``post_index_in_slot``, ``post_index_for_provider``) and can rebuild the used
  budgets from a previously written ledger (load/resume).

A reserve always checks the slot budget AND the provider budget before any
counter is incremented or any ledger line is recorded, so a rejected POST never
leaves a trace in the ledger.

This module performs no network access and depends only on the standard
library.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Optional

VALID_KINDS = frozenset({"initial", "transport_retry", "semantic_repair"})
DEFAULT_SLOT_LIMIT = 3
DEFAULT_PROVIDER_LIMIT = 108

LEDGER_FIELDS = (
    "ts_utc",
    "slot_id",
    "model",
    "provider",
    "kind",
    "attempt_index",
    "post_index_in_slot",
    "post_index_for_provider",
)


class BudgetExceededError(Exception):
    """Raised when a POST reserve would exceed a frozen budget."""

    def __init__(
        self,
        *,
        slot_id: Optional[str] = None,
        provider: Optional[str] = None,
        limit: Optional[int] = None,
        kind: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        self.slot_id = slot_id
        self.provider = provider
        self.limit = limit
        self.kind = kind
        if message is None:
            context: list[str] = []
            if slot_id is not None:
                context.append(f"slot={slot_id!r}")
            if provider is not None:
                context.append(f"provider={provider!r}")
            if limit is not None:
                context.append(f"limit={limit}")
            if kind is not None:
                context.append(f"kind={kind!r}")
            suffix = f" ({', '.join(context)})" if context else ""
            message = f"POST budget exceeded{suffix}"
        super().__init__(message)


def _validate_kind(kind: str) -> None:
    if not isinstance(kind, str) or kind not in VALID_KINDS:
        raise ValueError(f"invalid kind {kind!r}; expected one of {sorted(VALID_KINDS)}")


class SlotBudget:
    """Per-slot POST budget shared by all three kinds."""

    def __init__(self, slot_id: str, limit: int = DEFAULT_SLOT_LIMIT) -> None:
        if not isinstance(slot_id, str) or not slot_id.strip():
            raise ValueError("slot_id must be a non-empty string")
        if int(limit) <= 0:
            raise ValueError("limit must be a positive integer")
        self.slot_id = slot_id.strip()
        self.limit = int(limit)
        self.post_count = 0

    def check(self, kind: str) -> None:
        _validate_kind(kind)
        if self.post_count >= self.limit:
            raise BudgetExceededError(slot_id=self.slot_id, limit=self.limit, kind=kind)

    def reserve(self, kind: str) -> int:
        """Reserve one POST of ``kind``; returns the new per-slot POST index."""
        self.check(kind)
        self.post_count += 1
        return self.post_count


class ProviderBudget:
    """Per-provider POST budget across all slots."""

    def __init__(self, provider: str, limit: int = DEFAULT_PROVIDER_LIMIT) -> None:
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("provider must be a non-empty string")
        if int(limit) <= 0:
            raise ValueError("limit must be a positive integer")
        self.provider = provider.strip()
        self.limit = int(limit)
        self.post_count = 0

    def check(self, kind: Optional[str] = None) -> None:
        if kind is not None:
            _validate_kind(kind)
        if self.post_count >= self.limit:
            raise BudgetExceededError(provider=self.provider, limit=self.limit, kind=kind)

    def reserve(self, kind: Optional[str] = None) -> int:
        """Reserve one POST; returns the new per-provider POST index."""
        self.check(kind)
        self.post_count += 1
        return self.post_count


class D3QLedger:
    """JSONL ledger that records POST reserves and can resume from disk.

    The ledger is the single source of truth for used budgets: ``load()``
    rebuilds the per-slot and per-provider counters from the recorded events
    and fails closed on malformed lines, index mismatches, or budgets that
    exceed the frozen limits.
    """

    def __init__(
        self,
        path: Optional[os.PathLike] = None,
        *,
        slot_limit: int = DEFAULT_SLOT_LIMIT,
        provider_limit: int = DEFAULT_PROVIDER_LIMIT,
    ) -> None:
        self.path: Optional[str] = None if path is None else os.fspath(path)
        self.slot_limit = int(slot_limit)
        self.provider_limit = int(provider_limit)
        self._slots: Dict[str, SlotBudget] = {}
        self._providers: Dict[str, ProviderBudget] = {}
        self._lock = threading.Lock()
        if self.path is not None and os.path.exists(self.path):
            self.load()

    def _slot_budget(self, slot_id: str) -> SlotBudget:
        budget = self._slots.get(slot_id)
        if budget is None:
            budget = SlotBudget(slot_id, self.slot_limit)
            self._slots[slot_id] = budget
        return budget

    def _provider_budget(self, provider: str) -> ProviderBudget:
        budget = self._providers.get(provider)
        if budget is None:
            budget = ProviderBudget(provider, self.provider_limit)
            self._providers[provider] = budget
        return budget

    def slot_post_count(self, slot_id: str) -> int:
        budget = self._slots.get(slot_id)
        return 0 if budget is None else budget.post_count

    def provider_post_count(self, provider: str) -> int:
        budget = self._providers.get(provider)
        return 0 if budget is None else budget.post_count

    def reserve(
        self,
        *,
        ts_utc: str,
        slot_id: str,
        model: str,
        provider: str,
        kind: str,
        attempt_index: int,
    ) -> Dict[str, Any]:
        """Check both budgets, then record the reserve event.

        Raises :class:`BudgetExceededError` (slot or provider limit) or
        ``ValueError`` (invalid kind / attempt_index) without recording any
        ledger line when the reserve is rejected.
        """
        _validate_kind(kind)
        if not isinstance(attempt_index, int) or attempt_index < 1:
            raise ValueError("attempt_index must be a positive integer")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(ts_utc, str) or not ts_utc.strip():
            raise ValueError("ts_utc must be a non-empty string")

        with self._lock:
            slot = self._slot_budget(slot_id)
            provider_budget = self._provider_budget(provider)
            # Both budgets are checked BEFORE any counter is incremented or
            # any ledger line is recorded.
            slot.check(kind)
            provider_budget.check(kind)
            post_index_in_slot = slot.reserve(kind)
            post_index_for_provider = provider_budget.reserve(kind)
            event: Dict[str, Any] = {
                "ts_utc": ts_utc.strip(),
                "slot_id": slot.slot_id,
                "model": model.strip(),
                "provider": provider_budget.provider,
                "kind": kind,
                "attempt_index": int(attempt_index),
                "post_index_in_slot": post_index_in_slot,
                "post_index_for_provider": post_index_for_provider,
            }
            if self.path is not None:
                self._append(event)
            return dict(event)

    def _append(self, event: Dict[str, Any]) -> None:
        line = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def load(self) -> None:
        """Rebuild used budgets from the ledger file (fail closed)."""
        if self.path is None:
            raise ValueError("ledger has no path; nothing to load")
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as handle:
            for lineno, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"ledger line {lineno} is not valid JSON: {exc}") from exc
                self._apply(event, lineno)

    def _apply(self, event: Dict[str, Any], lineno: int) -> None:
        for field in LEDGER_FIELDS:
            if field not in event:
                raise ValueError(f"ledger line {lineno} missing required field {field!r}")
        kind = event["kind"]
        _validate_kind(kind)
        attempt_index = event["attempt_index"]
        if not isinstance(attempt_index, int) or attempt_index < 1:
            raise ValueError(f"ledger line {lineno} invalid attempt_index {attempt_index!r}")
        slot = self._slot_budget(event["slot_id"])
        provider_budget = self._provider_budget(event["provider"])
        # Validate recorded indices against the rebuilt counters BEFORE mutating.
        if event["post_index_in_slot"] != slot.post_count + 1:
            raise ValueError(
                f"ledger line {lineno} post_index_in_slot {event['post_index_in_slot']!r} "
                f"!= rebuilt {slot.post_count + 1}"
            )
        if event["post_index_for_provider"] != provider_budget.post_count + 1:
            raise ValueError(
                f"ledger line {lineno} post_index_for_provider "
                f"{event['post_index_for_provider']!r} != rebuilt {provider_budget.post_count + 1}"
            )
        # reserve() re-checks the budget, so an over-budget ledger fails closed.
        slot.reserve(kind)
        provider_budget.reserve(kind)