"""Preserve terminal evidence when wrappers auto-reset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


@dataclass(frozen=True)
class TerminalTransition:
    previous_state: Any
    action_metadata: Any
    reward: Any
    done: bool
    terminal_state: Any
    returned_state: Any
    events: tuple[Any, ...] = ()
    info: Mapping[str, Any] = field(default_factory=dict)
    autoreset_detected: bool = False
    reset_state: Any = None
    transition_hash: str = ""

    def __post_init__(self) -> None:
        if not self.done and self.terminal_state is not None:
            raise ValueError("terminal_state is only valid for done transitions")
        if self.done and self.autoreset_detected and self.reset_state is None:
            raise ValueError("autoreset transition must preserve reset_state")
        if not self.transition_hash:
            payload = {"previous_state": self.previous_state, "action_metadata": self.action_metadata,
                       "reward": self.reward, "done": self.done, "terminal_state": self.terminal_state,
                       "returned_state": self.returned_state, "events": self.events,
                       "autoreset_detected": self.autoreset_detected, "reset_state": self.reset_state}
            blob = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":"), default=str)
            object.__setattr__(self, "transition_hash", hashlib.sha256(blob.encode()).hexdigest())


class TerminalEventAdapter:
    """Adapter with explicit terminal/returned/reset state separation."""

    def adapt(self, *, previous_state: Any, action_metadata: Any, reward: Any, done: bool,
              returned_state: Any, terminal_state: Any = None, events: Sequence[Any] = (),
              info: Mapping[str, Any] | None = None, reset_state: Any = None) -> TerminalTransition:
        info = dict(info or {})
        autoreset = bool(done and (reset_state is not None or info.get("autoreset", False)))
        if done and terminal_state is None:
            terminal_state = info.get("terminal_state") or info.get("final_state")
        if done and terminal_state is None and not autoreset:
            terminal_state = returned_state
        if done and terminal_state is None and autoreset:
            raise ValueError("done+autoreset requires terminal_state in info or adapter input")
        return TerminalTransition(previous_state, action_metadata, reward, bool(done), terminal_state,
                                  returned_state, tuple(events), info, autoreset, reset_state)

    def goal_state(self, transition: TerminalTransition) -> Any:
        """Return the authoritative state for goal checks, never the auto-reset state."""
        return transition.terminal_state if transition.done else transition.returned_state
