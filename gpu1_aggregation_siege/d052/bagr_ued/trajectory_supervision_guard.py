"""Guard A — TrajectorySupervisionGuard (task sections 3 / 15).

Training trajectories are READ-ONLY EVIDENCE. This guard rejects, in ANY output
of the BA-BAGR-UED pipeline, the two forbidden transformations of evidence:

  1. SUPERVISION KEYS — any object carrying a key from
     FORBIDDEN_SUPERVISION_KEYS (recommended_actions, action_sequence_to_follow,
     waypoints, expert_demonstration, policy_override, hidden_state_override,
     reward_delta, reward_shaping). If such a key appears anywhere in a nested
     dict/list, the output would turn evidence into Student supervision.

  2. DIRECT ACTION ADVICE — any free-text field containing an imperative action
     instruction (bilingual patterns: "不要睡觉 / 远离怪物 / 应该攻击" and
     "don't sleep / move away / flee / walk left / attack the monster ...").
     The board may describe environment STRUCTURE; it may never tell the
     Student what to do.

Fail-closed: any finding -> GuardViolation with a specific, greppable code.
``scan`` is non-raising (returns a report); ``assert_clean`` raises.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from d052.bagr_ued import constants as C


class GuardViolation(Exception):
    TRAJECTORY_SUPERVISION_KEY_FORBIDDEN = "TRAJECTORY_SUPERVISION_KEY_FORBIDDEN"
    DIRECT_ACTION_ADVICE_FORBIDDEN = "DIRECT_ACTION_ADVICE_FORBIDDEN"

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


#: (compiled pattern, human label) — imperative action advice is forbidden in
#: ANY free-text field of ANY role output.
_ACTION_ADVICE_PATTERNS = [
    (re.compile(p, re.IGNORECASE), label)
    for p, label in (
        (r"不要(睡觉|休息|睡眠)", "zh: 不要睡觉/休息"),
        (r"远离(怪物|敌人|威胁| hostile)", "zh: 远离威胁"),
        (r"应该(攻击|逃跑|向左|向右|走|移动|躲|逃)", "zh: 应该<动作>"),
        (r"必须(攻击|逃跑|移动|躲)", "zh: 必须<动作>"),
        (r"快(跑|逃|攻击)", "zh: 快跑/快攻击"),
        (r"\bdon'?t\s+(sleep|rest|attack|fight)\b", "en: don't sleep/rest/attack"),
        (r"\bdo\s+not\s+(sleep|rest|attack|fight|approach)\b",
         "en: do not sleep/rest/attack"),
        (r"\bavoid\s+(sleeping|resting|attacking)\b", "en: avoid <gerund>"),
        (r"\bmove\s+away\b", "en: move away"),
        (r"\bflee(\s|$)", "en: flee"),
        (r"\brun\s+away\b", "en: run away"),
        (r"\bwalk\s+(left|right|towards|away)\b", "en: walk <direction>"),
        (r"\byou\s+should\s+(attack|flee|move|walk|run|rest|sleep|avoid)\b",
         "en: you should <action>"),
        (r"\bthe\s+student\s+should\s+(attack|flee|move|walk|run|rest|sleep)\b",
         "en: the student should <action>"),
        (r"\battack\s+the\s+(monster|hostile|enemy|zombie|kobold)\b",
         "en: attack the <entity>"),
    )
]


def _normalize_key(k: str) -> str:
    return re.sub(r"[_\-\s]+", "", k).casefold()


_FORBIDDEN_KEYS_NORMALIZED = {_normalize_key(k): k for k in C.FORBIDDEN_SUPERVISION_KEYS}


class GuardFinding(dict):
    """Plain dict subclass for canonical-JSON friendliness."""


class TrajectorySupervisionGuard:
    """Stateless scanner; safe to share across roles."""

    def scan(self, obj: Any, *, label: str = "output") -> Dict[str, Any]:
        """Non-raising scan -> report dict {passed, findings, label}."""
        findings: List[Dict[str, str]] = []
        self._walk(obj, path="$", label=label, findings=findings)
        return dict(guard="TrajectorySupervisionGuard",
                    label=label,
                    passed=not findings,
                    findings=findings)

    def assert_clean(self, obj: Any, *, label: str = "output") -> Dict[str, Any]:
        """Fail-closed: raise GuardViolation on the first finding class."""
        report = self.scan(obj, label=label)
        if report["passed"]:
            return report
        first = report["findings"][0]
        raise GuardViolation(first["code"],
                             f"{label}: {first['code']} at {first['path']} "
                             f"({first['detail']}); total_findings="
                             f"{len(report['findings'])}")

    # -- internals ----------------------------------------------------------
    def _walk(self, obj: Any, *, path: str, label: str,
              findings: List[Dict[str, str]]) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str):
                    original = _FORBIDDEN_KEYS_NORMALIZED.get(_normalize_key(k))
                    if original is not None:
                        findings.append(dict(
                            code=GuardViolation.TRAJECTORY_SUPERVISION_KEY_FORBIDDEN,
                            path=f"{path}.{k}",
                            detail=f"forbidden supervision key "
                                   f"{original!r} present"))
                self._walk(v, path=f"{path}.{k}", label=label, findings=findings)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                self._walk(v, path=f"{path}[{i}]", label=label, findings=findings)
        elif isinstance(obj, str):
            for rx, desc in _ACTION_ADVICE_PATTERNS:
                m = rx.search(obj)
                if m:
                    findings.append(dict(
                        code=GuardViolation.DIRECT_ACTION_ADVICE_FORBIDDEN,
                        path=path,
                        detail=f"direct action advice pattern {desc!r} "
                               f"matched {m.group(0)!r}"))
        elif hasattr(obj, "model_dump"):  # pydantic canonical model
            self._walk(obj.model_dump(), path=path, label=label, findings=findings)
