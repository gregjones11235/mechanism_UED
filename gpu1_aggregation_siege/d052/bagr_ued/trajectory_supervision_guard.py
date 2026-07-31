"""Guard A — TrajectorySupervisionGuard (task sections 3 / 15).

Training trajectories are READ-ONLY EVIDENCE. This guard rejects, in ANY output
of the BA-BAGR-UED pipeline, the two forbidden transformations of evidence:

  1. SUPERVISION KEYS — any object carrying a key from
     FORBIDDEN_SUPERVISION_KEYS or FORBIDDEN_SUPERVISION_KEY_ALIASES
     (recommended_actions / suggested_actions / recommended_move / route /
     navigation_route / path_to_follow / expert_plan / bank_blob /
     formal_state_blob / formal_state_payload / state_payload / reward_delta /
     reward_shaping / ...). Keys are matched AFTER normalization (casefold +
     separator stripping) at EVERY level of every nested mapping/sequence.
     If such a key appears anywhere, the output would turn evidence into
     Student supervision.

  2. DIRECT ACTION ADVICE — any free-text field containing an imperative
     action instruction (bilingual patterns: "不要睡觉 / 远离怪物 / 应该攻击 /
     向左走 / 往北走 / 攻击怪物 / 朝梯子移动" and "don't sleep / move away /
     flee / walk left / go left / head north / move toward the ladder /
     attack the monster ..."). Behavior DESCRIPTIONS ("the student
     repeatedly attacks without effect") are allowed; the patterns target
     imperative/second-person advice forms. The board may describe
     environment STRUCTURE; it may never tell the Student what to do.

CC1 audit fix1 hardening (task §5): a string that, after trimming, looks like
a JSON object or array is parsed and the FULL guard is re-run over the parsed
structure — smuggling a forbidden key/advice inside a serialized string does
NOT evade the guard. Parsing is bounded by MAX_SERIALIZED_PARSE_DEPTH /
MAX_SERIALIZED_STRING_LENGTH / MAX_SERIALIZED_CONTAINER_ITEMS; exceeding any
limit is a fail-closed SERIALIZED_GUARD_LIMIT_EXCEEDED finding (never a
lenient skip). A string that merely LOOKS like JSON but fails to parse is
treated as plain text and still runs through the natural-language patterns.

Fail-closed: any finding -> GuardViolation with a specific, greppable code.
``scan`` is non-raising (returns a report); ``assert_clean`` raises.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from d052.bagr_ued import constants as C


class GuardViolation(Exception):
    TRAJECTORY_SUPERVISION_KEY_FORBIDDEN = "TRAJECTORY_SUPERVISION_KEY_FORBIDDEN"
    DIRECT_ACTION_ADVICE_FORBIDDEN = "DIRECT_ACTION_ADVICE_FORBIDDEN"
    #: CC1 audit fix1 (§5): serialized-string parse limit exceeded -> the
    #: guard refuses to let an unbounded payload pass uninspected.
    SERIALIZED_GUARD_LIMIT_EXCEEDED = "SERIALIZED_GUARD_LIMIT_EXCEEDED"

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


#: (compiled pattern, human label) — imperative action advice is forbidden in
#: ANY free-text field of ANY role output. Behavior descriptions (third
#: person, past/present tense observation) MUST NOT match these patterns.
_ACTION_ADVICE_PATTERNS = [
    (re.compile(p, re.IGNORECASE), label)
    for p, label in (
        (r"不要(睡觉|休息|睡眠)", "zh: 不要睡觉/休息"),
        (r"远离(怪物|敌人|威胁| hostile)", "zh: 远离威胁"),
        (r"应该(攻击|逃跑|向左|向右|走|移动|躲|逃)", "zh: 应该<动作>"),
        (r"必须(攻击|逃跑|移动|躲)", "zh: 必须<动作>"),
        (r"快(跑|逃|攻击)", "zh: 快跑/快攻击"),
        # CC1 audit fix1 (§7): directional / target imperative advice
        (r"攻击(怪物|敌人|威胁)", "zh: 攻击<目标>"),
        (r"往(北|南|东|西|左|右|前)(走|跑|移动)", "zh: 往<方向>走"),
        (r"向(北|南|东|西|左|右|前)(走|跑|移动)", "zh: 向<方向>走"),
        (r"朝(梯子|出口|门|高处|低处|目标)(移动|走|跑|前进)", "zh: 朝<目标>移动"),
        (r"\bdon'?t\s+(sleep|rest|attack|fight)\b", "en: don't sleep/rest/attack"),
        (r"\bdo\s+not\s+(sleep|rest|attack|fight|approach)\b",
         "en: do not sleep/rest/attack"),
        (r"\bavoid\s+(sleeping|resting|attacking)\b", "en: avoid <gerund>"),
        (r"\bmove\s+away\b", "en: move away"),
        (r"\bflee(\s|$)", "en: flee"),
        (r"\brun\s+away\b", "en: run away"),
        (r"\bwalk\s+(left|right|towards|away)\b", "en: walk <direction>"),
        # CC1 audit fix1 (§7): directional imperative advice
        (r"\bgo\s+(left|right|north|south|east|west|toward|towards)\b",
         "en: go <direction>"),
        (r"\bhead\s+(north|south|east|west|left|right|toward|towards)\b",
         "en: head <direction>"),
        (r"\bmove\s+toward(s)?\b", "en: move toward"),
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


#: CC1 audit fix1 (§6): base keys UNION alias vocabulary, all normalized.
#: Normalization keeps casefold + separator stripping, so "Suggested-Actions"
#: / "RECOMMENDED ACTION" / "suggested_actions" all collapse to one entry.
_FORBIDDEN_KEYS_NORMALIZED = {
    _normalize_key(k): k
    for k in (C.FORBIDDEN_SUPERVISION_KEYS | C.FORBIDDEN_SUPERVISION_KEY_ALIASES)
}


def _serialized_structure_violation(parsed: Any) -> str | None:
    """Return a limit-exceeded detail string if the parsed structure breaks
    MAX_SERIALIZED_PARSE_DEPTH or MAX_SERIALIZED_CONTAINER_ITEMS, else None."""
    stack: List[tuple] = [(parsed, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > C.MAX_SERIALIZED_PARSE_DEPTH:
            return (f"serialized nesting depth exceeds "
                    f"MAX_SERIALIZED_PARSE_DEPTH={C.MAX_SERIALIZED_PARSE_DEPTH}")
        if isinstance(node, dict):
            if len(node) > C.MAX_SERIALIZED_CONTAINER_ITEMS:
                return (f"serialized container has {len(node)} items > "
                        f"MAX_SERIALIZED_CONTAINER_ITEMS="
                        f"{C.MAX_SERIALIZED_CONTAINER_ITEMS}")
            stack.extend((v, depth + 1) for v in node.values())
        elif isinstance(node, list):
            if len(node) > C.MAX_SERIALIZED_CONTAINER_ITEMS:
                return (f"serialized container has {len(node)} items > "
                        f"MAX_SERIALIZED_CONTAINER_ITEMS="
                        f"{C.MAX_SERIALIZED_CONTAINER_ITEMS}")
            stack.extend((v, depth + 1) for v in node)
    return None


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
              findings: List[Dict[str, str]], _parse_depth: int = 0) -> None:
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
                self._walk(v, path=f"{path}.{k}", label=label,
                           findings=findings, _parse_depth=_parse_depth)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                self._walk(v, path=f"{path}[{i}]", label=label,
                           findings=findings, _parse_depth=_parse_depth)
        elif isinstance(obj, str):
            # 1. natural-language action-advice patterns (plain text always
            #    runs through these — a JSON parse failure is NEVER a skip)
            for rx, desc in _ACTION_ADVICE_PATTERNS:
                m = rx.search(obj)
                if m:
                    findings.append(dict(
                        code=GuardViolation.DIRECT_ACTION_ADVICE_FORBIDDEN,
                        path=path,
                        detail=f"direct action advice pattern {desc!r} "
                               f"matched {m.group(0)!r}"))
            # 2. CC1 audit fix1 (§5): bounded recursive parse of strings that
            #    look like serialized JSON (object / array — the required
            #    scope — plus JSON string literals, which closes the
            #    double-encoding evasion: '"{\\"reward_delta\\": 1}"'); the
            #    FULL guard re-runs inside every successful parse.
            stripped = obj.strip()
            if stripped[:1] in ("{", "[", '"'):
                if len(obj) > C.MAX_SERIALIZED_STRING_LENGTH:
                    findings.append(dict(
                        code=GuardViolation.SERIALIZED_GUARD_LIMIT_EXCEEDED,
                        path=path,
                        detail=f"serialized string length {len(obj)} exceeds "
                               f"MAX_SERIALIZED_STRING_LENGTH="
                               f"{C.MAX_SERIALIZED_STRING_LENGTH} — "
                               f"fail-closed, not scanned leniently"))
                    return
                if _parse_depth + 1 > C.MAX_SERIALIZED_PARSE_DEPTH:
                    findings.append(dict(
                        code=GuardViolation.SERIALIZED_GUARD_LIMIT_EXCEEDED,
                        path=path,
                        detail=f"serialized parse chain depth exceeds "
                               f"MAX_SERIALIZED_PARSE_DEPTH="
                               f"{C.MAX_SERIALIZED_PARSE_DEPTH} — "
                               f"fail-closed"))
                    return
                try:
                    parsed = json.loads(stripped)
                except (json.JSONDecodeError, ValueError):
                    parsed = None  # not JSON after all: plain text (already
                    # pattern-scanned above — never a lenient skip)
                if parsed is not None:
                    limit_issue = _serialized_structure_violation(parsed)
                    if limit_issue is not None:
                        findings.append(dict(
                            code=GuardViolation.SERIALIZED_GUARD_LIMIT_EXCEEDED,
                            path=path,
                            detail=f"{limit_issue} — fail-closed"))
                        return
                    self._walk(parsed, path=f"{path}<parsed>", label=label,
                               findings=findings,
                               _parse_depth=_parse_depth + 1)
        elif hasattr(obj, "model_dump"):  # pydantic canonical model
            self._walk(obj.model_dump(), path=path, label=label,
                       findings=findings, _parse_depth=_parse_depth)
