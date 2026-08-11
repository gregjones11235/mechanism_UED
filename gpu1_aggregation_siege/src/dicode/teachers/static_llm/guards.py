"""Deterministic output guards for the static-LLM-UED teacher (design contract S3).

Fail-closed scanners for the forbidden-content contract F1-F7. The LLM (any
role) must never emit, in output that enters the system:

* F1 action sequences, routes, or waypoints;
* F2 step-by-step instructions telling the Student what to do next;
* F3 reward-function modification/shaping;
* F4 expert trajectories or demonstrations;
* F5 formal-evaluation data (any reference or marker);
* F6 hidden states, logits, or policy parameters;
* F7 direct modification of the Student policy / optimizer.

Pure standard library, pure functions over immutable compiled patterns: no
state, thread-safe, deterministic. Scanning is RECURSIVE over any JSON-shaped
LLM output (dict keys AND values, lists, tuples); unsupported container item
types fail closed rather than being silently skipped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Pattern, Tuple

from .schemas import SchemaError, assert_admissible_provenance


class GuardCode:
    """Greppable guard decision codes."""

    GUARD_PASS = "GUARD_PASS"
    # F1/F2
    ACTION_SEQUENCE_DETECTED = "ACTION_SEQUENCE_DETECTED"
    # F1
    WAYPOINT_DETECTED = "WAYPOINT_DETECTED"
    # F3
    REWARD_MODIFICATION_DETECTED = "REWARD_MODIFICATION_DETECTED"
    # F4
    EXPERT_TRAJECTORY_DETECTED = "EXPERT_TRAJECTORY_DETECTED"
    # F5
    FORMAL_DATA_DETECTED = "FORMAL_DATA_DETECTED"
    # F6
    HIDDEN_STATE_DETECTED = "HIDDEN_STATE_DETECTED"
    # F7
    POLICY_MODIFICATION_DETECTED = "POLICY_MODIFICATION_DETECTED"
    # structural fail-closed
    UNSUPPORTED_TYPE = "UNSUPPORTED_TYPE"


@dataclass(frozen=True)
class GuardDecision:
    """Immutable guard verdict. ``allowed=False`` is always fail-closed."""

    allowed: bool
    code: str
    detail: str = ""
    scan: str = ""
    path: str = "$"


def _pass(path: str = "$") -> GuardDecision:
    return GuardDecision(allowed=True, code=GuardCode.GUARD_PASS, path=path)


def _flag(code: str, scan: str, match_text: str, path: str) -> GuardDecision:
    snippet = match_text.strip().replace("\n", " ")
    if len(snippet) > 80:
        snippet = snippet[:77] + "..."
    return GuardDecision(
        allowed=False,
        code=code,
        detail=f"forbidden content near {snippet!r}",
        scan=scan,
        path=path,
    )


# ---------------------------------------------------------------------------
# Scanner definitions (fixed order => deterministic first-hit)
# ---------------------------------------------------------------------------
def _ci(pattern: str) -> Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


#: (scanner name, guard code, patterns) — order is part of the contract.
SCANNERS: Tuple[Tuple[str, str, Tuple[Pattern[str], ...]], ...] = (
    (
        "F1F2_action_sequence",
        GuardCode.ACTION_SEQUENCE_DETECTED,
        (
            # numbered step-by-step instructions
            _ci(r"\bstep\s+\d+\s*[:\.\)]"),
            _ci(r"\bfirst\s+(?:press|move|walk|turn|go|attack|use|open|mine)\b[^.]{0,120}\bthen\b"),
            # imperative navigation/press chains ("move north, then turn, then press ...")
            _ci(
                r"\b(?:press|move|walk|turn)\b[^.;]{0,30}"
                r"(?:\s*(?:,|;|->|→)\s*(?:then\s+)?|\bthen\s+)"
                r"(?:press|move|walk|turn)\b[^.;]{0,30}"
                r"(?:\s*(?:,|;|->|→)\s*(?:then\s+)?|\bthen\s+)"
                r"(?:press|move|walk|turn)\b"
            ),
            _ci(r"\baction\s*[- ]?sequence\b"),
            _ci(r"\bdo(?:es)?\s+the\s+following\s+(?:actions|steps)\b"),
            _ci(r"\bnext\s*,?\s+(?:press|move|walk|turn|attack)\b"),
            _ci(r"\byou\s+should\s+(?:press|move|walk|turn|attack)\b"),
        ),
    ),
    (
        "F1_waypoint",
        GuardCode.WAYPOINT_DETECTED,
        (
            # NOTE: ``\w*`` (not ``s?``) so underscore-joined identifiers such
            # as dict keys "waypoint_list" are still detected (``\b`` does not
            # fire between '_' and a letter).
            _ci(r"\bwaypoint\w*\b"),
            # numeric coordinate pairs anchored by spatial verbs/nouns
            _ci(r"\b(?:at|to|toward|position(?:ed)?|coordinates?)\s*\(?\s*-?\d{1,4}\s*,\s*-?\d{1,4}\s*\)?"),
            _ci(r"\bgo\s+to\s+\(?\s*-?\d{1,4}\s*,\s*-?\d{1,4}\s*\)?"),
            _ci(r"\broutes?\s+(?:from|to|through|via)\b"),
        ),
    ),
    (
        "F3_reward_modification",
        GuardCode.REWARD_MODIFICATION_DETECTED,
        (
            # NOTE: ``\w*reward\w*`` (not ``\breward``) so underscore-joined
            # identifiers such as "total_reward += bonus" are detected; ``\b``
            # does not fire between '_' and a letter.
            _ci(r"\w*reward\w*\s*=[^=]"),
            _ci(r"\w*reward\w*\s*\+=[^=]"),
            _ci(r"\bdef\s+\w*reward\w*\s*\("),
            _ci(r"\breward[- _]?shaping\b"),
            _ci(r"\b(?:modify|modifying|tamper|override|overwrite)\w*\s+(?:the\s+)?reward\b"),
            _ci(r"\bshaped\s+reward\b"),
        ),
    ),
    (
        "F4_expert_trajectory",
        GuardCode.EXPERT_TRAJECTORY_DETECTED,
        (
            _ci(r"\bexpert\s+(?:trajector\w+|demonstration\w*|polic\w+|actions?)\b"),
            _ci(r"\bdemonstration\s+trajector\w*\b"),
            _ci(r"\bimitation\s+learning\b"),
        ),
    ),
    (
        "F5_formal_data",
        GuardCode.FORMAL_DATA_DETECTED,
        (
            _ci(r"\bFORMAL_(?:FRONT|BACK|FULL)\b"),
            _ci(r"\bformal\s+(?:front|back|full)\b"),
            _ci(r"\bformal\s+evaluation\s+(?:trajector\w*|output\w*|results?)\b"),
            _ci(r"\bevaluation\s+trajector\w*\b"),
        ),
    ),
    (
        "F6_hidden_state",
        GuardCode.HIDDEN_STATE_DETECTED,
        (
            _ci(r"\blogits?\b"),
            _ci(r"\bhidden[- _]?states?\b"),
            _ci(r"\bpolicy[- _]?weights?\b"),
            _ci(r"\bnetwork\s+weights?\b"),
            _ci(r"\bparameter\s+gradients?\b"),
        ),
    ),
    (
        "F7_policy_modification",
        GuardCode.POLICY_MODIFICATION_DETECTED,
        (
            _ci(
                r"\b(?:modify|modifying|overwrite|overwriting|fine[- ]?tun\w*|retrain\w*|"
                r"updat\w*|train\w*)\s+(?:the\s+)?(?:student|policy|optimizer)\b"
            ),
            _ci(r"\boptimizer\s+state\b"),
            _ci(r"\bgradient\s+(?:descent|updates?)\b"),
        ),
    ),
)


def scan_text(text: str, path: str = "$") -> GuardDecision:
    """Scans a single string against all F1-F7 scanners (fixed order)."""
    if not isinstance(text, str):
        return GuardDecision(
            allowed=False,
            code=GuardCode.UNSUPPORTED_TYPE,
            detail=f"expected str at {path}, got {type(text).__name__}",
            scan="type_check",
            path=path,
        )
    for scan_name, code, patterns in SCANNERS:
        for pattern in patterns:
            m = pattern.search(text)
            if m:
                return _flag(code, scan_name, m.group(0), path)
    return _pass(path)


def scan_llm_output(obj: Any, path: str = "$") -> GuardDecision:
    """Recursively scans JSON-shaped LLM output; fail-closed on violations.

    Dict keys AND values are scanned; lists/tuples recurse; strings are
    scanned; numbers/bools/None are ignored; any other type fails closed.
    """
    if isinstance(obj, str):
        return scan_text(obj, path)
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, float)):
        return _pass(path)
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                return GuardDecision(
                    allowed=False,
                    code=GuardCode.UNSUPPORTED_TYPE,
                    detail=f"non-string dict key at {path}",
                    scan="type_check",
                    path=path,
                )
            key_path = f"{path}.{key}"
            decision = scan_text(key, key_path)
            if not decision.allowed:
                return decision
            decision = scan_llm_output(value, key_path)
            if not decision.allowed:
                return decision
        return _pass(path)
    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            decision = scan_llm_output(item, f"{path}[{i}]")
            if not decision.allowed:
                return decision
        return _pass(path)
    return GuardDecision(
        allowed=False,
        code=GuardCode.UNSUPPORTED_TYPE,
        detail=f"unsupported type {type(obj).__name__} at {path}",
        scan="type_check",
        path=path,
    )


def provenance_guard(value: Any, context: str) -> GuardDecision:
    """Non-raising provenance admissibility gate (design contract S4).

    Wraps ``schemas.assert_admissible_provenance`` into a ``GuardDecision``:
    missing/unknown/FORMAL provenance yields ``allowed=False`` with the
    greppable ``SchemaError`` code.
    """
    try:
        assert_admissible_provenance(value, context)
    except SchemaError as e:
        return GuardDecision(
            allowed=False, code=e.code, detail=str(e), scan="provenance_gate", path=context
        )
    return _pass(context)


def raise_if_forbidden(obj: Any, context: str) -> GuardDecision:
    """Raises ``SchemaError`` (fail-closed) when guards reject ``obj``."""
    decision = scan_llm_output(obj)
    if not decision.allowed:
        raise SchemaError(
            decision.code, f"{context}: guard {decision.scan} rejected output: {decision.detail}"
        )
    return decision
