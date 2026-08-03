"""Deterministic JSON extraction from LLM text (fail-closed).

Single-pass, pure standard library: locate the first balanced JSON
object/array in the text and parse it. No retries, no fallbacks, no
repair (repair calls are tracked separately in the LLM accounting
ledger; this round allows none).
"""
from __future__ import annotations

import json
from typing import Any

from .schemas import E1Code, E1SchemaError

_OPEN = "{"
_CLOSE = "}"
_OPEN_ARR = "["
_CLOSE_ARR = "]"


def extract_json_block(text: Any, context: str = "json_parse") -> Any:
    """Extract and parse the first balanced JSON object/array in ``text``.

    Raises:
        E1SchemaError: JSON_NOT_FOUND / JSON_PARSE_FAILED (greppable).
    """
    if not isinstance(text, str):
        raise E1SchemaError(
            E1Code.JSON_NOT_FOUND,
            f"{context}: expected str, got {type(text).__name__}",
        )
    start = -1
    for i, ch in enumerate(text):
        if ch in (_OPEN, _OPEN_ARR):
            start = i
            break
    if start < 0:
        raise E1SchemaError(
            E1Code.JSON_NOT_FOUND, f"{context}: no JSON object/array found"
        )

    opener = text[start]
    closer = _CLOSE if opener == _OPEN else _CLOSE_ARR
    depth = 0
    in_string = False
    escaped = False
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        raise E1SchemaError(
            E1Code.JSON_PARSE_FAILED,
            f"{context}: unbalanced JSON starting at offset {start}",
        )
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise E1SchemaError(
            E1Code.JSON_PARSE_FAILED,
            f"{context}: invalid JSON at offset {start}: {e.msg}",
        ) from e
