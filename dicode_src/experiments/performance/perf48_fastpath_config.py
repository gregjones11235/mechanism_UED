#!/usr/bin/env python3
"""Exact config overlays for the fused-preflight benchmark.

Two independent comparisons are supported:

``B4_SINGLE``
    ``B4_OFF`` versus ``B4_ON``. B2 and C are enabled in both arms;
    validation cache is disabled. Only the fused learnability flag differs.

``FINAL_COMBO``
    ``BASELINE`` versus ``FAST_COMBO``. B2, C, fused summary and validation
    cache are disabled in BASELINE and enabled in FAST_COMBO.

Old B3 compaction, train compile cache and embedding cache are forced off in
every arm. Runtime profiling is measurement-only and enabled everywhere.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


COMPARISONS = {
    "B4_SINGLE": ("B4_OFF", "B4_ON"),
    "FINAL_COMBO": ("BASELINE", "FAST_COMBO"),
}
ARMS = tuple(arm for arms in COMPARISONS.values() for arm in arms)
SWITCHES = (
    "preflight_reuse_loaded_tasks",
    "eval_compile_cache",
    "learnability_fused_preflight_summary",
    "validation_cache",
)
FORCED_FALSE = (
    "compact_preflight_payload",
    "train_compile_cache",
    "embedding_cache",
)
EXPECTED_FLAGS = {
    "B4_OFF": {
        "preflight_reuse_loaded_tasks": True,
        "eval_compile_cache": True,
        "learnability_fused_preflight_summary": False,
        "validation_cache": False,
    },
    "B4_ON": {
        "preflight_reuse_loaded_tasks": True,
        "eval_compile_cache": True,
        "learnability_fused_preflight_summary": True,
        "validation_cache": False,
    },
    "BASELINE": {key: False for key in SWITCHES},
    "FAST_COMBO": {key: True for key in SWITCHES},
}
EXPECTED_DIFF_PATHS = {
    "B4_SINGLE": frozenset({"performance.learnability_fused_preflight_summary"}),
    "FINAL_COMBO": frozenset(f"performance.{key}" for key in SWITCHES),
}


def canonical(value: Any) -> Any:
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "Inf"
        if value == float("-inf"):
            return "-Inf"
    if isinstance(value, Mapping):
        return {str(key): canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    return value


def fingerprint(value: Any) -> str:
    payload = json.dumps(canonical(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _set_path(config: dict, path: str, value: Any) -> None:
    current = config
    parts = path.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _get_path(config: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def comparison_arms(comparison: str) -> tuple[str, str]:
    try:
        return COMPARISONS[comparison]
    except KeyError as exc:
        raise ValueError(f"unknown comparison {comparison!r}") from exc


def build_overlay_dict(base: Mapping[str, Any], *, comparison: str, arm: str) -> dict[str, Any]:
    arms = comparison_arms(comparison)
    if arm not in arms:
        raise ValueError(f"arm {arm!r} does not belong to {comparison}: {arms}")
    config = copy.deepcopy(dict(base))
    _set_path(config, "runtime_profiling.enabled", True)
    perf = config.setdefault("performance", {})
    for key, value in EXPECTED_FLAGS[arm].items():
        perf[key] = value
    for key in FORCED_FALSE:
        perf[key] = False
    score = _get_path(config, "dicode_manager.score_function")
    if score is None:
        score = _get_path(config, "training.score_function")
    if EXPECTED_FLAGS[arm]["learnability_fused_preflight_summary"] and score != "learnability":
        raise ValueError("fused preflight benchmark requires score_function=learnability")
    return config


def _diff_paths(left: Mapping[str, Any], right: Mapping[str, Any], prefix: str = "") -> list[str]:
    paths: list[str] = []
    for key in sorted(set(left) | set(right), key=str):
        path = f"{prefix}{key}" if prefix else str(key)
        left_value, right_value = left.get(key), right.get(key)
        if isinstance(left_value, Mapping) and isinstance(right_value, Mapping):
            paths.extend(_diff_paths(left_value, right_value, f"{path}."))
        elif canonical(left_value) != canonical(right_value):
            paths.append(path)
    return sorted(paths)


def normalized_diff(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    return _diff_paths(left, right)


def verify_overlay_pair(
    left: Mapping[str, Any], right: Mapping[str, Any], *, comparison: str
) -> dict[str, Any]:
    arms = comparison_arms(comparison)
    actual = frozenset(normalized_diff(left, right))
    expected = EXPECTED_DIFF_PATHS[comparison]
    if actual != expected:
        raise ValueError(
            f"{comparison} overlay diff mismatch: expected {sorted(expected)}, got {sorted(actual)}"
        )
    for arm, config in zip(arms, (left, right)):
        perf = config.get("performance", {})
        for key, expected_value in EXPECTED_FLAGS[arm].items():
            if bool(perf.get(key, False)) is not expected_value:
                raise ValueError(f"{arm} requires performance.{key}={expected_value}")
        for key in FORCED_FALSE:
            if bool(perf.get(key, False)):
                raise ValueError(f"{arm} requires performance.{key}=false")
        if not bool(_get_path(config, "runtime_profiling.enabled", False)):
            raise ValueError(f"{arm} requires runtime_profiling.enabled=true")
        score = _get_path(config, "dicode_manager.score_function")
        if score is None:
            score = _get_path(config, "training.score_function")
        if EXPECTED_FLAGS[arm]["learnability_fused_preflight_summary"] and score != "learnability":
            raise ValueError("fused preflight benchmark requires score_function=learnability")
    return {
        "valid": True,
        "comparison": comparison,
        "arms": list(arms),
        "diff_paths": sorted(actual),
        "expected_diff_paths": sorted(expected),
        "left_sha256": fingerprint(left),
        "right_sha256": fingerprint(right),
    }


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config is not a mapping: {path}")
    return data


def write_overlay_yaml(
    base_path: str | Path, *, comparison: str, arm: str, out_path: str | Path
) -> dict[str, Any]:
    import yaml

    config = build_overlay_dict(_load_yaml(base_path), comparison=comparison, arm=arm)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(canonical(config), sort_keys=True, allow_unicode=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
        os.replace(temporary, target)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    return config


def load_overlay(path: str | Path) -> dict[str, Any]:
    return _load_yaml(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--comparison", choices=tuple(COMPARISONS), required=True)
    parser.add_argument("--out-left", required=True)
    parser.add_argument("--out-right", required=True)
    args = parser.parse_args()
    left_arm, right_arm = comparison_arms(args.comparison)
    left = write_overlay_yaml(
        args.base, comparison=args.comparison, arm=left_arm, out_path=args.out_left
    )
    right = write_overlay_yaml(
        args.base, comparison=args.comparison, arm=right_arm, out_path=args.out_right
    )
    print(json.dumps(verify_overlay_pair(left, right, comparison=args.comparison), indent=2))
