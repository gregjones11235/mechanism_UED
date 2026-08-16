#!/usr/bin/env python3
"""Config overlay generator for the BC combination experiment (B2 + C).

BC_OFF and BC_ON are derived from the SAME base config (configs/perf48_off.yaml,
the frozen P0 base) and differ ONLY in two performance switches:

  * preflight_reuse_loaded_tasks  (B2)  -- BC_ON true, BC_OFF false
  * eval_compile_cache            (C)   -- BC_ON true, BC_OFF false

Every other optimization switch is FORCED false in BOTH arms:
  * compact_preflight_payload     (B3)  -- must never enter the combination
  * train_compile_cache / embedding_cache / validation_cache

runtime_profiling.enabled is set true in BOTH arms (measurement only; not an
optimization variable). ``normalized_diff`` is the audit tool proving the two
overlays differ on exactly those two switch paths and nothing else.

The module is deliberately pure-python (PyYAML only) so it runs on the CPU-only
dev box and on the server alike; no JAX/OmegaConf import is required for the
overlay logic itself.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

ARMS = ("BC_OFF", "BC_ON")
B2_SWITCH = "preflight_reuse_loaded_tasks"
C_SWITCH = "eval_compile_cache"
# Optimization switches forced false in BOTH arms.
FORCED_FALSE = (
    "compact_preflight_payload",  # B3 must not enter the combination
    "train_compile_cache",
    "embedding_cache",
    "validation_cache",
)
# Switches that are allowed to differ between the two arms (the normalized diff
# gate must match exactly this set).
ALLOWED_DIFF_PATHS = frozenset({
    f"performance.{B2_SWITCH}",
    f"performance.{C_SWITCH}",
})
PERF_SECTION = "performance"
PROFILING_SECTION = "runtime_profiling"


def canonical(value: Any) -> Any:
    """Canonicalize values before comparison/hashing (stable NaN/Inf)."""
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "Inf"
        if value == float("-inf"):
            return "-Inf"
    if isinstance(value, dict):
        return {str(k): canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonical(x) for x in value]
    return value


def fingerprint(value: Any) -> str:
    data = json.dumps(canonical(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _set_path(cfg: dict, path: str, value: Any) -> None:
    """Set cfg[path] creating intermediate dicts; path is dot-separated."""
    parts = path.split(".")
    cur = cfg
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _get_path(cfg: Mapping, path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def build_overlay_dict(base: Mapping[str, Any], *, arm: str) -> dict[str, Any]:
    """Return a deep copy of ``base`` with the BC overlay applied for ``arm``.

    Raises ValueError for an unknown arm. Both arms get
    ``runtime_profiling.enabled = true``; all FORCED_FALSE switches are set
    false; only B2/C differ by arm.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    cfg = copy.deepcopy(dict(base))
    # runtime_profiling.enabled: measurement-only, true for BOTH arms.
    _set_path(cfg, f"{PROFILING_SECTION}.enabled", True)
    perf = cfg.setdefault(PERF_SECTION, {})
    perf[B2_SWITCH] = bool(arm == "BC_ON")
    perf[C_SWITCH] = bool(arm == "BC_ON")
    for key in FORCED_FALSE:
        perf[key] = False
    return cfg


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"config is not a mapping: {path}")
    return data


def write_overlay_yaml(base_path: str | Path, *, arm: str, out_path: str | Path) -> dict[str, Any]:
    """Build the overlay for ``arm`` and write it to ``out_path``.

    The base is loaded once and BOTH arms are derived from the same in-memory
    dict so the serialized pair can only differ on the two switches (key order
    and scalar formatting are identical by construction).
    """
    import yaml

    base = _load_yaml_mapping(base_path)
    cfg = build_overlay_dict(base, arm=arm)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(canonical(cfg), sort_keys=True, default_flow_style=False,
                          allow_unicode=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
        os.replace(tmp, target)
    finally:
        if Path(tmp).exists():
            Path(tmp).unlink()
    return cfg


def _diff_paths(a: Mapping[str, Any], b: Mapping[str, Any], prefix: str = "") -> list[str]:
    """Return the sorted dot-paths where canonicalized ``a`` and ``b`` differ."""
    out: list[str] = []
    keys = sorted(set(a) | set(b), key=str)
    for key in keys:
        path = f"{prefix}{key}" if prefix else str(key)
        av, bv = a.get(key), b.get(key)
        if isinstance(av, Mapping) and isinstance(bv, Mapping):
            if canonical(av) != canonical(bv):
                out.extend(_diff_paths(av, bv, f"{path}."))
        elif canonical(av) != canonical(bv):
            out.append(path)
    return sorted(out)


def normalized_diff(off: Mapping[str, Any], on: Mapping[str, Any]) -> list[str]:
    """Dot-paths where BC_OFF and BC_ON differ (canonicalized, sorted).

    For a valid overlay pair this returns exactly the two switch paths.
    """
    return _diff_paths(off, on)


def verify_overlay_pair(off: Mapping[str, Any], on: Mapping[str, Any]) -> dict[str, Any]:
    """Gate: the only differences between BC_OFF and BC_ON are the B2/C switches.

    Returns a summary dict::

        {
          "valid": bool,
          "diff_paths": [...],
          "unexpected_diff_paths": [...],
          "bc_off_sha256": ..., "bc_on_sha256": ...,
        }

    Raises ValueError when the overlay contract is violated.
    """
    diff = normalized_diff(off, on)
    unexpected = sorted(p for p in diff if p not in ALLOWED_DIFF_PATHS)
    if unexpected:
        raise ValueError(f"overlay diff exceeds B2/C switches: {unexpected}")
    for arm, data in (("BC_OFF", off), ("BC_ON", on)):
        perf = data.get(PERF_SECTION, {})
        for key in FORCED_FALSE:
            if bool(perf.get(key, False)):
                raise ValueError(f"{arm} must force {key}=false")
        if not bool(_get_path(data, f"{PROFILING_SECTION}.enabled", False)):
            raise ValueError(f"{arm} requires runtime_profiling.enabled=true")
    return {
        "valid": True,
        "diff_paths": diff,
        "unexpected_diff_paths": unexpected,
        "bc_off_sha256": fingerprint(off),
        "bc_on_sha256": fingerprint(on),
    }


def load_overlay(path: str | Path) -> dict[str, Any]:
    return _load_yaml_mapping(path)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, help="frozen base config (configs/perf48_off.yaml)")
    p.add_argument("--out-off", required=True)
    p.add_argument("--out-on", required=True)
    args = p.parse_args()
    off = write_overlay_yaml(args.base, arm="BC_OFF", out_path=args.out_off)
    on = write_overlay_yaml(args.base, arm="BC_ON", out_path=args.out_on)
    print(json.dumps(verify_overlay_pair(off, on), indent=2, sort_keys=True))
