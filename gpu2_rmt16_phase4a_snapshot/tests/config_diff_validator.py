#!/usr/bin/env python3
"""Phase4A-v2 Persistent/Reset128 config univariate-contract validator (CC2 directive §九; GATE 14).

The ONLY allowed difference between the two formal Carry-experiment arms is `carry_mode`.
Everything scientific — step0, optimizer, target params, env, task, rollout, sequence_length,
replay mode, batch size, replay schedule, PPO, V-trace, KL, EMA, seed, training budget,
evaluator — MUST be identical.

This validator loads both pre-registered YAMLs, recursively diffs the `scientific_config`
block, and FAILS (non-zero exit / AssertionError) if any leaf other than `carry_mode`
differs, or if either key set differs. The non-scientific top-level `arm` label and the
`runtime_assignment` block (GPU device + output path = hardware/log placement) are EXCLUDED
by design and documented as such.

It also re-asserts the §六 pre-registration invariants (sequence_length=129 crosses the 128
boundary; replay_mode=original_vtrace; hindsight=awr=false).

Usage:
    python config_diff_validator.py [persistent.yaml reset128.yaml]
Exit code 0 => contract holds; 1 => violation (with the offending paths printed).
"""
import os
import sys

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PERSIST = os.path.join(_HERE, "..", "configs", "rmt16_phase4a_v2_persistent.yaml")
_DEFAULT_RESET = os.path.join(_HERE, "..", "configs", "rmt16_phase4a_v2_reset128.yaml")

# Leaf paths within scientific_config that are PERMITTED to differ across arms.
ALLOWED_DIFF_PATHS = {"carry_mode"}


def _flatten(d, prefix=""):
    """Flatten a nested dict/list into {dotted_path: leaf_value}."""
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, p))
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = d
    return out


def validate(persist_path=_DEFAULT_PERSIST, reset_path=_DEFAULT_RESET):
    """Return (ok: bool, report: dict). Raises nothing; the caller decides exit behaviour."""
    with open(persist_path, encoding="utf-8") as f:
        persist = yaml.safe_load(f)
    with open(reset_path, encoding="utf-8") as f:
        reset = yaml.safe_load(f)

    report = {"persistent": os.path.abspath(persist_path),
              "reset128": os.path.abspath(reset_path), "violations": []}

    p_sci = persist.get("scientific_config")
    r_sci = reset.get("scientific_config")
    if p_sci is None or r_sci is None:
        report["violations"].append("missing scientific_config block in one of the configs")
        return False, report

    p_flat = _flatten(p_sci)
    r_flat = _flatten(r_sci)

    # 1. identical key sets (no extra/missing scientific field on either arm)
    only_p = sorted(set(p_flat) - set(r_flat))
    only_r = sorted(set(r_flat) - set(p_flat))
    if only_p:
        report["violations"].append(f"keys only in persistent scientific_config: {only_p}")
    if only_r:
        report["violations"].append(f"keys only in reset128 scientific_config: {only_r}")

    # 2. every differing leaf must be in the allowlist (== {carry_mode})
    diff_paths = sorted(p for p in set(p_flat) & set(r_flat) if p_flat[p] != r_flat[p])
    report["differing_paths"] = diff_paths
    illegal = [p for p in diff_paths if p not in ALLOWED_DIFF_PATHS]
    if illegal:
        report["violations"].append(
            f"scientific_config differs on non-allowed paths: {illegal} "
            f"(only {sorted(ALLOWED_DIFF_PATHS)} may differ)")

    # 3. the one allowed difference must actually be the two carry modes
    if p_sci.get("carry_mode") != "persistent":
        report["violations"].append(
            f"persistent arm carry_mode={p_sci.get('carry_mode')!r} != 'persistent'")
    if r_sci.get("carry_mode") != "reset128":
        report["violations"].append(
            f"reset128 arm carry_mode={r_sci.get('carry_mode')!r} != 'reset128'")

    # 4. §六 pre-registration invariants (identical on both arms by construction)
    for name, sci in (("persistent", p_sci), ("reset128", r_sci)):
        if int(sci.get("sequence_length")) != 129:
            report["violations"].append(f"{name}: sequence_length != 129")
        if int(sci.get("segment_len")) != 128:
            report["violations"].append(f"{name}: segment_len != 128")
        if bool(sci.get("crosses_boundary")) is not True:
            report["violations"].append(f"{name}: crosses_boundary != true")
        if int(sci.get("sequence_length")) <= int(sci.get("segment_len")):
            report["violations"].append(f"{name}: sequence_length does not cross segment_len")
        if sci.get("replay_mode") != "original_vtrace":
            report["violations"].append(f"{name}: replay_mode != original_vtrace")
        if bool(sci.get("hindsight")) is not False:
            report["violations"].append(f"{name}: hindsight != false")
        if bool(sci.get("awr")) is not False:
            report["violations"].append(f"{name}: awr != false")
        if float(sci.get("w_original_vtrace")) != 1.0:
            report["violations"].append(f"{name}: w_original_vtrace != 1.0")

    ok = (not report["violations"]) and (set(diff_paths) == ALLOWED_DIFF_PATHS)
    report["ok"] = ok
    return ok, report


def main(argv):
    persist = argv[1] if len(argv) > 1 else _DEFAULT_PERSIST
    reset = argv[2] if len(argv) > 2 else _DEFAULT_RESET
    ok, report = validate(persist, reset)
    print("GATE14_CONFIG_DIFF_UNIVARIATE=" + ("PASS" if ok else "FAIL"))
    print(f"  differing scientific_config paths: {report.get('differing_paths')}")
    print(f"  allowed diff paths              : {sorted(ALLOWED_DIFF_PATHS)}")
    if report["violations"]:
        print("  VIOLATIONS:")
        for v in report["violations"]:
            print(f"    - {v}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
