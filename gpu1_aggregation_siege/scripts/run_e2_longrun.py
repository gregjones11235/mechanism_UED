#!/usr/bin/env python
"""P0-16 entrypoint: the DiCode-clock E2 launcher (director smoke handoff).

The historical 98304 / 8-window / 12288-steps-per-update long-run protocol
is GONE. The FORMAL training timeline is the frozen DiCode resolved
config's ``training.total_timesteps``, clocked by ``global_env_steps`` —
E2 mode changes ONLY the Feedback View, never the training clock.

* the three E2 configurations (normal / no-feedback / shuffled) SHARE the
  SAME frozen DiCode config — only ``feedback_view_label`` differs;
* probe + LLM overhead is tracked separately in the AuxiliaryComputeLedger
  (a director Runtime Bundle asset) — it is never mixed into the training
  timestep budget;
* the FORMAL entry ONLY PREPARES the Formal Manifest
  (``--formal-manifest-only``) and NEVER launches: the formal experiment
  start requires a HUMAN-approved Formal Manifest, and
  ``FORMAL_EXPERIMENT_AUTHORIZED`` is False this round.

The E2 smoke itself (two windows, window0 delta=0 / window1 delta=1,
total=1) is the DIRECTOR's job — this launcher validates the DiCode clock
consumption and prepares the handoff.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import yaml

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.auxiliary_compute_ledger import (
    AuxiliaryComputeLedger,
)

#: the frozen DiCode resolved config (repo-relative; the formal timeline)
DICODE_TRAINING_YAML = "gpu1_aggregation_siege/conf/training/default.yaml"
DICODE_MANAGER_YAML = \
    "gpu1_aggregation_siege/conf/dicode_manager/default.yaml"
DICODE_CLOCK_FIELD = "global_env_steps"

#: repo root = worktree containing gpu1_aggregation_siege/
REPO_ROOT = Path(__file__).resolve().parents[2]

#: CLI mode name -> frozen loop mode (NO new modes are forked); the E2
#: mode changes ONLY the Feedback View
MODE_TO_LOOP_MODE = {
    "normal_feedback": C.MODE_NORMAL_FEEDBACK,
    "no_feedback_control": C.MODE_STATIC_LLM,
    "shuffled_feedback": C.MODE_SHUFFLED_FEEDBACK,
}

#: feedback-view label per E2 mode (the ONLY thing that differs)
MODE_TO_FEEDBACK_VIEW = {
    "normal_feedback": "normal",
    "no_feedback_control": "masked",
    "shuffled_feedback": "permuted",
}


class DiCodeConfigBlocked(RuntimeError):
    """Fail-closed refusal of the DiCode resolved-config consumption."""


def load_dicode_resolved_config() -> Dict[str, object]:
    """Consume the FROZEN DiCode resolved config (read-only).

    The formal training timeline is ``training.total_timesteps`` clocked
    by ``global_env_steps``; the manager section declares the DiCode 15+1
    batch semantics (original_task_proportion / active_task_capacity /
    training_sample_size_n). A fingerprint over the two files' bytes is
    carried so any drift in the frozen config fails loudly.
    """
    training_path = REPO_ROOT / DICODE_TRAINING_YAML
    manager_path = REPO_ROOT / DICODE_MANAGER_YAML
    try:
        training = yaml.safe_load(training_path.read_text(encoding="utf-8"))
        manager = yaml.safe_load(manager_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DiCodeConfigBlocked(
            f"DICODE_RESOLVED_CONFIG_UNREADABLE: {exc}") from exc
    total_timesteps = int(training.get("total_timesteps", 0))
    if total_timesteps <= 0:
        raise DiCodeConfigBlocked(
            "DICODE_TOTAL_TIMESTEPS_MISSING: the frozen DiCode training "
            f"config must declare a positive training.total_timesteps "
            f"(got {total_timesteps!r}) — the formal timeline is the DiCode "
            "clock, not a direction-two window budget")
    manager_keys = ("original_task_proportion", "active_task_capacity",
                    "training_sample_size_n")
    manager_cfg = {k: manager.get(k) for k in manager_keys}
    fingerprint = hashlib.sha256(
        training_path.read_bytes() + b"::" + manager_path.read_bytes()
    ).hexdigest()
    return dict(
        training=dict(total_timesteps=total_timesteps,
                      clock_field=DICODE_CLOCK_FIELD),
        manager=manager_cfg,
        resolved_config_hash=fingerprint,
        training_yaml=DICODE_TRAINING_YAML,
        manager_yaml=DICODE_MANAGER_YAML)


def dicode_launch_config(cli_mode: str,
                        dicode_config: Dict[str, object]) -> Dict[str, object]:
    """One E2 launch configuration. The DiCode config is SHARED (identical
    across the three modes); only ``feedback_view_label`` differs."""
    return dict(
        cli_mode=cli_mode,
        loop_mode=MODE_TO_LOOP_MODE[cli_mode],
        feedback_view_label=MODE_TO_FEEDBACK_VIEW[cli_mode],
        dicode_config=dict(dicode_config),
        #: auxiliary (non-training-clock) compute per window — recorded in
        #: the AuxiliaryComputeLedger, never in the training budget
        llm_calls_per_window=C.BOARD_CALLS_PER_WINDOW + 1,
        probe_transitions_per_window=(
            (C.RAW_CANDIDATES
             * (C.STAGE1_STUDENT_EPISODES + C.STAGE1_REFERENCE_EPISODES)
             + C.STAGE1_KEEP
             * (C.STAGE2_STUDENT_EPISODES_MAX
                + C.STAGE2_REFERENCE_EPISODES_MAX)) * C.ROLLOUT_LENGTH))


def assert_modes_share_dicode_clock(
        configs: List[Dict[str, object]]) -> List[str]:
    """P0-16: the three E2 modes SHARE the same frozen DiCode config and
    the same training clock; only the Feedback View differs. Fail closed
    on ANY drift. Returns the problem list (empty = passing)."""
    problems: List[str] = []
    shared_keys = ("dicode_config", "llm_calls_per_window",
                   "probe_transitions_per_window")
    for key in shared_keys:
        values = {hashlib.sha256(
            json.dumps(cfg[key], sort_keys=True).encode()).hexdigest()[:16]
            for cfg in configs}
        if len(values) != 1:
            problems.append(
                f"E2_MODES_DICODE_CLOCK_DIVERGED: field {key!r} differs "
                f"across modes: {sorted(values)}")
    views = {cfg["feedback_view_label"] for cfg in configs}
    if views != {"normal", "masked", "permuted"}:
        problems.append(
            f"E2_FEEDBACK_VIEW_SET_INVALID: expected {{normal, masked, "
            f"permuted}}, got {sorted(views)}")
    return problems


def prepare_formal_manifest(cli_mode: str,
                            dicode_config: Dict[str, object]
                            ) -> Dict[str, object]:
    """The FORMAL entry prepares ONLY the Formal Manifest (a preview);
    it NEVER launches. The formal experiment start requires a human-
    approved Formal Manifest; FORMAL_EXPERIMENT_AUTHORIZED is False."""
    return dict(
        kind="DICODE_FORMAL_MANIFEST_PREVIEW",
        prepared_at_entrypoint="scripts/run_e2_longrun.py",
        requested_mode=cli_mode,
        dicode_config=dicode_config,
        formal_experiment_authorized=C.FORMAL_EXPERIMENT_AUTHORIZED,
        status="PREPARED_ONLY_NOT_AUTHORIZED",
        note="the formal experiment start comes ONLY from a HUMAN-approved "
             "Formal Manifest; direction two never auto-starts a formal run")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direction two: DiCode-clock E2 launcher (formal "
                    "launch waits for a human-approved Formal Manifest; "
                    "FORMAL_EXPERIMENT_AUTHORIZED=false this round)")
    parser.add_argument("--mode", required=True,
                        choices=sorted(MODE_TO_LOOP_MODE),
                        help="which E2 configuration to validate / prepare")
    parser.add_argument("--check-only", action="store_true",
                        help="validate the DiCode clock consumption and "
                             "the shared-clock contract; never launch")
    parser.add_argument("--formal-manifest-only", action="store_true",
                        help="prepare ONLY the Formal Manifest preview "
                             "(never launch)")
    parser.add_argument("--manifest-out", default="",
                        help="where to write the Formal Manifest preview")
    parser.add_argument("--report-out", default="",
                        help="optional path to ALSO write the JSON report")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        dicode_config = load_dicode_resolved_config()
    except DiCodeConfigBlocked as exc:
        print(f"\nDICODE CLOCK CONSUMPTION FAILED: {exc}", file=sys.stderr)
        return 1
    all_configs = [dicode_launch_config(name, dicode_config)
                   for name in sorted(MODE_TO_LOOP_MODE)]
    problems = assert_modes_share_dicode_clock(all_configs)
    selected = dicode_launch_config(args.mode, dicode_config)

    report = dict(
        entrypoint="scripts/run_e2_longrun.py",
        requested_mode=args.mode,
        loop_mode=selected["loop_mode"],
        feedback_view_label=selected["feedback_view_label"],
        #: P0-16: the formal timeline is the frozen DiCode clock
        dicode_config=selected["dicode_config"],
        timeline=dict(clock_field=DICODE_CLOCK_FIELD,
                      total_timesteps=(
                          selected["dicode_config"]["training"]
                          ["total_timesteps"])),
        modes_share_dicode_clock=not problems,
        modes_share_dicode_clock_problems=problems,
        #: the 98304 / 8-window / 12288 protocol is REMOVED
        legacy_98304_budget_removed=True,
        formal_experiment_authorized=C.FORMAL_EXPERIMENT_AUTHORIZED,
        e2_real_smoke_authorized=C.E2_REAL_SMOKE_AUTHORIZED)
    if args.manifest_out or args.formal_manifest_only:
        manifest = prepare_formal_manifest(args.mode, dicode_config)
        report["formal_manifest"] = manifest
        if args.manifest_out:
            Path(args.manifest_out).write_text(
                json.dumps(manifest, indent=2, sort_keys=True,
                           ensure_ascii=False, default=str),
                encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False,
                     default=str))
    if args.report_out:
        Path(args.report_out).write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False,
                       default=str), encoding="utf-8")
    if problems:
        print("\nE2 LAUNCH REFUSED: the three modes do not share the DiCode "
              "clock (only the Feedback View may differ); no mode may "
              "launch until the contract holds", file=sys.stderr)
        return 1
    if args.check_only:
        print("\nDiCode clock consumption validated; --check-only stops "
              "here (no launch)")
        return 0
    if args.formal_manifest_only:
        print("\nFORMAL MANIFEST PREPARED (preview only); launch waits for "
              f"a HUMAN-approved Formal Manifest — "
              f"FORMAL_EXPERIMENT_AUTHORIZED="
              f"{C.FORMAL_EXPERIMENT_AUTHORIZED}", file=sys.stderr)
        return 0
    if not C.FORMAL_EXPERIMENT_AUTHORIZED:
        print("\nE2 LAUNCH REFUSED: FORMAL_EXPERIMENT_AUTHORIZED=false — the "
              "formal experiment start requires a human-approved Formal "
              "Manifest; direction two never auto-starts a formal run",
              file=sys.stderr)
        return 1
    print("\nE2_LAUNCH_NOT_IMPLEMENTED_THIS_ROUND: even with formal "
          "authorization the launch would require the director's shared "
          "DiCode runtime + assets (see run_e2_real_two_window.py gate); "
          "this round refuses instead of improvising", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
