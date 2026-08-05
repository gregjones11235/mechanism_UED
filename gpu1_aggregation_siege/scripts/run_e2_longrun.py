#!/usr/bin/env python
"""P0-5 entrypoint: direction two's compute-matched LONG-RUN launchers.

Three configurations, ONE frozen architecture (no mode forks):

* ``normal_feedback``      -> ``MODE_NORMAL_FEEDBACK`` (honest
                              candidate<->feedback binding);
* ``no_feedback_control``  -> ``MODE_STATIC_LLM``: the board reads the
                              P0-12 shape-matched MaskedFeedbackView — the
                              same feedback item count / prompt field set,
                              every value a controlled NULL/MASK value (no
                              feedback content);
* ``shuffled_feedback``    -> ``MODE_SHUFFLED_FEEDBACK``: the honest store
                              stays untouched; the board reads a frozen,
                              recomputable permutation under anonymized ids
                              with identity side channels masked.

Compute-match contract (fail closed as ``COMPUTE_MATCH_BROKEN``):

* every mode runs the SAME six-role board (6 logical LLM calls per window),
  the SAME EnvCoder budget (1 unique-template call per window + repair cap
  ``ENVCODER_MAX_REPAIR_ATTEMPTS``), the SAME funnel (64 -> 24 -> 12
  dynamic + 4 anchors), the SAME per-stage Student/Reference episode
  counts, the SAME seed schedule, the SAME anchor-slot count and the SAME
  checkpoint cadence;
* probe transitions per window are therefore IDENTICAL across modes
  (64*(2+1)*128 + 24*(8+4)*128 = 61440) and are accounted as UED overhead;
* ``total_env_steps`` counts STUDENT TRAINING environment steps and must
  equal ``TOTAL_ENV_STEPS_LONG_RUN`` (98304) in every mode — the same
  budget the strong-Student baseline consumed to produce
  PERSISTENT_RMT16_ORIGINAL_VTRACE_98304; the budget is spread as exactly
  one optimizer update per window (98304 / 8 = 12288 steps each).

THIS ROUND THE LONG RUN IS NOT STARTED: ``E2_PILOT_AUTHORIZED`` is False
and the launcher refuses before touching anything. Even with the pilot
flag set, the fail-closed asset gate of the two-window entrypoint applies
unchanged (no real transport, no shared runtime assets locally).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.shared_runtime_binding import SharedRuntimeBundle

#: CLI mode name -> frozen loop mode (NO new modes are forked)
MODE_TO_LOOP_MODE = {
    "normal_feedback": C.MODE_NORMAL_FEEDBACK,
    #: the no-feedback control is the static mode: the shape-matched
    #: MaskedFeedbackView masks every feedback value (controlled NULL/MASK
    #: values, same item count / field set as the normal mode)
    "no_feedback_control": C.MODE_STATIC_LLM,
    "shuffled_feedback": C.MODE_SHUFFLED_FEEDBACK,
}

#: probe transitions per window, derived from the frozen funnel constants:
#: 64 raw * (2 student + 1 reference) fast episodes
#: + 24 stage-1 survivors * (8 student + 4 reference) full episodes,
#: each episode ROLLOUT_LENGTH transitions — identical in every mode
PROBE_TRANSITIONS_PER_WINDOW = (
    (C.RAW_CANDIDATES
     * (C.STAGE1_STUDENT_EPISODES + C.STAGE1_REFERENCE_EPISODES)
     + C.STAGE1_KEEP
     * (C.STAGE2_STUDENT_EPISODES_MAX + C.STAGE2_REFERENCE_EPISODES_MAX))
    * C.ROLLOUT_LENGTH)


def longrun_config(cli_mode: str) -> Dict[str, object]:
    """The compute-match budget of one longrun configuration. Every field
    except ``cli_mode`` / ``loop_mode`` MUST be identical across modes."""
    loop_mode = MODE_TO_LOOP_MODE[cli_mode]
    windows = C.MAX_WINDOWS
    training_env_steps_per_update = C.TOTAL_ENV_STEPS_LONG_RUN // windows
    return dict(
        cli_mode=cli_mode,
        loop_mode=loop_mode,
        windows=windows,
        board_llm_calls_per_window=C.BOARD_CALLS_PER_WINDOW,
        envcoder_calls_per_window=1,
        envcoder_template_id=C.ENVCODER_UNIQUE_TEMPLATE_ID,
        envcoder_repair_cap=C.ENVCODER_MAX_REPAIR_ATTEMPTS,
        llm_calls_per_window=C.LLM_CALLS_PER_WINDOW,
        raw_candidates=C.RAW_CANDIDATES,
        stage1_keep=C.STAGE1_KEEP,
        stage2_keep=C.STAGE2_KEEP,
        dynamic_selected=C.DYNAMIC_UED_SLOTS,
        anchor_slots=C.GLOBAL_ANCHOR_SLOTS,
        stage1_student_episodes=C.STAGE1_STUDENT_EPISODES,
        stage1_reference_episodes=C.STAGE1_REFERENCE_EPISODES,
        stage2_student_episodes=C.STAGE2_STUDENT_EPISODES_MAX,
        stage2_reference_episodes=C.STAGE2_REFERENCE_EPISODES_MAX,
        rollout_length=C.ROLLOUT_LENGTH,
        probe_transitions_per_window=PROBE_TRANSITIONS_PER_WINDOW,
        probe_transitions_total=PROBE_TRANSITIONS_PER_WINDOW * windows,
        optimizer_updates_per_window=1,
        training_env_steps_per_update=training_env_steps_per_update,
        training_env_steps_total=(training_env_steps_per_update * windows),
        total_env_steps=training_env_steps_per_update * windows,
        seed_schedule_hash=C.SEED_SCHEDULE_HASH,
        checkpoint_cadence_windows=1)


def assert_compute_match(configs: List[Dict[str, object]]) -> List[str]:
    """Fail closed on ANY compute mismatch. Returns the problem list; an
    empty list is the only passing state."""
    problems: List[str] = []
    #: 1. every configuration must reconcile to the frozen total budget
    for cfg in configs:
        if cfg["total_env_steps"] != C.TOTAL_ENV_STEPS_LONG_RUN:
            problems.append(
                f"COMPUTE_MATCH_BROKEN: {cfg['cli_mode']} total_env_steps="
                f"{cfg['total_env_steps']} != TOTAL_ENV_STEPS_LONG_RUN="
                f"{C.TOTAL_ENV_STEPS_LONG_RUN}")
        if cfg["training_env_steps_total"] != cfg["total_env_steps"]:
            problems.append(
                f"COMPUTE_MATCH_BROKEN: {cfg['cli_mode']} training budget "
                f"sum {cfg['training_env_steps_total']} != declared total "
                f"{cfg['total_env_steps']}")
    #: 2. every compute field except the mode labels must be IDENTICAL
    #:    across configurations (compute-matched, fail closed)
    comparable_keys = sorted(
        k for k in configs[0] if k not in ("cli_mode", "loop_mode"))
    for key in comparable_keys:
        values = {cfg[key] for cfg in configs}
        if len(values) != 1:
            problems.append(
                f"COMPUTE_MATCH_BROKEN: field {key!r} differs across "
                f"modes: {sorted(map(str, values))}")
    return problems


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direction two: compute-matched longrun launcher "
                    "(NOT started this round: E2_PILOT_AUTHORIZED=false)")
    parser.add_argument("--mode", required=True,
                        choices=sorted(MODE_TO_LOOP_MODE),
                        help="which longrun configuration to validate/"
                             "launch")
    parser.add_argument("--check-only", action="store_true",
                        help="validate the compute-match contract and "
                             "report; never attempt a launch")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    all_configs = [longrun_config(name) for name in sorted(MODE_TO_LOOP_MODE)]
    problems = assert_compute_match(all_configs)
    selected = longrun_config(args.mode)
    report = dict(
        entrypoint="scripts/run_e2_longrun.py",
        requested_mode=args.mode,
        loop_mode=selected["loop_mode"],
        compute_match_problems=problems,
        config=selected,
        all_modes_compute_fields_identical=not problems,
        total_env_steps_required=C.TOTAL_ENV_STEPS_LONG_RUN,
        e2_pilot_authorized=C.E2_PILOT_AUTHORIZED)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False,
                     default=str))
    if problems:
        print("\nLONGRUN REFUSED: compute-match contract is broken; no "
              "mode may launch until every configuration reconciles to "
              f"total_env_steps={C.TOTAL_ENV_STEPS_LONG_RUN} with identical"
              " compute fields", file=sys.stderr)
        return 1
    if args.check_only:
        print("\ncompute-match contract holds; --check-only stops here")
        return 0
    if not C.E2_PILOT_AUTHORIZED:
        print("\nLONGRUN REFUSED: E2_PILOT_AUTHORIZED=false this round — "
              "the long run is NOT started; starting it requires the "
              "explicit pilot authorization plus the shared runtime assets "
              "(see scripts/run_e2_real_two_window.py blocker gate)",
              file=sys.stderr)
        return 1
    #: beyond this point the launch would reuse the SAME fail-closed asset
    #: gate as the two-window entrypoint (real transport + the five shared
    #: assets); this round never reaches it
    missing = SharedRuntimeBundle().missing_assets()
    if missing:
        print(f"\nLONGRUN REFUSED: {C.BLOCKED_WAITING_SHARED_RUNTIME}: "
              f"missing shared assets: {missing}", file=sys.stderr)
        return 1
    print("\nLONGRUN_LAUNCH_NOT_IMPLEMENTED_THIS_ROUND: pilot authorized "
          "and assets present is a state this worktree cannot reach; the "
          "launcher refuses instead of improvising", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
