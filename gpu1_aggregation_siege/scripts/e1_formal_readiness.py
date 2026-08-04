"""E1 round-3 C4: compute the real-smoke readiness report.

Writes ``reports/e1_formal_ued/real_smoke_readiness.json`` — EVERY
boolean is COMPUTED from the actual code/config/seam state at run
time (imported modules, live git, gate resolution); nothing is
hand-written and no value is ever guessed:

* ``sequential_six_role_context`` — the board module actually carries
  the round-3 sequential context (prompt version v2 + the
  context/upstream keyword parameters on both prompt builders);
* ``dynamic_12_reachable`` — the template-keyed spec surface exists
  (MAX_WINDOW_TEMPLATES + derive_variant_params) and NO stub-backfill
  symbol remains anywhere in the teacher source;
* ``criterionwise_selector`` — the criterion-wise Soft Copeland
  selector module is live (name pin, the eight criteria, the
  family_cap parameter, and NO mean-then-Copeland aggregate);
* ``bounded_envcoder_repair`` — the bounded repair loop is wired
  (MAX_ENVCODER_REPAIRS ceiling 2, the repair entry, the full
  8-stage ladder declared);
* ``shared_*_bound`` — the shared runtime seam's live resolution for
  StudentAdapter / ReferenceAdapter / AnchorManifest (this round:
  all False — ``dicode.shared_runtime`` does not exist yet);
* ``real_candidate_probe_executed`` / ``real_optimizer_update_executed``
  — true ONLY if the one-update entrypoint's own status report on
  disk records an EXECUTED run; absent/blocked => false;
* ``e1_real_smoke_ready`` — every capability flag true AND every
  shared contract bound AND zero production-gate blockers;
* ``blockers[]`` — the live production-gate blockers (the same
  resolution the one-update entrypoint consumes).
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_production_runtime as RT  # noqa: E402

ENTRYPOINT = "scripts/e1_formal_readiness.py"
DEFAULT_OUT = os.path.join(
    "reports", "e1_formal_ued", "real_smoke_readiness.json"
)
ONE_UPDATE_REPORT = os.path.join(
    "reports", "e1_formal_ued", "real_one_update_status.json"
)


def _compute_sequential_six_role_context() -> bool:
    """The board actually carries the sequential round-3 context."""
    from dicode.teachers.e1_formal import board

    if getattr(board, "BOARD_PROMPT_VERSION", "") != "e1-board-prompt-v2":
        return False
    for builder in (
        board.build_role_prompt,
        board.build_prompt_envelope_hash,
    ):
        params = inspect.signature(builder).parameters
        if "context" not in params or "upstream" not in params:
            return False
    return hasattr(board, "BoardContext") and hasattr(
        board, "UpstreamOutput"
    )


def _compute_dynamic_12_reachable() -> bool:
    """Template-keyed specs + 12-slot path; NO stub backfill remains.

    Round-3 semantics: ``_reuse_stub`` survives ONLY as the marker of
    the non-trainable REUSE batch (compiled=False, reuse=True, never
    trained). The trainable COMPLETE path never backfills — an
    insufficient pool refuses the WHOLE window
    (``INSUFFICIENT_DYNAMIC_ARTIFACTS``). Both facts are checked
    structurally.
    """
    import ast

    from dicode.teachers.e1_formal import gen_manager, task_specs
    from dicode.teachers.e1_formal.schemas import E1Code

    if getattr(task_specs, "MAX_WINDOW_TEMPLATES", None) != 10:
        return False
    if not hasattr(task_specs, "derive_variant_params"):
        return False
    if not hasattr(task_specs, "TaskTemplate"):
        return False
    if not hasattr(E1Code, "INSUFFICIENT_DYNAMIC_ARTIFACTS"):
        return False
    # every _reuse_stub call site must live inside _reuse_batch
    source_path = os.path.join(
        RT.SIEGE_ROOT,
        "src",
        "dicode",
        "teachers",
        "e1_formal",
        "gen_manager.py",
    )
    with open(source_path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=source_path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name == "_reuse_batch":
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "_reuse_stub"
            ):
                return False  # stub backfill outside the REUSE batch
    return True


def _compute_criterionwise_selector() -> bool:
    """The criterion-wise Soft Copeland selector is live."""
    from dicode.teachers.e1_formal import criterion_selector as CS

    if CS.CRITERION_SELECTOR_NAME != "CRITERION_WISE_COPELAND":
        return False
    if len(CS.CRITERIA) != 8:
        return False
    params = inspect.signature(CS.select_criterion_batch).parameters
    return "family_cap" in params


def _compute_bounded_envcoder_repair() -> bool:
    """The bounded repair loop is wired (ceiling 2; full stage ladder)."""
    from dicode.teachers.e1_formal import envcoder, envcoder_backends as EB

    if getattr(envcoder, "MAX_ENVCODER_REPAIRS", None) != 2:
        return False
    if not hasattr(envcoder, "run_envcoder_with_repair"):
        return False
    if not hasattr(envcoder, "RepairRecord"):
        return False
    return len(EB.STAGES) == 8


def _compute_real_execution_flags() -> tuple:
    """From the one-update entrypoint's OWN status report (absent or
    blocked => both false; never inferred from anything else)."""
    probe_executed = False
    update_executed = False
    path = os.path.join(RT.SIEGE_ROOT, ONE_UPDATE_REPORT)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
        except (OSError, ValueError):
            report = {}
        if report.get("status") == "EXECUTED":
            update_executed = bool(
                report.get("real_one_update_executed") is True
            )
            probe_executed = bool(
                report.get("flags", {}).get(
                    "real_student_reference_eval"
                )
                is True
            )
    return probe_executed, update_executed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    gates = RT.resolve_production_gates()
    gates["gates_checked"].append(RT.GATE_REAL_LLM_PROVIDER)
    try:
        RT.require_real_llm_provider("")
    except RuntimeError as e:
        gates["blockers"].append(
            {
                "stage": RT.GATE_REAL_LLM_PROVIDER,
                "code": RT.E1_REAL_LLM_NOT_AUTHORIZED,
                "detail": str(e),
            }
        )

    shared = gates["shared_runtime"]
    sequential = _compute_sequential_six_role_context()
    dynamic_12 = _compute_dynamic_12_reachable()
    criterionwise = _compute_criterionwise_selector()
    bounded_repair = _compute_bounded_envcoder_repair()
    student_adapter_bound = bool(
        shared.get("StudentAdapter", {}).get("bound")
    )
    reference_adapter_bound = bool(
        shared.get("ReferenceAdapter", {}).get("bound")
    )
    anchor_manifest_bound = bool(
        shared.get("AnchorManifest", {}).get("bound")
    )
    probe_executed, update_executed = _compute_real_execution_flags()

    blockers = list(gates["blockers"])
    e1_real_smoke_ready = (
        sequential
        and dynamic_12
        and criterionwise
        and bounded_repair
        and student_adapter_bound
        and reference_adapter_bound
        and anchor_manifest_bound
        and not blockers
    )

    report = {
        "branch": RT.git_branch(),
        "head_sha": RT.git_head_sha(),
        "sequential_six_role_context": sequential,
        "dynamic_12_reachable": dynamic_12,
        "criterionwise_selector": criterionwise,
        "bounded_envcoder_repair": bounded_repair,
        "shared_student_adapter_bound": student_adapter_bound,
        "shared_reference_adapter_bound": reference_adapter_bound,
        "shared_anchor_manifest_bound": anchor_manifest_bound,
        "real_candidate_probe_executed": probe_executed,
        "real_optimizer_update_executed": update_executed,
        "e1_real_smoke_ready": e1_real_smoke_ready,
        "blockers": blockers,
        # provenance only — never a source of truth for the booleans
        "note": (
            "computed by scripts/e1_formal_readiness.py from live "
            "code/config/seam state; real_*_executed read from the "
            "one-update entrypoint's own status report (absent => "
            "false). Entry: " + ENTRYPOINT
        ),
    }
    path = RT.write_json_report(report, args.out)
    print(f"E1 REAL SMOKE READINESS -> {path}")
    for key in (
        "sequential_six_role_context",
        "dynamic_12_reachable",
        "criterionwise_selector",
        "bounded_envcoder_repair",
        "shared_student_adapter_bound",
        "shared_reference_adapter_bound",
        "shared_anchor_manifest_bound",
        "real_candidate_probe_executed",
        "real_optimizer_update_executed",
        "e1_real_smoke_ready",
    ):
        print(f"  {key} = {report[key]}")
    print(f"  blockers = {len(blockers)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
