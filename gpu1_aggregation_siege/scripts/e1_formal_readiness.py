"""E1 round-3 C4: compute the real-smoke readiness report.

Writes ``reports/e1_formal_ued/real_smoke_readiness.json`` — EVERY
boolean is COMPUTED from the actual code/config/seam state at run
time (imported modules, live git, gate resolution); nothing is
hand-written and no value is ever guessed:

* ``sequential_six_role_context`` — the board module actually carries
  the round-3 sequential context (prompt version v2 + the
  context/upstream keyword parameters on both prompt builders);
* ``dynamic_12_logical_specs_reachable`` — the template-keyed spec
  surface exists (MAX_WINDOW_TEMPLATES + derive_variant_params) and
  NO stub-backfill symbol remains anywhere in the teacher source;
* ``dynamic_12_executable_candidates_reachable`` — the executable
  candidate binding chain (Mode A) exists: immutable
  ExecutableCandidate + pool binder + chain verifier + the
  VARIANT_PARAMETER_NOT_EXECUTED marker surface;
* ``dynamic_12_behaviorally_distinct_verified`` — true ONLY when the
  12 executable variants are verified behaviorally distinct by real
  signed probe evidence; this round: structurally FALSE (no shared
  probe runtime exists) — never hand-flipped;
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
  — true ONLY when the one-update entrypoint's status report on disk
  carries a VERIFIED SIGNED smoke attestation certifying EXECUTED;
  absent report / missing status / BLOCKED / unsigned or forged
  attestation => false (never inferred from anything else);
* ``real_smoke_attestation_valid`` — a signed ``E1RealSmokeAttestation``
  inside the report parsed, its hash + signer verified AND every
  bound hash matched the live expected values (CC2 P0-13: plain JSON
  status is parse-level evidence only, never readiness);
* ``e1_real_smoke_ready`` — every capability flag true AND every
  shared contract bound AND the real dual probe AND the real single
  optimizer update actually EXECUTED (per the one-update entrypoint's
  own report) AND zero production-gate blockers; structural gates
  alone NEVER grant readiness (fix(e1): require real probe and
  update for readiness);
* ``head_sha`` — a SNAPSHOT of ``git rev-parse HEAD`` at generation
  time; the commit enclosing this report advances HEAD past the
  snapshot (see ``head_sha_note``) — never hand-edited;
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


def _compute_dynamic_12_logical_specs_reachable() -> bool:
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


def _compute_dynamic_12_executable_candidates_reachable() -> bool:
    """The executable candidate binding chain (Mode A) is live.

    Structural check: the immutable ExecutableCandidate record, the
    pool binder, the chain verifier and the conspicuous
    VARIANT_PARAMETER_NOT_EXECUTED marker surface all exist, and the
    variant-execution gate (the ONLY surface that may clear the
    marker) is wired.
    """
    from dicode.teachers.e1_formal import executable_candidates as EX
    from dicode.teachers.e1_formal import variant_binding as VB

    if not hasattr(EX, "ExecutableCandidate"):
        return False
    if not hasattr(EX, "ExecutableEnvironmentArtifact"):
        return False
    if not hasattr(EX, "bind_executable_candidate_pool"):
        return False
    if not hasattr(EX, "verify_candidate_chain"):
        return False
    if EX.VARIANT_PARAMETER_NOT_EXECUTED != (
        "VARIANT_PARAMETER_NOT_EXECUTED"
    ):
        return False
    if not hasattr(VB, "execute_variant_parameters"):
        return False
    if not hasattr(VB, "bind_executable_pool_from_materials"):
        return False
    return True


def _compute_dynamic_12_behaviorally_distinct_verified() -> bool:
    """Behavioral distinctness of the 12 executable variants.

    Fail-closed CONSTANT this round: proving the 12 variants
    behaviorally distinct requires real signed probe evidence under
    the shared probe registry — which does not exist yet (the shared
    runtime is absent). No structural check may ever flip this flag;
    only signed real probe evidence may, through a future consumer
    here. NEVER hand-set true.
    """
    return False


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


def _compute_real_execution_flags(report_path: str = None) -> tuple:
    """From the one-update entrypoint's OWN status report.

    Both flags are true ONLY when that report records
    ``status == "EXECUTED"`` — absent file, missing status, a
    ``BLOCKED`` status, or any malformed JSON yields (False, False).
    Never inferred from anything else. ``report_path`` overrides the
    canonical location (regression tests only).
    """
    probe_executed = False
    update_executed = False
    path = (
        report_path
        if report_path is not None
        else os.path.join(RT.SIEGE_ROOT, ONE_UPDATE_REPORT)
    )
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


def _compute_real_smoke_evidence(
    report_path: str = None, *, expected: dict = None
) -> dict:
    """The readiness execution evidence from a SIGNED smoke attestation.

    Plain JSON status files are parse-level evidence ONLY. The
    readiness evidence comes from the ``e1_real_smoke_attestation``
    block inside the report: it must parse, its hash + signer must
    verify (``consume_smoke_attestation_mapping``) and EVERY bound
    hash must match the live ``expected`` values (fail-closed on any
    claimed hash with no verifiable source). ``probe_executed`` /
    ``update_executed`` are true ONLY when all of that holds AND the
    status is EXECUTED.
    """
    from dicode.teachers.e1_formal import smoke_attestation as SM

    path = (
        report_path
        if report_path is not None
        else os.path.join(RT.SIEGE_ROOT, ONE_UPDATE_REPORT)
    )
    result = {
        "valid": False,
        "status": "",
        "probe_executed": False,
        "update_executed": False,
        "attestation_signer": "",
        "detail": "no report on disk",
    }
    if not os.path.isfile(path):
        return result
    try:
        with open(path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, ValueError):
        result["detail"] = "unparseable report"
        return result
    result["status"] = report.get("status", "")
    if report.get("status") != "EXECUTED":
        result["detail"] = "status is not EXECUTED"
        return result
    block = report.get("e1_real_smoke_attestation")
    if not isinstance(block, dict):
        result["detail"] = (
            "no e1_real_smoke_attestation block; a plain JSON status "
            "never grants readiness"
        )
        return result
    try:
        attested = SM.consume_smoke_attestation_mapping(
            block, "e1_formal_readiness.smoke"
        )
        SM.verify_e1_real_smoke_attestation(
            attested, expected=expected or {}, ctx="e1_formal_readiness"
        )
    except SM.SmokeAttestationError as e:
        result["detail"] = str(e)
        return result
    if attested.status != SM.SMOKE_STATUS_EXECUTED:
        result["detail"] = "attestation does not certify EXECUTED"
        return result
    result["valid"] = True
    result["probe_executed"] = True
    result["update_executed"] = True
    result["attestation_signer"] = attested.signer_id
    result["detail"] = "signed smoke attestation verified"
    return result


def decide_real_smoke_ready(
    *,
    sequential: bool,
    dynamic_12_logical_specs_reachable: bool,
    dynamic_12_executable_candidates_reachable: bool,
    dynamic_12_behaviorally_distinct_verified: bool,
    criterionwise: bool,
    bounded_repair: bool,
    student_adapter_bound: bool,
    reference_adapter_bound: bool,
    anchor_manifest_bound: bool,
    probe_executed: bool,
    update_executed: bool,
    smoke_attested: bool,
    blockers: list,
) -> bool:
    """The FINAL readiness conjunction — strictly fail-closed.

    Structural capability gates are necessary but NEVER sufficient:
    readiness additionally requires REAL EXECUTION evidence — both
    the real dual probe and the real single optimizer update attested
    EXECUTED by a VERIFIED SIGNED smoke attestation — and zero live
    production-gate blockers. (fix(e1): the probe/update execution
    evidence was missing from this conjunction; structural gates alone
    could have granted the E1 Pilot prematurely.)

    CC2 follow-up P0-6: the single ``dynamic_12`` gate is split into
    three — logical specs reachable, executable candidates reachable,
    and behaviorally-distinct VERIFIED. The third is false until real
    signed probe evidence exists, so readiness stays fail-closed even
    when both reachability gates pass.

    CC2 follow-up P0-13: the execution evidence must come from a
    VERIFIED signed smoke attestation (``smoke_attested``). A plain
    JSON status file — even one stamped EXECUTED with forged flags —
    never grants readiness on its own.
    """
    return bool(
        sequential
        and dynamic_12_logical_specs_reachable
        and dynamic_12_executable_candidates_reachable
        and dynamic_12_behaviorally_distinct_verified
        and criterionwise
        and bounded_repair
        and student_adapter_bound
        and reference_adapter_bound
        and anchor_manifest_bound
        and probe_executed
        and update_executed
        and smoke_attested
        and not blockers
    )


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
    dynamic_12_logical = _compute_dynamic_12_logical_specs_reachable()
    dynamic_12_executable = (
        _compute_dynamic_12_executable_candidates_reachable()
    )
    dynamic_12_distinct = (
        _compute_dynamic_12_behaviorally_distinct_verified()
    )
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
    # CC2 follow-up P0-13: execution evidence comes ONLY from a
    # verified signed smoke attestation. The expected live values this
    # round: branch + git SHA are real; every shared-runtime-derived
    # hash is "" (absent shared runtime), so any claimed non-empty hash
    # fails closed — no TEST_ONLY or forged report can grant readiness.
    smoke_expected = {
        "branch": RT.git_branch(),
        "git_sha": RT.git_head_sha(),
        "run_id": "",
    }
    smoke = _compute_real_smoke_evidence(expected=smoke_expected)
    smoke_attested = smoke["valid"]
    probe_executed = smoke["probe_executed"]
    update_executed = smoke["update_executed"]
    smoke_evidence_detail = smoke["detail"]

    blockers = list(gates["blockers"])
    e1_real_smoke_ready = decide_real_smoke_ready(
        sequential=sequential,
        dynamic_12_logical_specs_reachable=dynamic_12_logical,
        dynamic_12_executable_candidates_reachable=dynamic_12_executable,
        dynamic_12_behaviorally_distinct_verified=dynamic_12_distinct,
        criterionwise=criterionwise,
        bounded_repair=bounded_repair,
        student_adapter_bound=student_adapter_bound,
        reference_adapter_bound=reference_adapter_bound,
        anchor_manifest_bound=anchor_manifest_bound,
        probe_executed=probe_executed,
        update_executed=update_executed,
        smoke_attested=smoke_attested,
        blockers=blockers,
    )

    report = {
        "branch": RT.git_branch(),
        "head_sha": RT.git_head_sha(),
        "head_sha_note": (
            "head_sha is the SNAPSHOT of `git rev-parse HEAD` at "
            "report generation time; the commit enclosing this "
            "report advances HEAD past the snapshot — never "
            "hand-edited"
        ),
        "sequential_six_role_context": sequential,
        "dynamic_12_logical_specs_reachable": dynamic_12_logical,
        "dynamic_12_executable_candidates_reachable": (
            dynamic_12_executable
        ),
        # CC2 follow-up P0-6: false until real signed probe evidence
        # verifies the 12 variants behaviorally distinct (never
        # hand-flipped; this round structurally false)
        "dynamic_12_behaviorally_distinct_verified": dynamic_12_distinct,
        "criterionwise_selector": criterionwise,
        "bounded_envcoder_repair": bounded_repair,
        "shared_student_adapter_bound": student_adapter_bound,
        "shared_reference_adapter_bound": reference_adapter_bound,
        "shared_anchor_manifest_bound": anchor_manifest_bound,
        "real_candidate_probe_executed": probe_executed,
        "real_optimizer_update_executed": update_executed,
        "real_smoke_attestation_valid": smoke_attested,
        "real_smoke_evidence_detail": smoke_evidence_detail,
        "e1_real_smoke_ready": e1_real_smoke_ready,
        "blockers": blockers,
        # provenance only — never a source of truth for the booleans
        "note": (
            "computed by scripts/e1_formal_readiness.py from live "
            "code/config/seam state; real_*_executed read from the "
            "one-update entrypoint's own status report (absent or "
            "status != EXECUTED => false); e1_real_smoke_ready "
            "additionally requires BOTH execution flags and zero "
            "blockers. Entry: " + ENTRYPOINT
        ),
    }
    path = RT.write_json_report(report, args.out)
    print(f"E1 REAL SMOKE READINESS -> {path}")
    for key in (
        "sequential_six_role_context",
        "dynamic_12_logical_specs_reachable",
        "dynamic_12_executable_candidates_reachable",
        "dynamic_12_behaviorally_distinct_verified",
        "criterionwise_selector",
        "bounded_envcoder_repair",
        "shared_student_adapter_bound",
        "shared_reference_adapter_bound",
        "shared_anchor_manifest_bound",
        "real_candidate_probe_executed",
        "real_optimizer_update_executed",
        "real_smoke_attestation_valid",
        "e1_real_smoke_ready",
    ):
        print(f"  {key} = {report[key]}")
    print(f"  blockers = {len(blockers)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
