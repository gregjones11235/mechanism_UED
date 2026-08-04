"""E1 CC2 follow-up P0-1/P0-21: the SOLE single-real-update gate.

Refactored entry contract::

    python scripts/run_e1_real_one_update.py \
        --runtime-bundle <signed-runtime-bundle-manifest> \
        --report-out <path> \
        --check-only

``--check-only``
    verifies the bundle MANIFEST, verifies every capability contract
    declaration, verifies the one-window driver data flow is
    constructible — and does NOT call any LLM, does NOT probe, does
    NOT train and NEVER writes ``status=EXECUTED``;
full run
    additionally requires EVERY production gate clear AND a
    PRODUCTION bundle signed on the supervisor-owned whitelist AND an
    authorized real LLM provider. This round none of those holds, so
    the full run is honestly BLOCKED (exit non-zero).

CC2 follow-up removals (P0-1): this entrypoint NO LONGER contains
``teacher.evolve()`` (nonexistent), ``stage_real_probe(())`` (empty
candidate set), ``gen_manager=None``, ``rl_train_state=None``, fake
seed banks, fake reset protocols or string adapters. Stage boundaries
carry the REAL objects defined by
``dicode.teachers.e1_formal.one_window_driver``; a bundle capability
is a REAL object or the pipeline stays blocked.

Production hygiene: no tests, no fixtures, no mock defaults, no paid
calls without explicit authorization, no training while any gate is
blocked. Honesty contract: ``real_one_update_executed`` is only true
after the complete pipeline actually ran; it is never hand-set.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_production_runtime as RT  # noqa: E402

ENTRYPOINT = "scripts/run_e1_real_one_update.py"
DEFAULT_REPORT = os.path.join(
    "reports", "e1_formal_ued", "real_one_update_status.json"
)

#: CC2 follow-up stage codes (greppable)
E1_RUNTIME_BUNDLE_MISSING = "E1_RUNTIME_BUNDLE_MISSING"
E1_RUNTIME_BUNDLE_FILE_MISSING = "E1_RUNTIME_BUNDLE_FILE_MISSING"
E1_CHECK_ONLY_OK = "CHECK_ONLY_OK"
E1_CHECK_ONLY_BLOCKED = "CHECK_ONLY_BLOCKED"
E1_PRODUCTION_PIPELINE_UNAUTHORIZED = "E1_PRODUCTION_PIPELINE_UNAUTHORIZED"


def _blocker(stage: str, code: str, detail: str) -> dict:
    return {"stage": stage, "code": code, "detail": detail}


def _resolve_bundle_manifest(args, blockers: list):
    """Load + verify the signed runtime bundle manifest (fail-closed).

    Returns the manifest-level ``E1RuntimeBundle`` record or None with
    an honest blocker appended. The manifest is REQUIRED: the pipeline
    never runs (and never checks) without a signed carrier.
    """
    from dicode.teachers.e1_formal import runtime_bundle as RB

    if not args.runtime_bundle:
        blockers.append(
            _blocker(
                "runtime_bundle",
                E1_RUNTIME_BUNDLE_MISSING,
                "no --runtime-bundle manifest supplied; the one-window "
                "pipeline consumes shared runtime objects ONLY through "
                "a signed bundle (never string contract names)",
            )
        )
        return None
    path = args.runtime_bundle
    if not os.path.isabs(path):
        path = os.path.join(RT.SIEGE_ROOT, path)
    if not os.path.isfile(path):
        blockers.append(
            _blocker(
                "runtime_bundle",
                E1_RUNTIME_BUNDLE_FILE_MISSING,
                f"runtime bundle manifest not found: {path}",
            )
        )
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            mapping = json.load(handle)
    except (OSError, ValueError) as e:
        blockers.append(
            _blocker(
                "runtime_bundle",
                "E1_RUNTIME_BUNDLE_PARSE_FAILED",
                f"cannot parse runtime bundle manifest {path}: {e}",
            )
        )
        return None
    try:
        return RB.load_verified_runtime_bundle(
            mapping, "run_e1_real_one_update.runtime_bundle"
        )
    except RB.RuntimeBundleError as e:
        blockers.append(
            _blocker("runtime_bundle", e.code, str(e))
        )
        return None


def _check_only_report(bundle, gates: dict) -> dict:
    """``--check-only``: verify manifest, capabilities, data flow.

    NO LLM call, NO probe, NO training, and the report NEVER carries
    ``status=EXECUTED`` — check-only is a verification surface only.
    """
    from dicode.teachers.e1_formal import one_window_driver as DRV
    from dicode.teachers.e1_formal import runtime_bundle as RB

    capability_contracts_declared = False
    bundle_mode = ""
    if bundle is not None:
        declared = dict(bundle.object_identity_hashes)
        capability_contracts_declared = all(
            contract in declared and len(declared[contract]) == 64
            for contract in RB.RUNTIME_CAPABILITY_CONTRACTS
        )
        bundle_mode = bundle.mode

    # shared runtime object binding (honest per-contract state)
    shared = gates["shared_runtime"]
    objects_bound = {
        contract: state["bound"] for contract, state in shared.items()
    }

    # data-flow constructibility: the driver surface exists and the
    # full window record declares every required field
    from dataclasses import fields as dataclass_fields

    driver_constructible = all(
        hasattr(DRV, name)
        for name in (
            "E1OneWindowArtifacts",
            "E1WindowResult",
            "E1CandidateMaterials",
            "execute_real_review_window",
            "execute_real_envcoder_and_compile",
            "validate_one_window_artifacts",
            "validate_runtime_surface",
        )
    )
    artifact_fields = [
        field.name
        for field in dataclass_fields(DRV.E1OneWindowArtifacts)
    ]
    required_fields = [
        "runtime_bundle_hash",
        "student_identity",
        "reference_identity",
        "student_adapter",
        "reference_adapter",
        "student_checkpoint_identity",
        "reference_checkpoint_identity",
        "gen_manager",
        "review_window",
        "candidate_materials",
        "executable_candidate_pool",
        "probe_result_pool",
        "criterion_signals_pool",
        "selection_outcome",
        "verified_batch",
        "update_attestation",
        "roundtrip_attestation",
        "run_id",
        "source_commit",
    ]
    dataflow_complete = all(
        name in artifact_fields for name in required_fields
    )

    ok = bool(
        bundle is not None
        and capability_contracts_declared
        and driver_constructible
        and dataflow_complete
    )
    production_blockers = list(gates["blockers"])
    if bundle is not None and bundle.mode == RB.BUNDLE_MODE_TEST_ONLY:
        production_blockers.append(
            _blocker(
                "runtime_bundle",
                RB.RUNTIME_BUNDLE_TEST_ONLY_REJECTED,
                "check-only ran against a TEST_ONLY bundle: it proves "
                "contract connectivity only and is NEVER admissible on "
                "the production path",
            )
        )
    return {
        "entrypoint": ENTRYPOINT,
        "branch": RT.git_branch(),
        "head_sha": RT.git_head_sha(),
        "status": E1_CHECK_ONLY_OK if ok else E1_CHECK_ONLY_BLOCKED,
        "check_only": True,
        "executed": False,  # check-only NEVER executes
        "gates_checked": list(gates["gates_checked"]),
        "checks": {
            "bundle_manifest_verified": bundle is not None,
            "bundle_mode": bundle_mode,
            "bundle_id": bundle.bundle_id if bundle else "",
            "bundle_hash": bundle.bundle_hash if bundle else "",
            "capability_contracts_declared": capability_contracts_declared,
            "shared_runtime_objects_bound": objects_bound,
            "driver_dataflow_constructible": driver_constructible,
            "driver_dataflow_fields_complete": dataflow_complete,
        },
        "production_blockers": production_blockers,
        "note": (
            "check-only verifies manifest + capabilities + data-flow "
            "constructibility; it never calls an LLM, never probes, "
            "never trains and never writes EXECUTED"
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-bundle",
        default="",
        help="path to the signed runtime bundle manifest (REQUIRED for "
        "any execution; absolute or relative to gpu1_aggregation_siege/)",
    )
    parser.add_argument(
        "--teacher-config",
        default=RT.TEACHER_CONFIG_PATH,
        help="teacher config path relative to gpu1_aggregation_siege/",
    )
    parser.add_argument(
        "--llm-provider",
        default="",
        help="real LLM provider identity (must be on the "
        "supervisor-owned whitelist; empty this round)",
    )
    parser.add_argument("--report-out", default=DEFAULT_REPORT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify bundle + capabilities + data-flow constructibility "
        "ONLY (no LLM, no probe, no training, never EXECUTED)",
    )
    args = parser.parse_args(argv)

    # ---- resolve EVERY gate honestly, before anything else -----------
    gates = RT.resolve_production_gates(
        teacher_config_path=args.teacher_config
    )
    gates["gates_checked"].append(RT.GATE_REAL_LLM_PROVIDER)
    try:
        RT.require_real_llm_provider(args.llm_provider)
    except RuntimeError as e:
        gates["blockers"].append(
            _blocker(
                RT.GATE_REAL_LLM_PROVIDER,
                RT.E1_REAL_LLM_NOT_AUTHORIZED,
                str(e),
            )
        )

    # ---- the signed runtime bundle manifest (REQUIRED) ---------------
    gates["gates_checked"].append("runtime_bundle")
    bundle = _resolve_bundle_manifest(args, gates["blockers"])

    if args.check_only:
        report = _check_only_report(bundle, gates)
        path = RT.write_json_report(report, args.report_out)
        print(
            f"E1 REAL ONE UPDATE CHECK-ONLY [{report['status']}] -> "
            f"{path}"
        )
        for name, value in sorted(report["checks"].items()):
            print(f"  {name} = {value}")
        print(
            f"  production blockers = "
            f"{len(report['production_blockers'])}"
        )
        return 0 if report["status"] == E1_CHECK_ONLY_OK else 2

    # ---- full run: every gate + a PRODUCTION bundle ------------------
    from dicode.teachers.e1_formal import runtime_bundle as RB

    if bundle is not None:
        try:
            RB.require_bundle_admissible_for_production(
                bundle, "run_e1_real_one_update.production"
            )
        except RB.RuntimeBundleError as e:
            gates["blockers"].append(
                _blocker("runtime_bundle", e.code, str(e))
            )

    if gates["blockers"]:
        report = RT.blocked_status_report(
            ENTRYPOINT,
            gates,
            extra={
                "llm_provider_requested": args.llm_provider,
                "runtime_bundle_verified": bundle is not None,
                "note": (
                    "every blocker above must clear (signed PRODUCTION "
                    "runtime bundle, shared runtime objects bound, real "
                    "EnvCoder backend authorized, frozen Reference "
                    "contract, frozen anchor manifest, authorized real "
                    "LLM provider) before the single-update pipeline "
                    "may run; stage boundaries then carry the real "
                    "one_window_driver objects — never None, never "
                    "string placeholders, never summary dicts"
                ),
            },
        )
        path = RT.write_json_report(report, args.report_out)
        print(
            f"E1 REAL ONE UPDATE BLOCKED: "
            f"{len(report['blockers'])} blocker(s); report at {path}"
        )
        for blocker in report["blockers"]:
            print(
                f"  - [{blocker['stage']}] {blocker['code']}: "
                f"{blocker['detail']}"
            )
        return 2

    # All gates clear + PRODUCTION bundle verified: the driver's real
    # object flow runs here (board authorization + seam-bound objects
    # land with the CC2 follow-up wiring commits). Reaching this point
    # this round is impossible (the shared runtime is absent and the
    # production signer whitelist is empty), and the driver refuses to
    # continue on any placeholder — fail closed, never fabricated.
    report = RT.blocked_status_report(
        ENTRYPOINT,
        gates,
        extra_blockers=[
            _blocker(
                "pipeline",
                E1_PRODUCTION_PIPELINE_UNAUTHORIZED,
                "all external gates cleared but the authorized six-role "
                "runtime injection is not wired yet (CC2 follow-up); "
                "the pipeline never falls back to a null LLM client",
            )
        ],
    )
    path = RT.write_json_report(report, args.report_out)
    print(f"E1 REAL ONE UPDATE BLOCKED mid-pipeline; report at {path}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
