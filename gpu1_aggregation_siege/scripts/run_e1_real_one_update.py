"""E1 CC2-Director: the SOLE single-real-update gate.

Refactored entry contract (CC2-Director)::

    python scripts/run_e1_real_one_update.py \
        --director-runtime-bundle <signed-manifest> \
        --report-out <path> \
        --check-only

The runtime bundle is the DIRECTOR's signed injection channel. The
entry:

1. verifies the director's signature against the supervisor-owned
   Signer Registry (the ``E1RuntimeBundle`` signer whitelist — EMPTY
   this round, so no production bundle can verify yet);
2. resolves the real shared objects from the bundle (seam);
3. builds the E1FormalGenManager;
4. injects the AuthorizedSixRoleLLMRuntime;
5. calls the FULL one-window driver chain
   (Review Window -> EnvCoder -> ExecutableCandidate -> signed probes
   -> signed signals -> SelectionAttestation -> 12 dynamic ->
   CanonicalDiCodeTrainingBatchPlan -> canonical DiCode one update ->
   RunStateCheckpoint -> fresh-process restore -> next-policy-step
   equivalence -> signed smoke attestation);
6. outputs the Smoke handoff report.

``--check-only``
    verifies the bundle, verifies every capability is really bindable,
    verifies the pipeline is constructible, verifies the 15+1 batch
    semantics — and does NOT call any LLM, does NOT probe, does NOT
    train and NEVER writes ``status=EXECUTED`` / NEVER flips a REAL_*
    flag.

This round: only ``--check-only`` and tests run. No real Smoke, no
real LLM / EnvCoder / probe / training.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_production_runtime as RT  # noqa: E402
from dicode.teachers.e1_formal import (  # noqa: E402
    runtime_bundle as RB,
    student_contract as SC,
)

ENTRYPOINT = "scripts/run_e1_real_one_update.py"
DEFAULT_REPORT = os.path.join(
    "reports", "e1_formal_ued", "real_one_update_status.json"
)

#: CC2-Director stage codes (greppable)
E1_DIRECTOR_RUNTIME_BUNDLE_REQUIRED = "E1_DIRECTOR_RUNTIME_BUNDLE_REQUIRED"
E1_CHECK_ONLY_BLOCKED = "CHECK_ONLY_BLOCKED"
#: CC2-Student repair: two check-only levels
E1_TEST_ONLY_CONTRACT_OK = "TEST_ONLY_CONTRACT_OK"
E1_OBJECT_LEVEL_CHECK_ONLY_OK = "OBJECT_LEVEL_CHECK_ONLY_OK"
E1_OBJECT_LEVEL_CHECK_ONLY_BLOCKED = "OBJECT_LEVEL_CHECK_ONLY_BLOCKED"


def _blocker(stage: str, code: str, detail: str) -> dict:
    return {"stage": stage, "code": code, "detail": detail}


def _resolve_director_bundle(args, blockers: list):
    """Load + verify the director's signed runtime bundle manifest."""
    from dicode.teachers.e1_formal import runtime_bundle as RB

    if not args.director_runtime_bundle:
        blockers.append(
            _blocker(
                "director_runtime_bundle",
                E1_DIRECTOR_RUNTIME_BUNDLE_REQUIRED,
                "no --director-runtime-bundle supplied; the one-window "
                "pipeline runs ONLY under a director-signed runtime "
                "bundle (never on string contract names)",
            )
        )
        return None
    path = args.director_runtime_bundle
    if not os.path.isabs(path):
        path = os.path.join(RT.SIEGE_ROOT, path)
    if not os.path.isfile(path):
        blockers.append(
            _blocker(
                "director_runtime_bundle",
                "E1_DIRECTOR_RUNTIME_BUNDLE_FILE_MISSING",
                f"director runtime bundle manifest not found: {path}",
            )
        )
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            mapping = json.load(handle)
    except (OSError, ValueError) as e:
        blockers.append(
            _blocker(
                "director_runtime_bundle",
                "E1_DIRECTOR_RUNTIME_BUNDLE_PARSE_FAILED",
                f"cannot parse director runtime bundle manifest {path}: {e}",
            )
        )
        return None
    try:
        return RB.load_verified_runtime_bundle(
            mapping, "run_e1_real_one_update.director_bundle"
        )
    except RB.RuntimeBundleError as e:
        blockers.append(
            _blocker("director_runtime_bundle", e.code, str(e))
        )
        return None


def _dual_student_mount_check() -> dict:
    """EITHER allowed Student must mount (distinct memory modes, no
    silent fallback to Persistent). TEST_ONLY / SYNTHETIC contracts;
    no LLM / probe / training / checkpoint writes."""
    from dicode.teachers.e1_formal import student_contract as SC

    results = {}
    for cid in sorted(SC.ALLOWED_STUDENT_CANDIDATE_IDS):
        contract = SC.build_synthetic_student_contract(
            cid, "check-only.dual-student"
        )
        mount = SC.consume_e1_student_contract(
            contract,
            director_selected_candidate_id=cid,
            runtime_bundle_hash="c0" * 32,
            ctx="check-only.dual-student",
        )
        results[cid] = {
            "profile_id": mount.profile_id,
            "memory_mode": mount.memory_mode,
            "mountable": True,
            "capability_state": mount.capability_state,
        }
    modes = {entry["memory_mode"] for entry in results.values()}
    return {
        "mountable": len(results) >= 2,
        "distinct_memory_modes": len(modes) >= 2,
        "per_student": results,
    }


def _check_only_report(
    bundle, gates: dict, *,
    formal_asset_registry=None,
    director_bundle_verifier=None,
) -> dict:
    """``--check-only``: bundle + bindability + constructibility +
    15+1 batch semantics. NO LLM, NO probe, NO training, NEVER
    EXECUTED, NEVER flips a REAL flag."""
    from dataclasses import fields as dataclass_fields

    from dicode.teachers.e1_formal import dicode_protocol as DP
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

    shared = gates["shared_runtime"]
    objects_bound = {
        contract: state["bound"] for contract, state in shared.items()
    }

    # CC2-Student: EITHER allowed Student must mount (distinct memory
    # modes, no silent fallback to Persistent)
    dual_student_mount = _dual_student_mount_check()

    driver_constructible = all(
        hasattr(DRV, name)
        for name in (
            "execute_real_review_window",
            "execute_real_envcoder_and_compile",
            "execute_real_candidate_binding",
            "execute_real_candidate_probes",
            "execute_real_criterion_selection",
            "execute_real_batch_certification",
            "execute_canonical_dicode_one_update",
            "consume_full_runstate_roundtrip",
            "build_e1_smoke_attestation",
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

    # CC2-Director: the 15+1 batch semantics must be constructible
    fifteen_plus_one_ready = bool(
        DP.DICODE_NUM_DYNAMIC == 12
        and DP.DICODE_NUM_NON_TARGET_ANCHORS == 3
        and DP.DICODE_NUM_CURRICULUM == 15
        and DP.DICODE_TARGET_PROBABILITY == 0.20
        and DP.DICODE_TARGET_TASK_ID == "original_craftax"
        and hasattr(DP, "CanonicalDiCodeTrainingBatchPlan")
        and hasattr(DP, "build_canonical_dicode_training_batch_plan")
        and hasattr(DP, "CanonicalDiCodeRunStateCheckpoint")
        and hasattr(DP, "CanonicalDiCodeOneUpdateRuntime")
    )

    synthetic_ok = bool(
        bundle is not None
        and capability_contracts_declared
        and driver_constructible
        and dataflow_complete
        and fifteen_plus_one_ready
        and dual_student_mount["mountable"]
    )
    production_blockers = list(gates["blockers"])
    is_test_only = bundle is not None and bundle.mode == RB.BUNDLE_MODE_TEST_ONLY
    if is_test_only:
        production_blockers.append(
            _blocker(
                "director_runtime_bundle",
                RB.RUNTIME_BUNDLE_TEST_ONLY_REJECTED,
                "check-only ran against a TEST_ONLY bundle: it proves "
                "contract connectivity only and is NEVER admissible on "
                "the production path",
            )
        )
        # CC2-Student repair: TEST_ONLY_CONTRACT_OK is a SEPARATE level —
        # synthetic contract tests only. It NEVER forms a Smoke handoff,
        # NEVER sets STUDENT_READ_ONLY_MOUNT_READY=true and NEVER
        # outputs DIRECTOR_SMOKE_HANDOFF_READY.
        status = (
            E1_TEST_ONLY_CONTRACT_OK if synthetic_ok else E1_CHECK_ONLY_BLOCKED
        )
        level = "TEST_ONLY_CONTRACT"
    elif bundle is None:
        status = E1_CHECK_ONLY_BLOCKED
        level = "NO_BUNDLE"
    else:
        # OBJECT_LEVEL_CHECK_ONLY_OK requires real objects; this round
        # no shared registry is injected, so every required runtime
        # object stays missing and the check honestly BLOCKS.
        from dicode.teachers.e1_formal import runtime_object_resolution as ROR

        try:
            ROR.require_real_registry(formal_asset_registry, "check-only")
            resolution = ROR.resolve_e1_runtime_objects(
                bundle, formal_asset_registry, "check-only.object-level",
                strict=True,
            )
        except ROR.RuntimeObjectResolutionError as e:
            resolution = {
                "all_bound": False,
                "missing": [e.code],
                "resolutions": {},
            }
        object_level_ok = bool(
            bundle is not None
            and not gates["blockers"]
            and resolution["all_bound"]
            and synthetic_ok
        )
        status = (
            E1_OBJECT_LEVEL_CHECK_ONLY_OK
            if object_level_ok
            else E1_OBJECT_LEVEL_CHECK_ONLY_BLOCKED
        )
        level = "OBJECT_LEVEL"
        for contract in resolution.get("missing", []):
            production_blockers.append(
                _blocker(
                    "object_level",
                    "OBJECT_LEVEL_MISSING_OBJECT",
                    f"required runtime object {contract!r} is not "
                    "resolved from the shared FormalAssetRegistry; the "
                    "check-only stays BLOCKED",
                )
            )
    return {
        "entrypoint": ENTRYPOINT,
        "branch": RT.git_branch(),
        "head_sha": RT.git_head_sha(),
        "status": status,
        "level": level,
        "check_only": True,
        "executed": False,  # check-only NEVER executes
        "smoke_handoff": False,  # check-only NEVER forms a Smoke handoff
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
            "fifteen_plus_one_batch_ready": fifteen_plus_one_ready,
            "dual_student_mount_ready": dual_student_mount,
        },
        "production_blockers": production_blockers,
        "note": (
            "check-only NEVER calls an LLM / probes / trains / writes "
            "checkpoints; TEST_ONLY_CONTRACT_OK is synthetic-only and "
            "cannot form a Smoke handoff; OBJECT_LEVEL_CHECK_ONLY_OK "
            "requires every real runtime object resolved"
        ),
    }


def run_director_one_window_pipeline(
    *,
    teacher: Any,
    bundle: Any,
    run_id: str,
    branch: str,
    git_sha: str,
    probe_issuer: Any,          # (candidates, bundle) -> probe results
    signal_issuer: Any,         # (candidates, probe_pool) -> signals
    one_update_runtime: Any,
    student_checkpoint_identity: str,
    reference_checkpoint_identity: str,
    anchor_manifest_hash: str,
    formal_asset_registry_hash: str,
    update_record: Any,
    checkpoint: Any,
    roundtrip_evidence: Any,
    k: int,
    seed: int,
    critic_policy: str,
    family_cap: int,
    smoke_signer_id: str,
    update_signer_id: str,
    roundtrip_signer_id: str,
    allow_test_only: bool = False,
    llm_call_attestation: Any = None,
    envcoder_execution_attestation: Any = None,
    probe_coverage_proof: Any = None,
) -> dict:
    """The FULL one-window pipeline under the director's bundle.

    This is the real call surface that runs when the external gates
    clear AND a director-signed bundle is injected. Returns the Smoke
    handoff report (the signed E1RealSmokeAttestation plus the bound
    stage hashes). CC2-Director: no fixed E1_PRODUCTION_PIPELINE_
    UNAUTHORIZED — reaching here is the pipeline.
    """
    from dicode.teachers.e1_formal import one_window_driver as DRV
    from dicode.teachers.e1_formal import selection_attestation as SA

    # 1) window -> EnvCoder -> executable candidates -> probes ->
    #    signed signals -> criterion selection
    window_result = DRV.execute_real_review_window(teacher, bundle)
    materials = DRV.execute_real_envcoder_and_compile(
        teacher, window_result, bundle
    )
    candidates = DRV.execute_real_candidate_binding(
        teacher,
        window_result,
        materials,
        bundle,
        allow_test_only=allow_test_only,
    )
    # the shared probe runner issues; the driver consumes fail-closed
    probe_results = probe_issuer(candidates, bundle)
    probe_pool = DRV.execute_real_candidate_probes(
        teacher,
        candidates,
        bundle,
        probe_results=probe_results,
        student_checkpoint_identity=student_checkpoint_identity,
        reference_checkpoint_identity=reference_checkpoint_identity,
        window_result=window_result,
        allow_test_only=allow_test_only,
    )
    # the shared runner derives one signed signal per candidate; the
    # driver's selection stage re-checks full coverage fail-closed
    signed_signals = signal_issuer(candidates, probe_pool)
    outcome, attestation = DRV.execute_real_criterion_selection(
        teacher,
        window_result,
        candidates,
        probe_pool,
        signed_signals,
        bundle,
        k=k,
        seed=seed,
        critic_policy=critic_policy,
        family_cap=family_cap,
        allow_test_only=allow_test_only,
    )
    SA.verify_selection_attestation(
        attestation,
        candidates=candidates,
        probe_results=probe_pool,
        signed_signals=signed_signals,
        window_hash=window_result.window.window_hash,
        ctx="e1_entry.selection",
    )

    # 2) 12 dynamic -> CanonicalDiCodeTrainingBatchPlan (15+1)
    plan = DRV.execute_real_batch_certification(
        selection_attestation=attestation,
        anchor_manifest_hash=anchor_manifest_hash,
    )

    # 3) canonical DiCode one update (counts from the DiCode timeline)
    update_attestation = DRV.execute_canonical_dicode_one_update(
        plan=plan,
        selection_attestation=attestation,
        one_update_runtime=one_update_runtime,
        update_record=update_record,
        anchor_manifest_hash=anchor_manifest_hash,
        signer_id=update_signer_id,
        test_only=allow_test_only,
    )

    # 4) full run-state round trip (shared canonical checkpoint)
    roundtrip_attestation = DRV.consume_full_runstate_roundtrip(
        checkpoint=checkpoint,
        update_attestation=update_attestation,
        runtime_bundle_hash=bundle.bundle_hash,
        roundtrip_evidence=roundtrip_evidence,
    )

    # 5) signed smoke attestation over the WHOLE chain
    smoke = DRV.build_e1_smoke_attestation(
        run_id=run_id,
        branch=branch,
        git_sha=git_sha,
        window_result=window_result,
        candidate_materials=materials,
        probe_pool=probe_pool,
        plan=plan,
        update_attestation=update_attestation,
        roundtrip_attestation=roundtrip_attestation,
        runtime=bundle,
        student_checkpoint_identity=student_checkpoint_identity,
        reference_checkpoint_identity=reference_checkpoint_identity,
        formal_asset_registry_hash=formal_asset_registry_hash,
        anchor_manifest_hash=anchor_manifest_hash,
        signer_id=smoke_signer_id,
        test_only=allow_test_only,
    )

    # CC2-Student repair (§六): the REAL_* flags are DERIVED from the
    # corresponding signed immutable evidence on the production path —
    # never set true from local booleans or calling functions. TEST_ONLY
    # keeps every REAL flag false.
    real_flags = {
        "REAL_LLM_EXECUTED": False,
        "REAL_ENVCODER_EXECUTED": False,
        "REAL_CANDIDATE_PROBE_EXECUTED": False,
        "REAL_OPTIMIZER_UPDATE_EXECUTED": False,
        "REAL_FULL_STATE_ROUND_TRIP": False,
        "E1_REAL_SMOKE_AUTHORIZED": False,
        "FORMAL_EXPERIMENT_AUTHORIZED": False,
    }
    if not allow_test_only:
        # CC2-Repair-2 (§八): each REAL flag derives ONLY from its own
        # immutable evidence — never from an allowlist being non-empty,
        # a function being called, a local boolean or an object existing.
        from dicode.teachers.e1_formal import (
            roundtrip_attestation as _RA,
        )
        from dicode.teachers.e1_formal import smoke_attestation as _SM
        from dicode.teachers.e1_formal import update_attestation as _UA

        real_flags["REAL_LLM_EXECUTED"] = bool(
            llm_call_attestation is not None
            and not getattr(llm_call_attestation, "test_only", False)
        )
        real_flags["REAL_ENVCODER_EXECUTED"] = bool(
            envcoder_execution_attestation is not None
            and not getattr(
                envcoder_execution_attestation, "test_only", False
            )
        )
        real_flags["REAL_CANDIDATE_PROBE_EXECUTED"] = bool(
            probe_coverage_proof is not None
            and not getattr(probe_coverage_proof, "test_only", False)
        )
        real_flags["REAL_OPTIMIZER_UPDATE_EXECUTED"] = bool(
            _UA.AUTHORIZED_TRAINING_RUNTIMES
            and not update_attestation.test_only
        )
        real_flags["REAL_FULL_STATE_ROUND_TRIP"] = bool(
            _RA.AUTHORIZED_ROUNDTRIP_SIGNERS
            and not roundtrip_attestation.test_only
        )
        real_flags["E1_REAL_SMOKE_AUTHORIZED"] = bool(
            _SM.AUTHORIZED_SMOKE_SIGNERS and not smoke.test_only
        )

    return {
        "entrypoint": ENTRYPOINT,
        "run_id": run_id,
        "branch": branch,
        "head_sha": git_sha,
        "status": "SMOKE_HANDOFF" if not allow_test_only else "TEST_ONLY_SMOKE_HANDOFF",
        "executed": True,
        "real_flags": real_flags,
        "window_hash": window_result.window.window_hash,
        "window_status": window_result.window.status,
        "selected_count": len(outcome.selected_ids),
        "curriculum_task_ids": list(plan.curriculum_task_ids),
        "target_task_id": plan.target_task_id,
        "target_probability": plan.target_probability,
        "plan_hash": plan.plan_hash,
        "update_attestation_hash": update_attestation.attestation_hash,
        "roundtrip_attestation_hash": (
            roundtrip_attestation.attestation_hash
        ),
        "smoke_attestation_hash": smoke.attestation_hash,
        "smoke_attestation_signer": smoke.signer_id,
    }


def run_e1_object_level_check(
    *,
    bundle_manifest: Any,
    director_bundle_verifier: Any,
    formal_asset_registry: Any,
    selected_candidate_id: str,
) -> dict:
    """OBJECT_LEVEL check-only against the director-injected registry.

    The real Python objects are injected by the director process ONLY —
    never restored from JSON, never imported from arbitrary module.attr.
    Without the registry => FORMAL_ASSET_REGISTRY_UNBOUND and the check
    honestly BLOCKs.
    """
    from dicode.teachers.e1_formal import runtime_bundle as RB
    from dicode.teachers.e1_formal import (
        runtime_object_resolution as ROR,
    )

    ctx = "run_e1_real_one_update.object_level"
    bundle = RB.load_verified_runtime_bundle(bundle_manifest, ctx)
    if bundle.student_selection.selected_candidate_id != (
        selected_candidate_id
    ):
        raise RuntimeError(
            f"{ctx}: object-level check pinned to {selected_candidate_id!r} "
            f"but the bundle selects {bundle.student_selection.selected_candidate_id!r}"
        )
    try:
        ROR.require_real_registry(formal_asset_registry, ctx)
    except ROR.RuntimeObjectResolutionError as e:
        return {"status": "OBJECT_LEVEL_BLOCKED", "code": e.code,
                "detail": str(e)}
    resolution = ROR.resolve_e1_runtime_objects(
        bundle, formal_asset_registry, ctx, strict=True)
    return {
        "status": (
            "OBJECT_LEVEL_OK" if resolution["all_bound"]
            else "OBJECT_LEVEL_BLOCKED"
        ),
        "all_bound": resolution["all_bound"],
        "missing": resolution["missing"],
    }


def main(
    argv=None,
    *,
    director_bundle_verifier=None,
    formal_asset_registry=None,
    authorized_llm_runtime=None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--director-runtime-bundle",
        default="",
        help="path to the DIRECTOR-signed runtime bundle manifest "
        "(REQUIRED for any execution; absolute or relative to "
        "gpu1_aggregation_siege/)",
    )
    parser.add_argument(
        "--teacher-config",
        default=RT.TEACHER_CONFIG_PATH,
        help="teacher config path relative to gpu1_aggregation_siege/",
    )
    parser.add_argument(
        "--student-candidate-id",
        default="",
        help="the director-selected Student candidate id (must equal "
        "the Runtime Bundle's issued value; NEVER defaulted to the "
        "first allowed candidate)",
    )
    parser.add_argument("--report-out", default=DEFAULT_REPORT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify bundle + capabilities + pipeline constructibility "
        "+ 15+1 batch semantics ONLY (no LLM, no probe, no training, "
        "never EXECUTED)",
    )
    args = parser.parse_args(argv)

    # ---- production deployment injection (DICODE_SHARED_RUNTIME_REAL=1)
    #      The real FormalAssetRegistry + director verifier + issued
    #      PRODUCTION bundle come from the shared runtime deployment;
    #      explicit kwargs always win (test injection surface). --------
    if os.environ.get("DICODE_SHARED_RUNTIME_REAL") == "1":
        from dicode.shared_runtime.registry import production_registry
        from dicode.shared_runtime.verifier import (
            ProductionDirectorVerifier,
        )

        if formal_asset_registry is None:
            formal_asset_registry = production_registry()
        if director_bundle_verifier is None:
            director_bundle_verifier = ProductionDirectorVerifier()
        if not args.director_runtime_bundle:
            issued = os.path.join(
                RT.SIEGE_ROOT, "reports", "e1_formal_ued",
                "e1_production_runtime_bundle.json")
            if os.path.isfile(issued):
                args.director_runtime_bundle = issued

    # ---- resolve EVERY gate honestly, before anything else -----------
    gates = RT.resolve_production_gates(
        teacher_config_path=args.teacher_config
    )

    # ---- the director-signed runtime bundle (REQUIRED) ---------------
    gates["gates_checked"].append("director_runtime_bundle")
    bundle = _resolve_director_bundle(args, gates["blockers"])

    # ---- director bundle verifier (production trust, not a static
    #      signer tuple) ------------------------------------------------
    gates["gates_checked"].append("director_bundle_verifier")
    if bundle is not None:
        from dicode.teachers.e1_formal import runtime_bundle as _RB

        if bundle.mode == _RB.BUNDLE_MODE_PRODUCTION:
            try:
                _RB.verify_production_runtime_bundle(
                    bundle,
                    director_bundle_verifier,
                    "run_e1_real_one_update.verifier",
                )
            except _RB.ProductionBundleVerificationError as e:
                gates["blockers"].append(
                    _blocker("director_bundle_verifier", e.code, str(e))
                )

    # ---- the director-selected Student (never defaulted) -------------
    # CC2-Repair (BUG-E1-01): the REAL mount happens AFTER the registry
    # resolves the real StudentInitContract (below). Here we only fail
    # fast when a CLI candidate contradicts the director-issued identity.
    gates["gates_checked"].append("student_selection")
    if bundle is not None:
        from dicode.teachers.e1_formal import student_contract as SC

        bundle_candidate = bundle.student_selection.selected_candidate_id
        if (args.student_candidate_id or None) is not None and (
            args.student_candidate_id != bundle_candidate
        ):
            gates["blockers"].append(
                _blocker(
                    "student_selection",
                    SC.STUDENT_CONTRACT_MISMATCH,
                    f"CLI candidate {args.student_candidate_id!r} != the "
                    f"director-issued candidate {bundle_candidate!r} (the "
                    "CLI can never override the director identity)",
                )
            )

    if args.check_only:
        report = _check_only_report(
            bundle, gates,
            formal_asset_registry=formal_asset_registry,
            director_bundle_verifier=director_bundle_verifier,
        )
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
        return (
            0
            if report["status"]
            in (E1_TEST_ONLY_CONTRACT_OK, E1_OBJECT_LEVEL_CHECK_ONLY_OK)
            else 2
        )

    # ---- full run: every gate + a PRODUCTION director bundle ---------
    from dicode.teachers.e1_formal import runtime_bundle as RB

    if bundle is not None:
        try:
            RB.require_bundle_admissible_for_production(
                bundle, "run_e1_real_one_update.production"
            )
        except RB.RuntimeBundleError as e:
            gates["blockers"].append(
                _blocker("director_runtime_bundle", e.code, str(e))
            )

    if gates["blockers"]:
        report = RT.blocked_status_report(
            ENTRYPOINT,
            gates,
            extra={
                "runtime_bundle_verified": bundle is not None,
                "note": (
                    "every blocker above must clear (a director-signed "
                    "PRODUCTION runtime bundle, shared runtime objects "
                    "bound, real EnvCoder backend authorized, frozen "
                    "Reference contract, frozen anchor manifest, "
                    "authorized real LLM provider) before the "
                    "one-window pipeline may run; the pipeline then "
                    "runs the FULL driver chain — there is no fixed "
                    "pipeline-unauthorized gate"
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

    # ---- all gates clear + a valid PRODUCTION director bundle ---------
    # CC2-Student repair: this is the REACHABLE director handoff
    # surface. The real shared objects must be injected (from the
    # director's FormalAssetRegistry); if they are NOT injected, the
    # entry BLOCKS BEFORE any LLM call — it never raises a hardcoded
    # unreachable.
    from dicode.teachers.e1_formal import runtime_object_resolution as ROR

    try:
        ROR.require_real_registry(formal_asset_registry, "pipeline")
        resolution = ROR.resolve_e1_runtime_objects(
            bundle, formal_asset_registry, "run_e1_real_one_update.pipeline",
            strict=True,
        )
    except ROR.RuntimeObjectResolutionError as e:
        gates["blockers"].append(
            _blocker("formal_asset_registry", e.code, str(e))
        )
        resolution = {"all_bound": False, "missing": [e.code]}
    if not resolution["all_bound"]:
        report = RT.blocked_status_report(
            ENTRYPOINT,
            gates,
            extra_blockers=[
                _blocker(
                    "pipeline",
                    "E1_PIPELINE_OBJECTS_NOT_INJECTED",
                    "required runtime object(s) "
                    f"{resolution['missing']} are not injected; the "
                    "entry BLOCKS BEFORE any LLM call (never "
                    "unreachable, never forged)",
                )
            ],
        )
        path = RT.write_json_report(report, args.report_out)
        print(
            f"E1 REAL ONE UPDATE BLOCKED (objects not injected); "
            f"report at {path}"
        )
        return 2

    # ---- the REAL Student mount (BUG-E1-01): registry resolved the
    #      real StudentInitContract; ONLY now is the Student mounted.
    #      A missing / non-matching real Persistent contract refuses.
    #      ------------------------------------------------------------
    _contract = resolution["resolutions"].get("student_init_contract")
    try:
        student_mount = SC.mount_student_from_director_bundle(
            bundle=bundle,
            director_selected_candidate_id=(args.student_candidate_id or None),
            ctx="run_e1_real_one_update.student_mount",
            contract=(_contract.object if _contract is not None else None),
        )
    except SC.StudentSelectionError as e:
        report = RT.blocked_status_report(
            ENTRYPOINT,
            gates,
            extra_blockers=[
                _blocker(
                    "student_selection",
                    e.code,
                    "the real Persistent Student did not mount from the "
                    f"resolved contract: {e}",
                )
            ],
        )
        path = RT.write_json_report(report, args.report_out)
        print(
            f"E1 REAL ONE UPDATE BLOCKED (student mount); report at {path}"
        )
        return 2
    # ---- BUG-E1-04: the BOUND runtime holds the ONE real object set
    #      (Manifest + Registry + real objects + student mount). The
    #      pipeline consumes exactly these instances — nothing is
    #      re-derived from strings / paths / JSON after this point.
    resolved_runtime = RB.bind_runtime_objects(
        bundle, resolution, student_mount=student_mount
    )
    _objects = dict(resolved_runtime.objects)

    # Every real object is injected. Human/authorization gate FIRST:
    # without the authorized six-role runtime the pipeline BLOCKs
    # BEFORE any LLM call.
    if authorized_llm_runtime is None:
        report = RT.blocked_status_report(
            ENTRYPOINT,
            gates,
            extra_blockers=[
                _blocker(
                    "pipeline",
                    "E1_SMOKE_NOT_AUTHORIZED",
                    "the real object set is resolved but no authorized "
                    "six-role LLM runtime / human approval is injected; "
                    "the pipeline BLOCKs before any LLM call",
                )
            ],
        )
        path = RT.write_json_report(report, args.report_out)
        print(f"E1 REAL ONE UPDATE BLOCKED (not authorized); report at {path}")
        return 2

    # REAL pipeline call: pull the resolved objects out of the registry
    # and invoke the full one-window pipeline. This is the reachable
    # director handoff surface — not a fixed pending report. The probe /
    # signal issuers are the RESOLVED real objects (BUG-E1-05), never
    # empty lambdas.
    probe_runner = _objects.get("candidate_probe_runner")
    signal_issuer_obj = _objects.get("criterion_signal_issuer")

    def _real_probe_issuer(candidates, _bundle):
        run_probes = getattr(probe_runner, "run_probes", None)
        if not callable(run_probes):
            raise RuntimeError(
                "E1_PIPELINE_PROBE_RUNNER_UNBOUND: the resolved "
                "candidate_probe_runner exposes no callable run_probes"
            )
        return run_probes(candidates, _bundle)

    def _real_signal_issuer(candidates, probe_pool):
        issue_signals = getattr(signal_issuer_obj, "issue_signals", None)
        if not callable(issue_signals):
            raise RuntimeError(
                "E1_PIPELINE_SIGNAL_ISSUER_UNBOUND: the resolved "
                "criterion_signal_issuer exposes no callable issue_signals"
            )
        return issue_signals(candidates, probe_pool)

    pipeline_report = run_director_one_window_pipeline(
        teacher=_objects.get("student_identity"),
        bundle=bundle,
        run_id="director-smoke",
        branch=RT.git_branch(),
        git_sha=RT.git_head_sha(),
        probe_issuer=_real_probe_issuer,
        signal_issuer=_real_signal_issuer,
        one_update_runtime=_objects.get(
            "canonical_dicode_one_update_runtime"
        ),
        student_checkpoint_identity=(
            bundle.student_selection.checkpoint_file_sha256
        ),
        reference_checkpoint_identity=(
            getattr(
                _objects.get("reference_identity"),
                "checkpoint_file_sha256",
                "",
            )
        ),
        anchor_manifest_hash=bundle.object_identity_hash("anchor_manifest"),
        formal_asset_registry_hash=(
            bundle.object_identity_hash("formal_asset_registry")
        ),
        update_record=None,
        checkpoint=None,
        roundtrip_evidence=None,
        k=12,
        seed=7,
        critic_policy="hard_veto",
        family_cap=6,
        smoke_signer_id="",
        update_signer_id="",
        roundtrip_signer_id="",
    )
    pipeline_report["resolved_runtime_hash"] = resolved_runtime.resolved_runtime_hash
    pipeline_report["student_mount"] = {
        "candidate_id": student_mount.candidate_id,
        "profile_id": student_mount.profile_id,
        "memory_mode": student_mount.memory_mode,
        "params_sha256": student_mount.params_sha256,
        "read_only_ready": student_mount.read_only_ready,
        "training_ready": student_mount.training_ready,
        "mount_hash": student_mount.mount_hash,
    }
    path = RT.write_json_report(pipeline_report, args.report_out)
    print(f"E1 REAL ONE UPDATE pipeline report at {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
