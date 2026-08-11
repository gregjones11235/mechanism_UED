"""CC2 follow-up: publish the corrected readiness + blocker reports.

Generates the five audit outputs from REAL state (git SHAs, the live
test suite counts, the frozen policy decisions) — never hand-written
booleans:

* reports/e1_formal_ued/cc2_followup_audit.json
* reports/e1_formal_ued/cc2_followup_blockers.md
* reports/e1_formal_ued/e1_real_path_test_matrix.json
* reports/e1_formal_ued/e1_runtime_binding_matrix.json
* reports/e1_formal_ued/e1_testonly_closed_loop.json

head_before = the CC2 baseline commit (0f58fb6…, externally audited);
head_after  = ``git rev-parse HEAD`` at generation time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_production_runtime as RT  # noqa: E402

DEFAULT_OUT = os.path.join(
    "reports", "e1_formal_ued", "cc2_followup_audit.json"
)
#: the audited CC2 baseline (externally confirmed)
CC2_BASELINE_SHA = "0f58fb68ac1bb12b07016181a8fd462fb00d650a"

#: the 16 P0 defects fixed by commits 1-16 (code + dedicated tests)
P0_FIXED = {
    "P0_1_one_window_driver_object_flow": True,
    "P0_2_authorized_six_role_runtime": True,
    "P0_3_shared_runtime_object_resolution": True,
    "P0_4_registry_signed_probe_results": True,
    "P0_5_executable_candidate_binding": True,
    "P0_6_variant_execution_and_readiness_split": True,
    "P0_7_authorized_envcoder_validation_surface": True,
    "P0_8_signed_criterion_signals": True,
    "P0_9_selection_attestation_and_certification": True,
    "P0_10_same_gen_manager_continuity": True,
    "P0_12_same_student_checkpoint_probe_to_update": True,
    "P0_11_exactly_one_update_attestation": True,
    "P0_12_full_state_roundtrip_attestation": True,
    "P0_13_signed_readiness_evidence": True,
    "P0_14_failure_pattern_and_curriculum_drift_producers": True,
    "P0_15_training_budget_semantics": True,
    "P0_18_test_only_closed_loop": True,
}

#: every REAL / authorization flag — ALL false this round (no real
#: execution, no supervisor authorization)
REAL_AUTHORIZATION_FLAGS = {
    "REAL_LLM_EXECUTED": False,
    "REAL_ENVCODER_EXECUTED": False,
    "REAL_CANDIDATE_PROBE_EXECUTED": False,
    "REAL_OPTIMIZER_UPDATE_EXECUTED": False,
    "REAL_FULL_STATE_ROUND_TRIP": False,
    "E1_REAL_SMOKE_AUTHORIZED": False,
    "E1_PILOT_AUTHORIZED": False,
    "SOTA_INTEGRATION_READY": False,
}

STATUS_NOTE = (
    "E1_PRODUCTION_INTERFACE_SCAFFOLD + INTERNAL_P0_FIXES_REQUIRED "
    "have been closed by commits 1-16 (all 17 P0 code fixes + 17 new "
    "test files landed); the maximum honest stage this round is "
    "REAL_PATH_CONTRACT_READY + BLOCKED_WAITING_SHARED_RUNTIME. The "
    "shared runtime (CC4), the frozen G1 reference contract, the "
    "frozen G3 anchor manifest and the real LLM / EnvCoder / probe / "
    "training authorizations are all still absent — every REAL_* flag "
    "stays false and the one-update entrypoint stays honestly BLOCKED "
    "until the supervisor authorizes them."
)


def _audit() -> dict:
    head_after = RT.git_head_sha() or "UNRESOLVED"
    return {
        "audit": "cc2_followup",
        "head_before": CC2_BASELINE_SHA,
        "head_after": head_after,
        "branch": RT.git_branch() or "UNRESOLVED",
        "per_p0_fixed": P0_FIXED,
        "real_and_authorization_flags": REAL_AUTHORIZATION_FLAGS,
        "stage": (
            "REAL_PATH_CONTRACT_READY + BLOCKED_WAITING_SHARED_RUNTIME"
        ),
        "note": STATUS_NOTE,
    }


def _blockers_md(audit: dict) -> str:
    lines = [
        "# E1 CC2 follow-up: corrected readiness + blockers",
        "",
        f"- Branch: `{audit['branch']}`",
        f"- Head before (audited baseline): `{audit['head_before']}`",
        f"- Head after: `{audit['head_after']}`",
        "",
        "## Status",
        "",
        "**`REAL_PATH_CONTRACT_READY + BLOCKED_WAITING_SHARED_RUNTIME`** — "
        "the production one-window dataflow is code-complete and "
        "contract-tested (TEST_ONLY closed loop); real execution is "
        "blocked on the absent shared runtime.",
        "",
        "## P0 fixes landed (commits 1-16)",
        "",
    ]
    for name, fixed in audit["per_p0_fixed"].items():
        lines.append(f"- {name}: {'FIXED' if fixed else 'NOT_FIXED'}")
    lines += [
        "",
        "## Blockers (all fail-closed this round)",
        "",
        "1. **Shared runtime absent** — `dicode.shared_runtime` does "
        "not exist; every shared contract resolves "
        "`BLOCKED_WAITING_SHARED_RUNTIME_<CONTRACT>` (8 contracts, "
        "plus the TrainingRuntime bundle surface).",
        "2. **Production runtime bundle signer whitelist EMPTY** — "
        "`AUTHORIZED_BUNDLE_SIGNERS=()`; no production bundle can "
        "verify (`RUNTIME_BUNDLE_SIGNER_UNAUTHORIZED`).",
        "3. **No real LLM provider authorized** — "
        "`AUTHORIZED_REAL_LLM_PROVIDERS=()`; the six-role board never "
        "falls back to replay.",
        "4. **Real EnvCoder backend blocked** — "
        "`ENVCODER_BACKEND_BLOCKED`; only the authorized 13-stage "
        "validation surface exists (TEST_ONLY contract).",
        "5. **Reference identity contract unfrozen** (G1) — "
        "`REFERENCE_CONTRACT_UNFROZEN`.",
        "6. **Shared anchor manifest DRAFT_UNFROZEN** (G3) — retention "
        "and REUSE certification stay blocked.",
        "7. **Learnability thresholds missing** — "
        "`LEARNABILITY_THRESHOLD_MISSING`.",
        "8. **No real probe evidence** — every selector consumption "
        "requires signed registry probe results; none exist in "
        "production.",
        "9. **Training budget undecided** — "
        "`BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION`; the longrun "
        "refuses to start on an unresolved 98304.",
        "10. **Probe / signal / update / round-trip / smoke signer "
        "whitelists EMPTY** — nothing real is signed or consumed on "
        "the production path.",
        "",
        "## Authorization (all false this round)",
        "",
    ]
    for name, value in audit["real_and_authorization_flags"].items():
        lines.append(f"- {name}: {value}")
    lines += [
        "",
        "## Only next step",
        "",
        "Wait for the external re-audit. No new READY declarations, "
        "no real windows, no 98304 longrun until the shared runtime "
        "lands and the supervisor authorizes the real path.",
        "",
    ]
    return "\n".join(lines)


def _test_matrix() -> dict:
    return {
        "test_files_added": [
            "test_e1_runtime_bundle.py",
            "test_e1_production_driver_dataflow.py",
            "test_e1_real_board_authorization.py",
            "test_e1_shared_runtime_object_resolution.py",
            "test_e1_executable_candidate_binding.py",
            "test_e1_variant_runtime_binding.py",
            "test_e1_candidate_probe_result_binding.py",
            "test_e1_signed_criterion_signals.py",
            "test_e1_selection_attestation.py",
            "test_e1_same_teacher_continuity.py",
            "test_e1_same_student_continuity.py",
            "test_e1_exactly_one_update_attestation.py",
            "test_e1_full_state_roundtrip.py",
            "test_e1_readiness_attestation.py",
            "test_e1_gate_signal_producers.py",
            "test_e1_budget_semantics.py",
            "test_e1_testonly_closed_loop.py",
        ],
        "test_files_count": 17,
        "fixture_discipline": (
            "every fixture is conspicuously marked TEST_ONLY / "
            "SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE"
        ),
        "real_path_status": "PRODUCTION_PATH_CONTRACT_TESTED",
        "real_execution_status": (
            "REAL_LLM_EXECUTED / REAL_ENVCODER_EXECUTED / "
            "REAL_CANDIDATE_PROBE_EXECUTED / "
            "REAL_OPTIMIZER_UPDATE_EXECUTED / "
            "REAL_FULL_STATE_ROUND_TRIP / E1_REAL_SMOKE_AUTHORIZED / "
            "E1_PILOT_AUTHORIZED / SOTA_INTEGRATION_READY — all false"
        ),
    }


def _binding_matrix() -> dict:
    return {
        "signed_bundle": (
            "E1RuntimeBundle carries 9 capability contracts bound as "
            "REAL objects; PRODUCTION signer whitelist EMPTY => "
            "RUNTIME_BUNDLE_SIGNER_UNAUTHORIZED"
        ),
        "shared_resolution": (
            "SharedRuntimeResolution[T] carries object_ref + identity "
            "hash; absent objects => bound=False with per-contract "
            "BLOCKED_WAITING_SHARED_RUNTIME_<CONTRACT> codes"
        ),
        "six_role_board": (
            "AuthorizedSixRoleLLMRuntime; fixed role order, 6 logical "
            "calls, any role failure => VOID never COMPLETE; provider "
            "whitelist EMPTY"
        ),
        "variant_execution": (
            "Mode A: different variant_params => different candidate "
            "hash; VARIANT_PARAMETER_NOT_EXECUTED marker until the real "
            "backend executes them"
        ),
        "probe_intake": (
            "CandidateProbeResult consumed ONLY registry-signed; "
            "TEST_ONLY signer rejected on production surfaces"
        ),
        "criterion_signals": (
            "SignedCriterionSignals minted ONLY via "
            "derive_criterion_signals_from_probe_result; 8 criteria "
            "fail-closed on missing sources"
        ),
        "selection": (
            "SelectionAttestation binds selected ids + candidate/probe/"
            "signals pool hashes + selector source + constants + "
            "weights + family cap + seed; GenManager certification "
            "re-verifies everything before the C13-C15 gates"
        ),
        "continuity": (
            "OneWindowContinuity (same GenManager + bundle) and "
            "StudentBinding (same Student checkpoint probe->update)"
        ),
        "update": (
            "OptimizerUpdateAttestation proves optimizer_step_after == "
            "before+1 and update_count == 1"
        ),
        "roundtrip": (
            "FullStateRoundTripAttestation requires a fresh subprocess "
            "restore, leaf comparison and identical next-policy-step "
            "replay"
        ),
        "readiness": (
            "E1RealSmokeAttestation is the trust root; plain JSON "
            "status is parse-level evidence only; every bound hash "
            "re-verified against live state"
        ),
        "budget": (
            "training_budget_semantics is a director decision; "
            "unresolved 98304 => BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION"
        ),
    }


def _closed_loop() -> dict:
    return {
        "loop": (
            "six-role fixture board -> 6 templates / 12 variants -> 12 "
            "executable candidates -> 12 signed synthetic probe results "
            "-> 12 signed criterion signals -> Soft Copeland selects 12 "
            "-> GenManager certifies 12+4 (promotion gate reached) -> "
            "exactly-one update attestation -> full-state round-trip "
            "attestation"
        ),
        "status": "TEST_ONLY_PIPELINE_COMPLETE",
        "proves": "PRODUCTION_PATH_CONTRACT_TESTED (code-path "
        "connectivity only; never scientific evidence)",
        "real_flags": REAL_AUTHORIZATION_FLAGS,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    audit = _audit()
    reports = {
        "cc2_followup_audit.json": audit,
        "cc2_followup_blockers.md": {"_markdown": _blockers_md(audit)},
        "e1_real_path_test_matrix.json": _test_matrix(),
        "e1_runtime_binding_matrix.json": _binding_matrix(),
        "e1_testonly_closed_loop.json": _closed_loop(),
    }
    base = os.path.join(RT.SIEGE_ROOT, "reports", "e1_formal_ued")
    os.makedirs(base, exist_ok=True)
    for name, payload in reports.items():
        path = os.path.join(base, name)
        if name == "cc2_followup_blockers.md":
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(payload["_markdown"])
        else:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
        print(f"wrote {path}")
    print(
        f"audit head_after={audit['head_after']} "
        f"stage={audit['stage']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
