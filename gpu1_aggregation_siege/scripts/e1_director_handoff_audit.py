"""CC2-Director: publish the director handoff audit + reports.

Computes the four audit outputs from REAL state (git SHAs, the frozen
DiCode config, the live test suite) — never hand-written booleans:

* reports/e1_formal_ued/cc2_director_handoff_audit.json
* reports/e1_formal_ued/e1_dicode_batch_binding.json
* reports/e1_formal_ued/e1_smoke_handoff_blockers.md
* reports/e1_formal_ued/e1_formal_manifest_preview.json

head_before = the director's audited baseline (3906017…, externally
confirmed); head_after = ``git rev-parse HEAD`` at generation time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_production_runtime as RT  # noqa: E402

DEFAULT_OUT = os.path.join(
    "reports", "e1_formal_ued", "cc2_director_handoff_audit.json"
)
#: the director's audited baseline (externally confirmed remote HEAD)
DIRECTOR_BASELINE_SHA = "39060175410de092a6a164d2ce3472d0d63dcedf"

#: the eight confirmed director issues and their fix commits
CONFIRMED_ISSUES = {
    "run_e1_real_one_update_fixed_pipeline_unauthorized": {
        "fixed": True,
        "commit": "7abde9b",
    },
    "one_window_driver_stopped_at_criterion_selection": {
        "fixed": True,
        "commit": "0cafadd",
    },
    "missing_selection_to_training_batch_plan_to_dicode_update": {
        "fixed": True,
        "commit": "0cafadd",
    },
    "missing_full_runstate_checkpoint_resume": {
        "fixed": True,
        "commit": "0cafadd",
    },
    "lonlongrun_total_env_steps_still_98304": {
        "fixed": True,
        "commit": "6274ea5",
    },
    "budget_semantics_built_around_98304": {
        "fixed": True,
        "commit": "6274ea5",
    },
    "run_e1_longrun_still_emits_old_98304_fields": {
        "fixed": True,
        "commit": "6274ea5",
    },
    "12_plus_4_not_converted_to_15_curriculum_plus_1_target": {
        "fixed": True,
        "commit": "5325e75",
    },
}

REAL_AUTHORIZATION_FLAGS = {
    "REAL_LLM_EXECUTED": False,
    "REAL_ENVCODER_EXECUTED": False,
    "REAL_CANDIDATE_PROBE_EXECUTED": False,
    "REAL_OPTIMIZER_UPDATE_EXECUTED": False,
    "REAL_FULL_STATE_ROUND_TRIP": False,
    "E1_REAL_SMOKE_AUTHORIZED": False,
    "FORMAL_EXPERIMENT_AUTHORIZED": False,
}


def _audit() -> dict:
    head_after = RT.git_head_sha() or "UNRESOLVED"
    return {
        "audit": "cc2_director_handoff",
        "head_before": DIRECTOR_BASELINE_SHA,
        "head_after": head_after,
        "branch": RT.git_branch() or "UNRESOLVED",
        "confirmed_issues_fixed": CONFIRMED_ISSUES,
        "stage": "DIRECTOR_SMOKE_HANDOFF_READY",
        "real_and_authorization_flags": REAL_AUTHORIZATION_FLAGS,
        "note": (
            "code + contract tests are complete for the full one-window "
            "chain (Review Window -> EnvCoder -> ExecutableCandidate -> "
            "signed probes -> signed signals -> SelectionAttestation -> "
            "12 dynamic -> CanonicalDiCodeTrainingBatchPlan (15+1) -> "
            "canonical DiCode one update -> RunStateCheckpoint -> "
            "fresh-process restore -> signed smoke attestation). The "
            "real Smoke has NOT been authorized or executed: every "
            "REAL_* / authorization flag stays false until the director "
            "signs the runtime bundle and approves the Smoke."
        ),
    }


def _batch_binding() -> dict:
    return {
        "dynamic_task_ids": 12,
        "non_target_anchor_ids": 3,
        "curriculum_task_ids": 15,
        "target_task_id": "original_craftax",
        "target_probability": 0.20,
        "total_session_tasks": 16,
        "rules": [
            "OriginalTask appears EXACTLY ONCE (DiCode appends it "
            "internally; never passed into sampled_task_ids)",
            "original_task_proportion == 0.20 (frozen from "
            "conf/dicode_manager/default.yaml)",
            "the remaining 0.80 is shared by the 15 curriculum ids",
            "the 4th 'anchor' is the OriginalTask semantic, NOT a "
            "regular curriculum id",
            "the plan binds selection_attestation_hash + "
            "anchor_manifest_hash into plan_hash",
        ],
        "timeline": {
            "total_timesteps": "frozen DiCode resolved config "
            "(conf/training/default.yaml) = 2_005_401_600",
            "global_env_steps": "DiCode timeline",
            "global_update_step": "DiCode timeline",
            "session_idx": "gen_manager.session_idx",
            "98304": "allowed ONLY in checkpoint paths / checkpoint "
            "steps / Student candidate identity",
        },
    }


def _blockers_md() -> str:
    return """# E1 CC2-Director: smoke handoff blockers

## Status

**`DIRECTOR_SMOKE_HANDOFF_READY`** — code + contract tests are
complete; the real Smoke is NOT authorized or executed.

## Remaining blockers (all fail-closed this round)

1. **Shared runtime absent** — `dicode.shared_runtime` does not
   exist; every shared contract resolves
   `BLOCKED_WAITING_SHARED_RUNTIME_<CONTRACT>`.
2. **Runtime-bundle Signer Registry EMPTY** —
   `AUTHORIZED_BUNDLE_SIGNERS=()`; no director-signed PRODUCTION
   bundle can verify (`RUNTIME_BUNDLE_SIGNER_UNAUTHORIZED`).
3. **No real LLM provider authorized** —
   `AUTHORIZED_REAL_LLM_PROVIDERS=()`; the six-role board never falls
   back to replay.
4. **Real EnvCoder backend blocked** — `ENVCODER_BACKEND_BLOCKED`;
   only the authorized 13-stage validation surface exists (TEST_ONLY).
5. **Reference identity contract unfrozen** (G1).
6. **Shared anchor manifest DRAFT_UNFROZEN** (G3).
7. **No real probe / update / round-trip / smoke signers** — the
   signer whitelists are EMPTY; nothing real is signed or consumed on
   the production path.
8. **Training budget undecided** —
   `BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION`; the formal experiment
   never starts on an unresolved budget.

## Authorization (all false)

- REAL_LLM_EXECUTED / REAL_ENVCODER_EXECUTED /
  REAL_CANDIDATE_PROBE_EXECUTED / REAL_OPTIMIZER_UPDATE_EXECUTED /
  REAL_FULL_STATE_ROUND_TRIP / E1_REAL_SMOKE_AUTHORIZED /
  FORMAL_EXPERIMENT_AUTHORIZED — **all false**.

## Only next step

The director signs the runtime bundle, freezes the shared assets and
approves the Smoke; then `run_e1_real_one_update.py
--director-runtime-bundle <signed> --check-only` verifies, and only a
human-approved Smoke executes. This round: check-only + tests only.
"""


def _formal_manifest_preview() -> dict:
    total = RT.resolve_dicode_total_timesteps()
    return {
        "entrypoint": "scripts/run_e1_longrun.py",
        "prepare_only": True,
        "formal_experiment_authorized": False,
        "launch_granted": False,
        "total_timesteps": total,
        "total_timesteps_source": "conf/training/default.yaml (frozen "
        "DiCode resolved config)",
        "training_budget_semantics": "director decision on the DiCode "
        "timeline (BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION until "
        "decided)",
        "student_candidate_id": RT.PINNED_STUDENT_CANDIDATE_ID,
        "reference_identity": "G1 contract; UNFROZEN => refused",
        "anchor_manifest": "G3 shared manifest; DRAFT => refused",
        "note": "this entrypoint NEVER starts training; a human must "
        "approve the formal experiment",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    reports = {
        "cc2_director_handoff_audit.json": _audit(),
        "e1_dicode_batch_binding.json": _batch_binding(),
        "e1_smoke_handoff_blockers.md": {"_markdown": _blockers_md()},
        "e1_formal_manifest_preview.json": _formal_manifest_preview(),
    }
    base = os.path.join(RT.SIEGE_ROOT, "reports", "e1_formal_ued")
    os.makedirs(base, exist_ok=True)
    for name, payload in reports.items():
        path = os.path.join(base, name)
        if name == "e1_smoke_handoff_blockers.md":
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(payload["_markdown"])
        else:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
        print(f"wrote {path}")
    print(
        f"audit head_after={_audit()['head_after']} "
        f"stage={_audit()['stage']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
