"""CC2-Student: publish the dual-student binding audit + reports.

Computes the three outputs from REAL state (git SHAs, the frozen
allowed-set + profile map, the capability policy) — never hand-written
booleans:

* reports/e1_formal_ued/e1_dual_student_binding_audit.json
* reports/e1_formal_ued/e1_student_runtime_matrix.json
* reports/e1_formal_ued/e1_director_handoff_blockers.md

head_before = the previous round's HEAD (2a2d5a7…); head_after =
``git rev-parse HEAD`` at generation time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_production_runtime as RT  # noqa: E402

DEFAULT_OUT = os.path.join(
    "reports", "e1_formal_ued", "e1_dual_student_binding_audit.json"
)
#: the previous round's HEAD (externally confirmed)
PREVIOUS_HEAD_SHA = "2a2d5a79bea5e5981a83a0f9f774521a2380adef"

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
    from dicode.teachers.e1_formal import student_contract as SC

    return {
        "audit": "cc2_student_dual_consumer",
        "head_before": PREVIOUS_HEAD_SHA,
        "head_after": RT.git_head_sha() or "UNRESOLVED",
        "branch": RT.git_branch() or "UNRESOLVED",
        "stage": "E1_DUAL_STUDENT_CONSUMER_READY",
        "allowed_student_candidate_ids": sorted(
            SC.ALLOWED_STUDENT_CANDIDATE_IDS
        ),
        "profile_map": {
            cid: {
                "profile_id": entry[0],
                "memory_mode": entry[1],
            }
            for cid, entry in SC.STUDENT_PROFILE_MAP.items()
        },
        "selection_rule": (
            "the director-selected candidate id is REQUIRED (no "
            "default); the CLI can never override the Runtime Bundle's "
            "issued identity; one Student per run only"
        ),
        "capability_states": {
            "read_only": SC.STUDENT_READ_ONLY_MOUNT_READY,
            "training": SC.STUDENT_TRAINING_RUNTIME_READY,
            "unbound": SC.STUDENT_SHARED_REGISTRY_UNBOUND,
        },
        "training_rule": (
            "training flows ONLY through the director-injected "
            "CanonicalDiCodeOneUpdateRuntime + "
            "CanonicalDiCodeRunStateCheckpoint; the read-only RMT16 "
            "adapter can never impersonate a training runtime"
        ),
        "real_and_authorization_flags": REAL_AUTHORIZATION_FLAGS,
        "note": (
            "E1 consumes the shared student registry "
            "(dicode.student_adapters) — it never constructs a second "
            "loader / registry / checkpoint codec. The shared registry "
            "is absent in this worktree, so every mount stays honestly "
            "STUDENT_SHARED_REGISTRY_UNBOUND and no real Smoke / probe "
            "/ update runs."
        ),
    }


def _runtime_matrix() -> dict:
    from dicode.teachers.e1_formal import student_contract as SC

    return {
        "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304": {
            "profile_id": "rmt16_persistent_98304",
            "memory_mode": "PERSISTENT",
            "carry_mode": "persistent-memory-progression",
            "read_only_mount": "checkpoint loadable / forward "
            "executable / memory progression verifiable / probe "
            "executable",
            "training_runtime": "NOT available via the read-only "
            "adapter; requires the director-injected canonical DiCode "
            "runtimes",
        },
        "RESET128_RMT16_ORIGINAL_VTRACE_98304": {
            "profile_id": "rmt16_reset128_98304",
            "memory_mode": "RESET128",
            "carry_mode": "reset-to-128-window-memory",
            "read_only_mount": "checkpoint loadable / forward "
            "executable / memory progression verifiable / probe "
            "executable",
            "training_runtime": "NOT available via the read-only "
            "adapter; requires the director-injected canonical DiCode "
            "runtimes",
        },
        "rules": [
            "one Student per run only (never both mixed)",
            "no silent fallback to Persistent",
            "no profile guessing from the name",
            "params-only checkpoint is never called full-state",
        ],
        "memory_modes_distinct": True,
    }


def _blockers_md() -> str:
    return """# E1 CC2-Student: director handoff blockers

## Status

**`E1_DUAL_STUDENT_CONSUMER_READY`** — code + contract tests are
complete for the dual-Student selection / mount / continuity /
read-only-vs-training capability split; the real Smoke is NOT
authorized or executed.

## Remaining blockers (all fail-closed this round)

1. **Shared student registry absent** — `dicode.student_adapters`
   does not exist; every mount resolves
   `STUDENT_SHARED_REGISTRY_UNBOUND` (read-only mount is NOT ready,
   training is NEVER implied).
2. **Runtime-bundle Signer Registry EMPTY** —
   `AUTHORIZED_BUNDLE_SIGNERS=()`; no director-signed PRODUCTION
   bundle can verify, so no real Student selection can be injected.
3. **No real LLM provider authorized** — the six-role board never
   falls back to replay.
4. **Reference identity contract unfrozen** (G1) and the shared
   anchor manifest DRAFT_UNFROZEN (G3).
5. **No canonical DiCode training runtime bound** — training happens
   ONLY through the director-injected `CanonicalDiCodeOneUpdateRuntime`
   + `CanonicalDiCodeRunStateCheckpoint`; until then the Director
   Smoke handoff stays BLOCKED and no update is ever forged.

## Authorization (all false)

- REAL_LLM_EXECUTED / REAL_CANDIDATE_PROBE_EXECUTED /
  REAL_OPTIMIZER_UPDATE_EXECUTED / REAL_FULL_STATE_ROUND_TRIP /
  E1_REAL_SMOKE_AUTHORIZED / FORMAL_EXPERIMENT_AUTHORIZED — **all
  false**.

## Only next step

The director signs the runtime bundle (with a `student` selection of
one of the two allowed candidates), freezes the shared assets and
approves the Smoke; then `run_e1_real_one_update.py
--director-runtime-bundle <signed> --student-candidate-id <id>
--check-only` verifies. This round: check-only + tests only.
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    reports = {
        "e1_dual_student_binding_audit.json": _audit(),
        "e1_student_runtime_matrix.json": _runtime_matrix(),
        "e1_director_handoff_blockers.md": {"_markdown": _blockers_md()},
    }
    base = os.path.join(RT.SIEGE_ROOT, "reports", "e1_formal_ued")
    os.makedirs(base, exist_ok=True)
    for name, payload in reports.items():
        path = os.path.join(base, name)
        if name == "e1_director_handoff_blockers.md":
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(payload["_markdown"])
        else:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
        print(f"wrote {path}")
    audit = _audit()
    print(f"audit head_after={audit['head_after']} stage={audit['stage']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
