"""Pinned role manifest for the E1 LLM roles (replay provider this round).

Accounting model (supervisor gate G5) — the manifest intentionally does
NOT describe any fixed per-window call count:

    N1 = 6*G1 + T1 + K1 + F1

* G1 — number of TRIGGERED review windows; each triggered window runs
  exactly the six board roles once (invocation_unit="window");
* T1 — TaskGenerator calls; E1 has NO TaskGenerator role, so T1 is the
  explicit constant 0 (this round reconcile fails closed if nonzero);
* K1 — EnvCoder calls counted per ACTUAL UNIQUE artifact produced
  (invocation_unit="artifact"); the EnvCoder is an independent
  artifact producer OUTSIDE the board, not a window member;
* F1 — repair calls, counted separately; round-3 wires the bounded
  EnvCoder repair loop into this counter (F1 = the real per-template
  repair-call count, bounded by ``teacher.envcoder.max_repairs`` <= 2;
  it never mixes with K1).

Every entry is pinned to the replay provider this round; a real LLM
provider would change REAL_ENVCODER_USED, which must stay false.
"""
from __future__ import annotations

from typing import Dict

from .schemas import E1SchemaError

E1_REPLAY_PROVIDER = "replay"
E1_REPLAY_MODEL_ID = "e1-replay-mock-v1"

#: Fixed board role order; every triggered window runs ALL six.
BOARD_ROLE_ORDER = (
    "student_modeler",
    "behavior_auditor",
    "causal_failure_analyst",
    "intervention_tutor",
    "explorer",
    "critic",
)

#: Independent artifact producer, OUTSIDE the board window.
ENVCODER_ROLE = "envcoder"

#: v2 (round-3 P0-1): the per-role prompt now binds the board context
#: (window identity, student candidate id, evidence hash) and every
#: successfully-parsed upstream role output, so the envelope hash — and
#: therefore every replay key — changes with the sequential chain.
BOARD_PROMPT_VERSION = "e1-board-prompt-v2"
ENVCODER_PROMPT_VERSION = "e1-envcoder-prompt-v1"
ROLE_OUTPUT_SCHEMA_VERSION = "e1-role-output-v1"
ENVCODER_OUTPUT_SCHEMA_VERSION = "e1-envcoder-output-v1"

INVOCATION_UNIT_WINDOW = "window"
INVOCATION_UNIT_ARTIFACT = "artifact"


def _entry(role: str, prompt_version: str, schema_version: str, unit: str) -> Dict[str, str]:
    return {
        "role": role,
        "provider": E1_REPLAY_PROVIDER,
        "exact_model_id": E1_REPLAY_MODEL_ID,
        "prompt_version": prompt_version,
        "schema_version": schema_version,
        "invocation_unit": unit,
    }


def build_role_manifest() -> Dict[str, Dict[str, str]]:
    """Fresh copy of the pinned manifest (6 board roles + 1 artifact producer)."""
    manifest = {
        role: _entry(role, BOARD_PROMPT_VERSION, ROLE_OUTPUT_SCHEMA_VERSION, INVOCATION_UNIT_WINDOW)
        for role in BOARD_ROLE_ORDER
    }
    manifest[ENVCODER_ROLE] = _entry(
        ENVCODER_ROLE,
        ENVCODER_PROMPT_VERSION,
        ENVCODER_OUTPUT_SCHEMA_VERSION,
        INVOCATION_UNIT_ARTIFACT,
    )
    return manifest


def assert_manifest_well_formed(manifest: Dict[str, Dict[str, str]]) -> None:
    """Fail-closed structural audit of the role manifest."""
    if not isinstance(manifest, dict):
        raise E1SchemaError(
            "ROLE_MANIFEST_BAD_TYPE", "role manifest must be a dict"
        )
    expected_roles = set(BOARD_ROLE_ORDER) | {ENVCODER_ROLE}
    if set(manifest) != expected_roles:
        raise E1SchemaError(
            "ROLE_MANIFEST_ROLE_SET_MISMATCH",
            f"manifest roles {sorted(manifest)} != expected "
            f"{sorted(expected_roles)}",
        )
    window_roles = []
    artifact_roles = []
    for role, entry in manifest.items():
        for key in ("role", "provider", "exact_model_id", "prompt_version", "schema_version", "invocation_unit"):
            if not entry.get(key):
                raise E1SchemaError(
                    "ROLE_MANIFEST_MISSING_FIELD",
                    f"manifest entry {role!r} missing {key!r}",
                )
        if entry["role"] != role:
            raise E1SchemaError(
                "ROLE_MANIFEST_ROLE_SET_MISMATCH",
                f"manifest entry key {role!r} != entry role {entry['role']!r}",
            )
        if entry["invocation_unit"] == INVOCATION_UNIT_WINDOW:
            window_roles.append(role)
        elif entry["invocation_unit"] == INVOCATION_UNIT_ARTIFACT:
            artifact_roles.append(role)
        else:
            raise E1SchemaError(
                "ROLE_MANIFEST_BAD_UNIT",
                f"manifest entry {role!r} has unknown invocation_unit "
                f"{entry['invocation_unit']!r}",
            )
    if tuple(sorted(window_roles)) != tuple(sorted(BOARD_ROLE_ORDER)):
        raise E1SchemaError(
            "ROLE_MANIFEST_ROLE_SET_MISMATCH",
            f"window-unit roles {sorted(window_roles)} != board roles "
            f"{sorted(BOARD_ROLE_ORDER)}",
        )
    if artifact_roles != [ENVCODER_ROLE]:
        raise E1SchemaError(
            "ROLE_MANIFEST_ROLE_SET_MISMATCH",
            f"artifact-unit roles {artifact_roles} != ['{ENVCODER_ROLE}']",
        )
