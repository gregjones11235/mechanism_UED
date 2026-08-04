"""E1 round-3 P0-6: shared production runtime resolution.

Both production entrypoints (``run_e1_real_one_update.py`` and
``run_e1_longrun.py``) resolve EVERY asset through this module before
any real execution:

* the shared runtime seam — the eight CC4 shared contracts, resolved
  lazily against ``dicode.shared_runtime`` (the seam only RESOLVES;
  it never constructs, mints or disguises a shared identity);
* the real EnvCoder backend authorization (the ``RealBackendAdapter``
  fails closed with ``ENVCODER_BACKEND_BLOCKED`` while the craftax
  runtime is unauthorized — never a silent downgrade);
* the G1 Reference identity contract (must be FROZEN);
* the G3 shared anchor manifest (must be FROZEN, not DRAFT);
* an honest JSON status report (every blocker with stage + code +
  detail; nothing omitted, nothing fabricated).

Production hygiene (round-3 directives): this module imports NO
tests and NO fixtures, never enables mock/replay defaults, performs
NO paid LLM calls and NO training. Any missing asset produces an
explicit BLOCKED code and the entrypoints exit non-zero.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

#: repo layout — scripts/ lives inside gpu1_aggregation_siege/
SIEGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(SIEGE_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
# siege root: required ONLY for the sanctioned d052.achievements
# REGISTRY import in task_specs (pure stdlib; same convention as
# tests/e1_formal/conftest.py) — nothing else is imported from d052
if SIEGE_ROOT not in sys.path:
    sys.path.insert(0, SIEGE_ROOT)

from dicode.teachers.e1_formal import anchor_manifest as AM  # noqa: E402
from dicode.teachers.e1_formal import envcoder_backends as EB  # noqa: E402
from dicode.teachers.e1_formal import shared_runtime_seam as SRS  # noqa: E402
from dicode.teachers.e1_formal.reference_contract import (  # noqa: E402
    ReferenceContractError,
    consume_reference_identity_contract,
)
from dicode.teachers.e1_formal.student_contract import (  # noqa: E402
    PINNED_STUDENT_CANDIDATE_ID,
)

#: the pinned longrun horizon (supervisor-frozen)
LONGRUN_TOTAL_ENV_STEPS = 98304

#: production gate stages, in the order the entrypoints audit them
GATE_SHARED_RUNTIME = "shared_runtime_resolution"
GATE_REAL_ENVCODER_BACKEND = "real_envcoder_backend"
GATE_REFERENCE_CONTRACT = "reference_contract_frozen"
GATE_ANCHOR_MANIFEST = "anchor_manifest_frozen"
GATE_REAL_LLM_PROVIDER = "real_llm_provider_authorized"

#: round-3: no real LLM provider is authorized this worktree; the
#: whitelist is supervisor-owned and currently EMPTY — the six-role
#: board never falls back to replay silently
AUTHORIZED_REAL_LLM_PROVIDERS = ()
E1_REAL_LLM_NOT_AUTHORIZED = "E1_REAL_LLM_NOT_AUTHORIZED"

#: default asset paths (relative to gpu1_aggregation_siege/)
TEACHER_CONFIG_PATH = os.path.join("conf", "teacher", "e1_formal.yaml")
FROZEN_MANIFEST_PATH = os.path.join("configs", "e1_formal_ued.yaml")
ANCHOR_MANIFEST_PATH = os.path.join(
    "configs", "e1_formal_ued_anchor_manifest.DRAFT.json"
)


def load_yaml(path: str) -> dict:
    """Load one YAML file (stdlib-safe: PyYAML only, no hydra magic)."""
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def file_sha256(path: str) -> str:
    """sha256 over the file bytes (config/checkpoint identity)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head_sha() -> str:
    """The LIVE ``git rev-parse HEAD`` of this worktree ("" on
    failure — callers treat an unresolvable SHA as unfrozen)."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=SIEGE_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def git_branch() -> str:
    """The current branch name ("" on failure)."""
    try:
        completed = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=SIEGE_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def resolve_shared_runtime() -> dict:
    """The seam's honest per-contract state (never mints/disguises)."""
    return {
        contract: {
            "bound": resolution.bound,
            "code": resolution.code,
            "detail": resolution.detail,
        }
        for contract, resolution in SRS.resolve_all_shared_runtime().items()
    }


def resolve_production_gates(
    teacher_config_path: str = TEACHER_CONFIG_PATH,
    anchor_manifest_path: str = ANCHOR_MANIFEST_PATH,
) -> dict:
    """Resolve EVERY production gate honestly, in audit order.

    Returns ``{"gates_checked": [...], "blockers": [{stage, code,
    detail}], "shared_runtime": {...}, "reference_contract_frozen":
    bool, "anchor_manifest_frozen": bool}``. Blockers are NEVER
    silently merged: every failing gate contributes its own entry.
    """
    gates_checked = []
    blockers = []

    # ---- gate 1: the shared runtime seam (eight contracts) ----------
    gates_checked.append(GATE_SHARED_RUNTIME)
    shared = resolve_shared_runtime()
    for contract in sorted(shared):
        state = shared[contract]
        if not state["bound"]:
            blockers.append(
                {
                    "stage": GATE_SHARED_RUNTIME,
                    "code": state["code"],
                    "detail": (
                        f"shared contract {contract!r} is unbound: "
                        f"{state['detail']}"
                    ),
                }
            )

    # ---- gate 2: the REAL EnvCoder backend (never degrades) ---------
    gates_checked.append(GATE_REAL_ENVCODER_BACKEND)
    try:
        EB.make_backend(EB.BACKEND_REAL)
    except EB.EnvCoderBackendError as e:
        blockers.append(
            {
                "stage": GATE_REAL_ENVCODER_BACKEND,
                "code": e.code,
                "detail": str(e),
            }
        )

    # ---- gate 3: the G1 Reference identity contract ------------------
    gates_checked.append(GATE_REFERENCE_CONTRACT)
    teacher_config = load_yaml(
        os.path.join(SIEGE_ROOT, teacher_config_path)
    )
    rc_block = teacher_config["teacher"]["reference_contract"]
    try:
        contract = consume_reference_identity_contract(
            rc_block, "e1_production.reference_contract"
        )
    except ReferenceContractError as e:
        contract = None
        blockers.append(
            {
                "stage": GATE_REFERENCE_CONTRACT,
                "code": e.code,
                "detail": str(e),
            }
        )

    # ---- gate 4: the G3 shared anchor manifest (FROZEN, not DRAFT) ---
    gates_checked.append(GATE_ANCHOR_MANIFEST)
    anchor_path = os.path.join(SIEGE_ROOT, anchor_manifest_path)
    with open(anchor_path, "r", encoding="utf-8") as handle:
        manifest_mapping = json.load(handle)
    try:
        manifest = AM.consume_anchor_manifest(
            manifest_mapping, "e1_production.anchor_manifest"
        )
        if not manifest.is_frozen:
            blockers.append(
                {
                    "stage": GATE_ANCHOR_MANIFEST,
                    "code": AM.BLOCKED_SHARED_ANCHOR_MANIFEST,
                    "detail": (
                        "shared anchor manifest status is "
                        f"{manifest.status!r}; retention and REUSE "
                        "certification stay BLOCKED until the "
                        "supervisor freezes it"
                    ),
                }
            )
    except AM.AnchorManifestError as e:
        manifest = None
        blockers.append(
            {
                "stage": GATE_ANCHOR_MANIFEST,
                "code": getattr(e, "code", "ANCHOR_MANIFEST_ERROR"),
                "detail": str(e),
            }
        )

    return {
        "gates_checked": gates_checked,
        "blockers": blockers,
        "shared_runtime": shared,
        "reference_contract": contract,
        "anchor_manifest": manifest,
        "teacher_config": teacher_config,
    }


def require_real_llm_provider(provider: str) -> None:
    """Fail closed unless ``provider`` is on the supervisor-owned
    whitelist (EMPTY this round — no paid/real LLM is authorized)."""
    if provider not in AUTHORIZED_REAL_LLM_PROVIDERS:
        raise RuntimeError(
            f"{E1_REAL_LLM_NOT_AUTHORIZED}: no real LLM provider is "
            f"authorized this round (whitelist is empty; got "
            f"{provider!r}). The six-role board never falls back to "
            "replay silently."
        )


def blocked_status_report(
    entrypoint: str,
    gates: dict,
    extra_blockers: list = (),
    extra: dict = None,
) -> dict:
    """The honest BLOCKED JSON report (every blocker listed)."""
    report = {
        "entrypoint": entrypoint,
        "branch": git_branch(),
        "head_sha": git_head_sha(),
        "status": "BLOCKED",
        "gates_checked": list(gates["gates_checked"]),
        "blockers": list(gates["blockers"]) + list(extra_blockers),
        "shared_runtime": gates["shared_runtime"],
        # round-3 honesty pins: every REAL_* flag stays false while
        # the shared runtime is unbound — never hand-set true
        "flags": {
            "real_envcoder_used": False,
            "real_student_reference_eval": False,
            "real_training_update_executed": False,
        },
        "real_one_update_executed": False,
        "student_candidate_id": PINNED_STUDENT_CANDIDATE_ID,
    }
    if extra:
        report.update(extra)
    return report


def write_json_report(report: dict, relative_path: str) -> str:
    """Write the JSON report under the siege root; returns the path."""
    path = os.path.join(SIEGE_ROOT, relative_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
