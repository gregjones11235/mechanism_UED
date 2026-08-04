"""E1 round-3 P0-6: the longrun entrypoint — PREPARE ONLY, never start.

This script freezes the COMPLETE run identity into a manifest:

* ``total_env_steps``       — the pinned horizon 98304;
* Student identity          — the pinned strong-Student candidate id
                              (CC4 owns the real checkpoint);
* Reference identity        — the G1 contract; UNFROZEN => refused;
* seed                      — the teacher ``selection.seed`` pin;
* anchor manifest           — the G3 shared manifest; DRAFT => refused;
* Git SHA                   — the LIVE ``git rev-parse HEAD``; an
                              unresolvable SHA is refused, never guessed;
* config hash               — sha256 over the teacher config bytes;
* checkpoint hash           — via the shared FullStateCheckpoint
                              contract (duck-typed); unbound => refused;
* output directory          — derived deterministically from the SHA.

ANY unfrozen field => the run is REFUSED (exit non-zero) and every
unfrozen field is listed with its blocker code. Default semantics are
``--prepare-only``: print the frozen manifest, write the JSON, and
exit WITHOUT entering any training loop. ``--launch`` additionally
requires every production gate (shared runtime bound, real EnvCoder
backend authorized, real LLM provider authorized) — this round none
of those holds, so launch is honestly refused; the script NEVER
starts the run on its own.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_production_runtime as RT  # noqa: E402

ENTRYPOINT = "scripts/run_e1_longrun.py"
DEFAULT_MANIFEST_OUT = os.path.join(
    "reports", "e1_formal_ued", "e1_longrun_manifest.json"
)

#: unfrozen-field codes (greppable)
E1_REFERENCE_UNFROZEN = "REFERENCE_CONTRACT_UNFROZEN"
E1_ANCHOR_UNFROZEN = "ANCHOR_MANIFEST_NOT_FROZEN"
E1_SHA_UNRESOLVED = "E1_GIT_SHA_UNRESOLVED"
E1_CHECKPOINT_UNBOUND = "E1_CHECKPOINT_HASH_UNBOUND"
E1_SEED_MISSING = "E1_SELECTION_SEED_MISSING"
E1_LAUNCH_UNAUTHORIZED = "E1_LONGRUN_LAUNCH_UNAUTHORIZED"


def _field(value, frozen: bool, code: str = "", detail: str = "") -> dict:
    return {
        "value": value,
        "frozen": frozen,
        "code": code,
        "detail": detail,
    }


def build_frozen_manifest(teacher_config_path: str) -> dict:
    """Compute every manifest field from REAL state — never guess."""
    fields = {}
    blockers = []

    # ---- total_env_steps (supervisor-pinned constant) ----------------
    fields["total_env_steps"] = _field(
        RT.LONGRUN_TOTAL_ENV_STEPS, True
    )

    # ---- Student identity (pinned; CC4 owns the checkpoint) ----------
    fields["student_candidate_id"] = _field(
        RT.PINNED_STUDENT_CANDIDATE_ID, True
    )

    # ---- Reference identity (G1 contract; must be FROZEN) ------------
    teacher_config = RT.load_yaml(
        os.path.join(RT.SIEGE_ROOT, teacher_config_path)
    )
    from dicode.teachers.e1_formal.reference_contract import (
        ReferenceContractError,
        consume_reference_identity_contract,
        reference_identity_sha256,
    )

    rc_block = teacher_config["teacher"]["reference_contract"]
    try:
        contract = consume_reference_identity_contract(
            rc_block, "e1_longrun.reference_contract"
        )
        fields["reference_identity_sha256"] = _field(
            reference_identity_sha256(contract), True
        )
        fields["reference_candidate_id"] = _field(
            contract.candidate_id, True
        )
    except ReferenceContractError as e:
        fields["reference_identity_sha256"] = _field(
            None, False, e.code, str(e)
        )
        fields["reference_candidate_id"] = _field(
            None, False, e.code, str(e)
        )
        blockers.append(
            {
                "field": "reference_identity",
                "code": e.code,
                "detail": str(e),
            }
        )

    # ---- seed (the teacher selection pin) ------------------------------
    seed = teacher_config["teacher"]["selection"].get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        fields["seed"] = _field(
            seed,
            False,
            E1_SEED_MISSING,
            f"selection.seed must be an int pin, got {seed!r}",
        )
        blockers.append(
            {
                "field": "seed",
                "code": E1_SEED_MISSING,
                "detail": f"selection.seed is not an int pin: {seed!r}",
            }
        )
    else:
        fields["seed"] = _field(seed, True)

    # ---- anchor manifest (G3; must be FROZEN, never DRAFT) -----------
    from dicode.teachers.e1_formal import anchor_manifest as AM

    anchor_path = os.path.join(RT.SIEGE_ROOT, RT.ANCHOR_MANIFEST_PATH)
    with open(anchor_path, "r", encoding="utf-8") as handle:
        manifest_mapping = json.load(handle)
    try:
        manifest = AM.consume_anchor_manifest(
            manifest_mapping, "e1_longrun.anchor_manifest"
        )
        if manifest.is_frozen:
            fields["anchor_manifest_sha256"] = _field(
                manifest.manifest_sha256, True
            )
        else:
            fields["anchor_manifest_sha256"] = _field(
                None,
                False,
                E1_ANCHOR_UNFROZEN,
                f"anchor manifest status is {manifest.status!r}; the "
                "supervisor must freeze it before any longrun",
            )
            blockers.append(
                {
                    "field": "anchor_manifest",
                    "code": E1_ANCHOR_UNFROZEN,
                    "detail": (
                        f"status {manifest.status!r} (expected "
                        f"{AM.STATUS_FROZEN!r})"
                    ),
                }
            )
    except AM.AnchorManifestError as e:
        fields["anchor_manifest_sha256"] = _field(
            None, False, getattr(e, "code", "ANCHOR_MANIFEST_ERROR"), str(e)
        )
        blockers.append(
            {
                "field": "anchor_manifest",
                "code": getattr(e, "code", "ANCHOR_MANIFEST_ERROR"),
                "detail": str(e),
            }
        )

    # ---- Git SHA (LIVE; an unresolvable SHA is unfrozen) ---------------
    head_sha = RT.git_head_sha()
    if head_sha:
        fields["git_head_sha"] = _field(head_sha, True)
    else:
        fields["git_head_sha"] = _field(
            "",
            False,
            E1_SHA_UNRESOLVED,
            "git rev-parse HEAD failed; the run identity can never "
            "rest on an unresolvable SHA",
        )
        blockers.append(
            {
                "field": "git_head_sha",
                "code": E1_SHA_UNRESOLVED,
                "detail": "git rev-parse HEAD failed",
            }
        )

    # ---- config hash (sha256 over the teacher config bytes) -----------
    config_path = os.path.join(RT.SIEGE_ROOT, teacher_config_path)
    fields["teacher_config_sha256"] = _field(
        RT.file_sha256(config_path), True
    )
    fields["teacher_config_path"] = _field(teacher_config_path, True)

    # ---- checkpoint hash (shared contract only; NEVER a second loader) -
    from dicode.teachers.e1_formal import shared_runtime_seam as SRS

    resolution = SRS.resolve_all_shared_runtime()["FullStateCheckpoint"]
    if resolution.bound:
        checkpoint_hash = getattr(
            resolution.contract, "checkpoint_sha256", ""
        )
        if checkpoint_hash:
            fields["checkpoint_sha256"] = _field(checkpoint_hash, True)
        else:
            fields["checkpoint_sha256"] = _field(
                "",
                False,
                E1_CHECKPOINT_UNBOUND,
                "the shared FullStateCheckpoint contract exposes no "
                "checkpoint_sha256 surface",
            )
            blockers.append(
                {
                    "field": "checkpoint_sha256",
                    "code": E1_CHECKPOINT_UNBOUND,
                    "detail": "no checkpoint_sha256 on the shared "
                    "contract",
                }
            )
    else:
        fields["checkpoint_sha256"] = _field(
            None, False, resolution.code, resolution.detail
        )
        blockers.append(
            {
                "field": "checkpoint_sha256",
                "code": resolution.code,
                "detail": resolution.detail,
            }
        )

    # ---- output directory (deterministically derived) ------------------
    if head_sha:
        output_dir = os.path.join("outputs", "e1_longrun", head_sha[:12])
        fields["output_dir"] = _field(output_dir, True)
    else:
        fields["output_dir"] = _field(
            "", False, E1_SHA_UNRESOLVED, "derived from the git SHA"
        )

    return {
        "entrypoint": ENTRYPOINT,
        "branch": RT.git_branch(),
        "fields": fields,
        "blockers": blockers,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-config", default=RT.TEACHER_CONFIG_PATH
    )
    parser.add_argument(
        "--manifest-out",
        default=DEFAULT_MANIFEST_OUT,
        help="manifest JSON path relative to gpu1_aggregation_siege/",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="request launch (STILL gated; unauthorized this round — "
        "the script never starts the run on its own)",
    )
    args = parser.parse_args(argv)

    manifest = build_frozen_manifest(args.teacher_config)
    unfrozen = [b for b in manifest["blockers"]]

    if args.launch:
        # launch additionally requires EVERY production gate
        gates = RT.resolve_production_gates(
            teacher_config_path=args.teacher_config
        )
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
        for blocker in gates["blockers"]:
            unfrozen.append(
                {
                    "field": blocker["stage"],
                    "code": blocker["code"],
                    "detail": blocker["detail"],
                }
            )

    status = "REFUSED" if unfrozen else "PREPARED"
    report = {
        "entrypoint": ENTRYPOINT,
        "branch": manifest["branch"],
        "head_sha": RT.git_head_sha(),
        "status": status,
        "prepare_only": True,  # this script NEVER starts the run
        "launch_requested": bool(args.launch),
        "launch_granted": False,  # unauthorized this round, always
        "total_env_steps": RT.LONGRUN_TOTAL_ENV_STEPS,
        "student_candidate_id": RT.PINNED_STUDENT_CANDIDATE_ID,
        "fields": manifest["fields"],
        "blockers": unfrozen,
    }
    path = RT.write_json_report(report, args.manifest_out)

    print(f"E1 LONGRUN MANIFEST [{status}] -> {path}")
    for name in sorted(manifest["fields"]):
        field = manifest["fields"][name]
        marker = "FROZEN  " if field["frozen"] else "UNFROZEN"
        print(f"  [{marker}] {name} = {field['value']!r}")
    for blocker in unfrozen:
        print(
            f"  - REFUSED [{blocker['field']}] {blocker['code']}: "
            f"{blocker['detail']}"
        )
    if status == "PREPARED":
        print(
            "all fields frozen; manifest prepared. Launch is a "
            "SEPARATE supervisor-authorized step — this entrypoint "
            "never starts the training loop."
        )
        return 0
    print(
        f"longrun REFUSED: {len(unfrozen)} field(s) unfrozen; the "
        "run is not started and no manifest is treated as launch "
        "permission."
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
