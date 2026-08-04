#!/usr/bin/env python3
"""E3 production driver: the SIGNED RUNTIME BUNDLE entrypoint (P0-16).

This is the ONE real asset injection channel for a production frontier
window.  Everything the window needs — Student checkpoint, memory artifact,
capture provenance, frozen formal asset registry, fresh-process restore
request, shared anchor manifest, retention contract, original training
runtime, capability descriptor and the injected surfaces — arrives through a
single controller-signed bundle manifest.  NOTHING is guessed: no
environment variables, no ad-hoc key=value overrides, no defaults.

Usage (venv python, PYTHONPATH=<repo>/gpu1_aggregation_siege/src,
JAX_PLATFORMS=cpu):

    python run_e3_runtime_bundle.py \
        --runtime-bundle=<signed manifest path> [--check-only] [--out=<DIR>]

``--check-only`` validates the bundle, mounts the Student, rebuilds the
typed assets and runs the production preflight, then STOPS before the
pipeline.  Without it the pipeline runs only if the preflight is green.
Unknown arguments fail closed (exit 4).

Exit codes: 0 PASS, 4 FAIL, 5 BLOCKED.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "gpu1_aggregation_siege" / "reports" / "simulator_frontier_foundation"

PASS, FAIL, BLOCKED = 0, 4, 5


def _log(msg: str) -> None:
    print(f"[e3-runtime-bundle] {msg}", flush=True)


def parse_args(argv):
    bundle_path = None
    check_only = False
    out_dir = str(DEFAULT_OUT_DIR)
    for arg in argv:
        if arg.startswith("--runtime-bundle="):
            bundle_path = arg.split("=", 1)[1]
        elif arg == "--check-only":
            check_only = True
        elif arg.startswith("--out="):
            out_dir = arg.split("=", 1)[1]
        else:
            raise ValueError(
                f"unknown argument {arg!r}: the signed runtime bundle is the ONLY "
                "injection channel — no ad-hoc overrides, no env-var guessing")
    if not bundle_path:
        raise ValueError("--runtime-bundle=<signed manifest path> is required")
    return bundle_path, check_only, out_dir


def _finish(report: dict, out_dir: str) -> None:
    try:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "e3_runtime_bundle.json"
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=False)
        os.replace(tmp, path)
        print(f"[e3-runtime-bundle] report: {path} ({path.stat().st_size} B)",
              flush=True)
    except Exception as exc:
        print(f"[e3-runtime-bundle] REPORT_WRITE_FAILED: {exc!r}", flush=True)
    print(f"[e3-runtime-bundle] verdict={report.get('verdict')} "
          f"reason={report.get('reason', '-')}", flush=True)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    started = time.time()
    report: dict = {
        "schema": "simulator_frontier.e3_runtime_bundle/v1",
        "REAL_ACTUAL_N_EXECUTED": False,
        "REAL_TWO_LLM_EXECUTED": False,
        "REAL_ONE_UPDATE_EXECUTED": False,
        "CHECKPOINT_RELOAD": False,
        "disclaimers": [
            "the signed runtime bundle is the only asset injection channel: "
            "no environment variables, no ad-hoc overrides, nothing guessed",
            "a blocked preflight is reported as BLOCKED, never as executed",
            "the Student network/reward/action head/optimizer/original loss are "
            "never modified by this driver",
        ],
    }
    try:
        bundle_path, check_only, out_dir = parse_args(argv)
    except ValueError as exc:
        report["verdict"] = "FAIL"
        report["reason"] = f"RUNTIME_BUNDLE_USAGE: {exc}"
        _finish(report, str(DEFAULT_OUT_DIR))
        return FAIL
    report["runtime_bundle_path"] = bundle_path
    report["check_only"] = bool(check_only)

    try:
        from dicode.simulator_frontier import runtime_bundle as rb
        from dicode.simulator_frontier.discovery_provenance import (
            clear_injected_production_registry,
            inject_frozen_formal_asset_registry,
            production_registry_bound,
        )
        from dicode.simulator_frontier.e3_window import (
            E3WindowConfig,
            one_window_pipeline,
            run_e3_preflight,
        )
        from dicode.simulator_frontier.errors import (
            InvalidEvidenceError,
            ProductionBlockedError,
        )
        from dicode.simulator_frontier.branch_search_runner import MemoryArtifactRef
        from dicode.simulator_frontier.memory_modes import (
            MemoryRestoreMode,
            MemoryRestoreRequest,
        )
        from dicode.simulator_frontier.training_runtime import (
            mint_original_training_runtime,
        )
    except Exception as exc:
        report["verdict"] = "BLOCKED"
        report["reason"] = f"BLOCKED_ENVIRONMENT: {exc!r}"
        _finish(report, out_dir)
        return BLOCKED

    # ------------------------------------------------------------------
    # 1. load + strictly validate the signed manifest, resolve file assets
    # ------------------------------------------------------------------
    try:
        manifest = rb.load_runtime_bundle_manifest(bundle_path)
        rb.validate_runtime_bundle_manifest(manifest)
        resolved = rb.resolve_bundle_asset_files(manifest)
    except InvalidEvidenceError as exc:
        report["verdict"] = "FAIL"
        report["reason"] = f"RUNTIME_BUNDLE_INVALID: {exc}"
        _finish(report, out_dir)
        return FAIL
    report["bundle_id"] = manifest["bundle_id"]
    report["run_id"] = manifest["run_id"]
    report["controller_signature_ref"] = manifest["controller_signature_ref"]
    report["resolved_assets"] = resolved
    _log(f"bundle {manifest['bundle_id']} manifest validated; "
         f"{len(resolved)} file assets resolved")

    # ------------------------------------------------------------------
    # 2. mount the Student exactly as the bundle names it
    # ------------------------------------------------------------------
    try:
        from dicode.student_adapters.registry import (
            default_profile_dir,
            load_student_profile,
        )
        from dicode.student_adapters.rmt16_adapter import RMT16StudentAdapter
    except Exception as exc:
        report["verdict"] = "BLOCKED"
        report["reason"] = f"BLOCKED_ENVIRONMENT: {exc!r}"
        _finish(report, out_dir)
        return BLOCKED

    student_spec = manifest["student"]
    try:
        profile = load_student_profile(
            default_profile_dir() / f"{student_spec['profile']}.yaml")
    except Exception as exc:
        report["verdict"] = "FAIL"
        report["reason"] = f"profile load failed (fail closed): {exc!r}"
        _finish(report, out_dir)
        return FAIL
    if profile.architecture_family != "RMT16":
        report["verdict"] = "BLOCKED"
        report["reason"] = (f"architecture_family {profile.architecture_family} has "
                            "no production adapter this round (RMT16 only)")
        _finish(report, out_dir)
        return BLOCKED
    try:
        adapter = RMT16StudentAdapter(profile)
        loaded = adapter.load_full_state(
            str(student_spec["checkpoint_path"]), profile.expected_identity())
        report["REAL_CHECKPOINT_LOADED"] = True
    except Exception as exc:
        report["verdict"] = "FAIL"
        report["REAL_CHECKPOINT_LOADED"] = False
        report["reason"] = f"checkpoint load gate chain failed: {exc!r}"
        _finish(report, out_dir)
        return FAIL
    identity = adapter.identity()
    if str(student_spec["abi_identity_hash"]) != str(identity.identity_hash()):
        report["verdict"] = "FAIL"
        report["reason"] = (
            "RUNTIME_BUNDLE_IDENTITY_MISMATCH: bundle declares abi_identity_hash "
            f"{str(student_spec['abi_identity_hash'])[:16]}… but the mounted "
            f"checkpoint identity is {str(identity.identity_hash())[:16]}… "
            "(fail closed)")
        _finish(report, out_dir)
        return FAIL
    report["candidate_id"] = profile.candidate_id

    # ------------------------------------------------------------------
    # 3. rebuild every typed asset from the bundle (nothing guessed)
    # ------------------------------------------------------------------
    def _read_json(name: str, path: str):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError) as exc:
            raise InvalidEvidenceError(
                f"RUNTIME_BUNDLE: {name} payload unreadable: {exc!r}") from exc

    try:
        capability = rb.capability_from_payload(
            manifest["training_surface_capability"],
            adapter_identity_hash=str(identity.identity_hash()))
        capture_provenance = rb.capture_provenance_from_payload(
            manifest["capture_provenance"])
        registry_payload = _read_json(
            "formal asset registry",
            str(manifest["formal_asset_registry_payload_path"]))
        registry = rb.discovery_registry_from_payload(registry_payload)
        restore_request = rb.restore_request_from_payload(_read_json(
            "restore request", str(manifest["restore_request_payload_path"])))
        anchor_manifest = rb.anchor_manifest_from_payload(_read_json(
            "anchor manifest", str(manifest["anchor_manifest_payload_path"])))
        from dicode.simulator_frontier.anchor_manifest import validate_anchor_manifest
        validate_anchor_manifest(anchor_manifest)
        retention = rb.retention_from_payload(manifest["retention"])
        loss_fn = rb.import_entrypoint(
            str(manifest["training_runtime"]["loss_entrypoint"]), "original loss")
        update_fn = rb.import_entrypoint(
            str(manifest["training_runtime"]["update_entrypoint"]),
            "optimizer update")
        training_runtime = mint_original_training_runtime(
            loss_fn=loss_fn,
            optimizer_update_fn=update_fn,
            runtime_id=str(manifest["training_runtime"]["runtime_id"]),
            loss_name=str(manifest["training_runtime"]["loss_name"]),
            optimizer_name=str(manifest["training_runtime"]["optimizer_name"]),
            contract_ref=str(manifest["training_runtime"]["contract_ref"]))
        taskparam_apply_fn = rb.import_entrypoint(
            str(manifest["taskparam_apply_entrypoint"]), "taskparam application")
        success_predicate = rb.import_entrypoint(
            str(manifest["predicates"]["success_entrypoint"]), "success predicate")
        progress_fn = rb.import_entrypoint(
            str(manifest["predicates"]["progress_entrypoint"]), "progress function")
        memory_loader = rb.import_entrypoint(
            str(manifest["memory"]["loader_entrypoint"]), "memory loader")
    except InvalidEvidenceError as exc:
        report["verdict"] = "FAIL"
        report["reason"] = f"RUNTIME_BUNDLE_ASSET_INVALID: {exc}"
        _finish(report, out_dir)
        return FAIL
    except Exception as exc:
        report["verdict"] = "BLOCKED"
        report["reason"] = f"RUNTIME_BUNDLE_ASSET_UNRESOLVED: {exc!r}"
        _finish(report, out_dir)
        return BLOCKED
    for purpose, target in (("loss_fn", loss_fn), ("update_fn", update_fn),
                            ("taskparam_apply_fn", taskparam_apply_fn),
                            ("success_predicate", success_predicate),
                            ("progress_fn", progress_fn),
                            ("memory_loader", memory_loader)):
        if not callable(target):
            report["verdict"] = "FAIL"
            report["reason"] = (
                f"RUNTIME_BUNDLE_ASSET_INVALID: {purpose} entry point resolved to "
                f"a non-callable ({type(target).__name__})")
            _finish(report, out_dir)
            return FAIL

    # Cross-bindings: the restore request must describe THIS checkpoint and
    # THIS adapter identity; the memory artifact must bind THIS adapter.
    if restore_request.checkpoint_sha256 != str(student_spec["checkpoint_sha256"]):
        report["verdict"] = "FAIL"
        report["reason"] = (
            "RUNTIME_BUNDLE_CROSS_BINDING: restore request checkpoint_sha256 does "
            "not equal the bundle's student.checkpoint_sha256 (fail closed)")
        _finish(report, out_dir)
        return FAIL
    if restore_request.student_abi_identity_hash != str(identity.identity_hash()):
        report["verdict"] = "FAIL"
        report["reason"] = (
            "RUNTIME_BUNDLE_CROSS_BINDING: restore request "
            "student_abi_identity_hash does not equal the mounted adapter "
            "identity (fail closed)")
        _finish(report, out_dir)
        return FAIL
    memory_spec = manifest["memory"]
    if str(memory_spec["student_identity_hash"]) != str(identity.identity_hash()):
        report["verdict"] = "FAIL"
        report["reason"] = (
            "RUNTIME_BUNDLE_CROSS_BINDING: memory artifact student_identity_hash "
            "does not equal the mounted adapter identity (fail closed)")
        _finish(report, out_dir)
        return FAIL
    if str(memory_spec["memory_spec_hash"]) != str(adapter.memory_spec().spec_hash()):
        report["verdict"] = "FAIL"
        report["reason"] = (
            "RUNTIME_BUNDLE_CROSS_BINDING: memory artifact memory_spec_hash does "
            "not equal the mounted adapter memory spec hash (fail closed)")
        _finish(report, out_dir)
        return FAIL
    memory_artifact = MemoryArtifactRef(
        path=str(memory_spec["artifact_path"]),
        sha256=str(memory_spec["artifact_sha256"]),
        memory_spec_hash=str(memory_spec["memory_spec_hash"]),
        student_identity_hash=str(memory_spec["student_identity_hash"]))
    memory_request = MemoryRestoreRequest(
        mode=MemoryRestoreMode(str(memory_spec["mode"])),
        policy_architecture_id=str(profile.architecture_family),
        checkpoint_id=str(profile.params_sha256))

    # ------------------------------------------------------------------
    # 4. inject the frozen formal asset registry (single shot, PRODUCTION)
    # ------------------------------------------------------------------
    try:
        if production_registry_bound():
            clear_injected_production_registry()
        inject_frozen_formal_asset_registry(registry)
    except Exception as exc:
        report["verdict"] = "FAIL"
        report["reason"] = f"RUNTIME_BUNDLE_REGISTRY_REJECTED: {exc!r}"
        _finish(report, out_dir)
        return FAIL

    # ------------------------------------------------------------------
    # 5. assemble the window config and run the production preflight
    # ------------------------------------------------------------------
    search = manifest["search"]
    paths = manifest["paths"]
    config = E3WindowConfig(
        run_id=str(manifest["run_id"]),
        student=adapter,
        student_params=loaded.get("params"),
        loaded_state=loaded,
        training_surface_capability=capability,
        max_timesteps=int(search["max_timesteps"]),
        reset_seed=int(search["reset_seed"]),
        capture_at_step=int(search["capture_at_step"]),
        capture_provenance=capture_provenance,
        memory_mode=str(memory_spec["mode"]),
        memory_request=memory_request,
        memory_artifact=memory_artifact,
        memory_loader=memory_loader,
        success_predicate=success_predicate,
        progress_fn=progress_fn,
        requested_n=int(search["requested_n"]),
        horizon=int(search["horizon"]),
        seed_base=int(search["seed_base"]),
        restore_request=restore_request,
        scratch_dir=str(paths["scratch_dir"]),
        two_llm_runtime=None,  # the bundle schema enforces null this round
        anchor_manifest=anchor_manifest,
        retention=retention,
        mixed_episodes=int(search["mixed_episodes"]),
        episode_horizon=int(search["episode_horizon"]),
        training_runtime=training_runtime,
        taskparam_apply_fn=taskparam_apply_fn,
        archive_path=str(paths["archive_path"]),
        checkpoint_dir=str(paths["checkpoint_dir"]),
    )

    pre = run_e3_preflight(config)
    report["preflight"] = {
        "ready": pre.ready,
        "gates": dict(pre.gates),
        "blockers": list(pre.blockers),
        "preflight_version": pre.preflight_version,
    }
    _log(f"preflight ready={pre.ready} blockers={list(pre.blockers)}")

    if not pre.ready:
        report["verdict"] = "BLOCKED"
        report["reason"] = "E3_PREFLIGHT_BLOCKED: " + "; ".join(pre.blockers)
        report["elapsed_s"] = round(time.time() - started, 2)
        _finish(report, out_dir)
        return BLOCKED

    if check_only:
        report["verdict"] = "PASS"
        report["reason"] = ("preflight green under --check-only: pipeline NOT "
                            "started (check-only never executes the window)")
        report["elapsed_s"] = round(time.time() - started, 2)
        _finish(report, out_dir)
        return PASS

    try:
        result = one_window_pipeline(config)
    except ProductionBlockedError as exc:
        report["verdict"] = "BLOCKED"
        report["reason"] = str(exc)
        _finish(report, out_dir)
        return BLOCKED
    except Exception as exc:
        report["verdict"] = "FAIL"
        report["reason"] = f"one_window_pipeline failed: {exc!r}"
        _finish(report, out_dir)
        return FAIL

    report["pipeline_status"] = result["status"]
    report["REAL_ACTUAL_N_EXECUTED"] = bool(result.get("real_actual_n_executed"))
    report["REAL_TWO_LLM_EXECUTED"] = bool(result.get("real_two_llm_executed"))
    report["REAL_ONE_UPDATE_EXECUTED"] = bool(result.get("real_one_update_executed"))
    report["CHECKPOINT_RELOAD"] = bool(result.get("checkpoint_reload"))
    report["steps"] = result.get("steps", {})
    report["elapsed_s"] = round(time.time() - started, 2)
    if result["status"] == "PASS":
        report["verdict"] = "PASS"
        _finish(report, out_dir)
        return PASS
    if result["status"] == "SELECTOR_REJECTED":
        report["verdict"] = "BLOCKED"
        report["reason"] = "evidence selector rejected the plan (no update executed)"
        _finish(report, out_dir)
        return BLOCKED
    report["verdict"] = "FAIL"
    report["reason"] = f"pipeline status {result['status']}"
    _finish(report, out_dir)
    return FAIL


if __name__ == "__main__":
    raise SystemExit(main())
