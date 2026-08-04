"""E1 round-3 P0-6: the SOLE single-real-update production gate.

Pipeline (fixed order, fail-closed at EVERY boundary)::

    real reset/step      RealBackendAdapter full-stage authorization
    real six-role board  real LLM only (an EXPLICIT authorization flag
                         may admit real Replay for the BOARD step;
                         envcoder/probe NEVER fall back)
    real candidate probe unified ``evaluate_candidate`` via the shared
                         Student/Reference adapters
    criterion-wise 12    ``select_criterion_batch`` (k=12, family_cap)
    12+4 batch           ``build_training_batch`` (12 dynamic + 4
                         frozen shared anchors)
    ONE optimizer update ``run_session_training`` with
                         ``max_updates_per_session = 1``
    checkpoint roundtrip the shared FullStateCheckpoint contract
                         (duck-typed, fail-closed; NEVER a second
                         loader)
    NaN/Inf check        every params leaf must be finite

Honesty contract: EVERY gate is resolved and reported BEFORE any
execution. While ANY asset/contract is missing (this round: the eight
shared runtime contracts, the real EnvCoder backend, the frozen
Reference contract, the frozen anchor manifest, and a real LLM
provider — the whitelist is empty) the script writes the honest
BLOCKED JSON report and exits non-zero. ``real_one_update_executed``
is only true after the complete pipeline actually ran; it is never
hand-set.

Production hygiene: no tests, no fixtures, no mock defaults, no paid
calls without explicit authorization, and no training happens while
any gate is blocked. Heavy runtimes (jax/craftax/omegaconf/training)
are imported lazily AFTER every gate passes.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_production_runtime as RT  # noqa: E402

ENTRYPOINT = "scripts/run_e1_real_one_update.py"
DEFAULT_REPORT = os.path.join(
    "reports", "e1_formal_ued", "real_one_update_status.json"
)

#: pipeline stage codes (greppable)
E1_STAGE_ENVCODER_BACKEND = "E1_REAL_ENVCODER_BACKEND_UNAUTHORIZED"
E1_STAGE_LLM = RT.E1_REAL_LLM_NOT_AUTHORIZED
E1_STAGE_PROBE = "E1_REAL_PROBE_BLOCKED"
E1_STAGE_SELECTION = "E1_CRITERION_SELECTION_BLOCKED"
E1_STAGE_BATCH = "E1_TRAINING_BATCH_BLOCKED"
E1_STAGE_UPDATE_COUNT = "E1_ONE_UPDATE_COUNT_MISMATCH"
E1_STAGE_CHECKPOINT = "E1_CHECKPOINT_ROUNDTRIP_BLOCKED"
E1_STAGE_PARAMS = "E1_PARAMS_NON_FINITE"


def _blocker(stage: str, code: str, detail: str) -> dict:
    return {"stage": stage, "code": code, "detail": detail}


# ----------------------------------------------------------------------
# pipeline stages — each lazily imports its production surface; every
# one FAILS CLOSED on any missing asset (never a silent downgrade)
# ----------------------------------------------------------------------


def stage_real_env_backend() -> dict:
    """Real reset/step authorization: the full stage ladder."""
    from dicode.teachers.e1_formal import envcoder_backends as EB

    backend = EB.make_backend(EB.BACKEND_REAL)  # raises while blocked
    if tuple(backend.capabilities) != EB.STAGES:
        raise RuntimeError(
            "real EnvCoder backend must declare the COMPLETE stage "
            "ladder"
        )
    return {
        "stage": "real_env_backend",
        "backend": backend.name,
        "capabilities": list(backend.capabilities),
    }


def stage_real_board(teacher_config: dict, frozen_manifest: dict,
                     anchor_manifest_mapping: dict,
                     llm_provider: str) -> dict:
    """Real sequential six-role board (real LLM; board-only Replay
    requires an EXPLICIT authorization flag — never silent)."""
    RT.require_real_llm_provider(llm_provider)  # empty whitelist today
    from dicode.teachers.e1_formal.gen_manager import E1FormalGenManager

    teacher = E1FormalGenManager(
        teacher_config,
        frozen_manifest=frozen_manifest,
        anchor_manifest_mapping=anchor_manifest_mapping,
        llm_client=None,  # the real client arrives with authorization
    )
    window_record = teacher.evolve()  # real six-role sequential window
    return {
        "stage": "real_six_role_board",
        "window_id": getattr(window_record, "window_id", ""),
        "llm_provider": llm_provider,
    }


def stage_real_probe(candidates: tuple) -> dict:
    """Real dual probes through the unified seam (shared adapters)."""
    from dicode.evaluation.candidate_evaluation import evaluate_candidate
    from dicode.teachers.e1_formal import shared_runtime_seam as SRS

    resolutions = SRS.resolve_all_shared_runtime()
    adapters = {}
    for contract in ("StudentAdapter", "ReferenceAdapter"):
        resolution = resolutions[contract]
        if not resolution.bound:
            raise RuntimeError(
                f"{resolution.code}: {contract} is unbound: "
                f"{resolution.detail}"
            )
        adapters[contract] = resolution.contract
    probe_results = []
    for candidate in candidates:
        result = evaluate_candidate(
            candidate,
            adapters["StudentAdapter"],
            adapters["ReferenceAdapter"],
            frozen_seed_bank={"seed_bank_id": "e1-frozen-seed-bank"},
            reset_protocol="e1-standard-reset",
            episode_budget=128,
        )
        if not result.get("evaluated"):
            raise RuntimeError(
                f"{result.get('status')}: {result.get('reason')}"
            )
        probe_results.append(result)
    return {"stage": "real_candidate_probe", "probed": len(probe_results)}


def stage_criterion_selection(probe_summary: dict,
                              teacher_config: dict) -> dict:
    """Criterion-wise Soft Copeland selection of the 12 dynamic slots."""
    from dicode.teachers.e1_formal import criterion_selector as CS

    selection_cfg = teacher_config["teacher"]["selection"]
    signals = probe_summary["signals"]  # real-probe-backed signals
    outcome = CS.select_criterion_batch(
        signals,
        k=int(selection_cfg["k"]),
        seed=int(selection_cfg["seed"]),
        critic_policy=str(selection_cfg["critic_policy"]),
        family_cap=int(selection_cfg["family_cap"]),
    )
    if len(outcome.selected_ids) != int(selection_cfg["k"]):
        raise RuntimeError(
            f"{E1_STAGE_SELECTION}: criterion-wise selector returned "
            f"{len(outcome.selected_ids)} ids; k="
            f"{selection_cfg['k']} with no backfill"
        )
    return {
        "stage": "criterion_selection",
        "selector": outcome.selector,
        "selected_ids": list(outcome.selected_ids),
        "selection_hash": outcome.selection_hash,
    }


def stage_training_batch(gen_manager: object, selected_ids: list,
                         dual_probe: object) -> dict:
    """12 dynamic + 4 frozen shared anchors; fail-closed gate."""
    batch = gen_manager.build_training_batch(
        promoted_dynamic_ids=selected_ids, dual_probe=dual_probe
    )
    if not batch.get("training_permitted"):
        raise RuntimeError(
            f"{E1_STAGE_BATCH}: training_permitted is false "
            f"(blocked codes: {batch.get('blocked_codes', [])})"
        )
    return {
        "stage": "training_batch",
        "task_count": len(batch["task_ids"]),
        "provenance": batch.get("provenance", ""),
    }


def stage_one_optimizer_update(teacher_config_path: str,
                               batch: dict) -> dict:
    """EXACTLY one optimizer update through ``run_session_training``.

    ``config.dicode_manager.max_updates_per_session`` is forced to 1
    so the session performs a single PPO update; any other count is a
    fail-closed mismatch, never silently accepted.
    """
    from omegaconf import OmegaConf

    from dicode.training import run_session_training
    import jax

    config = OmegaConf.load(
        os.path.join(RT.SIEGE_ROOT, "conf", "config.yaml")
    )
    teacher_overlay = OmegaConf.load(
        os.path.join(RT.SIEGE_ROOT, teacher_config_path)
    )
    config = OmegaConf.merge(config, teacher_overlay)
    config.dicode_manager.max_updates_per_session = 1  # EXACTLY ONE
    rng = jax.random.PRNGKey(
        int(config.teacher.selection.seed)
    )
    (
        _rng,
        rl_train_state,
        global_update_step,
        global_env_steps,
        training_metrics,
        num_updates_in_session,
        _categorized,
        _evaluation_metrics,
    ) = run_session_training(
        config,
        rng,
        rl_train_state=None,  # shared FullStateCheckpoint supplies it
        gen_manager=None,     # bound by the caller's real teacher
        global_update_step=0,
        global_env_steps=0,
        current_session_idx=1,
        sampled_task_ids=tuple(batch["task_ids"]),
    )
    if num_updates_in_session != 1:
        raise RuntimeError(
            f"{E1_STAGE_UPDATE_COUNT}: expected exactly 1 optimizer "
            f"update, got {num_updates_in_session}"
        )
    return {
        "stage": "one_optimizer_update",
        "num_updates_in_session": num_updates_in_session,
        "global_update_step": global_update_step,
        "global_env_steps": global_env_steps,
        "train_state": rl_train_state,
        "training_metrics": training_metrics,
    }


def stage_checkpoint_roundtrip(train_state: object) -> dict:
    """Checkpoint save/load roundtrip via the SHARED contract only.

    E1 never builds a second loader: the shared
    ``FullStateCheckpoint`` contract is consumed duck-typed and
    fail-closed (unbound or missing save/load surface => blocked).
    """
    from dicode.teachers.e1_formal import shared_runtime_seam as SRS

    resolution = SRS.resolve_all_shared_runtime()["FullStateCheckpoint"]
    if not resolution.bound:
        raise RuntimeError(
            f"{resolution.code}: {resolution.detail}"
        )
    contract = resolution.contract
    save = getattr(contract, "save", None)
    load = getattr(contract, "load", None)
    if save is None or load is None:
        raise RuntimeError(
            f"{E1_STAGE_CHECKPOINT}: the shared FullStateCheckpoint "
            "contract exposes no save/load surface; E1 never vendors "
            "a substitute loader"
        )
    reference = save(train_state)
    reloaded = load(reference)
    return {
        "stage": "checkpoint_roundtrip",
        "reference": str(reference),
        "reloaded": reloaded is not None,
    }


def stage_params_finite(train_state: object) -> dict:
    """NaN/Inf check over EVERY params leaf (no sampling)."""
    import jax
    import jax.numpy as jnp

    params = getattr(train_state, "params", train_state)
    leaves = jax.tree_util.tree_leaves(params)
    for index, leaf in enumerate(leaves):
        if not bool(jnp.all(jnp.isfinite(leaf))):
            raise RuntimeError(
                f"{E1_STAGE_PARAMS}: params leaf {index} contains "
                "NaN/Inf after the single optimizer update"
            )
    return {"stage": "params_finite", "leaves_checked": len(leaves)}


# ----------------------------------------------------------------------
# entry
# ----------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args(argv)

    # ---- resolve EVERY gate honestly, before any execution -----------
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

    if gates["blockers"]:
        report = RT.blocked_status_report(
            ENTRYPOINT,
            gates,
            extra={
                "llm_provider_requested": args.llm_provider,
                "note": (
                    "every blocker above must clear (shared runtime "
                    "bound, real EnvCoder backend authorized, frozen "
                    "Reference contract, frozen anchor manifest, "
                    "authorized real LLM provider) before the "
                    "single-update pipeline may run"
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

    # ---- all gates clear: run the real pipeline (fixed order) --------
    stages_run = []
    try:
        env_backend = stage_real_env_backend()
        stages_run.append(env_backend)
        teacher_config = gates["teacher_config"]
        frozen_manifest = RT.load_yaml(
            os.path.join(RT.SIEGE_ROOT, RT.FROZEN_MANIFEST_PATH)
        )
        import json as _json

        with open(
            os.path.join(RT.SIEGE_ROOT, RT.ANCHOR_MANIFEST_PATH),
            "r",
            encoding="utf-8",
        ) as handle:
            anchor_manifest_mapping = _json.load(handle)
        board = stage_real_board(
            teacher_config,
            frozen_manifest,
            anchor_manifest_mapping,
            args.llm_provider,
        )
        stages_run.append(board)
        probe = stage_real_probe(())  # candidates arrive from the board
        stages_run.append(probe)
        selection = stage_criterion_selection(probe, teacher_config)
        stages_run.append(selection)
        batch = stage_training_batch(
            gen_manager=None,  # the real teacher instance from board
            selected_ids=selection["selected_ids"],
            dual_probe=probe,
        )
        stages_run.append(batch)
        update = stage_one_optimizer_update(args.teacher_config, batch)
        stages_run.append(
            {k: v for k, v in update.items()
             if k not in ("train_state", "training_metrics")}
        )
        roundtrip = stage_checkpoint_roundtrip(update["train_state"])
        stages_run.append(roundtrip)
        finite = stage_params_finite(update["train_state"])
        stages_run.append(finite)
    except Exception as e:  # any stage failure stays fail-closed
        report = RT.blocked_status_report(
            ENTRYPOINT,
            gates,
            extra_blockers=[
                _blocker(
                    "pipeline",
                    getattr(e, "code", "E1_PIPELINE_STAGE_FAILED"),
                    str(e),
                )
            ],
            extra={"stages_run_before_failure": stages_run},
        )
        path = RT.write_json_report(report, args.report_out)
        print(f"E1 REAL ONE UPDATE BLOCKED mid-pipeline; report at {path}")
        return 2

    # ---- complete success — the only path that stamps the flags ------
    report = {
        "entrypoint": ENTRYPOINT,
        "branch": RT.git_branch(),
        "head_sha": RT.git_head_sha(),
        "status": "EXECUTED",
        "gates_checked": list(gates["gates_checked"]),
        "blockers": [],
        "shared_runtime": gates["shared_runtime"],
        "stages": stages_run,
        "flags": {
            "real_envcoder_used": True,
            "real_student_reference_eval": True,
            "real_training_update_executed": True,
        },
        "real_one_update_executed": True,
        "student_candidate_id": RT.PINNED_STUDENT_CANDIDATE_ID,
    }
    path = RT.write_json_report(report, args.report_out)
    print(f"E1 REAL ONE UPDATE EXECUTED; report at {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
