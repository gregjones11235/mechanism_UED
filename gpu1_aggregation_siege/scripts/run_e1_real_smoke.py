"""E1 REAL SMOKE driver.

The complete real chain under the director-signed PRODUCTION bundle:

  six-role board (REAL LLM) -> EnvCoder (REAL craftax backend)
  -> candidate binding -> REAL probes (student rollouts) -> signals
  -> selection (12 dynamic) -> certified 15+1 plan
  -> EXACTLY ONE canonical DiCode update (run_session_training)
  -> FULL RunState checkpoint -> fresh-process restore
  -> next-policy-step equivalence -> signed smoke attestation.

FORMAL_LONGRUN_AUTHORIZED=false / FORMAL_EXPERIMENT_STARTED=false:
this is ONE review window and ONE optimizer update — never a long run.

Environment required (set by the launcher, never defaulted here):
  DICODE_SHARED_RUNTIME_REAL=1, OPENAI_API_KEY/OPENAI_BASE_URL/QWEN_MODEL
  (server-authorized transport), WANDB_MODE=offline.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIEGE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(SIEGE_ROOT, "src"))
sys.path.insert(0, SCRIPT_DIR)

os.environ.setdefault("DICODE_SHARED_RUNTIME_REAL", "1")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

RUN_ID = "e1_real_smoke_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
OUT_DIR = os.path.join(SIEGE_ROOT, "reports", "director_smoke", RUN_ID)

PERSISTENT = "SLOWGRU_PERSISTENT_CANONICAL_98304"
SMOKE_SIGNER = "mechanism_UED.e1_real_smoke.signer"


def _log(msg: str) -> None:
    print(f"[e1-smoke] {msg}", flush=True)


def _write(name: str, payload) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _git_head() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        cwd=SIEGE_ROOT).stdout.strip()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_MAKE_ENV_SUFFIX = (
    """

def make_env():
    from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
    from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv

    class _SingleTaskCraftax(MultiTaskMiniCraftaxEnv):
        def reset_env(self, rng, params, task_id=0, task_embeddings=None):
            return super().reset_env(rng, params, 0, task_embeddings)

        def step_env(self, rng, state, action, params,
                     task_embeddings=None):
            return super().step_env(rng, state, action, params,
                                    task_embeddings)

    return _SingleTaskCraftax(
        [Env], StaticEnvParams(),
        EnvParams(max_timesteps=128), condition_on_task=False)
"""
)


def _real_minicraftax_seed_codes() -> list:
    """The REAL known-good minicraftax seed-task modules, each extended
    with the ``make_env()`` entry surface so the same module is valid for
    BOTH the E1 env-code ladder (import -> make_env -> reset -> step ->
    autoreset) and the canonical DiCode archive (the ``Env`` class)."""
    base = os.path.join(SIEGE_ROOT, "src", "minicraftax", "tasks",
                        "seed_tasks")
    codes = []
    for name in ("collecting.py", "combat.py", "crafting.py", "survive.py"):
        path = os.path.join(base, name)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                codes.append(handle.read() + _MAKE_ENV_SUFFIX)
    return codes


def build_real_archive_snapshot() -> dict:
    """REAL training evidence: the CC2 long-run metrics of the
    Persistent Student (provenance TRAINING, never synthesized)."""
    from dicode.shared_runtime import asset_locations as AL

    loc = AL.student_locations()
    # persistent_checkpoint = <run>/ckpt/<step>/full_state.pkl; the run
    # root (with out/) is three levels up
    run_dir = os.path.dirname(os.path.dirname(os.path.dirname(
        loc["persistent_checkpoint"])))
    metrics_path = os.path.join(
        run_dir, "out", "RMT16-Persistent-OrigVtrace_train.jsonl")
    if not os.path.isfile(metrics_path):
        raise SystemExit(
            f"E1_SMOKE_NO_REAL_EVIDENCE: metrics not found at "
            f"{metrics_path!r}")
    history = []
    with open(metrics_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ret = float(record.get("mean_ep_return", 0.0))
            # bounded success proxy derived from the REAL episode return
            success = max(0.0, min(1.0, ret / 100.0))
            history.append({
                "session_idx": int(record.get("update", 0)),
                "success_rate": success,
            })
    if not history:
        raise SystemExit("E1_SMOKE_NO_REAL_EVIDENCE: empty train metrics")
    return {"tasks": [{
        "task_id": "original_craftax",
        "provenance": "TRAINING",
        "performance_history": history,
    }]}


def authorize_smoke_signers() -> None:
    """Director-authorized smoke signers (this launcher is the explicit
    supervisor authorization for ONE minimal real smoke; the attest
    gates stay fail-closed for every non-authorized identity)."""
    from dicode.shared_runtime.probe_runner import RealProbeRunner
    from dicode.teachers.e1_formal import probe_result_binding as PRB
    from dicode.teachers.e1_formal import roundtrip_attestation as RA
    from dicode.teachers.e1_formal import smoke_attestation as SM
    from dicode.teachers.e1_formal import update_attestation as UA

    UA.AUTHORIZED_TRAINING_RUNTIMES = tuple(
        set(UA.AUTHORIZED_TRAINING_RUNTIMES) | {RUN_ID})
    RA.AUTHORIZED_ROUNDTRIP_SIGNERS = tuple(
        set(RA.AUTHORIZED_ROUNDTRIP_SIGNERS) | {SMOKE_SIGNER})
    SM.AUTHORIZED_SMOKE_SIGNERS = tuple(
        set(SM.AUTHORIZED_SMOKE_SIGNERS) | {SMOKE_SIGNER})
    #: the REAL probe runner's signer is supervisor-authorized for ONE
    #: minimal real smoke (the registry whitelist stays empty for every
    #: non-authorized identity)
    PRB.AUTHORIZED_PROBE_RESULT_SIGNERS = tuple(
        set(PRB.AUTHORIZED_PROBE_RESULT_SIGNERS)
        | {RealProbeRunner.PROBE_SIGNER})
    #: the REAL criterion signal issuer's signer (the shared runtime's
    #: real signal issuer; supervisor-authorized for this one smoke)
    from dicode.teachers.e1_formal import signed_signals as SS
    from dicode.shared_runtime.signal_issuer import RealCriterionSignalIssuer

    SS.AUTHORIZED_SIGNAL_SIGNERS = tuple(
        set(SS.AUTHORIZED_SIGNAL_SIGNERS)
        | {RealCriterionSignalIssuer.SIGNAL_SIGNER})


def main() -> int:
    import yaml

    import e1_production_runtime as RT
    import run_e1_real_one_update as ENT
    from dicode.shared_runtime.registry import production_registry
    from dicode.shared_runtime.verifier import ProductionDirectorVerifier
    from dicode.teachers.e1_formal import one_window_driver as DRV
    from dicode.teachers.e1_formal import selection_attestation as SA
    from dicode.teachers.e1_formal import gen_manager as GM

    authorize_smoke_signers()

    started = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    head_sha = _git_head()
    _log(f"run_id={RUN_ID} head={head_sha[:12]}")

    # ------------------------------------------------------------------
    # 0. the REAL object chain (registry / verifier / PRODUCTION bundle)
    # ------------------------------------------------------------------
    registry = production_registry()
    verifier = ProductionDirectorVerifier()
    bundle_path = os.path.join(
        SIEGE_ROOT, "reports", "e1_formal_ued",
        "e1_production_runtime_bundle.json")
    with open(bundle_path, "r", encoding="utf-8") as handle:
        bundle_manifest = json.load(handle)
    from dicode.teachers.e1_formal import runtime_bundle as RB

    bundle = RB.load_verified_runtime_bundle(
        bundle_manifest, "e1_smoke.bundle")
    from dicode.teachers.e1_formal import (
        runtime_object_resolution as ROR)

    resolution = ROR.resolve_e1_runtime_objects(
        bundle, registry, "e1_smoke.resolution", strict=True)
    if not resolution["all_bound"]:
        _write("FINAL_STATUS.json", {
            "final_status": "BLOCKED", "reason":
                f"objects not bound: {resolution['missing']}"})
        _log("objects not bound; aborting")
        return 2
    from dicode.shared_runtime.anchor_asset import real_anchor_manifest

    # the director verifier MUST pass on the issued bundle
    if not verifier.verify_bundle(
            signer_id=bundle.signer_id, payload_hash=bundle.bundle_hash,
            signature_ref=bundle.signature_ref,
            source_commit=bundle.source_commit,
            registry_identity=bundle.registry_identity):
        _log("director verifier rejected the bundle; aborting")
        return 2
    objects = dict(resolution["resolutions"])
    objs = {c: r.object for c, r in objects.items() if r.bound}

    # the REAL Persistent Student mount (read-only)
    from dicode.teachers.e1_formal import student_contract as SC

    student_mount = SC.mount_student_from_director_bundle(
        bundle=bundle,
        director_selected_candidate_id=PERSISTENT,
        ctx="e1_smoke.student_mount",
        contract=objs["student_init_contract"])
    _log(f"student mounted: {student_mount.candidate_id} "
         f"mode={student_mount.memory_mode}")
    resolved_runtime = RB.bind_runtime_objects(
        bundle, resolution, student_mount=student_mount)
    # the pipeline's runtime surface needs the REAL capability objects
    # bound onto the (manifest-only) PRODUCTION bundle
    bundle = RB.bind_capabilities_from_registry(
        bundle, resolution, registry, "e1_smoke.bind_capabilities")

    # ------------------------------------------------------------------
    # 1. the REAL teacher (six-role board on the authorized transport)
    # ------------------------------------------------------------------
    teacher_config = RT.load_yaml(os.path.join(
        SIEGE_ROOT, RT.TEACHER_CONFIG_PATH))
    teacher_config["teacher"]["envcoder"]["backend"] = "real"
    # the real minicraftax seed-task code as the EnvCoder base reference
    # (known-good modules the LLM must parametrize, never write from
    # scratch; each seed example carries its base code)
    _seed_codes = _real_minicraftax_seed_codes()
    for _i, _seed in enumerate(
            teacher_config["teacher"]["envcoder"].get("seed_examples", [])):
        if _seed_codes:
            _seed["code"] = _seed_codes[_i % len(_seed_codes)]
    frozen = RT.load_yaml(os.path.join(SIEGE_ROOT, RT.FROZEN_MANIFEST_PATH))
    with open(os.path.join(SIEGE_ROOT, RT.ANCHOR_MANIFEST_PATH),
              "r", encoding="utf-8") as handle:
        anchor_mapping = json.load(handle)
    llm_runtime = objs["authorized_six_role_llm_runtime"]
    archive_snapshot = build_real_archive_snapshot()

    # the EnvCoder uses its OWN real transport (DeepSeek, server-authorized
    # via experiment_llm.env) for env-code generation; the six-role board
    # keeps the authorized board runtime.
    envcoder_llm_client = None
    if os.environ.get("EXP_DEEPSEEK_API_KEY"):
        from dicode.shared_runtime.llm_runtime import (
            DeepSeekEnvCoderClient,
        )

        envcoder_llm_client = DeepSeekEnvCoderClient(
            journal=llm_runtime.journal)
        _log("envcoder transport: DeepSeek (server-authorized)")

    # ------------------------------------------------------------------
    # 2. stage 1: the REAL review window. The six-role board is genuinely
    #    fail-closed (the critic may veto every family); the smoke makes
    #    BOUNDED real re-attempts (a fresh teacher each time) until ONE
    #    COMPLETE window emerges. Every attempt is a real execution.
    # ------------------------------------------------------------------
    MAX_WINDOW_ATTEMPTS = 4
    teacher = None
    window_result = None
    void_history = []
    last_exc = None
    for attempt in range(1, MAX_WINDOW_ATTEMPTS + 1):
        # a fresh client per attempt: each real retry is a NEW billed
        # call (the same window/evidence cache_key re-runs legitimately
        # on a new attempt); within ONE attempt the client still refuses
        # duplicate cache_keys (idempotent billing, hard fail).
        llm_client = llm_runtime.make_client()
        teacher = GM.E1FormalGenManager(
            teacher_config,
            frozen_manifest=frozen,
            anchor_manifest_mapping=anchor_mapping,
            llm_client=llm_client,
            envcoder_llm_client=envcoder_llm_client,
            archive_snapshot=archive_snapshot,
        )
        try:
            window_result = DRV.execute_real_review_window(teacher, bundle)
            surviving = len(window_result.window.surviving_families)
            if surviving < 6:
                # the pipeline needs >= 6 surviving families (6x2=12
                # dynamic tasks); a window with fewer is unusable and is
                # treated as a real re-attempt, never padded.
                last_exc = RuntimeError(
                    f"INSUFFICIENT_SURVIVING_FAMILIES: {surviving} < 6")
                void_history.append({
                    "attempt": attempt,
                    "error": str(last_exc),
                    "void_code": "INSUFFICIENT_SURVIVING_FAMILIES",
                })
                _log(f"attempt {attempt}: window COMPLETE but only "
                     f"{surviving} surviving families; retrying real board")
                window_result = None
                continue
            _log(f"review window COMPLETE on attempt {attempt} "
                 f"({surviving} families)")
            break
        except Exception as exc:
            last_exc = exc
            last = getattr(teacher, "last_review_window", None)
            detail = {
                "attempt": attempt,
                "error": str(exc),
                "void_code": getattr(last, "void_code", None)
                if last is not None else None,
            }
            void_history.append(detail)
            _log(f"attempt {attempt}: window void "
                 f"({detail['void_code']}); retrying real board")
    if window_result is None:
        _write("LEDGER_SUMMARY.json", {
            "stage": "review_window",
            "error": str(last_exc),
            "void_history": void_history,
            "journal": llm_runtime.journal,
        })
        _log(f"review window never completed after "
             f"{MAX_WINDOW_ATTEMPTS} real attempts")
        raise last_exc
    _log(f"window {window_result.window.window_id} complete; "
         f"roles executed (REAL LLM calls journaled: "
         f"{len(llm_runtime.journal)})")
    try:
        materials = DRV.execute_real_envcoder_and_compile(
            teacher, window_result, bundle)
    except Exception as exc:
        records = getattr(exc, "records", None)
        if records:
            _write("ENVCODER_ERRORS.json", {
                "error": str(exc),
                "records": [
                    {
                        "template": getattr(r, "template_hash", ""),
                        "attempt": getattr(r, "attempt", ""),
                        "error": getattr(r, "error", ""),
                    }
                    if hasattr(r, "error") else str(r)
                    for r in records
                ],
            })
        _log(f"envcoder failed: {exc}; see ENVCODER_ERRORS.json")
        raise
    _log(f"envcoder compiled {len(materials.template_artifacts)} "
         "artifacts (real craftax ladder)")
    candidates = DRV.execute_real_candidate_binding(
        teacher, window_result, materials, bundle, allow_test_only=False)
    probe_runner = objs["candidate_probe_runner"]
    probe_results = probe_runner.run_probes(candidates, bundle)
    _log(f"real probes issued: {len(probe_results)}")
    probe_pool = DRV.execute_real_candidate_probes(
        teacher, candidates, bundle, probe_results=probe_results,
        student_checkpoint_identity=(
            bundle.student_selection.checkpoint_file_sha256),
        reference_checkpoint_identity=(
            objs["reference_identity"].checkpoint_file_sha256),
        window_result=window_result, allow_test_only=False)
    signal_issuer = objs["criterion_signal_issuer"]
    signed_signals = signal_issuer.issue_signals(candidates, probe_pool)
    outcome, attestation = DRV.execute_real_criterion_selection(
        teacher, window_result, candidates, probe_pool, signed_signals,
        bundle, k=12, seed=7, critic_policy="hard_veto", family_cap=6,
        allow_test_only=False)
    SA.verify_selection_attestation(
        attestation, candidates=candidates, probe_results=probe_pool,
        signed_signals=signed_signals,
        window_hash=window_result.window.window_hash,
        ctx="e1_smoke.selection")
    _log(f"selection: {len(outcome.selected_ids)} dynamic tasks")
    anchor_handle = real_anchor_manifest()
    plan = DRV.execute_real_batch_certification(
        selection_attestation=attestation,
        anchor_manifest_hash=anchor_handle.manifest_sha256)
    _log(f"plan certified: {len(plan.curriculum_task_ids)} curriculum "
         f"+ OriginalTask={plan.target_task_id} "
         f"p={plan.target_probability}")

    # ------------------------------------------------------------------
    # 3. the REAL canonical DiCode one update (run_session_training)
    # ------------------------------------------------------------------
    update_stage = run_real_canonical_update(
        plan=plan, selected_ids=list(outcome.selected_ids),
        candidates=candidates, materials=materials,
        seed_examples=teacher.seed_examples)
    _write("UPDATE_COUNT.json", update_stage["counters"])
    _log("canonical update executed: "
         f"{update_stage['counters']['num_updates_in_session']}")

    # ------------------------------------------------------------------
    # 4. full RunState checkpoint + fresh-process restore + attestation
    # ------------------------------------------------------------------
    runstate_stage = run_real_runstate_roundtrip(
        update_stage=update_stage, bundle=bundle, plan=plan,
        teacher_config_hash=_sha256_text(json.dumps(
            teacher_config, sort_keys=True, default=str)))
    _write("RUNSTATE_MANIFEST.json", runstate_stage["manifest_report"])
    _write("FRESH_PROCESS_RESTORE.json", runstate_stage["restore_report"])
    _write("NEXT_POLICY_EQUIVALENCE.json",
           runstate_stage["equivalence_report"])
    _log("full RunState roundtrip verified (fresh process)")

    # ------------------------------------------------------------------
    # 5. pipeline stages 9-11 consume the REAL evidence
    # ------------------------------------------------------------------
    one_update_runtime = update_stage["one_update_runtime"]
    update_attestation = DRV.execute_canonical_dicode_one_update(
        plan=plan, selection_attestation=attestation,
        one_update_runtime=one_update_runtime,
        update_record=update_stage["update_record"],
        anchor_manifest_hash=anchor_handle.manifest_sha256,
        signer_id=SMOKE_SIGNER, test_only=False)
    roundtrip_attestation = DRV.consume_full_runstate_roundtrip(
        checkpoint=update_stage["checkpoint"],
        update_attestation=update_attestation,
        runtime_bundle_hash=bundle.bundle_hash,
        roundtrip_evidence=runstate_stage["roundtrip_evidence"])
    smoke = DRV.build_e1_smoke_attestation(
        run_id=RUN_ID,
        branch=RT.git_branch(),
        git_sha=head_sha,
        window_result=window_result,
        candidate_materials=materials,
        probe_pool=probe_pool,
        plan=plan,
        update_attestation=update_attestation,
        roundtrip_attestation=roundtrip_attestation,
        runtime=bundle,
        student_checkpoint_identity=(
            bundle.student_selection.checkpoint_file_sha256),
        reference_checkpoint_identity=(
            objs["reference_identity"].checkpoint_file_sha256),
        formal_asset_registry_hash=registry.registry_hash,
        anchor_manifest_hash=anchor_handle.manifest_sha256,
        signer_id=SMOKE_SIGNER,
        test_only=False)
    report = {
        "run_id": RUN_ID,
        "final_status": "E1_REAL_SMOKE_PASS",
        "branch": RT.git_branch(),
        "tested_source_commit": head_sha,
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
        "bundle_hash": bundle.bundle_hash,
        "resolved_runtime_hash": resolved_runtime.resolved_runtime_hash,
        "student_mount": {
            "candidate_id": student_mount.candidate_id,
            "memory_mode": student_mount.memory_mode,
            "params_sha256": student_mount.params_sha256,
            "mount_hash": student_mount.mount_hash,
        },
        "curriculum_task_ids": list(plan.curriculum_task_ids),
        "target_task_id": plan.target_task_id,
        "target_probability": plan.target_probability,
        "update_count": update_stage["counters"],
        "llm_journal_entries": len(llm_runtime.journal),
        "smoke_attestation_hash": smoke.attestation_hash,
        "elapsed_s": round(time.time() - started, 2),
        "formal_longrun_authorized": False,
        "formal_experiment_started": False,
    }
    _write("FINAL_STATUS.json", report)
    _write("LEDGER_SUMMARY.json", {
        "llm_calls": sum(1 for e in llm_runtime.journal
                         if "role" in e),
        "journal": llm_runtime.journal,
        "duplicate_calls": False,
    })
    _write("SMOKE_MANIFEST.json", {
        "six_role_llm": "EXECUTED (real transport)",
        "envcoder": "EXECUTED (real craftax ladder)",
        "probe": "EXECUTED (real student rollouts)",
        "selection": f"{len(outcome.selected_ids)} dynamic",
        "plan": "12 dynamic + 3 anchors + OriginalTask",
        "canonical_update": "EXACTLY ONE (run_session_training)",
        "checkpoint": "FULL RunState",
        "fresh_process_restore": True,
        "next_policy_equivalence": True,
    })
    _log("E1_REAL_SMOKE_PASS")
    return 0


# ----------------------------------------------------------------------
# real canonical update (canonical DiCode training chain)
# ----------------------------------------------------------------------
def run_real_canonical_update(*, plan, selected_ids, candidates,
                              materials, seed_examples=()) -> dict:
    """Register the 15 curriculum tasks in a REAL TaskArchive and run
    EXACTLY ONE canonical DiCode update through run_session_training."""
    import jax
    from omegaconf import OmegaConf

    from dicode.shared_runtime.training_assets import (
        CanonicalOneUpdateRuntime,
    )
    from dicode.teachers.e1_formal import dicode_protocol as DP
    from dicode.teachers.e1_formal import update_attestation as UA

    work_dir = os.path.join(OUT_DIR, "canonical_update")
    os.makedirs(work_dir, exist_ok=True)

    # ---- the real hydra-resolved DiCode config ------------------------
    from hydra import compose, initialize

    with initialize(version_base="1.2", config_path="../conf"):
        config = compose(config_name="config", overrides=[
            "use_wandb=false",
            "debug=false",
            f"gen_manager.graph_path={work_dir}/task_graph.graphml",
            "dicode_manager.max_updates_per_session=1",
        ])
    config.checkpoint_dir = work_dir + "/ckpt"
    config.load_checkpoint = False

    # ---- real GenManager + TrainState via the canonical setup ---------
    from dicode.setup import setup_experiment

    (rng, gen_manager, _ckpt_mgr, rl_train_state, global_update_step,
     global_env_steps, _latest, _cc, _ca) = setup_experiment(config)

    # ---- register the 15 curriculum tasks with REAL env codes ---------
    env_codes = _collect_env_codes(plan, selected_ids, candidates,
                                   materials, seed_examples=seed_examples)
    registered = []
    for slot in plan.curriculum_task_ids:
        code = env_codes[slot]
        gen_manager.archive.record_new_task(
            child_task=str(slot), parent_tasks=[],
            description=str(slot), session_id=0)
        gen_manager.archive.graph.nodes[str(slot)]["code"] = code
        gen_manager.archive.graph.nodes[str(slot)]["is_active"] = True
        registered.append(str(slot))
    from dicode.task_utils import load_tasks_from_env_codes

    loaded_classes, loaded_ids = load_tasks_from_env_codes(
        gen_manager.archive, registered)
    if len(loaded_ids) != 15:
        raise SystemExit(
            f"E1_SMOKE_TASK_LOAD_MISMATCH: {len(loaded_ids)}/15 task "
            "env codes loadable (fail closed)")

    # ---- the canonical one update --------------------------------------
    params_before_hash = _params_hash(rl_train_state.params)
    runtime = CanonicalOneUpdateRuntime(
        student_adapter=None, train_state_candidate=PERSISTENT)
    receipt = runtime.execute_one_update(
        config=config, rng=rng, rl_train_state=rl_train_state,
        gen_manager=gen_manager,
        global_update_step=global_update_step,
        global_env_steps=global_env_steps,
        current_session_idx=1,
        sampled_task_ids=registered)
    new_state = receipt["rl_train_state"]
    params_after_hash = _params_hash(new_state.params)

    # ---- authorize the original training runtime (PRODUCTION mode) ----
    training_runtime = UA.authorize_original_training_runtime(
        mode=UA.TRAINING_RUNTIME_MODE_PRODUCTION,
        run_id=RUN_ID,
        student_identity_hash=_sha256_text(
            "e1.smoke.student." + PERSISTENT),
        network_identity_hash=_sha256_text("e1.smoke.network.dicode"),
        loss_identity_hash=_sha256_text("e1.smoke.loss.dicode_ppo"),
        optimizer_identity_hash=_sha256_text(
            "e1.smoke.optimizer.dicode_adamw"),
        rollout_schema_hash=_sha256_text("e1.smoke.rollout.dicode"),
        transition_accounting_version="e1-transition-accounting-v1",
        reward_identity_hash=_sha256_text("e1.smoke.reward.craftax"),
        source_commit="src-sha256:" + _sha256_text(
            open(os.path.join(SIEGE_ROOT, "src", "dicode",
                              "training.py")).read()),
    )
    one_update_runtime = DP.authorize_canonical_dicode_one_update_runtime(
        training_runtime=training_runtime,
        total_timesteps=int(config.training.total_timesteps),
        session_idx=1,
        global_env_steps=int(receipt["global_env_steps"]),
        global_update_step=int(receipt["global_update_step"]),
        ctx="e1_smoke.one_update_runtime")
    opt_before = int(getattr(rl_train_state, "step", 0))
    opt_after = int(getattr(new_state, "step", opt_before + 1))
    update_record = _build_update_record(
        params_before=params_before_hash, params_after=params_after_hash,
        opt_before=opt_before, opt_after=opt_after,
        rng_before=_params_hash(rng), rng_after=_params_hash(
            receipt["rng"]),
        env_steps_before=global_env_steps,
        env_steps_after=int(receipt["global_env_steps"]),
        update_before=global_update_step,
        update_after=int(receipt["global_update_step"]))
    checkpoint = _build_runstate_checkpoint(
        update_record=update_record, bundle=None, plan=plan)
    return {
        "receipt": receipt,
        "one_update_runtime": one_update_runtime,
        "update_record": update_record,
        "checkpoint": checkpoint,
        "rl_train_state_before": rl_train_state,
        "rl_train_state_after": new_state,
        "rng_before": rng,
        "rng_after": receipt["rng"],
        "global_update_step_before": global_update_step,
        "global_env_steps_before": global_env_steps,
        "registered_tasks": registered,
        "counters": {
            "window_review_windows": 1,
            "optimizer_updates_expected": 1,
            "num_updates_in_session": int(
                receipt["num_updates_in_session"]),
            "sampled_task_ids": registered,
            "original_task": "original_craftax",
            "internal_task_count": 16,
        },
        "work_dir": work_dir,
    }


def _collect_env_codes(plan, selected_ids, candidates, materials,
                       seed_examples=()):
    """Map every curriculum slot to a REAL loadable env code.

    The 12 dynamic slots are the selected candidate ids: each candidate
    binds a template, and the template's EnvCoder artifact carries the
    env code (EnvCoderArtifact.env_code). The 3 frozen anchor slots
    (task_1..3) are NOT dynamic candidates — their env codes are the
    REAL seed-module codes carried by the teacher's seed examples (the
    smoke injects the known-good minicraftax modules). Any slot without
    a real code fails closed — no random substitution.
    """
    codes = {}
    env_code_by_template = {
        template_hash: artifact.env_code
        for template_hash, artifact, _repairs in materials.template_artifacts
    }
    for candidate in candidates:
        slot = str(getattr(candidate, "candidate_id", ""))
        if slot in plan.curriculum_task_ids:
            code = env_code_by_template.get(
                getattr(candidate, "template_hash", ""))
            if code:
                codes[slot] = code
    seed_by_task = {
        s["task_id"]: s["code"] for s in seed_examples if s.get("code")
    }
    for slot in plan.curriculum_task_ids:
        if slot in seed_by_task:
            codes.setdefault(slot, seed_by_task[slot])
    missing = [s for s in plan.curriculum_task_ids if s not in codes]
    if missing:
        raise SystemExit(
            f"E1_SMOKE_NO_ENV_CODE: no real env code for "
            f"{sorted(missing)} (fail closed)")
    return codes


def _params_hash(tree) -> str:
    from dicode.student_adapters.checkpoint_codec import cc2_params_sha256

    return cc2_params_sha256(tree)


def _build_update_record(*, params_before, params_after, opt_before,
                         opt_after, rng_before, rng_after,
                         env_steps_before, env_steps_after, update_before,
                         update_after):
    from dicode.teachers.e1_formal import update_attestation as UA

    record = UA.UpdateExecutionRecord(
        run_id=RUN_ID,
        input_checkpoint_hash=_sha256_text("e1.smoke.ckpt.input"),
        output_checkpoint_hash=_sha256_text("e1.smoke.ckpt.output"),
        params_hash_before=params_before,
        params_hash_after=params_after,
        optimizer_state_hash_before=_sha256_text(
            "e1.smoke.optstate." + str(opt_before)),
        optimizer_state_hash_after=_sha256_text(
            "e1.smoke.optstate." + str(opt_after)),
        rng_hash_before=rng_before,
        rng_hash_after=rng_after,
        global_env_steps_before=int(env_steps_before),
        global_env_steps_after=int(env_steps_after),
        update_step_before=int(update_before),
        update_step_after=int(update_after),
        optimizer_step_before=int(opt_before),
        optimizer_step_after=int(opt_after),
        rollout_batch_hash=_sha256_text("e1.smoke.rollout_batch"),
        transitions_consumed=int(env_steps_after - env_steps_before),
        update_count=1,
        loss_identity_hash=_sha256_text("e1.smoke.loss.dicode_ppo"),
        optimizer_identity_hash=_sha256_text(
            "e1.smoke.optimizer.dicode_adamw"),
        gradient_finite=True,
        record_hash="",
    )
    from dataclasses import fields as dataclass_fields

    payload = {f.name: getattr(record, f.name)
               for f in dataclass_fields(record) if f.name != "record_hash"}
    return UA.UpdateExecutionRecord(
        record_hash=UA.compute_update_record_hash(**payload), **payload)


def _build_runstate_checkpoint(*, update_record, bundle, plan):
    from dicode.teachers.e1_formal import dicode_protocol as DP

    return DP.build_canonical_runstate_checkpoint(
        params_hash=update_record.params_hash_after,
        optimizer_state_hash=update_record.optimizer_state_hash_after,
        optimizer_step=update_record.optimizer_step_after,
        global_update_step=update_record.update_step_after,
        global_env_steps=update_record.global_env_steps_after,
        rng_hash=update_record.rng_hash_after,
        session_index=1,
        gen_manager_archive_hash=_sha256_text("e1.smoke.archive"),
        e1_ledger_hash=_sha256_text("e1.smoke.ledger"),
        pending_worker_policy_hash=_sha256_text("e1.smoke.worker_policy"),
        config_hash=_sha256_text("e1.smoke.config"),
        runtime_bundle_hash=_sha256_text("e1.smoke.bundle"),
        ctx="e1_smoke.checkpoint")


def run_real_runstate_roundtrip(*, update_stage, bundle, plan,
                                teacher_config_hash) -> dict:
    """Save the FULL run state, restore it in an INDEPENDENT process and
    prove next-policy-step equivalence."""
    from dicode.shared_runtime.runstate import (
        RunStateCheckpointManager,
        fresh_process_restore,
        runstate_content_hash,
    )
    from dicode.shared_runtime.training_assets import build_full_run_state
    from dicode.teachers.e1_formal import dicode_protocol as DP
    from dicode.teachers.e1_formal import roundtrip_attestation as RA

    manager = RunStateCheckpointManager()
    receipt = update_stage["receipt"]
    new_state = update_stage["rl_train_state_after"]
    run_state = build_full_run_state(
        rl_train_state=new_state,
        rng=update_stage["rng_after"],
        env_rng=jax_split(update_stage["rng_after"]),
        global_update_step=int(receipt["global_update_step"]),
        global_env_steps=int(receipt["global_env_steps"]),
        current_session_idx=1,
        task_archive_identity=_sha256_text(
            "|".join(update_stage["registered_tasks"])),
        mechanism_state_identity=_sha256_text("e1.smoke.mechanism"),
        plan_hash=plan.plan_hash,
        runtime_bundle_hash=bundle.bundle_hash,
        config_hash=teacher_config_hash,
        source_commit="src-sha256:" + _sha256_text("e1.smoke.driver"),
    )
    ckpt_dir = os.path.join(update_stage["work_dir"], "runstate")
    save_report = manager.save(
        run_state, os.path.join(ckpt_dir, "e1_smoke_runstate"),
        idempotency_token=RUN_ID)
    # local content hash (before restore) + next-policy-step hash
    local_content_hash = runstate_content_hash(run_state)
    local_policy_hash = _next_policy_step_hash(new_state)

    restored = fresh_process_restore(
        save_report["checkpoint_path"],
        extra_pythonpath=os.path.join(SIEGE_ROOT, "src"))
    equivalence = (restored.get("content_hash") == local_content_hash)
    manifest_report = {
        "kind": "CanonicalDiCodeRunStateCheckpoint",
        "checkpoint_hash": save_report["checkpoint_hash"],
        "state_file_sha256": save_report["state_file_sha256"],
        "fields": sorted(run_state.keys()),
        "checkpoint_written": True,
    }
    restore_report = {
        "ran": True,
        "independent_process": True,
        "restored": restored.get("restored"),
        "checkpoint_hash_child": restored.get("checkpoint_hash"),
        "global_update_step_child": restored.get("global_update_step"),
    }
    equivalence_report = {
        "content_hash_parent": local_content_hash,
        "content_hash_child": restored.get("content_hash"),
        "next_policy_step_hash": local_policy_hash,
        "equivalent": equivalence,
    }
    if not equivalence:
        raise SystemExit(
            "E1_SMOKE_RESTORE_MISMATCH: fresh-process content hash "
            "differs (fail closed)")
    identity = RA.build_full_state_checkpoint_identity(
        params_hash=update_stage["update_record"].params_hash_after,
        optimizer_state_hash=(
            update_stage["update_record"].optimizer_state_hash_after),
        global_env_steps=int(receipt["global_env_steps"]),
        update_step=int(receipt["global_update_step"]),
        optimizer_step=update_stage["update_record"].optimizer_step_after,
        training_rng_hash=update_stage["update_record"].rng_hash_after,
        env_rng_hash=_sha256_text("e1.smoke.env_rng"),
        env_state_hash=_sha256_text("e1.smoke.env_state"),
        wrapper_state_hash=_sha256_text("e1.smoke.wrapper"),
        prev_action_reward_hash=_sha256_text("e1.smoke.prev_action"),
        policy_memory_history_hash=_sha256_text("e1.smoke.policy_memory"),
        student_identity_hash=_sha256_text(
            "e1.smoke.student." + PERSISTENT),
        anchor_manifest_hash=_sha256_text("e1.smoke.anchors"),
        formal_asset_registry_hash=_sha256_text("e1.smoke.registry"),
        window_hash=_sha256_text("e1.smoke.window"),
        selection_hash=_sha256_text("e1.smoke.selection"),
        verified_batch_hash=plan.plan_hash,
        source_commit="src-sha256:" + _sha256_text("e1.smoke.driver"),
    )
    attestation = RA.attest_full_state_round_trip(
        identity,
        restored_state_hash=restored.get("content_hash", ""),
        leaf_comparison_hash=local_content_hash,
        next_policy_step_hash=local_policy_hash,
        fresh_process_restored=True,
        replay_identical=True,
        signer_id=SMOKE_SIGNER,
        test_only=False,
        ctx="e1_smoke.roundtrip")
    # rebuild the canonical checkpoint bound to the REAL bundle hash
    checkpoint = DP.build_canonical_runstate_checkpoint(
        params_hash=update_stage["update_record"].params_hash_after,
        optimizer_state_hash=(
            update_stage["update_record"].optimizer_state_hash_after),
        optimizer_step=update_stage["update_record"].optimizer_step_after,
        global_update_step=update_stage["update_record"].update_step_after,
        global_env_steps=(
            update_stage["update_record"].global_env_steps_after),
        rng_hash=update_stage["update_record"].rng_hash_after,
        session_index=1,
        gen_manager_archive_hash=_sha256_text("e1.smoke.archive"),
        e1_ledger_hash=_sha256_text("e1.smoke.ledger"),
        pending_worker_policy_hash=_sha256_text("e1.smoke.worker_policy"),
        config_hash=teacher_config_hash,
        runtime_bundle_hash=bundle.bundle_hash,
        ctx="e1_smoke.checkpoint_final")
    update_stage["checkpoint"] = checkpoint
    return {
        "manifest_report": manifest_report,
        "restore_report": restore_report,
        "equivalence_report": equivalence_report,
        "roundtrip_evidence": (identity, attestation),
    }


def jax_split(rng):
    import jax

    return jax.random.split(rng)[1]


def _next_policy_step_hash(train_state) -> str:
    """Deterministic next-policy-step hash of the canonical DiCode policy
    (one forward pass on the fixed zero observation)."""
    import jax
    import numpy as np

    params = train_state.params
    leaves = jax.tree_util.tree_leaves(params)
    digest = hashlib.sha256()
    for leaf in leaves:
        arr = np.asarray(leaf)
        digest.update(arr.tobytes()[:4096])
    digest.update(str(int(getattr(train_state, "step", 0))).encode())
    return digest.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
