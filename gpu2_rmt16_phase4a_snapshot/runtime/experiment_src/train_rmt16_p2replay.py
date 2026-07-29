#!/usr/bin/env python3
"""RMT16 × P2-Replay — unified training driver (Phase4A).

ONE driver serves all four RMT16 arms via two flags:
  --carry_mode {persistent, reset128}   the ONLY Persistent vs Reset128 difference
                                        (tokens carried across 128-step segment boundaries
                                         vs cleared at them; both clear on true done).
  --replay {on, off}                    enable/disable the P2-Full-A replay channel.
                                        off => clean RMT16 on-policy PPO (gate-4 reference).

Per outer update (directive §三 / frozen design §7):
  1. collect_rollout_rmt  (16 env x 128 steps; emits complete episodes + sparse GTrXL/RMT
                           anchors; persists episodes across rollouts via pending buffers)
  2. PPO MAIN update      (Original PPO, frozen hyperparams; rmt_ppo.ppo_update_rmt)
  3. REPLAY update        (if buffer.can_sample(): K=4 relabelable sequences ->
                           V-trace original-goal + hindsight AWR relabeled; transactional
                           KL gate <=0.05 with actor step-scale retry + rollback; EMA target)
  4. checkpoints at 0/4096/8192/12288/16384/20480/24576 (smoke: 0/4096). NO auto-98304.

Frozen coefficients (PPO + Replay) are inherited from the bakeoff Cfg + FullP2Config and
are NOT tuned here. GPU2=persistent, GPU3=reset128 (directive §二).
"""
import os, sys, json, time, hashlib, pickle, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--carry_mode", required=True, choices=["persistent", "reset128"])
# ---- Phase4A-v2 (CC2 directive §四): EXPLICIT replay mode (replaces ambiguous --replay on/off) ----
# required=True with NO default -> a missing --replay_mode is a hard argparse failure (exit 2).
# No old parameter is auto-inferred: the mode is taken verbatim, nothing else implies it.
ap.add_argument("--replay_mode", required=True,
                choices=["off", "original_vtrace", "full_p2_legacy"],
                help="off=online PPO only (no replay learner/hindsight/AWR); "
                     "original_vtrace=online PPO + Original-goal Replay V-trace (Hindsight calls==0, "
                     "AWR calls==0, no relabeled sample, no second relabeled RMT scan); "
                     "full_p2_legacy=V-trace+AWR audit/legacy path, DEFAULT-FORBIDDEN for formal "
                     "science, requires --allow-full-p2-legacy.")
ap.add_argument("--allow-full-p2-legacy", action="store_true",
                help="EXPLICIT authorization required to run --replay_mode full_p2_legacy (GATE 15).")
# ---- Phase4A-v2 (CC2 directive §六): EXPLICIT formal sequence length ----
ap.add_argument("--sequence_length", type=int, default=129,
                help="Replay loss-window length. Phase4A-v2 formal clean Carry experiment is "
                     "PRE-REGISTERED at 129 (crosses one 128-step RMT segment boundary). 512 is "
                     "retained only as ENGINEERING_LONG_WINDOW_MODE.")
ap.add_argument("--ckpt17500", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--gpu_uuid", required=True)
# ---- Phase4A-v2.2 (CC2 §六): pre-registered formal config binding (fail closed) ----
# REQUIRED for replay_mode=original_vtrace (enforced BEFORE `import jax` below; missing ->
# FORMAL_CONFIG_REQUIRED_FOR_ORIGINAL_VTRACE). replay_mode=off / probe may omit it (legacy dev
# compat). There is NO bypass parameter that relaxes the binding once a config is supplied.
ap.add_argument("--formal_config", default=None,
                help="path to the pre-registered YAML (configs/rmt16_phase4a_v2_<arm>.yaml); "
                     "required for replay_mode=original_vtrace. The driver binds it to the REAL "
                     "runtime scientific config / runtime assignment / base checkpoint identity "
                     "and refuses to proceed (FORMAL_CONFIG_RUNTIME_MISMATCH) on any difference.")
# ---- Phase4A-v2.3 (CC2 §三/§四): canonical path identity + strict run-root placement ----
# --snapshot_root pins the canonical pre-registered formal-config PATH (§三.2): realpath(args.
# formal_config) must equal realpath(snapshot_root/configs/rmt16_phase4a_v2_<arm>.yaml); byte
# copies, symlink escapes and `..` traversals are rejected. --run_root pins the STRICT out_dir
# identity (§四.3): realpath(args.out) must EQUAL realpath(run_root/<formal out_dir>) — relative
# path only, no `..`, no suffix match. Both are required for replay_mode=original_vtrace and are
# enforced BEFORE `import jax` (fail closed, no bypass).
ap.add_argument("--snapshot_root", default=None,
                help="root of the frozen experiment snapshot; pins the canonical formal-config "
                     "PATH identity (§三.2). Required for replay_mode=original_vtrace.")
ap.add_argument("--run_root", default=None,
                help="root under which the formal runtime_assignment.out_dir (a RELATIVE path) "
                     "must resolve; realpath(run_root/out_dir) must equal realpath(--out) exactly "
                     "(§四.3; no suffix match). Required for replay_mode=original_vtrace.")
ap.add_argument("--total_updates", type=int, default=12)     # 12 * 2048 = 24576
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--save_every", type=int, default=2)         # updates between saves (2 => every 4096)
# ---- Phase4A-direct-98304 (§一.2 / §二 / §三 / §六): run CLASS (NON-scientific management) ----
# formal_vtrace (DEFAULT) = the pre-registered formal Carry experiment; the ONLY class whose
# exit gate requires the replay horizon and the ONLY class authorized for a scientific claim.
# engineering_smoke = 4096-step (total_updates=2) correctness smoke; replay horizon NOT required
# for exit PASS (replay_update_count==0 / REPLAY_HORIZON_NOT_REACHED does NOT block the 98k run).
# long_run_98304 = 98304-step (total_updates=48) single-seed long horizon; no performance
# early-stop (§六). The run CLASS is bound to the config's top-level run_management block PRE-JAX
# (RTC.validate_run_class_binding) and recorded in the certificate; the frozen scientific protocol
# itself is unchanged across all three classes.
ap.add_argument("--run_class", default="formal_vtrace",
                choices=["formal_vtrace", "engineering_smoke", "long_run_98304"],
                help="run management class (NOT a scientific variable): formal_vtrace (formal "
                     "Carry experiment; replay horizon required for exit PASS; claim-authorized) "
                     "| engineering_smoke (4096-step correctness smoke; horizon NOT required) | "
                     "long_run_98304 (98304-step single-seed long run; no performance early-stop).")
# Phase4A probe (CC2 directive 2/3 + addendum): reachability probe + A/B no-perturbation gate.
ap.add_argument("--probe", action="store_true",
                help="L512 reachability probe: record-only, fixed full horizon, replay learner+hindsight OFF")
ap.add_argument("--early_stop_len", type=int, default=0,
                help="DEBUG-ONLY non-comparative early stop (0=OFF; formal probe MUST keep 0)")
ap.add_argument("--equiv_dump", action="store_true",
                help="emit per-update deterministic equivalence hashes for the A/B no-perturbation gate")
args = ap.parse_args()

# ---- Phase4A-v2 (CC2 directive §四): replay-mode validation (no auto-inference) ----
REPLAY_MODE = args.replay_mode
if REPLAY_MODE == "full_p2_legacy" and not args.allow_full_p2_legacy:
    # GATE 15: full_p2_legacy is default-forbidden for formal science.
    ap.error("--replay_mode full_p2_legacy requires explicit --allow-full-p2-legacy "
             "(default-forbidden for formal science).")

# ---- Phase4A-v2.2/2.3 (§六.2/§六.3 + §三/§四/§五.2/§六.1/§七): PRE-JAX identity + FULL binding ----
# Everything here is PURE Python (yaml/json/hashlib/os/inspect; NO jax/numpy) and runs BEFORE
# `import jax`, BEFORE CUDA env vars, BEFORE env build / ckpt load / training, in this order:
#   1. preflight  : original_vtrace REQUIRES --formal_config                (v2.2 §六.2)
#   2. arm binding: schema/arm/carry_mode/replay_mode                       (v2.2 §六.3)
#   3. FORMAL IDENTITY: canonical PATH + CONTENT (frozen file SHA / scientific SHA) (§三)
#   4. runtime_assignment: completeness + 4-way arm + exact gpu + STRICT out_dir      (§四)
#   5. FULL scientific binding: the pre-JAX runtime scientific config built from the frozen
#      pure-Python spec (phase4a_v2_frozen_spec) must equal the YAML scientific_config (§五.2)
#   6. provisional PENDING_CHECKPOINT_IDENTITY certificate written ATOMICALLY + sidecar (§六/§七)
# Only when the precheck certificate is PENDING_CHECKPOINT_IDENTITY does the driver continue to
# `import jax`. The REAL imported objects are re-bound + diffed against the frozen spec AFTER
# import (§五.3: IMPORTED_RUNTIME_CONSTANTS_MISMATCH on drift) — still before env build. The
# base checkpoint params SHA is verified after load and FINALIZES the certificate (§六.2/§六.3).
import phase4a_v2_runtime_config as RTC      # noqa: E402  (pure: yaml/json/hashlib/os)
import phase4a_v2_frozen_spec as FSPEC       # noqa: E402  (pure: stdlib only)
import phase4a_v2_formal_identity as FID     # noqa: E402  (pure: stdlib + yaml via RTC)
FORMAL_CONFIG_RECORD = None
FORMAL_CONFIG_IDENTITY = None
RUNTIME_ASSIGNMENT_RECORD = None
RUNTIME_CONFIG_CERTIFICATE = None
RUNTIME_CONFIG_CERTIFICATE_PATH = None
RUNTIME_CONFIG_CERTIFICATE_SIDECAR_PATH = None
RUNTIME_CONFIG_CERTIFICATE_FILE_SHA256 = None
# Phase4A-v2.4 (§五): the PAYLOAD SHA of the last written certificate artifact. Adopted from
# write_certificate_atomic's return at EVERY write site and pinned in manifest / summary /
# launch-status (non-null length-64 hex once a certificate has been written).
RUNTIME_CONFIG_CERTIFICATE_PAYLOAD_SHA256 = None
EXECUTED_PROTOCOL_IDENTITY = None
# Phase4A-direct-98304 (§一.2): the run_class management identity bound PRE-JAX from the config's
# run_management block. None for formal_vtrace (no run_management block -> certificate stays
# byte-identical to V2.4); a bound record for engineering_smoke / long_run_98304.
RUN_CLASS_IDENTITY = None
# Phase4A-v2.4 (§八): the DECLARED protocol definition + the EFFECTIVE protocol definition
# (declared + executed learner/sampler source identity + executed RNG identity) and its stable
# SHA256. Bound BEFORE the certificate reaches PASS (§六 ordering); None on non-formal runs.
DECLARED_PROTOCOL_DEFINITION = None
EFFECTIVE_PROTOCOL_DEFINITION = None
EFFECTIVE_PROTOCOL_SHA256 = None
try:
    RTC.preflight_require_formal_config(REPLAY_MODE, args.formal_config)
    if args.formal_config:
        FORMAL_CONFIG_RECORD = RTC.load_formal_config(args.formal_config)
        RTC.validate_arm_binding(FORMAL_CONFIG_RECORD, args.carry_mode, replay_mode=REPLAY_MODE)
        # §三 canonical formal-config PATH + CONTENT identity (frozen pre-registration).
        # Phase4A-direct-98304 (§一.3): formal_vtrace uses the FROZEN formal identity (file SHA +
        # scientific SHA pinned to the two canonical YAMLs). engineering_smoke / long_run_98304 use
        # the NON-frozen engineering identity: SAME path anti-copy protection, but the content is
        # bound by SELF-CONSISTENCY + the runtime deep_diff (their engineering budget legitimately
        # differs from the frozen formal SHAs). The frozen spec still binds every non-budget
        # scientific constant in all three classes.
        if args.run_class == "formal_vtrace":
            FORMAL_CONFIG_IDENTITY = FID.verify_formal_config_identity(
                args.snapshot_root, args.carry_mode, FORMAL_CONFIG_RECORD)
        else:
            FORMAL_CONFIG_IDENTITY = FID.verify_engineering_config_identity(
                args.snapshot_root, args.carry_mode, FORMAL_CONFIG_RECORD, args.run_class)
        # §四 fail-closed runtime_assignment (completeness + 4-way arm + exact gpu + strict
        # realpath out_dir anchored at --run_root; the v2.2 suffix match is gone).
        RUNTIME_ASSIGNMENT_RECORD = RTC.validate_runtime_assignment(
            FORMAL_CONFIG_RECORD["config"],
            cli_carry=args.carry_mode, cli_gpu=args.gpu_uuid, cli_out=args.out,
            run_root=args.run_root)
        # Phase4A-direct-98304 (§一.2): bind the run CLASS to the config's run_management block
        # PRE-JAX. formal_vtrace -> None (no block; certificate stays byte-identical to V2.4);
        # engineering_smoke / long_run_98304 -> a fail-closed bound record (run_class must match
        # the block; interruption_policy must be RESTART_FROM_STEP0; §七). This is a MANAGEMENT
        # binding only — it never touches scientific_config, so the deep_diff scientific gate and
        # the frozen spec are unaffected.
        RUN_CLASS_IDENTITY = (
            None if args.run_class == "formal_vtrace"
            else RTC.validate_run_class_binding(FORMAL_CONFIG_RECORD["config"], args.run_class))
        # §五.2 FULL scientific binding PRE-JAX (Phase4A-v2.4 §三.1: ACTUAL CLI BINDING): start
        # from the frozen pure-Python spec (single source of truth; no jax needed), then
        # OVERRIDE the seven CLI-facing keys with the ACTUAL command-line values, so the
        # precheck certificate binds what THIS process was really launched with — not the
        # frozen defaults. Any wrong CLI value (--seed / --total_updates / --save_every /
        # --sequence_length / --allow-full-p2-legacy; a wrong --replay_mode / --carry_mode is
        # caught by validate_arm_binding above) makes the runtime scientific config differ
        # from the formal YAML scientific_config => FORMAL_CONFIG_RUNTIME_MISMATCH HERE, BEFORE
        # `import jax` / CUDA env / env build / ckpt load (§三.2/§三.3).
        _prejax_kwargs = FSPEC.build_kwargs(args.carry_mode)
        _prejax_kwargs.update(
            carry_mode=args.carry_mode,
            replay_mode=REPLAY_MODE,
            allow_full_p2_legacy=bool(args.allow_full_p2_legacy),
            sequence_length=int(args.sequence_length),
            seed=int(args.seed),
            total_updates=int(args.total_updates),
            save_every=int(args.save_every))
        _PREJAX_SCIENTIFIC = RTC.build_runtime_scientific_config(**_prejax_kwargs)
        RUNTIME_CONFIG_CERTIFICATE = RTC.build_precheck_certificate(
            FORMAL_CONFIG_RECORD, _PREJAX_SCIENTIFIC,
            formal_identity_record=FORMAL_CONFIG_IDENTITY,
            assignment_record=RUNTIME_ASSIGNMENT_RECORD,
            run_class_identity=RUN_CLASS_IDENTITY,
            checkpoint_identity=RTC.build_checkpoint_identity(args.ckpt17500),
            frozen_spec_sha256=FSPEC.FROZEN_SPEC_SHA256,
            cli_args={k: v for k, v in vars(args).items()},
            runtime_constants=dict(FROZEN_SPEC_SHA256=FSPEC.FROZEN_SPEC_SHA256),
            snapshot_root=args.snapshot_root, run_root=args.run_root)
        if RUNTIME_CONFIG_CERTIFICATE["certificate_status"] != RTC.CERTIFICATE_STATUS_PENDING:
            raise ValueError(
                "FORMAL_CONFIG_RUNTIME_MISMATCH: prejax precheck certificate_status="
                + RUNTIME_CONFIG_CERTIFICATE["certificate_status"]
                + " (required: PENDING_CHECKPOINT_IDENTITY); no `import jax` / env build / ckpt "
                "load will proceed. errors: "
                + " | ".join(RUNTIME_CONFIG_CERTIFICATE["validation_errors"]))
        # §六.1/§六.4/§七 provisional PENDING certificate: ATOMIC write + payload SHA + detached
        # file-SHA sidecar. This file is OVERWRITTEN by the finalized certificate after the base
        # checkpoint SHA is verified (§六.2/§六.3); a failure can never leave a stale PASS here
        # because the status is PENDING and certificate_finalized is false.
        # Phase4A-v2.4 (§五): write_certificate_atomic returns the EXACT serialized artifact;
        # the caller adopts it immediately (RUNTIME_CONFIG_CERTIFICATE = written_certificate),
        # so the in-memory object can never drift from the disk file.
        (RUNTIME_CONFIG_CERTIFICATE_PATH, RUNTIME_CONFIG_CERTIFICATE_SIDECAR_PATH,
         RUNTIME_CONFIG_CERTIFICATE_FILE_SHA256, RUNTIME_CONFIG_CERTIFICATE_PAYLOAD_SHA256,
         RUNTIME_CONFIG_CERTIFICATE) = (
            RTC.write_certificate_atomic(
                RUNTIME_CONFIG_CERTIFICATE,
                os.path.join(args.out, "runtime_config_certificate.json")))
        # Phase4A-v2.4 (§五.2): immediately re-verify the just-written artifact (payload SHA +
        # file SHA + STRICT sidecar basename/format) after the write.
        RTC.verify_certificate_artifact(
            RUNTIME_CONFIG_CERTIFICATE_PATH, RUNTIME_CONFIG_CERTIFICATE_SIDECAR_PATH,
            expected_file_sha256=RUNTIME_CONFIG_CERTIFICATE_FILE_SHA256,
            expected_payload_sha256=RUNTIME_CONFIG_CERTIFICATE_PAYLOAD_SHA256)
        print(f"[formal-config] prejax precheck certificate_status="
              f"{RUNTIME_CONFIG_CERTIFICATE['certificate_status']} "
              f"frozen_spec_sha={FSPEC.FROZEN_SPEC_SHA256[:16]} "
              f"certificate_payload_sha={RUNTIME_CONFIG_CERTIFICATE_PAYLOAD_SHA256[:16]} "
              f"certificate_file_sha={RUNTIME_CONFIG_CERTIFICATE_FILE_SHA256[:16]} "
              f"path={RUNTIME_CONFIG_CERTIFICATE_PATH}", flush=True)
except ValueError as e:
    raise SystemExit(str(e))

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

SRC = os.path.dirname(os.path.abspath(__file__))
V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB"
for p in [SRC, V7 + "/src", V7]:
    if p not in sys.path:
        sys.path.insert(0, p)

import jax, jax.numpy as jnp, numpy as np, optax
import orbax.checkpoint as ocp
from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.task_utils import get_achievement_multi_hot
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv

from network_rmt16 import ActorCriticTransformerRMT16
import rmt16_memory as rmtm
import rng_utils as RU
import rmt_collect as RC
import rmt_ppo as PPO
import rmt_replay_learner as RL
import rmt_hindsight as RH
# Phase4A-v2.2 (§六.4): ANCHOR_INTERVAL / MIN_SEQUENCE_LENGTH are the REAL replay-protocol
# constants the formal YAML must match (not CLI args).
from rmt_replay_buffer import RMTReplayBuffer, ANCHOR_INTERVAL, MIN_SEQUENCE_LENGTH
from full_p2_learner import FullP2Config, build_optimizer
from rmt_memory_anchor import make_apply_eval_rmt
# Phase4A-v2 (CC2 directive §三/§八): split counters + Hindsight/AWR firewall.
from phase4a_v2_counters import Phase4ACounters
# Phase4A-v2.1 (CC2 §三/§四/§五): policy-lag gate identity, replay label split, fail-closed gates.
import phase4a_v2_contract as CONTRACT

REPLAY_ON = (REPLAY_MODE != "off")                       # any active replay channel
REPLAY_USES_HINDSIGHT = (REPLAY_MODE == "full_p2_legacy")  # ONLY the legacy path touches Hindsight/AWR
PROBE = bool(args.probe)
if PROBE:
    assert REPLAY_MODE == "off", "probe requires --replay_mode off (replay learner + hindsight must be OFF)"
# Phase4A-v2 (§六): pre-registered formal sequence length + segment-boundary provenance.
SEQUENCE_LENGTH = int(args.sequence_length)
SEGMENT_LEN = 128                                          # RMT16 segment boundary (ANCHOR_INTERVAL)
ENGINEERING_LONG_WINDOW_MODE = 512                         # legacy engineering window (NOT the formal one)
if REPLAY_MODE == "original_vtrace" and SEQUENCE_LENGTH <= SEGMENT_LEN:
    # The formal clean Carry experiment must CROSS one 128-step boundary (step 129 reads the
    # cross-segment token). Guard against an accidental non-crossing pre-registration.
    raise SystemExit(f"FATAL: replay_mode=original_vtrace requires sequence_length > {SEGMENT_LEN} "
                     f"(got {SEQUENCE_LENGTH}); the formal Carry experiment crosses one boundary.")
ARM_REPLAY_TAG = {"off": "-PPO", "original_vtrace": "-OrigVtrace",
                  "full_p2_legacy": "-P2ReplayLegacy"}[REPLAY_MODE]
ARM = f"RMT16-{args.carry_mode.capitalize()}{ARM_REPLAY_TAG}"

# Phase4A-v2.1 (§三.2) FAIL-CLOSED runtime alignment: the policy-lag gate identity is derived
# SOLELY from REPLAY_MODE, so original_vtrace/off can NEVER carry an active hard lag gate at
# runtime (only full_p2_legacy does). If this invariant is ever broken, refuse to run rather
# than silently imply a lag gate that the code does not implement.
_pl_manifest = CONTRACT.policy_lag_runtime_manifest(REPLAY_MODE)
if REPLAY_MODE in ("off", "original_vtrace") and _pl_manifest["policy_lag_gate_active"]:
    raise SystemExit("ORIGINAL_VTRACE_POLICY_LAG_CONFIG_CONFLICT: runtime policy-lag gate active "
                     f"under replay_mode={REPLAY_MODE} (must be inactive; V-trace importance "
                     "sampling is the only off-policy correction).")

# ----------------------- config (bakeoff frozen + P2 frozen) -----------------------
class Cfg:
    activation="relu"; embed_size=256; hidden_layers=256; num_heads=8; qkv_features=256
    num_layers=2; gating=True; gating_bias=2.0; window_mem=128; window_grad=64
    lr=2e-5; max_grad_norm=1.0; gamma=0.999; gae_lambda=0.8; clip_eps=0.2; vf_coef=0.5
    ent_coef=0.002; update_epochs=1; num_minibatches=2; num_envs=16; num_steps=128
    optimistic_reset_ratio=16; condition_on_task=True
    value_target_clip_min=-50.0; value_target_clip_max=300.0; rmt_num_tokens=16
cfg = Cfg()
ppo_cfg = dict(window_mem=cfg.window_mem, num_heads=cfg.num_heads, num_layers=cfg.num_layers,
               embed=cfg.embed_size, lr=cfg.lr, max_grad_norm=cfg.max_grad_norm,
               gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, clip_eps=cfg.clip_eps,
               vf_coef=cfg.vf_coef, ent_coef=cfg.ent_coef, update_epochs=cfg.update_epochs,
               num_minibatches=cfg.num_minibatches)
fp_cfg = FullP2Config(window_mem=cfg.window_mem, num_heads=cfg.num_heads,
                      num_layers=cfg.num_layers, embed=cfg.embed_size,
                      gamma=cfg.gamma, vt_clip_min=cfg.value_target_clip_min,
                      vt_clip_max=cfg.value_target_clip_max)
rmt_cfg = rmtm.RMT16Config(num_tokens=cfg.rmt_num_tokens, segment_len=cfg.num_steps,
                           encoder_size=cfg.embed_size)
K_BATCH = 4
L_SEQ = 512
STEPS_PER_UPDATE = cfg.num_envs * cfg.num_steps     # 2048

# ----------------------- Phase4A-v2.3 (§五.3/§八): imported constants + executed-protocol binding -----------------------
# Runs AFTER `import jax` and the REAL constant materialization (Cfg / FullP2Config / K_BATCH /
# ANCHOR_INTERVAL / MIN_SEQUENCE_LENGTH / RL.W_ORIGINAL_VTRACE) and BEFORE env build / network
# init / checkpoint load / training. The pre-JAX precheck certificate already bound the formal
# YAML to the FROZEN pure-Python spec (§五.2). Here the runtime scientific config is rebuilt from
# the REAL imported objects and diffed against the frozen spec: any drift =>
# IMPORTED_RUNTIME_CONSTANTS_MISMATCH (the finalized FAIL certificate is written and the driver
# exits). By transitivity: formal YAML == frozen spec == REAL executing constants.
# Then the replay protocol's EXECUTED source identity is bound via inspect on the real imported
# learner + sampler (§八; NOT string declarations) and reconciled with the declared labels.
if FORMAL_CONFIG_RECORD is not None:
    _imported_scientific = RTC.build_runtime_scientific_config(
        carry_mode=args.carry_mode,
        replay_mode=REPLAY_MODE,
        allow_full_p2_legacy=bool(args.allow_full_p2_legacy),
        sequence_length=SEQUENCE_LENGTH,
        segment_len=SEGMENT_LEN,
        hindsight=bool(REPLAY_USES_HINDSIGHT),
        awr=bool(REPLAY_USES_HINDSIGHT),
        w_original_vtrace=float(RL.W_ORIGINAL_VTRACE),
        base_checkpoint="ckpt17500",
        seed=int(args.seed),
        total_updates=int(args.total_updates),
        save_every=int(args.save_every),
        num_envs=int(cfg.num_envs),
        num_steps=int(cfg.num_steps),
        task="DEFEAT_KOBOLD",
        optimistic_reset_ratio=int(cfg.optimistic_reset_ratio),
        condition_on_task=bool(cfg.condition_on_task),
        replay_batch_size=int(K_BATCH),
        replay_buffer_capacity=64,               # == RMTReplayBuffer(capacity=64) below
        anchor_interval=int(ANCHOR_INTERVAL),    # frozen P2 constant (128)
        min_sequence_length=int(MIN_SEQUENCE_LENGTH),   # frozen P2 constant (129)
        eligible_only_sampling=True,             # sample_eligible is the only sampling path
        ppo_lr=float(cfg.lr), ppo_max_grad_norm=float(cfg.max_grad_norm),
        ppo_gamma=float(cfg.gamma), ppo_gae_lambda=float(cfg.gae_lambda),
        ppo_clip_eps=float(cfg.clip_eps), ppo_vf_coef=float(cfg.vf_coef),
        ppo_ent_coef=float(cfg.ent_coef), ppo_update_epochs=int(cfg.update_epochs),
        ppo_num_minibatches=int(cfg.num_minibatches),
        ppo_value_target_clip_min=float(cfg.value_target_clip_min),
        ppo_value_target_clip_max=float(cfg.value_target_clip_max),
        vtrace_rho_bar=float(fp_cfg.rho_bar), vtrace_c_bar=float(fp_cfg.c_bar),
        vtrace_vt_clip_min=float(fp_cfg.vt_clip_min),
        vtrace_vt_clip_max=float(fp_cfg.vt_clip_max),
        kl_replay_max=float(fp_cfg.kl_replay_max), kl_run_max=float(fp_cfg.kl_run_max),
        actor_step_scales=list(fp_cfg.actor_step_scales),
        policy_lag_gate_active=bool(_pl_manifest["policy_lag_gate_active"]),
        policy_lag_gate_mode=_pl_manifest["policy_lag_gate_mode"],
        policy_lag_max_policy_lag=_pl_manifest["max_policy_lag"],
        legacy_full_p2_active=False, legacy_full_p2_max_policy_lag=16,
        ema_tau=float(fp_cfg.ema_tau), ent_floor=float(fp_cfg.ent_floor),
        grad_clip=float(fp_cfg.grad_clip), adam_eps=float(fp_cfg.adam_eps),
        net_activation=cfg.activation, net_embed_size=int(cfg.embed_size),
        net_num_heads=int(cfg.num_heads), net_qkv_features=int(cfg.qkv_features),
        net_num_layers=int(cfg.num_layers), net_gating=bool(cfg.gating),
        net_gating_bias=float(cfg.gating_bias), net_window_mem=int(cfg.window_mem),
        net_rmt_num_tokens=int(cfg.rmt_num_tokens))
    _FROZEN_SCIENTIFIC = RTC.build_runtime_scientific_config(
        **FSPEC.build_kwargs(args.carry_mode))
    _IMPORTED_CONSTANTS_DRIFT = RTC.deep_diff(
        RTC.canonical_scientific_config(_FROZEN_SCIENTIFIC),
        RTC.canonical_scientific_config(_imported_scientific))
    if _IMPORTED_CONSTANTS_DRIFT:
        _drift_msg = " | ".join(
            f"{d['path']}: frozen={d['formal']!r} imported={d['runtime']!r} ({d['kind']})"
            for d in _IMPORTED_CONSTANTS_DRIFT)
        # §六: a constants-drift failure FINALIZES the certificate FAIL and overwrites the
        # provisional PENDING file before exit (no stale PENDING/PASS survives).
        RUNTIME_CONFIG_CERTIFICATE = RTC.finalize_certificate(
            RUNTIME_CONFIG_CERTIFICATE,
            checkpoint_error="IMPORTED_RUNTIME_CONSTANTS_MISMATCH: " + _drift_msg)
        (RUNTIME_CONFIG_CERTIFICATE_PATH, RUNTIME_CONFIG_CERTIFICATE_SIDECAR_PATH,
         RUNTIME_CONFIG_CERTIFICATE_FILE_SHA256, RUNTIME_CONFIG_CERTIFICATE_PAYLOAD_SHA256,
         RUNTIME_CONFIG_CERTIFICATE) = (
            RTC.write_certificate_atomic(
                RUNTIME_CONFIG_CERTIFICATE, RUNTIME_CONFIG_CERTIFICATE_PATH))
        RTC.verify_certificate_artifact(
            RUNTIME_CONFIG_CERTIFICATE_PATH, RUNTIME_CONFIG_CERTIFICATE_SIDECAR_PATH,
            expected_file_sha256=RUNTIME_CONFIG_CERTIFICATE_FILE_SHA256,
            expected_payload_sha256=RUNTIME_CONFIG_CERTIFICATE_PAYLOAD_SHA256)
        raise SystemExit(
            "IMPORTED_RUNTIME_CONSTANTS_MISMATCH: the REAL imported runtime constants drifted "
            "from the frozen pre-JAX spec (FROZEN_SPEC_SHA256=" + FSPEC.FROZEN_SPEC_SHA256
            + "); no env build / ckpt load / training will proceed. drift: " + _drift_msg)
    RUNTIME_CONFIG_CERTIFICATE["runtime_constants_imported"] = dict(
        SEQUENCE_LENGTH=SEQUENCE_LENGTH, SEGMENT_LEN=SEGMENT_LEN,
        ENGINEERING_LONG_WINDOW_MODE=ENGINEERING_LONG_WINDOW_MODE,
        K_BATCH=K_BATCH, STEPS_PER_UPDATE=STEPS_PER_UPDATE, ARM=ARM,
        ANCHOR_INTERVAL=int(ANCHOR_INTERVAL),
        MIN_SEQUENCE_LENGTH=int(MIN_SEQUENCE_LENGTH))
    RUNTIME_CONFIG_CERTIFICATE["imported_constants_binding"] = dict(
        imported_constants_match=True,
        imported_vs_frozen_drift=[],
        frozen_spec_sha256=FSPEC.FROZEN_SPEC_SHA256,
        imported_scientific_config_sha256=RTC.scientific_config_sha256(_imported_scientific))
# Phase4A-v2.4 (§六 ordering, step 5): the DEDICATED deterministically-seeded replay-sampling
# RNG is constructed HERE — after the imported-constants binding and the learner/sampler
# SOURCE binding below, BEFORE checkpoint load + certificate finalize — and the SAME instance
# is reused by training (search: replay_sample_rng). No random state is consumed before the
# certificate PASS: only its type/identity is verified (verify_rng_instance_identity draws no
# random numbers).
replay_sample_rng = np.random.RandomState(args.seed + 7)
if FORMAL_CONFIG_RECORD is not None:
    # §六/§七/§八: the ACTUALLY EXECUTED replay protocol — learner SOURCE identity, sampler
    # SOURCE identity, RNG instance identity, and the resulting EFFECTIVE protocol definition
    # + SHA — must be FULLY bound BEFORE checkpoint load / certificate finalize. ANY failure
    # of this binding FINALIZES the certificate FAIL, rewrites the provisional file ATOMICALLY
    # (and re-verifies it) and exits nonzero: the certificate can never reach PASS with an
    # unbound / partially-bound executed protocol (no declared-strings-only fallback).
    _EXECUTED_PROTOCOL_ERROR = None
    try:
        EXECUTED_PROTOCOL_IDENTITY = CONTRACT.executed_function_source_identity(
            RL.original_vtrace_update_rmt, RMTReplayBuffer.sample_eligible)
        _DECLARED_PROTOCOL_DEFINITION = CONTRACT.replay_protocol_labels(
            REPLAY_MODE, SEQUENCE_LENGTH, K_BATCH)["protocol_definition"]
        EXECUTED_PROTOCOL_IDENTITY["declaration_match"] = (
            CONTRACT.verify_executed_protocol_matches_declared(
                EXECUTED_PROTOCOL_IDENTITY, _DECLARED_PROTOCOL_DEFINITION))
        EXECUTED_PROTOCOL_IDENTITY["declared_protocol_definition"] = _DECLARED_PROTOCOL_DEFINITION
        EXECUTED_PROTOCOL_IDENTITY["rng_instance"] = CONTRACT.verify_rng_instance_identity(
            replay_sample_rng)
        DECLARED_PROTOCOL_DEFINITION = _DECLARED_PROTOCOL_DEFINITION
        EFFECTIVE_PROTOCOL_DEFINITION, EFFECTIVE_PROTOCOL_SHA256 = (
            CONTRACT.build_effective_protocol_definition(
                _DECLARED_PROTOCOL_DEFINITION, EXECUTED_PROTOCOL_IDENTITY,
                EXECUTED_PROTOCOL_IDENTITY["rng_instance"]))
        RUNTIME_CONFIG_CERTIFICATE["executed_protocol_identity"] = dict(
            learner=EXECUTED_PROTOCOL_IDENTITY["learner"],
            sampler=EXECUTED_PROTOCOL_IDENTITY["sampler"],
            rng_instance=EXECUTED_PROTOCOL_IDENTITY["rng_instance"],
            declaration_match=EXECUTED_PROTOCOL_IDENTITY["declaration_match"])
        RUNTIME_CONFIG_CERTIFICATE["declared_protocol_definition"] = DECLARED_PROTOCOL_DEFINITION
        RUNTIME_CONFIG_CERTIFICATE["effective_protocol_definition"] = (
            EFFECTIVE_PROTOCOL_DEFINITION)
        RUNTIME_CONFIG_CERTIFICATE["effective_protocol_sha256"] = EFFECTIVE_PROTOCOL_SHA256
    except Exception as exc:
        _EXECUTED_PROTOCOL_ERROR = f"{type(exc).__name__}: {exc}"
    if _EXECUTED_PROTOCOL_ERROR is not None:
        RUNTIME_CONFIG_CERTIFICATE = RTC.finalize_certificate(
            RUNTIME_CONFIG_CERTIFICATE,
            checkpoint_error="EXECUTED_PROTOCOL_BINDING_FAILURE: " + _EXECUTED_PROTOCOL_ERROR)
        (RUNTIME_CONFIG_CERTIFICATE_PATH, RUNTIME_CONFIG_CERTIFICATE_SIDECAR_PATH,
         RUNTIME_CONFIG_CERTIFICATE_FILE_SHA256, RUNTIME_CONFIG_CERTIFICATE_PAYLOAD_SHA256,
         RUNTIME_CONFIG_CERTIFICATE) = (
            RTC.write_certificate_atomic(
                RUNTIME_CONFIG_CERTIFICATE, RUNTIME_CONFIG_CERTIFICATE_PATH))
        RTC.verify_certificate_artifact(
            RUNTIME_CONFIG_CERTIFICATE_PATH, RUNTIME_CONFIG_CERTIFICATE_SIDECAR_PATH,
            expected_file_sha256=RUNTIME_CONFIG_CERTIFICATE_FILE_SHA256,
            expected_payload_sha256=RUNTIME_CONFIG_CERTIFICATE_PAYLOAD_SHA256)
        raise SystemExit(
            "EXECUTED_PROTOCOL_BINDING_FAILURE: the executed replay protocol (learner source "
            "identity / sampler source identity / RNG instance identity / effective protocol "
            "definition) could not be FULLY bound BEFORE the checkpoint finalize; the "
            "certificate is finalized FAIL and no checkpoint load / training will proceed. "
            "cause: " + _EXECUTED_PROTOCOL_ERROR)
    print(f"[formal-config] imported constants binding=PASS (drift=0 vs frozen spec) "
          f"executed_learner={EXECUTED_PROTOCOL_IDENTITY['learner']['qualname']} "
          f"executed_sampler={EXECUTED_PROTOCOL_IDENTITY['sampler']['qualname']} "
          f"learner_src_sha="
          f"{EXECUTED_PROTOCOL_IDENTITY['learner']['function_source_sha256'][:16]} "
          f"rng_class={EXECUTED_PROTOCOL_IDENTITY['rng_instance']['class_name']} "
          f"effective_protocol_sha={EFFECTIVE_PROTOCOL_SHA256[:16]}",
          flush=True)

# ----------------------- Stage4 DEFEAT_KOBOLD task (identical to bakeoff) -----------------------
S4_TASK_CODE = '''
import jax
from craftax.craftax.constants import Achievement, ItemType
from minicraftax.craftax_state import TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder
class Env(BaseTask):
    def __init__(self, sp, p):
        super().__init__(sp, p)
        self.relevant_achievements=[Achievement.DEFEAT_KOBOLD]; self.completed_achievements=[]; self.label="DEFEAT_KOBOLD"
    def get_task_params(self): return TaskParams(needs_depletion_multiplier=0.3)
    def generate_world(self, rng):
        rng,_r=jax.random.split(rng); b=WorldBuilder(_r,self.static_params,self.params)
        b.set_starting_floor(2); b.set_monsters_killed(2,8)
        b.set_player_inventory({"wood":7,"stone":27,"coal":3,"iron":3,"sapling":1,"pickaxe":3,"sword":3,"bow":1,"arrows":7,"torches":10})
        s=b.build(rng); up=b.ladders_up[2]
        return s.replace(item_map=s.item_map.at[2,up[0],up[1]].set(ItemType.NONE.value))
'''
ns4 = {}; exec(S4_TASK_CODE, ns4); S4Cls = ns4["Env"]
ctor = EnvParams(max_timesteps=4096)
table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
EMB = int(table.shape[1])
target_achievement_1d = np.asarray(table[0]).astype(np.float32)   # [n_ach] (P2 hindsight convention)
from hindsight import DEFAULT_EMBEDDING_SIZE
assert EMB == DEFAULT_EMBEDDING_SIZE, f"EMB {EMB} != DEFAULT_EMBEDDING_SIZE {DEFAULT_EMBEDDING_SIZE}"

# ----------------------- helpers -----------------------
def _params_sha(params):
    h = hashlib.sha256()
    for v in jax.tree_util.tree_leaves(params):
        h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    return h.hexdigest()

def _arr_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()

def _to_np(pt):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), pt)

def _merge(base, full):
    if isinstance(base, dict) and isinstance(full, dict):
        out = dict(full)
        for k in base:
            if k in full:
                out[k] = _merge(base[k], full[k])
        return out
    return base

# ----------------------- Phase4A-v2.4 (§四 + §六): base checkpoint load + staged fail-closed identity -----------------------
# Runs AFTER `import jax`, the imported-constants binding, the learner/sampler SOURCE binding,
# the replay-RNG construction + identity binding and the EFFECTIVE protocol build — and BEFORE
# env build / network init / optimizer / training (§六 ordering). The ENTIRE checkpoint flow is
# under ONE staged try: CheckpointManager init -> restore -> structure (raw["params"]) ->
# params extraction (base_params["params"]) -> params hash -> SHA compare. ANY failure at ANY
# stage is captured with its stage label (RTC.CHECKPOINT_FAILURE_STAGES) and FINALIZES the
# certificate FAIL: the finalized FAIL certificate is rewritten ATOMICALLY (payload SHA + file
# SHA + strict sidecar), re-verified, and the driver exits nonzero — no stale PENDING/PASS can
# survive a checkpoint failure, and a PASS certificate is only written when EVERY stage (incl.
# the SHA compare) succeeded.
base_params = None
base_inner = None
base_sha = None
_CHECKPOINT_ERROR = None
_CHECKPOINT_FAILURE_STAGE = "NONE"
try:
    _CHECKPOINT_FAILURE_STAGE = "CHECKPOINT_MANAGER_INIT"
    t0 = time.time()
    ckpt_mgr = ocp.CheckpointManager(os.path.dirname(args.ckpt17500))
    _CHECKPOINT_FAILURE_STAGE = "CHECKPOINT_RESTORE"
    raw = ckpt_mgr.restore(int(os.path.basename(args.ckpt17500)))
    _CHECKPOINT_FAILURE_STAGE = "CHECKPOINT_STRUCTURE"
    base_params = raw["params"]                 # {"params": {...}} wrapped (repo convention)
    _CHECKPOINT_FAILURE_STAGE = "CHECKPOINT_PARAMS_EXTRACTION"
    base_inner = base_params["params"]          # INNER (apply convention used by make_apply_eval_rmt)
    _CHECKPOINT_FAILURE_STAGE = "CHECKPOINT_PARAMS_HASH"
    base_sha = _params_sha(base_inner)
except Exception as exc:
    _CHECKPOINT_ERROR = f"{type(exc).__name__}: {exc}"
if base_sha is not None:
    print(f"[load] ckpt17500 leaves={len(jax.tree_util.tree_leaves(base_inner))} "
          f"sha={base_sha[:16]} ({time.time()-t0:.1f}s)", flush=True)
else:
    print(f"[load] ckpt17500 FAILED checkpoint_failure_stage={_CHECKPOINT_FAILURE_STAGE} "
          f"error={_CHECKPOINT_ERROR}", flush=True)

# Phase4A-v2.4 (§四): staged SHA compare + UNIFIED finalize. The loaded base params SHA is
# compared against the FROZEN expectation from reviewed evidence (both arms' frozen raw-probe
# summaries record this base_sha256), then the certificate is finalized: PENDING + checkpoint
# PASS/NOT_FROZEN -> PASS (stage NONE); ANY staged failure -> FAIL with its
# checkpoint_failure_stage recorded in the certificate (§四.1). The FINALIZED certificate is
# rewritten ATOMICALLY and this is the LAST write to the file — checkpoint manifests and the
# summary pin this exact file SHA (§七.3).
if RUNTIME_CONFIG_CERTIFICATE is not None:
    if _CHECKPOINT_ERROR is None:
        _CHECKPOINT_FAILURE_STAGE = "CHECKPOINT_SHA_COMPARE"
        try:
            RUNTIME_CONFIG_CERTIFICATE["checkpoint_identity"] = RTC.verify_checkpoint_params_sha(
                RUNTIME_CONFIG_CERTIFICATE["checkpoint_identity"], base_sha)
            _CHECKPOINT_FAILURE_STAGE = "NONE"
        except Exception as exc:
            _CHECKPOINT_ERROR = f"{type(exc).__name__}: {exc}"
    RUNTIME_CONFIG_CERTIFICATE = RTC.finalize_certificate(
        RUNTIME_CONFIG_CERTIFICATE,
        RUNTIME_CONFIG_CERTIFICATE["checkpoint_identity"],
        checkpoint_error=_CHECKPOINT_ERROR,
        checkpoint_failure_stage=_CHECKPOINT_FAILURE_STAGE)
    (RUNTIME_CONFIG_CERTIFICATE_PATH, RUNTIME_CONFIG_CERTIFICATE_SIDECAR_PATH,
     RUNTIME_CONFIG_CERTIFICATE_FILE_SHA256, RUNTIME_CONFIG_CERTIFICATE_PAYLOAD_SHA256,
     RUNTIME_CONFIG_CERTIFICATE) = (
        RTC.write_certificate_atomic(
            RUNTIME_CONFIG_CERTIFICATE, RUNTIME_CONFIG_CERTIFICATE_PATH))
    # Phase4A-v2.4 (§五.2): re-verify the FINAL artifact (payload SHA + file SHA + strict
    # sidecar basename/format) immediately after the write.
    RTC.verify_certificate_artifact(
        RUNTIME_CONFIG_CERTIFICATE_PATH, RUNTIME_CONFIG_CERTIFICATE_SIDECAR_PATH,
        expected_file_sha256=RUNTIME_CONFIG_CERTIFICATE_FILE_SHA256,
        expected_payload_sha256=RUNTIME_CONFIG_CERTIFICATE_PAYLOAD_SHA256)
    if RUNTIME_CONFIG_CERTIFICATE["certificate_status"] == RTC.CERTIFICATE_STATUS_PASS:
        # Phase4A-v2.4 (§五.2): on a final PASS certificate, re-read the artifact from disk and
        # require it to EQUAL the adopted in-memory RUNTIME_CONFIG_CERTIFICATE, with a non-null
        # certificate_payload_sha256 — the payload SHA in the manifest/summary is the disk truth.
        with open(RUNTIME_CONFIG_CERTIFICATE_PATH, "r") as _cert_f:
            _disk_certificate = json.load(_cert_f)
        if _disk_certificate != RUNTIME_CONFIG_CERTIFICATE:
            raise SystemExit(
                "CERTIFICATE_DISK_OBJECT_MISMATCH: the re-read disk certificate does not equal "
                "the adopted in-memory RUNTIME_CONFIG_CERTIFICATE after the final PASS write.")
        if not RUNTIME_CONFIG_CERTIFICATE.get("certificate_payload_sha256"):
            raise SystemExit(
                "CERTIFICATE_PAYLOAD_SHA_MISSING: the final PASS certificate carries no "
                "non-null certificate_payload_sha256.")
    _bc_match = RUNTIME_CONFIG_CERTIFICATE["checkpoint_identity"]["base_checkpoint_match"]
    _bc_expected = RUNTIME_CONFIG_CERTIFICATE["checkpoint_identity"][
        "base_checkpoint_expected_sha256"]
    print(f"[formal-config] FINAL certificate_status="
          f"{RUNTIME_CONFIG_CERTIFICATE['certificate_status']} "
          f"finalized={RUNTIME_CONFIG_CERTIFICATE['certificate_finalized']} "
          f"checkpoint_failure_stage="
          f"{RUNTIME_CONFIG_CERTIFICATE.get('checkpoint_failure_stage')} "
          f"base_checkpoint_match={_bc_match} expected={_bc_expected[:16]} "
          f"loaded={base_sha[:16] if base_sha else None} "
          f"certificate_payload_sha={RUNTIME_CONFIG_CERTIFICATE_PAYLOAD_SHA256[:16]} "
          f"certificate_file_sha="
          f"{RUNTIME_CONFIG_CERTIFICATE_FILE_SHA256[:16]}", flush=True)
    if RUNTIME_CONFIG_CERTIFICATE["certificate_status"] != RTC.CERTIFICATE_STATUS_PASS:
        raise SystemExit(
            "FORMAL_CONFIG_RUNTIME_MISMATCH: runtime_config_certificate_status=FAIL "
            "(checkpoint_failure_stage="
            + str(RUNTIME_CONFIG_CERTIFICATE.get("checkpoint_failure_stage"))
            + "); no training step will proceed (the finalized FAIL certificate overwrote the "
            "provisional file). errors: "
            + " | ".join(RUNTIME_CONFIG_CERTIFICATE["validation_errors"]))
elif _CHECKPOINT_ERROR is not None:
    # No formal certificate (replay_mode=off/probe legacy dev compat): a checkpoint load
    # failure is still a hard stop (pre-v2.4 this raised uncaught from the same operations).
    raise SystemExit(
        f"CHECKPOINT_LOAD_FAILURE: checkpoint_failure_stage={_CHECKPOINT_FAILURE_STAGE} "
        f"error={_CHECKPOINT_ERROR}")

# ----------------------- env + network + compat init -----------------------
print("=" * 78, flush=True)
print(f"{ARM}  driver  (Phase4A)", flush=True)
print(f"  carry_mode={args.carry_mode} replay_mode={REPLAY_MODE} "
      f"sequence_length={SEQUENCE_LENGTH} gpu={args.gpu_uuid}", flush=True)
print(f"  devices={[str(d) for d in jax.devices()]}", flush=True)
print("=" * 78, flush=True)

base_env = MultiTaskMiniCraftaxEnv([S4Cls], StaticEnvParams(), ctor, True,
                                   conditioning_type="embedding", embedding_size=EMB)
env = DistributedMultiTaskOptimisticLogWrapper(
    base_env, jax.random.PRNGKey(0), cfg.num_envs, 1, cfg.optimistic_reset_ratio,
    # RMT16 Phase4A (CC2): wire the probe so probe_term=True emits the additive _term_* keys.
    # PROBE=False (frozen runs / A arm) -> probe_term=False -> info dict bit-exact original.
    jnp.array([1.0]), table, probe_term=PROBE)
env_params = env.default_params
assert env_params is not None, 'env_params (Craftax EnvParams) must resolve before collect/env.step'
ACTION_DIM = int(env.action_space(env_params).n)
OBS_DIM = int(env.observation_space(env_params).shape[0])
fp_cfg.action_dim = ACTION_DIM; fp_cfg.obs_dim = OBS_DIM

network = ActorCriticTransformerRMT16(
    action_dim=ACTION_DIM, activation=cfg.activation, encoder_size=cfg.embed_size,
    hidden_layers=cfg.hidden_layers, num_heads=cfg.num_heads, qkv_features=cfg.qkv_features,
    num_layers=cfg.num_layers, gating=cfg.gating, gating_bias=cfg.gating_bias,
    rmt_num_tokens=cfg.rmt_num_tokens)

# compat init from ckpt17500 (base params loaded; RMT params fresh, rmt_gate=0 -> bit-exact).
# Phase4A-v2.4 (§四/§六): the base checkpoint LOAD + staged fail-closed identity + certificate
# FINALIZATION moved ABOVE the env build (search: "base checkpoint load + staged fail-closed
# identity") so the executed protocol is bound and the certificate is finalized BEFORE any env
# reset / optimizer / training work. base_inner / base_sha are in scope here.
rng_init = jax.random.PRNGKey(args.seed); rng_init, _rng = jax.random.split(rng_init)
full_params = network.init(
    _rng, jnp.zeros((2, cfg.window_mem, cfg.num_layers, cfg.embed_size)),
    jnp.zeros((2, OBS_DIM)),
    jnp.zeros((2, cfg.num_heads, 1, cfg.window_mem + 1), jnp.bool_),
    mem_tokens=jnp.zeros((2, cfg.rmt_num_tokens, cfg.embed_size)),
    seg_buf=jnp.zeros((2, cfg.num_steps, cfg.embed_size)),
    method=network.init_all)
full_inner = full_params["params"]                  # flax init returns {"params": {...}}
params = _merge(base_inner, full_inner)             # INNER: ckpt17500 base weights + fresh RMT
assert _params_sha(_merge(base_inner, base_inner)) == base_sha, "base merge sanity"
print(f"[merge] inner_leaves={len(jax.tree_util.tree_leaves(params))} base_sha={base_sha[:16]} "
      f"(rmt_gate zero-init -> bit-exact at step0)", flush=True)

# ----------------------- optimizers / EMA / buffers / state -----------------------
ppo_opt = PPO.build_ppo_optimizer(ppo_cfg)
replay_opt = build_optimizer(cfg.lr, fp_cfg)
params = jax.tree_util.tree_map(jnp.asarray, params)
ppo_opt_state = ppo_opt.init(params)
replay_opt_state = replay_opt.init(params)
target_params = params                                  # EMA target init = online (ckpt17500)

replay = RMTReplayBuffer(capacity=64, seed=args.seed)
pending = RC.RMTPendingEpisodeBuffers(cfg.num_envs, first_episode_id=0, first_policy_version=0)
apply_eval_rmt = make_apply_eval_rmt(network)

# ---- RMT read-path runtime monitor (Phase4A env_params-fix gate; READ-ONLY probe) ----
# Driver-local probe; does NOT modify network / loss / optimizer / env / eval logic.
def _leaf_norm_rm(subtree):
    return float(np.sqrt(sum(float(np.sum(np.square(np.asarray(v))))
        for v in jax.tree_util.tree_leaves(subtree))))

def _read_path_monitor(p, mem, obs, mask, tok):
    """READ-ONLY read-path activity probe on current (post-update) params + carried state.
    read-path grad = grad of (logits.mean()+value.mean()); mem on/off = tokens vs zeros."""
    def _probe_loss(pp):
        lg, vl, _mo, _ht = apply_eval_rmt(pp, mem, obs, mask, tok)
        return jnp.asarray(lg).mean() + jnp.asarray(vl).mean()
    g = jax.grad(_probe_loss)(p)
    lg_on, _v1, _m1, _h1 = apply_eval_rmt(p, mem, obs, mask, tok)
    lg_off, _v2, _m2, _h2 = apply_eval_rmt(p, mem, obs, mask, jnp.zeros_like(jnp.asarray(tok)))
    lon = jnp.asarray(lg_on); loff = jnp.asarray(lg_off)
    diff = float(jnp.max(jnp.abs(lon - loff)))
    pon = jax.nn.softmax(lon, axis=-1); poff = jax.nn.softmax(loff, axis=-1)
    kl = float(jnp.mean(jnp.sum(pon * (jnp.log(pon + 1e-12) - jnp.log(poff + 1e-12)), axis=-1)))
    return dict(gate_value=float(np.asarray(p["rmt_gate"]).reshape(-1)[0]),
                gate_probe_grad=float(np.abs(np.asarray(g["rmt_gate"])).max()),
                read_attn_probe_grad=_leaf_norm_rm(g["rmt_read_attn"]),
                read_ln_probe_grad=_leaf_norm_rm(g["rmt_read_ln"]),
                mem_on_off_logit_diff=diff, mem_on_off_KL=kl)

read_ever_nonzero = False
mem_on_off_ever_nonzero = False
# ---- Phase4A per-arm gate + replay instrumentation state (driver-only; CC2 directive) ----
online_ppo_update_count = 0
replay_not_ready_skip_count = 0
replay_sample_success_count = 0
replay_update_success_count = 0
replay_first_success_update = None
replay_last_sampled_seq_len = None
replay_last_sampled_traj_len = None
replay_sequences_consumed = 0          # Phase4A-v2 (§七): MATCHED_REPLAY_EXPOSURE numerator
ever_eligible_512 = False
# ---- Phase4A-v2 (CC2 directive §三): authoritative SPLIT counters (replace overloaded update_count) ----
counters = Phase4ACounters()
# Phase4A-v2.4 (§六): replay_sample_rng (the DEDICATED deterministically-seeded RNG for
# eligible-only replay sampling: given the same buffer state + this RNG state, sample_eligible
# produces bit-identical sample_ids/start_offsets/sequence_lengths; seeded args.seed + 7 so it
# is reproducible and independent of the JAX rollout/action RNG streams) was constructed ABOVE
# — after the imported-constants + learner/sampler source binding, BEFORE the checkpoint load +
# certificate finalize — and its identity was bound into EXECUTED_PROTOCOL_IDENTITY there. The
# SAME instance is reused below; no random state was consumed before the certificate PASS.
# ---- Phase4A-v2.1 (CC2 §五): per-arm MATCHED_REPLAY_EXPOSURE certificate accumulation ----
# These per-outer-update lists are the raw exposure record the two-arm validator compares at
# level 2 (EXPOSURE_COUNT_MATCH). sample_ids_by_outer_update / start_offsets_by_outer_update are
# per-arm INTERNAL provenance ONLY — endogenous buffers have no shared trajectory identity, so
# those two are NEVER compared across arms (compare_exposure excludes them).
replay_attempt_mask = []                 # bool per outer update: True if a replay update was ATTEMPTED
replay_attempt_outer_updates = []        # outer update indices where a replay update was attempted
replay_not_ready_outer_updates = []      # outer update indices where buffer was NOT eligible -> skip
replay_update_outer_updates = []         # outer update indices where a replay update actually RAN
replay_batch_sizes = []                  # per executed replay update: batch size (== K_BATCH)
replay_sequence_lengths = []             # per executed replay update: list of sampled seq lengths
eligible_count_by_outer_update = []      # eligible trajectory count at each attempt
sample_ids_by_outer_update = []          # INTERNAL: sampled trajectory ids (not compared cross-arm)
start_offsets_by_outer_update = []       # INTERNAL: sampled start offsets (not compared cross-arm)
persistent_carry_nonzero_all = True     # persistent: carried RMT tokens nonzero every rollout
reset128_boundary_clear_all = True      # reset128: carried RMT tokens strictly zero every rollout
gtrxl_window_finite_all = True

def _replay_stats(buf):
    """READ-ONLY eligibility query over the (frozen) replay buffer; modifies nothing."""
    lengths = [int(t.length) for t in buf]
    return dict(
        replay_buffer_trajectory_count=len(lengths),
        replay_max_trajectory_length=(max(lengths) if lengths else 0),
        replay_eligible_count_128=sum(1 for L in lengths if L >= 128),
        replay_eligible_count_256=sum(1 for L in lengths if L >= 256),
        replay_eligible_count_512=sum(1 for L in lengths if L >= 512),
        replay_lengths=lengths)

# ---- RMT read-path CONNECTIVITY probe (Phase4A; READ-ONLY, synthetic non-zero tokens) ----
# Discriminates (a) "cleared -> read output zero but write/read branch still CONNECTED" (expected
# reset128 control) from (b) "write->carry->read broken / read branch never wired to the output"
# (engineering defect). It forces the gate OPEN on an in-memory param COPY (tanh(gate)!=0) and
# injects NON-ZERO tokens, so branch WIRING is tested independently of the trained gate value
# (zero at init; may stay zero for reset128 whose rollout tokens are cleared) and independently of
# cross-boundary carry. Does NOT modify training params / optimizer / network / loss.
read_conn_ever_nonzero = False

def _read_connectivity_probe(p, mem, obs, mask, tok_shape, key):
    inj = jax.random.normal(key, tok_shape)          # synthetic NON-ZERO tokens (not from rollout)
    zeros = jnp.zeros_like(inj)
    p_open = {**p, "rmt_gate": jnp.ones_like(jnp.asarray(p["rmt_gate"]))}   # READ-ONLY forced-open copy
    lg_on, _v1, _m1, _h1 = apply_eval_rmt(p_open, mem, obs, mask, inj)
    lg_off, _v2, _m2, _h2 = apply_eval_rmt(p_open, mem, obs, mask, zeros)
    lon = jnp.asarray(lg_on); loff = jnp.asarray(lg_off)
    diff = float(jnp.max(jnp.abs(lon - loff)))
    pon = jax.nn.softmax(lon, axis=-1); poff = jax.nn.softmax(loff, axis=-1)
    kl = float(jnp.mean(jnp.sum(pon * (jnp.log(pon + 1e-12) - jnp.log(poff + 1e-12)), axis=-1)))
    top_changed = float(jnp.mean(jnp.asarray(jnp.argmax(lon, axis=-1) != jnp.argmax(loff, axis=-1))))
    def _loss(pp):
        lg, vl, _mo, _ht = apply_eval_rmt(pp, mem, obs, mask, inj)
        return jnp.asarray(lg).mean() + jnp.asarray(vl).mean()
    g = jax.grad(_loss)(p_open)   # grad at forced-open gate: nonzero iff read params are WIRED to output
    return dict(conn_logit_diff=diff, conn_KL=kl, conn_top_action_frac=top_changed,
                conn_read_attn_grad=_leaf_norm_rm(g["rmt_read_attn"]),
                conn_read_ln_grad=_leaf_norm_rm(g["rmt_read_ln"]))

scan_fn = RL._make_scan_rmt(network, fp_cfg, rmt_cfg, args.carry_mode) if REPLAY_ON else None

rng = jax.random.PRNGKey(args.seed + 1)
rng, _rng = jax.random.split(rng)
obsv, env_state = env.reset(_rng, env_params)
memories = jnp.zeros((cfg.num_envs, cfg.window_mem, cfg.num_layers, cfg.embed_size))
mem_mask = jnp.zeros((cfg.num_envs, cfg.num_heads, 1, cfg.window_mem + 1), jnp.bool_)
mem_idx = jnp.full((cfg.num_envs,), cfg.window_mem, jnp.int32)   # P2 convention (derive_anchor matches)
rmt_state = rmtm.rmt16_init(cfg.num_envs, rmt_cfg)
action_rng = RU.make_action_rng(args.seed)

CKPT_DIR = os.path.join(args.out, "ckpt"); LOG_DIR = os.path.join(args.out, "out")
os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, f"{ARM}_train.jsonl")
audit_path = os.path.join(LOG_DIR, f"{ARM}_replay_audit.jsonl")
# Phase4A directive 2/3: probe output files (only written when PROBE) + equiv gate file.
probe_episodes_path = os.path.join(LOG_DIR, f"{ARM}_probe_episodes.jsonl")
probe_updates_path = os.path.join(LOG_DIR, f"{ARM}_probe_updates.jsonl")
equiv_path = os.path.join(LOG_DIR, f"{ARM}_equiv.jsonl")
PROBE_GE_LEN = 512   # directive length>=512 eligibility threshold (RECORD only; never a stop)

update_count = 0
accepted_policy_updates = 0; kl_rejected_updates = 0; hindsight_eligible = 0; hindsight_attempts = 0
# Phase4A directive 2/3: probe cumulative state (used only when PROBE).
probe_completed_episodes = []     # cumulative per-episode termination records
probe_first_ge512 = None          # first length>=512 episode (RECORD only; never a stop)

# ----------------------- checkpoint -----------------------
def _phase4a_v2_manifest_fields():
    """Phase4A-v2 (CC2 directive §六): provenance fields recorded in every checkpoint manifest.

    sequence_length, segment_len=128, crosses_boundary (= sequence_length > 128), replay_mode,
    and the structural hindsight/awr flags. For replay_mode in {off, original_vtrace} hindsight
    and awr are STRUCTURALLY False (the path never references them); only full_p2_legacy sets
    them True."""
    fields = dict(
        sequence_length=SEQUENCE_LENGTH,
        segment_len=SEGMENT_LEN,
        crosses_boundary=bool(SEQUENCE_LENGTH > SEGMENT_LEN),
        replay_mode=REPLAY_MODE,
        hindsight=bool(REPLAY_USES_HINDSIGHT),
        awr=bool(REPLAY_USES_HINDSIGHT),
        w_original_vtrace=float(RL.W_ORIGINAL_VTRACE),
        allow_full_p2_legacy=bool(args.allow_full_p2_legacy),
        # Phase4A-direct-98304 (§一.2): the run management class recorded in every checkpoint
        # manifest (NON-scientific provenance; formal_vtrace / engineering_smoke / long_run_98304).
        run_class=args.run_class)
    # Phase4A-v2.1 (§三.2): policy-lag GATE identity. For original_vtrace this records
    # policy_lag_gate_active=false / max_policy_lag=null / off_policy_correction=vtrace; the
    # V-trace importance correction stays active, only an ADDITIONAL hard lag gate is absent.
    fields.update(CONTRACT.policy_lag_runtime_manifest(REPLAY_MODE))
    # Phase4A-v2.2 (§三.1/§三.2): the manifest's ACTIVE replay block carries an explicit
    # active_replay_config (replay_mode + policy_lag_gate_active=false + max_policy_lag=null +
    # vtrace_importance_sampling for original_vtrace). The numeric legacy lag 16 is quarantined
    # in legacy_full_p2_only with active=false — it never appears in any active manifest block.
    fields["active_replay_config"] = CONTRACT.active_replay_config_manifest(REPLAY_MODE)
    fields["legacy_full_p2_only"] = CONTRACT.legacy_full_p2_manifest(
        active=False, max_policy_lag=16)
    # Phase4A-v2.1 (§四): the four-way replay label split — SAME_REPLAY_PROTOCOL=READY does NOT
    # imply MATCHED_REPLAY_EXPOSURE (NOT_RUN) nor MATCHED_REPLAY_CONTENT (NOT_CLAIMED).
    fields["replay_labels"] = CONTRACT.replay_protocol_labels(
        REPLAY_MODE, SEQUENCE_LENGTH, K_BATCH)
    # Phase4A-v2.2 (§六.8): every checkpoint manifest (step0 full_state + train_state and all
    # later checkpoints) references the runtime_config_certificate SHAs + status + path, so the
    # binding evidence is anchored inside the artifacts themselves.
    if RUNTIME_CONFIG_CERTIFICATE is not None:
        # Phase4A-v2.3 (§七.3): the manifest pins the finalized certificate's OWN artifact
        # identity — payload SHA, FINAL file SHA, sidecar path — so checkpoint readers can
        # re-verify the certificate file byte-for-byte (verify_certificate_artifact).
        fields["runtime_config_certificate"] = RTC.certificate_shas_record(
            RUNTIME_CONFIG_CERTIFICATE,
            certificate_file_sha256=RUNTIME_CONFIG_CERTIFICATE_FILE_SHA256,
            certificate_sidecar_path=RUNTIME_CONFIG_CERTIFICATE_SIDECAR_PATH)
        fields["runtime_config_certificate_path"] = RUNTIME_CONFIG_CERTIFICATE_PATH
        fields["runtime_config_certificate_sidecar_path"] = (
            RUNTIME_CONFIG_CERTIFICATE_SIDECAR_PATH)
        fields["runtime_config_certificate_file_sha256"] = (
            RUNTIME_CONFIG_CERTIFICATE_FILE_SHA256)
        # Phase4A-v2.4 (§五.3): the certificate PAYLOAD SHA (the SHA the sidecar's file SHA is
        # derived from) and the loaded base checkpoint params SHA are first-class manifest
        # fields — non-null length-64 hex, equal to the on-disk certificate artifact / the
        # loaded base params.
        fields["runtime_config_certificate_payload_sha256"] = (
            RUNTIME_CONFIG_CERTIFICATE_PAYLOAD_SHA256)
        fields["base_checkpoint_params_sha256"] = base_sha
    return fields


def save_ckpt(step, params, ppo_opt_state, replay_opt_state, target_params, tag):
    d = os.path.join(CKPT_DIR, str(step)); os.makedirs(d, exist_ok=True)
    p_sha = _params_sha(params)
    # full_state.pkl for the frozen evaluator (params wrapped + manifest)
    with open(os.path.join(d, "full_state.pkl"), "wb") as f:
        pickle.dump({"params": _to_np(params),
                     "manifest": {"params_sha256": p_sha, "step": step, "arm": ARM,
                                  "carry_mode": args.carry_mode, "replay_mode": REPLAY_MODE,
                                  "gpu_uuid": args.gpu_uuid, "seed": args.seed,
                                  "config": {k: v for k, v in vars(cfg).items()},
                                  "phase4a_v2": _phase4a_v2_manifest_fields(),
                                  "tag": tag}}, f, protocol=4)
    # train_state.pkl for exact resume (opt/EMA/replay/rng/counters)
    with open(os.path.join(d, "train_state.pkl"), "wb") as f:
        pickle.dump({"params": _to_np(params), "ppo_opt_state": _to_np(ppo_opt_state),
                     "replay_opt_state": _to_np(replay_opt_state),
                     "target_params": _to_np(target_params),
                     "replay_buffer": replay.state_dict(), "pending": pending.state_dict(),
                     "rng": np.asarray(rng), "action_rng": RU.action_rng_state(action_rng),
                     "update_count": update_count, "global_step": step,
                     "memories": np.asarray(memories), "mem_mask": np.asarray(mem_mask),
                     "mem_idx": np.asarray(mem_idx),
                     "rmt_state": _to_np(rmt_state), "obsv": np.asarray(obsv),
                     # GATE 12: checkpoint carries params/PPO opt/Replay opt/EMA/RNG/action RNG/
                     # buffer/pending episodes/GTrXL state/RMT state AND all split counters.
                     "counters": {"accepted_policy_updates": accepted_policy_updates,
                                  "kl_rejected_updates": kl_rejected_updates,
                                  "hindsight_eligible": hindsight_eligible,
                                  "hindsight_attempts": hindsight_attempts,
                                  "replay_sequences_consumed": replay_sequences_consumed,
                                  "replay_sample_rng_state": replay_sample_rng.get_state(),
                                  "phase4a_v2": counters.snapshot()},
                     "manifest": {"params_sha256": p_sha, "step": step, "arm": ARM,
                                  "carry_mode": args.carry_mode, "replay_mode": REPLAY_MODE,
                                  "phase4a_v2": _phase4a_v2_manifest_fields()}},
                    f, protocol=4)
    print(f"[ckpt] step={step} params_sha={p_sha[:16]} tag={tag}", flush=True)
    return p_sha

# ----------------------- step-0 checkpoint -----------------------
save_ckpt(0, params, ppo_opt_state, replay_opt_state, target_params, "step0")

# ----------------------- training loop -----------------------
for u in range(args.total_updates):
    t_u = time.time()
    # 1. collect
    trajs, carry, rollout, stats = RC.collect_rollout_rmt(
        env, env_state, network, params, obsv, memories, mem_mask, mem_idx, rmt_state,
        rng, action_rng, pending, target_achievement_1d, cfg.num_steps, cfg.window_mem,
        cfg.num_heads, rmt_cfg, args.carry_mode, collected_update_count=update_count,
        apply_eval_rmt=apply_eval_rmt, env_params=env_params,
        # Phase4A-v2 (§二/§三): episode update index = outer loop index; pending-episode
        # policy_version = ACCEPTED policy version (not the loop index). In replay_mode=off
        # these coincide with the legacy values -> bit-exact (GATE 13).
        outer_update_index=u, policy_version=counters.policy_version)
    env_state = carry["env_state"]; obsv = carry["obsv"]
    memories = carry["memories"]; mem_mask = carry["mem_mask"]
    mem_idx = carry["mem_idx"]; rmt_state = carry["rmt_state"]; rng = carry["rng"]

    for t in trajs:
        assert bool(np.asarray(t.dones)[-1]), "HARD STOP episode-boundary: non-terminal trajectory"
        replay.insert(t)                       # validates GTrXL + RMT anchor conservation
        replay.counters.trajectories_collected += 1
    assert replay.counters.trajectories_collected == replay.counters.trajectories_inserted, \
        "HARD STOP conservation: collected != inserted"
    # ---- Phase4A probe: per-episode records + per-update aggregation (directive 2/3; RECORD only) ----
    if PROBE:
        _new_eps = stats.get("episode_records", [])
        probe_completed_episodes.extend(_new_eps)
        with open(probe_episodes_path, "a") as f:
            for _ep in _new_eps:
                f.write(json.dumps(_ep, default=str) + "\n")
        for _ep in _new_eps:
            if probe_first_ge512 is None and int(_ep["length"]) >= PROBE_GE_LEN:
                probe_first_ge512 = dict(
                    first_ge512_update=int(_ep["update_index"]),
                    # Phase4A-v2 (§二): PRECISE resolved env step (authoritative).
                    first_ge512_resolved_env_step=int(_ep["completion_resolved_env_step"]),
                    # DEPRECATED (§二): kept only for historical recomparison.
                    first_ge512_global_step=int(_ep["completion_global_step"]),
                    first_ge512_global_step_deprecated=True,
                    first_ge512_env_id=int(_ep["env_id"]),
                    first_ge512_rollout_step=int(_ep["rollout_step"]),
                    first_ge512_episode_id=int(_ep["episode_id"]),
                    first_ge512_length=int(_ep["length"]))
        _lens = [int(_e["length"]) for _e in probe_completed_episodes]
        _reasons = {}
        for _e in probe_completed_episodes:
            _reasons[_e["done_reason"]] = _reasons.get(_e["done_reason"], 0) + 1
        _la = np.array(_lens, float) if _lens else np.array([0.0])
        probe_upd = dict(update=u, global_step=(u + 1) * STEPS_PER_UPDATE, arm=ARM,
            completed_episode_count_cumulative=len(probe_completed_episodes),
            completed_episode_count_this_update=len(_new_eps),
            pending_episode_count=sum(1 for _ps in pending.slots if len(_ps["obs"]) > 0),
            replay_buffer_trajectory_count=len(replay),
            P50=float(np.percentile(_la, 50)), P75=float(np.percentile(_la, 75)),
            P90=float(np.percentile(_la, 90)), P95=float(np.percentile(_la, 95)),
            P99=float(np.percentile(_la, 99)), max_len=int(max(_lens) if _lens else 0),
            count_ge_129=sum(1 for _L in _lens if _L >= 129),
            count_ge_256=sum(1 for _L in _lens if _L >= 256),
            count_ge_512=sum(1 for _L in _lens if _L >= 512),
            fraction_ge_512=(sum(1 for _L in _lens if _L >= 512) / len(_lens) if _lens else 0.0),
            termination_reason_counts=_reasons,
            first_ge512=probe_first_ge512)
        with open(probe_updates_path, "a") as f:
            f.write(json.dumps(probe_upd, default=str) + "\n")
        print(f"[probe u{u}] eps_cum={len(probe_completed_episodes)} this={len(_new_eps)} "
              f"max_len={probe_upd['max_len']} ge512={probe_upd['count_ge_512']} "
              f"first_ge512={probe_first_ge512}", flush=True)

    # 2. last_value + GAE + PPO main update
    _lg, last_value, _mo, _ht = apply_eval_rmt(
        params, memories, obsv, mem_mask, rmt_state["mem_tokens"])
    advantages, targets = PPO.compute_gae(
        rollout["rewards"], rollout["values"], rollout["dones"], np.asarray(last_value),
        cfg.gamma, cfg.gae_lambda, cfg.value_target_clip_min, cfg.value_target_clip_max)
    params, ppo_opt_state, ppo_metrics = PPO.ppo_update_rmt(
        network, params, ppo_opt_state, ppo_opt, rollout, advantages, targets,
        ppo_cfg, rmt_cfg, args.carry_mode, rng)
    update_count += 1
    online_ppo_update_count += 1
    # Phase4A-v2 (§三): one outer rollout+PPO iteration completed; PPO always commits its
    # policy step -> policy_version advances. (off-path: policy_version stays == legacy
    # update_count, bit-exact.)
    counters.on_outer_update(cfg.num_envs, cfg.num_steps)
    counters.on_ppo_accepted()
    assert ppo_metrics["ppo_finite"], "HARD STOP NaN/Inf in PPO update"
    # ---- A/B training-no-perturbation gate artifacts (CC2 addendum; only with --equiv_dump) ----
    # Deterministic hashes of the rollout + post-update params/optimizer/RMT state. A (probe OFF)
    # and B (probe ON) both emit these; an exact match proves the probe instrumentation does not
    # perturb training. Host-side reads only; no effect on training numerics / RNG stream.
    if args.equiv_dump:
        equiv = dict(update=u, global_step=(u + 1) * STEPS_PER_UPDATE,
                     actions_hash=_arr_hash(rollout["actions"]),
                     rewards_hash=_arr_hash(rollout["rewards"]),
                     dones_hash=_arr_hash(rollout["dones"]),
                     ard_hash=_arr_hash(rollout["actions"], rollout["rewards"], rollout["dones"]),
                     params_sha=_params_sha(params),
                     ppo_opt_sha=_params_sha(ppo_opt_state),
                     rmt_state_sha=_params_sha(rmt_state),
                     memories_sha=_params_sha(memories),
                     mem_mask_sha=_params_sha(mem_mask),
                     mem_idx_sha=_params_sha(mem_idx),
                     ppo_actor=float(ppo_metrics["ppo_actor"]),
                     ppo_entropy=float(ppo_metrics["ppo_entropy"]),
                     ppo_value=float(ppo_metrics.get("ppo_value", 0.0)),
                     online_ppo_update_count=online_ppo_update_count)
        with open(equiv_path, "a") as f:
            f.write(json.dumps(equiv, default=str) + "\n")

    # 3. replay update (Phase4A-v2: mode-dispatched; CC2 directive §四/§五/§七/§八)
    rep = {}
    # ---- Replay eligibility instrumentation (every update, every arm; read-only query) ----
    _rstats = _replay_stats(replay._buffer)
    _pending_traj_count = sum(1 for _ps in pending.slots if len(_ps["obs"]) > 0)
    if _rstats["replay_eligible_count_512"] > 0:
        ever_eligible_512 = True
    did_replay_update = False
    # ---- Phase4A-v2.1 (§五): per-outer-update exposure scratch (recorded after dispatch) ----
    _replay_attempted_this_update = False
    _replay_not_ready_this_update = False
    _eligible_count_this_update = int(_rstats.get("replay_eligible_count_512", 0))
    _replay_batch_size_this_update = 0
    _replay_seq_lengths_this_update = []
    _sample_ids_this_update = []
    _start_offsets_this_update = []
    if REPLAY_ON and REPLAY_MODE == "original_vtrace":
        # ============ ORIGINAL-GOAL V-TRACE ONLY (no relabel, no AWR; firewall §八) ============
        # Eligible-ONLY deterministic sampling (§七): pre-filters length>=SEQUENCE_LENGTH, fixed
        # batch size, explicit NOT_READY (no random-short-then-retry, no silent redraw).
        _batch = replay.sample_eligible(SEQUENCE_LENGTH, replay_sample_rng, K_BATCH)
        _replay_attempted_this_update = True          # §五: sampling was attempted this update
        _eligible_count_this_update = int(_batch.eligible_count)
        if _batch.status == "NOT_READY":
            _replay_not_ready_this_update = True      # §五: attempted but buffer not eligible
            replay_not_ready_skip_count += 1
            print(f"REPLAY_NOT_READY requested_sequence_length={SEQUENCE_LENGTH} "
                  f"max_trajectory_length={_rstats['replay_max_trajectory_length']} "
                  f"eligible_count={_batch.eligible_count}", flush=True)
        else:
            so = _batch.samples
            # §五 exposure scratch: the drawn sample identity / geometry for this update.
            _replay_batch_size_this_update = int(len(so))
            _replay_seq_lengths_this_update = [int(x) for x in _batch.sequence_lengths]
            _sample_ids_this_update = [int(x) for x in _batch.sample_ids]
            _start_offsets_this_update = [int(x) for x in _batch.start_offsets]
            counters.on_replay_attempt(len(so))
            replay_sample_success_count += len(so)
            replay_last_sampled_seq_len = SEQUENCE_LENGTH
            _src0 = replay._get_by_id(_batch.sample_ids[0]) if _batch.sample_ids else None
            replay_last_sampled_traj_len = int(_src0.length) if _src0 is not None else None
            params, target_params, replay_opt_state, m = RL.original_vtrace_update_rmt(
                network, params, target_params, replay_opt_state, replay_opt,
                apply_eval_rmt, scan_fn, so, fp_cfg, rmt_cfg, args.carry_mode)
            update_count += 1
            counters.on_replay_update_executed()
            did_replay_update = True
            assert bool(m["finite"]), "HARD STOP NaN/Inf in original_vtrace replay loss"
            if bool(m.get("policy_committed")):
                assert float(m["policy_kl"]) <= fp_cfg.kl_replay_max + 1e-12, \
                    f"HARD STOP accepted policy_kl={float(m['policy_kl']):.5f} > {fp_cfg.kl_replay_max}"
                accepted_policy_updates += 1
                counters.on_replay_policy_committed()
            if bool(m.get("kl_rejected_update")):
                assert not bool(m.get("policy_committed")), "HARD STOP KL-rejected committed policy"
                kl_rejected_updates += 1
                counters.on_replay_kl_rejected()   # policy_version does NOT advance on rollback
            assert float(m["entropy"]) >= fp_cfg.ent_floor, \
                f"HARD STOP entropy collapse {float(m['entropy']):.4f} < {fp_cfg.ent_floor}"
            rep = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in m.items() if not isinstance(v, (list, dict))}
            rep["batch"] = len(so)
            rep["replay_sample_ids"] = list(_batch.sample_ids)
            rep["replay_start_offsets"] = list(_batch.start_offsets)
            rep["replay_sequence_lengths"] = list(_batch.sequence_lengths)
            replay_update_success_count += 1
            replay_sequences_consumed += len(so)
            if replay_first_success_update is None:
                replay_first_success_update = update_count
            # ---- Hindsight/AWR firewall (§八): original_vtrace MUST keep all four == 0 ----
            counters.assert_hindsight_awr_disabled()
    elif REPLAY_ON and REPLAY_MODE == "full_p2_legacy":
        # ============ LEGACY V-trace+AWR (audit only; GATE 15 requires explicit flag) ============
        # Preserves the original K_BATCH relabel path for audit/legacy comparison. This is the
        # ONLY path that touches Hindsight/AWR; it is default-forbidden for formal science.
        if replay.can_sample():
            _replay_attempted_this_update = True      # §五: sampling was attempted this update
            so, sr = [], []
            if _rstats["replay_eligible_count_512"] == 0:
                _replay_not_ready_this_update = True  # §五: attempted but buffer not eligible
                replay_not_ready_skip_count += 1
                print(f"REPLAY_NOT_READY requested_sequence_length={L_SEQ} "
                      f"max_trajectory_length={_rstats['replay_max_trajectory_length']} "
                      f"eligible_count_512={_rstats['replay_eligible_count_512']}", flush=True)
            else:
                _too_short_prefix = f"sequence_length {L_SEQ} > trajectory length"
                for _ in range(K_BATCH):
                    hindsight_attempts += 1
                    counters.on_hindsight_attempt(1)
                    try:
                        s = replay.sample(sequence_length=L_SEQ)
                    except ValueError as _e:
                        # Legacy benign redraw: a random pick can be < L_SEQ and raise. ONLY the
                        # precise "requested {L_SEQ} > picked trajectory length" case retries; any
                        # other ValueError is re-raised UNCHANGED. (original_vtrace never reaches
                        # this — sample_eligible pre-filters and never draws a short trajectory.)
                        if str(_e).startswith(_too_short_prefix):
                            continue
                        raise
                    replay_sample_success_count += 1
                    replay_last_sampled_seq_len = int(getattr(s, "length", L_SEQ))
                    _src = replay._get_by_id(getattr(s, "source_trajectory_id", None))
                    replay_last_sampled_traj_len = int(_src.length) if _src is not None else None
                    try:
                        rel = RH.relabel_sample_rmt(s, embedding_size=EMB)   # min achieved goal
                    except ValueError:
                        continue                            # not relabelable -> skip
                    hindsight_eligible += 1
                    counters.on_hindsight_eligible(1)
                    counters.on_relabeled_sample(1)
                    so.append(s); sr.append(rel)
            if len(so) >= 2:
                params, target_params, replay_opt_state, m = RL.full_p2_update_rmt(
                    network, params, target_params, replay_opt_state, replay_opt,
                    apply_eval_rmt, scan_fn, so, sr, fp_cfg, rmt_cfg, args.carry_mode, update_count)
                update_count += 1
                counters.on_replay_attempt(len(so))
                counters.on_replay_update_executed()
                counters.on_awr_update(1)
                did_replay_update = True
                assert bool(m["finite"]), "HARD STOP NaN/Inf in replay loss"
                if bool(m.get("policy_committed")):
                    assert float(m["policy_kl"]) <= fp_cfg.kl_replay_max + 1e-12, \
                        f"HARD STOP accepted policy_kl={float(m['policy_kl']):.5f} > {fp_cfg.kl_replay_max}"
                    accepted_policy_updates += 1
                    counters.on_replay_policy_committed()
                if bool(m.get("kl_rejected_update")):
                    assert not bool(m.get("policy_committed")), "HARD STOP KL-rejected committed policy"
                    kl_rejected_updates += 1
                    counters.on_replay_kl_rejected()
                assert float(m["entropy"]) >= fp_cfg.ent_floor, \
                    f"HARD STOP entropy collapse {float(m['entropy']):.4f} < {fp_cfg.ent_floor}"
                rep = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                       for k, v in m.items() if not isinstance(v, (list, dict))}
                rep["batch"] = len(so)
                replay_update_success_count += 1
                replay_sequences_consumed += len(so)
                if replay_first_success_update is None:
                    replay_first_success_update = update_count
                # §五 exposure scratch (legacy path): fixed L_SEQ windows from the relabel path.
                _replay_batch_size_this_update = int(len(so))
                _replay_seq_lengths_this_update = [int(getattr(s, "length", L_SEQ)) for s in so]
                _sample_ids_this_update = [int(getattr(s, "source_trajectory_id", -1)) for s in so]
                _start_offsets_this_update = [int(getattr(s, "start_offset", 0)) for s in so]
    # ---- Phase4A-v2.1 (§五): record this outer update's exposure certificate row ----
    # One row per outer update, in outer-update order. replay_attempt_mask[k] is True iff outer
    # update k attempted a replay sample. sample_ids_by_outer_update / start_offsets_by_outer_update
    # are per-arm INTERNAL provenance (endogenous buffers => never compared cross-arm).
    replay_attempt_mask.append(bool(_replay_attempted_this_update))
    if _replay_attempted_this_update:
        replay_attempt_outer_updates.append(int(u))
    if _replay_not_ready_this_update:
        replay_not_ready_outer_updates.append(int(u))
    if did_replay_update:
        replay_update_outer_updates.append(int(u))
    eligible_count_by_outer_update.append(int(_eligible_count_this_update))
    if did_replay_update:
        replay_batch_sizes.append(int(_replay_batch_size_this_update))
        replay_sequence_lengths.append(list(_replay_seq_lengths_this_update))
        sample_ids_by_outer_update.append(list(_sample_ids_this_update))
        start_offsets_by_outer_update.append(list(_start_offsets_this_update))
    # ---- EMA target tracking on PPO-only outer iterations (no replay update this iteration) ----
    # Identical to legacy off-path (GATE 13): replay_mode=off EMAs every iteration; a replay
    # update performs its own EMA internally, so we skip the PPO-only EMA on those iterations.
    if not did_replay_update:
        target_params = RL.FPL.ema_update(params, target_params, fp_cfg.ema_tau)

    # ---- RMT read-path monitor probe (post-update params + carried state) ----
    mon = _read_path_monitor(params, memories, obsv, mem_mask, rmt_state["mem_tokens"])
    mon["update"] = u
    mon["global_step"] = (u + 1) * STEPS_PER_UPDATE
    mon["arm"] = ARM
    mon["accepted_policy_updates"] = accepted_policy_updates
    # carried RMT memory-token analysis (per-arm carry / boundary-clear verification)
    _carried_tok = np.asarray(rmt_state["mem_tokens"])
    _carried_tok_maxabs = float(np.max(np.abs(_carried_tok)))
    _gtrxl_mem_maxabs = float(np.max(np.abs(np.asarray(memories))))
    mon["carried_rmt_token_maxabs"] = _carried_tok_maxabs
    mon["gtrxl_window_mem_maxabs"] = _gtrxl_mem_maxabs
    if not (bool(np.all(np.isfinite(_carried_tok))) and bool(np.all(np.isfinite(np.asarray(memories))))):
        gtrxl_window_finite_all = False
    if args.carry_mode == "persistent":
        persistent_carry_nonzero_all = persistent_carry_nonzero_all and (_carried_tok_maxabs > 0.0)
    elif args.carry_mode == "reset128":
        reset128_boundary_clear_all = reset128_boundary_clear_all and (_carried_tok_maxabs == 0.0)
    if mon["read_attn_probe_grad"] > 1e-12 or mon["read_ln_probe_grad"] > 1e-12:
        read_ever_nonzero = True
    if mon["mem_on_off_KL"] > 0.0 or mon["mem_on_off_logit_diff"] > 0.0:
        mem_on_off_ever_nonzero = True
    # ---- read-path CONNECTIVITY probe (forced-open gate + injected tokens; both arms) ----
    conn = _read_connectivity_probe(params, memories, obsv, mem_mask,
                                    np.asarray(rmt_state["mem_tokens"]).shape,
                                    jax.random.PRNGKey(args.seed + 1000 + u))
    if (conn["conn_logit_diff"] > 0.0 or conn["conn_KL"] > 0.0 or conn["conn_top_action_frac"] > 0.0
            or conn["conn_read_attn_grad"] > 1e-12 or conn["conn_read_ln_grad"] > 1e-12):
        read_conn_ever_nonzero = True
    mon.update(conn)
    mon["read_conn_ever_nonzero"] = read_conn_ever_nonzero
    mon["read_ever_nonzero"] = read_ever_nonzero
    mon["mem_on_off_ever_nonzero"] = mem_on_off_ever_nonzero
    mon["persistent_carry_nonzero_all"] = persistent_carry_nonzero_all
    mon["reset128_boundary_clear_all"] = reset128_boundary_clear_all
    # ---- replay instrumentation record (per update, per arm) ----
    mon.update(_rstats)
    mon["replay_pending_trajectory_count"] = _pending_traj_count
    mon["replay_not_ready_skip_count"] = replay_not_ready_skip_count
    mon["replay_sample_success_count"] = replay_sample_success_count
    mon["replay_update_success_count"] = replay_update_success_count
    mon["replay_first_success_update"] = replay_first_success_update
    mon["replay_sampled_sequence_length"] = replay_last_sampled_seq_len
    mon["replay_sampled_trajectory_length"] = replay_last_sampled_traj_len
    mon["online_ppo_update_count"] = online_ppo_update_count
    with open(os.path.join(LOG_DIR, "RMT_PHASE4A_MONITOR.jsonl"), "a") as f:
        f.write(json.dumps(mon, default=str) + "\n")
    print(f"[readmon u{u}] gate={mon['gate_value']:.3e} gate_g={mon['gate_probe_grad']:.3e} "
          f"read_attn_g={mon['read_attn_probe_grad']:.3e} read_ln_g={mon['read_ln_probe_grad']:.3e} "
          f"mem_diff={mon['mem_on_off_logit_diff']:.3e} mem_KL={mon['mem_on_off_KL']:.3e} "
          f"carry_tok={_carried_tok_maxabs:.3e} elig512={_rstats['replay_eligible_count_512']} "
          f"rep_upd={replay_update_success_count}", flush=True)

    global_step = (u + 1) * STEPS_PER_UPDATE
    entry = dict(update=u, global_step=global_step, arm=ARM,
                 completed_episodes=stats["completed_episodes"],
                 mean_ep_return=stats["mean_ep_return"], mean_ep_length=stats["mean_ep_length"],
                 replay_size=len(replay), replay_can_sample=replay.can_sample(),
                 update_count=update_count, **ppo_metrics, **rep,
                 t_s=round(time.time() - t_u, 1))
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    print(f"[u{u}] gs={global_step} ppo_actor={ppo_metrics['ppo_actor']:.4f} "
          f"ent={ppo_metrics['ppo_entropy']:.4f} eps={stats['completed_episodes']} "
          f"replay={len(replay)} kl={rep.get('policy_kl','-')} "
          f"({time.time()-t_u:.1f}s)", flush=True)

    # 4. checkpoint at save points
    if (u + 1) % args.save_every == 0 or (u + 1) == args.total_updates:
        save_ckpt(global_step, params, ppo_opt_state, replay_opt_state, target_params, "save")
    # Phase4A probe DEBUG-ONLY early stop (default OFF; NON-COMPARATIVE debugging only). The formal
    # probe MUST keep args.early_stop_len == 0 so both arms run the identical full fixed horizon.
    if PROBE and args.early_stop_len and probe_first_ge512 is not None \
            and int(probe_first_ge512["first_ge512_length"]) >= args.early_stop_len:
        print(f"PROBE_DEBUG_EARLY_STOP fired at u{u} (early_stop_len={args.early_stop_len}); "
              f"NON-COMPARATIVE debug only -- formal probe must NOT stop here.", flush=True)
        break

    # replay noise audit (directive §七)
    if REPLAY_ON:
        audit = dict(global_step=global_step, buffer_episodes=len(replay),
                     stored_transitions=sum(t.length for t in replay._buffer),
                     replay_drawn=replay.counters.replay_samples_drawn,
                     accepted_policy_updates=accepted_policy_updates,
                     kl_rejected_updates=kl_rejected_updates,
                     hindsight_eligible=hindsight_eligible, hindsight_attempts=hindsight_attempts,
                     ratio_max=rep.get("ratio_max"), ess=rep.get("ess"),
                     policy_kl=rep.get("policy_kl"), awr_w_mean=rep.get("awr_w_mean"),
                     awr_kl=rep.get("awr_kl"), entropy=rep.get("entropy"),
                     ep_lengths=[int(t.length) for t in replay._buffer])
        with open(audit_path, "a") as f:
            f.write(json.dumps(audit, default=str) + "\n")

# ---- Phase4A probe final summary (directive 4; RECORD only; NOT formal science) ----
if PROBE:
    _lens = [int(_e["length"]) for _e in probe_completed_episodes]
    _reasons = {}
    for _e in probe_completed_episodes:
        _reasons[_e["done_reason"]] = _reasons.get(_e["done_reason"], 0) + 1
    probe_summary = dict(
        arm=ARM, carry_mode=args.carry_mode, probe="REACHABILITY_ONLY",
        not_for_formal_science=True,
        replay_mode=REPLAY_MODE,
        replay_note="replay learner + hindsight OFF; buffer collection of complete done episodes ON",
        total_updates=args.total_updates,
        total_env_steps=args.total_updates * STEPS_PER_UPDATE,
        online_ppo_update_count=online_ppo_update_count,
        completed_episode_count=len(probe_completed_episodes),
        all_episode_lengths_sorted=sorted(_lens),
        count_ge_512=sum(1 for _L in _lens if _L >= 512),
        fraction_ge_512=(sum(1 for _L in _lens if _L >= 512) / len(_lens) if _lens else 0.0),
        first_ge512=probe_first_ge512,
        termination_reason_counts=_reasons,
        final_params_sha256=_params_sha(params), base_sha256=base_sha,
        step0_params_in="ckpt/0/full_state.pkl",
        early_stop_used=bool(args.early_stop_len and probe_first_ge512 is not None),
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(os.path.join(LOG_DIR, f"{ARM}_probe_summary.json"), "w") as f:
        json.dump(probe_summary, f, indent=2, default=str)
    print("RMT16_PROBE=REACHABILITY_ONLY NOT_FOR_FORMAL_SCIENCE", flush=True)
    print("PROBE_SUMMARY=" + json.dumps(probe_summary, default=str), flush=True)
    sys.exit(0)

# ---- Phase4A per-arm final gates v2 (CC2 directive: reset128 read branch must be CONNECTED) ----
RMT_READ_PATH_ACTIVE = bool(read_ever_nonzero and mem_on_off_ever_nonzero)  # carried-token activity
READ_BRANCH_CONNECTED = bool(read_conn_ever_nonzero)                         # synthetic-token connectivity
_carried_final_maxabs = float(np.max(np.abs(np.asarray(rmt_state["mem_tokens"]))))
REPLAY_HORIZON_REACHED = bool(replay_update_success_count > 0)
# Phase4A-direct-98304 (§四 / §六): the replay horizon is a PASS requirement ONLY for the formal
# run class. engineering_smoke (4096 steps) and long_run_98304 do NOT require a replay update to
# exit PASS — at 4096 steps the 512-anchor horizon is structurally unreachable, so
# replay_update_count==0 / REPLAY_HORIZON_NOT_REACHED must NOT block the run (and must NOT block
# the conditional 98k launch; §五). The CORRECTNESS gates (carry/boundary + read branch + finite
# gtrxl window) stay REQUIRED for every class. When correctness holds but the horizon is not
# reached under a non-formal class, the arm exits PASS with ARM_STATUS=PASS_REPLAY_HORIZON_NOT_REACHED.
REPLAY_HORIZON_REQUIRED_FOR_PASS = bool(args.run_class == "formal_vtrace")
if args.carry_mode == "persistent":
    # Persistent: write->carry->read connected; cross-boundary carry NON-ZERO; read affects output >=1.
    CARRY_NONZERO = bool(persistent_carry_nonzero_all and _carried_final_maxabs > 0.0)
    READ_AFFECTS_OUTPUT = bool(RMT_READ_PATH_ACTIVE or READ_BRANCH_CONNECTED)
    PERSISTENT_READ_DEFECT = bool(CARRY_NONZERO and (not READ_AFFECTS_OUTPUT))
    _PERSISTENT_CORRECTNESS_OK = bool(CARRY_NONZERO and RMT_READ_PATH_ACTIVE
                                      and READ_BRANCH_CONNECTED and gtrxl_window_finite_all)
    ARM_GATES_PASS = bool(_PERSISTENT_CORRECTNESS_OK
                          and (REPLAY_HORIZON_REACHED or not REPLAY_HORIZON_REQUIRED_FOR_PASS))
    if PERSISTENT_READ_DEFECT:
        ARM_STATUS = "RMT_READ_PATH_ENGINEERING_DEFECT"
    elif not REPLAY_HORIZON_REACHED and REPLAY_HORIZON_REQUIRED_FOR_PASS:
        ARM_STATUS = "REPLAY_HORIZON_NOT_REACHED"
    elif ARM_GATES_PASS and REPLAY_HORIZON_REACHED:
        ARM_STATUS = "PASS"
    elif ARM_GATES_PASS:
        ARM_STATUS = "PASS_REPLAY_HORIZON_NOT_REACHED"
    else:
        ARM_STATUS = "FAIL"
else:  # reset128 -- cross-boundary long-term carry read NOT required; read branch MUST be connected.
    # (a) expected control: carried tokens zero (reset works) AND read branch connected (forced-open
    #     gate + injected non-zero tokens produce detectable logit/KL/top-action change / read grad).
    # (b) defect: read branch NEVER connected -> RMT_RESET128_READ_PATH_ENGINEERING_DEFECT.
    BOUNDARY_CLEAR = bool(reset128_boundary_clear_all and _carried_final_maxabs == 0.0)
    RESET128_READ_DEFECT = bool(not READ_BRANCH_CONNECTED)
    _RESET128_CORRECTNESS_OK = bool(BOUNDARY_CLEAR and READ_BRANCH_CONNECTED
                                    and gtrxl_window_finite_all)
    ARM_GATES_PASS = bool(_RESET128_CORRECTNESS_OK
                          and (REPLAY_HORIZON_REACHED or not REPLAY_HORIZON_REQUIRED_FOR_PASS))
    if RESET128_READ_DEFECT:
        ARM_STATUS = "RMT_RESET128_READ_PATH_ENGINEERING_DEFECT"
    elif not BOUNDARY_CLEAR:
        ARM_STATUS = "RESET128_BOUNDARY_CLEAR_FAILED"
    elif not REPLAY_HORIZON_REACHED and REPLAY_HORIZON_REQUIRED_FOR_PASS:
        ARM_STATUS = "REPLAY_HORIZON_NOT_REACHED"
    elif ARM_GATES_PASS and REPLAY_HORIZON_REACHED:
        ARM_STATUS = "PASS"
    elif ARM_GATES_PASS:
        ARM_STATUS = "PASS_REPLAY_HORIZON_NOT_REACHED"
    else:
        ARM_STATUS = "FAIL"
print(f"[gates] arm={ARM} carry_mode={args.carry_mode} STATUS={ARM_STATUS} "
      f"replay_upd={replay_update_success_count} read_active={RMT_READ_PATH_ACTIVE} "
      f"read_conn={READ_BRANCH_CONNECTED} carry_final_maxabs={_carried_final_maxabs:.3e} "
      f"persistent_carry_nonzero_all={persistent_carry_nonzero_all} "
      f"reset128_boundary_clear_all={reset128_boundary_clear_all} "
      f"ever_eligible_512={ever_eligible_512}", flush=True)
# ----------------------- summary -----------------------
# Phase4A-v2 (§八): FINAL Hindsight/AWR firewall. For replay_mode in {off, original_vtrace} the
# four firewall counters MUST be 0 for the entire run (structural non-entry). Any breach here is
# a HARD STOP. full_p2_legacy is the only mode permitted to make them nonzero.
if REPLAY_MODE in ("off", "original_vtrace"):
    counters.assert_hindsight_awr_disabled()
# Phase4A-v2.1 (§五): MATCHED_REPLAY_EXPOSURE readiness. A formal two-arm Carry causal
# conclusion requires the Persistent and Reset128 arms to have consumed IDENTICAL replay
# exposure across the full EXPOSURE_MATCH_FIELDS set (replay_attempt_mask,
# replay_update_outer_updates, replay_update_count, replay_sequences_consumed,
# replay_batch_sizes, replay_sequence_lengths). Each arm records its own exposure_certificate
# below; the cross-arm equality is adjudicated host-side by
# tests/phase4a_v2_exposure_validator.py from the two summaries. This arm reports its
# certificate + the four-way label split (phase4a_v2.replay_labels); it does NOT self-declare
# MATCHED_REPLAY_EXPOSURE=PASS (no two-arm run this round -> NOT_RUN).
summary = dict(arm=ARM, carry_mode=args.carry_mode, replay_mode=REPLAY_MODE,
               # Phase4A-direct-98304 (§一.2): the run management class + its bound identity
               # (NON-scientific; None on formal_vtrace). Recorded so the smoke PASS evaluation
               # (§四) and the conditional 98k launch (§五) can key off the run class directly.
               run_class=args.run_class,
               run_class_identity=RUN_CLASS_IDENTITY,
               replay_horizon_required_for_pass=REPLAY_HORIZON_REQUIRED_FOR_PASS,
               total_updates=args.total_updates, global_step=args.total_updates * STEPS_PER_UPDATE,
               final_params_sha256=_params_sha(params), base_sha256=base_sha,
               accepted_policy_updates=accepted_policy_updates,
               kl_rejected_updates=kl_rejected_updates,
               hindsight_eligible=hindsight_eligible, hindsight_attempts=hindsight_attempts,
               replay_buffer_hash=replay.hash_digest() if REPLAY_ON else None,
               config={k: v for k, v in vars(cfg).items()},
               phase4a_v2=_phase4a_v2_manifest_fields(),
               phase4a_v2_counters=counters.snapshot(),
               replay_update_count=counters.replay_update_count,
               accepted_replay_policy_update_count=counters.accepted_replay_policy_update_count,
               replay_attempt_count=counters.replay_attempt_count,
               replay_sequences_consumed=replay_sequences_consumed,
               policy_version=counters.policy_version,
               outer_update_index=counters.outer_update_index,
               global_env_steps=counters.global_env_steps,
               # Phase4A-v2.1 (§四): the single conflated MATCHED_REPLAY_PROTOCOL_READY flag is
               # REMOVED. The four distinct labels live in phase4a_v2.replay_labels
               # (SAME_REPLAY_PROTOCOL=READY / MATCHED_REPLAY_EXPOSURE=NOT_RUN /
               #  MATCHED_REPLAY_CONTENT=NOT_CLAIMED / ENDOGENOUS_REPLAY_SCREENING=READY_AFTER_SMOKE).
               # Phase4A-v2.1 (§五): the per-arm MATCHED_REPLAY_EXPOSURE certificate — the raw
               # exposure record the two-arm validator (tests/phase4a_v2_exposure_validator.py)
               # compares. One row per outer update, in order. sample_ids_*/start_offsets_* are
               # per-arm INTERNAL provenance ONLY (endogenous buffers: NOT compared cross-arm).
               exposure_certificate=dict(
                   outer_update_count=int(counters.outer_update_index),
                   replay_attempt_mask=[bool(x) for x in replay_attempt_mask],
                   replay_attempt_outer_updates=list(replay_attempt_outer_updates),
                   replay_not_ready_outer_updates=list(replay_not_ready_outer_updates),
                   replay_update_outer_updates=list(replay_update_outer_updates),
                   replay_update_count=int(counters.replay_update_count),
                   accepted_replay_policy_update_count=int(
                       counters.accepted_replay_policy_update_count),
                   kl_rejected_replay_update_count=int(counters.kl_rejected_replay_update_count),
                   replay_sequences_consumed=int(replay_sequences_consumed),
                   replay_batch_sizes=list(replay_batch_sizes),
                   replay_sequence_lengths=[list(x) for x in replay_sequence_lengths],
                   eligible_count_by_outer_update=list(eligible_count_by_outer_update),
                   sample_ids_by_outer_update=[list(x) for x in sample_ids_by_outer_update],
                   start_offsets_by_outer_update=[list(x) for x in start_offsets_by_outer_update]),
               # Phase4A-v2.2 (§三.1): p2_frozen keeps the frozen V-trace/AWR correction
               # constants but MUST NOT carry a numeric policy lag for original_vtrace: the old
               # max_policy_lag=fp_cfg.max_policy_lag (=legacy 16) leaked alongside
               # max_policy_lag=null elsewhere. Now max_policy_lag=null +
               # policy_lag_gate_active=false here; the documentary legacy 16 lives ONLY in the
               # inactive legacy_full_p2_only block below.
               p2_frozen=dict(rho_bar=fp_cfg.rho_bar, c_bar=fp_cfg.c_bar, beta=fp_cfg.beta,
                              w_max=fp_cfg.w_max, w_vtrace=fp_cfg.w_vtrace, w_awr=fp_cfg.w_awr,
                              kl_replay_max=fp_cfg.kl_replay_max, ema_tau=fp_cfg.ema_tau,
                              policy_lag_gate_active=False,
                              max_policy_lag=None),
               # Phase4A-v2.2 (§三.1): explicit ACTIVE replay config + inactive legacy scope.
               active_replay_config=CONTRACT.active_replay_config_manifest(REPLAY_MODE),
               legacy_full_p2_only=CONTRACT.legacy_full_p2_manifest(active=False,
                                                                    max_policy_lag=16),
               # Phase4A-v2.2 (§六.8): the formal-config binding record (SHAs + status + path +
               # frozen base-checkpoint expectation). None when no --formal_config was supplied
               # (replay_mode=off / probe legacy dev compat).
               # Phase4A-v2.3 (§七.3 + §八): the summary pins the FINALIZED certificate's file
               # SHA / sidecar / payload SHA and the executed-protocol SOURCE identity (learner
               # + sampler source SHA via inspect, RNG instance class) — not string labels.
               runtime_config_certificate=(
                   RTC.certificate_shas_record(
                       RUNTIME_CONFIG_CERTIFICATE,
                       certificate_file_sha256=RUNTIME_CONFIG_CERTIFICATE_FILE_SHA256,
                       certificate_sidecar_path=RUNTIME_CONFIG_CERTIFICATE_SIDECAR_PATH)
                   if RUNTIME_CONFIG_CERTIFICATE is not None else None),
               runtime_config_certificate_path=RUNTIME_CONFIG_CERTIFICATE_PATH,
               runtime_config_certificate_sidecar_path=RUNTIME_CONFIG_CERTIFICATE_SIDECAR_PATH,
               runtime_config_certificate_file_sha256=RUNTIME_CONFIG_CERTIFICATE_FILE_SHA256,
               executed_protocol_identity=EXECUTED_PROTOCOL_IDENTITY,
               # Phase4A-v2.4 (§八): BOTH arms' summaries carry the DECLARED protocol
               # definition + the EFFECTIVE protocol definition (declared + executed learner /
               # sampler source identity + executed RNG identity) + its stable SHA256, so the
               # cross-arm validator compares the EFFECTIVE protocol, not declared strings
               # alone. None on non-formal runs (replay_mode=off / probe legacy dev compat).
               declared_protocol_definition=DECLARED_PROTOCOL_DEFINITION,
               effective_protocol_definition=EFFECTIVE_PROTOCOL_DEFINITION,
               effective_protocol_sha256=EFFECTIVE_PROTOCOL_SHA256,
               # Phase4A-v2.4 (§五.3): non-null length-64 SHA pins in the summary (payload SHA
               # of the on-disk certificate artifact + loaded base checkpoint params SHA).
               runtime_config_certificate_payload_sha256=(
                   RUNTIME_CONFIG_CERTIFICATE_PAYLOAD_SHA256),
               base_checkpoint_params_sha256=base_sha,
               status="COMPLETE",
               arm_status=ARM_STATUS,
               arm_gates_pass=bool(ARM_GATES_PASS),
               rmt_read_path_active=RMT_READ_PATH_ACTIVE,
               read_ever_nonzero=read_ever_nonzero,
               mem_on_off_ever_nonzero=mem_on_off_ever_nonzero,
               replay_horizon_reached=REPLAY_HORIZON_REACHED,
               online_ppo_update_count=online_ppo_update_count,
               replay_sample_success_count=replay_sample_success_count,
               replay_update_success_count=replay_update_success_count,
               replay_not_ready_skip_count=replay_not_ready_skip_count,
               replay_first_success_update=replay_first_success_update,
               ever_eligible_512=ever_eligible_512,
               persistent_carry_nonzero_all=persistent_carry_nonzero_all,
               reset128_boundary_clear_all=reset128_boundary_clear_all,
               carried_rmt_token_final_maxabs=_carried_final_maxabs,
               gtrxl_window_finite_all=gtrxl_window_finite_all,
               read_branch_connected=READ_BRANCH_CONNECTED,
               read_conn_ever_nonzero=read_conn_ever_nonzero,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
# Phase4A-v2.2 (§三.3): fail-closed active-policy-lag leak scan over the WHOLE summary before it
# is written to disk. For replay_mode=original_vtrace any numeric max_policy_lag in an active
# block (phase4a_v2 / active_replay_config / p2_frozen / scientific_config.policy_lag / run
# manifest / top level) aborts the run with ORIGINAL_VTRACE_ACTIVE_POLICY_LAG_LEAK; the legacy
# 16 is legal ONLY under legacy_full_p2_only with active=false.
try:
    summary["phase4a_v2_active_policy_lag_leak_scan"] = (
        CONTRACT.assert_no_active_policy_lag_leak(summary))
except ValueError as e:
    raise SystemExit(str(e))
with open(os.path.join(LOG_DIR, f"{ARM}_train_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("\n" + "=" * 78, flush=True)
print(f"{ARM} COMPLETE  final_params_sha={summary['final_params_sha256'][:16]}", flush=True)
print(f"  accepted_policy_updates={accepted_policy_updates} kl_rejected={kl_rejected_updates} "
      f"hindsight={hindsight_eligible}/{hindsight_attempts}", flush=True)
print("=" * 78, flush=True)
print(f"PHASE4A_ARM_FINAL_STATUS={ARM_STATUS}", flush=True)
# Phase4A-v2.2/2.3 (§六.8 + §七): launch-status line for the formal-config binding — now also
# carries the FINALIZED flag and the certificate's own FINAL file SHA + sidecar (§七.3), so the
# launch log alone lets a reviewer re-fetch + re-verify the exact certificate artifact.
if RUNTIME_CONFIG_CERTIFICATE is not None:
    _cert_status = RUNTIME_CONFIG_CERTIFICATE["certificate_status"]
    _cert_finalized = RUNTIME_CONFIG_CERTIFICATE["certificate_finalized"]
    _cert_sci_sha = RUNTIME_CONFIG_CERTIFICATE["scientific_config_sha256"]
    _cert_rt_sha = RUNTIME_CONFIG_CERTIFICATE["runtime_scientific_config_sha256"]
    _cert_bc_status = RUNTIME_CONFIG_CERTIFICATE["checkpoint_identity"][
        "base_checkpoint_expected_sha256_status"]
    _cert_bc_match = RUNTIME_CONFIG_CERTIFICATE["checkpoint_identity"]["base_checkpoint_match"]
    print(f"RUNTIME_CONFIG_CERTIFICATE_STATUS={_cert_status} "
          f"certificate_finalized={_cert_finalized} "
          f"scientific_config_sha256={_cert_sci_sha} "
          f"runtime_scientific_config_sha256={_cert_rt_sha} "
          f"base_checkpoint_expected_sha256_status={_cert_bc_status} "
          f"base_checkpoint_match={_cert_bc_match} "
          f"certificate_payload_sha256={RUNTIME_CONFIG_CERTIFICATE_PAYLOAD_SHA256} "
          f"certificate_file_sha256={RUNTIME_CONFIG_CERTIFICATE_FILE_SHA256} "
          f"base_checkpoint_params_sha256={base_sha} "
          f"certificate_sidecar_path={RUNTIME_CONFIG_CERTIFICATE_SIDECAR_PATH}", flush=True)
else:
    print("RUNTIME_CONFIG_CERTIFICATE_STATUS=NOT_REQUIRED_NO_FORMAL_CONFIG "
          "(replay_mode=off/probe legacy dev compat)", flush=True)
if not ARM_GATES_PASS:
    sys.exit(1)
