#!/usr/bin/env python3
"""Phase3 GPU3 ATTRIBUTION evaluator (directive section 10 + GAIN_SOURCE section 11).

Imports the FROZEN Phase2 unified evaluator (eval_phase2_unified.py, sha prefix 22451402) as a module and
REUSES its 256-world Stage4 protocol (run_variant) and statistics (paired_compare / mcnemar_paired /
paired_bootstrap_ci) UNCHANGED. The frozen code is NOT modified; its sha is asserted at import.

Four arms, all @24576, all paired by world index (seed42, 256 worlds, stochastic, max 4096 steps, Stage4):
  (1) Control        ActorCriticTransformer  canonical control_RUN2/ckpt/24576
  (2) Full           ActorCriticSlowGRU       Phase2 SlowGRU-Reset128@24576 (sha 2ffdd269...)
  (3) Detach         ActorCriticSlowGRUDetach gpu3 train_detach/ckpt/24576 (stop_gradient slow->backbone)
  (4) MatchedMLP     ActorCriticSlowGRUMLP    gpu3 train_mlp/ckpt/24576 (non-recurrent matched MLP)

Effect decomposition (paired, + = first arm better):
  BACKBONE_GRADIENT_SHAPING_EFFECT  = Full - Detach
  WITHIN_ROLLOUT_RECURRENCE_EFFECT  = Full - MatchedMLP
  CAPACITY_REGULARIZATION_EFFECT    = MatchedMLP - Control

GAIN_SOURCE in {BACKBONE_GRADIENT_SHAPING, WITHIN_ROLLOUT_RECURRENCE,
                CAPACITY_OR_RESIDUAL_REGULARIZATION, MIXED, UNRESOLVED, ENGINEERING_FAIL}.
GPU3 only; deterministic ops; read-only w.r.t. checkpoints.
"""
import hashlib, json, os, sys, time

GPU_UUID = "GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd"   # GPU3
os.environ["PHASE2_EVAL_GPU"] = GPU_UUID                # frozen module reads this for CUDA_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import numpy as np
W = "/home/oseasy/experiments/student_upgrade_wave1_4gpu"
ATTR_SRC = f"{W}/gpu3_slowgru_reset128_attribution/src"
for p in (W, ATTR_SRC):
    if p not in sys.path: sys.path.insert(0, p)
import eval_phase2_unified as E                          # FROZEN evaluator (sha 22451402...)
from slowgru_detach_network import ActorCriticSlowGRUDetach, init_longstate as det_init_longstate
from slowgru_mlp_network import ActorCriticSlowGRUMLP, init_longstate as mlp_init_longstate

EVAL_FROZEN_PATH = os.path.join(W, "eval_phase2_unified.py")
EVAL_FROZEN_SHA = hashlib.sha256(open(EVAL_FROZEN_PATH, "rb").read()).hexdigest()
assert EVAL_FROZEN_SHA.startswith("22451402"), f"FROZEN evaluator sha mismatch: {EVAL_FROZEN_SHA}"

CTL_24576 = "/home/oseasy/experiments/exploratory_delayed_onset_20260724/control_RUN2/ckpt/24576/full_state.pkl"
FULL_24576 = f"{W}/gpu2_lc_slowgru_reset128/train_24576/ckpt/24576/full_state.pkl"
DETACH_24576 = f"{W}/gpu3_slowgru_reset128_attribution/train_detach/ckpt/24576/full_state.pkl"
MLP_24576 = f"{W}/gpu3_slowgru_reset128_attribution/train_mlp/ckpt/24576/full_state.pkl"
KNOWN_FULL_SHA16 = "2ffdd269b94e1e6b"
KNOWN_TEACHER_SHA16 = "d4e85af58b7f87d6"

with open(__file__, "rb") as f:
    DRIVER_SHA = hashlib.sha256(f.read()).hexdigest()

# Detach/MLP networks share the EXACT frozen constructor signature -> reuse E._net_kw / E.ACTION_DIM
det_on = ActorCriticSlowGRUDetach(**E._net_kw, use_longmem=True)
mlp_on = ActorCriticSlowGRUMLP(**E._net_kw, use_longmem=True)


def finite(pytree):
    import jax
    return bool(all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree_util.tree_leaves(pytree)
                    if np.asarray(v).dtype.kind in "fi"))


def clearly_better(cmp, min_pp=2.0, p_thresh=0.10):
    """'明显优于' = magnitude >= min_pp AND statistical support (McNemar p<thresh OR paired CI lower > 0)."""
    mag = bool(cmp["SR_delta_pp"] >= min_pp)
    sig = bool(cmp["mcnemar"]["mcnemar_p"] < p_thresh or cmp["bootstrap_SR"]["ci95_low_pp"] > 0.0)
    return bool(mag and sig)


def main():
    out_dir = f"{W}/gpu3_slowgru_reset128_attribution/eval"
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 78, flush=True)
    print("PHASE3 GPU3 ATTRIBUTION evaluator (frozen 256-world protocol reused)", flush=True)
    print(f"  GPU={GPU_UUID} frozen_eval_sha={EVAL_FROZEN_SHA[:16]} driver_sha={DRIVER_SHA[:16]}", flush=True)
    print("=" * 78, flush=True)

    # ---- load all params + integrity ----
    ctl_params, ctl_full = E.load_full_params(CTL_24576)
    full_params, full_full = E.load_full_params(FULL_24576)
    det_params, det_full = E.load_full_params(DETACH_24576)
    mlp_params, mlp_full = E.load_full_params(MLP_24576)
    full_sha = full_full["manifest"]["params_sha256"]
    full_sha_ok = bool(full_sha.startswith(KNOWN_FULL_SHA16))
    fin = dict(control=finite(ctl_params), full=finite(full_params),
               detach=finite(det_params), mlp=finite(mlp_params))
    print(f"[load] Full@24576 sha={full_sha[:16]} (expect {KNOWN_FULL_SHA16}) ok={full_sha_ok}", flush=True)
    print(f"[load] Detach sha={det_full['manifest']['params_sha256'][:16]} "
          f"MLP sha={mlp_full['manifest']['params_sha256'][:16]} finite={fin}", flush=True)

    R = {}
    print("\n=== four arms @24576 (paired, 256 worlds) ===", flush=True)
    R["Control"] = E.run_variant("Control_24576", E.teacher_net, ctl_params, "teacher", E.sg_init_longstate)
    R["Full"] = E.run_variant("Full_Reset128_24576", E.sg_on, full_params, "on", E.sg_init_longstate)
    R["Detach"] = E.run_variant("Detach_24576", det_on, det_params, "on", det_init_longstate)
    R["MatchedMLP"] = E.run_variant("MatchedMLP_24576", mlp_on, mlp_params, "on", mlp_init_longstate)

    # ---- effect decomposition (paired) ----
    print("\n=== effect decomposition (paired, + = first arm better) ===", flush=True)
    eff = {}
    eff["BACKBONE_GRADIENT_SHAPING_EFFECT"] = E.paired_compare(R["Full"], R["Detach"],
        "Full - Detach (backbone gradient shaping)")
    eff["WITHIN_ROLLOUT_RECURRENCE_EFFECT"] = E.paired_compare(R["Full"], R["MatchedMLP"],
        "Full - MatchedMLP (within-rollout recurrence)")
    eff["CAPACITY_REGULARIZATION_EFFECT"] = E.paired_compare(R["MatchedMLP"], R["Control"],
        "MatchedMLP - Control (capacity / residual regularization)")
    eff["FULL_TOTAL_EFFECT"] = E.paired_compare(R["Full"], R["Control"],
        "Full - Control (total SlowGRU-Reset128 effect, sanity)")
    for k, v in eff.items():
        print(f"  [{k}] dSR={v['SR_delta_pp']:+.2f}pp ({v['SR_delta_worlds']:+d}w) "
              f"dF3={v['floor3_delta_pp']:+.2f} dDeath={v['death_delta_pp']:+.2f} "
              f"McNemar p={v['mcnemar']['mcnemar_p']} "
              f"CI95=[{v['bootstrap_SR']['ci95_low_pp']:+.2f},{v['bootstrap_SR']['ci95_high_pp']:+.2f}]",
              flush=True)

    # ---- GAIN_SOURCE decision ----
    backbone = clearly_better(eff["BACKBONE_GRADIENT_SHAPING_EFFECT"])
    recurrence = clearly_better(eff["WITHIN_ROLLOUT_RECURRENCE_EFFECT"])
    # capacity: MatchedMLP clearly > Control AND MatchedMLP close to Full (|Full-MLP| < 2pp)
    cap_clear = clearly_better(eff["CAPACITY_REGULARIZATION_EFFECT"])
    mlp_close_to_full = bool(abs(eff["WITHIN_ROLLOUT_RECURRENCE_EFFECT"]["SR_delta_pp"]) < 2.0)
    capacity = bool(cap_clear and mlp_close_to_full)
    n_effects = int(backbone) + int(recurrence) + int(capacity)

    # ---- infrastructure gate ----
    ctl_sr = R["Control"]["SR"] * 100
    ctl_anchor_ok = bool(abs(ctl_sr - 36.328125) < 0.10)
    all_finite = bool(all(fin.values()))
    infra_ok = bool(ctl_anchor_ok and full_sha_ok and all_finite)

    if not infra_ok:
        gain = "ENGINEERING_FAIL"
        reason = (f"infra gate failed: control_anchor_ok={ctl_anchor_ok} (Control@24576={ctl_sr:.2f}%), "
                  f"full_sha_ok={full_sha_ok} (sha={full_sha[:16]}), all_finite={all_finite} {fin}")
    elif n_effects >= 2:
        gain = "MIXED"
        reason = (f"multiple effects hold: backbone={backbone} recurrence={recurrence} capacity={capacity} "
                  f"(Full-Detach={eff['BACKBONE_GRADIENT_SHAPING_EFFECT']['SR_delta_pp']:+.2f}pp, "
                  f"Full-MLP={eff['WITHIN_ROLLOUT_RECURRENCE_EFFECT']['SR_delta_pp']:+.2f}pp, "
                  f"MLP-Control={eff['CAPACITY_REGULARIZATION_EFFECT']['SR_delta_pp']:+.2f}pp)")
    elif backbone:
        gain = "BACKBONE_GRADIENT_SHAPING"
        reason = (f"Full clearly > Detach (dSR={eff['BACKBONE_GRADIENT_SHAPING_EFFECT']['SR_delta_pp']:+.2f}pp, "
                  f"p={eff['BACKBONE_GRADIENT_SHAPING_EFFECT']['mcnemar']['mcnemar_p']}) while "
                  f"recurrence={recurrence} capacity={capacity} -> gain from slow-branch gradient shaping "
                  f"of the shared CNN/GTrXL")
    elif recurrence:
        gain = "WITHIN_ROLLOUT_RECURRENCE"
        reason = (f"Full clearly > MatchedMLP (dSR={eff['WITHIN_ROLLOUT_RECURRENCE_EFFECT']['SR_delta_pp']:+.2f}pp, "
                  f"p={eff['WITHIN_ROLLOUT_RECURRENCE_EFFECT']['mcnemar']['mcnemar_p']}) while "
                  f"backbone={backbone} capacity={capacity} -> gain from within-rollout recurrence")
    elif capacity:
        gain = "CAPACITY_OR_RESIDUAL_REGULARIZATION"
        reason = (f"MatchedMLP clearly > Control (dSR={eff['CAPACITY_REGULARIZATION_EFFECT']['SR_delta_pp']:+.2f}pp) "
                  f"and MatchedMLP close to Full (|Full-MLP|="
                  f"{abs(eff['WITHIN_ROLLOUT_RECURRENCE_EFFECT']['SR_delta_pp']):.2f}pp<2) while "
                  f"backbone={backbone} recurrence={recurrence} -> gain from extra params/depth/residual branch")
    else:
        gain = "UNRESOLVED"
        reason = (f"no effect clearly holds: backbone={backbone} recurrence={recurrence} capacity={capacity} "
                  f"(Full-Detach={eff['BACKBONE_GRADIENT_SHAPING_EFFECT']['SR_delta_pp']:+.2f}pp, "
                  f"Full-MLP={eff['WITHIN_ROLLOUT_RECURRENCE_EFFECT']['SR_delta_pp']:+.2f}pp, "
                  f"MLP-Control={eff['CAPACITY_REGULARIZATION_EFFECT']['SR_delta_pp']:+.2f}pp); "
                  f"differences small or statistically unstable")

    print("\n" + "=" * 78, flush=True)
    print(f"GAIN_SOURCE = {gain}", flush=True)
    print(f"  reason: {reason}", flush=True)
    print(f"  effects: backbone={backbone} recurrence={recurrence} capacity={capacity} (n={n_effects})", flush=True)
    print("=" * 78, flush=True)

    out = dict(label="P3_ATTRIBUTION_EVAL",
        protocol="FROZEN 256-world Stage4 stochastic seed42 (eval_phase2_unified.py reused, sha "
                 + EVAL_FROZEN_SHA[:16] + "); 4 arms @24576 paired by world index",
        gpu_uuid=GPU_UUID, frozen_evaluator_sha256=EVAL_FROZEN_SHA, driver_sha256=DRIVER_SHA,
        arms=dict(Control=CTL_24576, Full=FULL_24576, Detach=DETACH_24576, MatchedMLP=MLP_24576),
        integrity=dict(full_sha256=full_sha, full_sha_ok=full_sha_ok, known_full_sha16=KNOWN_FULL_SHA16,
                       detach_sha256=det_full["manifest"]["params_sha256"],
                       mlp_sha256=mlp_full["manifest"]["params_sha256"],
                       control_sha256=ctl_full["manifest"]["params_sha256"], finite=fin),
        infra_gate=dict(control_anchor_ok=ctl_anchor_ok, control_24576_SR_pp=ctl_sr,
                        full_sha_ok=full_sha_ok, all_finite=all_finite, infra_ok=infra_ok),
        results=R, effects=eff,
        effect_flags=dict(backbone_gradient_shaping=backbone, within_rollout_recurrence=recurrence,
                          capacity_or_residual_regularization=capacity, n_effects=n_effects,
                          clearly_better_threshold="SR_delta>=2pp AND (McNemar p<0.10 OR CI_low>0)"),
        GAIN_SOURCE=gain, gain_source_reason=reason,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    jp = os.path.join(out_dir, "phase3_attribution_eval.json")
    json.dump(out, open(jp, "w"), indent=2, default=str)
    print(f"\nwrote {jp}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
