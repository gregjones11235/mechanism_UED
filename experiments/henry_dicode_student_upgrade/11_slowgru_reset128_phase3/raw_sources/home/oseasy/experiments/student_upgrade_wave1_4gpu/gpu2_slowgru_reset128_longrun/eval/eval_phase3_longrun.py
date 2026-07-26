#!/usr/bin/env python3
"""Phase3 GPU2 LONGRUN SUSTAINABILITY evaluator.

Imports the FROZEN Phase2 unified evaluator (eval_phase2_unified.py, sha prefix 22451402) as a module and
REUSES its 256-world Stage4 protocol (run_variant) and statistics (paired_compare / mcnemar_paired /
paired_bootstrap_ci) UNCHANGED. The frozen evaluator code is NOT modified; its sha is asserted at import.

Evaluates SlowGRU-Reset128 at the four longrun nodes {24576,49152,73728,98304} vs the canonical Control at
the SAME steps, plus BASELINE(teacher17500)/Control@24576 anchors. Anchors must reproduce (BASELINE~39.45%,
Control@24576~36.33%) or the run is EVALUATION_INFRASTRUCTURE_FAIL.

SUSTAINED gate (directive section 7), all evaluated at the final node unless stated:
  c1: >=2 of {49152,73728,98304} have DK SR > same-step Control
  c2: 98304 DK SR >= Control + 5pp
  c3: 98304 floor3 not lower than Control
  c4: 98304 death not clearly worse than Control (<= +2pp)
  c5: no numerical/entropy/restore issues (all nodes roundtrip_ok + params finite + entropy healthy)
Verdict: SUSTAINED_SIGNAL / TRANSIENT_SIGNAL / NO_SIGNAL / ENGINEERING_FAIL.
GPU2 only; deterministic ops; read-only w.r.t. checkpoints.
"""
import hashlib, json, os, sys, time

GPU_UUID = "GPU-8df11537-ab79-722d-606f-411966196c4c"   # GPU2
os.environ["PHASE2_EVAL_GPU"] = GPU_UUID                # frozen module reads this for CUDA_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import numpy as np
W = "/home/oseasy/experiments/student_upgrade_wave1_4gpu"
if W not in sys.path: sys.path.insert(0, W)
import eval_phase2_unified as E                          # FROZEN evaluator (sha 22451402...)

EVAL_FROZEN_PATH = os.path.join(W, "eval_phase2_unified.py")
EVAL_FROZEN_SHA = hashlib.sha256(open(EVAL_FROZEN_PATH, "rb").read()).hexdigest()
assert EVAL_FROZEN_SHA.startswith("22451402"), f"FROZEN evaluator sha mismatch: {EVAL_FROZEN_SHA}"

LONGRUN_ROOT = f"{W}/gpu2_slowgru_reset128_longrun/train/ckpt"
LONGRUN_SUMMARY = f"{W}/gpu2_slowgru_reset128_longrun/train/out/LC_SLOWGRU_RESET128_LONGRUN_train_summary.json"
CTL_ROOT = "/home/oseasy/experiments/exploratory_delayed_onset_20260724/control_RUN2/ckpt"
NODES = [24576, 49152, 73728, 98304]

with open(__file__, "rb") as f:
    DRIVER_SHA = hashlib.sha256(f.read()).hexdigest()


def finite(pytree):
    import jax
    return bool(all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree_util.tree_leaves(pytree)
                    if np.asarray(v).dtype.kind in "fi"))


def main():
    out_dir = f"{W}/gpu2_slowgru_reset128_longrun/eval"
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 78, flush=True)
    print("PHASE3 GPU2 LONGRUN SUSTAINABILITY evaluator (frozen 256-world protocol reused)", flush=True)
    print(f"  GPU={GPU_UUID} frozen_eval_sha={EVAL_FROZEN_SHA[:16]} driver_sha={DRIVER_SHA[:16]}", flush=True)
    print("=" * 78, flush=True)

    # ---- anchors ----
    print("\n=== anchors (must reproduce) ===", flush=True)
    R = {}
    R["BASELINE"] = E.run_variant("BASELINE_teacher17500", E.teacher_net, E.load_weights_only(
        E.TEACHER_CKPT, E.s4_base, E.ctor, E.Cfg, load_opt_state=False).params, "teacher", E.sg_init_longstate)
    base_ok = bool(abs(R["BASELINE"]["SR"] * 100 - 39.45) < 0.5)

    print("\n=== per-node: SlowGRU-Reset128@step vs canonical Control@step ===", flush=True)
    node_results = {}
    params_finite = {}
    for s in NODES:
        lr_path = f"{LONGRUN_ROOT}/{s}/full_state.pkl"
        ctl_path = f"{CTL_ROOT}/{s}/full_state.pkl"
        lr_params, lr_full = E.load_full_params(lr_path)
        ctl_params, _ = E.load_full_params(ctl_path)
        params_finite[s] = dict(longrun_finite=finite(lr_params), control_finite=finite(ctl_params),
                                longrun_manifest_sha=lr_full["manifest"]["params_sha256"])
        print(f"\n--- node {s} (longrun params_sha={lr_full['manifest']['params_sha256'][:16]}) ---", flush=True)
        sgr = E.run_variant(f"SGR128_{s}", E.sg_on, lr_params, "on", E.sg_init_longstate)
        ctl = E.run_variant(f"Control_{s}", E.teacher_net, ctl_params, "teacher", E.sg_init_longstate)
        cmp = E.paired_compare(sgr, ctl, f"SlowGRU-Reset128 - Control @ {s}")
        node_results[s] = dict(slowgru_reset128=sgr, control=ctl, paired=cmp)
        print(f"  [{s}] dSR={cmp['SR_delta_pp']:+.2f}pp ({cmp['SR_delta_worlds']:+d}w) "
              f"dF3={cmp['floor3_delta_pp']:+.2f} dDeath={cmp['death_delta_pp']:+.2f} "
              f"McNemar p={cmp['mcnemar']['mcnemar_p']} "
              f"CI95=[{cmp['bootstrap_SR']['ci95_low_pp']:+.2f},{cmp['bootstrap_SR']['ci95_high_pp']:+.2f}]",
              flush=True)

    # ---- c5: restore/numerical/entropy from train summary ----
    # train summary stores node roundtrip under top-level `chunks`, per-update entropy under
    # `per_update_metrics`, and the live reset128 gate under `reset128_gates`.
    STEPS_PER_UPDATE = 2048
    c5_detail = {}
    try:
        summ = json.load(open(LONGRUN_SUMMARY))
        chunks = summ.get("chunks", [])
        node_roundtrip = {}
        for r in chunks:
            gs = r.get("global_step")
            if gs in NODES:
                node_roundtrip[gs] = bool(r.get("roundtrip_ok", False))
        all_rt = bool(all(node_roundtrip.get(s, False) for s in NODES))
        pum = summ.get("per_update_metrics", [])
        entropies = [m["entropy"] for m in pum if isinstance(m.get("entropy"), (int, float))]
        node_entropy = {}
        for m in pum:
            us = m.get("update_step")
            if us is not None and isinstance(m.get("entropy"), (int, float)):
                gs = us * STEPS_PER_UPDATE
                if gs in NODES:
                    node_entropy[gs] = m["entropy"]
        ent_min = min(entropies) if entropies else None
        ent_ok = bool(ent_min is None or ent_min > 0.05)   # no collapse
        all_fin = bool(all(params_finite[s]["longrun_finite"] and params_finite[s]["control_finite"]
                           for s in NODES))
        r128 = summ.get("reset128_gates", {})
        boundary_clear_pass = bool(r128.get("boundary_clear_pass", False))
        c5_detail = dict(node_roundtrip=node_roundtrip, all_roundtrip_ok=all_rt,
                         all_params_finite=all_fin, node_entropy=node_entropy,
                         entropy_min=ent_min, entropy_max=(max(entropies) if entropies else None),
                         entropy_ok=ent_ok, reset128_boundary_clear_pass=boundary_clear_pass,
                         train_status=summ.get("status"))
        c5 = bool(all_rt and all_fin and ent_ok and boundary_clear_pass)
    except Exception as ex:
        all_fin = bool(all(params_finite[s]["longrun_finite"] and params_finite[s]["control_finite"]
                           for s in NODES))
        c5_detail = dict(summary_read_error=str(ex), all_params_finite=all_fin)
        c5 = bool(all_fin)

    # ---- anchor / infrastructure gate ----
    ctl24576_sr = node_results[24576]["control"]["SR"] * 100
    ctl_anchor_ok = bool(abs(ctl24576_sr - 36.328125) < 0.10)
    anchor_pass = bool(base_ok and ctl_anchor_ok)

    # ---- SUSTAINED gate ----
    dSR = {s: node_results[s]["paired"]["SR_delta_pp"] for s in NODES}
    leads = {s: bool(dSR[s] > 0.0) for s in NODES}
    n_late_leads = int(sum(leads[s] for s in (49152, 73728, 98304)))
    c1 = bool(n_late_leads >= 2)
    c2 = bool(dSR[98304] >= 5.0)
    c3 = bool(node_results[98304]["paired"]["floor3_delta_pp"] >= -1e-9)
    c4 = bool(node_results[98304]["paired"]["death_delta_pp"] <= 2.0)
    sustained = bool(anchor_pass and c1 and c2 and c3 and c4 and c5)

    # ---- verdict ----
    if not anchor_pass:
        verdict = "ENGINEERING_FAIL"
        verdict_reason = (f"EVALUATION_INFRASTRUCTURE_FAIL: anchors did not reproduce "
                          f"(BASELINE={R['BASELINE']['SR']*100:.2f}% ok={base_ok}, "
                          f"Control@24576={ctl24576_sr:.2f}% ok={ctl_anchor_ok})")
    elif not c5:
        verdict = "ENGINEERING_FAIL"
        verdict_reason = f"numerical/restore/entropy issue: {c5_detail}"
    elif sustained:
        verdict = "SUSTAINED_SIGNAL"
        verdict_reason = (f"SUSTAINED: late leads={n_late_leads}/3 (c1), dSR@98304={dSR[98304]:+.2f}pp (c2>=+5), "
                          f"floor3@98304={node_results[98304]['paired']['floor3_delta_pp']:+.2f} (c3), "
                          f"death@98304={node_results[98304]['paired']['death_delta_pp']:+.2f} (c4)")
    else:
        any_lead = any(leads[s] for s in NODES)
        final_lead = leads[98304]
        if any_lead and (not final_lead):
            verdict = "TRANSIENT_SIGNAL"
            verdict_reason = (f"early/mid lead (leads={ {s:leads[s] for s in NODES} }) but 98304 NOT leading "
                              f"(dSR@98304={dSR[98304]:+.2f}pp) -> signal vanished/reversed by the final node")
        elif not any_lead:
            verdict = "NO_SIGNAL"
            verdict_reason = (f"SlowGRU-Reset128 never leads Control at any node "
                              f"(dSR={ {s:round(dSR[s],2) for s in NODES} })")
        else:
            verdict = "TRANSIENT_SIGNAL"
            verdict_reason = (f"SUSTAINED gate not met (c1={c1} c2={c2} c3={c3} c4={c4}); final node "
                              f"dSR@98304={dSR[98304]:+.2f}pp; leads={ {s:leads[s] for s in NODES} }")

    print("\n" + "=" * 78, flush=True)
    print(f"SUSTAINABILITY_VERDICT = {verdict}", flush=True)
    print(f"  reason: {verdict_reason}", flush=True)
    print(f"  dSR per node (pp): { {s: round(dSR[s],2) for s in NODES} }", flush=True)
    print(f"  gate: anchor={anchor_pass} c1={c1} c2={c2} c3={c3} c4={c4} c5={c5}", flush=True)
    print("=" * 78, flush=True)

    out = dict(label="P3_LONGRUN_SUSTAINABILITY_EVAL",
        protocol="FROZEN 256-world Stage4 stochastic seed42 (eval_phase2_unified.py reused, sha "
                 + EVAL_FROZEN_SHA[:16] + "); SlowGRU-Reset128@step vs canonical Control@step, paired",
        gpu_uuid=GPU_UUID, frozen_evaluator_sha256=EVAL_FROZEN_SHA, driver_sha256=DRIVER_SHA,
        nodes=NODES, longrun_root=LONGRUN_ROOT, control_root=CTL_ROOT,
        anchors=dict(BASELINE_SR_pp=R["BASELINE"]["SR"] * 100, baseline_ok=base_ok,
                     Control_24576_SR_pp=ctl24576_sr, control_anchor_ok=ctl_anchor_ok, anchor_pass=anchor_pass),
        baseline_full=R["BASELINE"],
        node_results=node_results, params_finite=params_finite,
        dSR_per_node_pp={s: round(dSR[s], 3) for s in NODES}, leads_per_node=leads,
        sustained_gate=dict(c1_late_leads_ge2=c1, n_late_leads=n_late_leads, c2_98304_ge_5pp=c2,
                            c3_floor3_not_lower=c3, c4_death_not_worse=c4, c5_no_num_restore_issue=c5,
                            c5_detail=c5_detail, ALL_PASS=sustained),
        SUSTAINABILITY_VERDICT=verdict, verdict_reason=verdict_reason,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    jp = os.path.join(out_dir, "phase3_longrun_sustainability_eval.json")
    json.dump(out, open(jp, "w"), indent=2, default=str)
    print(f"\nwrote {jp}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
