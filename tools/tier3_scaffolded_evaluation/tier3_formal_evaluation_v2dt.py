#!/usr/bin/env python3
"""CC4 Tier3 — FORMAL global performance evaluation DRIVER (V2_DYNAMIC_TOPOLOGY).

Authorized by the independent secondary audit (verdict
PASS_FOR_FORMAL_GLOBAL_PERFORMANCE_EVALUATION_START). This is the formal
counterpart of the V2DT binding smoke driver (tier3_projection_binding_smoke_-
v2.py): the SAME stages 0–6 gate flow over the SAME frozen primitives, with
stage 7 swapped from the 2-episode/32-step smoke schedule to the FROZEN formal
performance schedule at the full horizon:

  * FULL  = 64 held-out canonical-reset seeds 200000..200063,
  * FRONT = all 8 frozen FRONT bank states, each exactly once,
  * BACK  = all 8 frozen BACK bank states, each exactly once,
  * max_steps = 4096, action_mode = greedy_argmax (frozen contract),

through the V2 engine (tier3_evaluator_v2; BFS over the CURRENT environment
topology — legal mining is inside the metric domain).

Start authorization is gated on the SECONDARY_AUDIT_PASS marker (written by
tier3_formal_start_marker_v2dt.py BEFORE any formal run): a formal run
fail-closes if the marker is absent, its SHA sidecar mismatches, or its
verdict is not the verbatim audit verdict. COMMON_EVALUATOR_V2_READY.json's
FORMAL_RANKING_STARTED flag is flipped to true ONLY at closing time by the
ranking tool (tier3_formal_ranking_v2dt.py) — a single READY writer, no race;
while runs execute, the marker + per-candidate certificates ARE the start
record. After closing, this driver fail-closes (the frozen V2 verifier
requires FORMAL_RANKING_STARTED=false), so formal runs cannot silently repeat
once the ranking is published.

Teacher (BASELINE_TEACHER_CKPT17500) runs the IDENTICAL formal schedule for
reference comparability, certified reference_only=true and EXCLUDED from the
student ranking (counts_toward_student_binding_count=false).

Interface smoke evidence is NOT a performance result; this driver's outputs
ARE formal performance evidence (run_class=FORMAL_EVALUATION), but
scientific_claim_authorized stays false forever this round (single training
seed) and scaffolded results can never replace the full task.

FailClosed discipline is identical to the binding driver: predicate FailClosed
(corruption class only under V2) → structured formal_abort, candidate BLOCKED,
never relaxed, never faked; every other exception crashes fail-closed.

Usage (server, locked CC4 venv, GPU2 or GPU3, CWD = repo root):
  CUDA_VISIBLE_DEVICES=GPU-8df11537-ab79-722d-606f-411966196c4c \
  python tools/tier3_scaffolded_evaluation/tier3_formal_evaluation_v2dt.py \
      --candidate-id BASE_GTRXL_ORIGINAL_VTRACE_98304 \
      [--common-dir /home/oseasy/student_pool_v1/common_v2] \
      [--v1-common-dir /home/oseasy/student_pool_v1/common] \
      [--frozen-bank-artifacts /home/oseasy/student_pool_v1/common/frozen_bank_artifacts] \
      [--out-root /home/oseasy/student_pool_v1/cc4]

Rehearsal (bounded plumbing check, NOT a formal result; scratch dir required):
  ... --rehearsal-scratch <pool>/cc4/_rehearsal_<UTC> \
      --limit-full 2 --limit-front 2 --limit-back 2
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import tier3_projection_runtime as proj                            # noqa: E402
import tier3_projection_binding_smoke_v2 as smokev2                # noqa: E402
import tier3_evaluation_certificate_v2dt as certmod                # noqa: E402

RESULT_SCHEMA = "mechanism_UED.tier3_formal_evaluation_result/v2dt"
READY_FORMAL_SCHEMA = "mechanism_UED.tier3_formal_ready/v2dt"
PROVENANCE_SCHEMA = "mechanism_UED.tier3_formal_provenance/v2dt"
COMMON_EVALUATOR_PROTOCOL_VERSION = certmod.COMMON_EVALUATOR_PROTOCOL_VERSION
RUN_CLASS = certmod.RUN_CLASS                       # "FORMAL_EVALUATION"
FORMAL_SCENARIO_ORDER = certmod.FORMAL_SCENARIO_ORDER
SECONDARY_AUDIT_VERDICT = certmod.SECONDARY_AUDIT_VERDICT
SECONDARY_AUDIT_MARKER_NAME = "SECONDARY_AUDIT_PASS.json"

# Frozen closing-evidence pins (committed, b736c8c evidence bundle).
POOL_BINDING_GATE_V2DT_SHA256 = \
    "cec167117a7aa8e67a3d5eb60839e711e72d950135553e4035a87e6c9859a352"


# ---------------------------------------------------------------------------
# pure gates (unit-testable without JAX)
# ---------------------------------------------------------------------------
def check_rehearsal_args(limit_full, limit_front, limit_back, rehearsal_scratch):
    limits = {"full": limit_full, "front_l2": limit_front, "back_l2": limit_back}
    any_limit = any(v is not None for v in limits.values())
    if not any_limit and not rehearsal_scratch:
        return False, None                      # formal mode
    proj.require(rehearsal_scratch,
                 "FAIL CLOSED: --limit-* requires --rehearsal-scratch <DIR> "
                 "(bounded rehearsal runs never write formal output dirs)")
    formal_counts = {"full": proj.FROZEN_FULL_EPISODE_COUNT,
                     "front_l2": proj.FROZEN_FRONT_EPISODE_COUNT,
                     "back_l2": proj.FROZEN_BACK_EPISODE_COUNT}
    resolved = {}
    for sc, v in limits.items():
        n = formal_counts[sc] if v is None else int(v)
        proj.require(1 <= n <= formal_counts[sc],
                     "FAIL CLOSED: rehearsal limit %s=%s outside [1, %d]"
                     % (sc, v, formal_counts[sc]))
        resolved[sc] = n
    return True, resolved


def verify_formal_start(common_dir, pool_cc4_dir):
    """The formal start-authorization gate (stage 1b). Returns the marker
    reference recorded into every formal certificate. Fail-closed on ANY
    deviation; stronger than a boolean flag: the marker file must exist, its
    SHA sidecar must match, and its verdict must be the verbatim audit
    verdict."""
    ready_path = os.path.join(common_dir, "COMMON_EVALUATOR_V2_READY.json")
    proj.require(os.path.isfile(ready_path),
                 "FAIL CLOSED (FORMAL_START): %s missing" % ready_path)
    ready = proj.read_json(ready_path)
    proj.require(ready.get("COMMON_EVALUATOR_V2_READY") is True,
                 "FAIL CLOSED (FORMAL_START): COMMON_EVALUATOR_V2_READY is not "
                 "true — the §七 gate loop did not pass")
    proj.require(ready.get("FORMAL_RANKING_STARTED") is False,
                 "FAIL CLOSED (FORMAL_START): FORMAL_RANKING_STARTED is %r — "
                 "formal runs execute BEFORE the closing flip; after the "
                 "ranking is published this driver must not run again"
                 % ready.get("FORMAL_RANKING_STARTED"))
    marker_path = os.path.join(pool_cc4_dir, SECONDARY_AUDIT_MARKER_NAME)
    proj.require(os.path.isfile(marker_path),
                 "FAIL CLOSED (FORMAL_START): %s missing — the secondary audit "
                 "pass marker must be recorded BEFORE any formal run"
                 % marker_path)
    side_path = marker_path + ".sha256"
    proj.require(os.path.isfile(side_path),
                 "FAIL CLOSED (FORMAL_START): marker SHA sidecar missing")
    side = open(side_path, encoding="utf-8").read().split()
    proj.require(len(side) >= 1 and len(side[0]) == 64,
                 "FAIL CLOSED (FORMAL_START): marker SHA sidecar malformed")
    want = side[0]
    got = proj.sha256_file(marker_path)
    proj.require(got == want,
                 "FAIL CLOSED (FORMAL_START): marker sha %s != sidecar %s"
                 % (got, want))
    marker = proj.read_json(marker_path)
    proj.require(marker.get("verdict") == SECONDARY_AUDIT_VERDICT,
                 "FAIL CLOSED (FORMAL_START): marker verdict %r != %r"
                 % (marker.get("verdict"), SECONDARY_AUDIT_VERDICT))
    # Canonical marker schema (sole producer: tier3_formal_start_marker_v2dt.
    # write_marker): the binding-gate SHA lives under the evidence block. A
    # top-level binding_gate_sha256 is the legacy hand-built shape and is
    # rejected (regression: 2026-07-31 pre-launch catch).
    evidence = marker.get("evidence")
    proj.require(isinstance(evidence, dict),
                 "FAIL CLOSED (FORMAL_START): marker has no evidence block "
                 "(not written by tier3_formal_start_marker_v2dt?)")
    gate_sha = evidence.get("binding_gate_sha256")
    proj.require(gate_sha == POOL_BINDING_GATE_V2DT_SHA256,
                 "FAIL CLOSED (FORMAL_START): marker binding-gate sha %r != "
                 "frozen %s" % (gate_sha, POOL_BINDING_GATE_V2DT_SHA256))
    proj.require(os.path.normpath(marker.get("pool_cc4_dir") or "")
                 == os.path.normpath(pool_cc4_dir),
                 "FAIL CLOSED (FORMAL_START): marker pool_cc4_dir %r != %r"
                 % (marker.get("pool_cc4_dir"), pool_cc4_dir))
    proj.require(bool(marker.get("recorded_at_utc")),
                 "FAIL CLOSED (FORMAL_START): marker has no recorded_at_utc")
    return {"path": marker_path,
            "sha256": got,
            "verdict": marker["verdict"],
            "recorded_at_utc": marker["recorded_at_utc"],
            "binding_gate_sha256": gate_sha}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate-id")
    ap.add_argument("--common-dir", default="/home/oseasy/student_pool_v1/common_v2")
    ap.add_argument("--v1-common-dir", default="/home/oseasy/student_pool_v1/common")
    ap.add_argument("--frozen-bank-artifacts",
                    default="/home/oseasy/student_pool_v1/common/frozen_bank_artifacts")
    ap.add_argument("--out-root", default=None,
                    help="pool cc4 dir; default <pool>/cc4 (parent of --common-dir)")
    ap.add_argument("--out", default=None,
                    help="explicit output dir (overrides --out-root)")
    ap.add_argument("--rehearsal-scratch", default=None,
                    help="scratch dir for bounded rehearsal runs (NOT formal)")
    ap.add_argument("--limit-full", type=int, default=None)
    ap.add_argument("--limit-front", type=int, default=None)
    ap.add_argument("--limit-back", type=int, default=None)
    ap.add_argument("--self-test", action="store_true",
                    help="structural + live-schedule checks (server venv)")
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test()
    proj.require(bool(args.candidate_id),
                 "FAIL CLOSED: --candidate-id is required")

    # --- Stage 0: anti-pollution + registry + mode + launch contract ---------
    hook = os.environ.get("RMT16_POSTJAX_BINDING_SELFTEST", "")
    proj.require(hook.strip() in ("", "0"),
                 "FAIL CLOSED (anti-pollution): RMT16_POSTJAX_BINDING_SELFTEST=%r "
                 "is set (false-success hook). Unset before any formal run." % hook)
    spec = proj.get_spec(args.candidate_id)
    rehearsal, rehearsal_limits = check_rehearsal_args(
        args.limit_full, args.limit_front, args.limit_back, args.rehearsal_scratch)
    pool_root = os.path.dirname(os.path.normpath(args.common_dir))
    pool_cc4_dir = args.out_root or os.path.join(pool_root, "cc4")
    if args.out:
        out_dir = args.out
    elif rehearsal:
        out_dir = os.path.join(args.rehearsal_scratch, args.candidate_id,
                               "formal_evaluation_v2dt")
    else:
        out_dir = os.path.join(pool_cc4_dir, args.candidate_id,
                               "formal_evaluation_v2dt")
    tools_dir = HERE
    repo_root = os.path.dirname(os.path.dirname(tools_dir))
    cwd_real = os.path.realpath(os.getcwd())
    root_real = os.path.realpath(repo_root)
    proj.require(cwd_real == root_real,
                 "FAIL CLOSED (launch contract): cwd %s != repo root %s"
                 % (cwd_real, root_real))
    print("[stage0] candidate=%s family=%s class=%s mode=%s out=%s protocol=%s"
          % (args.candidate_id, spec["runtime_family"], spec["candidate_class"],
             "REHEARSAL" if rehearsal else "FORMAL", out_dir,
             COMMON_EVALUATOR_PROTOCOL_VERSION), flush=True)

    # --- engine import (V2) ---------------------------------------------------
    import tier3_evaluator_v2 as ev
    import tier3_event_predicates_v2 as predm
    import tier3_state_serializer as ser
    import tier3_failure_taxonomy as taxonomy
    proj.require(predm.COMMON_EVALUATOR_PROTOCOL_VERSION
                 == COMMON_EVALUATOR_PROTOCOL_VERSION,
                 "FAIL CLOSED: predicate module protocol version %r"
                 % predm.COMMON_EVALUATOR_PROTOCOL_VERSION)
    proj.require(ser.have_jax_craftax(),
                 "FAIL CLOSED (BLOCKED_ENVIRONMENT): JAX+craftax required "
                 "(jax=%s, craftax=%s)" % (ser.have_jax(), ser.have_craftax()))
    proj.require(int(ev.MAX_TIMESTEPS) == proj.FROZEN_MAX_TIMESTEPS,
                 "FAIL CLOSED: engine MAX_TIMESTEPS %r" % ev.MAX_TIMESTEPS)
    proj.require(ev.ACTION_MODE == proj.FROZEN_ACTION_MODE,
                 "FAIL CLOSED: engine ACTION_MODE %r" % ev.ACTION_MODE)
    ev.assert_output_dir_fresh(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # --- Stage 1: V2 engine + V2 common identity + V1 preservation -----------
    print("[stage1] verifying V2 engine + common_v2/ identity + V1 preservation "
          "...", flush=True)
    common_ev = smokev2.verify_engine_and_common_v2(args.common_dir,
                                                    args.v1_common_dir, tools_dir)
    print("[stage1] %s; engine modules LF-SHA %d; profile counts %s; v1 %s"
          % (common_ev["common_v2_sha256sums_self_check"],
             common_ev["engine_modules_lf_sha_verified"],
             common_ev["profile_episode_counts"],
             common_ev["v1_preservation"]["v1_status"]), flush=True)

    # --- Stage 1b: formal start authorization (marker gate) ------------------
    marker_ref = None
    if rehearsal:
        ready_path = os.path.join(args.common_dir, "COMMON_EVALUATOR_V2_READY.json")
        ready = proj.read_json(ready_path)
        proj.require(ready.get("COMMON_EVALUATOR_V2_READY") is True,
                     "FAIL CLOSED (REHEARSAL): COMMON_EVALUATOR_V2_READY not true")
        print("[stage1b] REHEARSAL mode: secondary-audit marker gate SKIPPED "
              "(bounded plumbing check only; outputs are NOT formal results)",
              flush=True)
    else:
        marker_ref = verify_formal_start(args.common_dir, pool_cc4_dir)
        print("[stage1b] secondary audit marker VERIFIED: sha=%s verdict=%s "
              "recorded_at=%s" % (marker_ref["sha256"][:16],
                                  marker_ref["verdict"],
                                  marker_ref["recorded_at_utc"]), flush=True)

    # --- Stage 2: GPU discipline ----------------------------------------------
    gpu_ev = smokev2.verify_gpu_allowed()
    print("[stage2] visible GPUs %s (allowlist enforced)"
          % gpu_ev["visible_gpu_uuids"], flush=True)

    # --- Stage 3: dicode resolution pin + canonical env -----------------------
    dicode_ev = proj.pin_dicode_resolution(repo_root)
    import jax
    import jax.numpy as jnp
    print("[stage3] building canonical env ...", flush=True)
    entry = ev.make_canonical_env()
    proj.require(tuple(entry["observation_shape"]) == proj.FROZEN_OBSERVATION_SHAPE,
                 "FAIL CLOSED: observation shape %s" % (entry["observation_shape"],))
    proj.require(int(entry["action_count"]) == proj.FROZEN_ACTION_DIM,
                 "FAIL CLOSED: action count %d" % entry["action_count"])

    # --- Stage 4: frozen bank artifacts ---------------------------------------
    import tier3_frozen_bank_artifacts as art
    print("[stage4] loading frozen bank artifacts (read-only) ...", flush=True)
    schedule = ev.performance_start_schedule()
    bindings, bank_ev = {}, {}
    for sc in (ev.FRONT, ev.BACK):
        b = art.load_bank(sc, args.frozen_bank_artifacts)
        want = (proj.FROZEN_FRONT_BANK_CONTENT_SHA256 if sc == ev.FRONT
                else proj.FROZEN_BACK_BANK_CONTENT_SHA256)
        proj.require(b.get("state_bank_hash") == want,
                     "FAIL CLOSED (FROZEN_BANK_CONTENT_MISMATCH): %s %r != %s"
                     % (sc, b.get("state_bank_hash"), want))
        bank_n = (proj.FROZEN_FRONT_EPISODE_COUNT if sc == ev.FRONT
                  else proj.FROZEN_BACK_EPISODE_COUNT)
        proj.require(len(b.get("states", [])) == bank_n,
                     "FAIL CLOSED: bank %s has %d states, frozen %d"
                     % (sc, len(b.get("states", [])), bank_n))
        proj.require([int(s) for s in b.get("seeds", [])]
                     == [int(s) for s in schedule[sc]["seeds"]],
                     "FAIL CLOSED: bank %s seed order %r != formal schedule %r"
                     % (sc, b.get("seeds"), schedule[sc]["seeds"]))
        bindings[sc] = b
        bank_ev[sc] = {"state_bank_hash": b.get("state_bank_hash"),
                       "bank_source": b.get("bank_source"),
                       "artifact_file_sha256": b.get("artifact_file_sha256"),
                       "loaded_content_sha256": b.get("loaded_content_sha256"),
                       "device_provenance": b.get("device_provenance"),
                       "n_states": len(b.get("states", []))}

    # --- Stage 5: capsule verification + owner runtime load -------------------
    print("[stage5] verifying capsule + loading owner runtime ...", flush=True)
    capsule_ev = proj.verify_capsule_files(spec)
    ctx = proj.load_owner_runtime(spec)
    params_before = proj.recompute_params_sha_owner(ctx)
    declared_params = spec["declared_params_sha256"]["value"]
    proj.require(params_before == declared_params,
                 "FAIL CLOSED (PARAMS_SHA_CC4_RECOMPUTE_MISMATCH): %s != %s"
                 % (params_before, declared_params))
    file_sha = proj.recompute_checkpoint_file_sha_owner(spec, ctx)
    declared_file = spec["declared_checkpoint_file_sha256"]["value"]
    proj.require(file_sha == declared_file,
                 "FAIL CLOSED (CHECKPOINT_FILE_SHA_CC4_RECOMPUTE_MISMATCH): "
                 "%s != %s" % (file_sha, declared_file))
    print("[stage5] params/checkpoint file SHA (owner protocol) == declared "
          "MATCH", flush=True)

    # --- Stage 6: policy adapter ----------------------------------------------
    print("[stage6] building policy adapter ...", flush=True)
    policy = proj.build_policy(spec, ctx)
    boundary_ev = None
    if spec["loader_kind"] == "cc3_slowgru":
        boundary_ev = proj.slowgru_boundary_unit_check(ctx["module"],
                                                       spec["carry_mode"])
        print("[stage6] slowgru boundary unit check: carry_mode=%s"
              % boundary_ev["carry_mode"], flush=True)
    batch1_ev = getattr(policy, "batch1_workaround", None)
    if batch1_ev and batch1_ev.get("applied"):
        print("[stage6] batch-1 workaround ACTIVE (disclosed): effective_batch=%d"
              % batch1_ev["effective_batch"], flush=True)
    elif batch1_ev:
        print("[stage6] batch-1 workaround NOT applied (disclosed): %s"
              % batch1_ev.get("reason"), flush=True)

    # --- Stage 7: FORMAL rollouts (frozen schedule, full horizon) -------------
    counts = {ev.FULL: proj.FROZEN_FULL_EPISODE_COUNT,
              ev.FRONT: proj.FROZEN_FRONT_EPISODE_COUNT,
              ev.BACK: proj.FROZEN_BACK_EPISODE_COUNT}
    if rehearsal:
        counts = {ev.FULL: rehearsal_limits["full"],
                  ev.FRONT: rehearsal_limits["front_l2"],
                  ev.BACK: rehearsal_limits["back_l2"]}
    max_steps = int(ev.MAX_TIMESTEPS)
    print("[stage7] %s rollouts (V2 dynamic topology): FULL=%d FRONT=%d BACK=%d "
          "max_steps=%d action_mode=%s ..."
          % ("REHEARSAL" if rehearsal else "FORMAL", counts[ev.FULL],
             counts[ev.FRONT], counts[ev.BACK], max_steps, ev.ACTION_MODE),
          flush=True)
    reset_fn = ev._jit_reset(entry)
    records_by_scenario, results_by_scenario = {}, {}
    entry_ids_by_scenario, timing_by_scenario = {}, {}
    formal_abort = None
    scenario_wall = {}
    for sc in FORMAL_SCENARIO_ORDER:
        t_sc0 = time.perf_counter()
        seeds = [int(s) for s in schedule[sc]["seeds"]][:counts[sc]]
        entry_ids = ev.state_entry_ids_for(sc, seeds)
        entry_ids_by_scenario[sc] = entry_ids
        eps, timings = [], []
        for i, seed in enumerate(seeds):
            policy.reset()
            if sc == ev.FULL:
                _obs0, start_state = reset_fn(jax.random.PRNGKey(seed))
            else:
                start_state = jax.tree.map(jnp.asarray,
                                           bindings[sc]["states"][i])
            t0 = time.perf_counter()
            try:
                rec = ev.rollout_episode(entry, start_state, sc, policy,
                                         entry_ids[i], seed, max_steps=max_steps)
            except predm.FailClosed as exc:
                # Identical discipline to the V2DT binding driver: under V2 the
                # legal-mining abort class cannot occur; remaining FailClosed
                # verdicts are corruption class only. Recorded, never relaxed,
                # never faked; candidate → BLOCKED.
                formal_abort = {
                    "exception_type": "tier3_event_predicates.FailClosed (via "
                                      "tier3_event_predicates_v2 re-export)",
                    "engine_message": str(exc),
                    "scenario": sc,
                    "episode_index": i,
                    "episodes_planned": len(seeds),
                    "episodes_completed_before_abort": i,
                    "entry_id": entry_ids[i],
                    "seed": seed,
                    "scenarios_completed_before_abort":
                        list(records_by_scenario.keys()),
                    "verdict": "ENGINE_PREDICATE_REJECTED_FORMAL_ROLLOUT_V2",
                    "v2_failclosed_class": "CORRUPTION_CLASS_ONLY (out-of-bounds / "
                        "non-finite / undecodable / player-vs-current-map "
                        "contradiction); legal topology mutation is in-domain "
                        "under V2_DYNAMIC_TOPOLOGY",
                    "authority": "frozen engine predicate (not relaxable by CC4)",
                    "consequence": "candidate BLOCKED; NOT recorded as a formal "
                        "score; no candidate-level exemption; no retraining",
                }
                print("  [%s %d/%d %s seed=%d] ENGINE PREDICATE ABORT (recorded):"
                      " %s" % (sc, i + 1, len(seeds), entry_ids[i], seed, exc),
                      flush=True)
                break
            wall = time.perf_counter() - t0
            rec["episode_record_sha256"] = proj.sha256_bytes(
                proj.canonical_json_bytes(rec))
            eps.append(rec)
            timings.append({"episode_id": rec["episode_id"],
                            "entry_id": entry_ids[i], "seed": seed,
                            "wall_seconds": round(wall, 6),
                            "timesteps": int(rec["timesteps"])})
            print("  [%s %d/%d %s seed=%d] steps=%d defeat=%s died=%s "
                  "transition=%s engaged=%s progress=%s wall=%.2fs"
                  % (sc, i + 1, len(seeds), entry_ids[i], seed, rec["timesteps"],
                     rec["defeat_kobold"], rec["player_died"],
                     rec["front_floor_transition_reached"], rec["kobold_engaged"],
                     rec["graph_distance_progress"], wall), flush=True)
        scenario_wall[sc] = round(time.perf_counter() - t_sc0, 6)
        timing_by_scenario[sc] = {"episodes": timings,
                                  "scenario_wall_seconds": scenario_wall[sc],
                                  "peak_rss_kb": _peak_rss_kb()}
        if eps:
            lines = [proj.canonical_json_bytes(r).decode("utf-8") for r in eps]
            records_by_scenario[sc] = {
                "seeds": seeds[:len(eps)],
                "entry_ids": entry_ids[:len(eps)],
                "episode_records": eps,
                "partial": bool(formal_abort and formal_abort["scenario"] == sc),
                "episode_records_sha256": proj.sha256_bytes(
                    ("\n".join(lines) + "\n").encode("utf-8"))}
            if not (formal_abort and formal_abort["scenario"] == sc):
                try:
                    results_by_scenario[sc] = ev.evaluate(sc, eps)
                except taxonomy.FailClosed as exc:
                    # The frozen failure taxonomy refused to CLASSIFY the
                    # completed rollouts (e.g. NEG20 ambiguous/contradictory
                    # terminal signals: floor-transition AND defeat_kobold).
                    # The episode records are intact; the protocol simply
                    # cannot produce metrics for this candidate. Same BLOCKED
                    # discipline as a rollout-predicate abort: recorded, never
                    # relaxed, never faked; a rerun reproduces deterministically.
                    formal_abort = {
                        "exception_type": "tier3_failure_taxonomy.FailClosed "
                                          "(via frozen tier3_evaluator.evaluate)",
                        "engine_message": str(exc),
                        "scenario": sc,
                        "aborted_phase": "evaluate_classification",
                        "episode_index": None,
                        "episodes_planned": len(seeds),
                        "episodes_completed_before_abort": len(eps),
                        "entry_id": None,
                        "seed": None,
                        "scenarios_completed_before_abort":
                            list(records_by_scenario.keys()),
                        "verdict": "ENGINE_TAXONOMY_REJECTED_FORMAL_EVALUATION_V2",
                        "v2_failclosed_class": "TERMINAL_SIGNAL_AMBIGUITY "
                            "(frozen classifier refuses contradictory terminal "
                            "signals; NOT corruption, NOT relaxable by CC4)",
                        "authority": "frozen engine taxonomy (not relaxable by CC4)",
                        "consequence": "candidate BLOCKED; NOT recorded as a "
                            "formal score; no candidate-level exemption; no "
                            "retraining; rerun would reproduce deterministically",
                    }
                    print("  [%s] ENGINE TAXONOMY ABORT after %d/%d rollouts "
                          "(recorded, candidate BLOCKED): %s"
                          % (sc, len(eps), len(seeds), exc), flush=True)
        # flush per-scenario result immediately (legible partial evidence on
        # mid-run crash; the freshness gate quarantines any partial dir)
        _write_scenario_result(out_dir, sc, args.candidate_id, schedule[sc],
                               entry_ids, counts[sc], records_by_scenario.get(sc),
                               results_by_scenario.get(sc),
                               timing_by_scenario[sc],
                               bool(formal_abort and formal_abort["scenario"] == sc),
                               rehearsal)
        if formal_abort:
            break
    if formal_abort:
        # flush skeleton result files for scenarios never reached, so the
        # six-file evidence shape (and SHA256SUMS_FORMAL_V2DT) stays complete
        # and legible: episodes_executed=0, evaluation=null
        for sc in FORMAL_SCENARIO_ORDER:
            if sc in timing_by_scenario:
                continue
            _write_scenario_result(
                out_dir, sc, args.candidate_id, schedule[sc],
                ev.state_entry_ids_for(
                    sc, [int(s) for s in schedule[sc]["seeds"]][:counts[sc]]),
                counts[sc], None, None,
                {"episodes": [], "scenario_wall_seconds": 0.0,
                 "peak_rss_kb": _peak_rss_kb()},
                False, rehearsal)

    # --- NEG23 analog (params unchanged by the evaluation) --------------------
    params_after = proj.recompute_params_sha_owner(ctx)
    proj.require(params_after == params_before,
                 "FAIL CLOSED (PARAMS_CHANGED_BY_EVALUATION): before %s != after "
                 "%s" % (params_before, params_after))
    params_unchanged = True

    # --- provenance ------------------------------------------------------------
    provenance = _collect_provenance(ev, tools_dir, gpu_ev, scenario_wall)

    # --- episode_records.jsonl -------------------------------------------------
    jsonl_lines = []
    for sc in FORMAL_SCENARIO_ORDER:
        if sc not in records_by_scenario:
            continue
        for r in records_by_scenario[sc]["episode_records"]:
            jsonl_lines.append(proj.canonical_json_bytes(r).decode("utf-8"))
    jsonl_path = os.path.join(out_dir, "episode_records.jsonl")
    with open(jsonl_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(("\n".join(jsonl_lines) + "\n") if jsonl_lines else "")
    episode_records_jsonl_sha256 = proj.sha256_file(jsonl_path)

    smokev2.write_json(os.path.join(out_dir, "provenance.json"), provenance)

    # --- evaluation certificate ------------------------------------------------
    common_v2_ready_sha256 = proj.sha256_file(
        os.path.join(args.common_dir, "COMMON_EVALUATOR_V2_READY.json"))
    episodes_executed = {sc: len(records_by_scenario.get(sc, {})
                                     .get("episode_records", []))
                         for sc in FORMAL_SCENARIO_ORDER}
    valid_start_counts = {
        sc: int((results_by_scenario.get(sc) or {}).get("valid_start_count", 0))
        for sc in FORMAL_SCENARIO_ORDER}
    cert_input = {
        "candidate_id": args.candidate_id,
        "spec": spec,
        "generated_at_utc": smokev2.utc_now_iso(),
        "bfs_graph_source": predm.BFS_GRAPH_SOURCE,
        "common_dir": args.common_dir,
        "results_by_scenario": results_by_scenario,
        "schedule": {sc: schedule[sc] for sc in FORMAL_SCENARIO_ORDER},
        "entry_ids_by_scenario": entry_ids_by_scenario,
        "records_sha256_by_scenario": {
            sc: records_by_scenario[sc]["episode_records_sha256"]
            for sc in records_by_scenario},
        "episode_records_jsonl_sha256": episode_records_jsonl_sha256,
        "episodes_executed": episodes_executed,
        "valid_start_counts": valid_start_counts,
        "params_before": params_before,
        "declared_params": declared_params,
        "file_sha": file_sha,
        "declared_file": declared_file,
        "params_unchanged": params_unchanged,
        "capsule_ev": capsule_ev,
        "common_ev": common_ev,
        "gpu_ev": gpu_ev,
        "dicode_ev": dicode_ev,
        "provenance": {"pid": provenance["pid"],
                       "argv": provenance["argv"],
                       "cwd": provenance["cwd"],
                       "host": provenance["host"],
                       "git_commit_head": provenance["git_commit_head"],
                       "device_identity": provenance["device_identity"],
                       "scenario_wall_seconds": scenario_wall,
                       "timing_by_scenario": timing_by_scenario},
        "engine_module_shas": {
            "tier3_evaluator_v2.py": provenance["evaluator_v2_module_sha256"],
            "tier3_event_predicates_v2.py":
                provenance["predicates_v2_module_sha256"]},
        "policy_class": type(policy).__name__,
        "batch1_ev": batch1_ev,
        "boundary_ev": boundary_ev,
        "formal_abort": formal_abort,
        "rehearsal": rehearsal,
        "rehearsal_limits": rehearsal_limits,
        "marker_ref": marker_ref,
        "binding_gate_sha256": (marker_ref["binding_gate_sha256"]
                                if marker_ref else None),
        "common_v2_ready_sha256": common_v2_ready_sha256,
        "rmt16_engine_metadata": ctx.get("engine_metadata"),
        "rmt16_frozen_identities": ctx.get("frozen_identities"),
        "rmt16_common_runner_sha256": ctx.get("common_runner_sha256"),
        "rmt16_engine_lf_sha256": ctx.get("engine_lf_sha256"),
    }
    cert = certmod.build_evaluation_certificate(cert_input)
    cert_path = os.path.join(out_dir, "evaluation_certificate_v2dt.json")
    smokev2.write_json(cert_path, cert)
    cert_problems = certmod.verify_evaluation_certificate(cert, evidence_dir=None)
    proj.require(not cert_problems,
                 "FAIL CLOSED (CERTIFICATE_SELF_VERIFY): %s" % cert_problems)

    # --- SHA256SUMS_FORMAL_V2DT --------------------------------------------------
    summed = ["episode_records.jsonl", "provenance.json",
              "evaluation_certificate_v2dt.json"] + [
        "evaluation_result_v2dt.%s.json" % sc for sc in FORMAL_SCENARIO_ORDER]
    sums_entries = ["%s  %s" % (proj.sha256_file(os.path.join(out_dir, fn)), fn)
                    for fn in summed]
    with open(os.path.join(out_dir, "SHA256SUMS_FORMAL_V2DT"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(sums_entries) + "\n")

    # --- READY_FORMAL_V2DT.json (sums-excluded) ----------------------------------
    formal_counts = {ev.FULL: proj.FROZEN_FULL_EPISODE_COUNT,
                     ev.FRONT: proj.FROZEN_FRONT_EPISODE_COUNT,
                     ev.BACK: proj.FROZEN_BACK_EPISODE_COUNT}
    is_teacher = spec["candidate_class"] == "TEACHER_REFERENCE"
    gates = {
        "G1_CAPSULE_FILE_SHA_MATCH": all(
            capsule_ev[fn]["match"] for fn in proj.CAPSULE_FILES),
        "G2_CHECKPOINT_FILE_SHA_OWNER_RECOMPUTE_MATCH": file_sha == declared_file,
        "G3_PARAMS_SHA_OWNER_RECOMPUTE_MATCH": params_before == declared_params,
        "G5_PARAMS_UNCHANGED": params_unchanged,
        "G6_COMMON_V2_SUMS": common_ev["common_v2_sha256sums_self_check"]
            == "PASS (%d/%d)" % (smokev2.FROZEN_V2_COMMON_SUMS_ENTRY_COUNT,
                                 smokev2.FROZEN_V2_COMMON_SUMS_ENTRY_COUNT),
        "G7_EPISODE_COUNTS_FROM_V2_PROFILE":
            common_ev["profile_episode_counts"]["front_episode_count"] == 8
            and common_ev["profile_episode_counts"]["back_episode_count"] == 8
            and common_ev["profile_episode_counts"]["full_episode_count"] == 64,
        "G8_GPU_ALLOWED": bool(gpu_ev["visible_gpu_uuids"]) and all(
            u in proj.CC4_GPU_ALLOWED_UUIDS
            for u in gpu_ev["visible_gpu_uuids"]),
        "G9_V1_FROZEN_PRESERVED":
            common_ev["v1_preservation"]["v1_sha256sums_self_check"]
            == "PASS (%d/%d)" % (proj.FROZEN_COMMON_SUMS_ENTRY_COUNT,
                                 proj.FROZEN_COMMON_SUMS_ENTRY_COUNT)
            and common_ev["v1_preservation"]["v1_formal_ranking_authorized"]
            is False,
        "G10_PROTOCOL_VERSION_V2":
            common_ev["common_evaluator_protocol_version"]
            == COMMON_EVALUATOR_PROTOCOL_VERSION,
        "G12_CERTIFICATE_VERIFIED": not cert_problems,
    }
    if rehearsal:
        gates["G4_REHEARSAL_SCHEDULE_EXECUTED"] = (formal_abort is None) and all(
            episodes_executed[sc] == counts[sc] for sc in FORMAL_SCENARIO_ORDER)
        gates["G11_REHEARSAL_LIMITS_RESPECTED"] = all(
            counts[sc] <= formal_counts[sc] for sc in FORMAL_SCENARIO_ORDER)
    else:
        gates["G4_FORMAL_SCHEDULE_COMPLETE"] = (formal_abort is None) and all(
            episodes_executed[sc] == formal_counts[sc]
            and sc in results_by_scenario
            for sc in FORMAL_SCENARIO_ORDER)
        gates["G11_SECONDARY_AUDIT_MARKER_VERIFIED"] = marker_ref is not None
    if spec["loader_kind"] == "cc3_slowgru":
        gates["G13_BOUNDARY_SEMANTICS_UNIT_CHECK"] = bool(boundary_ev)
    evaluation_status = ("BLOCKED" if formal_abort is not None
                         else "REHEARSAL_NOT_FORMAL" if rehearsal else "PASS")
    ready = {
        "schema": READY_FORMAL_SCHEMA,
        "candidate_id": args.candidate_id,
        "runtime_family": spec["runtime_family"],
        "common_evaluator_protocol_version": COMMON_EVALUATOR_PROTOCOL_VERSION,
        "READY_FORMAL_V2DT": all(gates.values()),
        "evaluation_status": evaluation_status,
        "run_class": RUN_CLASS,
        "rehearsal": rehearsal,
        "rehearsal_limits": rehearsal_limits,
        "formal_abort": formal_abort,
        "candidate_class": spec["candidate_class"],
        "counts_toward_student_binding_count": (False if is_teacher else True),
        "reference_only": spec["reference_only"],
        "teacher_included_in_student_ranking": False,
        "student_rank": None,
        "ranking_publication": "FORMAL_RANKING_SUMMARY_V2DT.json (published by "
            "tier3_formal_ranking_v2dt.py after ALL runs; certificates are "
            "never rewritten)",
        "performance_evaluation_executed": (not rehearsal)
                                           and formal_abort is None,
        "scientific_claim_authorized": False,
        "scaffolded_results_can_replace_full_task": False,
        "secondary_audit_marker": marker_ref,
        "gates": gates,
        "generated_at_utc": smokev2.utc_now_iso(),
        "evidence_files": summed + ["SHA256SUMS_FORMAL_V2DT"],
        "honest_false_discipline": "ANY failed gate keeps READY_FORMAL_V2DT "
            "false; a BLOCKED candidate is never recorded as a formal score; "
            "interface smoke is not a performance result",
    }
    smokev2.write_json(os.path.join(out_dir, "READY_FORMAL_V2DT.json"), ready)

    print("[done] %s evaluation_status=%s READY_FORMAL_V2DT=%s failed_gates=%s"
          % (args.candidate_id, evaluation_status, ready["READY_FORMAL_V2DT"],
             {k: v for k, v in gates.items() if not v} or "none"), flush=True)
    if formal_abort:
        ei = formal_abort.get("episode_index")
        pos = ("episode %d/%d" % (ei + 1, formal_abort["episodes_planned"])
               if ei is not None else
               "after %d/%d rollouts (evaluate phase)"
               % (formal_abort["episodes_completed_before_abort"],
                  formal_abort["episodes_planned"]))
        print("[done] BLOCKED by frozen engine: %s @ %s %s — verdict %s; "
              "recorded as minimum blocking evidence; NOT relaxed, NOT a "
              "formal score"
              % (args.candidate_id, formal_abort["scenario"], pos,
                 formal_abort["verdict"]), flush=True)
    print("[done] out=%s" % out_dir, flush=True)
    return 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _peak_rss_kb():
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None


def _write_scenario_result(out_dir, sc, candidate_id, sched, entry_ids,
                           planned, records, evaluation, timing, aborted_here,
                           rehearsal):
    obj = {
        "schema": RESULT_SCHEMA,
        "candidate_id": candidate_id,
        "scenario": sc,
        "run_class": RUN_CLASS,
        "common_evaluator_protocol_version": COMMON_EVALUATOR_PROTOCOL_VERSION,
        "rehearsal": rehearsal,
        "schedule": sched,
        "entry_ids_planned": entry_ids,
        "episodes_planned": planned,
        "episodes_executed": len((records or {}).get("episode_records", [])),
        "episode_records_sha256": (records or {}).get("episode_records_sha256"),
        "aborted_in_scenario": aborted_here,
        "evaluation": evaluation,        # ev.evaluate output or None
        "timing": timing,
        "generated_at_utc": smokev2.utc_now_iso(),
    }
    smokev2.write_json(os.path.join(out_dir,
                                    "evaluation_result_v2dt.%s.json" % sc), obj)


def _collect_provenance(ev, tools_dir, gpu_ev, scenario_wall):
    try:
        device_identity = ev._eval_device_identity()
    except Exception as exc:
        device_identity = {"error": repr(exc)}
    try:
        git_head = ev._git_commit_head()
    except Exception as exc:
        git_head = "unavailable: %r" % exc
    try:
        # _runtime_versions lives on the frozen V1 engine module (V2 re-exports
        # the public rollout surface, not this provenance helper)
        import tier3_evaluator as v1_ev
        runtime_versions = v1_ev._runtime_versions()
    except Exception as exc:
        runtime_versions = {"error": repr(exc)}
    xla_flags = {k: os.environ.get(k) for k in sorted(os.environ)
                 if k.startswith(("XLA_", "JAX_"))}
    return {
        "schema": PROVENANCE_SCHEMA,
        "pid": os.getpid(),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "host": socket.gethostname(),
        "python_executable": sys.executable,
        "generated_at_utc": smokev2.utc_now_iso(),
        "git_commit_head": git_head,
        "device_identity": device_identity,
        "runtime_versions": runtime_versions,
        "gpu": gpu_ev,
        "xla_jax_env": xla_flags,
        "scenario_wall_seconds": scenario_wall,
        "peak_rss_kb": _peak_rss_kb(),
        "projection_module_sha256": proj.sha256_file(
            os.path.join(tools_dir, "tier3_projection_runtime.py")),
        "driver_module_sha256": proj.sha256_file(os.path.abspath(__file__)),
        "projection_module_lf_sha256": proj.lf_sha256_file(
            os.path.join(tools_dir, "tier3_projection_runtime.py")),
        "driver_module_lf_sha256": proj.lf_sha256_file(os.path.abspath(__file__)),
        "evaluator_v2_module_sha256": proj.sha256_file(
            os.path.join(tools_dir, "tier3_evaluator_v2.py")),
        "predicates_v2_module_sha256": proj.sha256_file(
            os.path.join(tools_dir, "tier3_event_predicates_v2.py")),
        "certificate_module_sha256": proj.sha256_file(
            os.path.join(tools_dir, "tier3_evaluation_certificate_v2dt.py")),
        "binding_driver_module_sha256": proj.sha256_file(
            os.path.join(tools_dir, "tier3_projection_binding_smoke_v2.py")),
    }


# ---------------------------------------------------------------------------
# self-test (server venv: imports the engine; pins/schedule checks live)
# ---------------------------------------------------------------------------
def run_self_test():
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        checks += 1
        proj.require(cond, "FORMAL_DRIVER_SELF_TEST FAIL: %s" % msg)

    # pins identical to the frozen binding driver (single source of truth)
    ok(certmod.pins_snapshot()["common_evaluator_sha256"]
       == smokev2.FROZEN_V2_COMMON_EVALUATOR_SHA256, "pin evaluator")
    ok(certmod.pins_snapshot()["metric_schema_sha256"]
       == smokev2.FROZEN_V2_METRIC_SCHEMA_SHA256, "pin schema")
    ok(len(certmod.pins_snapshot()) == 11, "pin set size")

    import tier3_evaluator_v2 as ev
    ok(int(ev.MAX_TIMESTEPS) == 4096, "MAX_TIMESTEPS")
    ok(ev.ACTION_MODE == "greedy_argmax", "ACTION_MODE")
    ok(tuple(FORMAL_SCENARIO_ORDER) == ("full", "front_l2", "back_l2"),
       "scenario order")

    # the frozen formal schedule, live
    sched = ev.performance_start_schedule()
    ok(sched[ev.FULL]["seeds"] == [200000 + i for i in range(64)], "FULL seeds")
    ok(sched[ev.FULL]["count"] == 64, "FULL count")
    ok(sched[ev.FRONT]["seeds"] == [10000 + i for i in range(8)], "FRONT seeds")
    ok(sched[ev.BACK]["seeds"] == [1010000 + i for i in range(8)], "BACK seeds")
    ok(ev.state_entry_ids_for(ev.FULL, sched[ev.FULL]["seeds"])[0]
       == "full-seed200000", "entry id FULL")
    ok(ev.state_entry_ids_for(ev.FRONT, sched[ev.FRONT]["seeds"])
       == ["front_l2-bank%d" % i for i in range(8)], "entry ids FRONT")

    # rehearsal arg gates (pure)
    is_r, lim = check_rehearsal_args(None, None, None, None)
    ok(not is_r and lim is None, "formal mode default")
    is_r, lim = check_rehearsal_args(2, 2, 2, "/tmp/scratch")
    ok(is_r and lim == {"full": 2, "front_l2": 2, "back_l2": 2}, "rehearsal ok")
    try:
        check_rehearsal_args(2, None, None, None)
        ok(False, "limit without scratch accepted")
    except proj.FailClosed:
        checks += 1
    try:
        check_rehearsal_args(65, None, None, "/tmp/scratch")
        ok(False, "limit 65 accepted")
    except proj.FailClosed:
        checks += 1

    # start gate (pure, temp dirs)
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        common = os.path.join(td, "common_v2")
        cc4 = os.path.join(td, "cc4")
        os.makedirs(common)
        os.makedirs(cc4)
        try:
            verify_formal_start(common, cc4)
            ok(False, "missing READY accepted")
        except proj.FailClosed:
            checks += 1
        smokev2.write_json(os.path.join(common, "COMMON_EVALUATOR_V2_READY.json"),
                           {"COMMON_EVALUATOR_V2_READY": True,
                            "FORMAL_RANKING_STARTED": False,
                            "STUDENT_COMMON_BINDING_PASS_COUNT": "6/6"})
        try:
            verify_formal_start(common, cc4)
            ok(False, "missing marker accepted")
        except proj.FailClosed:
            checks += 1
        # happy path via the REAL marker producer (drift-proof: the driver
        # must accept exactly what tier3_formal_start_marker_v2dt writes).
        # The temp gate file cannot match the frozen pool pin, so the pin is
        # swapped for the synthetic sha for this block only; pin enforcement
        # against the real value is covered by the marker self-test.
        import tier3_formal_start_marker_v2dt as marker_tool
        gp = os.path.join(cc4, "POOL_BINDING_GATE_V2DT.json")
        with open(gp, "wb") as fh:
            fh.write(b'{"synthetic_formal_gate": true}\n')
        gsha = proj.sha256_file(gp)
        mp = os.path.join(cc4, SECONDARY_AUDIT_MARKER_NAME)
        marker_tool.write_marker(cc4, common,
                                 recorded_at_utc="1970-01-01T00:00:00+00:00",
                                 expected_gate_sha=gsha)
        good_bytes = open(mp, "rb").read()
        good_side = open(mp + ".sha256", "rb").read()
        saved_pin = globals()["POOL_BINDING_GATE_V2DT_SHA256"]
        globals()["POOL_BINDING_GATE_V2DT_SHA256"] = gsha
        try:
            sha = proj.sha256_file(mp)
            ref = verify_formal_start(common, cc4)
            ok(ref["sha256"] == sha and ref["verdict"] == SECONDARY_AUDIT_VERDICT
               and ref["binding_gate_sha256"] == gsha,
               "marker verified (real marker-tool shape)")
            # wrong verdict rejected
            marker2 = proj.read_json(mp)
            marker2["verdict"] = "PASS_SOMETHING_ELSE"
            smokev2.write_json(mp, marker2)
            sha2 = proj.sha256_file(mp)
            with open(mp + ".sha256", "w", encoding="utf-8", newline="\n") as fh:
                fh.write("%s  %s\n" % (sha2, SECONDARY_AUDIT_MARKER_NAME))
            try:
                verify_formal_start(common, cc4)
                ok(False, "wrong verdict accepted")
            except proj.FailClosed:
                checks += 1
        finally:
            globals()["POOL_BINDING_GATE_V2DT_SHA256"] = saved_pin
        # regression: the legacy hand-built shape (gate sha at TOP level, no
        # evidence block) must be REJECTED under the real frozen pin — this is
        # the exact mismatch caught at the 2026-07-31 pre-launch attempt.
        with open(mp, "wb") as fh:
            fh.write(good_bytes)
        with open(mp + ".sha256", "wb") as fh:
            fh.write(good_side)
        legacy = proj.read_json(mp)
        legacy["binding_gate_sha256"] = legacy["evidence"]["binding_gate_sha256"]
        del legacy["evidence"]
        smokev2.write_json(mp, legacy)
        sha3 = proj.sha256_file(mp)
        with open(mp + ".sha256", "w", encoding="utf-8", newline="\n") as fh:
            fh.write("%s  %s\n" % (sha3, SECONDARY_AUDIT_MARKER_NAME))
        try:
            verify_formal_start(common, cc4)
            ok(False, "legacy top-level gate-sha shape accepted")
        except proj.FailClosed as exc:
            ok("no evidence block" in str(exc),
               "legacy top-level gate-sha shape rejected")
        # FORMAL_RANKING_STARTED=true (post-closing) rejected
        with open(mp, "wb") as fh:
            fh.write(good_bytes)
        with open(mp + ".sha256", "wb") as fh:
            fh.write(good_side)
        smokev2.write_json(os.path.join(common, "COMMON_EVALUATOR_V2_READY.json"),
                           {"COMMON_EVALUATOR_V2_READY": True,
                            "FORMAL_RANKING_STARTED": True,
                            "STUDENT_COMMON_BINDING_PASS_COUNT": "6/6"})
        globals()["POOL_BINDING_GATE_V2DT_SHA256"] = gsha
        try:
            verify_formal_start(common, cc4)
            ok(False, "post-closing rerun accepted")
        except proj.FailClosed:
            checks += 1
        finally:
            globals()["POOL_BINDING_GATE_V2DT_SHA256"] = saved_pin

    # certificate module self-test (jax-free)
    checks += certmod.self_test()

    print("FORMAL_DRIVER_SELF_TEST_PASS checks=%d" % checks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
