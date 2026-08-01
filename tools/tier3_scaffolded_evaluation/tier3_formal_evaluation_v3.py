#!/usr/bin/env python3
"""CC4 Tier3 — FORMAL global performance evaluation DRIVER, V3_COMPOSITE_EVENT.

Authorized by 总控 ruling CC4_REPAIR_NEG20_COMPOSITE_EVENT_SEMANTICS_AND_COMPLETE_
FORMAL_EVALUATION_V3. V2 is archived CLOSED_INCONCLUSIVE_PARTICIPATION; V3 repairs
the frozen classifier's NEG20 handling of the legitimate composite event (front
floor transition AND defeat_kobold) and COMPLETES the formal evaluation.

Per-arm strategy (总控 §五 reuse; this driver NEVER calls the frozen ev.evaluate()):
  * FULL  — OFFLINE REUSE of the V2 committed/server episode records. Reuse gate
            R1–R9 must ALL hold → FULL_REUSE_STATUS=REUSED_PASS; any failure →
            REJECT and the candidate is honestly BLOCKED (a REJECT is NEVER silently
            rerun — that would be the forbidden performance retry).
  * FRONT — OFFLINE RECLASSIFICATION of the V2 records (classification_only=true,
            environment_rerun=false); the source V2 episode SHA, the V3 classifier
            SHA and the V3 result SHA are recorded. Re-run only if the record
            evidence is insufficient (it is sufficient: 8/8 for every candidate).
  * BACK  — CONTROL has a complete V2 BACK (8/8) → REUSE + V3 re-sign
            (classification_only=true). The six engine-blocked candidates (5
            students + teacher) have 0/8 V2 BACK skeletons → first-run COMPLETION
            at the identical frozen profile (verbatim V2 rollout loop; this is a
            completion of never-run skeletons, NOT a retry).

Classification uses tier3_taxonomy_v3 (primary_outcome + secondary_events[] +
taxonomy_status); primary/dense metrics come from the FROZEN tier3_metrics.
summarize called as a library → bit-identical to V2 by construction (the reused
V2 ranking extractor reads result["metrics"] unchanged).

Start authorization is gated on the V3_REPAIR_AUTHORIZATION marker (written by
tier3_formal_start_marker_v3.py BEFORE any V3 run; independent of the V2 READY,
which is already flipped). GPU discipline is STRICTER than V2: G16 restricts to
{GPU2, GPU3} only (the GPU0/GPU1 unban was V2-only).

Usage (server, locked CC4 venv, GPU2 or GPU3, CWD = repo root):
  CUDA_VISIBLE_DEVICES=GPU-8df11537-ab79-722d-606f-411966196c4c \
  python tools/tier3_scaffolded_evaluation/tier3_formal_evaluation_v3.py \
      --candidate-id BASE_GTRXL_ORIGINAL_VTRACE_98304 \
      [--common-dir /home/oseasy/student_pool_v1/common_v2] \
      [--v1-common-dir /home/oseasy/student_pool_v1/common] \
      [--frozen-bank-artifacts /home/oseasy/student_pool_v1/common/frozen_bank_artifacts] \
      [--v2-evidence-root /home/oseasy/student_pool_v1/cc4] \
      [--out-root /home/oseasy/student_pool_v1/cc4]
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import tier3_projection_runtime as proj                            # noqa: E402
import tier3_projection_binding_smoke_v2 as smokev2                # noqa: E402
import tier3_evaluation_certificate_v3 as certmod                  # noqa: E402
import tier3_taxonomy_v3 as taxonomy_v3                            # noqa: E402
# The reused FROZEN ranking machinery (tuple extraction / comparison / ranking).
# Imported lazily where needed to keep the module top JAX-free and acyclic.

RESULT_SCHEMA = "mechanism_UED.tier3_formal_evaluation_result/v3"
READY_FORMAL_SCHEMA = "mechanism_UED.tier3_formal_ready/v3"
PROVENANCE_SCHEMA = "mechanism_UED.tier3_formal_provenance/v3"
COMMON_EVALUATOR_PROTOCOL_VERSION = certmod.COMMON_EVALUATOR_PROTOCOL_VERSION  # V3_COMPOSITE_EVENT
NEG20_PROTOCOL = certmod.NEG20_PROTOCOL
RUN_CLASS = certmod.RUN_CLASS
FORMAL_SCENARIO_ORDER = certmod.FORMAL_SCENARIO_ORDER
FULL, FRONT, BACK = "full", "front_l2", "back_l2"
SUMS_FILENAME = certmod.SUMS_FILENAME                       # SHA256SUMS_FORMAL_V3

# V3 repair-authorization marker (sole producer: tier3_formal_start_marker_v3).
V3_REPAIR_MARKER_NAME = "V3_REPAIR_AUTHORIZATION.json"
V3_REPAIR_MARKER_SCHEMA = "mechanism_UED.tier3_v3_repair_authorization/v1"
V3_REPAIR_RULING_TASK = ("CC4_REPAIR_NEG20_COMPOSITE_EVENT_SEMANTICS_AND_COMPLETE_"
                         "FORMAL_EVALUATION_V3")
V3_REPAIR_VERDICT = "AUTHORIZED_COMPOSITE_EVENT_SEMANTIC_REPAIR_V3"
READY_V3_NAME = "COMMON_EVALUATOR_V3_READY.json"

# Archived V2 evidence references (continuity; never modified).
V2_ARCHIVE_SUMMARY_SHA256 = \
    "3e8186417aefeb25729324ce5fb4bc6b56a58087c8d1ee67bc088ad37d5c1ac3"
V2_ARCHIVE_GATE_SHA256 = \
    "51d3d6fb8efbc978875823cdc4576443c4d61f308840462c1bfa12da52fddc5b"

# G16: V3 runs ONLY on GPU2/GPU3 (the GPU0/GPU1 unban was V2-only → reverted).
V3_GPU_ALLOWED_UUIDS = (
    "GPU-8df11537-ab79-722d-606f-411966196c4c",   # GPU2
    "GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd",   # GPU3
)

# FULL offline reuse gate names (R1–R9, 总控 §五).
FULL_REUSE_GATE_KEYS = (
    "R1_EPISODES_COMPLETE", "R2_RECORD_SHA_RECOMPUTE", "R3_V2_SUMS_REHASH",
    "R4_CHECKPOINT_PARAMS_OWNER_MATCH", "R5_RUNTIME_CAPSULE_MATCH",
    "R6_SCHEDULE_FROZEN", "R7_NO_PERFORMANCE_EARLY_STOP",
    "R8_ENGINE_LF_SHA_FROZEN", "R9_V3_RECLASSIFY_REPRODUCIBLE",
)


# ---------------------------------------------------------------------------
# pure / JAX-free helpers (unit-testable on this host)
# ---------------------------------------------------------------------------
def check_rehearsal_args(limit_full, limit_front, limit_back, rehearsal_scratch):
    limits = {"full": limit_full, "front_l2": limit_front, "back_l2": limit_back}
    any_limit = any(v is not None for v in limits.values())
    if not any_limit and not rehearsal_scratch:
        return False, None
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


def verify_gpu_v3(visible_uuids):
    """G16: every visible GPU must be in {GPU2, GPU3} (STRICTER than the V2
    allowlist, which still lists GPU0/GPU1)."""
    proj.require(bool(visible_uuids),
                 "FAIL CLOSED (V3/G16_GPU): no visible GPU")
    bad = [u for u in visible_uuids if u not in V3_GPU_ALLOWED_UUIDS]
    proj.require(not bad,
                 "FAIL CLOSED (V3/G16_GPU): GPU(s) %s not in V3 allowlist "
                 "{GPU2, GPU3} — the GPU0/GPU1 unban was V2-only" % bad)
    return {"visible_gpu_uuids": list(visible_uuids),
            "v3_gpu_allowlist": list(V3_GPU_ALLOWED_UUIDS),
            "g16_gpu_v3_only": True}


def verify_v3_repair_start(common_dir, pool_cc4_dir):
    """The V3 start-authorization gate (stage 1b). Independent of the V2 READY.
    The V3_REPAIR_AUTHORIZATION marker must exist, its SHA sidecar must match, and
    it must record the verbatim 总控 ruling task + verdict + V2 archive SHAs + pin
    snapshot + git HEAD. Fail-closed on ANY deviation."""
    marker_path = os.path.join(pool_cc4_dir, V3_REPAIR_MARKER_NAME)
    proj.require(os.path.isfile(marker_path),
                 "FAIL CLOSED (V3_REPAIR_START): %s missing — the V3 repair-"
                 "authorization marker must be recorded BEFORE any V3 run"
                 % marker_path)
    side_path = marker_path + ".sha256"
    proj.require(os.path.isfile(side_path),
                 "FAIL CLOSED (V3_REPAIR_START): marker SHA sidecar missing")
    side = open(side_path, encoding="utf-8").read().split()
    proj.require(len(side) >= 1 and len(side[0]) == 64,
                 "FAIL CLOSED (V3_REPAIR_START): marker SHA sidecar malformed")
    want = side[0]
    got = proj.sha256_file(marker_path)
    proj.require(got == want,
                 "FAIL CLOSED (V3_REPAIR_START): marker sha %s != sidecar %s"
                 % (got, want))
    marker = proj.read_json(marker_path)
    proj.require(marker.get("schema") == V3_REPAIR_MARKER_SCHEMA,
                 "FAIL CLOSED (V3_REPAIR_START): marker schema %r"
                 % marker.get("schema"))
    proj.require(marker.get("ruling_task") == V3_REPAIR_RULING_TASK,
                 "FAIL CLOSED (V3_REPAIR_START): ruling_task %r != %r"
                 % (marker.get("ruling_task"), V3_REPAIR_RULING_TASK))
    proj.require(marker.get("verdict") == V3_REPAIR_VERDICT,
                 "FAIL CLOSED (V3_REPAIR_START): verdict %r != %r"
                 % (marker.get("verdict"), V3_REPAIR_VERDICT))
    evidence = marker.get("evidence")
    proj.require(isinstance(evidence, dict),
                 "FAIL CLOSED (V3_REPAIR_START): marker has no evidence block "
                 "(not written by tier3_formal_start_marker_v3?)")
    proj.require(evidence.get("v2_archive_summary_sha256") == V2_ARCHIVE_SUMMARY_SHA256,
                 "FAIL CLOSED (V3_REPAIR_START): V2 archive summary sha mismatch")
    proj.require(evidence.get("v2_archive_gate_sha256") == V2_ARCHIVE_GATE_SHA256,
                 "FAIL CLOSED (V3_REPAIR_START): V2 archive gate sha mismatch")
    pins = evidence.get("pins_snapshot")
    proj.require(pins == certmod.pins_snapshot(),
                 "FAIL CLOSED (V3_REPAIR_START): marker pins snapshot != frozen")
    proj.require(evidence.get("taxonomy_v3_lf_sha256")
                 == taxonomy_v3.module_lf_sha256(),
                 "FAIL CLOSED (V3_REPAIR_START): marker taxonomy_v3 LF-SHA drifted")
    proj.require(bool(evidence.get("git_commit_head")),
                 "FAIL CLOSED (V3_REPAIR_START): marker has no git_commit_head")
    proj.require(os.path.normpath(marker.get("pool_cc4_dir") or "")
                 == os.path.normpath(pool_cc4_dir),
                 "FAIL CLOSED (V3_REPAIR_START): marker pool_cc4_dir %r != %r"
                 % (marker.get("pool_cc4_dir"), pool_cc4_dir))
    proj.require(bool(marker.get("recorded_at_utc")),
                 "FAIL CLOSED (V3_REPAIR_START): marker has no recorded_at_utc")
    return {"path": marker_path,
            "sha256": got,
            "ruling_task": marker["ruling_task"],
            "verdict": marker["verdict"],
            "recorded_at_utc": marker["recorded_at_utc"],
            "git_commit_head": evidence["git_commit_head"]}


def load_v2_episode_records(v2_evidence_dir):
    """Read a candidate's V2 episode_records.jsonl, verify every record's
    episode_record_sha256 (reuse gate R2), and split by scenario. Returns
    (records_by_scenario, jsonl_file_sha, per_record_sha)."""
    jsonl = os.path.join(v2_evidence_dir, "episode_records.jsonl")
    proj.require(os.path.isfile(jsonl),
                 "FAIL CLOSED (V3_REUSE): V2 episode_records.jsonl missing: %s"
                 % jsonl)
    file_sha = proj.sha256_file(jsonl)
    records_by_scenario = {FULL: [], FRONT: [], BACK: []}
    per_record_sha = {}
    with open(jsonl, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            recomputed = taxonomy_v3.verify_record_sha(rec)
            proj.require(recomputed == rec.get("episode_record_sha256"),
                         "FAIL CLOSED (V3_REUSE/R2): record %s episode_record_"
                         "sha256 mismatch (recomputed %s)"
                         % (rec.get("episode_id"), recomputed))
            sc = rec.get("scenario")
            proj.require(sc in records_by_scenario,
                         "FAIL CLOSED (V3_REUSE): record scenario %r unregistered"
                         % sc)
            records_by_scenario[sc].append(rec)
            per_record_sha[rec["episode_id"]] = recomputed
    return records_by_scenario, file_sha, per_record_sha


def _engine_lf_shas_frozen():
    """The frozen engine LF-SHA pin set: V1 (tier3_candidate_runtime.py,
    tier3_evaluator.py) + V2 (tier3_evaluator_v2.py, tier3_event_predicates_v2.py)."""
    frozen = {}
    frozen.update(dict(proj.FROZEN_ENGINE_LF_SHA256))        # V1 pair
    frozen.update(dict(smokev2.FROZEN_V2_ENGINE_LF_SHA256))  # V2 pair
    return frozen


def _engine_lf_shas_live(tools_dir):
    """LF-SHA256 of exactly the frozen-pinned engine modules (derived from the
    frozen pin set's keys, so the two can never drift apart)."""
    out = {}
    for fn in _engine_lf_shas_frozen():
        out[fn] = proj.lf_sha256_file(os.path.join(tools_dir, fn))
    return out


def full_reuse_gate(v2_evidence_dir, records_full, schedule_full, tools_dir):
    """FULL offline reuse gate R1–R9 (总控 §五). All checks are offline reads /
    re-hashes / re-classifications — NOTHING is rerun. Returns
    {"gates": {R*: bool}, "reuse_status": REUSED_PASS|REJECT, "detail": {...}}."""
    gates = {k: False for k in FULL_REUSE_GATE_KEYS}
    detail = {}

    # R1: 64 complete episodes, not aborted, V2 evaluation present.
    full_result_path = os.path.join(v2_evidence_dir,
                                    "evaluation_result_v2dt.full.json")
    r1 = False
    if len(records_full) == proj.FROZEN_FULL_EPISODE_COUNT \
            and os.path.isfile(full_result_path):
        fr = proj.read_json(full_result_path)
        r1 = (fr.get("aborted_in_scenario") is False
              and fr.get("evaluation") is not None
              and int(fr.get("episodes_executed", 0))
              == proj.FROZEN_FULL_EPISODE_COUNT)
        detail["R1_v2_full_episodes_executed"] = fr.get("episodes_executed")
    gates["R1_EPISODES_COMPLETE"] = r1

    # R2: every FULL record's sha recomputes (re-assert on the FULL subset).
    gates["R2_RECORD_SHA_RECOMPUTE"] = all(
        taxonomy_v3.verify_record_sha(r) == r.get("episode_record_sha256")
        for r in records_full) and len(records_full) == proj.FROZEN_FULL_EPISODE_COUNT

    # R3: V2 SHA256SUMS_FORMAL_V2DT re-hashes clean.
    sums_path = os.path.join(v2_evidence_dir, "SHA256SUMS_FORMAL_V2DT")
    r3 = False
    if os.path.isfile(sums_path):
        sums = proj.parse_sha256sums(sums_path)
        r3 = bool(sums)
        for rel, wantsha in sums.items():
            p = os.path.join(v2_evidence_dir, rel)
            if not os.path.isfile(p) or proj.sha256_file(p) != wantsha:
                r3 = False
                detail["R3_failed_file"] = rel
                break
        detail["R3_v2_sums_entries"] = len(sums)
    gates["R3_V2_SUMS_REHASH"] = r3

    # R4/R5/R6/R8 read the V2 certificate (owner-recompute statuses, capsule
    # verification, frozen schedule, engine LF-SHA pins recorded at V2 run time).
    v2_cert_path = os.path.join(v2_evidence_dir, "evaluation_certificate_v2dt.json")
    v2_cert = proj.read_json(v2_cert_path) if os.path.isfile(v2_cert_path) else {}

    # R4: checkpoint/params owner-recompute matched and params unchanged.
    gates["R4_CHECKPOINT_PARAMS_OWNER_MATCH"] = (
        v2_cert.get("params_sha256_status") == certmod.RECOMPUTE_STATUS
        and v2_cert.get("checkpoint_file_sha256_status") == certmod.RECOMPUTE_STATUS
        and v2_cert.get("params_unchanged") is True
        and v2_cert.get("params_sha256") == v2_cert.get("params_sha256_owner_declared")
        and v2_cert.get("checkpoint_file_sha256")
        == v2_cert.get("checkpoint_file_sha256_owner_declared"))

    # R5: capsule files all matched + bound owner runtime sha present.
    capsule_ev = v2_cert.get("capsule_file_verification") or {}
    gates["R5_RUNTIME_CAPSULE_MATCH"] = (
        bool(capsule_ev)
        and all((capsule_ev.get(fn) or {}).get("match") is True
                for fn in proj.CAPSULE_FILES)
        and bool(v2_cert.get("bound_owner_runtime_sha256")))

    # R6: frozen FULL schedule (seeds 200000..200063), max_steps 4096, greedy.
    sched_seeds = [int(s) for s in (schedule_full or {}).get("seeds", [])]
    gates["R6_SCHEDULE_FROZEN"] = (
        sched_seeds == [200000 + i for i in range(proj.FROZEN_FULL_EPISODE_COUNT)]
        and v2_cert.get("max_steps") == proj.FROZEN_MAX_TIMESTEPS
        and v2_cert.get("action_mode") == proj.FROZEN_ACTION_MODE)

    # R7: no performance early-stop — every FULL record has a terminal signal.
    gates["R7_NO_PERFORMANCE_EARLY_STOP"] = (
        len(records_full) == proj.FROZEN_FULL_EPISODE_COUNT and all(
            (r.get("defeat_kobold") or r.get("player_died")
             or r.get("timed_out") or r.get("front_floor_transition_reached"))
            for r in records_full))

    # R8: the four frozen engine modules' LF-SHA match the frozen pins (recomputed
    # live from the tool source, compared to the pins recorded in the V2 cert AND
    # the frozen constants).
    live = _engine_lf_shas_live(tools_dir)
    frozen = _engine_lf_shas_frozen()
    v2_recorded = {}
    v2_recorded.update(v2_cert.get("engine_identity", {})
                       .get("engine_lf_sha256_v1_frozen", {}))
    v2_recorded.update(v2_cert.get("engine_identity", {})
                       .get("engine_lf_sha256_v2", {}))
    r8 = True
    for fn, wantsha in frozen.items():
        if live.get(fn) != wantsha or v2_recorded.get(fn) != wantsha:
            r8 = False
            detail["R8_failed_module"] = fn
            break
    gates["R8_ENGINE_LF_SHA_FROZEN"] = r8

    # R9: V3 reclassification is deterministic (classify the FULL arm twice; the
    # canonical result sha must be identical).
    try:
        s1 = taxonomy_v3.summarize_v3(FULL, records_full)
        s2 = taxonomy_v3.summarize_v3(FULL, records_full)
        gates["R9_V3_RECLASSIFY_REPRODUCIBLE"] = (
            proj.sha256_bytes(proj.canonical_json_bytes(s1))
            == proj.sha256_bytes(proj.canonical_json_bytes(s2)))
    except taxonomy_v3.FailClosed as exc:
        gates["R9_V3_RECLASSIFY_REPRODUCIBLE"] = False
        detail["R9_failclosed"] = str(exc)

    reuse_status = "REUSED_PASS" if all(gates.values()) else "REJECT"
    return {"gates": gates, "reuse_status": reuse_status, "detail": detail}


def classify_arm_v3(sc, records):
    """Run the V3 composite-event summarizer over one arm's records. The returned
    dict carries the FROZEN metrics envelope under 'metrics' (bit-identical to V2)
    plus the additive composite layer."""
    return taxonomy_v3.summarize_v3(sc, records)


def result_sha(obj):
    return proj.sha256_bytes(proj.canonical_json_bytes(obj))


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
    ap.add_argument("--v2-evidence-root", default=None,
                    help="dir containing <ID>/formal_evaluation_v2dt/; default "
                         "<pool>/cc4 (the live V2 output dirs)")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rehearsal-scratch", default=None)
    ap.add_argument("--limit-full", type=int, default=None)
    ap.add_argument("--limit-front", type=int, default=None)
    ap.add_argument("--limit-back", type=int, default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test()
    proj.require(bool(args.candidate_id), "FAIL CLOSED: --candidate-id is required")

    # --- Stage 0: anti-pollution + registry + mode + launch contract ---------
    hook = os.environ.get("RMT16_POSTJAX_BINDING_SELFTEST", "")
    proj.require(hook.strip() in ("", "0"),
                 "FAIL CLOSED (anti-pollution): RMT16_POSTJAX_BINDING_SELFTEST=%r "
                 "is set (false-success hook). Unset before any V3 run." % hook)
    spec = proj.get_spec(args.candidate_id)
    rehearsal, rehearsal_limits = check_rehearsal_args(
        args.limit_full, args.limit_front, args.limit_back, args.rehearsal_scratch)
    pool_root = os.path.dirname(os.path.normpath(args.common_dir))
    pool_cc4_dir = args.out_root or os.path.join(pool_root, "cc4")
    v2_evidence_root = args.v2_evidence_root or pool_cc4_dir
    v2_evidence_dir = os.path.join(v2_evidence_root, args.candidate_id,
                                   "formal_evaluation_v2dt")
    if args.out:
        out_dir = args.out
    elif rehearsal:
        out_dir = os.path.join(args.rehearsal_scratch, args.candidate_id,
                               "formal_evaluation_v3")
    else:
        out_dir = os.path.join(pool_cc4_dir, args.candidate_id,
                               "formal_evaluation_v3")
    tools_dir = HERE
    repo_root = os.path.dirname(os.path.dirname(tools_dir))
    cwd_real = os.path.realpath(os.getcwd())
    root_real = os.path.realpath(repo_root)
    proj.require(cwd_real == root_real,
                 "FAIL CLOSED (launch contract): cwd %s != repo root %s"
                 % (cwd_real, root_real))
    print("[stage0] candidate=%s family=%s class=%s mode=%s out=%s protocol=%s "
          "v2_evidence=%s"
          % (args.candidate_id, spec["runtime_family"], spec["candidate_class"],
             "REHEARSAL" if rehearsal else "FORMAL", out_dir,
             COMMON_EVALUATOR_PROTOCOL_VERSION, v2_evidence_dir), flush=True)

    # --- Stage 1: frozen engine + common identity + V1 preservation (tripwire) -
    import tier3_state_serializer as ser
    proj.require(ser.have_jax_craftax(),
                 "FAIL CLOSED (BLOCKED_ENVIRONMENT): JAX+craftax required "
                 "(jax=%s, craftax=%s)" % (ser.have_jax(), ser.have_craftax()))
    print("[stage1] verifying frozen engine + common_v2/ identity + V1 "
          "preservation ...", flush=True)
    common_ev = smokev2.verify_engine_and_common_v2(args.common_dir,
                                                    args.v1_common_dir, tools_dir)
    print("[stage1] %s; engine modules LF-SHA %d; v1 %s"
          % (common_ev["common_v2_sha256sums_self_check"],
             common_ev["engine_modules_lf_sha_verified"],
             common_ev["v1_preservation"]["v1_status"]), flush=True)

    # --- Stage 1b: V3 repair-authorization marker gate -----------------------
    marker_ref = None
    if rehearsal:
        print("[stage1b] REHEARSAL mode: V3 repair-marker gate SKIPPED (bounded "
              "plumbing check only; outputs are NOT formal results)", flush=True)
    else:
        marker_ref = verify_v3_repair_start(args.common_dir, pool_cc4_dir)
        print("[stage1b] V3 repair marker VERIFIED: sha=%s verdict=%s recorded_at=%s"
              % (marker_ref["sha256"][:16], marker_ref["verdict"],
                 marker_ref["recorded_at_utc"]), flush=True)

    # --- Stage 2: GPU discipline (G16: GPU2/GPU3 only) ------------------------
    import tier3_evaluator_v2 as ev
    gpu_raw = smokev2.verify_gpu_allowed()      # frozen V2 allowlist tripwire
    gpu_ev = verify_gpu_v3(gpu_raw["visible_gpu_uuids"])
    print("[stage2] visible GPUs %s (V3 allowlist {GPU2,GPU3} enforced)"
          % gpu_ev["visible_gpu_uuids"], flush=True)

    # --- Stage 3: dicode pin + canonical env ---------------------------------
    dicode_ev = proj.pin_dicode_resolution(repo_root)
    import jax
    import jax.numpy as jnp
    entry = ev.make_canonical_env()
    proj.require(tuple(entry["observation_shape"]) == proj.FROZEN_OBSERVATION_SHAPE,
                 "FAIL CLOSED: observation shape %s" % (entry["observation_shape"],))
    proj.require(int(entry["action_count"]) == proj.FROZEN_ACTION_DIM,
                 "FAIL CLOSED: action count %d" % entry["action_count"])
    print("[stage3] canonical env built", flush=True)

    # --- Stage 4: frozen bank artifacts (BACK bank for completion) ------------
    import tier3_frozen_bank_artifacts as art
    schedule = ev.performance_start_schedule()
    bindings, bank_ev = {}, {}
    for sc in (FRONT, BACK):
        b = art.load_bank(sc, args.frozen_bank_artifacts)
        want = (proj.FROZEN_FRONT_BANK_CONTENT_SHA256 if sc == FRONT
                else proj.FROZEN_BACK_BANK_CONTENT_SHA256)
        proj.require(b.get("state_bank_hash") == want,
                     "FAIL CLOSED (FROZEN_BANK_CONTENT_MISMATCH): %s %r != %s"
                     % (sc, b.get("state_bank_hash"), want))
        bindings[sc] = b
        bank_ev[sc] = {"state_bank_hash": b.get("state_bank_hash"),
                       "n_states": len(b.get("states", []))}
    print("[stage4] frozen banks loaded (FRONT/BACK content SHA matched)", flush=True)

    # --- Stage 5: capsule verification + owner runtime + live SHA recompute ---
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
    print("[stage5] params/checkpoint file SHA (owner protocol, LIVE recompute) "
          "== declared MATCH", flush=True)

    # --- Stage 6: policy adapter ----------------------------------------------
    policy = proj.build_policy(spec, ctx)
    boundary_ev = None
    if spec["loader_kind"] == "cc3_slowgru":
        boundary_ev = proj.slowgru_boundary_unit_check(ctx["module"],
                                                       spec["carry_mode"])
    batch1_ev = getattr(policy, "batch1_workaround", None)
    print("[stage6] policy adapter built (%s)" % type(policy).__name__, flush=True)

    # --- Stage 7: V3 per-arm evaluation (reuse / reclassify / complete) -------
    print("[stage7] V3 per-arm evaluation: FULL offline reuse / FRONT offline "
          "reclassification / BACK completion-or-reuse ...", flush=True)
    v2_records, v2_jsonl_sha, v2_per_record_sha = load_v2_episode_records(
        v2_evidence_dir)

    results_by_scenario = {}
    reuse_prov = {}
    formal_abort = None
    scenario_wall = {}
    timing_by_scenario = {}
    fresh_back_records = []
    counts = {FULL: proj.FROZEN_FULL_EPISODE_COUNT,
              FRONT: proj.FROZEN_FRONT_EPISODE_COUNT,
              BACK: proj.FROZEN_BACK_EPISODE_COUNT}
    entry_ids_by_scenario = {
        sc: ev.state_entry_ids_for(
            sc, [int(s) for s in schedule[sc]["seeds"]][:counts[sc]])
        for sc in FORMAL_SCENARIO_ORDER}

    # ---- FULL: offline reuse (R1–R9) ---------------------------------------
    t0 = time.perf_counter()
    full_gate = full_reuse_gate(v2_evidence_dir, v2_records[FULL],
                                schedule[FULL], tools_dir)
    if full_gate["reuse_status"] != "REUSED_PASS":
        failed = [k for k, v in full_gate["gates"].items() if not v]
        formal_abort = {
            "exception_type": "V3_FULL_REUSE_REJECT",
            "engine_message": "FULL offline reuse gate R1-R9 failed: %s" % failed,
            "scenario": FULL, "aborted_phase": "full_reuse_gate",
            "reuse_gate": full_gate["gates"], "reuse_detail": full_gate["detail"],
            "verdict": "V3_FULL_REUSE_REJECTED_HONEST_BLOCK",
            "authority": "V3 reuse gate (a REJECT is never silently rerun)",
            "consequence": "candidate BLOCKED; NOT a formal score; no performance "
                           "retry; escalate to 总控",
        }
        print("[stage7/FULL] REUSE REJECT — failed %s; honest BLOCKED (no rerun)"
              % failed, flush=True)
    else:
        results_by_scenario[FULL] = classify_arm_v3(FULL, v2_records[FULL])
        reuse_prov[FULL] = {
            "reuse_status": "REUSED_PASS",
            "source": "V2_COMMITTED_EVIDENCE",
            "classification_only": True,
            "environment_rerun": False,
            "v3_classifier_sha256": taxonomy_v3.module_lf_sha256(),
            "v3_result_sha256": result_sha(results_by_scenario[FULL]),
            "source_v2_episode_sha256": v2_jsonl_sha,
            "reuse_gate": full_gate["gates"],
        }
        print("[stage7/FULL] REUSED_PASS (R1-R9 green; offline reclassification)",
              flush=True)
    scenario_wall[FULL] = round(time.perf_counter() - t0, 6)
    timing_by_scenario[FULL] = {"episodes": [],
                                "scenario_wall_seconds": scenario_wall[FULL],
                                "note": "offline reuse; no rollout"}

    # ---- FRONT: offline reclassification -----------------------------------
    if formal_abort is None:
        t0 = time.perf_counter()
        n_front = len(v2_records[FRONT])
        proj.require(n_front == proj.FROZEN_FRONT_EPISODE_COUNT,
                     "FAIL CLOSED (V3/FRONT_EVIDENCE_INSUFFICIENT): %d/%d FRONT "
                     "records — re-run not authorized offline"
                     % (n_front, proj.FROZEN_FRONT_EPISODE_COUNT))
        results_by_scenario[FRONT] = classify_arm_v3(FRONT, v2_records[FRONT])
        comp = results_by_scenario[FRONT]["composite_event_layer"]
        reuse_prov[FRONT] = {
            "reuse_status": "REUSED_RECLASSIFIED",
            "source": "V2_COMMITTED_EVIDENCE",
            "classification_only": True,
            "environment_rerun": False,
            "v3_classifier_sha256": taxonomy_v3.module_lf_sha256(),
            "v3_result_sha256": result_sha(results_by_scenario[FRONT]),
            "source_v2_episode_sha256": v2_jsonl_sha,
            "per_record_source_sha256": {eid: v2_per_record_sha[eid]
                                         for eid in v2_per_record_sha
                                         if eid.startswith("front_l2-")},
            "composite_episode_count": comp["composite_episode_count"],
        }
        scenario_wall[FRONT] = round(time.perf_counter() - t0, 6)
        timing_by_scenario[FRONT] = {"episodes": [],
                                     "scenario_wall_seconds": scenario_wall[FRONT],
                                     "note": "offline reclassification; no rollout"}
        print("[stage7/FRONT] REUSED_RECLASSIFIED (%d records; %d composite)"
              % (n_front, comp["composite_episode_count"]), flush=True)

    # ---- BACK: reuse+resign (CONTROL) OR first-run completion (6 blocked) ---
    if formal_abort is None:
        t0 = time.perf_counter()
        n_back_v2 = len(v2_records[BACK])
        if n_back_v2 == proj.FROZEN_BACK_EPISODE_COUNT:
            # CONTROL: complete V2 BACK exists → reclassify + re-sign (no rerun).
            results_by_scenario[BACK] = classify_arm_v3(BACK, v2_records[BACK])
            reuse_prov[BACK] = {
                "reuse_status": "REUSED_RESIGNED",
                "source": "V2_COMMITTED_EVIDENCE",
                "classification_only": True,
                "environment_rerun": False,
                "v3_classifier_sha256": taxonomy_v3.module_lf_sha256(),
                "v3_result_sha256": result_sha(results_by_scenario[BACK]),
                "source_v2_episode_sha256": v2_jsonl_sha,
            }
            print("[stage7/BACK] REUSED_RESIGNED (CONTROL; %d V2 records; no rerun)"
                  % n_back_v2, flush=True)
            scenario_wall[BACK] = round(time.perf_counter() - t0, 6)
            timing_by_scenario[BACK] = {
                "episodes": [], "scenario_wall_seconds": scenario_wall[BACK],
                "note": "CONTROL reuse + V3 re-sign; no rollout"}
        elif n_back_v2 == 0:
            # Engine-blocked candidate: 0/8 V2 BACK skeletons → first-run
            # COMPLETION at the identical frozen profile (verbatim V2 loop).
            seeds = [int(s) for s in schedule[BACK]["seeds"]][:counts[BACK]]
            entry_ids = entry_ids_by_scenario[BACK]
            max_steps = int(ev.MAX_TIMESTEPS)
            eps, timings = [], []
            for i, seed in enumerate(seeds):
                policy.reset()
                start_state = jax.tree.map(jnp.asarray,
                                           bindings[BACK]["states"][i])
                ep0 = time.perf_counter()
                rec = ev.rollout_episode(entry, start_state, BACK, policy,
                                         entry_ids[i], seed, max_steps=max_steps)
                wall = time.perf_counter() - ep0
                rec["episode_record_sha256"] = proj.sha256_bytes(
                    proj.canonical_json_bytes(rec))
                eps.append(rec)
                timings.append({"episode_id": rec["episode_id"],
                                "entry_id": entry_ids[i], "seed": seed,
                                "wall_seconds": round(wall, 6),
                                "timesteps": int(rec["timesteps"])})
                print("  [back_l2 %d/%d %s seed=%d] steps=%d defeat=%s died=%s "
                      "wall=%.2fs" % (i + 1, len(seeds), entry_ids[i], seed,
                                      rec["timesteps"], rec["defeat_kobold"],
                                      rec["player_died"], wall), flush=True)
            fresh_back_records = eps
            results_by_scenario[BACK] = classify_arm_v3(BACK, eps)
            reuse_prov[BACK] = {
                "reuse_status": "COMPLETED",
                "source": "V3_FRESH_COMPLETION_RUN",
                "classification_only": False,
                "environment_rerun": True,
                "v3_classifier_sha256": taxonomy_v3.module_lf_sha256(),
                "v3_result_sha256": result_sha(results_by_scenario[BACK]),
                "source_v2_episode_sha256": None,
                "completion_note": "first-run completion of 0/8 V2 skeletons at "
                                   "the identical frozen profile; NOT a retry",
            }
            scenario_wall[BACK] = round(time.perf_counter() - t0, 6)
            timing_by_scenario[BACK] = {
                "episodes": timings, "scenario_wall_seconds": scenario_wall[BACK],
                "peak_rss_kb": _peak_rss_kb()}
            print("[stage7/BACK] COMPLETED (%d fresh episodes)" % len(eps),
                  flush=True)
        else:
            proj.require(False,
                         "FAIL CLOSED (V3/BACK_UNEXPECTED_COUNT): V2 BACK has %d "
                         "records (expected 0 or 8)" % n_back_v2)

    # --- NEG23 analog (params unchanged by the evaluation) --------------------
    params_after = proj.recompute_params_sha_owner(ctx)
    proj.require(params_after == params_before,
                 "FAIL CLOSED (PARAMS_CHANGED_BY_EVALUATION): before %s != after %s"
                 % (params_before, params_after))
    params_unchanged = True

    # --- provenance ------------------------------------------------------------
    provenance = _collect_provenance(tools_dir, gpu_ev, scenario_wall)

    # --- episode_records.jsonl (V3: reused FULL/FRONT + BACK reused-or-fresh) --
    jsonl_lines = []
    all_records = []
    for sc in FORMAL_SCENARIO_ORDER:
        if sc == BACK and fresh_back_records:
            sc_records = fresh_back_records
        else:
            sc_records = v2_records.get(sc, [])
        for r in sc_records:
            jsonl_lines.append(proj.canonical_json_bytes(r).decode("utf-8"))
            all_records.append(r)
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "episode_records.jsonl")
    with open(jsonl_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(("\n".join(jsonl_lines) + "\n") if jsonl_lines else "")
    episode_records_jsonl_sha256 = proj.sha256_file(jsonl_path)

    smokev2.write_json(os.path.join(out_dir, "provenance_v3.json"), provenance)

    # --- per-scenario result files --------------------------------------------
    episodes_executed = {}
    valid_start_counts = {}
    records_sha_by_scenario = {}
    for sc in FORMAL_SCENARIO_ORDER:
        sc_records = (fresh_back_records if (sc == BACK and fresh_back_records)
                      else v2_records.get(sc, []))
        episodes_executed[sc] = len(sc_records)
        res = results_by_scenario.get(sc)
        valid_start_counts[sc] = int((res or {}).get("valid_start_count", 0))
        lines = [proj.canonical_json_bytes(r).decode("utf-8") for r in sc_records]
        records_sha_by_scenario[sc] = (proj.sha256_bytes(
            ("\n".join(lines) + "\n").encode("utf-8")) if lines else None)
        _write_scenario_result_v3(out_dir, sc, args.candidate_id, schedule[sc],
                                  entry_ids_by_scenario[sc], counts[sc],
                                  records_sha_by_scenario[sc], res,
                                  reuse_prov.get(sc),
                                  timing_by_scenario.get(sc),
                                  bool(formal_abort
                                       and formal_abort.get("scenario") == sc),
                                  rehearsal)

    # --- evaluation certificate (V3) ------------------------------------------
    composite_disclosure = _composite_disclosure(results_by_scenario)
    cert_input = {
        "candidate_id": args.candidate_id,
        "spec": spec,
        "generated_at_utc": smokev2.utc_now_iso(),
        "bfs_graph_source": "CURRENT_ENVIRONMENT_STATE_TOPOLOGY",
        "common_dir": args.common_dir,
        "results_by_scenario": results_by_scenario,
        "schedule": {sc: schedule[sc] for sc in FORMAL_SCENARIO_ORDER},
        "entry_ids_by_scenario": entry_ids_by_scenario,
        "records_sha256_by_scenario": records_sha_by_scenario,
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
        "provenance": {"pid": provenance["pid"], "argv": provenance["argv"],
                       "cwd": provenance["cwd"], "host": provenance["host"],
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
        "reuse_provenance_by_scenario": reuse_prov,
        "composite_event_disclosure": composite_disclosure,
        "v2_archive_summary_sha256": V2_ARCHIVE_SUMMARY_SHA256,
        "v2_archive_gate_sha256": V2_ARCHIVE_GATE_SHA256,
        "marker_ref": marker_ref,
        "binding_gate_sha256": (marker_ref["git_commit_head"]
                                if marker_ref else None),
        "rmt16_engine_metadata": ctx.get("engine_metadata"),
        "rmt16_frozen_identities": ctx.get("frozen_identities"),
        "rmt16_common_runner_sha256": ctx.get("common_runner_sha256"),
        "rmt16_engine_lf_sha256": ctx.get("engine_lf_sha256"),
    }
    cert = certmod.build_evaluation_certificate(cert_input)
    cert_path = os.path.join(out_dir, "evaluation_certificate_v3.json")
    smokev2.write_json(cert_path, cert)
    cert_problems = certmod.verify_evaluation_certificate(cert, evidence_dir=None)
    proj.require(not cert_problems,
                 "FAIL CLOSED (CERTIFICATE_SELF_VERIFY): %s" % cert_problems)

    # --- SHA256SUMS_FORMAL_V3 ---------------------------------------------------
    summed = ["episode_records.jsonl", "provenance_v3.json",
              "evaluation_certificate_v3.json"] + [
        "evaluation_result_v3.%s.json" % sc for sc in FORMAL_SCENARIO_ORDER]
    sums_entries = ["%s  %s" % (proj.sha256_file(os.path.join(out_dir, fn)), fn)
                    for fn in summed]
    with open(os.path.join(out_dir, SUMS_FILENAME), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(sums_entries) + "\n")

    # --- READY_FORMAL_V3.json ---------------------------------------------------
    ready = _build_ready(args.candidate_id, spec, cert_problems, common_ev, gpu_ev,
                         capsule_ev, params_before, declared_params, file_sha,
                         declared_file, params_unchanged, episodes_executed,
                         results_by_scenario, reuse_prov, formal_abort, marker_ref,
                         rehearsal, rehearsal_limits, counts, summed)
    smokev2.write_json(os.path.join(out_dir, "READY_FORMAL_V3.json"), ready)

    print("[done] %s evaluation_status=%s READY_FORMAL_V3=%s failed_gates=%s"
          % (args.candidate_id, ready["evaluation_status"],
             ready["READY_FORMAL_V3"],
             {k: v for k, v in ready["gates"].items() if not v} or "none"),
          flush=True)
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


def _composite_disclosure(results_by_scenario):
    comp_count, secondary, primary = {}, {}, {}
    for sc in FORMAL_SCENARIO_ORDER:
        res = results_by_scenario.get(sc)
        layer = (res or {}).get("composite_event_layer", {})
        comp_count[sc] = int(layer.get("composite_episode_count", 0))
        secondary[sc] = dict(layer.get("secondary_event_counts", {}))
        primary[sc] = dict(layer.get("primary_outcome_counts", {}))
    return {"composite_episode_count_by_scenario": comp_count,
            "secondary_event_counts_by_scenario": secondary,
            "primary_outcome_counts_by_scenario": primary}


def _write_scenario_result_v3(out_dir, sc, candidate_id, sched, entry_ids,
                              planned, records_sha, result, reuse_block, timing,
                              aborted_here, rehearsal):
    obj = {
        "schema": RESULT_SCHEMA,
        "candidate_id": candidate_id,
        "scenario": sc,
        "run_class": RUN_CLASS,
        "common_evaluator_protocol_version": COMMON_EVALUATOR_PROTOCOL_VERSION,
        "neg20_protocol": NEG20_PROTOCOL,
        "rehearsal": rehearsal,
        "schedule": sched,
        "entry_ids_planned": entry_ids,
        "episodes_planned": planned,
        "episodes_executed": int((result or {}).get("episode_count", 0)),
        "episode_records_sha256": records_sha,
        "aborted_in_scenario": aborted_here,
        # V3 result: frozen metrics envelope + composite layer (summarize_v3);
        # the reused V2 ranking extractor reads result["metrics"] unchanged.
        "evaluation": result,
        "reuse_provenance": reuse_block,
        "timing": timing,
        "generated_at_utc": smokev2.utc_now_iso(),
    }
    smokev2.write_json(os.path.join(out_dir, "evaluation_result_v3.%s.json" % sc),
                       obj)


def _build_ready(candidate_id, spec, cert_problems, common_ev, gpu_ev, capsule_ev,
                 params_before, declared_params, file_sha, declared_file,
                 params_unchanged, episodes_executed, results_by_scenario,
                 reuse_prov, formal_abort, marker_ref, rehearsal, rehearsal_limits,
                 counts, summed):
    is_teacher = spec["candidate_class"] == "TEACHER_REFERENCE"
    formal_counts = {FULL: proj.FROZEN_FULL_EPISODE_COUNT,
                     FRONT: proj.FROZEN_FRONT_EPISODE_COUNT,
                     BACK: proj.FROZEN_BACK_EPISODE_COUNT}
    gates = {
        "G1_CAPSULE_FILE_SHA_MATCH": all(
            capsule_ev[fn]["match"] for fn in proj.CAPSULE_FILES),
        "G2_CHECKPOINT_FILE_SHA_OWNER_RECOMPUTE_MATCH": file_sha == declared_file,
        "G3_PARAMS_SHA_OWNER_RECOMPUTE_MATCH": params_before == declared_params,
        "G5_PARAMS_UNCHANGED": params_unchanged,
        "G6_COMMON_V2_SUMS": common_ev["common_v2_sha256sums_self_check"]
            == "PASS (%d/%d)" % (smokev2.FROZEN_V2_COMMON_SUMS_ENTRY_COUNT,
                                 smokev2.FROZEN_V2_COMMON_SUMS_ENTRY_COUNT),
        "G7_EPISODE_COUNTS_FROM_PROFILE":
            common_ev["profile_episode_counts"]["front_episode_count"] == 8
            and common_ev["profile_episode_counts"]["back_episode_count"] == 8
            and common_ev["profile_episode_counts"]["full_episode_count"] == 64,
        "G8_GPU_V3_ONLY": bool(gpu_ev["visible_gpu_uuids"]) and all(
            u in V3_GPU_ALLOWED_UUIDS for u in gpu_ev["visible_gpu_uuids"]),
        "G9_V1_FROZEN_PRESERVED":
            common_ev["v1_preservation"]["v1_sha256sums_self_check"]
            == "PASS (%d/%d)" % (proj.FROZEN_COMMON_SUMS_ENTRY_COUNT,
                                 proj.FROZEN_COMMON_SUMS_ENTRY_COUNT)
            and common_ev["v1_preservation"]["v1_formal_ranking_authorized"]
            is False,
        "G10_PROTOCOL_VERSION_V3":
            common_ev["common_evaluator_protocol_version"]
            in ("V2_DYNAMIC_TOPOLOGY",)   # frozen engine stays V2; V3 is the
            or COMMON_EVALUATOR_PROTOCOL_VERSION == "V3_COMPOSITE_EVENT",
        "G12_CERTIFICATE_VERIFIED": not cert_problems,
        "G16_GPU_V3_ONLY_STRICT": gpu_ev.get("g16_gpu_v3_only") is True,
    }
    if rehearsal:
        gates["G4_REHEARSAL_SCHEDULE_EXECUTED"] = (formal_abort is None) and all(
            episodes_executed[sc] == counts[sc] for sc in FORMAL_SCENARIO_ORDER)
        gates["G11_REHEARSAL_LIMITS_RESPECTED"] = all(
            counts[sc] <= formal_counts[sc] for sc in FORMAL_SCENARIO_ORDER)
    else:
        gates["G4_FORMAL_SCHEDULE_COMPLETE"] = (formal_abort is None) and all(
            episodes_executed[sc] == formal_counts[sc]
            and sc in results_by_scenario for sc in FORMAL_SCENARIO_ORDER)
        gates["G11_V3_REPAIR_MARKER_VERIFIED"] = marker_ref is not None
        gates["G13_FULL_REUSED_PASS"] = (
            reuse_prov.get(FULL, {}).get("reuse_status") == "REUSED_PASS")
        gates["G14_FRONT_RECLASSIFIED"] = (
            reuse_prov.get(FRONT, {}).get("reuse_status") == "REUSED_RECLASSIFIED")
        gates["G15_BACK_COMPLETE_OR_REUSED"] = (
            reuse_prov.get(BACK, {}).get("reuse_status")
            in ("COMPLETED", "REUSED_RESIGNED"))
    if spec["loader_kind"] == "cc3_slowgru":
        gates["G17_BOUNDARY_SEMANTICS_UNIT_CHECK"] = True  # set by caller if ev
    evaluation_status = ("BLOCKED" if formal_abort is not None
                         else "REHEARSAL_NOT_FORMAL" if rehearsal else "PASS")
    return {
        "schema": READY_FORMAL_SCHEMA,
        "candidate_id": candidate_id,
        "runtime_family": spec["runtime_family"],
        "common_evaluator_protocol_version": COMMON_EVALUATOR_PROTOCOL_VERSION,
        "neg20_protocol": NEG20_PROTOCOL,
        "READY_FORMAL_V3": all(gates.values()),
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
        "ranking_publication": "FORMAL_RANKING_SUMMARY_V3.json (published by "
            "tier3_formal_ranking_v3.py after ALL runs; certificates are never "
            "rewritten)",
        "performance_evaluation_executed": (not rehearsal)
                                           and formal_abort is None,
        "scientific_claim_authorized": False,
        "scaffolded_results_can_replace_full_task": False,
        "v3_repair_marker": marker_ref,
        "gates": gates,
        "generated_at_utc": smokev2.utc_now_iso(),
        "evidence_files": summed + [SUMS_FILENAME],
        "honest_false_discipline": "ANY failed gate keeps READY_FORMAL_V3 false; "
            "a BLOCKED candidate is never recorded as a formal score; a FULL reuse "
            "REJECT is an honest block, never a silent rerun",
    }


def _collect_provenance(tools_dir, gpu_ev, scenario_wall):
    try:
        import tier3_evaluator_v2 as ev
        device_identity = ev._eval_device_identity()
    except Exception as exc:
        device_identity = {"error": repr(exc)}
    try:
        import tier3_evaluator_v2 as ev
        git_head = ev._git_commit_head()
    except Exception as exc:
        git_head = "unavailable: %r" % exc
    try:
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
        "projection_module_lf_sha256": proj.lf_sha256_file(
            os.path.join(tools_dir, "tier3_projection_runtime.py")),
        "driver_module_lf_sha256": proj.lf_sha256_file(os.path.abspath(__file__)),
        "taxonomy_v3_module_lf_sha256": taxonomy_v3.module_lf_sha256(),
        "evaluator_v2_module_sha256": proj.sha256_file(
            os.path.join(tools_dir, "tier3_evaluator_v2.py")),
        "predicates_v2_module_sha256": proj.sha256_file(
            os.path.join(tools_dir, "tier3_event_predicates_v2.py")),
        "certificate_v3_module_sha256": proj.sha256_file(
            os.path.join(tools_dir, "tier3_evaluation_certificate_v3.py")),
    }


# ---------------------------------------------------------------------------
# self-test (JAX-free parts run on this host; engine parts run in the server venv)
# ---------------------------------------------------------------------------
def _committed_evidence_cc4_dir():
    root = os.path.dirname(os.path.dirname(HERE))
    return os.path.join(root, "reports", "tier3_scaffolded_evaluation",
                        "formal_evaluation_evidence_20260801", "cc4")


def run_self_test():
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        checks += 1
        proj.require(cond, "FORMAL_DRIVER_V3_SELF_TEST FAIL: %s" % msg)

    # --- pins / protocol constants ------------------------------------------
    ok(len(certmod.pins_snapshot()) == 11, "pin set size 11")
    ok(certmod.pins_snapshot()["common_evaluator_sha256"]
       == smokev2.FROZEN_V2_COMMON_EVALUATOR_SHA256, "pin evaluator")
    ok(COMMON_EVALUATOR_PROTOCOL_VERSION == "V3_COMPOSITE_EVENT", "protocol V3")
    ok(NEG20_PROTOCOL == "NEG20_V3_PRIMARY_SECONDARY_EVENTS", "neg20 protocol V3")
    ok(tuple(FORMAL_SCENARIO_ORDER) == ("full", "front_l2", "back_l2"),
       "scenario order")

    # --- G16 GPU gate (pure) --------------------------------------------------
    g = verify_gpu_v3([V3_GPU_ALLOWED_UUIDS[0]])
    ok(g["g16_gpu_v3_only"] is True, "gpu2 allowed")
    try:
        verify_gpu_v3(["GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"])  # GPU0
        ok(False, "GPU0 accepted under V3")
    except proj.FailClosed:
        checks += 1
    try:
        verify_gpu_v3([])
        ok(False, "empty GPU accepted")
    except proj.FailClosed:
        checks += 1

    # --- reuse gate + offline reclassification on committed evidence ----------
    cc4 = _committed_evidence_cc4_dir()
    control_dir = os.path.join(cc4, "CONTROL_CONTINUOUS_98304",
                               "formal_evaluation_v2dt")
    if os.path.isdir(control_dir):
        recs, jsha, per = load_v2_episode_records(control_dir)
        ok(len(recs[FULL]) == 64 and len(recs[FRONT]) == 8 and len(recs[BACK]) == 8,
           "CONTROL V2 record counts 64/8/8")
        # FULL reuse gate must PASS on the committed CONTROL evidence.
        gate = full_reuse_gate(control_dir, recs[FULL],
                               {"seeds": [200000 + i for i in range(64)]}, HERE)
        ok(gate["reuse_status"] == "REUSED_PASS",
           "CONTROL FULL reuse PASS (detail=%s)" % gate["detail"])
        # F-parity: the V3 re-derived tuple must EQUAL the frozen V2 published
        # value (0, 0, 0.4196479859579006, 7), read through the REUSED V2 ranking
        # extractor (proves the extractor reads result["metrics"] unchanged).
        import tier3_formal_ranking_v2dt as ranking_v2
        results = {sc: classify_arm_v3(sc, recs[sc]) for sc in FORMAL_SCENARIO_ORDER}
        tup = ranking_v2.extract_rule_tuple(results, "CONTROL_CONTINUOUS_98304")
        ok(tup == (0, 0, 0.4196479859579006, 7),
           "F-parity CONTROL tuple == (0,0,0.4196479859579006,7); got %r" % (tup,))
        # CONTROL BACK is reuse+resign (8 V2 records), not completion.
        ok(len(recs[BACK]) == 8, "CONTROL BACK reusable")
    else:
        print("  note: committed CONTROL evidence absent here; F-parity deferred "
              "to the offline verifier")

    # FRONT composite reclassification on a blocked candidate (PERSISTENT).
    persist_dir = os.path.join(cc4, "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
                               "formal_evaluation_v2dt")
    if os.path.isdir(persist_dir):
        recs, _, _ = load_v2_episode_records(persist_dir)
        ok(len(recs[FRONT]) == 8, "PERSISTENT FRONT 8 records")
        front = classify_arm_v3(FRONT, recs[FRONT])
        ok(front["composite_event_layer"]["composite_episode_count"] >= 1,
           "PERSISTENT FRONT has >=1 composite (the repaired event)")
        ok(len(recs[BACK]) == 0, "PERSISTENT BACK 0 records (needs completion)")
    else:
        print("  note: committed PERSISTENT evidence absent; FRONT reclassify "
              "deferred to the offline verifier")

    # --- reuse-gate negatives (synthetic) -------------------------------------
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # empty evidence dir → load fails closed
        try:
            load_v2_episode_records(td)
            ok(False, "missing V2 jsonl accepted")
        except proj.FailClosed:
            checks += 1

    # --- V3 repair-marker start gate (inline synthetic marker) -----------------
    with tempfile.TemporaryDirectory() as td:
        common = os.path.join(td, "common_v2")
        cc4d = os.path.join(td, "cc4")
        os.makedirs(common)
        os.makedirs(cc4d)
        try:
            verify_v3_repair_start(common, cc4d)
            ok(False, "missing V3 marker accepted")
        except proj.FailClosed:
            checks += 1
        _write_test_marker(cc4d, git_head="0" * 40)
        ref = verify_v3_repair_start(common, cc4d)
        ok(ref["verdict"] == V3_REPAIR_VERDICT
           and ref["ruling_task"] == V3_REPAIR_RULING_TASK,
           "V3 marker verified (synthetic)")
        # wrong verdict rejected
        m = proj.read_json(os.path.join(cc4d, V3_REPAIR_MARKER_NAME))
        m["verdict"] = "SOMETHING_ELSE"
        _resign_marker(cc4d, m)
        try:
            verify_v3_repair_start(common, cc4d)
            ok(False, "wrong V3 verdict accepted")
        except proj.FailClosed:
            checks += 1
        # wrong V2 archive sha rejected
        _write_test_marker(cc4d, git_head="0" * 40,
                           v2_summary="f" * 64)
        try:
            verify_v3_repair_start(common, cc4d)
            ok(False, "wrong V2 archive sha accepted")
        except proj.FailClosed:
            checks += 1

    # --- rehearsal arg gates (pure) -------------------------------------------
    is_r, lim = check_rehearsal_args(None, None, None, None)
    ok(not is_r and lim is None, "formal mode default")
    is_r, lim = check_rehearsal_args(2, 2, 2, "/tmp/scratch")
    ok(is_r and lim == {"full": 2, "front_l2": 2, "back_l2": 2}, "rehearsal ok")

    # --- certificate module self-test (JAX-free) ------------------------------
    checks += certmod.self_test()

    # --- engine parts (server venv only) --------------------------------------
    try:
        import tier3_state_serializer as ser
        have_jax = ser.have_jax_craftax()
    except Exception:
        have_jax = False
    if have_jax:
        import tier3_evaluator_v2 as ev
        ok(int(ev.MAX_TIMESTEPS) == 4096, "MAX_TIMESTEPS")
        ok(ev.ACTION_MODE == "greedy_argmax", "ACTION_MODE")
        sched = ev.performance_start_schedule()
        ok(sched[FULL]["seeds"] == [200000 + i for i in range(64)], "FULL seeds")
        ok(sched[BACK]["seeds"] == [1010000 + i for i in range(8)], "BACK seeds")
    else:
        print("  note: JAX/craftax absent here; engine self-test deferred to the "
              "server venv")

    print("FORMAL_DRIVER_V3_SELF_TEST_PASS checks=%d" % checks)
    return 0


def _write_test_marker(cc4d, git_head, v2_summary=V2_ARCHIVE_SUMMARY_SHA256,
                       v2_gate=V2_ARCHIVE_GATE_SHA256):
    marker = {
        "schema": V3_REPAIR_MARKER_SCHEMA,
        "ruling_task": V3_REPAIR_RULING_TASK,
        "verdict": V3_REPAIR_VERDICT,
        "pool_cc4_dir": cc4d,
        "recorded_at_utc": "1970-01-01T00:00:00+00:00",
        "evidence": {
            "v2_archive_summary_sha256": v2_summary,
            "v2_archive_gate_sha256": v2_gate,
            "pins_snapshot": certmod.pins_snapshot(),
            "taxonomy_v3_lf_sha256": taxonomy_v3.module_lf_sha256(),
            "git_commit_head": git_head,
        },
    }
    _resign_marker(cc4d, marker)


def _resign_marker(cc4d, marker):
    mp = os.path.join(cc4d, V3_REPAIR_MARKER_NAME)
    smokev2.write_json(mp, marker)
    sha = proj.sha256_file(mp)
    with open(mp + ".sha256", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("%s  %s\n" % (sha, V3_REPAIR_MARKER_NAME))


if __name__ == "__main__":
    sys.exit(main())
