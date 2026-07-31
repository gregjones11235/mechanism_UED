#!/usr/bin/env python3
"""CC4 Tier3 — FORMAL 6-student ranking + closing gate (V2_DYNAMIC_TOPOLOGY).

Runs AFTER all seven formal evaluations (6 students + teacher reference) have
written <pool>/cc4/<ID>/formal_evaluation_v2dt/. It:

  1. verifies every bundle (READY_FORMAL_V2DT all-true, formal counts 64/8/8,
     run_class=FORMAL_EVALUATION, NOT rehearsal, no abort, params unchanged,
     certificate self-verify incl. SHA256SUMS_FORMAL_V2DT re-hash, pin set
     uniform and == the frozen V2 pins, git HEAD uniform and == the marker
     HEAD, registry student_rank still null);
  2. extracts the 4-metric rule tuple per the FROZEN metric_schema rule
     (verbatim order string, lexicographic desc, tie tolerance 1e-12, full
     4-level tie -> INCONCLUSIVE); teacher gets metrics but rank=null,
     excluded from the student ranking;
  3. writes <pool>/cc4/FORMAL_RANKING_SUMMARY_V2DT.json (+.sha256) and
     FORMAL_EVALUATION_GATE_V2DT.json;
  4. ONLY IF the gate passes (and not --dry-run): performs the single
     closing READY update — the allowlisted completion-state RMW of
     <pool>/common_v2/COMMON_EVALUATOR_V2_READY.json (FORMAL_RANKING_STARTED=
     true, FORMAL_RANKING_PUBLISHED=true, started-at from the audit marker,
     summary SHA, marker reference, pending gate retired). This tool is the
     SOLE writer of the READY ranking flags: the start marker tool writes
     only the audit marker, the formal driver requires FORMAL_RANKING_STARTED
     to still be false — one writer, zero races, and after the flip the
     driver fail-closes so formal runs cannot silently repeat.

Never rewrites certificates; never writes student_rank back into the registry;
never records an engine-BLOCKED candidate as a formal score; never overclaims
(forbidden-claim scan before any write). <6 eligible students ->
INCONCLUSIVE_PARTICIPATION + gate FAIL + no flip + escalate 总控.

Usage (server):
  python tools/tier3_scaffolded_evaluation/tier3_formal_ranking_v2dt.py \
      --pool-cc4-dir /home/oseasy/student_pool_v1/cc4 \
      --common-dir /home/oseasy/student_pool_v1/common_v2 \
      [--dry-run]

  --self-test   structural + synthetic-bundle checks (JAX-free, any host)
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import tier3_projection_runtime as proj                            # noqa: E402
import tier3_projection_binding_smoke_v2 as smokev2                # noqa: E402
import tier3_evaluation_certificate_v2dt as certmod                # noqa: E402
import tier3_formal_evaluation_v2dt as driver                      # noqa: E402

SUMMARY_SCHEMA = "mechanism_UED.tier3_formal_ranking_summary/v2dt"
GATE_SCHEMA = "mechanism_UED.tier3_formal_evaluation_gate/v2dt"
COMMON_EVALUATOR_PROTOCOL_VERSION = certmod.COMMON_EVALUATOR_PROTOCOL_VERSION
RUN_CLASS = certmod.RUN_CLASS
SCENARIO_ORDER = certmod.FORMAL_SCENARIO_ORDER

# Frozen rule (metric_schema.json selection_predicate_rule), VERBATIM.
FROZEN_RULE_ORDER = [
    "full success_count",
    "front_l2 transition_count",
    "front_l2 mean graph_distance_progress",
    "back_l2 defeat_count",
]
FROZEN_RULE_ALL_EQUAL_RESULT = "INCONCLUSIVE"
SELECTION_TIE_TOLERANCE = 1e-12

STUDENTS = [
    "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
    "RESET128_RMT16_ORIGINAL_VTRACE_98304",
    "BASE_GTRXL_ORIGINAL_VTRACE_98304",
    "CONTROL_CONTINUOUS_98304",
    "SLOWGRU_RESET128_CANONICAL_98304",
    "SLOWGRU_PERSISTENT_CANONICAL_98304",
]
TEACHER = "BASELINE_TEACHER_CKPT17500"
ALL_CANDIDATES = STUDENTS + [TEACHER]
FORMAL_COUNTS = {"full": 64, "front_l2": 8, "back_l2": 8}

SUMMARY_NAME = "FORMAL_RANKING_SUMMARY_V2DT.json"
GATE_NAME = "FORMAL_EVALUATION_GATE_V2DT.json"

# The ONLY keys this tool may add/change in COMMON_EVALUATOR_V2_READY.json.
READY_FLIP_ALLOWED_KEYS = frozenset([
    "FORMAL_RANKING_STARTED",
    "FORMAL_RANKING_PUBLISHED",
    "formal_evaluation_started_at_utc",
    "formal_ranking_published_at_utc",
    "formal_ranking_summary_sha256",
    "formal_evaluation_gate_sha256",
    "secondary_audit_marker",
    "pending_gates",
])


# ---------------------------------------------------------------------------
# pure rule machinery (unit-testable)
# ---------------------------------------------------------------------------
def extract_rule_tuple(evaluation_by_scenario, candidate_id):
    """Map ev.evaluate() outputs to the frozen 4-tuple, verifying scenario
    identities and counts. Fail-closed on any missing/None value.

    Real ev.evaluate() shape (frozen tier3_evaluator):
      {"schema", "scenario", "metrics": {"primary", "dense",
       ["diagnostics" for back_l2], ...}, "valid_start_count",
       "episode_count", "terminal_label_counts", "contract", ...}
    """
    ev_full = evaluation_by_scenario["full"]
    ev_front = evaluation_by_scenario["front_l2"]
    ev_back = evaluation_by_scenario["back_l2"]
    proj.require(ev_full["scenario"] == "full"
                 and ev_front["scenario"] == "front_l2"
                 and ev_back["scenario"] == "back_l2",
                 "FAIL CLOSED (RULE_EXTRACT %s): scenario identity mismatch"
                 % candidate_id)
    m_full, m_front, m_back = (ev_full["metrics"], ev_front["metrics"],
                               ev_back["metrics"])
    proj.require(m_full["primary"]["scenario"] == "full"
                 and m_front["primary"]["scenario"] == "front_l2"
                 and m_back["primary"]["scenario"] == "back_l2",
                 "FAIL CLOSED (RULE_EXTRACT %s): metrics scenario mismatch"
                 % candidate_id)
    proj.require(m_full["primary"]["valid_starts"] == 64,
                 "FAIL CLOSED (RULE_EXTRACT %s): full valid_starts %r"
                 % (candidate_id, m_full["primary"]["valid_starts"]))
    proj.require(m_front["primary"]["valid_starts"] == 8,
                 "FAIL CLOSED (RULE_EXTRACT %s): front valid_starts %r"
                 % (candidate_id, m_front["primary"]["valid_starts"]))
    proj.require(m_back["primary"]["valid_starts"] == 8,
                 "FAIL CLOSED (RULE_EXTRACT %s): back valid_starts %r"
                 % (candidate_id, m_back["primary"]["valid_starts"]))
    front_progress = m_front["dense"]["value"]
    proj.require(front_progress is not None,
                 "FAIL CLOSED (RULE_EXTRACT %s): front dense value is None"
                 % candidate_id)
    back_defeat = int(m_back["primary"]["successes"])
    proj.require(back_defeat == int(
        m_back["diagnostics"]["survival"]["defeat_count"]),
        "FAIL CLOSED (RULE_EXTRACT %s): back primary/diagnostics mismatch"
        % candidate_id)
    return (int(m_full["primary"]["successes"]),
            int(m_front["primary"]["successes"]),
            float(front_progress),
            back_defeat)


def compare_rule_tuples(a, b, tol=SELECTION_TIE_TOLERANCE):
    """Lexicographic DESCENDING with per-level tie tolerance.
    -1: a strictly ranks above b; 1: below; 0: FULL four-level tie."""
    for av, bv in zip(a, b):
        if av - bv > tol:
            return -1
        if bv - av > tol:
            return 1
    return 0


def rank_students(entries):
    """entries: [{"candidate_id","rule_tuple"}]. Returns
    (ranks {cid: int|None}, ranking_status, inconclusive_groups, ordered_ids).
    Competition ranking over the strict descending order; any full four-level
    tie group -> ranks None for the group, status INCONCLUSIVE_FULL_TIE."""
    ordered = sorted(entries,
                     key=lambda e: tuple(-v for v in e["rule_tuple"])
                     + (e["candidate_id"],))
    groups = []
    for e in ordered:
        if (groups and compare_rule_tuples(groups[-1][0]["rule_tuple"],
                                           e["rule_tuple"]) == 0):
            groups[-1].append(e)
        else:
            groups.append([e])
    ranks, inconclusive = {}, []
    pos = 1
    for g in groups:
        if len(g) > 1:
            ids = sorted(e["candidate_id"] for e in g)
            inconclusive.append(ids)
            for e in g:
                ranks[e["candidate_id"]] = None
        else:
            ranks[g[0]["candidate_id"]] = pos
        pos += len(g)
    status = "ORDERED" if not inconclusive else "INCONCLUSIVE_FULL_TIE"
    return ranks, status, inconclusive, [e["candidate_id"] for e in ordered]


# ---------------------------------------------------------------------------
# bundle verification
# ---------------------------------------------------------------------------
def _check(cond, failures, name, detail=""):
    if not cond:
        failures.append("%s%s" % (name, (": %s" % detail) if detail else ""))


def verify_candidate_bundle(cid, cc4_dir, expected_head=None):
    """Verify one candidate's formal_evaluation_v2dt bundle. Returns
    {"candidate_id","eligible","participant_status","problems","cert","ready",
     "evaluation_by_scenario","rule_tuple","git_commit_head"} — never raises
    for eligibility problems (recorded as problems); fail-closed only on
    structurally unreadable JSON."""
    spec = proj.get_spec(cid)
    out = {"candidate_id": cid, "eligible": False,
           "participant_status": "NOT_ELIGIBLE", "problems": [],
           "cert": None, "ready": None, "evaluation_by_scenario": None,
           "rule_tuple": None, "git_commit_head": None, "spec": spec}
    f = out["problems"]
    formal_dir = os.path.join(cc4_dir, cid, "formal_evaluation_v2dt")
    if not os.path.isdir(formal_dir):
        out["participant_status"] = "NOT_ELIGIBLE_COMPLETE"
        f.append("formal_evaluation_v2dt directory missing")
        return out
    out["formal_dir"] = formal_dir

    ready = proj.read_json(os.path.join(formal_dir, "READY_FORMAL_V2DT.json"))
    out["ready"] = ready
    _check(ready.get("candidate_id") == cid, f, "READY_CANDIDATE_ID")
    _check(ready.get("READY_FORMAL_V2DT") is True, f, "READY_FORMAL_V2DT_FALSE",
           str({k: v for k, v in (ready.get("gates") or {}).items() if not v}))
    _check(bool(ready.get("gates")) and all(ready["gates"].values()), f,
           "READY_INNER_GATES", str(ready.get("gates")))
    _check(ready.get("rehearsal") is False, f, "REHEARSAL_IN_FORMAL_POOL")
    _check(ready.get("run_class") == RUN_CLASS, f, "RUN_CLASS",
           str(ready.get("run_class")))
    _check(ready.get("student_rank") is None, f, "READY_STUDENT_RANK_NOT_NULL")
    _check(ready.get("scientific_claim_authorized") is False, f,
           "SCIENTIFIC_CLAIM_FLAG")
    _check(ready.get("teacher_included_in_student_ranking") is False, f,
           "TEACHER_RANKING_FLAG")
    if ready.get("formal_abort") is not None:
        out["participant_status"] = "BLOCKED_ENGINE_ABORT"
        f.append("formal_abort recorded (engine predicate BLOCKED)")
        # honest BLOCKED: still record, never eligible
    elif ready.get("evaluation_status") != "PASS":
        out["participant_status"] = "NOT_ELIGIBLE_COMPLETE"
        f.append("evaluation_status=%r" % ready.get("evaluation_status"))

    cert = proj.read_json(os.path.join(formal_dir,
                                       "evaluation_certificate_v2dt.json"))
    out["cert"] = cert
    out["git_commit_head"] = (cert.get("provenance") or {}).get("git_commit_head")
    cert_problems = certmod.verify_evaluation_certificate(cert,
                                                          evidence_dir=formal_dir)
    _check(not cert_problems, f, "CERTIFICATE_VERIFY", str(cert_problems))
    _check(cert.get("common_pins") == certmod.PIN_FIELD_SOURCES, f,
           "CERT_PINS_DRIFT")
    _check(cert.get("student_rank") is None, f, "CERT_STUDENT_RANK_NOT_NULL")
    _check(cert.get("teacher_included_in_student_ranking") is False, f,
           "CERT_TEACHER_FLAG")
    _check((cert.get("identity") or {}).get("candidate_id") == cid
           or cert.get("candidate_id") == cid, f, "CERT_CANDIDATE_ID")
    if expected_head:
        _check(out["git_commit_head"] == expected_head, f, "GIT_HEAD",
               "%r != %r" % (out["git_commit_head"], expected_head))

    evaluation_by_scenario = {}
    counts_ok = True
    for sc in SCENARIO_ORDER:
        rp = os.path.join(formal_dir, "evaluation_result_v2dt.%s.json" % sc)
        if not os.path.isfile(rp):
            f.append("result file missing: %s" % sc)
            counts_ok = False
            continue
        r = proj.read_json(rp)
        _check(r.get("candidate_id") == cid, f, "RESULT_CANDIDATE_ID %s" % sc)
        _check(r.get("run_class") == RUN_CLASS, f, "RESULT_RUN_CLASS %s" % sc)
        _check(r.get("rehearsal") is False, f, "RESULT_REHEARSAL %s" % sc)
        _check(r.get("aborted_in_scenario") is False, f, "RESULT_ABORT %s" % sc)
        _check(r.get("episodes_executed") == FORMAL_COUNTS[sc], f,
               "RESULT_COUNT %s" % sc, str(r.get("episodes_executed")))
        evd = r.get("evaluation")
        if evd is None:
            f.append("evaluation missing in result %s" % sc)
            counts_ok = False
        else:
            evaluation_by_scenario[sc] = evd
    if len(evaluation_by_scenario) == 3 and counts_ok:
        out["evaluation_by_scenario"] = evaluation_by_scenario
        if out["participant_status"] not in ("BLOCKED_ENGINE_ABORT",):
            try:
                out["rule_tuple"] = extract_rule_tuple(evaluation_by_scenario,
                                                       cid)
            except proj.FailClosed as exc:
                f.append("rule extraction: %s" % exc)

    is_teacher = spec["candidate_class"] == "TEACHER_REFERENCE"
    if not f:
        out["eligible"] = True
        out["participant_status"] = ("TEACHER_REFERENCE_ONLY" if is_teacher
                                     else "ELIGIBLE_COMPLETE")
    return out


# ---------------------------------------------------------------------------
# READY closing flip (sole writer)
# ---------------------------------------------------------------------------
def apply_ready_flip(common_dir, marker_ref, summary_sha, gate_sha, now_utc):
    ready_path = os.path.join(common_dir, "COMMON_EVALUATOR_V2_READY.json")
    proj.require(os.path.isfile(ready_path),
                 "FAIL CLOSED (READY_FLIP): %s missing" % ready_path)
    before = proj.read_json(ready_path)
    proj.require(before.get("FORMAL_RANKING_STARTED") is False,
                 "FAIL CLOSED (READY_FLIP): FORMAL_RANKING_STARTED already %r "
                 "— the ranking close runs exactly once"
                 % before.get("FORMAL_RANKING_STARTED"))
    after = dict(before)
    after["FORMAL_RANKING_STARTED"] = True
    after["FORMAL_RANKING_PUBLISHED"] = True
    after["formal_evaluation_started_at_utc"] = marker_ref["recorded_at_utc"]
    after["formal_ranking_published_at_utc"] = now_utc
    after["formal_ranking_summary_sha256"] = summary_sha
    after["formal_evaluation_gate_sha256"] = gate_sha
    after["secondary_audit_marker"] = {
        "path": marker_ref["path"],
        "sha256": marker_ref["sha256"],
        "verdict": marker_ref["verdict"]}
    pending = [g for g in (before.get("pending_gates") or [])
               if g != "INDEPENDENT_SECONDARY_AUDIT"]
    after["pending_gates"] = pending
    changed = {k for k in set(before) | set(after)
               if before.get(k) != after.get(k)}
    unexpected = changed - READY_FLIP_ALLOWED_KEYS
    proj.require(not unexpected,
                 "FAIL CLOSED (READY_FLIP): non-allowlisted keys would change: "
                 "%s" % sorted(unexpected))
    tmp = ready_path + ".tmp_ranking"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        import json
        fh.write(json.dumps(after, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, ready_path)
    return {"before_sha256": proj.sha256_bytes(
                proj.canonical_json_bytes(before)),
            "after_sha256": proj.sha256_file(ready_path),
            "changed_keys": sorted(changed)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-cc4-dir", default=None,
                    help="default: parent of --common-dir + /cc4")
    ap.add_argument("--common-dir", default="/home/oseasy/student_pool_v1/common_v2")
    ap.add_argument("--out", default=None,
                    help="summary/gate output dir (default: --pool-cc4-dir)")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify + report only; write neither summary nor READY")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return run_self_test()

    pool_cc4_dir = args.pool_cc4_dir or os.path.join(
        os.path.dirname(os.path.normpath(args.common_dir)), "cc4")
    out_dir = args.out or pool_cc4_dir
    proj.require(os.path.isdir(pool_cc4_dir),
                 "FAIL CLOSED: pool cc4 dir missing: %s" % pool_cc4_dir)

    # --- frozen rule verbatim cross-check against the pinned schema file ------
    rule_source = {"schema_file": None, "verified_verbatim": False}
    schema_path = os.path.join(args.common_dir, "metric_schema.json")
    if os.path.isfile(schema_path):
        proj.require(proj.sha256_file(schema_path)
                     == smokev2.FROZEN_V2_METRIC_SCHEMA_SHA256,
                     "FAIL CLOSED: metric_schema.json sha != frozen %s"
                     % smokev2.FROZEN_V2_METRIC_SCHEMA_SHA256)
        schema = proj.read_json(schema_path)
        spr = schema["selection_predicate_rule"]
        proj.require(spr["order"] == FROZEN_RULE_ORDER,
                     "FAIL CLOSED: schema rule order %r != frozen %r"
                     % (spr["order"], FROZEN_RULE_ORDER))
        proj.require(spr["all_equal_result"] == FROZEN_RULE_ALL_EQUAL_RESULT,
                     "FAIL CLOSED: schema all_equal_result %r"
                     % spr["all_equal_result"])
        rule_source = {"schema_file": schema_path,
                       "schema_sha256": smokev2.FROZEN_V2_METRIC_SCHEMA_SHA256,
                       "verified_verbatim": True}

    # --- secondary audit marker (start authorization) -------------------------
    marker_ref = driver.verify_formal_start(args.common_dir, pool_cc4_dir)
    expected_head = None  # uniformity checked below against marker if recorded
    marker_json = proj.read_json(marker_ref["path"])
    expected_head = marker_json.get("git_commit_head")

    # --- verify all seven bundles ---------------------------------------------
    bundles = [verify_candidate_bundle(cid, pool_cc4_dir, expected_head)
               for cid in ALL_CANDIDATES]
    students = [b for b in bundles if b["spec"]["candidate_class"] == "STUDENT"]
    teacher = [b for b in bundles
               if b["spec"]["candidate_class"] == "TEACHER_REFERENCE"]
    eligible_students = [b for b in students if b["eligible"]]
    teacher_ok = bool(teacher) and teacher[0]["eligible"]

    # registry write-back guard (student_rank structurally null)
    for b in bundles:
        proj.require(b["spec"].get("student_rank") is None,
                     "FAIL CLOSED: registry student_rank for %s is %r — ranking "
                     "write-back is forbidden" % (b["candidate_id"],
                                                  b["spec"].get("student_rank")))

    # --- ranking ---------------------------------------------------------------
    ranks, ranking_status, inconclusive, ordered_ids = ({}, "INCONCLUSIVE_"
        "PARTICIPATION", [], [])
    if len(eligible_students) == len(STUDENTS):
        ranks, ranking_status, inconclusive, ordered_ids = rank_students(
            [{"candidate_id": b["candidate_id"], "rule_tuple": b["rule_tuple"]}
             for b in eligible_students])

    participants = []
    for b in bundles:
        cid = b["candidate_id"]
        is_teacher_c = b["spec"]["candidate_class"] == "TEACHER_REFERENCE"
        t = b["rule_tuple"]
        participants.append({
            "candidate_id": cid,
            "candidate_class": b["spec"]["candidate_class"],
            "runtime_family": b["spec"]["runtime_family"],
            "counts_toward_student_binding_count":
                b["spec"].get("counts_toward_student"),
            "reference_only": b["spec"]["reference_only"],
            "participant_status": b["participant_status"],
            "eligible": b["eligible"],
            "rule_tuple": ({"full success_count": t[0],
                            "front_l2 transition_count": t[1],
                            "front_l2 mean graph_distance_progress": t[2],
                            "back_l2 defeat_count": t[3]} if t else None),
            "student_rank": (None if is_teacher_c else ranks.get(cid)),
            "excluded_from_student_ranking": is_teacher_c,
            "problems": b["problems"],
            "formal_dir": b.get("formal_dir"),
            "certificate_sha256": (proj.sha256_file(os.path.join(
                b["formal_dir"], "evaluation_certificate_v2dt.json"))
                if b.get("formal_dir") and os.path.isfile(os.path.join(
                    b["formal_dir"], "evaluation_certificate_v2dt.json"))
                else None),
        })

    gate_failures = []
    for b in bundles:
        gate_failures.extend("%s: %s" % (b["candidate_id"], p)
                             for p in b["problems"])
    heads = {b["git_commit_head"] for b in bundles if b["git_commit_head"]}
    heads_uniform = len(heads) == 1
    if not heads_uniform:
        gate_failures.append("GIT_HEAD_NOT_UNIFORM: %s" % sorted(
            str(h) for h in heads))
    gates = {
        "G1_ALL_6_STUDENTS_ELIGIBLE_COMPLETE":
            len(eligible_students) == len(STUDENTS),
        "G2_TEACHER_REFERENCE_COMPLETE": teacher_ok,
        "G3_NO_ENGINE_ABORT": all(
            b["participant_status"] != "BLOCKED_ENGINE_ABORT" for b in bundles),
        "G4_NO_REHEARSAL_IN_FORMAL_POOL": all(
            (b["ready"] or {}).get("rehearsal") is False for b in bundles),
        "G5_CERTIFICATES_ALL_VERIFY": not any(
            "CERTIFICATE_VERIFY" in p for b in bundles for p in b["problems"]),
        "G6_PINS_UNIFORM_FROZEN": all(
            (b["cert"] or {}).get("common_pins") == certmod.PIN_FIELD_SOURCES
            for b in bundles if b["cert"]),
        "G7_GIT_HEAD_UNIFORM": heads_uniform,
        "G8_RULE_VERBATIM": rule_source.get("verified_verbatim", False)
            or not os.path.isfile(schema_path),
        "G9_REGISTRY_RANK_NULL": all(
            b["spec"].get("student_rank") is None for b in bundles),
        "G10_RANKING_COMPUTED_HONEST": ranking_status in (
            "ORDERED", "INCONCLUSIVE_FULL_TIE", "INCONCLUSIVE_PARTICIPATION"),
    }
    gate_pass = all(gates.values()) and not gate_failures

    now_utc = smokev2.utc_now_iso()
    summary = {
        "schema": SUMMARY_SCHEMA,
        "generated_at_utc": now_utc,
        "common_evaluator_protocol_version": COMMON_EVALUATOR_PROTOCOL_VERSION,
        "run_class": RUN_CLASS,
        "secondary_audit_marker": {"path": marker_ref["path"],
                                   "sha256": marker_ref["sha256"],
                                   "verdict": marker_ref["verdict"]},
        "selection_predicate_rule": {
            "order": FROZEN_RULE_ORDER,
            "tie_tolerance": SELECTION_TIE_TOLERANCE,
            "all_equal_result": FROZEN_RULE_ALL_EQUAL_RESULT,
            "source": rule_source,
            "note": "frozen lexicographic comparator from metric_schema.json "
                    "(sha-pinned); PROVISIONAL-class rule; no scientific-"
                    "superiority claim, multi-seed confirmation required"},
        "ranking_status": ranking_status,
        "inconclusive_groups": inconclusive,
        "ordered_by_strict_rule": ordered_ids,
        "top_ranked_student_id": (ordered_ids[0]
                                  if ranking_status == "ORDERED" else None),
        "participants": participants,
        "student_count_eligible": "%d/%d" % (len(eligible_students),
                                             len(STUDENTS)),
        "teacher_included_in_student_ranking": False,
        "scientific_claim_authorized": False,
        "scientific_claim_status":
            "FORMAL_SCIENTIFIC_CLAIM: NOT_AUTHORIZED_SINGLE_TRAINING_SEED",
        "scaffolded_results_can_replace_full_task": False,
        "interface_smoke_substituted_for_performance": False,
        "gate_failures": gate_failures,
        "escalation": ("INCONCLUSIVE_PARTICIPATION: fewer than 6 eligible "
                       "students — escalate to 总控 before any selection claim"
                       if ranking_status == "INCONCLUSIVE_PARTICIPATION"
                       else None),
    }
    overclaim = certmod.scan_forbidden_overclaims(summary)
    proj.require(not overclaim,
                 "FAIL CLOSED (OVERCLAIM_SCAN): summary contains %s" % overclaim)

    if args.dry_run:
        print("[dry-run] ranking_status=%s gate_pass=%s eligible=%d/%d "
              "teacher_ok=%s" % (ranking_status, gate_pass,
                                 len(eligible_students), len(STUDENTS),
                                 teacher_ok), flush=True)
        for p in participants:
            print("  %-42s status=%-22s rank=%s tuple=%s"
                  % (p["candidate_id"], p["participant_status"],
                     p["student_rank"], p["rule_tuple"]), flush=True)
        if gate_failures:
            print("[dry-run] gate_failures:", flush=True)
            for g in gate_failures:
                print("  -", g, flush=True)
        return 0 if gate_pass else 2

    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, SUMMARY_NAME)
    smokev2.write_json(summary_path, summary)
    summary_sha = proj.sha256_file(summary_path)
    with open(summary_path + ".sha256", "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write("%s  %s\n" % (summary_sha, SUMMARY_NAME))

    gate = {
        "schema": GATE_SCHEMA,
        "generated_at_utc": now_utc,
        "common_evaluator_protocol_version": COMMON_EVALUATOR_PROTOCOL_VERSION,
        "FORMAL_EVALUATION_GATE_V2DT_PASS": gate_pass,
        "gates": gates,
        "gate_failures": gate_failures,
        "ranking_status": ranking_status,
        "student_common_eligible_count": "%d/%d" % (len(eligible_students),
                                                    len(STUDENTS)),
        "teacher_reference_binding": "PASS" if teacher_ok else "FAIL",
        "formal_ranking_summary_sha256": summary_sha,
        "secondary_audit_marker_sha256": marker_ref["sha256"],
        "git_heads_uniform": sorted(heads),
        "server_git_head": sorted(heads)[0] if heads else None,
        "CHECKPOINTS_MODIFIED": False,
        "CONTROL_RETRAINED": False,
        "CANDIDATE_EXCEPTION_USED": False,
        "FROZEN_BANKS_MODIFIED": False,
        "RETRAINING_PERFORMED": False,
        "honest_discipline": "gate false => READY ranking flags NOT flipped; "
            "BLOCKED candidates never counted; teacher never ranked",
    }
    gate_path = os.path.join(out_dir, GATE_NAME)
    smokev2.write_json(gate_path, gate)
    gate_sha = proj.sha256_file(gate_path)
    with open(gate_path + ".sha256", "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write("%s  %s\n" % (gate_sha, GATE_NAME))
    print("[ranking] summary=%s sha=%s" % (summary_path, summary_sha),
          flush=True)
    print("[ranking] gate=%s PASS=%s ranking_status=%s"
          % (gate_path, gate_pass, ranking_status), flush=True)

    if not gate_pass:
        print("[ranking] GATE FAIL — READY flip SKIPPED; failures:",
              flush=True)
        for g in gate_failures:
            print("  -", g, flush=True)
        return 2

    flip_ev = apply_ready_flip(args.common_dir, marker_ref, summary_sha,
                               gate_sha, now_utc)
    print("[ranking] READY closing flip applied: changed_keys=%s after_sha=%s"
          % (flip_ev["changed_keys"], flip_ev["after_sha256"][:16]),
          flush=True)
    print("[ranking] FORMAL_RANKING_PUBLISHED — driver now fail-closes on any "
          "rerun (FORMAL_RANKING_STARTED=true)", flush=True)
    return 0


# ---------------------------------------------------------------------------
# self-test (JAX-free synthetic bundles)
# ---------------------------------------------------------------------------
def _synthetic_evaluation(scenario, successes, valid_starts, dense_value=None,
                          defeat_count=None):
    """Mirrors the REAL frozen ev.evaluate() envelope: top-level scenario /
    valid_start_count / contract, with the metrics block (primary / dense /
    back diagnostics) nested under 'metrics' (tier3_evaluator.evaluate wraps
    tier3_metrics.summarize)."""
    metrics = {
        "schema": "mechanism_UED.tier3_metrics/v1",
        "scenario": scenario,
        "primary": {"metric": "P_%s" % scenario.upper(),
                    "scenario": scenario,
                    "valid_starts": valid_starts,
                    "successes": successes,
                    "value": (successes / valid_starts) if valid_starts else 0.0,
                    "conditional_on": "valid_start",
                    "diagnostic_only": scenario != "full"},
        "dense": {"metric": "GRAPH_DISTANCE_PROGRESS",
                  "scenario": scenario,
                  "valid_starts": valid_starts,
                  "scored": valid_starts,
                  "value": dense_value,
                  "median": dense_value,
                  "per_state_progress": [],
                  "range": [0, 1],
                  "monotonicity_guaranteed": False,
                  "is_success_substitute": False},
        "scaffolded_results_can_replace_full_task": False,
    }
    if scenario == "back_l2":
        metrics["identity_class"] = "BOSS_COMBAT_SCAFFOLDED"
        metrics["na_metrics"] = ["boss_area_reached"]
        metrics["na_reason"] = "synthetic"
        metrics["diagnostics"] = {
            "valid_starts": valid_starts,
            "kobold_engaged_count": successes,
            "time_to_first_engagement": None,
            "time_to_kill": None,
            "damage": None,
            "schema_note": "synthetic",
            "survival": {"died_count": valid_starts - (defeat_count
                                                       if defeat_count
                                                       is not None
                                                       else successes),
                         "defeat_count": (defeat_count if defeat_count
                                          is not None else successes),
                         "mean_timesteps": 100.0,
                         "max_timesteps_observed": 200},
            "failure_taxonomy": {}}
    return {
        "schema": "mechanism_UED.tier3_evaluation_result/v1",
        "result_version": "tier3_evaluation_result/v1",
        "failure_rule_version": "tier3_failure_rules/v1",
        "scenario": scenario,
        "metrics": metrics,
        "valid_start_count": valid_starts,
        "episode_count": valid_starts,
        "terminal_label_counts": {},
        "contract": {"action_mode": "greedy_argmax",
                     "max_timesteps": 4096,
                     "action_space": "canonical_craftax_action_set",
                     "observation_schema": "canonical_craftax_symbolic",
                     "identical_for_all_arms": True},
        "rollout_status": "REAL_ENV_INTERFACE_READY",
        "materialization_status": "JAX_CRAFTAX_AVAILABLE",
        "checkpoint_params_sha256": None,
        "scaffolded_results_can_replace_full_task": False,
    }


def _build_synthetic_bundle(cc4_dir, cid, rule_tuple, git_head,
                            rehearsal=False, abort=False, corrupt_pin=False):
    """Build a complete, self-consistent synthetic formal bundle for cid."""
    d = os.path.join(cc4_dir, cid, "formal_evaluation_v2dt")
    os.makedirs(d, exist_ok=True)
    spec = proj.get_spec(cid)
    t = rule_tuple
    evaluation_by_scenario = {
        "full": _synthetic_evaluation("full", t[0], 64),
        "front_l2": _synthetic_evaluation("front_l2", t[1], 8,
                                          dense_value=t[2]),
        "back_l2": _synthetic_evaluation("back_l2", t[3], 8,
                                         defeat_count=t[3]),
    }
    # episode records jsonl + sums-consistent files
    jsonl_lines = [proj.canonical_json_bytes(
        {"candidate_id": cid, "scenario": sc, "synthetic": True}).decode()
        for sc in SCENARIO_ORDER]
    jsonl_bytes = ("\n".join(jsonl_lines) + "\n").encode("utf-8")
    with open(os.path.join(d, "episode_records.jsonl"), "wb") as fh:
        fh.write(jsonl_bytes)
    jsonl_sha = proj.sha256_bytes(jsonl_bytes)
    smokev2.write_json(os.path.join(d, "provenance.json"),
                       {"schema": "synthetic", "pid": 1, "argv": [],
                        "cwd": "/", "host": "synthetic",
                        "git_commit_head": git_head,
                        "generated_at_utc": "1970-01-01T00:00:00+00:00"})
    for sc in SCENARIO_ORDER:
        smokev2.write_json(os.path.join(
            d, "evaluation_result_v2dt.%s.json" % sc),
            {"schema": driver.RESULT_SCHEMA, "candidate_id": cid,
             "scenario": sc, "run_class": RUN_CLASS,
             "common_evaluator_protocol_version":
                 COMMON_EVALUATOR_PROTOCOL_VERSION,
             "rehearsal": rehearsal,
             "schedule": {"seeds": list(range(FORMAL_COUNTS[sc]))},
             "entry_ids_planned": ["%s-bank%d" % (sc, i)
                                   for i in range(FORMAL_COUNTS[sc])],
             "episodes_planned": FORMAL_COUNTS[sc],
             "episodes_executed": FORMAL_COUNTS[sc],
             "episode_records_sha256": jsonl_sha,
             "aborted_in_scenario": False,
             "evaluation": evaluation_by_scenario[sc],
             "timing": {"episodes": [], "scenario_wall_seconds": 1.0,
                        "peak_rss_kb": None},
             "generated_at_utc": "1970-01-01T00:00:00+00:00"})
    # certificate via the real builder (pins, honest labels, rank=null)
    ci = certmod._sample_cert_input(candidate_id=cid)
    ci["results_by_scenario"] = evaluation_by_scenario
    ci["episode_records_jsonl_sha256"] = jsonl_sha
    ci["generated_at_utc"] = "1970-01-01T00:00:00+00:00"
    ci["provenance"] = {"pid": 1, "argv": [], "cwd": "/", "host": "synthetic",
                        "git_commit_head": git_head,
                        "device_identity": {"synthetic": True},
                        "scenario_wall_seconds": {},
                        "timing_by_scenario": {}}
    ci["rehearsal"] = rehearsal
    ci["formal_abort"] = ({"verdict": "ENGINE_PREDICATE_REJECTED_FORMAL_"
                                       "ROLLOUT_V2", "scenario": "full",
                           "episode_index": 0} if abort else None)
    if abort:
        ci["results_by_scenario"] = {}
        ci["episodes_executed"] = {"full": 0, "front_l2": 0, "back_l2": 0}
        ci["valid_start_counts"] = {"full": 0, "front_l2": 0, "back_l2": 0}
    cert = certmod.build_evaluation_certificate(ci)
    if corrupt_pin:
        cert["common_pins"]["common_evaluator_sha256"] = "0" * 64
    smokev2.write_json(os.path.join(d, "evaluation_certificate_v2dt.json"),
                       cert)
    # SHA256SUMS_FORMAL_V2DT over the same six files the driver sums
    summed = ["episode_records.jsonl", "provenance.json",
              "evaluation_certificate_v2dt.json"] + [
        "evaluation_result_v2dt.%s.json" % sc for sc in SCENARIO_ORDER]
    with open(os.path.join(d, "SHA256SUMS_FORMAL_V2DT"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join("%s  %s" % (proj.sha256_file(os.path.join(d, fn)),
                                       fn) for fn in summed) + "\n")
    # READY_FORMAL_V2DT
    is_teacher = spec["candidate_class"] == "TEACHER_REFERENCE"
    gates = {"G1_CAPSULE_FILE_SHA_MATCH": True,
             "G2_CHECKPOINT_FILE_SHA_OWNER_RECOMPUTE_MATCH": True,
             "G3_PARAMS_SHA_OWNER_RECOMPUTE_MATCH": True,
             "G4_FORMAL_SCHEDULE_COMPLETE": not abort,
             "G5_PARAMS_UNCHANGED": True,
             "G6_COMMON_V2_SUMS": True,
             "G7_EPISODE_COUNTS_FROM_V2_PROFILE": True,
             "G8_GPU_ALLOWED": True,
             "G9_V1_FROZEN_PRESERVED": True,
             "G10_PROTOCOL_VERSION_V2": True,
             "G11_SECONDARY_AUDIT_MARKER_VERIFIED": True,
             "G12_CERTIFICATE_VERIFIED": not corrupt_pin}
    status = ("BLOCKED" if abort else "REHEARSAL_NOT_FORMAL" if rehearsal
              else "PASS")
    smokev2.write_json(os.path.join(d, "READY_FORMAL_V2DT.json"),
                       {"schema": driver.READY_FORMAL_SCHEMA,
                        "candidate_id": cid,
                        "runtime_family": spec["runtime_family"],
                        "common_evaluator_protocol_version":
                            COMMON_EVALUATOR_PROTOCOL_VERSION,
                        "READY_FORMAL_V2DT": all(gates.values()),
                        "evaluation_status": status,
                        "run_class": RUN_CLASS,
                        "rehearsal": rehearsal,
                        "rehearsal_limits": None,
                        "formal_abort": ({"verdict": "SYNTHETIC_ABORT"}
                                         if abort else None),
                        "candidate_class": spec["candidate_class"],
                        "counts_toward_student_binding_count": not is_teacher,
                        "reference_only": spec["reference_only"],
                        "teacher_included_in_student_ranking": False,
                        "student_rank": None,
                        "performance_evaluation_executed": not rehearsal
                                                           and not abort,
                        "scientific_claim_authorized": False,
                        "scaffolded_results_can_replace_full_task": False,
                        "secondary_audit_marker": {"path": "marker",
                                                   "sha256": "1" * 64,
                                                   "verdict": "x"},
                        "gates": gates,
                        "generated_at_utc": "1970-01-01T00:00:00+00:00",
                        "evidence_files": summed + ["SHA256SUMS_FORMAL_V2DT"],
                        "honest_false_discipline": "synthetic"})
    return d


def _write_marker(cc4_dir, git_head):
    marker = {"schema": "mechanism_UED.tier3_secondary_audit_marker/v2dt",
              "verdict": driver.SECONDARY_AUDIT_VERDICT,
              "binding_gate_sha256": driver.POOL_BINDING_GATE_V2DT_SHA256,
              "pool_cc4_dir": cc4_dir,
              "recorded_at_utc": "1970-01-01T00:00:01+00:00",
              "git_commit_head": git_head}
    mp = os.path.join(cc4_dir, driver.SECONDARY_AUDIT_MARKER_NAME)
    smokev2.write_json(mp, marker)
    sha = proj.sha256_file(mp)
    with open(mp + ".sha256", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("%s  %s\n" % (sha, driver.SECONDARY_AUDIT_MARKER_NAME))
    return mp, sha


def _write_ready_v2(common_dir, started=False):
    os.makedirs(common_dir, exist_ok=True)
    smokev2.write_json(os.path.join(common_dir,
                                    "COMMON_EVALUATOR_V2_READY.json"),
                       {"COMMON_EVALUATOR_V2_READY": True,
                        "FORMAL_RANKING_STARTED": started,
                        "pending_gates": (["INDEPENDENT_SECONDARY_AUDIT"]
                                          if not started else []),
                        "binding_gate_file_sha256":
                            driver.POOL_BINDING_GATE_V2DT_SHA256,
                        "common_evaluator_protocol_version":
                            COMMON_EVALUATOR_PROTOCOL_VERSION,
                        "STUDENT_COMMON_BINDING_PASS_COUNT": "6/6"})


def run_self_test():
    import tempfile
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        checks += 1
        proj.require(cond, "RANKING_SELF_TEST FAIL: %s" % msg)

    # pure rule machinery
    a = (10, 5, 0.5, 3)
    ok(compare_rule_tuples((11, 0, 0.0, 0), a) == -1, "level1 dominates")
    ok(compare_rule_tuples((10, 6, 0.0, 0), a) == -1, "level2 breaks tie")
    ok(compare_rule_tuples((10, 5, 0.5 + 2e-12, 0), a) == -1, "2e-12 ordered")
    ok(compare_rule_tuples((10, 5, 0.5 + 5e-13, 0), (10, 5, 0.5, 0)) == 0,
       "5e-13 within tolerance at level3 -> level4 decides")
    ok(compare_rule_tuples((10, 5, 0.5 + 5e-13, 3), a) == 0,
       "full tie within tolerance -> 0")
    ranks, status, incon, order = rank_students(
        [{"candidate_id": "A", "rule_tuple": (5, 4, 0.6, 2)},
         {"candidate_id": "B", "rule_tuple": (5, 4, 0.6, 2)},
         {"candidate_id": "C", "rule_tuple": (4, 4, 0.6, 2)}])
    ok(status == "INCONCLUSIVE_FULL_TIE" and ranks["A"] is None
       and ranks["B"] is None and ranks["C"] == 3
       and incon == [["A", "B"]], "tie group -> INCONCLUSIVE")
    ranks, status, incon, order = rank_students(
        [{"candidate_id": "A", "rule_tuple": (5, 4, 0.6, 2)},
         {"candidate_id": "B", "rule_tuple": (5, 4, 0.6, 1)},
         {"candidate_id": "C", "rule_tuple": (6, 0, 0.0, 0)}])
    ok(status == "ORDERED" and ranks == {"C": 1, "A": 2, "B": 3}
       and order == ["C", "A", "B"], "clean ordering with level4 break")

    with tempfile.TemporaryDirectory() as td:
        cc4 = os.path.join(td, "cc4")
        common = os.path.join(td, "common_v2")
        os.makedirs(cc4)
        head = "ab" * 20
        tuples = {
            STUDENTS[0]: (40, 6, 0.75, 5),
            STUDENTS[1]: (38, 6, 0.70, 5),
            STUDENTS[2]: (35, 5, 0.60, 4),
            STUDENTS[3]: (30, 5, 0.55, 3),
            STUDENTS[4]: (28, 4, 0.50, 3),
            STUDENTS[5]: (25, 4, 0.45, 2),
            TEACHER:     (64, 8, 0.99, 8),   # teacher strongest — still unranked
        }
        for cid, t in tuples.items():
            _build_synthetic_bundle(cc4, cid, t, head)
        _write_marker(cc4, head)
        _write_ready_v2(common, started=False)

        # happy path (dry run)
        rc = main(["--pool-cc4-dir", cc4, "--common-dir", common, "--dry-run"])
        checks += 1
        proj.require(rc == 0, "RANKING_SELF_TEST: dry-run rc=%d" % rc)
        ready_after = proj.read_json(os.path.join(
            common, "COMMON_EVALUATOR_V2_READY.json"))
        ok(ready_after["FORMAL_RANKING_STARTED"] is False,
           "dry-run must not flip READY")

        # bundle verifier sees everything
        b0 = verify_candidate_bundle(STUDENTS[0], cc4, head)
        ok(b0["eligible"] and b0["participant_status"] == "ELIGIBLE_COMPLETE"
           and b0["rule_tuple"] == tuples[STUDENTS[0]], "bundle verify clean")
        bt = verify_candidate_bundle(TEACHER, cc4, head)
        ok(bt["eligible"] and bt["participant_status"]
           == "TEACHER_REFERENCE_ONLY", "teacher reference-only")

        # real run: summary + gate + flip
        rc = main(["--pool-cc4-dir", cc4, "--common-dir", common])
        checks += 1
        proj.require(rc == 0, "RANKING_SELF_TEST: run rc=%d" % rc)
        summary = proj.read_json(os.path.join(cc4, SUMMARY_NAME))
        ok(summary["ranking_status"] == "ORDERED", "ordered")
        ok(summary["top_ranked_student_id"] == STUDENTS[0],
           "top student (teacher strongest yet excluded)")
        tp = [p for p in summary["participants"]
              if p["candidate_id"] == TEACHER][0]
        ok(tp["student_rank"] is None and tp["excluded_from_student_ranking"]
           and tp["rule_tuple"]["full success_count"] == 64,
           "teacher metrics present, rank null")
        ok(all(p["student_rank"] is not None
               for p in summary["participants"]
               if p["candidate_class"] == "STUDENT"), "all students ranked")
        ok(summary["selection_predicate_rule"]["order"] == FROZEN_RULE_ORDER
           and summary["selection_predicate_rule"]["tie_tolerance"] == 1e-12,
           "rule verbatim embedded")
        ok(not certmod.scan_forbidden_overclaims(summary), "overclaim scan")
        side = open(os.path.join(cc4, SUMMARY_NAME + ".sha256"),
                    encoding="utf-8").read().split()[0]
        ok(side == proj.sha256_file(os.path.join(cc4, SUMMARY_NAME)),
           "summary sidecar sha")
        gate = proj.read_json(os.path.join(cc4, GATE_NAME))
        ok(gate["FORMAL_EVALUATION_GATE_V2DT_PASS"] and all(
            gate["gates"].values()) and gate["gate_failures"] == [],
           "gate all pass")
        ready_after = proj.read_json(os.path.join(
            common, "COMMON_EVALUATOR_V2_READY.json"))
        ok(ready_after["FORMAL_RANKING_STARTED"] is True
           and ready_after["FORMAL_RANKING_PUBLISHED"] is True
           and ready_after["pending_gates"] == []
           and ready_after["formal_ranking_summary_sha256"] == side
           and ready_after["formal_evaluation_started_at_utc"]
           == "1970-01-01T00:00:01+00:00", "READY closing flip applied")
        ok(ready_after["COMMON_EVALUATOR_V2_READY"] is True
           and ready_after["STUDENT_COMMON_BINDING_PASS_COUNT"] == "6/6",
           "non-allowlisted keys preserved")

        # second close refuses (single writer)
        try:
            apply_ready_flip(common, {"path": "m", "sha256": "s",
                                      "verdict": "v",
                                      "recorded_at_utc": "x"}, "0" * 64,
                             "0" * 64, "1970-01-01T00:00:02+00:00")
            ok(False, "second flip accepted")
        except proj.FailClosed:
            checks += 1

        # rehearsal bundle rejected
        _build_synthetic_bundle(cc4, STUDENTS[0], tuples[STUDENTS[0]], head,
                                rehearsal=True)
        b = verify_candidate_bundle(STUDENTS[0], cc4, head)
        ok(not b["eligible"] and any("REHEARSAL_IN_FORMAL_POOL" in p
                                     for p in b["problems"]),
           "rehearsal bundle rejected")
        _build_synthetic_bundle(cc4, STUDENTS[0], tuples[STUDENTS[0]], head)

        # engine-abort bundle -> BLOCKED, never eligible
        _build_synthetic_bundle(cc4, STUDENTS[1], tuples[STUDENTS[1]], head,
                                abort=True)
        b = verify_candidate_bundle(STUDENTS[1], cc4, head)
        ok(not b["eligible"] and b["participant_status"]
           == "BLOCKED_ENGINE_ABORT", "abort -> BLOCKED")
        _build_synthetic_bundle(cc4, STUDENTS[1], tuples[STUDENTS[1]], head)

        # pin drift rejected
        _build_synthetic_bundle(cc4, STUDENTS[2], tuples[STUDENTS[2]], head,
                                corrupt_pin=True)
        b = verify_candidate_bundle(STUDENTS[2], cc4, head)
        ok(not b["eligible"] and any("CERT_PINS_DRIFT" in p or
                                     "CERTIFICATE_VERIFY" in p
                                     for p in b["problems"]),
           "pin drift rejected")
        _build_synthetic_bundle(cc4, STUDENTS[2], tuples[STUDENTS[2]], head)

        # git head mismatch rejected
        b = verify_candidate_bundle(STUDENTS[3], cc4, "cd" * 20)
        ok(not b["eligible"] and any("GIT_HEAD" in p for p in b["problems"]),
           "head mismatch rejected")

        # missing dir -> NOT_ELIGIBLE_COMPLETE, participation inconclusive
        import shutil
        shutil.rmtree(os.path.join(cc4, STUDENTS[4],
                                   "formal_evaluation_v2dt"))
        b = verify_candidate_bundle(STUDENTS[4], cc4, head)
        ok(not b["eligible"] and b["participant_status"]
           == "NOT_ELIGIBLE_COMPLETE", "missing dir not eligible")

        # full-tie -> INCONCLUSIVE_FULL_TIE (rebuild all six with tie pair)
        for cid in STUDENTS:
            _build_synthetic_bundle(cc4, cid, (30, 5, 0.55, 3), head)
        bundles = [verify_candidate_bundle(cid, cc4, head) for cid in STUDENTS]
        ranks, status, incon, order = rank_students(
            [{"candidate_id": b["candidate_id"], "rule_tuple": b["rule_tuple"]}
             for b in bundles])
        ok(status == "INCONCLUSIVE_FULL_TIE" and len(incon[0]) == 6
           and all(v is None for v in ranks.values()),
           "six-way full tie -> INCONCLUSIVE, all ranks null")

    print("RANKING_SELF_TEST_PASS checks=%d" % checks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
