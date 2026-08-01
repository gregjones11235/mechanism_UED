#!/usr/bin/env python3
"""CC4 Tier3 — FORMAL 6-student ranking + closing gate (V3_COMPOSITE_EVENT).

Runs AFTER all seven V3 formal evaluations (6 students + teacher reference) have
written <pool>/cc4/<ID>/formal_evaluation_v3/. It is the V3 semantic-repair
successor of tier3_formal_ranking_v2dt.py, authorized by the 总控 ruling
CC4_REPAIR_NEG20_COMPOSITE_EVENT_SEMANTICS_AND_COMPLETE_FORMAL_EVALUATION_V3.

Design (total-control §三/§五/§七):
  * The FROZEN ranking machinery — extract_rule_tuple / compare_rule_tuples /
    rank_students / FROZEN_RULE_ORDER / SELECTION_TIE_TOLERANCE — is imported
    VERBATIM from tier3_formal_ranking_v2dt (SAME code objects, not a copy):
    G8 "rule verbatim" is satisfied BY CONSTRUCTION. The V3 result envelope
    carries the frozen tier3_metrics.summarize output under its "metrics" key
    (bit-identical by construction), so the reused extractor reads it unchanged.
  * The V2 STUDENTS/TEACHER/ALL_CANDIDATES/FORMAL_COUNTS/SCENARIO_ORDER lists are
    likewise reused verbatim — no drift, no re-selection of the candidate set.

It:
  1. verifies every V3 bundle (READY_FORMAL_V3 all-true, formal counts 64/8/8,
     run_class=FORMAL_EVALUATION, NOT rehearsal, no abort, params unchanged,
     certificate self-verify incl. SHA256SUMS_FORMAL_V3 re-hash, the frozen V2
     pin set + taxonomy_v3 LF-SHA uniform, git HEAD uniform across bundles and
     each equal to, or a proven git descendant of, the V3 repair-marker HEAD,
     registry student_rank still null, per-arm reuse provenance complete
     (FULL REUSED_PASS / FRONT REUSED_RECLASSIFIED / BACK COMPLETED-or-RESIGNED),
     the V2 archive block untouched, GPU ⊆ {GPU2,GPU3});
  2. extracts the 4-metric rule tuple per the FROZEN metric_schema rule
     (verbatim order string, lexicographic desc, tie tolerance 1e-12, full
     4-level tie -> INCONCLUSIVE); teacher gets metrics but rank=null, excluded
     from the student ranking;
  3. writes <pool>/cc4/FORMAL_RANKING_SUMMARY_V3.json (+.sha256) and
     FORMAL_EVALUATION_GATE_V3.json (with the V2-archive reference: V2_STATUS=
     CLOSED_INCONCLUSIVE_PARTICIPATION, V2_WINNER=null, never modified);
  4. (unless --dry-run) performs the single closing READY update under one of
     three explicit, audited flip policies (recorded in the gate file):
       V3_GATE_GREEN — every gate G1..G16 true (the EXPECTED V3 outcome: all 6
         students + teacher complete). FORMAL_RANKING_AUTHORIZED_V3=true; winner
         = top student iff the strict rule ORDERED, else null (full four-level
         tie => INCONCLUSIVE_FULL_TIE, still authorized).
       V3_PUBLISH_HONEST_UNDER_COMPLETION_BLOCK — every INTEGRITY gate true and
         every remaining failure originates solely from BLOCKED candidates (an
         honest FULL reuse REJECT or a BACK completion that hit a retained
         fail-closed). FORMAL_RANKING_AUTHORIZED_V3=false, winner=null — STRICTER
         than V2: a completion block never yields a winner.
       NO_FLIP — any integrity failure -> exit 2, READY untouched.
     The update is the allowlisted completion-state RMW (sole writer/creator) of
     <pool>/cc4/COMMON_EVALUATOR_V3_READY.json (FORMAL_RANKING_STARTED=true,
     FORMAL_RANKING_PUBLISHED=true, started-at from the V3 repair marker, summary
     + gate SHAs, marker reference). After the flip the V3 driver fail-closes on
     any rerun (its start gate re-reads this file).

Never rewrites certificates; never writes student_rank back into the registry;
never records a BLOCKED candidate as a formal score; never overclaims (forbidden-
claim + no-V2-masquerade scans before any write); never touches the V2 evidence.

Usage (server):
  python tools/tier3_scaffolded_evaluation/tier3_formal_ranking_v3.py \
      --pool-cc4-dir /home/oseasy/student_pool_v1/cc4 \
      --common-dir /home/oseasy/student_pool_v1/common_v2 \
      [--v2-archive-dir <dir-with-V2 summary/gate>] [--dry-run]

  --self-test   structural + synthetic-V3-bundle checks (JAX-free, any host)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import tier3_projection_runtime as proj                            # noqa: E402
import tier3_projection_binding_smoke_v2 as smokev2                # noqa: E402
import tier3_evaluation_certificate_v3 as certmod                  # noqa: E402
import tier3_taxonomy_v3 as taxonomy_v3                            # noqa: E402
import tier3_formal_evaluation_v3 as driver                        # noqa: E402
# FROZEN ranking machinery + candidate set, reused VERBATIM (single source of
# truth; G8 "rule verbatim" holds by object identity, not by re-implementation).
import tier3_formal_ranking_v2dt as ranking_v2                     # noqa: E402

extract_rule_tuple = ranking_v2.extract_rule_tuple
compare_rule_tuples = ranking_v2.compare_rule_tuples
rank_students = ranking_v2.rank_students
FROZEN_RULE_ORDER = ranking_v2.FROZEN_RULE_ORDER
FROZEN_RULE_ALL_EQUAL_RESULT = ranking_v2.FROZEN_RULE_ALL_EQUAL_RESULT
SELECTION_TIE_TOLERANCE = ranking_v2.SELECTION_TIE_TOLERANCE
STUDENTS = ranking_v2.STUDENTS
TEACHER = ranking_v2.TEACHER
ALL_CANDIDATES = ranking_v2.ALL_CANDIDATES
FORMAL_COUNTS = ranking_v2.FORMAL_COUNTS
SCENARIO_ORDER = ranking_v2.SCENARIO_ORDER

SUMMARY_SCHEMA = "mechanism_UED.tier3_formal_ranking_summary/v3"
GATE_SCHEMA = "mechanism_UED.tier3_formal_evaluation_gate/v3"
READY_V3_SCHEMA = "mechanism_UED.tier3_common_evaluator_v3_ready/v1"
COMMON_EVALUATOR_PROTOCOL_VERSION = certmod.COMMON_EVALUATOR_PROTOCOL_VERSION  # V3_COMPOSITE_EVENT
NEG20_PROTOCOL = certmod.NEG20_PROTOCOL                  # NEG20_V3_PRIMARY_SECONDARY_EVENTS
RUN_CLASS = certmod.RUN_CLASS

# G16: V3 runs ONLY on GPU2/GPU3 (the GPU0/GPU1 unban was V2-only -> reverted).
V3_GPU_ALLOWED_UUIDS = driver.V3_GPU_ALLOWED_UUIDS
READY_V3_NAME = driver.READY_V3_NAME                     # COMMON_EVALUATOR_V3_READY.json
V2_ARCHIVE_SUMMARY_SHA256 = driver.V2_ARCHIVE_SUMMARY_SHA256
V2_ARCHIVE_GATE_SHA256 = driver.V2_ARCHIVE_GATE_SHA256
FULL = "full"
FRONT = "front_l2"
BACK = "back_l2"

SUMMARY_NAME = "FORMAL_RANKING_SUMMARY_V3.json"
GATE_NAME = "FORMAL_EVALUATION_GATE_V3.json"

# Anti-masquerade for THIS artifact (distinct from the certificate's
# scan_no_v2_masquerade, which is cert-schema-specific): the V3 ranking summary /
# gate may not embed the V2DT ranking summary/gate schema strings as leaf values.
V2DT_RANKING_SCHEMA_STRINGS = (
    "mechanism_UED.tier3_formal_ranking_summary/v2dt",
    "mechanism_UED.tier3_formal_evaluation_gate/v2dt",
)


def _leaf_strings(doc, acc):
    if isinstance(doc, str):
        acc.append(doc)
    elif isinstance(doc, dict):
        for v in doc.values():
            _leaf_strings(v, acc)
    elif isinstance(doc, (list, tuple)):
        for v in doc:
            _leaf_strings(v, acc)
    return acc


def scan_no_v2dt_ranking_masquerade(doc, own_schema):
    """The V3 ranking summary/gate must present its OWN v3 schema and must not
    embed the V2DT ranking summary/gate schema strings anywhere. The V2-archive
    reference block carries only V2 STATUS/WINNER/SHAs, never the V2DT schema
    string, so it is not tripped."""
    problems = []
    if doc.get("schema") != own_schema:
        problems.append("schema %r != %s" % (doc.get("schema"), own_schema))
    leaves = " ".join(_leaf_strings(doc, []))
    for s in V2DT_RANKING_SCHEMA_STRINGS:
        if s in leaves:
            problems.append("leaf value contains V2DT ranking schema string %s" % s)
    return problems

# The ONLY keys this tool may add/change in COMMON_EVALUATOR_V3_READY.json on a
# read-modify-write (when a prior stub exists). On fresh creation (no prior file)
# the whole document is written; on rerun the start gate fail-closes first.
READY_V3_FLIP_ALLOWED_KEYS = frozenset([
    "COMMON_EVALUATOR_V3_READY",
    "FORMAL_RANKING_STARTED",
    "FORMAL_RANKING_PUBLISHED",
    "FORMAL_RANKING_AUTHORIZED_V3",
    "formal_evaluation_started_at_utc",
    "formal_ranking_published_at_utc",
    "formal_ranking_summary_sha256",
    "formal_evaluation_gate_sha256",
    "flip_policy",
    "ranking_status",
    "formal_winner",
    "v3_repair_marker",
    "pending_gates",
])

# Per-arm reuse status vocabulary (certmod), referenced for the completion gates.
REUSED_PASS = "REUSED_PASS"
REUSED_RECLASSIFIED = "REUSED_RECLASSIFIED"
REUSED_RESIGNED = "REUSED_RESIGNED"
COMPLETED = "COMPLETED"
REJECT = "REJECT"

# INTEGRITY gates: must ALL be true before ANY closing flip is considered. They
# concern process integrity (pins, git, rule, marker, V2 archive, GPU, honesty)
# and are independent of candidate completion. The remaining gates (G1 all-6
# eligible, G2 teacher, G3 no engine abort, G11/G12/G13 per-arm completion) are
# PARTICIPATION/COMPLETION gates: they legitimately fail when a candidate is
# honestly BLOCKED (FULL reuse REJECT or a retained fail-closed on a BACK
# completion), and that failure is itself published — never papered over.
INTEGRITY_GATE_KEYS = (
    "G4_NO_REHEARSAL_IN_FORMAL_POOL",
    "G5_CERTIFICATES_ALL_VERIFY",
    "G6_PINS_UNIFORM_FROZEN",
    "G7_GIT_HEAD_UNIFORM",
    "G7b_GIT_HEAD_EQUAL_OR_DESCENDED_FROM_MARKER",
    "G8_RULE_VERBATIM",
    "G9_REGISTRY_RANK_NULL",
    "G10_RANKING_COMPUTED_HONEST",
    "G14_V2_ARCHIVE_UNTOUCHED",
    "G15_V3_REPAIR_MARKER_VERIFIED",
    "G16_GPU_V3_ONLY",
)


# ---------------------------------------------------------------------------
# bundle verification
# ---------------------------------------------------------------------------
def _check(cond, failures, name, detail=""):
    if not cond:
        failures.append("%s%s" % (name, (": %s" % detail) if detail else ""))


def verify_candidate_bundle(cid, cc4_dir, expected_head=None, is_ancestor=None,
                            marker_sha=None):
    """Verify one candidate's formal_evaluation_v3 bundle. Returns a dict with
    {"candidate_id","eligible","participant_status","problems","cert","ready",
     "evaluation_by_scenario","rule_tuple","reuse_by_scenario","git_commit_head",
     "git_head_relation_to_marker","spec","gpu_uuids"} — never raises for
    eligibility problems (recorded as problems); fail-closed only on structurally
    unreadable JSON. expected_head: the V3 repair-marker git HEAD; a bundle head
    is accepted when EQUAL to it, or when is_ancestor(expected_head, bundle_head)
    proves it a descendant (frozen engine modules stay LF-SHA byte-pinned).
    is_ancestor=None -> strict equality only (fail-closed default)."""
    spec = proj.get_spec(cid)
    out = {"candidate_id": cid, "eligible": False,
           "participant_status": "NOT_ELIGIBLE", "problems": [],
           "cert": None, "ready": None, "evaluation_by_scenario": None,
           "rule_tuple": None, "reuse_by_scenario": None, "gpu_uuids": [],
           "git_commit_head": None, "git_head_relation_to_marker": None,
           "spec": spec}
    f = out["problems"]
    formal_dir = os.path.join(cc4_dir, cid, "formal_evaluation_v3")
    if not os.path.isdir(formal_dir):
        out["participant_status"] = "NOT_ELIGIBLE_COMPLETE"
        f.append("formal_evaluation_v3 directory missing")
        return out
    out["formal_dir"] = formal_dir

    ready = proj.read_json(os.path.join(formal_dir, "READY_FORMAL_V3.json"))
    out["ready"] = ready
    _check(ready.get("candidate_id") == cid, f, "READY_CANDIDATE_ID")
    _check(ready.get("READY_FORMAL_V3") is True, f, "READY_FORMAL_V3_FALSE",
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
    _check((ready.get("gates") or {}).get("G16_GPU_V3_ONLY_STRICT") is True, f,
           "READY_G16_GPU_V3_ONLY_STRICT")

    cert = proj.read_json(os.path.join(formal_dir,
                                       "evaluation_certificate_v3.json"))
    out["cert"] = cert
    out["git_commit_head"] = (cert.get("provenance") or {}).get("git_commit_head")
    cert_problems = certmod.verify_evaluation_certificate(cert,
                                                          evidence_dir=formal_dir)
    _check(not cert_problems, f, "CERTIFICATE_VERIFY", str(cert_problems))
    _check(cert.get("common_pins") == certmod.PIN_FIELD_SOURCES, f,
           "CERT_PINS_DRIFT")
    _check(cert.get("taxonomy_v3_lf_sha256") == taxonomy_v3.module_lf_sha256(),
           f, "CERT_TAXONOMY_V3_SHA_DRIFT")
    _check(cert.get("common_evaluator_protocol_version")
           == COMMON_EVALUATOR_PROTOCOL_VERSION, f, "CERT_PROTOCOL_VERSION",
           str(cert.get("common_evaluator_protocol_version")))
    _check(cert.get("neg20_protocol") == NEG20_PROTOCOL, f, "CERT_NEG20_PROTOCOL",
           str(cert.get("neg20_protocol")))
    _check(cert.get("student_rank") is None, f, "CERT_STUDENT_RANK_NOT_NULL")
    _check(cert.get("teacher_included_in_student_ranking") is False, f,
           "CERT_TEACHER_FLAG")
    _check((cert.get("identity") or {}).get("candidate_id") == cid
           or cert.get("candidate_id") == cid, f, "CERT_CANDIDATE_ID")

    # V2 archive block: continuity only; the V2 evidence is NOT modified (G14).
    v2a = cert.get("v2_archive") or {}
    _check(v2a.get("v2_evidence_modified_by_v3") is False, f,
           "V2_ARCHIVE_MODIFIED_FLAG")
    if v2a.get("v2_summary_sha256") is not None:
        _check(v2a.get("v2_summary_sha256") == V2_ARCHIVE_SUMMARY_SHA256, f,
               "V2_ARCHIVE_SUMMARY_SHA_DRIFT")
    if v2a.get("v2_gate_sha256") is not None:
        _check(v2a.get("v2_gate_sha256") == V2_ARCHIVE_GATE_SHA256, f,
               "V2_ARCHIVE_GATE_SHA_DRIFT")

    # V3 repair-marker reference recorded by the bundle (G15 cross-check).
    bundle_marker = ((cert.get("audit") or {}).get("repair_authorization_marker")
                     or {})
    if marker_sha is not None:
        _check(bundle_marker.get("sha256") == marker_sha, f,
               "BUNDLE_MARKER_SHA_DRIFT",
               "bundle %r != ranking %r" % (bundle_marker.get("sha256"),
                                            marker_sha))

    # GPU discipline (G16): every bundle ran with GPU ⊆ {GPU2,GPU3}.
    gpu_uuids = (cert.get("gpu") or {}).get("visible_gpu_uuids") or []
    out["gpu_uuids"] = list(gpu_uuids)
    _check(bool(gpu_uuids) and all(u in V3_GPU_ALLOWED_UUIDS for u in gpu_uuids),
           f, "GPU_NOT_V3_ONLY", str(gpu_uuids))

    # per-arm reuse provenance (总控 §五 reuse chain) — completion gates G11/G12/G13.
    reuse = cert.get("reuse_provenance_by_scenario") or {}
    out["reuse_by_scenario"] = {sc: dict(reuse.get(sc) or {})
                                for sc in SCENARIO_ORDER}
    for sc in SCENARIO_ORDER:
        block = reuse.get(sc) or {}
        missing = [k for k in certmod.REUSE_PROVENANCE_REQUIRED_KEYS
                   if k not in block]
        _check(not missing, f, "REUSE_PROVENANCE_MISSING_KEYS %s" % sc,
               str(missing))
        _check(block.get("reuse_status") in certmod.REUSE_STATUS_VOCABULARY, f,
               "REUSE_STATUS_UNREGISTERED %s" % sc, str(block.get("reuse_status")))
    full_status = (reuse.get(FULL) or {}).get("reuse_status")
    front_status = (reuse.get(FRONT) or {}).get("reuse_status")
    back_status = (reuse.get(BACK) or {}).get("reuse_status")
    _check(full_status == REUSED_PASS, f, "FULL_REUSE_STATUS", str(full_status))
    _check(front_status == REUSED_RECLASSIFIED, f, "FRONT_REUSE_STATUS",
           str(front_status))
    _check(back_status in (COMPLETED, REUSED_RESIGNED), f, "BACK_REUSE_STATUS",
           str(back_status))
    if full_status == REJECT:
        # honest FULL reuse rejection: a block, never a silent rerun
        out["participant_status"] = "BLOCKED_REUSE_REJECT"

    if expected_head:
        head = out["git_commit_head"]
        if head == expected_head:
            out["git_head_relation_to_marker"] = "EQUAL"
        elif is_ancestor is not None and is_ancestor(expected_head, head):
            out["git_head_relation_to_marker"] = "DESCENDANT"
        else:
            out["git_head_relation_to_marker"] = "NOT_VERIFIED_DESCENDANT"
            f.append("GIT_HEAD: bundle head %r neither equals nor is a proven "
                     "descendant of the V3 repair-marker head %r"
                     % (head, expected_head))

    evaluation_by_scenario = {}
    counts_ok = True
    for sc in SCENARIO_ORDER:
        rp = os.path.join(formal_dir, "evaluation_result_v3.%s.json" % sc)
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
        if not out["participant_status"].startswith("BLOCKED"):
            try:
                out["rule_tuple"] = extract_rule_tuple(evaluation_by_scenario,
                                                       cid)
            except proj.FailClosed as exc:
                f.append("rule extraction: %s" % exc)

    # final participant status
    if ready.get("formal_abort") is not None or cert.get("formal_abort") is not None:
        out["participant_status"] = "BLOCKED_ENGINE_ABORT"
        f.append("formal_abort recorded (engine / retained fail-closed BLOCKED)")
    elif ready.get("evaluation_status") == "BLOCKED_REUSE_REJECT":
        out["participant_status"] = "BLOCKED_REUSE_REJECT"
    elif ready.get("evaluation_status") != "PASS":
        out["participant_status"] = "NOT_ELIGIBLE_COMPLETE"
        f.append("evaluation_status=%r" % ready.get("evaluation_status"))

    is_teacher = spec["candidate_class"] == "TEACHER_REFERENCE"
    if not f:
        out["eligible"] = True
        out["participant_status"] = ("TEACHER_REFERENCE_ONLY" if is_teacher
                                     else "ELIGIBLE_COMPLETE")
    return out


# ---------------------------------------------------------------------------
# READY closing flip (sole writer / creator of COMMON_EVALUATOR_V3_READY.json)
# ---------------------------------------------------------------------------
def apply_ready_flip_v3(pool_cc4_dir, marker_ref, summary_sha, gate_sha, now_utc,
                        flip_policy, auth_v3, ranking_status, formal_winner):
    """The single closing READY update. This tool is the SOLE writer/creator of
    <pool>/cc4/COMMON_EVALUATOR_V3_READY.json. If a prior file exists it must NOT
    already record FORMAL_RANKING_STARTED=true (rerun guard; the V3 driver start
    gate re-reads this and fail-closes after the flip). On a read-modify-write
    only allowlisted keys may change; on fresh creation the full document is
    written. Atomic via tmp + os.replace."""
    ready_path = os.path.join(pool_cc4_dir, READY_V3_NAME)
    before = {}
    if os.path.isfile(ready_path):
        before = proj.read_json(ready_path)
        proj.require(before.get("FORMAL_RANKING_STARTED") is not True,
                     "FAIL CLOSED (V3_READY_FLIP): FORMAL_RANKING_STARTED already "
                     "true — the V3 ranking close runs exactly once")
    closing = {
        "COMMON_EVALUATOR_V3_READY": True,
        "FORMAL_RANKING_STARTED": True,
        "FORMAL_RANKING_PUBLISHED": True,
        "FORMAL_RANKING_AUTHORIZED_V3": bool(auth_v3),
        "formal_evaluation_started_at_utc": marker_ref["recorded_at_utc"],
        "formal_ranking_published_at_utc": now_utc,
        "formal_ranking_summary_sha256": summary_sha,
        "formal_evaluation_gate_sha256": gate_sha,
        "flip_policy": flip_policy,
        "ranking_status": ranking_status,
        "formal_winner": formal_winner,
        "v3_repair_marker": {"path": marker_ref["path"],
                             "sha256": marker_ref["sha256"],
                             "verdict": marker_ref["verdict"],
                             "ruling_task": marker_ref["ruling_task"]},
        "pending_gates": [],
    }
    if before:
        after = dict(before)
        after.update(closing)
        changed = {k for k in set(before) | set(after)
                   if before.get(k) != after.get(k)}
        unexpected = changed - READY_V3_FLIP_ALLOWED_KEYS
        proj.require(not unexpected,
                     "FAIL CLOSED (V3_READY_FLIP): non-allowlisted keys would "
                     "change: %s" % sorted(unexpected))
    else:
        after = {
            "schema": READY_V3_SCHEMA,
            "common_evaluator_protocol_version":
                COMMON_EVALUATOR_PROTOCOL_VERSION,
            "neg20_protocol": NEG20_PROTOCOL,
            "run_class": RUN_CLASS,
            "single_training_seed": True,
            "scientific_claim_authorized": False,
            "teacher_included_in_student_ranking": False,
            "v2_archive": {"v2_status": certmod.V2_ARCHIVE_STATUS,
                           "v2_winner": certmod.V2_ARCHIVE_WINNER,
                           "v2_ranking_valid": False,
                           "v2_summary_sha256": V2_ARCHIVE_SUMMARY_SHA256,
                           "v2_gate_sha256": V2_ARCHIVE_GATE_SHA256,
                           "v2_evidence_modified_by_v3": False},
        }
        after.update(closing)
    tmp = ready_path + ".tmp_ranking_v3"
    import json
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(after, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, ready_path)
    return {"before_sha256": (proj.sha256_bytes(
                proj.canonical_json_bytes(before)) if before else None),
            "after_sha256": proj.sha256_file(ready_path),
            "created": not before,
            "changed_keys": sorted({k for k in set(before) | set(after)
                                    if before.get(k) != after.get(k)})}


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
    ap.add_argument("--v2-archive-dir", default=None,
                    help="optional dir holding the archived V2 summary/gate JSON; "
                         "if present they are re-hashed against the frozen pins")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify + report only; write neither summary nor READY")
    ap.add_argument("--repo", default=None,
                    help="git repo used for HEAD ancestry proofs (default: the "
                         "toplevel of the current directory)")
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
    rule_source = {"schema_file": None, "verified_verbatim": False,
                   "machinery_source": "tier3_formal_ranking_v2dt (imported "
                                        "verbatim; object identity)"}
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
        rule_source["schema_file"] = schema_path
        rule_source["schema_sha256"] = smokev2.FROZEN_V2_METRIC_SCHEMA_SHA256
        rule_source["verified_verbatim"] = True

    # --- V3 repair-authorization marker (start authorization) ------------------
    # verify_v3_repair_start ALSO enforces the rerun guard: if READY_V3 already
    # records FORMAL_RANKING_STARTED=true it fail-closes here, before any work.
    marker_ref = driver.verify_v3_repair_start(args.common_dir, pool_cc4_dir)
    marker_json = proj.read_json(marker_ref["path"])
    expected_head = marker_ref["git_commit_head"]

    # --- git head policy: uniform + equal-or-descendant of the marker head -----
    repo_dir = args.repo
    if repo_dir is None:
        try:
            repo_dir = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            repo_dir = None

    def is_ancestor(ancestor, descendant):
        if not repo_dir or not ancestor or not descendant:
            return False
        return subprocess.run(
            ["git", "-C", repo_dir, "merge-base", "--is-ancestor",
             ancestor, descendant], capture_output=True).returncode == 0

    closing_head = None
    if repo_dir:
        try:
            closing_head = subprocess.run(
                ["git", "-C", repo_dir, "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            closing_head = None

    # --- V2 archive untouched (G14): live re-hash if locatable -----------------
    v2_archive_live = {"checked": False, "summary_match": None, "gate_match": None}
    if args.v2_archive_dir and os.path.isdir(args.v2_archive_dir):
        v2sum = os.path.join(args.v2_archive_dir, "FORMAL_RANKING_SUMMARY_V2DT.json")
        v2gate = os.path.join(args.v2_archive_dir, "FORMAL_EVALUATION_GATE_V2DT.json")
        if os.path.isfile(v2sum) and os.path.isfile(v2gate):
            v2_archive_live = {
                "checked": True,
                "summary_match": proj.sha256_file(v2sum) == V2_ARCHIVE_SUMMARY_SHA256,
                "gate_match": proj.sha256_file(v2gate) == V2_ARCHIVE_GATE_SHA256}
            proj.require(v2_archive_live["summary_match"]
                         and v2_archive_live["gate_match"],
                         "FAIL CLOSED (V2_ARCHIVE): archived V2 summary/gate sha "
                         "drifted — V2 evidence must NOT be modified: %s"
                         % v2_archive_live)

    # --- verify all seven V3 bundles ------------------------------------------
    bundles = [verify_candidate_bundle(cid, pool_cc4_dir, expected_head,
                                       is_ancestor, marker_ref["sha256"])
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

    # --- ranking (frozen machinery, verbatim) ---------------------------------
    ranks, ranking_status, inconclusive, ordered_ids = ({}, "INCONCLUSIVE_"
        "PARTICIPATION", [], [])
    if len(eligible_students) == len(STUDENTS):
        ranks, ranking_status, inconclusive, ordered_ids = rank_students(
            [{"candidate_id": b["candidate_id"], "rule_tuple": b["rule_tuple"]}
             for b in eligible_students])
    formal_winner = (ordered_ids[0] if ranking_status == "ORDERED" else None)

    participants = []
    for b in bundles:
        cid = b["candidate_id"]
        is_teacher_c = b["spec"]["candidate_class"] == "TEACHER_REFERENCE"
        t = b["rule_tuple"]
        cert = b["cert"] or {}
        disclosure = cert.get("composite_event_disclosure") or {}
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
            "reuse_status_by_scenario": b["reuse_by_scenario"] and {
                sc: (b["reuse_by_scenario"][sc] or {}).get("reuse_status")
                for sc in SCENARIO_ORDER},
            "composite_episode_count_by_scenario":
                disclosure.get("composite_episode_count_by_scenario"),
            "secondary_event_counts_by_scenario":
                disclosure.get("secondary_event_counts_by_scenario"),
            "problems": b["problems"],
            "formal_dir": b.get("formal_dir"),
            "gpu_uuids": b["gpu_uuids"],
            "certificate_sha256": (proj.sha256_file(os.path.join(
                b["formal_dir"], "evaluation_certificate_v3.json"))
                if b.get("formal_dir") and os.path.isfile(os.path.join(
                    b["formal_dir"], "evaluation_certificate_v3.json"))
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
    execution_head = sorted(heads)[0] if len(heads) == 1 else None
    heads_descend = bool(heads) and all(
        b["git_head_relation_to_marker"] in ("EQUAL", "DESCENDANT")
        for b in bundles if b["git_commit_head"])
    if not heads_descend:
        gate_failures.append(
            "GIT_HEAD_NOT_EQUAL_OR_DESCENDED_FROM_MARKER: %s" % sorted(
                "%s=%s" % (b["candidate_id"], b["git_head_relation_to_marker"])
                for b in bundles))
    commits_after_marker = []
    if (repo_dir and execution_head and expected_head
            and execution_head != expected_head):
        try:
            commits_after_marker = [
                ln for ln in subprocess.run(
                    ["git", "-C", repo_dir, "log", "--oneline",
                     "%s..%s" % (expected_head, execution_head)],
                    capture_output=True, text=True,
                    check=True).stdout.splitlines() if ln.strip()]
        except (OSError, subprocess.CalledProcessError):
            commits_after_marker = None

    # per-arm completion aggregates (G11/G12/G13)
    def _reuse_status(b, sc):
        return ((b["reuse_by_scenario"] or {}).get(sc) or {}).get("reuse_status")
    full_pass_all = all(_reuse_status(b, FULL) == REUSED_PASS
                        for b in bundles if b["cert"])
    front_reclass_all = all(_reuse_status(b, FRONT) == REUSED_RECLASSIFIED
                            for b in bundles if b["cert"])
    back_complete_all = all(_reuse_status(b, BACK) in (COMPLETED, REUSED_RESIGNED)
                            for b in bundles if b["cert"])
    v2_untouched_all = all(
        ((b["cert"] or {}).get("v2_archive") or {}).get(
            "v2_evidence_modified_by_v3") is False for b in bundles if b["cert"])
    if v2_archive_live["checked"]:
        v2_untouched_all = v2_untouched_all and v2_archive_live["summary_match"] \
            and v2_archive_live["gate_match"]
    marker_uniform = all(
        (((b["cert"] or {}).get("audit") or {}).get(
            "repair_authorization_marker") or {}).get("sha256")
        == marker_ref["sha256"] for b in bundles if b["cert"])
    gpu_v3_all = all(bool(b["gpu_uuids"]) and all(
        u in V3_GPU_ALLOWED_UUIDS for u in b["gpu_uuids"]) for b in bundles)

    gates = {
        "G1_ALL_6_STUDENTS_ELIGIBLE_COMPLETE":
            len(eligible_students) == len(STUDENTS),
        "G2_TEACHER_REFERENCE_COMPLETE": teacher_ok,
        "G3_NO_ENGINE_ABORT": all(
            not b["participant_status"].startswith("BLOCKED") for b in bundles),
        "G4_NO_REHEARSAL_IN_FORMAL_POOL": all(
            (b["ready"] or {}).get("rehearsal") is False for b in bundles),
        "G5_CERTIFICATES_ALL_VERIFY": not any(
            "CERTIFICATE_VERIFY" in p for b in bundles for p in b["problems"]),
        "G6_PINS_UNIFORM_FROZEN": all(
            (b["cert"] or {}).get("common_pins") == certmod.PIN_FIELD_SOURCES
            and (b["cert"] or {}).get("taxonomy_v3_lf_sha256")
            == taxonomy_v3.module_lf_sha256() for b in bundles if b["cert"]),
        "G7_GIT_HEAD_UNIFORM": heads_uniform,
        "G7b_GIT_HEAD_EQUAL_OR_DESCENDED_FROM_MARKER": heads_descend,
        "G8_RULE_VERBATIM": rule_source.get("verified_verbatim", False)
            or not os.path.isfile(schema_path),
        "G9_REGISTRY_RANK_NULL": all(
            b["spec"].get("student_rank") is None for b in bundles),
        "G10_RANKING_COMPUTED_HONEST": ranking_status in (
            "ORDERED", "INCONCLUSIVE_FULL_TIE", "INCONCLUSIVE_PARTICIPATION"),
        "G11_FULL_REUSED_PASS_X7": full_pass_all,
        "G12_FRONT_RECLASSIFIED_PROVENANCE_X7": front_reclass_all,
        "G13_BACK_COMPLETE_OR_REUSED_X7": back_complete_all,
        "G14_V2_ARCHIVE_UNTOUCHED": v2_untouched_all,
        "G15_V3_REPAIR_MARKER_VERIFIED": marker_uniform,
        "G16_GPU_V3_ONLY": gpu_v3_all,
    }
    gate_pass = all(gates.values()) and not gate_failures
    # FORMAL_RANKING_AUTHORIZED_V3 is STRICT: true iff EVERY gate (incl. the
    # participation/completion gates) is green — i.e. all 6 students + teacher
    # complete, no ENGINE_BLOCKED, no candidate-level exemption. Any honest block
    # keeps it false.
    auth_v3 = gate_pass

    # --- flip policy -----------------------------------------------------------
    blocked_ids = {b["candidate_id"] for b in bundles
                   if b["participant_status"].startswith("BLOCKED")}
    foreign_failures = [g for g in gate_failures
                        if g.split(": ", 1)[0] not in blocked_ids]
    integrity_ok = all(gates[k] for k in INTEGRITY_GATE_KEYS)
    if gate_pass:
        flip_policy = "V3_GATE_GREEN"
    elif integrity_ok and not foreign_failures:
        flip_policy = "V3_PUBLISH_HONEST_UNDER_COMPLETION_BLOCK"
    else:
        flip_policy = "NO_FLIP"

    now_utc = smokev2.utc_now_iso()
    summary = {
        "schema": SUMMARY_SCHEMA,
        "generated_at_utc": now_utc,
        "common_evaluator_protocol_version": COMMON_EVALUATOR_PROTOCOL_VERSION,
        "neg20_protocol": NEG20_PROTOCOL,
        "run_class": RUN_CLASS,
        "v3_repair_marker": {"path": marker_ref["path"],
                             "sha256": marker_ref["sha256"],
                             "verdict": marker_ref["verdict"],
                             "ruling_task": marker_ref["ruling_task"]},
        "v2_archive": {"v2_status": certmod.V2_ARCHIVE_STATUS,
                       "v2_winner": certmod.V2_ARCHIVE_WINNER,
                       "v2_ranking_valid": False,
                       "v2_summary_sha256": V2_ARCHIVE_SUMMARY_SHA256,
                       "v2_gate_sha256": V2_ARCHIVE_GATE_SHA256,
                       "v2_evidence_modified_by_v3": False,
                       "v2_archive_live_rehash": v2_archive_live,
                       "note": "V2 is archived CLOSED_INCONCLUSIVE_PARTICIPATION "
                               "under the frozen single-label classifier; V3 is a "
                               "distinct evaluator semantic repair and does not "
                               "overwrite/delete/rewrite any V2 evidence"},
        "FORMAL_RANKING_AUTHORIZED_V3": auth_v3,
        "formal_winner": formal_winner,
        "git_head_policy": {
            "marker_git_commit_head": expected_head,
            "execution_git_commit_head": execution_head,
            "closing_git_commit_head": closing_head,
            "execution_heads_uniform": heads_uniform,
            "execution_heads_equal_or_descended_from_marker": heads_descend,
            "bundle_head_relations": sorted(
                "%s=%s" % (b["candidate_id"], b["git_head_relation_to_marker"])
                for b in bundles),
            "commits_marker_to_execution_oneline": commits_after_marker,
            "behavioral_identity_basis": (
                "frozen engine modules are LF-SHA byte-pinned and re-verified per "
                "bundle via certificate pins/engine identity; the V3 semantic "
                "repair touches ONLY the taxonomy representation / composite-event "
                "classification / certificate fields — never the frozen engine, "
                "banks, seeds, episode counts, horizon, or ranking rule"),
        },
        "selection_predicate_rule": {
            "order": FROZEN_RULE_ORDER,
            "tie_tolerance": SELECTION_TIE_TOLERANCE,
            "all_equal_result": FROZEN_RULE_ALL_EQUAL_RESULT,
            "source": rule_source,
            "note": "frozen lexicographic comparator from metric_schema.json "
                    "(sha-pinned), imported verbatim from tier3_formal_ranking_v2dt; "
                    "PROVISIONAL-class rule; no scientific-superiority claim, "
                    "multi-seed confirmation required"},
        "ranking_status": ranking_status,
        "inconclusive_groups": inconclusive,
        "ordered_by_strict_rule": ordered_ids,
        "top_ranked_student_id": formal_winner,
        "participants": participants,
        "student_count_eligible": "%d/%d" % (len(eligible_students),
                                             len(STUDENTS)),
        "teacher_included_in_student_ranking": False,
        "composite_event_summary": {
            "total_composite_episodes": sum(
                int(((p["composite_episode_count_by_scenario"] or {}).get(sc) or 0))
                for p in participants
                for sc in SCENARIO_ORDER),
            "disclosure": "every FRONT transition∧defeat_kobold episode is a "
                          "VALID_COMPOSITE_EVENT (primary=FRONT_TRANSITION_SUCCESS, "
                          "secondary includes DEFEAT_KOBOLD); no such episode is "
                          "FailClosed under V3",
        },
        "scientific_claim_authorized": False,
        "scientific_claim_status":
            "FORMAL_SCIENTIFIC_CLAIM: NOT_AUTHORIZED_SINGLE_TRAINING_SEED",
        "scaffolded_results_can_replace_full_task": False,
        "interface_smoke_substituted_for_performance": False,
        "gate_failures": gate_failures,
        "escalation": (
            "INCONCLUSIVE_PARTICIPATION: fewer than 6 eligible students — escalate "
            "to 总控 before any selection claim"
            if ranking_status == "INCONCLUSIVE_PARTICIPATION"
            else ("FORMAL_RANKING_AUTHORIZED_V3=false under an honest completion "
                  "block — no winner; escalate to 总控" if not auth_v3 else None)),
    }
    overclaim = certmod.scan_forbidden_overclaims(summary)
    proj.require(not overclaim,
                 "FAIL CLOSED (OVERCLAIM_SCAN): summary contains %s" % overclaim)
    masquerade = scan_no_v2dt_ranking_masquerade(summary, SUMMARY_SCHEMA)
    proj.require(not masquerade,
                 "FAIL CLOSED (MASQUERADE_SCAN): summary contains %s" % masquerade)

    if args.dry_run:
        print("[dry-run] ranking_status=%s gate_pass=%s AUTH_V3=%s eligible=%d/%d "
              "teacher_ok=%s flip_policy=%s winner=%s"
              % (ranking_status, gate_pass, auth_v3, len(eligible_students),
                 len(STUDENTS), teacher_ok, flip_policy, formal_winner), flush=True)
        print("[dry-run] git_head_policy: marker=%s execution=%s uniform=%s "
              "descend=%s" % (expected_head, execution_head, heads_uniform,
                              heads_descend), flush=True)
        for p in participants:
            print("  %-42s status=%-24s rank=%s tuple=%s"
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
        "neg20_protocol": NEG20_PROTOCOL,
        "FORMAL_EVALUATION_GATE_V3_PASS": gate_pass,
        "FORMAL_RANKING_AUTHORIZED_V3": auth_v3,
        "gates": gates,
        "gate_failures": gate_failures,
        "flip_policy": flip_policy,
        "integrity_gates": {k: gates[k] for k in INTEGRITY_GATE_KEYS},
        "foreign_gate_failures": foreign_failures,
        "blocked_candidate_ids": sorted(blocked_ids),
        "ranking_status": ranking_status,
        "formal_winner": formal_winner,
        "student_common_eligible_count": "%d/%d" % (len(eligible_students),
                                                    len(STUDENTS)),
        "teacher_reference_binding": "PASS" if teacher_ok else "FAIL",
        "formal_ranking_summary_sha256": summary_sha,
        "v3_repair_marker_sha256": marker_ref["sha256"],
        "v2_archive": {"v2_status": certmod.V2_ARCHIVE_STATUS,
                       "v2_winner": certmod.V2_ARCHIVE_WINNER,
                       "v2_summary_sha256": V2_ARCHIVE_SUMMARY_SHA256,
                       "v2_gate_sha256": V2_ARCHIVE_GATE_SHA256,
                       "v2_evidence_modified_by_v3": False},
        "git_heads_uniform": sorted(heads),
        "server_git_head": sorted(heads)[0] if heads else None,
        "CHECKPOINTS_MODIFIED": False,
        "STUDENTS_RETRAINED": 0,
        "CONTROL_RETRAINED": False,
        "CANDIDATE_EXCEPTIONS_USED": 0,
        "CANDIDATE_EXCEPTION_USED": False,
        "FULL_ONLY_RANKING_USED": False,
        "BACK_ONLY_RANKING_USED": False,
        "FROZEN_BANKS_MODIFIED": False,
        "RETRAINING_PERFORMED": False,
        "honest_discipline": "READY_V3 flip authorized only under V3_GATE_GREEN "
            "(all gates true => AUTH_V3=true) or V3_PUBLISH_HONEST_UNDER_COMPLETION_"
            "BLOCK (all integrity gates true; every remaining failure originates "
            "solely from BLOCKED candidates => AUTH_V3=false, winner=null, STRICTER "
            "than V2); any integrity failure => NO_FLIP. BLOCKED candidates never "
            "counted; teacher never ranked; a FULL reuse REJECT is an honest block, "
            "never a silent rerun; the V2 evidence is never modified",
    }
    gate_overclaim = certmod.scan_forbidden_overclaims(gate)
    proj.require(not gate_overclaim,
                 "FAIL CLOSED (OVERCLAIM_SCAN): gate contains %s" % gate_overclaim)
    gate_masquerade = scan_no_v2dt_ranking_masquerade(gate, GATE_SCHEMA)
    proj.require(not gate_masquerade,
                 "FAIL CLOSED (MASQUERADE_SCAN): gate contains %s" % gate_masquerade)
    gate_path = os.path.join(out_dir, GATE_NAME)
    smokev2.write_json(gate_path, gate)
    gate_sha = proj.sha256_file(gate_path)
    with open(gate_path + ".sha256", "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write("%s  %s\n" % (gate_sha, GATE_NAME))
    print("[ranking-v3] summary=%s sha=%s" % (summary_path, summary_sha),
          flush=True)
    print("[ranking-v3] gate=%s PASS=%s AUTH_V3=%s ranking_status=%s winner=%s"
          % (gate_path, gate_pass, auth_v3, ranking_status, formal_winner),
          flush=True)

    if flip_policy == "NO_FLIP":
        print("[ranking-v3] FLIP REFUSED — integrity failure; failures:",
              flush=True)
        for g in gate_failures:
            print("  -", g, flush=True)
        return 2

    flip_ev = apply_ready_flip_v3(pool_cc4_dir, marker_ref, summary_sha, gate_sha,
                                  now_utc, flip_policy, auth_v3, ranking_status,
                                  formal_winner)
    print("[ranking-v3] READY_V3 closing flip applied: policy=%s AUTH_V3=%s "
          "created=%s after_sha=%s" % (flip_policy, auth_v3, flip_ev["created"],
                                       flip_ev["after_sha256"][:16]), flush=True)
    print("[ranking-v3] FORMAL_RANKING_PUBLISHED — V3 driver now fail-closes on any "
          "rerun (FORMAL_RANKING_STARTED=true)", flush=True)
    return 0


# ---------------------------------------------------------------------------
# self-test (JAX-free synthetic V3 bundles)
# ---------------------------------------------------------------------------
def _synthetic_v3_evaluation(scenario, successes, valid_starts, dense_value=None,
                             defeat_count=None, composite_count=0):
    """Mirrors tier3_taxonomy_v3.summarize_v3 output: the FROZEN tier3_metrics
    envelope nested under 'metrics' (bit-identical by construction) plus the
    additive composite-event layer. The reused V2 extractor reads ['metrics']."""
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
                                                       if defeat_count is not None
                                                       else successes),
                         "defeat_count": (defeat_count if defeat_count is not None
                                          else successes),
                         "mean_timesteps": 100.0,
                         "max_timesteps_observed": 200},
            "failure_taxonomy": {}}
    return {
        "schema": "mechanism_UED.tier3_taxonomy_v3_summary/v1",
        "scenario": scenario,
        "formal_evaluator_protocol": COMMON_EVALUATOR_PROTOCOL_VERSION,
        "neg20_protocol": NEG20_PROTOCOL,
        "taxonomy_version_v3": taxonomy_v3.TAXONOMY_V3_VERSION,
        "failure_rule_version_v3": taxonomy_v3.FAILURE_RULE_VERSION_V3,
        "episode_count": valid_starts,
        "valid_start_count": valid_starts,
        "metrics": metrics,
        "terminal_label_counts": {},
        "composite_event_layer": {
            "per_episode": [],
            "primary_outcome_counts": {},
            "secondary_event_counts": ({"DEFEAT_KOBOLD": composite_count}
                                       if composite_count else {}),
            "composite_episode_count": composite_count,
            "secondary_event_vocabulary":
                sorted(taxonomy_v3.SECONDARY_EVENT_VOCABULARY),
        },
        "scaffolded_results_can_replace_full_task": False,
    }


def _synthetic_reuse(status, classification_only, environment_rerun,
                     source_v2_sha="a" * 64):
    return {
        "reuse_status": status,
        "source": ("V2_COMMITTED_EVIDENCE" if source_v2_sha is not None
                   else "V3_FRESH_COMPLETION_RUN"),
        "classification_only": classification_only,
        "environment_rerun": environment_rerun,
        "v3_classifier_sha256": taxonomy_v3.module_lf_sha256(),
        "v3_result_sha256": "b" * 64,
        "source_v2_episode_sha256": source_v2_sha,
    }


def _build_synthetic_bundle(cc4_dir, cid, rule_tuple, git_head, marker_sha,
                            rehearsal=False, abort=False, corrupt_pin=False,
                            full_reject=False, gpu_banned=False,
                            back_status=None):
    """Build a complete, self-consistent synthetic formal_evaluation_v3 bundle."""
    d = os.path.join(cc4_dir, cid, "formal_evaluation_v3")
    os.makedirs(d, exist_ok=True)
    spec = proj.get_spec(cid)
    is_teacher = spec["candidate_class"] == "TEACHER_REFERENCE"
    t = rule_tuple
    evaluation_by_scenario = {
        "full": _synthetic_v3_evaluation("full", t[0], 64),
        "front_l2": _synthetic_v3_evaluation("front_l2", t[1], 8,
                                             dense_value=t[2],
                                             composite_count=t[1]),
        "back_l2": _synthetic_v3_evaluation("back_l2", t[3], 8, defeat_count=t[3]),
    }
    # episode records jsonl (synthetic) + sums-consistent files
    jsonl_lines = [proj.canonical_json_bytes(
        {"candidate_id": cid, "scenario": sc, "synthetic": True}).decode()
        for sc in SCENARIO_ORDER]
    jsonl_bytes = ("\n".join(jsonl_lines) + "\n").encode("utf-8")
    with open(os.path.join(d, "episode_records.jsonl"), "wb") as fh:
        fh.write(jsonl_bytes)
    jsonl_sha = proj.sha256_bytes(jsonl_bytes)
    smokev2.write_json(os.path.join(d, "provenance_v3.json"),
                       {"schema": driver.PROVENANCE_SCHEMA, "pid": 1, "argv": [],
                        "cwd": "/", "host": "synthetic",
                        "git_commit_head": git_head,
                        "generated_at_utc": "1970-01-01T00:00:00+00:00"})
    # per-arm reuse provenance: FULL reused, FRONT reclassified, BACK completed
    # (fresh) for the originally-blocked students / teacher, RESIGNED for CONTROL.
    if back_status is None:
        back_status = REUSED_RESIGNED if cid == "CONTROL_CONTINUOUS_98304" \
            else COMPLETED
    reuse_prov = {
        "full": _synthetic_reuse(REJECT if full_reject else REUSED_PASS,
                                 True, False),
        "front_l2": _synthetic_reuse(REUSED_RECLASSIFIED, True, False),
        "back_l2": _synthetic_reuse(back_status,
                                    back_status == REUSED_RESIGNED,
                                    back_status == COMPLETED,
                                    source_v2_sha=("a" * 64
                                                   if back_status == REUSED_RESIGNED
                                                   else None)),
    }
    counts = ({"full": 0, "front_l2": 0, "back_l2": 0} if abort
              else {"full": 64, "front_l2": 8, "back_l2": 8})
    for sc in SCENARIO_ORDER:
        smokev2.write_json(os.path.join(
            d, "evaluation_result_v3.%s.json" % sc),
            {"schema": driver.RESULT_SCHEMA, "candidate_id": cid,
             "scenario": sc, "run_class": RUN_CLASS,
             "common_evaluator_protocol_version":
                 COMMON_EVALUATOR_PROTOCOL_VERSION,
             "neg20_protocol": NEG20_PROTOCOL,
             "rehearsal": rehearsal,
             "schedule": {"seeds": list(range(FORMAL_COUNTS[sc]))},
             "entry_ids_planned": ["%s-bank%d" % (sc, i)
                                   for i in range(FORMAL_COUNTS[sc])],
             "episodes_planned": FORMAL_COUNTS[sc],
             "episodes_executed": FORMAL_COUNTS[sc],
             "episode_records_sha256": jsonl_sha,
             "aborted_in_scenario": False,
             "evaluation": evaluation_by_scenario[sc],
             "reuse_provenance": reuse_prov[sc],
             "timing": {"episodes": [], "scenario_wall_seconds": 1.0,
                        "peak_rss_kb": None},
             "generated_at_utc": "1970-01-01T00:00:00+00:00"})
    # certificate via the real builder (pins, taxonomy LF-SHA, honest labels,
    # rank=null, per-arm reuse provenance, V2 archive, no-V2-masquerade)
    ci = certmod._sample_cert_input(candidate_id=cid)
    ci["results_by_scenario"] = ({} if abort else evaluation_by_scenario)
    ci["episode_records_jsonl_sha256"] = jsonl_sha
    ci["records_sha256_by_scenario"] = {sc: jsonl_sha for sc in SCENARIO_ORDER}
    ci["episodes_executed"] = counts
    ci["valid_start_counts"] = counts
    ci["generated_at_utc"] = "1970-01-01T00:00:00+00:00"
    ci["provenance"] = {"pid": 1, "argv": [], "cwd": "/", "host": "synthetic",
                        "git_commit_head": git_head,
                        "device_identity": {"synthetic": True},
                        "scenario_wall_seconds": {},
                        "timing_by_scenario": {}}
    banned = [u for u in proj.CC4_GPU_ALLOWED_UUIDS
              if u not in V3_GPU_ALLOWED_UUIDS]
    ci["gpu_ev"] = {"visible_gpu_uuids": [banned[0]] if gpu_banned
                    else [V3_GPU_ALLOWED_UUIDS[0]],
                    "g16_gpu_v3_only": not gpu_banned}
    ci["rehearsal"] = rehearsal
    ci["formal_abort"] = ({"verdict": "ENGINE_TAXONOMY_REJECTED_FORMAL_"
                                       "ROLLOUT_V3", "scenario": "back_l2",
                           "episode_index": 0} if abort else None)
    ci["reuse_provenance_by_scenario"] = reuse_prov
    ci["composite_event_disclosure"] = {
        "composite_episode_count_by_scenario": {
            sc: int(evaluation_by_scenario[sc]["composite_event_layer"]
                    ["composite_episode_count"]) for sc in SCENARIO_ORDER},
        "secondary_event_counts_by_scenario": {
            sc: evaluation_by_scenario[sc]["composite_event_layer"]
            ["secondary_event_counts"] for sc in SCENARIO_ORDER},
        "primary_outcome_counts_by_scenario": {sc: {} for sc in SCENARIO_ORDER},
    }
    ci["v2_archive_summary_sha256"] = V2_ARCHIVE_SUMMARY_SHA256
    ci["v2_archive_gate_sha256"] = V2_ARCHIVE_GATE_SHA256
    ci["marker_ref"] = {"path": os.path.join(cc4_dir,
                                             driver.V3_REPAIR_MARKER_NAME),
                        "sha256": marker_sha,
                        "ruling_task": driver.V3_REPAIR_RULING_TASK,
                        "verdict": driver.V3_REPAIR_VERDICT,
                        "recorded_at_utc": "1970-01-01T00:00:01+00:00"}
    cert = certmod.build_evaluation_certificate(ci)
    if corrupt_pin:
        cert["common_pins"]["common_evaluator_sha256"] = "0" * 64
    smokev2.write_json(os.path.join(d, "evaluation_certificate_v3.json"), cert)
    # SHA256SUMS_FORMAL_V3 over the same six files the driver sums
    summed = ["episode_records.jsonl", "provenance_v3.json",
              "evaluation_certificate_v3.json"] + [
        "evaluation_result_v3.%s.json" % sc for sc in SCENARIO_ORDER]
    with open(os.path.join(d, certmod.SUMS_FILENAME), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join("%s  %s" % (proj.sha256_file(os.path.join(d, fn)), fn)
                           for fn in summed) + "\n")
    # READY_FORMAL_V3
    gates = {"G1_CAPSULE_FILE_SHA_MATCH": True,
             "G2_CHECKPOINT_FILE_SHA_OWNER_RECOMPUTE_MATCH": True,
             "G3_PARAMS_SHA_OWNER_RECOMPUTE_MATCH": True,
             "G4_FORMAL_SCHEDULE_COMPLETE": not abort,
             "G5_PARAMS_UNCHANGED": True,
             "G6_COMMON_V2_SUMS": True,
             "G7_EPISODE_COUNTS_FROM_PROFILE": True,
             "G8_GPU_V3_ONLY": not gpu_banned,
             "G9_V1_FROZEN_PRESERVED": True,
             "G10_PROTOCOL_VERSION_V3": True,
             "G11_V3_REPAIR_MARKER_VERIFIED": True,
             "G12_CERTIFICATE_VERIFIED": not corrupt_pin,
             "G13_FULL_REUSED_PASS": not full_reject,
             "G14_FRONT_RECLASSIFIED": True,
             "G15_BACK_COMPLETE_OR_REUSED": not abort,
             "G16_GPU_V3_ONLY_STRICT": not gpu_banned}
    status = ("BLOCKED" if abort else "BLOCKED_REUSE_REJECT" if full_reject
              else "REHEARSAL_NOT_FORMAL" if rehearsal else "PASS")
    smokev2.write_json(os.path.join(d, "READY_FORMAL_V3.json"),
                       {"schema": driver.READY_FORMAL_SCHEMA,
                        "candidate_id": cid,
                        "runtime_family": spec["runtime_family"],
                        "common_evaluator_protocol_version":
                            COMMON_EVALUATOR_PROTOCOL_VERSION,
                        "neg20_protocol": NEG20_PROTOCOL,
                        "READY_FORMAL_V3": all(gates.values()),
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
                        "v3_repair_marker": ci["marker_ref"],
                        "gates": gates,
                        "generated_at_utc": "1970-01-01T00:00:00+00:00",
                        "evidence_files": summed + [certmod.SUMS_FILENAME],
                        "honest_false_discipline": "synthetic"})
    return d


def _write_marker(cc4_dir, git_head):
    """Write the V3 repair marker via the real marker tool (single source of
    truth) and return (path, sha)."""
    import tier3_formal_start_marker_v3 as marker_v3
    sha = marker_v3.write_marker(cc4_dir, recorded_at_utc="1970-01-01T00:00:01+00:00",
                                 git_head=git_head)
    return os.path.join(cc4_dir, driver.V3_REPAIR_MARKER_NAME), sha


def run_self_test():
    import tempfile
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        checks += 1
        proj.require(cond, "RANKING_V3_SELF_TEST FAIL: %s" % msg)

    # pure rule machinery (imported verbatim from ranking_v2 — identity check)
    ok(extract_rule_tuple is ranking_v2.extract_rule_tuple
       and compare_rule_tuples is ranking_v2.compare_rule_tuples
       and rank_students is ranking_v2.rank_students
       and FROZEN_RULE_ORDER is ranking_v2.FROZEN_RULE_ORDER
       and SELECTION_TIE_TOLERANCE == 1e-12,
       "frozen ranking machinery reused by object identity (G8 verbatim)")
    a = (10, 5, 0.5, 3)
    ok(compare_rule_tuples((11, 0, 0.0, 0), a) == -1, "level1 dominates")
    ok(compare_rule_tuples((10, 5, 0.5 + 2e-12, 0), a) == -1, "2e-12 ordered")
    ok(compare_rule_tuples((10, 5, 0.5 + 5e-13, 3), a) == 0,
       "full tie within tolerance -> 0")

    with tempfile.TemporaryDirectory() as td:
        cc4 = os.path.join(td, "cc4")
        common = os.path.join(td, "common_v2")
        os.makedirs(cc4)
        os.makedirs(common)
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
        _, marker_sha = _write_marker(cc4, head)
        for cid, t in tuples.items():
            _build_synthetic_bundle(cc4, cid, t, head, marker_sha)

        # happy path (dry run): all 6 students + teacher complete -> gate green
        rc = main(["--pool-cc4-dir", cc4, "--common-dir", common, "--dry-run"])
        checks += 1
        proj.require(rc == 0, "RANKING_V3_SELF_TEST: dry-run rc=%d" % rc)
        ok(not os.path.isfile(os.path.join(cc4, READY_V3_NAME)),
           "dry-run must not create READY_V3")

        # bundle verifier sees everything (CONTROL back is REUSED_RESIGNED)
        b0 = verify_candidate_bundle(STUDENTS[0], cc4, head, None, marker_sha)
        ok(b0["eligible"] and b0["participant_status"] == "ELIGIBLE_COMPLETE"
           and b0["rule_tuple"] == tuples[STUDENTS[0]]
           and b0["reuse_by_scenario"]["full"]["reuse_status"] == REUSED_PASS
           and b0["reuse_by_scenario"]["front_l2"]["reuse_status"]
           == REUSED_RECLASSIFIED
           and b0["reuse_by_scenario"]["back_l2"]["reuse_status"] == COMPLETED,
           "student bundle verify clean + reuse chain")
        bt = verify_candidate_bundle(TEACHER, cc4, head, None, marker_sha)
        ok(bt["eligible"] and bt["participant_status"] == "TEACHER_REFERENCE_ONLY",
           "teacher reference-only")

        # real run: summary + gate + flip (expected V3_GATE_GREEN)
        rc = main(["--pool-cc4-dir", cc4, "--common-dir", common, "--repo", td])
        checks += 1
        proj.require(rc == 0, "RANKING_V3_SELF_TEST: run rc=%d" % rc)
        summary = proj.read_json(os.path.join(cc4, SUMMARY_NAME))
        ok(summary["ranking_status"] == "ORDERED", "ordered")
        ok(summary["FORMAL_RANKING_AUTHORIZED_V3"] is True, "AUTH_V3 true")
        ok(summary["formal_winner"] == STUDENTS[0]
           and summary["top_ranked_student_id"] == STUDENTS[0],
           "top student (teacher strongest yet excluded)")
        ok(summary["v2_archive"]["v2_status"]
           == "CLOSED_INCONCLUSIVE_PARTICIPATION"
           and summary["v2_archive"]["v2_winner"] is None
           and summary["v2_archive"]["v2_evidence_modified_by_v3"] is False,
           "V2 archive reference recorded, untouched")
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
        ok(summary["composite_event_summary"]["total_composite_episodes"] > 0,
           "composite-event disclosure present")
        ok(not certmod.scan_forbidden_overclaims(summary)
           and not scan_no_v2dt_ranking_masquerade(summary, SUMMARY_SCHEMA),
           "overclaim+masquerade (V3 summary, no V2DT ranking schema strings)")
        side = open(os.path.join(cc4, SUMMARY_NAME + ".sha256"),
                    encoding="utf-8").read().split()[0]
        ok(side == proj.sha256_file(os.path.join(cc4, SUMMARY_NAME)),
           "summary sidecar sha")
        gate = proj.read_json(os.path.join(cc4, GATE_NAME))
        ok(gate["FORMAL_EVALUATION_GATE_V3_PASS"] and all(
            gate["gates"].values()) and gate["gate_failures"] == []
           and gate["FORMAL_RANKING_AUTHORIZED_V3"] is True
           and gate["flip_policy"] == "V3_GATE_GREEN"
           and gate["CHECKPOINTS_MODIFIED"] is False
           and gate["STUDENTS_RETRAINED"] == 0
           and gate["CANDIDATE_EXCEPTIONS_USED"] == 0
           and gate["FULL_ONLY_RANKING_USED"] is False
           and gate["BACK_ONLY_RANKING_USED"] is False,
           "gate all pass, §九 prohibition flags false")
        ready_after = proj.read_json(os.path.join(cc4, READY_V3_NAME))
        ok(ready_after["FORMAL_RANKING_STARTED"] is True
           and ready_after["FORMAL_RANKING_PUBLISHED"] is True
           and ready_after["FORMAL_RANKING_AUTHORIZED_V3"] is True
           and ready_after["pending_gates"] == []
           and ready_after["formal_ranking_summary_sha256"] == side
           and ready_after["formal_winner"] == STUDENTS[0]
           and ready_after["formal_evaluation_started_at_utc"]
           == "1970-01-01T00:00:01+00:00"
           and ready_after["v2_archive"]["v2_winner"] is None,
           "READY_V3 closing flip applied (created)")

        # rerun guard: the driver start gate now fail-closes
        try:
            driver.verify_v3_repair_start(common, cc4)
            ok(False, "driver rerun guard accepted a flipped READY_V3")
        except proj.FailClosed as exc:
            ok("FORMAL_RANKING_STARTED=true" in str(exc),
               "driver fail-closes after the V3 flip")
        # second flip refuses (single writer)
        try:
            apply_ready_flip_v3(cc4, {"path": "m", "sha256": "s", "verdict": "v",
                                       "ruling_task": "t",
                                       "recorded_at_utc": "x"}, "0" * 64,
                                "0" * 64, "1970-01-01T00:00:02+00:00",
                                "V3_GATE_GREEN", True, "ORDERED", STUDENTS[0])
            ok(False, "second flip accepted")
        except proj.FailClosed:
            checks += 1

        # rehearsal bundle rejected
        os.remove(os.path.join(cc4, READY_V3_NAME))  # reset flip for negatives
        _build_synthetic_bundle(cc4, STUDENTS[0], tuples[STUDENTS[0]], head,
                                marker_sha, rehearsal=True)
        b = verify_candidate_bundle(STUDENTS[0], cc4, head, None, marker_sha)
        ok(not b["eligible"] and any("REHEARSAL_IN_FORMAL_POOL" in p
                                     for p in b["problems"]),
           "rehearsal bundle rejected")
        _build_synthetic_bundle(cc4, STUDENTS[0], tuples[STUDENTS[0]], head,
                                marker_sha)

        # engine-abort bundle -> BLOCKED_ENGINE_ABORT, never eligible
        _build_synthetic_bundle(cc4, STUDENTS[1], tuples[STUDENTS[1]], head,
                                marker_sha, abort=True)
        b = verify_candidate_bundle(STUDENTS[1], cc4, head, None, marker_sha)
        ok(not b["eligible"] and b["participant_status"] == "BLOCKED_ENGINE_ABORT",
           "abort -> BLOCKED_ENGINE_ABORT")
        _build_synthetic_bundle(cc4, STUDENTS[1], tuples[STUDENTS[1]], head,
                                marker_sha)

        # pin drift rejected
        _build_synthetic_bundle(cc4, STUDENTS[2], tuples[STUDENTS[2]], head,
                                marker_sha, corrupt_pin=True)
        b = verify_candidate_bundle(STUDENTS[2], cc4, head, None, marker_sha)
        ok(not b["eligible"] and any("CERT_PINS_DRIFT" in p or
                                     "CERTIFICATE_VERIFY" in p
                                     for p in b["problems"]), "pin drift rejected")
        _build_synthetic_bundle(cc4, STUDENTS[2], tuples[STUDENTS[2]], head,
                                marker_sha)

        # FULL reuse REJECT -> honest BLOCKED_REUSE_REJECT (no rerun)
        _build_synthetic_bundle(cc4, STUDENTS[4], tuples[STUDENTS[4]], head,
                                marker_sha, full_reject=True)
        b = verify_candidate_bundle(STUDENTS[4], cc4, head, None, marker_sha)
        ok(not b["eligible"] and b["participant_status"] == "BLOCKED_REUSE_REJECT"
           and any("FULL_REUSE_STATUS" in p for p in b["problems"]),
           "FULL reuse REJECT -> BLOCKED_REUSE_REJECT")
        _build_synthetic_bundle(cc4, STUDENTS[4], tuples[STUDENTS[4]], head,
                                marker_sha)

        # GPU outside {GPU2,GPU3} rejected (G16)
        _build_synthetic_bundle(cc4, STUDENTS[5], tuples[STUDENTS[5]], head,
                                marker_sha, gpu_banned=True)
        b = verify_candidate_bundle(STUDENTS[5], cc4, head, None, marker_sha)
        ok(not b["eligible"] and any("GPU_NOT_V3_ONLY" in p
                                     for p in b["problems"]),
           "GPU0/1 (banned for V3) rejected")
        _build_synthetic_bundle(cc4, STUDENTS[5], tuples[STUDENTS[5]], head,
                                marker_sha)

        # marker sha drift rejected (G15 cross-check)
        b = verify_candidate_bundle(STUDENTS[3], cc4, head, None, "f" * 64)
        ok(not b["eligible"] and any("BUNDLE_MARKER_SHA_DRIFT" in p
                                     for p in b["problems"]),
           "bundle marker sha drift rejected")

        # git head mismatch rejected (no ancestry proof available)
        b = verify_candidate_bundle(STUDENTS[3], cc4, "cd" * 20, None, marker_sha)
        ok(not b["eligible"] and any("GIT_HEAD" in p for p in b["problems"])
           and b["git_head_relation_to_marker"] == "NOT_VERIFIED_DESCENDANT",
           "head mismatch rejected")
        # descendant head accepted with an ancestry proof
        desc_head = "cd" * 20
        _build_synthetic_bundle(cc4, STUDENTS[3], tuples[STUDENTS[3]], desc_head,
                                marker_sha)
        b = verify_candidate_bundle(
            STUDENTS[3], cc4, head,
            lambda a, dd: a == head and dd == desc_head, marker_sha)
        ok(b["eligible"] and b["git_head_relation_to_marker"] == "DESCENDANT",
           "descendant head accepted via ancestry proof")
        _build_synthetic_bundle(cc4, STUDENTS[3], tuples[STUDENTS[3]], head,
                                marker_sha)

        # missing dir -> NOT_ELIGIBLE_COMPLETE
        import shutil
        shutil.rmtree(os.path.join(cc4, STUDENTS[4], "formal_evaluation_v3"))
        b = verify_candidate_bundle(STUDENTS[4], cc4, head, None, marker_sha)
        ok(not b["eligible"] and b["participant_status"] == "NOT_ELIGIBLE_COMPLETE",
           "missing dir not eligible")

    # honest completion-block majority (a BACK completion hit a retained
    # fail-closed): integrity gates true, participation/completion gates false,
    # every failure from BLOCKED candidates -> publish honest, AUTH_V3=false,
    # winner=null (STRICTER than V2). And an integrity failure -> NO_FLIP.
    with tempfile.TemporaryDirectory() as td2:
        cc4b = os.path.join(td2, "cc4")
        commonb = os.path.join(td2, "common_v2")
        os.makedirs(cc4b)
        os.makedirs(commonb)
        head2 = "ef" * 20
        _, msha2 = _write_marker(cc4b, head2)
        _build_synthetic_bundle(cc4b, STUDENTS[0], (30, 5, 0.55, 3), head2, msha2)
        for cid in STUDENTS[1:]:
            _build_synthetic_bundle(cc4b, cid, (0, 0, 0.1, 0), head2, msha2,
                                    abort=True)
        _build_synthetic_bundle(cc4b, TEACHER, (64, 8, 0.99, 8), head2, msha2,
                                abort=True)
        rc = main(["--pool-cc4-dir", cc4b, "--common-dir", commonb, "--repo", td2])
        checks += 1
        proj.require(rc == 0, "RANKING_V3_SELF_TEST: blocked-policy rc=%d" % rc)
        summary = proj.read_json(os.path.join(cc4b, SUMMARY_NAME))
        ok(summary["ranking_status"] == "INCONCLUSIVE_PARTICIPATION"
           and summary["student_count_eligible"] == "1/6"
           and summary["FORMAL_RANKING_AUTHORIZED_V3"] is False
           and summary["formal_winner"] is None and summary["escalation"],
           "completion-block outcome: AUTH_V3=false, winner=null, escalation")
        gate = proj.read_json(os.path.join(cc4b, GATE_NAME))
        ok(gate["FORMAL_EVALUATION_GATE_V3_PASS"] is False
           and gate["FORMAL_RANKING_AUTHORIZED_V3"] is False
           and gate["gates"]["G1_ALL_6_STUDENTS_ELIGIBLE_COMPLETE"] is False
           and gate["gates"]["G3_NO_ENGINE_ABORT"] is False
           and gate["flip_policy"] == "V3_PUBLISH_HONEST_UNDER_COMPLETION_BLOCK"
           and gate["foreign_gate_failures"] == []
           and all(gate["integrity_gates"].values())
           and len(gate["blocked_candidate_ids"]) == 6
           and gate["formal_winner"] is None,
           "gate honestly false, integrity gates true, policy=publish-honest")
        ready_after = proj.read_json(os.path.join(cc4b, READY_V3_NAME))
        ok(ready_after["FORMAL_RANKING_STARTED"] is True
           and ready_after["FORMAL_RANKING_AUTHORIZED_V3"] is False
           and ready_after["formal_winner"] is None
           and ready_after["pending_gates"] == [],
           "publish-honest flip applied, AUTH_V3=false")

        # integrity failure (pin drift inside a BLOCKED bundle) -> NO_FLIP
        os.remove(os.path.join(cc4b, READY_V3_NAME))
        _build_synthetic_bundle(cc4b, STUDENTS[2], (0, 0, 0.1, 0), head2, msha2,
                                abort=True, corrupt_pin=True)
        rc = main(["--pool-cc4-dir", cc4b, "--common-dir", commonb, "--repo", td2])
        checks += 1
        proj.require(rc == 2, "RANKING_V3_SELF_TEST: integrity-fail rc=%d" % rc)
        gate = proj.read_json(os.path.join(cc4b, GATE_NAME))
        ok(gate["flip_policy"] == "NO_FLIP"
           and gate["integrity_gates"]["G5_CERTIFICATES_ALL_VERIFY"] is False,
           "integrity failure -> NO_FLIP")
        ok(not os.path.isfile(os.path.join(cc4b, READY_V3_NAME)),
           "integrity failure -> READY_V3 untouched")

    print("RANKING_V3_SELF_TEST_PASS checks=%d" % checks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
