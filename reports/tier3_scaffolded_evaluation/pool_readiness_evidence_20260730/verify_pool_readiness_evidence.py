#!/usr/bin/env python3
"""CC4 Tier3 — LOCAL PURE RE-VERIFICATION of the pool-readiness round
(closing contract 2, 2026-07-30). Any interpreter, no jax.

Cross-checks the NEW server artifacts (supplemented READY marker + five pending
binding records) against the PRIOR round's closing evidence in
../closing_evidence_20260730/ — every common SHA in a new binding must equal the
file identity established there, and every READY supplement field must
reproduce from those files. Exits non-zero on any mismatch.
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PRIOR = os.path.normpath(os.path.join(
    ROOT, "..", "closing_evidence_20260730", "common"))

FRONT_FROZEN = "21aeb7dcdcb4ffccfb1eedc80f2c6daec1995c242822a0efbec9b947e275d687"
BACK_FROZEN = "c632e30dcabea7ff812fa07ce855809ec54e7cbca87747fa2d5ab775431f2566"
STUDENTS = ["BASE_GTRXL_ORIGINAL_VTRACE_98304", "CONTROL_CONTINUOUS_98304",
            "SLOWGRU_PERSISTENT_CANONICAL_98304",
            "SLOWGRU_RESET128_CANONICAL_98304"]
TEACHER = "BASELINE_TEACHER_CKPT17500"

REQUIRED_BINDING_FIELDS = (
    "common_runner_sha256", "common_evaluator_sha256",
    "evaluation_profile_sha256", "metric_schema_sha256",
    "front_bank_content_sha256", "back_bank_content_sha256",
    "full_profile_sha256", "full_profile_status", "environment_lock_sha256",
    "max_steps", "action_mode", "observation_shape", "action_dim",
    "front_episode_count", "back_episode_count", "full_episode_count",
    "candidate_runtime_sha256", "evaluate_candidate_sha256",
    "candidate_manifest_sha256", "checkpoint_contract_sha256",
    "params_sha256", "checkpoint_file_sha256", "formal_eval_binding")

problems = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        problems.append(name)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def canonical(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def load(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


prior_manifest = load(os.path.join(PRIOR, "assembly_manifest.json"))
prior_profile = load(os.path.join(PRIOR, "evaluation_profile.json"))
prior_full_sha = canonical(prior_profile["scenarios"]["full"])
prior_lock_sha = sha_file(os.path.join(PRIOR, "environment_lock.json"))

# --- supplemented READY marker ------------------------------------------------
ready = load(os.path.join(ROOT, "common", "COMMON_EVALUATOR_READY.json"))
check("READY still true", ready.get("COMMON_EVALUATOR_READY") is True)
check("negative_test_report_sha256 reproduces from prior evidence",
      ready.get("negative_test_report_sha256")
      == sha_file(os.path.join(PRIOR, "negative_test_report.json")))
check("assembly_manifest_sha256 reproduces from prior evidence",
      ready.get("assembly_manifest_sha256")
      == sha_file(os.path.join(PRIOR, "assembly_manifest.json")))
check("full_profile_sha256 == canonical full block of prior profile",
      ready.get("full_profile_sha256") == prior_full_sha)
check("FULL_PROFILE_STATUS FROZEN", ready.get("FULL_PROFILE_STATUS") == "FROZEN")
check("common_sha256sums_self_check PASS",
      ready.get("common_sha256sums_self_check") == "PASS")
check("FORMAL_RANKING_AUTHORIZED false",
      ready.get("FORMAL_RANKING_AUTHORIZED") is False)
for key in ("common_evaluator_sha256", "common_runner_sha256",
            "evaluation_profile_sha256", "metric_schema_sha256",
            "environment_lock_sha256"):
    check("READY.%s unchanged == prior manifest" % key,
          ready.get(key) == prior_manifest[key])

# --- the five pending records ---------------------------------------------------
for cid in STUDENTS + [TEACHER]:
    d = os.path.join(ROOT, "cc4", cid)
    files = set(os.listdir(d))
    check("%s has the five record files" % cid,
          {"candidate_manifest.json", "common_evaluator_binding_result.json",
           "environment_lock.json", "SHA256SUMS", "READY.json"} <= files)
    # sums verify locally
    bad = []
    with open(os.path.join(d, "SHA256SUMS"), encoding="utf-8") as fh:
        for line in fh:
            sha, rel = line.rstrip("\n").split("  ", 1)
            if sha_file(os.path.join(d, rel)) != sha:
                bad.append(rel)
    check("%s SHA256SUMS 3/3" % cid, not bad)

    b = load(os.path.join(d, "common_evaluator_binding_result.json"))
    missing_fields = [f for f in REQUIRED_BINDING_FIELDS if f not in b]
    check("%s binding carries all 23 §三 fields" % cid, not missing_fields)
    check("%s formal_eval_binding honest (never PASS)" % cid,
          b.get("formal_eval_binding") in
          ("MISSING_EVIDENCE", "PENDING_COMMON_READY", "PENDING_FULL_PROFILE")
          and b.get("formal_eval_binding") != "PASS")
    check("%s common SHAs == prior manifest identities" % cid,
          b.get("common_runner_sha256") == prior_manifest["common_runner_sha256"]
          and b.get("common_evaluator_sha256")
          == prior_manifest["common_evaluator_sha256"]
          and b.get("evaluation_profile_sha256")
          == prior_manifest["evaluation_profile_sha256"]
          and b.get("metric_schema_sha256")
          == prior_manifest["metric_schema_sha256"]
          and b.get("environment_lock_sha256")
          == prior_manifest["environment_lock_sha256"])
    check("%s bank contents frozen" % cid,
          b.get("front_bank_content_sha256") == FRONT_FROZEN
          and b.get("back_bank_content_sha256") == BACK_FROZEN)
    check("%s full_profile_sha256 == canonical full block" % cid,
          b.get("full_profile_sha256") == prior_full_sha
          and b.get("full_profile_status") == "FROZEN")
    check("%s zero episodes, no certificate, no perf claim" % cid,
          b.get("front_episode_count") == 0
          and b.get("back_episode_count") == 0
          and b.get("full_episode_count") == 0
          and b.get("evaluation_certificate_status") == "PENDING_FORMAL_EVALUATION"
          and b.get("evaluation_certificate_file") is None
          and b.get("performance_claim_authorized") is False)
    check("%s frozen profile protocol" % cid,
          b.get("max_steps") == 4096 and b.get("action_mode") == "greedy_argmax"
          and b.get("observation_shape") == [8335] and b.get("action_dim") == 43)
    check("%s owner capsule file SHAs recorded (4)" % cid,
          all(isinstance(b.get(k), str) and len(b[k]) == 64 for k in
              ("candidate_manifest_sha256", "candidate_runtime_sha256",
               "evaluate_candidate_sha256", "checkpoint_contract_sha256")))
    check("%s params never recomputed by cc4" % cid,
          b.get("params_sha256_cc4_recomputation")
          == "NOT_PERFORMED_REQUIRES_OWNER_RUNTIME_AND_PROTOCOL")
    check("%s blockers include runtime-family gate" % cid,
          any("runtime_family_not_registered" in x for x in b.get("blockers", [])))

    m = load(os.path.join(d, "candidate_manifest.json"))
    r = load(os.path.join(d, "READY.json"))
    check("%s audit manifest immutable + unauthorized" % cid,
          m.get("immutable") is True
          and m.get("performance_claim_authorized") is False
          and m.get("runtime_family_registered_in_cc4_common_abi") is False
          and m.get("student_rank") is None)
    check("%s READY false with honest gates" % cid,
          r.get("READY") is False
          and r.get("gates", {}).get("runtime_family_registered_in_common_abi")
          == "FAIL"
          and r.get("gates", {}).get("binding_status_honest") == "PASS"
          and r.get("gates", {}).get("params_sha256_cc4_verified") == "BLOCKED")
    check("%s env lock byte-identical to common" % cid,
          sha_file(os.path.join(d, "environment_lock.json")) == prior_lock_sha)

# --- student-specific ------------------------------------------------------------
for cid in STUDENTS:
    m = load(os.path.join(ROOT, "cc4", cid, "candidate_manifest.json"))
    check("%s STUDENT eligible + MATCHED_98304" % cid,
          m.get("candidate_class") == "STUDENT"
          and m.get("formal_student_ranking_eligible") is True
          and m.get("budget_class") == "MATCHED_98304"
          and m.get("training_steps") == 98304
          and m.get("training_seed") == 42)
verified_ckpt = [cid for cid in STUDENTS if load(os.path.join(
    ROOT, "cc4", cid, "common_evaluator_binding_result.json"))
    .get("checkpoint_file_sha256_verified_by_cc4") is True]
check("4/5 checkpoint file SHAs independently recomputed by CC4",
      sorted(verified_ckpt) == sorted(
          [c for c in STUDENTS if c != "CONTROL_CONTINUOUS_98304"]))
ctrl = load(os.path.join(ROOT, "cc4", "CONTROL_CONTINUOUS_98304",
                         "common_evaluator_binding_result.json"))
check("control checkpoint blocked on owner orbax protocol (not fabricated)",
      "BLOCKED_PENDING_OWNER_PROTOCOL" in ctrl["checkpoint_file_sha256_status"]
      and ctrl["checkpoint_file_sha256_cc4_recomputed"] is None
      and ctrl["params_sha256"] == "PENDING_COMPUTED_IN_SMOKE")

# --- teacher-specific --------------------------------------------------------------
mt = load(os.path.join(ROOT, "cc4", TEACHER, "candidate_manifest.json"))
bt = load(os.path.join(ROOT, "cc4", TEACHER, "common_evaluator_binding_result.json"))
check("teacher class/flags per contract",
      mt.get("candidate_class") == "TEACHER_REFERENCE"
      and mt.get("formal_student_ranking_eligible") is False
      and mt.get("strong_student_selection_eligible") is False
      and mt.get("student_rank") is None
      and mt.get("reference_only") is True
      and mt.get("budget_class") == "UNMATCHED_REFERENCE"
      and mt.get("training_steps") == 17500)
check("teacher binding excluded from ranking",
      bt.get("reference_only") is True
      and bt.get("formal_student_ranking_eligible") is False
      and bt.get("teacher_included_in_student_ranking") is False
      and bt.get("student_rank") is None)
check("teacher checkpoint recomputed by CC4",
      bt.get("checkpoint_file_sha256_verified_by_cc4") is True
      and bt.get("checkpoint_file_sha256")
      == "a87924a34d898fceed874c16e7332703fe960f02abaa2f8443efaecdb7482d01")

print()
if problems:
    print("POOL_READINESS_LOCAL_REVERIFY_FAIL (%d problems)" % len(problems))
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("POOL_READINESS_LOCAL_REVERIFY_PASS")
