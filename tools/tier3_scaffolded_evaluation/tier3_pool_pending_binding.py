#!/usr/bin/env python3
"""CC4 Tier3 — PENDING COMMON-EVALUATOR BINDINGS for the rest of the pool
(closing contract 2 §三/§四: the four non-RMT16 students + the teacher reference).

CC4 owns the common evaluator. The other four students and the teacher
reference are OWNED artifacts of CC1/CC2/CC3 — this tool never modifies them;
it reads them read-only and writes CC4's binding AUDIT RECORDS under
<cc4-root>/<CANDIDATE_ID>/, each containing:

  candidate_manifest.json                CC4 audit projection (NOT an owner capsule)
  common_evaluator_binding_result.json   every §三 field, each a recomputed SHA or
                                         an explicit BLOCKED/PENDING status
  environment_lock.json                  byte copy of the common lock
  SHA256SUMS                             over the two documents + the lock
  READY.json                             READY=false with the honest blockers

HONESTY DISCIPLINE (fail closed, never faked):
- This tool can NEVER write formal_eval_binding=PASS. PASS requires every SHA
  recomputed against the real file AND an interface smoke through the common
  evaluator; neither is possible while a candidate's runtime family is not
  registered in the common ABI (ABI doc §1.2: registration is by owner; the
  registered set is exactly ("rmt16_gtrxl_cc2",)).
- params_sha256 is NEVER recomputed here: the params hash protocol is owned by
  the candidate's trainer; recomputing it a different way would fabricate
  evidence. Owner-claimed values are recorded verbatim with status
  NOT_RECOMPUTED_BY_CC4.
- No interface smoke, no rollout, no GPU, no certificate: every episode count
  is 0 and evaluation_certificate_status=PENDING_FORMAL_EVALUATION.
- common ready missing/false      -> formal_eval_binding=PENDING_COMMON_READY
- FULL profile not frozen         -> formal_eval_binding=PENDING_FULL_PROFILE
- source capsule file missing     -> formal_eval_binding=MISSING_EVIDENCE
- otherwise                       -> formal_eval_binding=MISSING_EVIDENCE with the
                                     standard blocker list (runtime family etc.)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCHEMA_MANIFEST = "mechanism_UED.candidate_audit_manifest/v1"
SCHEMA_BINDING = "mechanism_UED.common_evaluator_binding/v1"
SCHEMA_READY = "mechanism_UED.candidate_pending_ready/v1"

FRONT_FROZEN_CONTENT_SHA256 = (
    "21aeb7dcdcb4ffccfb1eedc80f2c6daec1995c242822a0efbec9b947e275d687")
BACK_FROZEN_CONTENT_SHA256 = (
    "c632e30dcabea7ff812fa07ce855809ec54e7cbca87747fa2d5ab775431f2566")
BASE_PARAMS_SHA256 = (
    "d4e85af58b7f87d689fadea12eec70c852fa098a09f5ea8907448684b3bf60f5")
PROFILE_MAX_STEPS = 4096
PROFILE_ACTION_MODE = "greedy_argmax"
PROFILE_OBSERVATION_SHAPE = [8335]
PROFILE_ACTION_DIM = 43

# Read-only source facts collected 2026-07-30 from each owner's capsule
# (candidate_manifest.json / checkpoint_contract.json). SHA claims are the
# OWNERS' values, recorded verbatim; CC4 recomputes only plain file SHAs.
CANDIDATES = {
    "BASE_GTRXL_ORIGINAL_VTRACE_98304": {
        "owner": "CC2",
        "source_capsule_rel": "cc2/BASE_GTRXL_ORIGINAL_VTRACE_98304",
        "candidate_class": "STUDENT",
        "network_family": "base_gtrxl",
        "memory_mode": "none",
        "replay_mode": "original_vtrace",
        "budget_class": "MATCHED_98304",
        "training_steps": 98304,
        "training_seed": 42,
        "checkpoint_step": 98304,
        "checkpoint_format": "pickle",
        "checkpoint_path": "/home/oseasy/cc2_data/cc2_basegtrxl_runs/runs/"
                           "BASEGTRXL-LONG98304/ckpt/98304/full_state.pkl",
        "checkpoint_file_sha256_claim":
            "d71e30aebb307c6fc5b404543a5ba3e32b30e25857905ae20300be46600713ea",
        "params_sha256_claim":
            "78a14cc6e9ccdeb2c9c3d827ff9e21366d7ceb73ca6c9e557fb1a3733fe6b3f2",
        "base_checkpoint_params_sha256": BASE_PARAMS_SHA256,
        "single_file_checkpoint": True,
    },
    "CONTROL_CONTINUOUS_98304": {
        "owner": "CC1",
        "source_capsule_rel": "cc1/CONTROL_CONTINUOUS_98304",
        "candidate_class": "STUDENT",
        "network_family": "GTrXL128",
        "memory_mode": "gtrxl_window128",
        "replay_mode": "OFF",
        "budget_class": "MATCHED_98304",
        "training_steps": 98304,
        "training_seed": 42,
        "checkpoint_step": 98304,
        "checkpoint_format": "orbax",
        "checkpoint_path": "/home/oseasy/student_pool_v1/cc1_retrain/"
                           "CONTROL_CONTINUOUS_CANONICAL_98304/long_98304/"
                           "checkpoints/98304",
        "checkpoint_file_sha256_claim": "PENDING_ORBAX_DIR_COMPUTED_IN_SMOKE",
        "params_sha256_claim": "PENDING_COMPUTED_IN_SMOKE",
        "base_checkpoint_params_sha256": None,
        "single_file_checkpoint": False,
    },
    "SLOWGRU_PERSISTENT_CANONICAL_98304": {
        "owner": "CC3",
        "source_capsule_rel": "cc3/SLOWGRU_PERSISTENT_CANONICAL_98304",
        "candidate_class": "STUDENT",
        "network_family": "SlowGRU",
        "memory_mode": "persistent",
        "replay_mode": "canonical_awr",
        "budget_class": "MATCHED_98304",
        "training_steps": 98304,
        "training_seed": 42,
        "checkpoint_step": 98304,
        "checkpoint_format": "pickle",
        "checkpoint_path": "/home/oseasy/student_pool_v1/cc3/"
                           "SLOWGRU_PERSISTENT_CANONICAL_98304/ckpt/98304/"
                           "full_state.pkl",
        "checkpoint_file_sha256_claim":
            "0bc92c9ee28684ba507d6d6d728110000f11d7115126fbaf9137b1f8390a9c47",
        "params_sha256_claim":
            "99d734b48acfd3499e5b836c7f632a52b1d17a732c3764a24c1935fd82a77ecc",
        "base_checkpoint_params_sha256": BASE_PARAMS_SHA256,
        "single_file_checkpoint": True,
    },
    "SLOWGRU_RESET128_CANONICAL_98304": {
        "owner": "CC3",
        "source_capsule_rel": "cc3/SLOWGRU_RESET128_CANONICAL_98304",
        "candidate_class": "STUDENT",
        "network_family": "SlowGRU",
        "memory_mode": "reset128",
        "replay_mode": "canonical_awr",
        "budget_class": "MATCHED_98304",
        "training_steps": 98304,
        "training_seed": 42,
        "checkpoint_step": 98304,
        "checkpoint_format": "pickle",
        "checkpoint_path": "/home/oseasy/experiments/student_upgrade_wave1_4gpu/"
                           "gpu2_slowgru_reset128_longrun/train/ckpt/98304/"
                           "full_state.pkl",
        "checkpoint_file_sha256_claim":
            "2c065fa88bcc8cfcb193deda6ef599522238b99bf7151f5eeab0b70e4420f2de",
        "params_sha256_claim":
            "9d92c5b9e2e2148b2375c59f7f595d53b95f924d62436ebdccf8bf9ea3d59247",
        "base_checkpoint_params_sha256": BASE_PARAMS_SHA256,
        "single_file_checkpoint": True,
    },
    "BASELINE_TEACHER_CKPT17500": {
        "owner": "CC1",
        "source_capsule_rel": "cc1/BASELINE_TEACHER_CKPT17500",
        "candidate_class": "TEACHER_REFERENCE",
        "network_family": "GTrXL128",
        "memory_mode": "gtrxl_window128",
        "replay_mode": "reference_only",
        "budget_class": "UNMATCHED_REFERENCE",
        "training_steps": 17500,
        "training_seed": "UNKNOWN_HENRY_BASE_PROVENANCE",
        "checkpoint_step": 17500,
        "checkpoint_format": "pickle",
        "checkpoint_path": "/home/oseasy/experiments/bakeoff_phase1/"
                           "shared_assets/ckpt17500_params.pkl",
        "checkpoint_file_sha256_claim":
            "a87924a34d898fceed874c16e7332703fe960f02abaa2f8443efaecdb7482d01",
        "params_sha256_claim": BASE_PARAMS_SHA256,
        "base_checkpoint_params_sha256": BASE_PARAMS_SHA256,
        "single_file_checkpoint": True,
    },
}

REQUIRED_SOURCE_FILES = ("candidate_manifest.json", "candidate_runtime.py",
                         "evaluate_candidate.py", "checkpoint_contract.json",
                         "READY.json", "SHA256SUMS")

STANDARD_BLOCKERS = (
    "runtime_family_not_registered_in_cc4_common_abi (registered set is "
    "exactly ('rmt16_gtrxl_cc2',); ABI doc: registration is by owner)",
    "no_interface_smoke_through_cc4_common_evaluator (impossible without a "
    "registered runtime; this round authorizes no new adapter)",
    "params_sha256_not_recomputed_by_cc4 (owner hash protocol; recomputing it "
    "differently would fabricate evidence)",
    "formal_evaluation_not_authorized_this_round",
)


class FailClosed(Exception):
    """Hard stop on any binding violation."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: str, doc: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def common_sha_set(common_root: str) -> dict:
    """Recompute every common SHA from the live files (fail closed)."""
    def sha(rel):
        return _sha256_file(os.path.join(common_root, rel))
    with open(os.path.join(common_root, "evaluation_profile.json"),
              encoding="utf-8") as fh:
        profile = json.load(fh)
    full = profile.get("scenarios", {}).get("full")
    require(isinstance(full, dict) and full, "FAIL CLOSED: profile full missing")
    with open(os.path.join(common_root, "statuses", "bank_identity.json"),
              encoding="utf-8") as fh:
        banks = json.load(fh)["banks"]
    front_c = banks["front_l2"]["canonical_content_sha256"]
    back_c = banks["back_l2"]["canonical_content_sha256"]
    require(front_c == FRONT_FROZEN_CONTENT_SHA256
            and back_c == BACK_FROZEN_CONTENT_SHA256,
            "FAIL CLOSED: common bank content SHAs are not the frozen identities")
    return {
        "common_runner_sha256": sha("common_runner.py"),
        "common_evaluator_sha256": sha("common_evaluator.py"),
        "evaluation_profile_sha256": sha("evaluation_profile.json"),
        "metric_schema_sha256": sha("metric_schema.json"),
        "environment_lock_sha256": sha("environment_lock.json"),
        "front_bank_content_sha256": front_c,
        "back_bank_content_sha256": back_c,
        "full_profile_sha256": _canonical_sha256(full),
        "full_profile_status":
            "FROZEN" if full.get("FULL_PROFILE_READY") is True else "PENDING",
    }


def determine_status(common_ready: bool, full_frozen: bool,
                     missing_source_files: list) -> str:
    """The ONLY status ladder this tool implements. PASS is unreachable by
    construction — see the module docstring."""
    if not common_ready:
        return "PENDING_COMMON_READY"
    if not full_frozen:
        return "PENDING_FULL_PROFILE"
    if missing_source_files:
        return "MISSING_EVIDENCE"
    return "MISSING_EVIDENCE"


def build(candidate_id: str, cc4_root: str, pool_root: str,
          common_root: str) -> dict:
    spec = CANDIDATES[candidate_id]
    out_dir = os.path.join(cc4_root, candidate_id)
    require(not os.path.exists(out_dir),
            "FAIL CLOSED: %s already exists — never overwrite an existing "
            "capsule/record" % out_dir)
    is_teacher = spec["candidate_class"] == "TEACHER_REFERENCE"

    # --- common readiness -----------------------------------------------------
    ready_path = os.path.join(common_root, "COMMON_EVALUATOR_READY.json")
    common_ready = False
    if os.path.isfile(ready_path):
        with open(ready_path, encoding="utf-8") as fh:
            common_ready = json.load(fh).get("COMMON_EVALUATOR_READY") is True
    common_shas = common_sha_set(common_root) if common_ready else None
    full_frozen = bool(common_shas and
                       common_shas["full_profile_status"] == "FROZEN")

    # --- read-only source capsule ---------------------------------------------
    src = os.path.join(pool_root, spec["source_capsule_rel"])
    missing = [f for f in REQUIRED_SOURCE_FILES
               if not os.path.isfile(os.path.join(src, f))]
    src_shas = {f: _sha256_file(os.path.join(src, f))
                for f in REQUIRED_SOURCE_FILES
                if os.path.isfile(os.path.join(src, f))}

    # --- checkpoint evidence (read-only) ----------------------------------------
    ckpt_claim = spec["checkpoint_file_sha256_claim"]
    ckpt_exists = os.path.exists(spec["checkpoint_path"])
    ckpt_recomputed, ckpt_verified = None, None
    if spec["single_file_checkpoint"]:
        if ckpt_exists and os.path.isfile(spec["checkpoint_path"]):
            ckpt_recomputed = _sha256_file(spec["checkpoint_path"])
            ckpt_verified = (ckpt_recomputed == ckpt_claim)
        ckpt_status = ("CC4_RECOMPUTED_MATCH" if ckpt_verified
                       else "CC4_RECOMPUTED_MISMATCH" if ckpt_recomputed
                       else "BLOCKED_CHECKPOINT_FILE_NOT_FOUND")
    else:
        ckpt_status = ("BLOCKED_PENDING_OWNER_PROTOCOL (orbax dir; the file SHA "
                       "definition is owned by %s — CC4 computes no competing "
                       "directory hash)" % spec["owner"])

    status = determine_status(common_ready, full_frozen, missing)
    blockers = list(STANDARD_BLOCKERS)
    if missing:
        blockers.insert(0, "source_capsule_missing_files: %s" % ",".join(missing))
    if not ckpt_exists:
        blockers.insert(0, "checkpoint_path_not_found: %s"
                        % spec["checkpoint_path"])
    if spec["single_file_checkpoint"] and ckpt_verified is False:
        blockers.insert(0, "checkpoint_file_sha256_mismatch_on_cc4_recompute")
    if is_teacher:
        blockers.append("teacher_reference: excluded from student ranking by "
                        "contract (candidate_class=TEACHER_REFERENCE)")

    now = _utc_now()
    params_claim = spec["params_sha256_claim"]
    params_full64 = isinstance(params_claim, str) and len(params_claim) == 64 \
        and all(c in "0123456789abcdef" for c in params_claim)

    # --- documents -------------------------------------------------------------
    manifest = {
        "schema": SCHEMA_MANIFEST,
        "generated_at_utc": now,
        "generated_by": "CC4 (common evaluator owner) — AUDIT PROJECTION, not "
                        "an owner capsule; owner artifacts are never modified",
        "candidate_id": candidate_id,
        "candidate_class": spec["candidate_class"],
        "owner": spec["owner"],
        "source_capsule_root": src,
        "network_family": spec["network_family"],
        "memory_mode": spec["memory_mode"],
        "replay_mode": spec["replay_mode"],
        "runtime_family_label": spec["network_family"],
        "runtime_family_registered_in_cc4_common_abi": False,
        "cc4_common_abi_registered_families": ["rmt16_gtrxl_cc2"],
        "budget_class": spec["budget_class"],
        "training_steps": spec["training_steps"],
        "training_seed": spec["training_seed"],
        "checkpoint_step": spec["checkpoint_step"],
        "checkpoint_path": spec["checkpoint_path"],
        "checkpoint_format": spec["checkpoint_format"],
        "checkpoint_file_sha256": ckpt_claim,
        "checkpoint_file_sha256_status": ckpt_status,
        "checkpoint_file_sha256_cc4_recomputed": ckpt_recomputed,
        "params_sha256": params_claim,
        "params_sha256_status":
            "FULL64_OWNER_CLAIMED_NOT_RECOMPUTED_BY_CC4" if params_full64
            else "BLOCKED_PENDING_OWNER",
        "base_checkpoint_params_sha256": spec["base_checkpoint_params_sha256"],
        "observation_shape": PROFILE_OBSERVATION_SHAPE,
        "action_dim": PROFILE_ACTION_DIM,
        "immutable": True,
        "performance_claim_authorized": False,
        "strong_student_selection_authorized": False,
        # eligibility (contract §二/§三 + teacher block)
        "candidate_class_is_student": not is_teacher,
        "formal_student_ranking_eligible": not is_teacher,
        "strong_student_selection_eligible": not is_teacher,
        "student_rank": None,
        "reference_only": is_teacher,
    }

    binding = {
        "schema": SCHEMA_BINDING,
        "generated_at_utc": now,
        "candidate_id": candidate_id,
        "candidate_class": spec["candidate_class"],
        "owner": spec["owner"],
        "common_root": common_root,
        "common_ready_at_binding_time": common_ready,
        # the §三 common SHA set (all recomputed live when common is ready)
        "common_runner_sha256": common_shas["common_runner_sha256"]
                               if common_shas else None,
        "common_evaluator_sha256": common_shas["common_evaluator_sha256"]
                                  if common_shas else None,
        "evaluation_profile_sha256": common_shas["evaluation_profile_sha256"]
                                    if common_shas else None,
        "metric_schema_sha256": common_shas["metric_schema_sha256"]
                               if common_shas else None,
        "front_bank_content_sha256": common_shas["front_bank_content_sha256"]
                                    if common_shas else None,
        "back_bank_content_sha256": common_shas["back_bank_content_sha256"]
                                   if common_shas else None,
        "full_profile_sha256": common_shas["full_profile_sha256"]
                              if common_shas else None,
        "full_profile_status": common_shas["full_profile_status"]
                              if common_shas else "UNKNOWN_COMMON_NOT_READY",
        "environment_lock_sha256": common_shas["environment_lock_sha256"]
                                  if common_shas else None,
        # frozen evaluation protocol
        "max_steps": PROFILE_MAX_STEPS,
        "action_mode": PROFILE_ACTION_MODE,
        "action_mode_source": "frozen_evaluation_profile",
        "observation_shape": PROFILE_OBSERVATION_SHAPE,
        "action_dim": PROFILE_ACTION_DIM,
        # NO episodes were run — interface smoke through the common evaluator
        # is impossible without a registered runtime family.
        "front_episode_count": 0,
        "back_episode_count": 0,
        "full_episode_count": 0,
        "episodes_run_note": "zero by construction: no registered runtime, no "
                             "smoke, no rollout — smoke reward/success would "
                             "never be formal performance anyway",
        # owner capsule files (read-only SHAs)
        "source_capsule_root": src,
        "candidate_manifest_sha256": src_shas.get("candidate_manifest.json"),
        "candidate_runtime_sha256": src_shas.get("candidate_runtime.py"),
        "evaluate_candidate_sha256": src_shas.get("evaluate_candidate.py"),
        "checkpoint_contract_sha256": src_shas.get("checkpoint_contract.json"),
        "source_capsule_missing_files": missing,
        # checkpoint / params
        "checkpoint_file_sha256": ckpt_claim,
        "checkpoint_file_sha256_cc4_recomputed": ckpt_recomputed,
        "checkpoint_file_sha256_verified_by_cc4": bool(ckpt_verified),
        "checkpoint_file_sha256_status": ckpt_status,
        "params_sha256": params_claim,
        "params_sha256_cc4_recomputation":
            "NOT_PERFORMED_REQUIRES_OWNER_RUNTIME_AND_PROTOCOL",
        # verdict
        "formal_eval_binding": status,
        "run_class": None,
        "blockers": blockers,
        "evaluation_certificate_status": "PENDING_FORMAL_EVALUATION",
        "evaluation_certificate_file": None,
        "performance_claim_authorized": False,
        "strong_student_selection_authorized": False,
        "formal_student_ranking_eligible": not is_teacher,
        "strong_student_selection_eligible": not is_teacher,
        "student_rank": None,
        "reference_only": is_teacher,
        "teacher_included_in_student_ranking": False,
    }

    gates = {
        "common_ready_verified":
            "PASS" if common_ready else "FAIL",
        "full_profile_frozen":
            "PASS" if full_frozen else "FAIL",
        "source_capsule_files_present":
            "PASS" if not missing else "FAIL",
        "checkpoint_file_sha256_cc4_verified":
            ("PASS" if ckpt_verified else
             "BLOCKED" if not spec["single_file_checkpoint"] or not ckpt_exists
             else "FAIL"),
        "params_sha256_cc4_verified": "BLOCKED",
        "runtime_family_registered_in_common_abi": "FAIL",
        "cc4_interface_smoke_through_common_evaluator": "FAIL",
        "binding_status_honest":
            "PASS" if status != "PASS" else "FAIL",
    }
    ready = {
        "schema": SCHEMA_READY,
        "generated_at_utc": now,
        "candidate_id": candidate_id,
        "READY": False,
        "gates": gates,
        "blockers": blockers,
        "formal_eval_binding": status,
        "formal_ranking_eligible_this_round": False,
        "honest_false_discipline":
            "this record can only become READY after an owner-registered "
            "runtime family + a CC4-run interface smoke through the common "
            "evaluator + owner-protocol params recomputation; nothing here is "
            "ever faked to PASS",
    }

    # --- write ------------------------------------------------------------------
    os.makedirs(out_dir)
    _atomic_json(os.path.join(out_dir, "candidate_manifest.json"), manifest)
    _atomic_json(os.path.join(out_dir, "common_evaluator_binding_result.json"),
                 binding)
    shutil.copyfile(os.path.join(common_root, "environment_lock.json"),
                    os.path.join(out_dir, "environment_lock.json"))
    entries = []
    for rel in ("candidate_manifest.json", "common_evaluator_binding_result.json",
                "environment_lock.json"):
        entries.append("%s  %s\n" % (_sha256_file(os.path.join(out_dir, rel)),
                                     rel))
    with open(os.path.join(out_dir, "SHA256SUMS"), "w", encoding="utf-8") as fh:
        fh.write("".join(sorted(entries, key=lambda s: s.split("  ", 1)[1])))
    _atomic_json(os.path.join(out_dir, "READY.json"), ready)

    print("PENDING_BINDING_WRITTEN %s status=%s ckpt_file_sha_verified=%s "
          "owner=%s" % (candidate_id, status,
                        ckpt_verified if ckpt_verified is not None else "n/a",
                        spec["owner"]))
    return binding


def self_test() -> int:
    problems = []

    def check(name, cond):
        if not cond:
            problems.append(name)

    # Status ladder.
    check("common_not_ready_wins",
          determine_status(False, True, []) == "PENDING_COMMON_READY")
    check("full_not_frozen_wins_over_missing",
          determine_status(True, False, ["x"]) == "PENDING_FULL_PROFILE")
    check("missing_files_is_missing_evidence",
          determine_status(True, True, ["candidate_runtime.py"])
          == "MISSING_EVIDENCE")
    check("happy_path_is_still_missing_evidence",
          determine_status(True, True, []) == "MISSING_EVIDENCE")
    check("pass_is_unreachable",
          determine_status(True, True, []) != "PASS")

    # Pool coverage + teacher discipline.
    students = [c for c, s in CANDIDATES.items()
                if s["candidate_class"] == "STUDENT"]
    teachers = [c for c, s in CANDIDATES.items()
                if s["candidate_class"] == "TEACHER_REFERENCE"]
    check("four_students_one_teacher",
          len(students) == 4 and teachers == ["BASELINE_TEACHER_CKPT17500"])
    check("all_ids_frozen", sorted(CANDIDATES) == sorted([
        "BASE_GTRXL_ORIGINAL_VTRACE_98304", "CONTROL_CONTINUOUS_98304",
        "SLOWGRU_PERSISTENT_CANONICAL_98304", "SLOWGRU_RESET128_CANONICAL_98304",
        "BASELINE_TEACHER_CKPT17500"]))
    for cid, spec in CANDIDATES.items():
        is_t = spec["candidate_class"] == "TEACHER_REFERENCE"
        if is_t:
            check("teacher_budget_unmatched",
                  spec["budget_class"] == "UNMATCHED_REFERENCE")
        else:
            check("%s matched 98304" % cid,
                  spec["budget_class"] == "MATCHED_98304"
                  and spec["training_steps"] == 98304
                  and spec["training_seed"] == 42)
        claims = [spec["checkpoint_file_sha256_claim"], spec["params_sha256_claim"]]
        for v in claims:
            check("%s claims are full64-or-pending" % cid,
                  (isinstance(v, str) and len(v) == 64
                   and all(c in "0123456789abcdef" for c in v))
                  or (isinstance(v, str) and v.startswith("PENDING")))

    # The required §三 binding field set is fixed.
    required_binding_fields = (
        "common_runner_sha256", "common_evaluator_sha256",
        "evaluation_profile_sha256", "metric_schema_sha256",
        "front_bank_content_sha256", "back_bank_content_sha256",
        "full_profile_sha256", "full_profile_status", "environment_lock_sha256",
        "max_steps", "action_mode", "observation_shape", "action_dim",
        "front_episode_count", "back_episode_count", "full_episode_count",
        "candidate_runtime_sha256", "evaluate_candidate_sha256",
        "candidate_manifest_sha256", "checkpoint_contract_sha256",
        "params_sha256", "checkpoint_file_sha256", "formal_eval_binding")
    # (documented here; the live build() writes every one of these keys)
    check("required_field_list_complete", len(required_binding_fields) == 23)

    check("canonical_sha_stable",
          _canonical_sha256({"b": 1, "a": [1, 2]})
          == _canonical_sha256({"a": [1, 2], "b": 1}))

    if problems:
        print("TIER3_POOL_PENDING_BINDING_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_POOL_PENDING_BINDING_SELF_TEST_PASS "
          "(status ladder; PASS unreachable; teacher discipline; ids frozen)")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return self_test()

    def opt(flag, default=None):
        if flag in argv:
            return argv[argv.index(flag) + 1]
        return default

    if "--build" in argv or "--build-all" in argv:
        require(opt("--cc4-root") and opt("--pool-root") and opt("--common-root"),
                "usage: tier3_pool_pending_binding.py --build --candidate ID "
                "--cc4-root DIR --pool-root DIR --common-root DIR | "
                "--build-all ... | --self-test")
        ids = ([opt("--candidate")] if "--build" in argv
               else sorted(CANDIDATES))
        for cid in ids:
            require(cid in CANDIDATES,
                    "FAIL CLOSED: unknown candidate %r (pool: %s)"
                    % (cid, sorted(CANDIDATES)))
            build(cid, opt("--cc4-root"), opt("--pool-root"),
                  opt("--common-root"))
        print("PENDING_BINDINGS_COMPLETE count=%d" % len(ids))
        return 0
    print("usage: tier3_pool_pending_binding.py --build --candidate ID "
          "--cc4-root DIR --pool-root DIR --common-root DIR | "
          "--build-all ... | --self-test")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
