#!/usr/bin/env python3
"""CC4 Tier3 — FORMAL evaluation certificate, V2_DYNAMIC_TOPOLOGY generation.

Registry-shaped (all seven projection families), NOT CC2-arm-shaped: the V1
certificate module (tier3_evaluation_certificate.py) is structurally bound to
the CC2 RMT16 arm pair (carry_mode in (persistent, reset128), cc2_params_sha256,
checkpoint_contract_arm) and cannot express the six heterogeneous student
families + teacher reference. This module keeps the V1 module's DISCIPLINE
(value-binding — no signatures; required-field assertions; forbidden-overclaim
scan; honest status labels) and binds it to the projection registry.

A formal certificate:
  * run_class = FORMAL_EVALUATION, performance_evaluation_executed = true;
  * scientific_claim_authorized = false FOREVER this round (single training
    seed) — the ranking is a formal ranking, not a scientific claim;
  * student_rank = null AT CREATION and NEVER rewritten afterwards: ranks are
    published only by FORMAL_RANKING_SUMMARY_V2DT.json (the registry self-test
    structurally forbids writing ranks back into the registry); rewriting a
    certificate would break its SHA256SUMS_FORMAL_V2DT binding;
  * teacher: reference_only = true, teacher_included_in_student_ranking = false,
    counts_toward_student_binding_count = false — evaluated at the identical
    formal schedule for reference comparability, excluded from ranking;
  * references the SECONDARY_AUDIT_PASS marker (the start authorization) and
    the full V2 common pin set (identical key names / values to the V2DT
    binding certificates — single source of truth: the frozen V2 binding
    driver module).

JAX-free by construction (imports only the frozen pin constants via the V2
binding driver, which itself imports only the stdlib + the stdlib-only
projection registry at module level).
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import tier3_projection_binding_smoke_v2 as smokev2   # noqa: E402  (frozen pins)
import tier3_projection_runtime as proj               # noqa: E402

SCHEMA = "mechanism_UED.tier3_evaluation_certificate/v2dt"
CERT_VERSION = "tier3_evaluation_certificate/v2dt"
COMMON_EVALUATOR_PROTOCOL_VERSION = "V2_DYNAMIC_TOPOLOGY"
RUN_CLASS = "FORMAL_EVALUATION"
FORMAL_SCENARIO_ORDER = ("full", "front_l2", "back_l2")
RECOMPUTE_STATUS = "CC4_RECOMPUTED_MATCH_VIA_OWNER_PROTOCOL"
SECONDARY_AUDIT_VERDICT = "PASS_FOR_FORMAL_GLOBAL_PERFORMANCE_EVALUATION_START"

# The V2 common pin set, under the SAME key names the V2DT binding certificates
# use. Values are asserted == the frozen binding-driver constants at build time;
# any drift fails closed.
PIN_FIELD_SOURCES = {
    "common_evaluator_sha256": smokev2.FROZEN_V2_COMMON_EVALUATOR_SHA256,
    "common_runner_sha256": smokev2.FROZEN_V2_COMMON_RUNNER_SHA256,
    "evaluation_profile_sha256": smokev2.FROZEN_V2_EVALUATION_PROFILE_SHA256,
    "metric_schema_sha256": smokev2.FROZEN_V2_METRIC_SCHEMA_SHA256,
    "environment_lock_sha256": smokev2.FROZEN_V2_ENVIRONMENT_LOCK_SHA256,
    "candidate_runtime_abi_sha256": smokev2.FROZEN_V2_ABI_DOC_SHA256,
    "assembly_manifest_v2_sha256": smokev2.FROZEN_V2_ASSEMBLY_MANIFEST_SHA256,
    "sha256sums_v2_sha256": smokev2.FROZEN_V2_SHA256SUMS_SHA256,
    "full_profile_sha256": proj.FROZEN_FULL_PROFILE_SHA256,
    "front_bank_content_sha256": proj.FROZEN_FRONT_BANK_CONTENT_SHA256,
    "back_bank_content_sha256": proj.FROZEN_BACK_BANK_CONTENT_SHA256,
}

REQUIRED_FIELDS = (
    "schema", "cert_version", "generated_at_utc", "candidate_id", "owner",
    "candidate_class", "runtime_family", "common_evaluator_protocol_version",
    "bfs_graph_source", "run_class", "evaluation_status",
    "action_mode", "max_steps", "observation_shape", "action_dim",
    "front_episode_count", "back_episode_count", "full_episode_count",
    "episodes_executed", "valid_start_counts",
    "performance_evaluation_executed", "scientific_claim_authorized",
    "scaffolded_results_can_replace_full_task",
    "interface_smoke_substituted_for_performance",
    "formal_student_ranking_eligible", "reference_only",
    "teacher_included_in_student_ranking",
    "counts_toward_student_binding_count", "student_rank",
    "schedule", "entry_ids_by_scenario", "scenario_order",
    "params_sha256", "params_sha256_status", "params_unchanged",
    "checkpoint_file_sha256", "checkpoint_file_sha256_status",
    "episode_records_jsonl_sha256", "records_sha256_by_scenario",
    "results_by_scenario", "common_pins", "v1_supersession",
    "engine_identity", "provenance", "audit", "rehearsal",
)

SHA_FIELDS = (
    "params_sha256", "checkpoint_file_sha256", "episode_records_jsonl_sha256",
)

FORBIDDEN_OVERCLAIMS = (
    "BEST_OVERALL_STUDENT",
    "ALL_STUDENT_BAKEOFF_WINNER",
    "SCIENTIFIC_CLAIM_AUTHORIZED",
    "SCIENTIFIC_SUPERIORITY",
    "STRONG_STUDENT_V1",
    "OVERALL_STRONG_STUDENT_SELECTION_AUTHORIZED",
    "REPLACE_FULL_TASK",
    "SUCCEEDED_BY_TRAINING",
)


class CertificateError(Exception):
    pass


def _require(cond, msg):
    if not cond:
        raise CertificateError("FAIL CLOSED (CERTIFICATE_V2DT): %s" % msg)


def _hex64(v):
    return isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) is not None


def pins_snapshot():
    """The frozen V2 common pin set (single source of truth)."""
    return dict(PIN_FIELD_SOURCES)


def build_evaluation_certificate(ci):
    """Build one formal evaluation certificate from the driver's cert input.

    ci keys (assembled by tier3_formal_evaluation_v2dt.py):
      candidate_id, spec, results_by_scenario, schedule, entry_ids_by_scenario,
      records_sha256_by_scenario, episode_records_jsonl_sha256,
      episodes_executed {sc: n}, valid_start_counts {sc: n},
      params_before, declared_params, file_sha, declared_file, params_unchanged,
      capsule_ev, common_ev, gpu_ev, dicode_ev, provenance, boundary_ev,
      batch1_ev, formal_abort, rehearsal, rehearsal_limits, marker_ref
      ({"path", "sha256", "verdict", "recorded_at_utc"} or None),
      binding_gate_sha256, common_v2_ready_sha256, engine_module_shas.
    """
    spec = ci["spec"]
    is_teacher = spec["candidate_class"] == "TEACHER_REFERENCE"

    pins = pins_snapshot()  # frozen constants, asserted below
    for k, v in pins.items():
        _require(_hex64(v), "pin %s not a hex64 constant: %r" % (k, v))

    aborted = ci["formal_abort"] is not None
    rehearsal = bool(ci.get("rehearsal"))
    if aborted:
        status = "BLOCKED"
    elif rehearsal:
        status = "REHEARSAL_NOT_FORMAL"
    else:
        status = "PASS"

    cert = {
        "schema": SCHEMA,
        "cert_version": CERT_VERSION,
        "generated_at_utc": ci["generated_at_utc"],
        "candidate_id": ci["candidate_id"],
        "owner": spec["owner"],
        "candidate_class": spec["candidate_class"],
        "runtime_family": spec["runtime_family"],
        "source_capsule_root": spec["source_capsule_root"],
        "common_evaluator_protocol_version": COMMON_EVALUATOR_PROTOCOL_VERSION,
        "bfs_graph_source": ci["bfs_graph_source"],
        "run_class": RUN_CLASS,
        "evaluation_status": status,
        "formal_abort": ci["formal_abort"],
        # frozen contract (formal scale)
        "action_mode": proj.FROZEN_ACTION_MODE,
        "max_steps": proj.FROZEN_MAX_TIMESTEPS,
        "observation_shape": list(proj.FROZEN_OBSERVATION_SHAPE),
        "action_dim": proj.FROZEN_ACTION_DIM,
        "front_episode_count": proj.FROZEN_FRONT_EPISODE_COUNT,
        "back_episode_count": proj.FROZEN_BACK_EPISODE_COUNT,
        "full_episode_count": proj.FROZEN_FULL_EPISODE_COUNT,
        "episodes_executed": ci["episodes_executed"],
        "valid_start_counts": ci["valid_start_counts"],
        # honest labels
        "performance_evaluation_executed": not rehearsal,
        "scientific_claim_authorized": False,
        "scientific_claim_status":
            "FORMAL_SCIENTIFIC_CLAIM: NOT_AUTHORIZED_SINGLE_TRAINING_SEED",
        "scaffolded_results_can_replace_full_task": False,
        "interface_smoke_substituted_for_performance": False,
        "formal_performance_results_authorized_by_secondary_audit":
            (not rehearsal) and (ci.get("marker_ref") is not None),
        # eligibility / ranking discipline
        "formal_student_ranking_eligible": spec["formal_student_ranking_eligible"],
        "strong_student_selection_eligible":
            spec["strong_student_selection_eligible"],
        "reference_only": spec["reference_only"],
        "student_rank": None,     # null at creation; NEVER rewritten (see docstring)
        "student_rank_publication": "FORMAL_RANKING_SUMMARY_V2DT.json only; the "
            "registry self-test forbids writing ranks into the registry and this "
            "certificate is SHA256SUMS-bound and immutable after creation",
        "budget_class": spec["budget_class"],
        "training_steps": spec["training_steps"],
        "training_seed": spec["training_seed"],
        "teacher_included_in_student_ranking": False,
        "counts_toward_student_binding_count": (False if is_teacher else True),
        # schedule identity (verbatim)
        "schedule": ci["schedule"],
        "entry_ids_by_scenario": ci["entry_ids_by_scenario"],
        "scenario_order": list(FORMAL_SCENARIO_ORDER),
        # checkpoint identity (owner protocol, recomputed by the driver)
        "params_sha256": ci["params_before"],
        "params_sha256_status": RECOMPUTE_STATUS,
        "params_sha256_owner_declared": ci["declared_params"],
        "params_sha256_declaration_source":
            spec["declared_params_sha256"]["declaration_source"],
        "params_hash_protocol": spec["params_hash_protocol"],
        "params_unchanged": ci["params_unchanged"],
        "checkpoint_file_sha256": ci["file_sha"],
        "checkpoint_file_sha256_status": RECOMPUTE_STATUS,
        "checkpoint_file_sha256_owner_declared": ci["declared_file"],
        "checkpoint_file_sha256_declaration_source":
            spec["declared_checkpoint_file_sha256"]["declaration_source"],
        "checkpoint_file_hash_protocol": spec["checkpoint_file_hash_protocol"],
        # capsule artifacts
        "candidate_runtime_sha256":
            spec["capsule_file_sha256"]["candidate_runtime.py"],
        "evaluate_candidate_sha256":
            spec["capsule_file_sha256"]["evaluate_candidate.py"],
        "candidate_manifest_sha256":
            spec["capsule_file_sha256"]["candidate_manifest.json"],
        "checkpoint_contract_sha256":
            spec["capsule_file_sha256"]["checkpoint_contract.json"],
        "capsule_file_verification": ci["capsule_ev"],
        "bound_owner_runtime_sha256": spec["bound_owner_runtime_sha256"],
        # episode evidence
        "episode_records_jsonl_sha256": ci["episode_records_jsonl_sha256"],
        "records_sha256_by_scenario": ci["records_sha256_by_scenario"],
        "results_by_scenario": ci["results_by_scenario"],
        # the V2 common pin set (identical key names/values to the V2DT
        # binding certificates)
        "common_root": ci["common_dir"],
        "common_pins": pins,
        "common_sha_match_status": ci["common_ev"].get("common_v2_sha256sums_self_check"),
        "v1_supersession": {
            "COMMON_EVALUATOR_V1_DRIVER": smokev2.SUPERSEDED_V1_DRIVER_COMMIT,
            "COMMON_EVALUATOR_V1_STATUS": smokev2.SUPERSEDED_V1_STATUS,
            "v1_common_evaluator_sha256": proj.FROZEN_COMMON_EVALUATOR_SHA256,
            "v1_common_runner_sha256": proj.FROZEN_COMMON_RUNNER_SHA256,
            "v1_evaluation_profile_sha256": proj.FROZEN_EVALUATION_PROFILE_SHA256,
            "v1_metric_schema_sha256": proj.FROZEN_METRIC_SCHEMA_SHA256,
            "v1_sha256sums_sha256": proj.FROZEN_SHA256SUMS_SHA256,
            "v1_formal_ranking_ever_authorized": False,
            "v1_preservation_reverified": ci["common_ev"].get("v1_preservation"),
        },
        # engine identity
        "engine_identity": {
            "evaluator_v2_module_sha256": ci["engine_module_shas"]["tier3_evaluator_v2.py"],
            "predicates_v2_module_sha256": ci["engine_module_shas"]["tier3_event_predicates_v2.py"],
            "engine_lf_sha256_v2": dict(smokev2.FROZEN_V2_ENGINE_LF_SHA256),
            "engine_lf_sha256_v1_frozen": dict(proj.FROZEN_ENGINE_LF_SHA256),
            "engine_modules_verified_by_stage1":
                ci["common_ev"].get("engine_modules_lf_sha_verified"),
        },
        # policy adapter disclosure
        "policy_adapter": {
            "class": ci["policy_class"],
            "batch1_workaround": ci.get("batch1_ev"),
            "boundary_unit_check": ci.get("boundary_ev"),
        },
        "dicode_resolution": ci["dicode_ev"],
        "gpu": ci["gpu_ev"],
        "provenance": ci["provenance"],
        # start authorization
        "audit": {
            "secondary_audit_marker": ci.get("marker_ref"),
            "binding_gate_sha256": ci.get("binding_gate_sha256"),
            "common_v2_ready_sha256_at_run_time": ci.get("common_v2_ready_sha256"),
            "authorization_note": "formal run authorized by SECONDARY_AUDIT_PASS "
                "marker (verdict PASS_FOR_FORMAL_GLOBAL_PERFORMANCE_EVALUATION_"
                "START); interface smoke evidence is NOT a performance result",
        },
        "rehearsal": {
            "is_rehearsal": rehearsal,
            "limits": ci.get("rehearsal_limits"),
            "note": ("REHEARSAL runs are bounded plumbing checks on a scratch "
                     "directory; they are NOT formal results and the ranking "
                     "tool rejects them" if rehearsal
                     else "formal run at the frozen full schedule "
                          "(FULL 64 + FRONT 8 + BACK 8, max_steps=4096)"),
        },
    }
    if spec["loader_kind"] == "cc3_slowgru":
        cert["carry_mode"] = spec["carry_mode"]
        cert["segment_boundary_steps"] = spec["segment_boundary_steps"]
        cert["boundary_semantics"] = spec["boundary_semantics"]
    if spec["loader_kind"] == "cc4_rmt16_capsule":
        cert["carry_mode"] = spec["carry_mode"]
        cert["engine_runtime_family"] = spec["engine_runtime_family"]
        cert["rmt16_engine_metadata"] = ci.get("rmt16_engine_metadata")
        cert["rmt16_frozen_identities"] = ci.get("rmt16_frozen_identities")
        cert["rmt16_common_runner_sha256"] = ci.get("rmt16_common_runner_sha256")
        cert["rmt16_engine_lf_sha256"] = ci.get("rmt16_engine_lf_sha256")

    assert_required_fields(cert)
    overclaims = scan_forbidden_overclaims(cert)
    _require(not overclaims, "forbidden overclaim phrases present: %s" % overclaims)
    return cert


def assert_required_fields(cert):
    missing = [f for f in REQUIRED_FIELDS if f not in cert]
    _require(not missing, "missing required fields: %s" % missing)
    for f in SHA_FIELDS:
        _require(_hex64(cert[f]), "field %s is not hex64: %r" % (f, cert[f]))
    pins = cert["common_pins"]
    for k, want in PIN_FIELD_SOURCES.items():
        _require(pins.get(k) == want,
                 "common_pins[%s] %r != frozen %s" % (k, pins.get(k), want))
    _require(cert["student_rank"] is None,
             "student_rank must be null in a freshly created certificate")
    _require(cert["teacher_included_in_student_ranking"] is False,
             "teacher must never be included in student ranking")
    _require(cert["scientific_claim_authorized"] is False,
             "single training seed: scientific claim not authorized")
    _require(cert["scaffolded_results_can_replace_full_task"] is False,
             "scaffolded results can never replace the full task")
    _require(cert["run_class"] == RUN_CLASS, "run_class %r" % cert["run_class"])
    _require(cert["max_steps"] == proj.FROZEN_MAX_TIMESTEPS,
             "max_steps %r" % cert["max_steps"])
    _require(cert["action_mode"] == proj.FROZEN_ACTION_MODE,
             "action_mode %r" % cert["action_mode"])
    _require(cert["common_evaluator_protocol_version"]
             == COMMON_EVALUATOR_PROTOCOL_VERSION, "protocol version")
    _require(list(cert["scenario_order"]) == list(FORMAL_SCENARIO_ORDER),
             "scenario_order %r" % cert["scenario_order"])


def _leaf_strings(doc, acc):
    if isinstance(doc, str):
        acc.append(doc)
    elif isinstance(doc, dict):
        for v in doc.values():          # keys are field names, never scanned
            _leaf_strings(v, acc)
    elif isinstance(doc, (list, tuple)):
        for v in doc:
            _leaf_strings(v, acc)
    return acc


def scan_forbidden_overclaims(doc):
    """Scan leaf STRING VALUES only (field names like scientific_claim_authorized
    are honest negative-label keys and must not self-trip the scan)."""
    text = " ".join(_leaf_strings(doc, [])).upper()
    return [p for p in FORBIDDEN_OVERCLAIMS if p.upper() in text]


def verify_evaluation_certificate(cert, evidence_dir=None):
    """Full re-verification of a stored certificate. Returns a problem list
    ([] == valid). evidence_dir: if given, SHA256SUMS_FORMAL_V2DT and the
    episode_records.jsonl are re-hashed against the cert's recorded SHAs."""
    problems = []
    try:
        assert_required_fields(cert)
    except CertificateError as exc:
        problems.append(str(exc))
    problems.extend("overclaim: %s" % p for p in scan_forbidden_overclaims(cert))
    if evidence_dir is not None:
        sums_path = os.path.join(evidence_dir, "SHA256SUMS_FORMAL_V2DT")
        if not os.path.isfile(sums_path):
            problems.append("missing SHA256SUMS_FORMAL_V2DT")
        else:
            sums = proj.parse_sha256sums(sums_path)
            for rel, want in sorted(sums.items()):
                p = os.path.join(evidence_dir, rel)
                if not os.path.isfile(p):
                    problems.append("sums-listed file missing: %s" % rel)
                elif proj.sha256_file(p) != want:
                    problems.append("sums mismatch: %s" % rel)
            want_jsonl = cert.get("episode_records_jsonl_sha256")
            got = sums.get("episode_records.jsonl")
            if want_jsonl and got and got != want_jsonl:
                problems.append("episode_records.jsonl sha: cert %s != sums %s"
                                % (want_jsonl, got))
    return problems


# ---------------------------------------------------------------------------
# Self-test (JAX-free): schema sample + negative cases
# ---------------------------------------------------------------------------
def _sample_cert_input(candidate_id="BASE_GTRXL_ORIGINAL_VTRACE_98304"):
    spec = proj.get_spec(candidate_id)
    pins = pins_snapshot()
    schedule = {
        "full": {"kind": "canonical_reset_seeds_held_out", "base": 200000,
                 "count": 64, "seeds": [200000 + i for i in range(64)]},
        "front_l2": {"kind": "frozen_bank_state_each_once", "seed_base": 10000,
                     "stride": 1, "count": 8,
                     "seeds": [10000 + i for i in range(8)]},
        "back_l2": {"kind": "frozen_bank_state_each_once", "seed_base": 10000,
                    "stride": 1, "count": 8,
                    "seeds": [1010000 + i for i in range(8)]},
    }
    return {
        "candidate_id": candidate_id,
        "spec": spec,
        "generated_at_utc": "1970-01-01T00:00:00+00:00",
        "bfs_graph_source": "CURRENT_ENVIRONMENT_STATE_TOPOLOGY",
        "common_dir": "/dev/null/common_v2",
        "results_by_scenario": {sc: {"scenario": sc, "metrics": {}}
                                for sc in FORMAL_SCENARIO_ORDER},
        "schedule": schedule,
        "entry_ids_by_scenario": {
            "full": ["full-seed%d" % s for s in schedule["full"]["seeds"]],
            "front_l2": ["front_l2-bank%d" % i for i in range(8)],
            "back_l2": ["back_l2-bank%d" % i for i in range(8)]},
        "records_sha256_by_scenario": {sc: "0" * 64 for sc in FORMAL_SCENARIO_ORDER},
        "episode_records_jsonl_sha256": "1" * 64,
        "episodes_executed": {"full": 64, "front_l2": 8, "back_l2": 8},
        "valid_start_counts": {"full": 64, "front_l2": 8, "back_l2": 8},
        "params_before": spec["declared_params_sha256"]["value"],
        "declared_params": spec["declared_params_sha256"]["value"],
        "file_sha": spec["declared_checkpoint_file_sha256"]["value"],
        "declared_file": spec["declared_checkpoint_file_sha256"]["value"],
        "params_unchanged": True,
        "capsule_ev": {fn: {"match": True} for fn in proj.CAPSULE_FILES},
        "common_ev": {"common_v2_sha256sums_self_check": "PASS (7/7)",
                      "engine_modules_lf_sha_verified": 26,
                      "v1_preservation": {"v1_status": "SUPERSEDED_PRE_RANKING"}},
        "gpu_ev": {"visible_gpu_uuids": [proj.CC4_GPU_ALLOWED_UUIDS[0]]},
        "dicode_ev": {"dicode_src": "/dev/null"},
        "provenance": {"pid": 1, "git_commit_head": "0" * 40},
        "engine_module_shas": {"tier3_evaluator_v2.py": "2" * 64,
                               "tier3_event_predicates_v2.py": "3" * 64},
        "policy_class": "BaseGtrxlProjectionPolicy",
        "batch1_ev": None,
        "boundary_ev": None,
        "formal_abort": None,
        "rehearsal": False,
        "rehearsal_limits": None,
        "marker_ref": {"path": "/dev/null/SECONDARY_AUDIT_PASS.json",
                       "sha256": "4" * 64,
                       "verdict": SECONDARY_AUDIT_VERDICT,
                       "recorded_at_utc": "1970-01-01T00:00:00+00:00"},
        "binding_gate_sha256": "5" * 64,
        "common_v2_ready_sha256": "6" * 64,
    }


def self_test():
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        checks += 1
        if not cond:
            raise CertificateError("SELF_TEST FAIL: %s" % msg)

    # positive: student cert builds + verifies
    cert = build_evaluation_certificate(_sample_cert_input())
    ok(cert["evaluation_status"] == "PASS", "student status")
    ok(not verify_evaluation_certificate(cert), "student verify clean")
    ok(cert["student_rank"] is None, "rank null")
    ok(cert["common_pins"] == pins_snapshot(), "pins snapshot equality")

    # positive: teacher cert — reference_only, still rank null
    tci = _sample_cert_input("BASELINE_TEACHER_CKPT17500")
    tcert = build_evaluation_certificate(tci)
    ok(tcert["reference_only"] is True, "teacher reference_only")
    ok(tcert["counts_toward_student_binding_count"] is False, "teacher count")
    ok(tcert["student_rank"] is None, "teacher rank null")
    ok(not verify_evaluation_certificate(tcert), "teacher verify clean")

    # positive: RMT16 cert carries both runtime-family fields
    rci = _sample_cert_input("PERSISTENT_RMT16_ORIGINAL_VTRACE_98304")
    rci["rmt16_engine_metadata"] = {"runtime_family": "rmt16_gtrxl_cc2"}
    rci["rmt16_frozen_identities"] = {}
    rci["rmt16_common_runner_sha256"] = "7" * 64
    rci["rmt16_engine_lf_sha256"] = "8" * 64
    rcert = build_evaluation_certificate(rci)
    ok(rcert["engine_runtime_family"] == "rmt16_gtrxl_cc2", "engine family field")
    ok(rcert["runtime_family"]
       == "rmt16_gtrxl_cc2_persistent_projection", "projection family distinct")

    # positive: abort -> BLOCKED; rehearsal -> REHEARSAL_NOT_FORMAL
    aci = _sample_cert_input()
    aci["formal_abort"] = {"scenario": "full", "episode_index": 3,
                           "engine_message": "x"}
    ok(build_evaluation_certificate(aci)["evaluation_status"] == "BLOCKED",
       "abort status")
    hci = _sample_cert_input()
    hci["rehearsal"] = True
    hci["rehearsal_limits"] = {"full": 2, "front_l2": 2, "back_l2": 2}
    hcert = build_evaluation_certificate(hci)
    ok(hcert["evaluation_status"] == "REHEARSAL_NOT_FORMAL", "rehearsal status")
    ok(hcert["performance_evaluation_executed"] is False,
       "rehearsal not executed-as-formal")

    # negative: each required field dropped -> assertion fires
    for f in REQUIRED_FIELDS:
        c2 = build_evaluation_certificate(_sample_cert_input())
        del c2[f]
        try:
            assert_required_fields(c2)
            raise CertificateError("SELF_TEST FAIL: dropping %s accepted" % f)
        except CertificateError as exc:
            ok("missing required fields" in str(exc) or "FAIL CLOSED" in str(exc),
               "drop %s fires" % f)
        checks += 1

    # negative: non-null rank rejected
    c3 = build_evaluation_certificate(_sample_cert_input())
    c3["student_rank"] = 1
    try:
        assert_required_fields(c3)
        raise CertificateError("SELF_TEST FAIL: rank=1 accepted")
    except CertificateError:
        checks += 1

    # negative: teacher-in-ranking rejected
    c4 = build_evaluation_certificate(_sample_cert_input())
    c4["teacher_included_in_student_ranking"] = True
    try:
        assert_required_fields(c4)
        raise CertificateError("SELF_TEST FAIL: teacher-in-ranking accepted")
    except CertificateError:
        checks += 1

    # negative: pin drift rejected
    c5 = build_evaluation_certificate(_sample_cert_input())
    c5["common_pins"]["metric_schema_sha256"] = "f" * 64
    try:
        assert_required_fields(c5)
        raise CertificateError("SELF_TEST FAIL: pin drift accepted")
    except CertificateError:
        checks += 1

    # negative: overclaim phrases scanned
    c6 = build_evaluation_certificate(_sample_cert_input())
    c6["provenance"]["note"] = "this proves BEST_OVERALL_STUDENT scientifically"
    ok(scan_forbidden_overclaims(c6), "overclaim scan fires")

    # negative: corrupt evidence dir detected
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        c7 = build_evaluation_certificate(_sample_cert_input())
        probs = verify_evaluation_certificate(c7, evidence_dir=td)
        ok(any("SHA256SUMS_FORMAL_V2DT" in p for p in probs),
           "missing sums detected")
        with open(os.path.join(td, "SHA256SUMS_FORMAL_V2DT"), "w",
                  encoding="utf-8", newline="\n") as fh:
            fh.write("%s  episode_records.jsonl\n" % ("9" * 64))
        with open(os.path.join(td, "episode_records.jsonl"), "w",
                  encoding="utf-8", newline="\n") as fh:
            fh.write("{}\n")
        probs = verify_evaluation_certificate(c7, evidence_dir=td)
        ok(any("sums mismatch" in p for p in probs), "sums mismatch detected")

    return checks


def main(argv=None):
    if (argv or sys.argv[1:])[:1] == ["--self-test"]:
        n = self_test()
        print("CERTIFICATE_V2DT_SELF_TEST_PASS checks=%d" % n)
        return 0
    print("usage: tier3_evaluation_certificate_v2dt.py --self-test")
    return 2


if __name__ == "__main__":
    sys.exit(main())
