#!/usr/bin/env python3
"""CC4 Tier3 — FORMAL evaluation certificate, V3_COMPOSITE_EVENT generation.

Authorized by 总控 ruling CC4_REPAIR_NEG20_COMPOSITE_EVENT_SEMANTICS_AND_COMPLETE_
FORMAL_EVALUATION_V3. V2 is archived CLOSED_INCONCLUSIVE_PARTICIPATION (winner
null); V3 is a NEW, independently-auditable evaluator semantic repair — it must
NEVER masquerade as V2 (distinct schema / cert_version / protocol ids / full SHAs).

This module keeps the V2DT certificate module's DISCIPLINE verbatim (value-binding,
no signatures; required-field assertions; forbidden-overclaim scan; rank=null at
creation and never rewritten; teacher reference-only; honest negative labels) and
adds the V3 surface:
  * common_evaluator_protocol_version = V3_COMPOSITE_EVENT;
    neg20_protocol = NEG20_V3_PRIMARY_SECONDARY_EVENTS;
  * taxonomy_v3_lf_sha256 pin (LF-normalized SHA256 of tier3_taxonomy_v3.py) —
    identical across all seven certificates (single classifier source of truth);
  * the SAME 11 frozen common/bank pins as V2DT (common evaluator / runner /
    evaluation profile / metric schema / environment lock / ABI / assembly
    manifest / sha256sums / full seed-profile / FRONT bank content / BACK bank
    content) — 总控 §六: all seven V3 certificates reference the identical set;
  * per-arm reuse provenance (FULL offline reuse / FRONT offline reclassification
    / BACK completion or CONTROL reuse+resign), each recording source V2 episode
    SHA, the V3 classifier SHA, the V3 result SHA, classification_only and
    environment_rerun flags;
  * a composite-event disclosure layer (per-scenario composite counts, secondary
    event counts, primary outcome counts) carried from tier3_taxonomy_v3.summarize_v3;
  * a no-V2-masquerade guard (schema/protocol must be V3; the V2DT schema/version
    strings may not appear as leaf values; legitimate V2-archive references use
    distinct keys such as v2_archive.* and are NOT tripped).

JAX-free by construction: imports only frozen pin constants (via the V2 binding
driver / projection registry, both stdlib-only at module level) and the JAX-free
tier3_taxonomy_v3 (which imports only the frozen tier3_metrics library).
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
import tier3_taxonomy_v3 as taxonomy_v3               # noqa: E402  (JAX-free)

SCHEMA = "mechanism_UED.tier3_evaluation_certificate/v3"
CERT_VERSION = "tier3_evaluation_certificate/v3"
COMMON_EVALUATOR_PROTOCOL_VERSION = taxonomy_v3.FORMAL_EVALUATOR_PROTOCOL  # V3_COMPOSITE_EVENT
NEG20_PROTOCOL = taxonomy_v3.NEG20_PROTOCOL                  # NEG20_V3_PRIMARY_SECONDARY_EVENTS
RUN_CLASS = "FORMAL_EVALUATION"
FORMAL_SCENARIO_ORDER = ("full", "front_l2", "back_l2")
RECOMPUTE_STATUS = "CC4_RECOMPUTED_MATCH_VIA_OWNER_PROTOCOL"
SUMS_FILENAME = "SHA256SUMS_FORMAL_V3"

# The archived V2 outcome (referenced for continuity; never overwritten).
V2_ARCHIVE_STATUS = "CLOSED_INCONCLUSIVE_PARTICIPATION"
V2_ARCHIVE_WINNER = None
# V2DT schema/version strings — forbidden as leaf VALUES in a V3 cert (masquerade).
_V2DT_SCHEMA_STRING = "mechanism_UED.tier3_evaluation_certificate/v2dt"
_V2DT_CERT_VERSION_STRING = "tier3_evaluation_certificate/v2dt"
_V2_PROTOCOL_STRING = "V2_DYNAMIC_TOPOLOGY"

# The 11 frozen common/bank pins, identical key names/values to the V2DT binding
# certificates (single source of truth: the frozen V2 binding driver + registry).
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
    "neg20_protocol", "taxonomy_version_v3", "failure_rule_version_v3",
    "taxonomy_v3_lf_sha256",
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
    "results_by_scenario", "reuse_provenance_by_scenario",
    "composite_event_disclosure", "common_pins",
    "v2_archive", "engine_identity", "provenance", "audit", "rehearsal",
)

SHA_FIELDS = (
    "params_sha256", "checkpoint_file_sha256", "episode_records_jsonl_sha256",
    "taxonomy_v3_lf_sha256",
)

# Per-arm reuse provenance block required keys (total control §五 reuse chain).
REUSE_PROVENANCE_REQUIRED_KEYS = (
    "reuse_status", "source", "classification_only", "environment_rerun",
    "v3_classifier_sha256", "v3_result_sha256", "source_v2_episode_sha256",
)
REUSE_STATUS_VOCABULARY = frozenset([
    "REUSED_PASS",            # FULL offline reuse R1-R9 all green
    "REUSED_RECLASSIFIED",    # FRONT offline reclassification (no env rerun)
    "REUSED_RESIGNED",        # BACK CONTROL reuse + V3 re-sign (no env rerun)
    "COMPLETED",              # BACK first-run completion of a 0/8 skeleton
    "REJECT",                 # FULL reuse gate rejected -> honest block (no rerun)
])

FORBIDDEN_OVERCLAIMS = (
    "BEST_OVERALL_STUDENT",
    "ALL_STUDENT_BAKEOFF_WINNER",
    "SCIENTIFIC_CLAIM_AUTHORIZED",
    "SCIENTIFIC_SUPERIORITY",
    "STRONG_STUDENT_V1",
    "OVERALL_STRONG_STUDENT_SELECTION_AUTHORIZED",
    "REPLACE_FULL_TASK",
    "SUCCEEDED_BY_TRAINING",
    # V3-specific: a V3 certificate may not claim to be a V2 certificate.
    "MASQUERADE_AS_V2",
    "IDENTICAL_TO_V2_CERTIFICATE",
)


class CertificateError(Exception):
    pass


def _require(cond, msg):
    if not cond:
        raise CertificateError("FAIL CLOSED (CERTIFICATE_V3): %s" % msg)


def _hex64(v):
    return isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) is not None


def pins_snapshot():
    """The frozen common/bank pin set (identical to V2DT; single source of truth)."""
    return dict(PIN_FIELD_SOURCES)


def build_evaluation_certificate(ci):
    """Build one V3 formal evaluation certificate from the driver's cert input.

    V3 additions to the V2DT cert input (assembled by tier3_formal_evaluation_v3.py):
      reuse_provenance_by_scenario {sc: {reuse_status, source, classification_only,
          environment_rerun, v3_classifier_sha256, v3_result_sha256,
          source_v2_episode_sha256, ...}},
      composite_event_disclosure {composite_episode_count_by_scenario,
          secondary_event_counts_by_scenario, primary_outcome_counts_by_scenario},
      marker_ref -> V3_REPAIR_AUTHORIZATION marker (总控 ruling).
    results_by_scenario values are tier3_taxonomy_v3.summarize_v3 outputs (frozen
    metrics envelope + composite layer).
    """
    spec = ci["spec"]
    is_teacher = spec["candidate_class"] == "TEACHER_REFERENCE"

    pins = pins_snapshot()
    for k, v in pins.items():
        _require(_hex64(v), "pin %s not a hex64 constant: %r" % (k, v))
    taxonomy_sha = taxonomy_v3.module_lf_sha256()
    _require(_hex64(taxonomy_sha), "taxonomy_v3 LF-SHA not hex64: %r" % taxonomy_sha)

    aborted = ci["formal_abort"] is not None
    rehearsal = bool(ci.get("rehearsal"))
    reuse_rejected = any(
        ci["reuse_provenance_by_scenario"][sc]["reuse_status"] == "REJECT"
        for sc in FORMAL_SCENARIO_ORDER)
    if aborted:
        status = "BLOCKED"
    elif rehearsal:
        status = "REHEARSAL_NOT_FORMAL"
    elif reuse_rejected:
        status = "BLOCKED_REUSE_REJECT"
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
        "neg20_protocol": NEG20_PROTOCOL,
        "taxonomy_version_v3": taxonomy_v3.TAXONOMY_V3_VERSION,
        "failure_rule_version_v3": taxonomy_v3.FAILURE_RULE_VERSION_V3,
        "taxonomy_v3_lf_sha256": taxonomy_sha,
        "bfs_graph_source": ci["bfs_graph_source"],
        "run_class": RUN_CLASS,
        "evaluation_status": status,
        "formal_abort": ci["formal_abort"],
        # frozen contract (formal scale; unchanged by the V3 semantic repair)
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
        "formal_performance_results_authorized_by_repair_marker":
            (not rehearsal) and (ci.get("marker_ref") is not None),
        # eligibility / ranking discipline
        "formal_student_ranking_eligible": spec["formal_student_ranking_eligible"],
        "strong_student_selection_eligible":
            spec["strong_student_selection_eligible"],
        "reference_only": spec["reference_only"],
        "student_rank": None,     # null at creation; NEVER rewritten (see docstring)
        "student_rank_publication": "FORMAL_RANKING_SUMMARY_V3.json only; the "
            "registry self-test forbids writing ranks into the registry and this "
            "certificate is %s-bound and immutable after creation" % SUMS_FILENAME,
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
        # V3 results: frozen metrics envelope + composite layer (summarize_v3)
        "results_by_scenario": ci["results_by_scenario"],
        # per-arm reuse provenance (总控 §五 reuse chain)
        "reuse_provenance_by_scenario": ci["reuse_provenance_by_scenario"],
        # composite-event disclosure (总控 §二)
        "composite_event_disclosure": ci["composite_event_disclosure"],
        # the frozen common/bank pin set (identical to V2DT) + V3 classifier pin
        "common_root": ci["common_dir"],
        "common_pins": pins,
        "common_sha_match_status": ci["common_ev"].get("common_v2_sha256sums_self_check"),
        # V2 archive reference (continuity; the V2 evidence is NOT modified)
        "v2_archive": {
            "v2_status": V2_ARCHIVE_STATUS,
            "v2_winner": V2_ARCHIVE_WINNER,
            "v2_ranking_valid": False,
            "v2_summary_sha256": ci.get("v2_archive_summary_sha256"),
            "v2_gate_sha256": ci.get("v2_archive_gate_sha256"),
            "v2_evidence_modified_by_v3": False,
            "note": "V2 is archived CLOSED_INCONCLUSIVE_PARTICIPATION under the "
                    "frozen single-label classifier; V3 is a distinct, separately-"
                    "auditable evaluator semantic repair and does not overwrite, "
                    "delete, or rewrite any V2 evidence",
        },
        # engine identity
        "engine_identity": {
            "evaluator_v2_module_sha256": ci["engine_module_shas"]["tier3_evaluator_v2.py"],
            "predicates_v2_module_sha256": ci["engine_module_shas"]["tier3_event_predicates_v2.py"],
            "engine_lf_sha256_v2": dict(smokev2.FROZEN_V2_ENGINE_LF_SHA256),
            "engine_lf_sha256_v1_frozen": dict(proj.FROZEN_ENGINE_LF_SHA256),
            "engine_modules_verified_by_stage1":
                ci["common_ev"].get("engine_modules_lf_sha_verified"),
            "evaluate_v1_called_by_v3": False,   # V3 NEVER calls frozen ev.evaluate()
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
        # start authorization (V3 repair marker, NOT the V2 SECONDARY_AUDIT marker)
        "audit": {
            "repair_authorization_marker": ci.get("marker_ref"),
            "binding_gate_sha256": ci.get("binding_gate_sha256"),
            "authorization_note": "V3 formal run authorized by V3_REPAIR_AUTHORIZATION "
                "marker recording the 总控 ruling (composite-event semantic repair); "
                "interface smoke evidence is NOT a performance result",
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
    masquerade = scan_no_v2_masquerade(cert)
    _require(not masquerade, "V2-masquerade detected: %s" % masquerade)
    return cert


def _validate_reuse_block(sc, block):
    _require(isinstance(block, dict),
             "reuse_provenance[%s] not a dict" % sc)
    missing = [k for k in REUSE_PROVENANCE_REQUIRED_KEYS if k not in block]
    _require(not missing,
             "reuse_provenance[%s] missing keys: %s" % (sc, missing))
    _require(block["reuse_status"] in REUSE_STATUS_VOCABULARY,
             "reuse_provenance[%s] reuse_status %r not registered"
             % (sc, block["reuse_status"]))
    _require(isinstance(block["classification_only"], bool),
             "reuse_provenance[%s] classification_only not bool" % sc)
    _require(isinstance(block["environment_rerun"], bool),
             "reuse_provenance[%s] environment_rerun not bool" % sc)
    # classification-only reuse never reruns the environment.
    _require(not (block["classification_only"] and block["environment_rerun"]),
             "reuse_provenance[%s] classification_only and environment_rerun both "
             "true (contradiction)" % sc)
    _require(_hex64(block["v3_classifier_sha256"]),
             "reuse_provenance[%s] v3_classifier_sha256 not hex64" % sc)
    _require(block["v3_classifier_sha256"] == taxonomy_v3.module_lf_sha256(),
             "reuse_provenance[%s] v3_classifier_sha256 != taxonomy_v3 LF-SHA" % sc)
    _require(_hex64(block["v3_result_sha256"]),
             "reuse_provenance[%s] v3_result_sha256 not hex64" % sc)
    src = block["source_v2_episode_sha256"]
    _require(src is None or _hex64(src),
             "reuse_provenance[%s] source_v2_episode_sha256 not null/hex64" % sc)
    # A fresh completion run has no V2 source episode evidence.
    if block["reuse_status"] == "COMPLETED":
        _require(block["environment_rerun"] is True,
                 "reuse_provenance[%s] COMPLETED must set environment_rerun" % sc)
        _require(block["classification_only"] is False,
                 "reuse_provenance[%s] COMPLETED is not classification_only" % sc)
    else:
        _require(block["environment_rerun"] is False,
                 "reuse_provenance[%s] reused arm must not rerun the environment" % sc)


def assert_required_fields(cert):
    missing = [f for f in REQUIRED_FIELDS if f not in cert]
    _require(not missing, "missing required fields: %s" % missing)
    for f in SHA_FIELDS:
        _require(_hex64(cert[f]), "field %s is not hex64: %r" % (f, cert[f]))
    pins = cert["common_pins"]
    for k, want in PIN_FIELD_SOURCES.items():
        _require(pins.get(k) == want,
                 "common_pins[%s] %r != frozen %s" % (k, pins.get(k), want))
    _require(cert["taxonomy_v3_lf_sha256"] == taxonomy_v3.module_lf_sha256(),
             "taxonomy_v3_lf_sha256 drifted from the live classifier module")
    # per-arm reuse provenance: all three arms present + validated
    reuse = cert["reuse_provenance_by_scenario"]
    _require(isinstance(reuse, dict), "reuse_provenance_by_scenario not a dict")
    for sc in FORMAL_SCENARIO_ORDER:
        _require(sc in reuse, "reuse_provenance missing scenario %s" % sc)
        _validate_reuse_block(sc, reuse[sc])
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
    _require(cert["neg20_protocol"] == NEG20_PROTOCOL, "neg20 protocol")
    _require(list(cert["scenario_order"]) == list(FORMAL_SCENARIO_ORDER),
             "scenario_order %r" % cert["scenario_order"])
    _require(cert["engine_identity"]["evaluate_v1_called_by_v3"] is False,
             "V3 must never call the frozen ev.evaluate()")


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


def scan_no_v2_masquerade(cert):
    """A V3 certificate may not present itself as a V2 certificate. Structural
    checks (schema/protocol must be V3) plus a leaf-value scan for the V2DT
    schema/version strings. Legitimate V2-archive references live under the
    v2_archive.* keys with distinct values (CLOSED_INCONCLUSIVE_PARTICIPATION,
    V2 summary/gate SHAs) and do NOT embed the V2DT schema string, so they are
    not tripped."""
    problems = []
    if cert.get("schema") != SCHEMA:
        problems.append("schema %r != %s" % (cert.get("schema"), SCHEMA))
    if cert.get("cert_version") != CERT_VERSION:
        problems.append("cert_version %r != %s" % (cert.get("cert_version"), CERT_VERSION))
    if cert.get("common_evaluator_protocol_version") != COMMON_EVALUATOR_PROTOCOL_VERSION:
        problems.append("protocol %r != %s"
                        % (cert.get("common_evaluator_protocol_version"),
                           COMMON_EVALUATOR_PROTOCOL_VERSION))
    leaves = " ".join(_leaf_strings(cert, []))
    if _V2DT_SCHEMA_STRING in leaves:
        problems.append("leaf value contains V2DT schema string")
    if _V2DT_CERT_VERSION_STRING in leaves:
        problems.append("leaf value contains V2DT cert_version string")
    return problems


def verify_evaluation_certificate(cert, evidence_dir=None):
    """Full re-verification of a stored V3 certificate. Returns a problem list
    ([] == valid). evidence_dir: if given, SHA256SUMS_FORMAL_V3 and the
    episode_records.jsonl are re-hashed against the cert's recorded SHAs."""
    problems = []
    try:
        assert_required_fields(cert)
    except CertificateError as exc:
        problems.append(str(exc))
    problems.extend("overclaim: %s" % p for p in scan_forbidden_overclaims(cert))
    problems.extend("masquerade: %s" % p for p in scan_no_v2_masquerade(cert))
    if evidence_dir is not None:
        sums_path = os.path.join(evidence_dir, SUMS_FILENAME)
        if not os.path.isfile(sums_path):
            problems.append("missing %s" % SUMS_FILENAME)
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
# Self-test (JAX-free): schema sample + negative cases + V3-specific guards
# ---------------------------------------------------------------------------
def _sample_reuse(status, classification_only, environment_rerun,
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


def _sample_cert_input(candidate_id="BASE_GTRXL_ORIGINAL_VTRACE_98304"):
    spec = proj.get_spec(candidate_id)
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
    results = {sc: {"scenario": sc,
                    "metrics": {"primary": {}, "dense": {}},
                    "composite_event_layer": {"composite_episode_count": 0}}
               for sc in FORMAL_SCENARIO_ORDER}
    return {
        "candidate_id": candidate_id,
        "spec": spec,
        "generated_at_utc": "1970-01-01T00:00:00+00:00",
        "bfs_graph_source": "CURRENT_ENVIRONMENT_STATE_TOPOLOGY",
        "common_dir": "/dev/null/common_v2",
        "results_by_scenario": results,
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
        "reuse_provenance_by_scenario": {
            "full": _sample_reuse("REUSED_PASS", True, False),
            "front_l2": _sample_reuse("REUSED_RECLASSIFIED", True, False),
            "back_l2": _sample_reuse("COMPLETED", False, True, source_v2_sha=None),
        },
        "composite_event_disclosure": {
            "composite_episode_count_by_scenario":
                {sc: 0 for sc in FORMAL_SCENARIO_ORDER},
            "secondary_event_counts_by_scenario":
                {sc: {} for sc in FORMAL_SCENARIO_ORDER},
            "primary_outcome_counts_by_scenario":
                {sc: {} for sc in FORMAL_SCENARIO_ORDER},
        },
        "v2_archive_summary_sha256": "c" * 64,
        "v2_archive_gate_sha256": "d" * 64,
        "marker_ref": {"path": "/dev/null/V3_REPAIR_AUTHORIZATION.json",
                       "sha256": "4" * 64,
                       "ruling_task": "CC4_REPAIR_NEG20_COMPOSITE_EVENT_SEMANTICS_"
                                      "AND_COMPLETE_FORMAL_EVALUATION_V3",
                       "recorded_at_utc": "1970-01-01T00:00:00+00:00"},
        "binding_gate_sha256": "5" * 64,
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
    ok(cert["common_evaluator_protocol_version"] == "V3_COMPOSITE_EVENT",
       "protocol is V3")
    ok(cert["neg20_protocol"] == "NEG20_V3_PRIMARY_SECONDARY_EVENTS",
       "neg20 protocol is V3")
    ok(cert["taxonomy_v3_lf_sha256"] == taxonomy_v3.module_lf_sha256(),
       "taxonomy pin bound to live module")
    ok(cert["v2_archive"]["v2_status"] == "CLOSED_INCONCLUSIVE_PARTICIPATION",
       "v2 archive referenced")
    ok(cert["v2_archive"]["v2_evidence_modified_by_v3"] is False,
       "v2 evidence untouched flag")
    ok(not scan_no_v2_masquerade(cert), "no masquerade on clean cert")

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

    # positive: abort -> BLOCKED; rehearsal -> REHEARSAL_NOT_FORMAL;
    #           FULL reuse REJECT -> BLOCKED_REUSE_REJECT
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
    rji = _sample_cert_input()
    rji["reuse_provenance_by_scenario"]["full"] = _sample_reuse(
        "REJECT", False, False)
    ok(build_evaluation_certificate(rji)["evaluation_status"]
       == "BLOCKED_REUSE_REJECT", "reuse-reject status")

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

    # negative: taxonomy pin drift rejected
    c5b = build_evaluation_certificate(_sample_cert_input())
    c5b["taxonomy_v3_lf_sha256"] = "e" * 64
    try:
        assert_required_fields(c5b)
        raise CertificateError("SELF_TEST FAIL: taxonomy pin drift accepted")
    except CertificateError:
        checks += 1

    # negative: classifier-sha mismatch in a reuse block rejected
    c5c = build_evaluation_certificate(_sample_cert_input())
    c5c["reuse_provenance_by_scenario"]["front_l2"]["v3_classifier_sha256"] = "9" * 64
    try:
        assert_required_fields(c5c)
        raise CertificateError("SELF_TEST FAIL: classifier-sha mismatch accepted")
    except CertificateError:
        checks += 1

    # negative: classification_only AND environment_rerun both true rejected
    c5d = build_evaluation_certificate(_sample_cert_input())
    c5d["reuse_provenance_by_scenario"]["front_l2"]["environment_rerun"] = True
    try:
        assert_required_fields(c5d)
        raise CertificateError("SELF_TEST FAIL: classify+rerun contradiction accepted")
    except CertificateError:
        checks += 1

    # negative: COMPLETED without environment_rerun rejected
    c5e = build_evaluation_certificate(_sample_cert_input())
    c5e["reuse_provenance_by_scenario"]["back_l2"] = _sample_reuse(
        "COMPLETED", False, False, source_v2_sha=None)
    try:
        assert_required_fields(c5e)
        raise CertificateError("SELF_TEST FAIL: COMPLETED w/o rerun accepted")
    except CertificateError:
        checks += 1

    # negative: evaluate_v1_called_by_v3 must be False
    c5f = build_evaluation_certificate(_sample_cert_input())
    c5f["engine_identity"]["evaluate_v1_called_by_v3"] = True
    try:
        assert_required_fields(c5f)
        raise CertificateError("SELF_TEST FAIL: evaluate_v1_called accepted")
    except CertificateError:
        checks += 1

    # negative: overclaim phrases scanned
    c6 = build_evaluation_certificate(_sample_cert_input())
    c6["provenance"]["note"] = "this proves BEST_OVERALL_STUDENT scientifically"
    ok(scan_forbidden_overclaims(c6), "overclaim scan fires")

    # negative: V2-masquerade leaf value scanned
    c6b = build_evaluation_certificate(_sample_cert_input())
    c6b["provenance"]["note"] = "identical to mechanism_UED.tier3_evaluation_certificate/v2dt"
    ok(scan_no_v2_masquerade(c6b), "masquerade scan fires on v2dt schema leaf")

    # negative: corrupt evidence dir detected (SHA256SUMS_FORMAL_V3)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        c7 = build_evaluation_certificate(_sample_cert_input())
        probs = verify_evaluation_certificate(c7, evidence_dir=td)
        ok(any(SUMS_FILENAME in p for p in probs), "missing sums detected")
        with open(os.path.join(td, SUMS_FILENAME), "w",
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
        print("CERTIFICATE_V3_SELF_TEST_PASS checks=%d" % n)
        return 0
    print("usage: tier3_evaluation_certificate_v3.py --self-test")
    return 2


if __name__ == "__main__":
    sys.exit(main())
