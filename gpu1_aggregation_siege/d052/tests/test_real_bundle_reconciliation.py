"""Auto regression tests for the REAL Phase 2.5 bundle reconciliation (spec §7).

Implements the bundle's regression_test_spec.md R1-R6 + the §3 replay anchors,
run against the REAL bundle (committed at orchestration/experiments/
d052_modeler_shadow_v1/). OFFLINE, NO LLM, NO TRAINING, read-only on the bundle.

These ADD to the 283-test canonical firewall suite; they never lower any gate.
"""
from __future__ import annotations

import json

import pytest

from d052.reconciliation.real_bundle import (
    REPLAY_INPUTS_REL, REPO_ROOT, bundle_dir, load_bundle_json, load_judgments,
    verify_bundle_integrity, verify_judgment_hashes,
)

pytest.importorskip("numpy")

pytestmark = pytest.mark.skipif(
    not bundle_dir().exists(),
    reason="real Phase 2.5 bundle not present in this checkout")


# --- bundle integrity -------------------------------------------------------
def test_bundle_integrity_13_of_13():
    r = verify_bundle_integrity()
    assert r["verified"] == 13, r
    assert r["ok"] is True, r


# --- R1. forbidden targets ---------------------------------------------------
def _legacy_pool():
    p = REPO_ROOT / REPLAY_INPUTS_REL / "frozen_candidate_pool_round4.json"
    return json.loads(p.read_text(encoding="utf-8"))["candidates"]


def test_r1_legacy_targets_are_firewalled():
    from d052.counterfactual.firewall import (
        TargetFirewallError, assert_execution_mapping_rejects,
        assert_target_firewall, classify_target,
    )
    cands = _legacy_pool()
    distinct = sorted({t for c in cands for t in c.get("target_achievements", [])})
    codes = {t: classify_target(t) for t in distinct}
    unknown = [t for t, code in codes.items()
               if code == TargetFirewallError.UNKNOWN_TARGET_FORBIDDEN]
    assert len(unknown) == 18, (len(unknown), unknown)
    legal = [t for t, code in codes.items() if code is None]
    assert sorted(legal) == ["collect_wood", "defeat_zombie", "place_table"]
    # the full legacy target set MUST be rejected with a specific code
    with pytest.raises(TargetFirewallError) as ei:
        assert_target_firewall(distinct)
    assert ei.value.code == TargetFirewallError.UNKNOWN_TARGET_FORBIDDEN
    # salted / hash-modulo schemes stay banned with their precise codes
    assert classify_target("collect_wood::salt=ab12") == \
        TargetFirewallError.SALTED_TARGET_FORBIDDEN
    assert assert_execution_mapping_rejects(["38"]) == \
        TargetFirewallError.HASH_MODULO_TARGET_FORBIDDEN
    assert assert_execution_mapping_rejects([""]) == \
        TargetFirewallError.EMPTY_TARGET_FORBIDDEN


# --- R2. B/C matched-field check ---------------------------------------------
def test_r2_prompt_registry_matched_contract():
    from d052.reconciliation.prompt_profile_contract import verify_prompt_registry
    checks = verify_prompt_registry()
    assert checks["all_6_prompt_hashes_match"] is True, checks
    assert checks["B_C_only_difference_is_appended_frozen_profile"] is True, checks
    assert checks["profile_block_byte_identical_across_roles"] is True, checks
    assert checks["raw_summary_sha256_reproduced"] is True, checks
    assert checks["candidate_block_sha256_reproduced"] is True, checks
    assert checks["profile_json_sha256_reproduced"] is True, checks
    assert checks["embedded_profile_deep_equal_llm_interpretation"] is True, checks
    assert checks["ok"] is True, checks


def test_r2_protocol_invariants():
    proto = load_bundle_json("protocol.json")
    assert proto["protocol_id"] == "d052_phase25_matched_counterfactual_v1"
    assert set(proto["arms"]) == {"B_NO_MODELER", "C_WITH_MODELER"}
    inv = proto["controlled_invariants"]
    assert inv["temperature"] == 0
    assert inv["http_timeout_s"] == 180
    assert inv["max_attempts"] == 8
    assert inv["candidate_order"] == "sorted(task_id)"
    assert str(inv["seed"]).startswith("none")      # no RNG, fully deterministic
    assert "StudentProfile" in proto["only_allowed_variation"]


# --- R3. judgment replay ------------------------------------------------------
def test_r3_judgment_hash_tamper_evidence_192_of_192():
    r = verify_judgment_hashes()
    assert r["checked"] == 192, r
    assert r["all_ok"] is True, r


def test_r3_record_coverage_and_schema():
    for arm in ("B", "C"):
        recs = load_judgments(arm)
        assert len(recs) == 96
        by_role = {}
        for r in recs:
            assert r["parse_status"] == "ok"
            assert r["decision"] in ("accept", "hold", "reject")
            by_role[r["role"]] = by_role.get(r["role"], 0) + 1
            for v in r["raw_scores"].values():
                assert isinstance(v, (int, float)) and v == v  # numeric, not NaN
            if r["role"] == "critic":
                assert set(r["flags"]) == {"too_hard", "already_mastered"}
        assert by_role == {"tutor": 32, "critic": 32, "explorer": 32}


def test_r3_adapter_instantiates_canonical_role_judgments():
    from d052.reconciliation.judgment_adapter import adapt_arm
    pr = load_bundle_json("prompt_registry.json")
    for arm in ("B", "C"):
        # critic policy MUST be named explicitly (fail-closed, no default)
        adapted = adapt_arm(load_judgments(arm),
                            critic_reject_rule="decision_reject",
                            prompt_version=pr["prompt_version"])
        assert len(adapted) == 96
        critics = [a for a in adapted if a.role_judgment.role.value == "critic"]
        assert len(critics) == 32
        assert all(a.derived.get("critic_reject_rule") == "decision_reject"
                   for a in critics)
        assert all(a.derived.get("critic_reject_value") is not None for a in critics)
        # non-critic judgments carry NO derived critic bit
        assert all(not a.derived for a in adapted
                   if a.role_judgment.role.value != "critic")
        # the original record is preserved verbatim in the envelope
        for a in adapted:
            assert a.original_record["judgment_hash_sha256"] == \
                a.audit_envelope["judgment_hash_sha256"]


def test_r3_critic_reject_rule_sensitivity_is_recorded():
    """The two candidate derivation rules differ on exactly 2 records/arm-pair.

    decision_reject -> 40 True over B+C; flags_too_hard -> 38. This pins the open
    director question as a FACT; the historical replay consumes neither (it uses
    raw critic_penalty exactly as the legacy selector did).
    """
    from d052.reconciliation.judgment_adapter import adapt_arm
    counts = {}
    for rule in ("decision_reject", "flags_too_hard"):
        n = 0
        for arm in ("B", "C"):
            adapted = adapt_arm(load_judgments(arm), critic_reject_rule=rule)
            n += sum(1 for a in adapted if a.derived.get("critic_reject_value"))
        counts[rule] = n
    assert counts == {"decision_reject": 40, "flags_too_hard": 38}, counts


def test_r3_glm_role_normalization_log():
    from d052.reconciliation.judgment_adapter import adapt_arm, normalization_log
    adapted = []
    for arm in ("B", "C"):
        adapted += adapt_arm(load_judgments(arm),
                             critic_reject_rule="decision_reject")
    log = normalization_log(adapted)
    assert len(log["records"]) == 192
    assert log["n_normalized"] == 18          # glm role-echo quirks
    assert normalization_log(adapted)["normalization_log_hash"] == \
        log["normalization_log_hash"]          # deterministic
    for e in log["records"]:
        assert e["normalization_log_hash"]
        if e["normalized"]:
            assert "glm" in e["normalization_reason"]
            assert e["raw_role_label"] != e["canonical_role_label"]


# --- R4. selector determinism + §3 anchors ------------------------------------
def test_r4_replay_reproduces_all_historical_anchors():
    from d052.reconciliation.replay import run_replay
    r = run_replay()
    assert r["ALL_ANCHORS_PASS"] is True, r["checks"]
    rec = r["recomputed"]
    assert rec["B_selection_hash"] == r["expected_anchors"]["B_selection_hash"] \
        == "82571538e5299ea9"
    assert rec["C_selection_hash"] == r["expected_anchors"]["C_selection_hash"] \
        == "868a57268d66b90b"
    assert rec["pool_hash"] == "1902b71a5d86fa00"
    assert rec["change"] == "4/8"
    assert r["checks"]["B_determinism_bitidentical"] is True
    assert r["checks"]["C_determinism_bitidentical"] is True
    assert r["checks"]["selector_rng_seed_null"] is True


# --- R5. canonical target semantics (post-repair path vs legacy) ---------------
def test_r5_canonical_pool_candidate_maps_as_intended():
    from d052.counterfactual.pipeline import build_phase25_pool
    from d052.execution.mapper import (
        build_execution_certificate, canonical_compiled_spec,
    )
    pool = build_phase25_pool()
    c = pool.candidates[0]
    cert = build_execution_certificate(c, canonical_compiled_spec(c, "r5_probe/x"))
    assert cert.executed_as_intended is True
    assert all(cert.gates.values())


def test_r5_legacy_unknown_targets_rejected_at_boundary():
    from d052.counterfactual.firewall import assert_execution_mapping_rejects
    cands = _legacy_pool()
    rejected = 0
    for c in cands:
        names = c.get("target_achievements", [])
        from d052.counterfactual.firewall import classify_target
        if any(classify_target(t) is not None for t in names):
            code = assert_execution_mapping_rejects(names)
            assert code  # specific banned class returned
            rejected += 1
    assert rejected == 25                      # 25/32 candidates blocked outright


# --- R6. profile integrity ----------------------------------------------------
def test_r6_student_profile_integrity():
    from d052.reconciliation.prompt_profile_contract import verify_student_profile
    checks, built = verify_student_profile()
    assert checks["profile_hash_sha256_reproduced"] is True, checks
    assert checks["insufficient_evidence_kept_null"] is True, checks
    assert built.measured_count == 7
    assert built.mastered_count == 1           # WAKE_UP (0.9844)
    assert checks["ok"] is True, checks


# --- §13 frozen labels ---------------------------------------------------------
def test_frozen_labels():
    from d052.reconciliation.prompt_profile_contract import verify_prompt_registry
    from d052.reconciliation.replay import run_replay
    r = run_replay()
    assert ("PASS" if r["ALL_ANCHORS_PASS"] else "FAIL") == "PASS"
    assert verify_prompt_registry()["ok"] is True    # REAL_PHASE25_MATCHED_PROTOCOL


# --- §8 evidence tier separation ------------------------------------------------
def test_evidence_tiers_and_blocked_templates():
    rep = REPO_ROOT / "gpu1_aggregation_siege" / "reports" / "phase25"
    tiers = json.loads((rep / "real_bundle_evidence_tiers.json").read_text("utf-8"))
    A = tiers["tier_A_REAL_LEGACY_PHASE25"]["frozen_labels"]
    assert A["MODELER_MATCHED_SELECTION_EFFECT"] == "CONFIRMED"
    assert A["LEGACY_MATCHED_SELECTED_SET_CHANGE"] == "4/8"
    assert A["MODELER_LEARNING_VALUE"] == "UNTESTED"
    assert A["LEGACY_PHASE25_SELECTION_EVIDENCE"] == "MECHANISM_ONLY"
    assert A["LEGACY_PHASE25_SELECTED8_TRAINING_READY"] is False
    assert A["LEGACY_PHASE25_PERFORMANCE_INTERPRETATION_ALLOWED"] is False
    B = tiers["tier_B_SYNTHETIC_CANONICAL_FIXTURE"]["frozen_labels"]
    assert B["CANONICAL_SYNTHETIC_FIXTURE_ENGINEERING_TEST"] == "PASS"
    assert B["CANONICAL_SYNTHETIC_MODELER_SELECTION_CHANGE"] == "1/8"
    assert B["CANONICAL_SYNTHETIC_RESULT_IS_SCIENTIFIC_EVIDENCE"] is False
    C = tiers["tier_C_REAL_CANONICAL_POOL"]["frozen_labels"]
    assert C["REAL_CANONICAL_POOL_EXPERIMENT"] == "NOT_RUN"
    assert C["REAL_CANONICAL_POOL_TRAINING_TIMESTEPS"] == 0
    # the two real-canonical cell templates exist and are BLOCKED, unregistered
    tdir = REPO_ROOT / "gpu1_aggregation_siege" / "phase25_real_canonical_cell_templates"
    for arm in ("B", "C"):
        t = json.loads((tdir / f"CELL_PHASE25_REAL_CANONICAL_{arm}.json")
                       .read_text("utf-8"))
        assert t["template_status"] == "BLOCKED_PENDING_REAL_CANONICAL_JUDGMENTS"
        assert t["training_authorized"] is False
        assert t["fields_PENDING_real_values"]["intended_total_timesteps"] == 0
        assert t["blockers"]


# --- D052_PREMERGE_CORRECTION_V2: critic policy fail-closed -------------------
def test_ccv2_critic_policy_fail_closed_whole_arm():
    from d052.reconciliation.judgment_adapter import AdapterError, adapt_arm
    with pytest.raises(AdapterError) as ei:
        adapt_arm(load_judgments("B"))          # no policy -> fail closed
    assert ei.value.code == AdapterError.CRITIC_POLICY_REQUIRED


def test_ccv2_critic_policy_fail_closed_single_critic_record():
    from d052.reconciliation.judgment_adapter import AdapterError, adapt_judgment
    rec = next(r for r in load_judgments("B") if r["role"] == "critic")
    with pytest.raises(AdapterError) as ei:
        adapt_judgment(rec)                     # no policy -> fail closed
    assert ei.value.code == AdapterError.CRITIC_POLICY_REQUIRED
    # non-critic records adapt fine without any policy, and derive nothing
    tutor = next(r for r in load_judgments("B") if r["role"] == "tutor")
    assert adapt_judgment(tutor).derived == {}


def test_ccv2_unknown_policy_rejected():
    from d052.reconciliation.judgment_adapter import (
        AdapterError, adapt_arm, adapt_judgment,
    )
    with pytest.raises(AdapterError) as ei:
        adapt_arm(load_judgments("B"), critic_reject_rule="invalid_policy")
    assert ei.value.code == AdapterError.UNKNOWN_RULE
    rec = next(r for r in load_judgments("B") if r["role"] == "critic")
    with pytest.raises(AdapterError) as ei2:
        adapt_judgment(rec, critic_reject_rule="invalid_policy")
    assert ei2.value.code == AdapterError.UNKNOWN_RULE


def test_ccv2_explicit_rules_full_provenance_and_counts():
    """Both candidate rules are allowed ONLY when named; each derived record
    carries rule + value + derived=True + the 'no raw bit' note. The two
    candidate counts stay pinned as FACTS (40 vs 38 over B+C)."""
    from d052.reconciliation.judgment_adapter import adapt_arm
    counts = {}
    for rule in ("decision_reject", "flags_too_hard"):
        n_true = 0
        for arm in ("B", "C"):
            for a in adapt_arm(load_judgments(arm), critic_reject_rule=rule):
                if a.role_judgment.role.value != "critic":
                    continue
                d = a.derived
                assert d["critic_reject_rule"] == rule
                assert d["derived"] is True
                assert "no raw critic_reject bit" in d["note"]
                assert a.audit_envelope["judgment_hash_sha256"]
                if d["critic_reject_value"]:
                    n_true += 1
        counts[rule] = n_true
    assert counts == {"decision_reject": 40, "flags_too_hard": 38}


def test_ccv2_replay_overlap_jaccard_and_anchors_unchanged():
    """The fail-closed adapter change must not touch any replay anchor: the
    replay consumes raw critic_penalty, never a derived critic_reject."""
    from d052.reconciliation.replay import run_replay
    r = run_replay()
    assert r["ALL_ANCHORS_PASS"] is True
    rec = r["recomputed"]
    assert rec["B_selection_hash"] == "82571538e5299ea9"
    assert rec["C_selection_hash"] == "868a57268d66b90b"
    assert rec["pool_hash"] == "1902b71a5d86fa00"
    assert rec["change"] == "4/8"
    assert len(rec["overlap"]) == 4
    assert abs(rec["jaccard"] - 0.3333) < 1e-4
    assert r["checks"]["overlap_4"] is True
    assert r["checks"]["jaccard_match"] is True


# --- D052_PREMERGE_CORRECTION_V2: Henry invalid-archive preservation -----------
def test_ccv2_henry_invalid_archive_preserved():
    base = REPO_ROOT / "experiments" / "henry_dicode_student_upgrade"
    readme = base / "01_d052" / "README.md"
    removed = base / "inventory" / "d052_data_removed_by_request.txt"
    assert readme.is_file(), "01_d052/README.md must be preserved"
    assert removed.is_file(), "d052_data_removed_by_request.txt must be preserved"
    text = readme.read_text(encoding="utf-8")
    assert ("INVALID_DATA_CODE_ONLY" in text
            or ("invalid experiment data, code retained only" in text
                and "must not be used as scientific evidence" in text)), \
        "README must keep the invalid-data / code-only declaration"
