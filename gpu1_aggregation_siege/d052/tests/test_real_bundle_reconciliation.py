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
                            critic_reject_derivation_rule="decision_reject",
                            prompt_version=pr["prompt_version"])
        assert len(adapted) == 96
        critics = [a for a in adapted if a.role_judgment.role.value == "critic"]
        assert len(critics) == 32
        assert all(a.derived.get("critic_reject_derivation_rule") == "decision_reject"
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
            adapted = adapt_arm(load_judgments(arm), critic_reject_derivation_rule=rule)
            n += sum(1 for a in adapted if a.derived.get("critic_reject_value"))
        counts[rule] = n
    assert counts == {"decision_reject": 40, "flags_too_hard": 38}, counts


def test_r3_glm_role_normalization_log():
    from d052.reconciliation.judgment_adapter import adapt_arm, normalization_log
    adapted = []
    for arm in ("B", "C"):
        adapted += adapt_arm(load_judgments(arm),
                             critic_reject_derivation_rule="decision_reject")
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
        adapt_arm(load_judgments("B"), critic_reject_derivation_rule="invalid_policy")
    assert ei.value.code == AdapterError.UNKNOWN_RULE
    rec = next(r for r in load_judgments("B") if r["role"] == "critic")
    with pytest.raises(AdapterError) as ei2:
        adapt_judgment(rec, critic_reject_derivation_rule="invalid_policy")
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
            for a in adapt_arm(load_judgments(arm), critic_reject_derivation_rule=rule):
                if a.role_judgment.role.value != "critic":
                    continue
                d = a.derived
                assert d["critic_reject_derivation_rule"] == rule
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


# --- D052_PREMERGE_SEMANTIC_CLEANUP_V3: critic scope + split + allowlist -------
def _reports():
    return REPO_ROOT / "gpu1_aggregation_siege" / "reports"


def test_no_ambiguous_critic_policy_pass_label():
    """V3 §2/§7: the unscoped D052_CRITIC_POLICY label must never read PASS.

    It is kept only as a deprecated compatibility field pointing at the split
    replacement fields; no auto-gate may consume it.
    """
    lab = json.loads((_reports() / "d052_canonical_frozen_labels.json")
                     .read_text("utf-8"))
    assert lab["D052_CRITIC_POLICY"] != "PASS"
    assert lab["D052_CRITIC_POLICY"] == "DEPRECATED_AMBIGUOUS_DO_NOT_USE"
    assert lab["D052_CRITIC_POLICY_deprecated"] is True
    repl = lab["D052_CRITIC_POLICY_replacement_fields"]
    assert "D052_SYNTHETIC_CRITIC_SELECTOR_ENGINEERING" in repl
    assert "REAL_CANONICAL_CRITIC_REJECT_DERIVATION_RULE" in repl
    assert "REAL_CANONICAL_CRITIC_SELECTION_POLICY" in repl
    # the phase25 label file carries the layering block, never a bare PASS
    p25 = json.loads((_reports() / "d052_phase25_canonical_frozen_labels.json")
                     .read_text("utf-8"))
    assert "D052_CRITIC_POLICY" not in p25
    assert "critic_label_layering" in p25


def test_real_canonical_critic_fields_are_split():
    """V3 §3/§7: both cell templates keep the two critic dimensions split,
    PENDING and separately blocking — no ambiguous single critic_policy."""
    from d052.reconciliation.tier_c_gate import validate_template_critic_fields
    tdir = REPO_ROOT / "gpu1_aggregation_siege" / \
        "phase25_real_canonical_cell_templates"
    for arm in ("B", "C"):
        t = json.loads((tdir / f"CELL_PHASE25_REAL_CANONICAL_{arm}.json")
                       .read_text("utf-8"))
        sel = t["fields_PENDING_real_values"]["selector"]
        assert "critic_policy" not in sel, f"arm {arm}: ambiguous field remains"
        assert sel["critic_reject_derivation_rule"] == "PENDING_DIRECTOR_DECISION"
        assert sel["critic_selection_policy"] == "PENDING_DIRECTOR_DECISION"
        rec = t["fields_PENDING_real_values"]["execution_certificate_policy_record"]
        assert rec["fail_closed_if_either_missing"] is True
        v = validate_template_critic_fields(t)
        assert v["ok"] is True, (arm, v["problems"])


def test_real_canonical_missing_derivation_rule_blocks():
    """V3 §4/§7: dimension A missing -> BOTH the adapter and the Tier-C gate
    fail closed, even if dimension B is frozen."""
    from d052.reconciliation.judgment_adapter import AdapterError, adapt_arm
    from d052.reconciliation.tier_c_gate import (
        TierCPolicyError, require_frozen_critic_fields,
    )
    with pytest.raises(AdapterError) as ei:
        adapt_arm(load_judgments("B"))      # no derivation rule named
    assert ei.value.code == AdapterError.CRITIC_POLICY_REQUIRED
    for bad in (None, "", "UNDECIDED", "NONE", "PENDING_DIRECTOR_DECISION"):
        with pytest.raises(TierCPolicyError) as ei2:
            require_frozen_critic_fields(bad, "hard_veto")   # B frozen, A not
        assert ei2.value.code == TierCPolicyError.CRITIC_DERIVATION_RULE_REQUIRED


def test_real_canonical_missing_selection_policy_blocks():
    """V3 §4/§7: dimension B missing -> the Tier-C gate fails closed, even if
    dimension A is frozen; the two dimensions never substitute for each other."""
    from d052.reconciliation.tier_c_gate import (
        LEGAL_CRITIC_SELECTION_POLICIES, TierCPolicyError,
        require_frozen_critic_fields,
    )
    assert set(LEGAL_CRITIC_SELECTION_POLICIES) == \
        {"hard_veto", "soft_penalty", "score_only"}
    for bad in (None, "", "UNDECIDED", "NONE", "PENDING_DIRECTOR_DECISION"):
        with pytest.raises(TierCPolicyError) as ei:
            require_frozen_critic_fields("decision_reject", bad)  # A frozen, B not
        assert ei.value.code == TierCPolicyError.CRITIC_SELECTION_POLICY_REQUIRED
    # BOTH frozen + legal -> the two-field policy record is returned
    rec = require_frozen_critic_fields("decision_reject", "soft_penalty")
    assert rec == {"critic_reject_derivation_rule": "decision_reject",
                   "critic_selection_policy": "soft_penalty",
                   "both_frozen": True}


def test_synthetic_engineering_pass_does_not_freeze_real_policy():
    """V3 §2/§7: the synthetic fixture engineering PASS stays scoped and must
    not appear as a frozen real canonical critic policy anywhere."""
    lab = json.loads((_reports() / "d052_canonical_frozen_labels.json")
                     .read_text("utf-8"))
    assert lab["D052_SYNTHETIC_CRITIC_SELECTOR_ENGINEERING"] == "PASS"
    assert "synthetic" in lab["D052_SYNTHETIC_CRITIC_SELECTOR_ENGINEERING_scope"]
    assert lab["REAL_CANONICAL_CRITIC_REJECT_DERIVATION_RULE"] == "UNDECIDED"
    assert lab["REAL_CANONICAL_CRITIC_SELECTION_POLICY"] == "UNDECIDED"
    assert lab["DEFAULT_CRITIC_REJECT_DERIVATION_RULE"] == "NONE"
    assert lab["DEFAULT_CRITIC_SELECTION_POLICY"] == "NONE"
    assert lab["REAL_CANONICAL_CONVERSION_WITHOUT_CRITIC_RULE"] == "BLOCKED"
    assert lab["REAL_CANONICAL_SELECTION_WITHOUT_CRITIC_POLICY"] == "BLOCKED"
    # mirrored in the phase25 layering block
    lay = json.loads((_reports() / "d052_phase25_canonical_frozen_labels.json")
                     .read_text("utf-8"))["critic_label_layering"]
    for k in ("D052_SYNTHETIC_CRITIC_SELECTOR_ENGINEERING",
              "REAL_CANONICAL_CRITIC_REJECT_DERIVATION_RULE",
              "REAL_CANONICAL_CRITIC_SELECTION_POLICY",
              "DEFAULT_CRITIC_REJECT_DERIVATION_RULE",
              "DEFAULT_CRITIC_SELECTION_POLICY",
              "REAL_CANONICAL_CONVERSION_WITHOUT_CRITIC_RULE",
              "REAL_CANONICAL_SELECTION_WITHOUT_CRITIC_POLICY"):
        assert lay[k] == lab[k], k
    # and in the evidence tiers: the real dimensions are NOT_RUN-era UNDECIDED
    tiers = json.loads((_reports() / "phase25" / "real_bundle_evidence_tiers.json")
                       .read_text("utf-8"))
    two = tiers["critic_policy_two_dimensions"]
    assert two["dimension_A_derivation"][
        "REAL_CANONICAL_CRITIC_REJECT_DERIVATION_RULE"] == "UNDECIDED"
    assert two["dimension_B_selection_consumption"][
        "REAL_CANONICAL_CRITIC_SELECTION_POLICY"] == "UNDECIDED"
    assert two["synthetic_pass_does_not_freeze_real_policy"] is True


def test_gitignore_only_allows_frozen_outputs():
    """V3 §5/§7: the outputs dir exemption is a strict per-file allowlist — a
    future unapproved file is ignored; every allowlisted file is trackable."""
    import subprocess
    out_rel = "orchestration/experiments/d052_modeler_shadow_v1/outputs"
    out_abs = REPO_ROOT / out_rel
    future = out_abs / "future_unapproved_output.json"

    def check_ignore(rel):
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "check-ignore", "-q", rel],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode

    future.write_text("{}\n", encoding="utf-8")
    try:
        assert check_ignore(f"{out_rel}/future_unapproved_output.json") == 0, \
            "future unapproved file must stay ignored"
        allow = json.loads((_reports() / "phase25" / "frozen_output_allowlist.json")
                           .read_text("utf-8"))
        assert allow["file_count"] == 21
        for e in allow["files"]:
            assert check_ignore(e["path"]) != 0, \
                f"allowlisted file must be trackable: {e['path']}"
    finally:
        future.unlink()          # spec: delete the temp file afterwards
    assert not future.exists()


def test_frozen_output_allowlist_matches_git():
    """V3 §5/§7: the allowlist equals `git ls-files outputs/` exactly, and
    every SHA256/size matches the file on disk."""
    import hashlib
    import subprocess
    allow = json.loads((_reports() / "phase25" / "frozen_output_allowlist.json")
                       .read_text("utf-8"))
    listed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files",
         "orchestration/experiments/d052_modeler_shadow_v1/outputs/"],
        capture_output=True, text=True, check=True
    ).stdout.split()
    assert sorted(e["path"] for e in allow["files"]) == sorted(listed)
    for e in allow["files"]:
        data = (REPO_ROOT / e["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == e["sha256"], e["path"]
        assert len(data) == e["size_bytes"], e["path"]
        assert e["contains_secret"] is False
        assert e["contains_checkpoint"] is False
        assert e["approved_for_git"] is True
    assert allow["scan"]["judgment_hash_reverified"] == "192/192"


def test_historical_replay_unchanged():
    """V3 §7: none of the V3 renames/splits may move a historical anchor —
    the replay consumes raw critic_penalty, never either critic dimension."""
    from d052.reconciliation.replay import run_replay
    r = run_replay()
    assert r["ALL_ANCHORS_PASS"] is True, r["checks"]
    rec = r["recomputed"]
    assert rec["B_selection_hash"] == "82571538e5299ea9"
    assert rec["C_selection_hash"] == "868a57268d66b90b"
    assert rec["pool_hash"] == "1902b71a5d86fa00"
    assert rec["change"] == "4/8"
    assert sorted(rec["overlap"]) == rec["overlap"] and len(rec["overlap"]) == 4
    assert abs(rec["jaccard"] - 0.3333) < 1e-4
    assert r["checks"]["overlap_4"] is True
    assert r["checks"]["jaccard_match"] is True
    assert r["checks"]["B_determinism_bitidentical"] is True
    assert r["checks"]["C_determinism_bitidentical"] is True


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
