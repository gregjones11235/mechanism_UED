#!/usr/bin/env python3
"""CC4 Tier3 — FROZEN RMT16 carry-mode comparison (task §九; scope-corrected).

CANDIDATE SCOPE (总控范围修正): this round evaluates ONLY the two RMT16 +
Original V-trace replay arms — Persistent (cross-segment carry) vs Reset128
(per-segment reset). It is a MECHANISM comparison
(PERSISTENT_CROSS_SEGMENT_CARRY_VS_RESET128), NOT an overall strong-student
selection. This module therefore NEVER outputs or implies
PROVISIONAL_STRONG_STUDENT_RECOMMENDATION / STRONG_STUDENT_V1 /
BEST_OVERALL_STUDENT / ALL_STUDENT_BAKEOFF_WINNER; every such string is a
forbidden over-claim and fails closed. The filename is retained to avoid churn.

The comparison rule is FIXED pre-run, machine-readable, and embedded verbatim in
the comparison output so the winner can be STRICTLY RECOMPUTED from the final
metrics by any auditor. It is lexicographic over the two arms' frozen-schedule
performance results:

    1. higher FULL success_count        (DEFEAT_KOBOLD_SR over the 64 held-out seeds)
    2. tie → higher FRONT transition_count (floor 2->3 over the 8 frozen FRONT states)
    3. tie → higher FRONT mean GRAPH_DISTANCE_PROGRESS (dense, tie tolerance 1e-12;
       task §九: if the mean progress also ties the rule MAY favor either arm — this
       implementation proceeds deterministically to level 4, which is one allowed
       deterministic choice)
    4. tie → higher BACK defeat_count   (over the 8 frozen BACK states)
    5. all tied → INCONCLUSIVE

Output: RMT16_CARRY_MODE_WINNER = PERSISTENT | RESET128 | INCONCLUSIVE, ALWAYS
with the fixed scope/boundary fields CANDIDATE_SCOPE=
RMT16_ORIGINAL_VTRACE_PAIR_ONLY, MECHANISM_QUESTION=
PERSISTENT_CROSS_SEGMENT_CARRY_VS_RESET128,
OVERALL_STRONG_STUDENT_SELECTION_AUTHORIZED=false, STRONG_STUDENT_V1=NOT_SELECTED,
EXISTING_STUDENT_BAKEOFF_REQUIRED=true, SINGLE_TRAINING_SEED=true,
SCIENTIFIC_SUPERIORITY_CLAIM=false, REQUIRES_MULTI_SEED_CONFIRMATION=true. The
rule is NEVER modified from interface-smoke results or post-hoc inspection
(task §九).

The comparator CLI verifies BOTH arms' finalized evidence first (SHA256SUMS + FULL
certificate verification + identical run class / contract SHA / frozen schedule
counts) and only then compares — never on raw logs.
"""
from __future__ import annotations

import json
import os
import sys

# Runnable-as-script AND importable-as-package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_evaluation_certificate as certmod      # noqa: E402
import tier3_checkpoint_contract as contractmod     # noqa: E402
import tier3_evaluation_runner as runner            # noqa: E402

SCHEMA = "mechanism_UED.tier3_rmt16_carry_mode_comparison/v1"

FULL = certmod.FULL
FRONT = certmod.FRONT
BACK = certmod.BACK
ARMS = ("persistent", "reset128")
RUN_CLASS = "PROVISIONAL_STRONG_STUDENT_SELECTION"
SELECTION_TIE_TOLERANCE = 1e-12
FROZEN_COUNTS = {FULL: certmod.PERF_FULL_N, FRONT: 8, BACK: 8}

# 总控范围修正 — fixed scope / boundary fields bound into EVERY comparison output.
# This round compares ONLY the two RMT16 + Original V-trace carry modes; it is not
# an overall strong-student selection.
CANDIDATE_SCOPE = "RMT16_ORIGINAL_VTRACE_PAIR_ONLY"
MECHANISM_QUESTION = "PERSISTENT_CROSS_SEGMENT_CARRY_VS_RESET128"
WINNER_VALUES = ("PERSISTENT", "RESET128", "INCONCLUSIVE")
FIXED_SCOPE_FIELDS = {
    "CANDIDATE_SCOPE": CANDIDATE_SCOPE,
    "MECHANISM_QUESTION": MECHANISM_QUESTION,
    "OVERALL_STRONG_STUDENT_SELECTION_AUTHORIZED": False,
    "STRONG_STUDENT_V1": "NOT_SELECTED",
    "EXISTING_STUDENT_BAKEOFF_REQUIRED": True,
    "SINGLE_TRAINING_SEED": True,
    "SCIENTIFIC_SUPERIORITY_CLAIM": False,
    "REQUIRES_MULTI_SEED_CONFIRMATION": True,
}

# The FROZEN comparison rule — embedded verbatim in every comparison output so the
# winner is strictly recomputable (task §九 extra clause).
SELECTION_RULE = {
    "rule_version": "tier3_rmt16_carry_mode_comparison_rule/v1",
    "kind": "lexicographic_fixed_pre_run",
    "question": MECHANISM_QUESTION,
    "candidate_scope": CANDIDATE_SCOPE,
    "output_field": "RMT16_CARRY_MODE_WINNER",
    "tie_tolerance": SELECTION_TIE_TOLERANCE,
    "criteria": [
        {"level": 1, "metric": "full_success_count", "scenario": FULL,
         "primary": "DEFEAT_KOBOLD_SR", "direction": "higher"},
        {"level": 2, "metric": "front_transition_count", "scenario": FRONT,
         "primary": "P_FRONT_FLOOR_TRANSITION_REACHED_GIVEN_VALID_START",
         "direction": "higher"},
        {"level": 3, "metric": "front_mean_graph_distance_progress",
         "scenario": FRONT, "dense": "GRAPH_DISTANCE_PROGRESS", "direction": "higher",
         "note": "task §九: if the mean progress also ties the rule may favor either "
                 "arm; this implementation continues deterministically to level 4 "
                 "(a deterministic continuation is one allowed choice)"},
        {"level": 4, "metric": "back_defeat_count", "scenario": BACK,
         "primary": "P_DEFEAT_KOBOLD_GIVEN_VALID_BACK_START", "direction": "higher"},
        {"level": 5, "metric": "all_criteria_tied", "decision": "INCONCLUSIVE"},
    ],
    "outputs": list(WINNER_VALUES),
    "never_modified_from": ["interface_smoke_results", "post_hoc_inspection",
                            "checkpoint_or_result_based_scaffold_filtering",
                            "other_students_or_bakeoff_results"],
}

# task §六 + 总控范围修正: the comparison may NEVER contain these claims. The
# strong-student / bakeoff vocabulary is forbidden outright (scope: RMT16 +
# Original V-trace carry-mode pair only), in any "KEY=VALUE" or bare form.
FORBIDDEN_COMPARISON_CLAIMS = certmod.FORBIDDEN_OVERCLAIMS | {
    "SOTA", "FORMAL_SCIENTIFIC_PASS", "PERSISTENT_PROVEN_BETTER",
    "RESET128_PROVEN_BETTER", "TIER3_SOLVED",
    # 总控范围修正: no overall student selection is authorized this round.
    "PROVISIONAL_STRONG_STUDENT_RECOMMENDATION",
    "STRONG_STUDENT_V1",
    "STRONG_STUDENT_V1=PERSISTENT",
    "STRONG_STUDENT_V1=RESET128",
    "BEST_OVERALL_STUDENT",
    "ALL_STUDENT_BAKEOFF_WINNER",
}


class FailClosed(Exception):
    """Hard stop on any selection / comparison violation."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


# ---------------------------------------------------------------------------
# Metric extraction (from a finalized arm's evaluation_result.json)
# ---------------------------------------------------------------------------
def extract_arm_metrics(result_doc: dict) -> dict:
    """Pull the four frozen selection counters out of one arm's evaluation_result
    document, validating the frozen 64/8/8 valid-start counts (task §七). Any drift
    (wrong run class, missing scenario, wrong valid_starts, missing dense progress)
    fails closed — the comparator never compares unverified numbers."""
    require(isinstance(result_doc, dict),
            "FAIL CLOSED: evaluation_result document is not a dict")
    require(result_doc.get("run_class") == RUN_CLASS,
            "FAIL CLOSED: evaluation_result run_class %r != %s (only finalized "
            "provisional selection runs are comparable)"
            % (result_doc.get("run_class"), RUN_CLASS))
    results = result_doc.get("results")
    require(isinstance(results, dict),
            "FAIL CLOSED: evaluation_result has no results dict")
    out = {"arm": None}
    cc = result_doc.get("checkpoint_contract") or {}
    if isinstance(cc, dict) and cc.get("arm"):
        out["arm"] = cc.get("arm")

    def _scenario(sc):
        r = results.get(sc)
        require(isinstance(r, dict),
                "FAIL CLOSED: evaluation_result missing scenario %r" % sc)
        m = r.get("metrics") or {}
        primary = m.get("primary") or {}
        require(primary.get("valid_starts") == FROZEN_COUNTS[sc],
                "FAIL CLOSED: scenario %s valid_starts %r != frozen %d (the arm did "
                "not run the full frozen schedule)"
                % (sc, primary.get("valid_starts"), FROZEN_COUNTS[sc]))
        successes = primary.get("successes")
        require(isinstance(successes, int) and not isinstance(successes, bool)
                and 0 <= successes <= FROZEN_COUNTS[sc],
                "FAIL CLOSED: scenario %s successes %r is not an int in [0, %d]"
                % (sc, successes, FROZEN_COUNTS[sc]))
        return r, successes

    _full, full_n = _scenario(FULL)
    front_r, front_n = _scenario(FRONT)
    _back, back_n = _scenario(BACK)
    dense = (front_r.get("metrics") or {}).get("dense") or {}
    mean = dense.get("value")
    require(isinstance(mean, (int, float)) and not isinstance(mean, bool)
            and 0.0 <= float(mean) <= 1.0,
            "FAIL CLOSED: FRONT mean GRAPH_DISTANCE_PROGRESS %r is not a float in "
            "[0,1] (every frozen FRONT state must carry a scored progress)" % (mean,))
    out.update({
        "full_success_count": full_n,
        "front_transition_count": front_n,
        "front_mean_progress": float(mean),
        "back_defeat_count": back_n,
        "valid_starts": dict(FROZEN_COUNTS),
    })
    return out


# ---------------------------------------------------------------------------
# The frozen lexicographic rule
# ---------------------------------------------------------------------------
def select_provisional(persistent: dict, reset128: dict) -> dict:
    """Apply the frozen rule to two arms' extracted metrics (task §九). Pure and
    deterministic: identical inputs → identical recommendation + trace."""
    for name, m in (("persistent", persistent), ("reset128", reset128)):
        require(isinstance(m, dict),
                "FAIL CLOSED: %s metrics is not a dict" % name)
        for k in ("full_success_count", "front_transition_count",
                  "front_mean_progress", "back_defeat_count"):
            require(k in m, "FAIL CLOSED: %s metrics missing %r" % (name, k))
    arms = {"persistent": persistent, "reset128": reset128}
    trace = []

    def decide(level, metric, pa, ra):
        if pa > ra:
            return "persistent"
        if ra > pa:
            return "reset128"
        trace.append({"level": level, "metric": metric,
                      "persistent": pa, "reset128": ra, "decision": "tie"})
        return None

    # levels 1, 2: exact integer counts
    d = decide(1, "full_success_count",
               persistent["full_success_count"], reset128["full_success_count"])
    if d is None:
        d = decide(2, "front_transition_count",
                   persistent["front_transition_count"],
                   reset128["front_transition_count"])
    if d is None:
        # level 3: float mean progress with the frozen tie tolerance
        pa = persistent["front_mean_progress"]
        ra = reset128["front_mean_progress"]
        if pa - ra > SELECTION_TIE_TOLERANCE:
            d = "persistent"
        elif ra - pa > SELECTION_TIE_TOLERANCE:
            d = "reset128"
        else:
            trace.append({"level": 3, "metric": "front_mean_graph_distance_progress",
                          "persistent": pa, "reset128": ra,
                          "tolerance": SELECTION_TIE_TOLERANCE, "decision": "tie"})
    if d is None:
        d = decide(4, "back_defeat_count",
                   persistent["back_defeat_count"], reset128["back_defeat_count"])
    decided_at_level = None
    if d is None:
        recommendation = "INCONCLUSIVE"
        trace.append({"level": 5, "metric": "all_criteria_tied",
                      "decision": "INCONCLUSIVE"})
    else:
        recommendation = d.upper()
        # the deciding entry is the last non-tie context: find its level
        levels = {1: "full_success_count", 2: "front_transition_count",
                  3: "front_mean_graph_distance_progress", 4: "back_defeat_count"}
        for lvl in (1, 2, 3, 4):
            tied_metrics = {t["metric"] for t in trace if t["decision"] == "tie"}
            if levels[lvl] not in tied_metrics:
                decided_at_level = lvl
                break
        trace.append({"decision": recommendation,
                      "decided_at_level": decided_at_level})
    require(recommendation in WINNER_VALUES,
            "FAIL CLOSED: winner %r not in frozen outputs %s"
            % (recommendation, WINNER_VALUES))
    result = dict(FIXED_SCOPE_FIELDS)
    result.update({
        "RMT16_CARRY_MODE_WINNER": recommendation,
        "decided_at_level": decided_at_level,
        "trace": trace,
        "rule": SELECTION_RULE,
        "arms": arms,
    })
    assert_no_forbidden_claims(result)
    return result


def assert_no_forbidden_claims(doc: dict):
    """task §六: the comparison may never contain SOTA / FORMAL_SCIENTIFIC_PASS /
    *_PROVEN_BETTER / TIER3_SOLVED — scan every string value, fail closed."""
    def walk(v):
        if isinstance(v, str):
            yield v
        elif isinstance(v, dict):
            for x in v.values():
                yield from walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                yield from walk(x)
    bad = sorted({s for s in walk(doc) if s in FORBIDDEN_COMPARISON_CLAIMS})
    require(not bad,
            "FAIL CLOSED (task §六): comparison carries forbidden claim(s) %s" % bad)


# ---------------------------------------------------------------------------
# Cross-arm comparator (verifies BOTH arms' evidence before comparing)
# ---------------------------------------------------------------------------
def _verify_arm_dir(arm: str, arm_dir: str, expect_contract_sha: str = None) -> dict:
    """SHA256SUMS + FULL certificate verification + frozen run identity for one
    finalized arm directory (task §九: compare only verified evidence)."""
    require(arm in ARMS, "FAIL CLOSED: arm %r not in %s" % (arm, ARMS))
    runner.verify_dir_sha256sums(arm_dir)          # fail closed on any mismatch
    with open(os.path.join(arm_dir, "evaluation_certificate.json"), "r",
              encoding="utf-8") as fh:
        certs = json.load(fh)
    require(isinstance(certs, dict) and set(certs) == {FULL, FRONT, BACK},
            "FAIL CLOSED: %s evaluation_certificate.json must cover exactly the "
            "three frozen scenarios %s (got %s)"
            % (arm, [FULL, FRONT, BACK], sorted(certs) if isinstance(certs, dict) else certs))
    contract_shas = set()
    for sc, cert in certs.items():
        certmod.assert_eval_binding_complete(cert)        # NEG27/NEG29/NEG37 full
        certmod.assert_scaffold_hash_not_global(cert)     # NEG24
        certmod.assert_scaffold_does_not_claim_full_success(cert)   # NEG25
        b = cert["eval_binding"]
        require(b["run_class"] == RUN_CLASS,
                "FAIL CLOSED: %s/%s certificate run_class %r != %s"
                % (arm, sc, b["run_class"], RUN_CLASS))
        require(b["checkpoint_contract_arm"] == arm,
                "FAIL CLOSED: %s directory certificate %s binds contract arm %r "
                "(arm mismatch)" % (arm, sc, b["checkpoint_contract_arm"]))
        require(b["carry_mode"] == arm,
                "FAIL CLOSED: %s/%s carry_mode %r != arm" % (arm, sc, b["carry_mode"]))
        require(b["action_mode"] == certmod.FROZEN_ACTION_MODE
                and b["max_timesteps"] == 4096,
                "FAIL CLOSED: %s/%s frozen action identity violated "
                "(action_mode=%r max_timesteps=%r)"
                % (arm, sc, b["action_mode"], b["max_timesteps"]))
        require(b["scientific_claim_authorized"] is False
                and b["single_training_seed"] is True
                and b["provisional_selection_only"] is True,
                "FAIL CLOSED: %s/%s scientific-boundary flags violated" % (arm, sc))
        contract_shas.add(b["checkpoint_contract_sha256"])
    require(len(contract_shas) == 1,
            "FAIL CLOSED: %s certificates disagree on checkpoint_contract_sha256 %s"
            % (arm, sorted(contract_shas)))
    contract_sha = contract_shas.pop()
    if expect_contract_sha is not None:
        require(contract_sha == expect_contract_sha,
                "FAIL CLOSED: %s checkpoint_contract_sha256 %s != contract %s"
                % (arm, contract_sha[:16], expect_contract_sha[:16]))
    with open(os.path.join(arm_dir, "evaluation_result.json"), "r",
              encoding="utf-8") as fh:
        result_doc = json.load(fh)
    metrics = extract_arm_metrics(result_doc)
    require(metrics.get("arm") in (None, arm),
            "FAIL CLOSED: %s evaluation_result checkpoint_contract.arm %r != %s"
            % (arm, metrics.get("arm"), arm))
    return {"dir": arm_dir, "sha256sums_verified": True,
            "certificate_binding_verified": True,
            "checkpoint_contract_sha256": contract_sha, "metrics": metrics}


def _repo_relative_or_raw(path: str) -> str:
    import tier3_source_audit as audit
    if not os.path.isabs(path):
        return path
    try:
        return os.path.relpath(path, str(audit.repo_root()))
    except ValueError:
        return path


def compare_dirs(persistent_dir: str, reset128_dir: str, out_path: str,
                 contract_path: str = None) -> dict:
    """Verify BOTH arms, then apply the frozen rule and write
    cross_arm_comparison.json (task §九/§十一)."""
    expect = None
    if contract_path is not None:
        expect = contractmod.load_contract(contract_path)["checkpoint_contract_sha256"]
    p = _verify_arm_dir("persistent", persistent_dir, expect)
    r = _verify_arm_dir("reset128", reset128_dir, expect)
    require(p["checkpoint_contract_sha256"] == r["checkpoint_contract_sha256"],
            "FAIL CLOSED: the two arms were evaluated against DIFFERENT checkpoint "
            "contracts (%s vs %s) — not comparable"
            % (p["checkpoint_contract_sha256"][:16],
               r["checkpoint_contract_sha256"][:16]))
    # task §七: both arms must have run the IDENTICAL frozen start schedule.
    require(p["metrics"]["valid_starts"] == r["metrics"]["valid_starts"]
            == dict(FROZEN_COUNTS),
            "FAIL CLOSED: arms ran different start schedules (%s vs %s)"
            % (p["metrics"]["valid_starts"], r["metrics"]["valid_starts"]))
    selection = select_provisional(p["metrics"], r["metrics"])
    doc = {
        "schema": SCHEMA,
        "run_class": RUN_CLASS,
        "checkpoint_contract_sha256": p["checkpoint_contract_sha256"],
        "arms_verified": {
            "persistent": {
                "dir": _repo_relative_or_raw(persistent_dir),
                "sha256sums_verified": True,
                "certificate_binding_verified": True,
            },
            "reset128": {
                "dir": _repo_relative_or_raw(reset128_dir),
                "sha256sums_verified": True,
                "certificate_binding_verified": True,
            },
        },
        "RMT16_CARRY_MODE_WINNER": selection["RMT16_CARRY_MODE_WINNER"],
        "CANDIDATE_SCOPE": CANDIDATE_SCOPE,
        "MECHANISM_QUESTION": MECHANISM_QUESTION,
        "OVERALL_STRONG_STUDENT_SELECTION_AUTHORIZED": False,
        "STRONG_STUDENT_V1": "NOT_SELECTED",
        "EXISTING_STUDENT_BAKEOFF_REQUIRED": True,
        "SINGLE_TRAINING_SEED": True,
        "SCIENTIFIC_SUPERIORITY_CLAIM": False,
        "REQUIRES_MULTI_SEED_CONFIRMATION": True,
        "selection": selection,
        "metrics": {"persistent": p["metrics"], "reset128": r["metrics"]},
        "forbidden_claims_absent": True,
    }
    assert_no_forbidden_claims(doc)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    return doc


# ---------------------------------------------------------------------------
# Self-test (pure; any host).
# ---------------------------------------------------------------------------
def _result_doc(full_n, front_n, front_mean, back_n, arm="persistent"):
    return {
        "schema": "mechanism_UED.tier3_evaluation_result/v1",
        "run_class": RUN_CLASS,
        "checkpoint_contract": {"arm": arm},
        "results": {
            FULL: {"metrics": {"primary": {"metric": "DEFEAT_KOBOLD_SR",
                                           "successes": full_n, "valid_starts": 64,
                                           "value": full_n / 64.0}}},
            FRONT: {"metrics": {"primary": {
                        "metric": "P_FRONT_FLOOR_TRANSITION_REACHED_GIVEN_VALID_START",
                        "successes": front_n, "valid_starts": 8,
                        "value": front_n / 8.0},
                        "dense": {"metric": "GRAPH_DISTANCE_PROGRESS",
                                  "value": front_mean}}},
            BACK: {"metrics": {"primary": {
                        "metric": "P_DEFEAT_KOBOLD_GIVEN_VALID_BACK_START",
                        "successes": back_n, "valid_starts": 8,
                        "value": back_n / 8.0}}},
        },
    }


def self_test() -> int:
    problems = []

    def check(name, cond):
        if not cond:
            problems.append(name)

    pm = extract_arm_metrics(_result_doc(5, 4, 0.6, 2, "persistent"))
    rm = extract_arm_metrics(_result_doc(3, 4, 0.6, 2, "reset128"))
    check("extract_counts",
          pm["full_success_count"] == 5 and pm["front_transition_count"] == 4
          and abs(pm["front_mean_progress"] - 0.6) < 1e-12
          and pm["back_defeat_count"] == 2 and pm["valid_starts"]
          == {FULL: 64, FRONT: 8, BACK: 8})

    # level 1: higher FULL success count decides.
    s1 = select_provisional(pm, rm)
    check("level1_persistent",
          s1["RMT16_CARRY_MODE_WINNER"] == "PERSISTENT"
          and s1["decided_at_level"] == 1)
    s1r = select_provisional(rm, pm)
    check("level1_symmetric",
          s1r["RMT16_CARRY_MODE_WINNER"] == "RESET128"
          and s1r["decided_at_level"] == 1)

    # level 2: FULL tied → FRONT transition count.
    pm2 = extract_arm_metrics(_result_doc(5, 5, 0.6, 2))
    rm2 = extract_arm_metrics(_result_doc(5, 3, 0.6, 2, "reset128"))
    s2 = select_provisional(pm2, rm2)
    check("level2_front_transition",
          s2["RMT16_CARRY_MODE_WINNER"] == "PERSISTENT"
          and s2["decided_at_level"] == 2)

    # level 3: FULL + FRONT counts tied → FRONT mean progress.
    pm3 = extract_arm_metrics(_result_doc(5, 4, 0.7, 2))
    rm3 = extract_arm_metrics(_result_doc(5, 4, 0.5, 2, "reset128"))
    s3 = select_provisional(pm3, rm3)
    check("level3_front_mean",
          s3["RMT16_CARRY_MODE_WINNER"] == "PERSISTENT"
          and s3["decided_at_level"] == 3)

    # level 4: mean progress tied within tolerance → BACK defeat count.
    pm4 = extract_arm_metrics(_result_doc(5, 4, 0.5, 3))
    rm4 = extract_arm_metrics(_result_doc(5, 4, 0.5 + 1e-13, 1, "reset128"))
    s4 = select_provisional(pm4, rm4)
    check("level4_back_defeat",
          s4["RMT16_CARRY_MODE_WINNER"] == "PERSISTENT"
          and s4["decided_at_level"] == 4)

    # level 5: everything tied → INCONCLUSIVE (never an arbitrary winner).
    pm5 = extract_arm_metrics(_result_doc(5, 4, 0.5, 2))
    rm5 = extract_arm_metrics(_result_doc(5, 4, 0.5, 2, "reset128"))
    s5 = select_provisional(pm5, rm5)
    check("level5_inconclusive",
          s5["RMT16_CARRY_MODE_WINNER"] == "INCONCLUSIVE"
          and s5["decided_at_level"] is None)

    # determinism: identical inputs → identical winner + trace.
    check("deterministic",
          select_provisional(pm, rm) == s1)
    # 总控范围修正: the fixed scope/boundary fields are ALWAYS present and exact.
    check("frozen_scope_fields",
          all(all(s[k] == v for k, v in FIXED_SCOPE_FIELDS.items())
              for s in (s1, s2, s3, s4, s5)))
    check("only_carry_mode_winner_field",
          all("RMT16_CARRY_MODE_WINNER" in s
              and "PROVISIONAL_STRONG_STUDENT_RECOMMENDATION" not in s
              for s in (s1, s2, s3, s4, s5)))
    # the frozen rule is embedded and recomputable.
    check("rule_embedded",
          s1["rule"]["rule_version"] == "tier3_rmt16_carry_mode_comparison_rule/v1"
          and s1["rule"]["output_field"] == "RMT16_CARRY_MODE_WINNER"
          and len(s1["rule"]["criteria"]) == 5)
    # forbidden claims never appear anywhere in the output — including the
    # 总控范围修正 vocabulary (strong-student / bakeoff selection is out of scope).
    for bad_claim in ("PERSISTENT_PROVEN_BETTER",
                      "PROVISIONAL_STRONG_STUDENT_RECOMMENDATION",
                      "STRONG_STUDENT_V1=PERSISTENT",
                      "STRONG_STUDENT_V1=RESET128",
                      "BEST_OVERALL_STUDENT",
                      "ALL_STUDENT_BAKEOFF_WINNER"):
        bad = dict(s1)
        bad["note"] = bad_claim
        try:
            assert_no_forbidden_claims(bad)
            check("forbidden_claim_rejected_%s" % bad_claim.split("=")[0], False)
        except FailClosed:
            check("forbidden_claim_rejected_%s" % bad_claim.split("=")[0], True)

    # extraction gates: wrong valid_starts / wrong run class / missing dense value.
    wrong_n = _result_doc(5, 4, 0.5, 2)
    wrong_n["results"][FULL]["metrics"]["primary"]["valid_starts"] = 63
    try:
        extract_arm_metrics(wrong_n)
        check("wrong_valid_starts_rejected", False)
    except FailClosed:
        check("wrong_valid_starts_rejected", True)
    wrong_rc = _result_doc(5, 4, 0.5, 2)
    wrong_rc["run_class"] = "INTERFACE_SMOKE"
    try:
        extract_arm_metrics(wrong_rc)
        check("wrong_run_class_rejected", False)
    except FailClosed:
        check("wrong_run_class_rejected", True)
    no_dense = _result_doc(5, 4, 0.5, 2)
    no_dense["results"][FRONT]["metrics"]["dense"]["value"] = None
    try:
        extract_arm_metrics(no_dense)
        check("missing_dense_value_rejected", False)
    except FailClosed:
        check("missing_dense_value_rejected", True)

    if problems:
        print("TIER3_PROVISIONAL_SELECTION_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_RMT16_CARRY_MODE_COMPARISON_SELF_TEST_PASS (frozen rule v1; "
          "recomputable; scope=RMT16_ORIGINAL_VTRACE_PAIR_ONLY; no student selection)")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return self_test()

    def _opt(flag, default=None):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                return argv[i + 1]
        return default

    persistent_dir = _opt("--persistent")
    reset128_dir = _opt("--reset128")
    out = _opt("--out")
    if not persistent_dir or not reset128_dir or not out:
        print("usage: tier3_provisional_selection.py --self-test\n"
              "       tier3_provisional_selection.py --persistent <DIR> "
              "--reset128 <DIR> --out <cross_arm_comparison.json> "
              "[--checkpoint-contract <PATH>]")
        return 3
    try:
        doc = compare_dirs(persistent_dir, reset128_dir, out,
                           contract_path=_opt("--checkpoint-contract"))
    except (FailClosed, certmod.FailClosed, contractmod.FailClosed,
            runner.FailClosed) as exc:
        print(str(exc))
        return 1
    print("TIER3_RMT16_CARRY_MODE_COMPARISON_DONE "
          "(RMT16_CARRY_MODE_WINNER=%s; CANDIDATE_SCOPE=%s; "
          "OVERALL_STRONG_STUDENT_SELECTION_AUTHORIZED=false; "
          "STRONG_STUDENT_V1=NOT_SELECTED; EXISTING_STUDENT_BAKEOFF_REQUIRED=true; "
          "SCIENTIFIC_SUPERIORITY_CLAIM=false; "
          "REQUIRES_MULTI_SEED_CONFIRMATION=true; out=%s)"
          % (doc["RMT16_CARRY_MODE_WINNER"], doc["CANDIDATE_SCOPE"], out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
