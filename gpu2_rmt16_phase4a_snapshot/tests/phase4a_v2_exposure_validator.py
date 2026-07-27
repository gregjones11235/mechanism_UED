#!/usr/bin/env python3
"""Phase4A-v2.1 (§五) — MATCHED_REPLAY_EXPOSURE two-arm certificate validator.

Compares the per-arm exposure certificates emitted by train_rmt16_p2replay.py
(summary["exposure_certificate"]) and adjudicates the three-level replay claim split:

    Level 1  PROTOCOL_MATCH        : FULL canonical protocol identity (Phase4A-v2.2 §二/§八):
                                     every REQUIRED_PROTOCOL_FIELDS present incl. learner and
                                     rng_rule, identical key sets, identical canonical dicts and
                                     identical protocol SHA256 across both arms. The old partial
                                     whitelist (missing learner/rng_rule) is gone.
    Level 2  EXPOSURE_COUNT_MATCH  : the six EXPOSURE_MATCH_FIELDS (replay_attempt_mask,
                                     replay_update_outer_updates, replay_update_count,
                                     replay_sequences_consumed, replay_batch_sizes,
                                     replay_sequence_lengths) are EQUAL across both arms
                                     -> MATCHED_REPLAY_EXPOSURE=PASS, else FAIL.
    Level 3  CONTENT_MATCH         : always NOT_APPLICABLE_ENDOGENOUS_BUFFERS (never PASS): the
                                     buffers are endogenous per-arm trajectory buffers with no
                                     shared trajectory identity.

FAIL-CLOSED GATES (raise -> exit 1, never a silent pass):
  * MATCHED_REPLAY_CERTIFICATE_REQUIRED          — an arm summary lacks (some of) the 14
                                                   EXPOSURE_CERTIFICATE_FIELDS, yet a PASS would
                                                   be claimed.
  * ENDOGENOUS_REPLAY_CONTENT_CANNOT_BE_CLAIMED_MATCHED — any attempt to assert content match.

NOTES:
  * sample_ids_by_outer_update / start_offsets_by_outer_update are per-arm INTERNAL provenance
    only — they are present in the certificate but NEVER compared across arms.
  * If one arm is NOT_READY (attempted zero replay updates) while the other arm DID replay, the
    comparison is FAIL but the pair remains a valid ENDOGENOUS_REPLAY_SCREENING=READY_AFTER_SMOKE
    record — the runs are NOT discarded and NOT silently rerun (CC2 §五.1).

Usage:
    python phase4a_v2_exposure_validator.py --self-test
    python phase4a_v2_exposure_validator.py \\
        --persistent runs/.../RMT16-Persistent-..._train_summary.json \\
        --reset128   runs/.../RMT16-Reset128-..._train_summary.json \\
        [--out reports/rmt16_phase4a_v2_1_exposure_report.json]

Pure Python (no JAX / no numpy) so it runs locally and on server CPU alike.
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SNAPSHOT = os.path.dirname(_HERE)
_EXP = os.path.join(_SNAPSHOT, "runtime", "experiment_src")
_FRZ = os.path.join(_SNAPSHOT, "runtime", "frozen_modules")
for _p in (_EXP, _FRZ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import phase4a_v2_contract as CONTRACT  # noqa: E402


def extract_certificate(arm_summary, arm_name="arm"):
    """Pull the exposure certificate out of an arm train-summary dict and attach its
    protocol_definition (from the §四 replay_labels block). Fail-closed if anything is missing.

    Returns a flat dict containing all EXPOSURE_CERTIFICATE_FIELDS + protocol_definition, suitable
    for CONTRACT.compare_exposure / check_certificate_complete."""
    if not isinstance(arm_summary, dict):
        raise ValueError(f"MATCHED_REPLAY_CERTIFICATE_REQUIRED: {arm_name} summary is not a dict")
    cert = arm_summary.get("exposure_certificate")
    if not isinstance(cert, dict):
        raise ValueError(
            f"MATCHED_REPLAY_CERTIFICATE_REQUIRED: {arm_name} summary has no "
            "'exposure_certificate' block; cannot assess MATCHED_REPLAY_EXPOSURE.")
    cert = dict(cert)
    missing = CONTRACT.check_certificate_complete(cert, arm_name)
    if missing:
        raise ValueError(
            f"MATCHED_REPLAY_CERTIFICATE_REQUIRED: {arm_name} exposure certificate missing "
            f"fields {missing}; cannot assess MATCHED_REPLAY_EXPOSURE.")
    labels = (arm_summary.get("phase4a_v2") or {}).get("replay_labels") or {}
    protocol = labels.get("protocol_definition")
    if not isinstance(protocol, dict):
        raise ValueError(
            f"MATCHED_REPLAY_CERTIFICATE_REQUIRED: {arm_name} summary has no "
            "phase4a_v2.replay_labels.protocol_definition block.")
    cert["protocol_definition"] = dict(protocol)
    return cert


def validate_two_arm(persistent_summary, reset128_summary):
    """Full three-level two-arm adjudication. Returns the report dict.

    Raises ValueError on fail-closed gate breaches (missing certificate / content claim)."""
    cert_p = extract_certificate(persistent_summary, "persistent")
    cert_r = extract_certificate(reset128_summary, "reset128")
    result = CONTRACT.compare_exposure(cert_p, cert_r)   # raises if a certificate is incomplete

    # §五.1: a NOT_READY-vs-replayed mismatch is FAIL but still valid screening material.
    p_updates = int(cert_p.get("replay_update_count", 0))
    r_updates = int(cert_r.get("replay_update_count", 0))
    one_arm_not_ready = bool((p_updates == 0) != (r_updates == 0))

    report = dict(
        validator="phase4a_v2_exposure_validator",
        spec="RMT16_PHASE4A_V2_1 §五 + V2_2 §二/§八 full canonical protocol identity",
        arms=dict(persistent="certificate_present", reset128="certificate_present"),
        # Level 1 (Phase4A-v2.2 §二/§八: FULL canonical protocol identity)
        PROTOCOL_MATCH=result["PROTOCOL_MATCH"],
        PROTOCOL_MISSING_FIELDS_ARM_A=result["PROTOCOL_MISSING_FIELDS_ARM_A"],
        PROTOCOL_MISSING_FIELDS_ARM_B=result["PROTOCOL_MISSING_FIELDS_ARM_B"],
        PROTOCOL_KEYSET_MISMATCH=result["PROTOCOL_KEYSET_MISMATCH"],
        PROTOCOL_DIFFERING_FIELDS=result["PROTOCOL_DIFFERING_FIELDS"],
        PROTOCOL_DEFINITION_SHA256_ARM_A=result["PROTOCOL_DEFINITION_SHA256_ARM_A"],
        PROTOCOL_DEFINITION_SHA256_ARM_B=result["PROTOCOL_DEFINITION_SHA256_ARM_B"],
        # Level 2
        EXPOSURE_COUNT_MATCH=result["EXPOSURE_COUNT_MATCH"],
        EXPOSURE_DIFFERING_FIELDS=result["EXPOSURE_DIFFERING_FIELDS"],
        MATCHED_REPLAY_EXPOSURE=result["MATCHED_REPLAY_EXPOSURE"],
        one_arm_not_ready_vs_replayed=one_arm_not_ready,
        # Level 3 (endogenous: never PASS)
        CONTENT_MATCH=result["CONTENT_MATCH"],
        MATCHED_REPLAY_CONTENT=result["MATCHED_REPLAY_CONTENT"],
        ENDOGENOUS_REPLAY_SCREENING=result["ENDOGENOUS_REPLAY_SCREENING"],
        # per-arm exposure echo (counts only; sample_ids/start_offsets are internal, NOT echoed
        # cross-arm because they carry no shared identity)
        persistent_exposure=dict(
            outer_update_count=cert_p["outer_update_count"],
            replay_update_count=p_updates,
            accepted_replay_policy_update_count=cert_p["accepted_replay_policy_update_count"],
            kl_rejected_replay_update_count=cert_p["kl_rejected_replay_update_count"],
            replay_sequences_consumed=cert_p["replay_sequences_consumed"],
            replay_attempt_outer_updates=cert_p["replay_attempt_outer_updates"],
            replay_not_ready_outer_updates=cert_p["replay_not_ready_outer_updates"],
            replay_update_outer_updates=cert_p["replay_update_outer_updates"],
            replay_batch_sizes=cert_p["replay_batch_sizes"],
            replay_sequence_lengths=cert_p["replay_sequence_lengths"]),
        reset128_exposure=dict(
            outer_update_count=cert_r["outer_update_count"],
            replay_update_count=r_updates,
            accepted_replay_policy_update_count=cert_r["accepted_replay_policy_update_count"],
            kl_rejected_replay_update_count=cert_r["kl_rejected_replay_update_count"],
            replay_sequences_consumed=cert_r["replay_sequences_consumed"],
            replay_attempt_outer_updates=cert_r["replay_attempt_outer_updates"],
            replay_not_ready_outer_updates=cert_r["replay_not_ready_outer_updates"],
            replay_update_outer_updates=cert_r["replay_update_outer_updates"],
            replay_batch_sizes=cert_r["replay_batch_sizes"],
            replay_sequence_lengths=cert_r["replay_sequence_lengths"]),
    )
    # Fail-closed: PASS may never be reported without both complete certificates (already enforced
    # by extract_certificate/compare_exposure) — double-guard the claim itself.
    if report["MATCHED_REPLAY_EXPOSURE"] == "PASS":
        CONTRACT.assert_matched_exposure_pass_allowed(cert_p, cert_r)
    # Fail-closed (§五.2): neither INPUT certificate may claim MATCHED_REPLAY_CONTENT=PASS —
    # endogenous buffers have no shared trajectory identity. (This validator's own report never
    # emits CONTENT_MATCH=PASS: compare_exposure fixes NOT_APPLICABLE_ENDOGENOUS_BUFFERS.)
    for _name, _s in (("persistent", persistent_summary), ("reset128", reset128_summary)):
        _claimed = ((_s.get("phase4a_v2") or {}).get("replay_labels") or {}).get(
            "MATCHED_REPLAY_CONTENT")
        if _claimed == "PASS":
            raise ValueError(
                "ENDOGENOUS_REPLAY_CONTENT_CANNOT_BE_CLAIMED_MATCHED: "
                f"{_name} certificate claims MATCHED_REPLAY_CONTENT=PASS; forbidden for "
                "endogenous per-arm buffers.")
    return report


# ----------------------------------------------------------------------------
# §五 self-test (synthetic certificates; no training, no JAX)
# ----------------------------------------------------------------------------

def _synthetic_summary(arm, *, replay_updates, consumed, batch_sizes, seq_lens,
                       attempt_mask, not_ready_updates, drop_field=None,
                       seq_length=129, batch_size=4,
                       learner=None, rng_rule=None, drop_protocol_field=None,
                       extra_protocol_field=None):
    n = len(attempt_mask)
    attempt_outer = [i for i, a in enumerate(attempt_mask) if a]
    update_outer = [i for i, a in enumerate(attempt_mask) if a][:replay_updates]
    cert = dict(
        outer_update_count=n,
        replay_attempt_mask=list(attempt_mask),
        replay_attempt_outer_updates=attempt_outer,
        replay_not_ready_outer_updates=list(not_ready_updates),
        replay_update_outer_updates=update_outer,
        replay_update_count=int(replay_updates),
        accepted_replay_policy_update_count=int(replay_updates),
        kl_rejected_replay_update_count=0,
        replay_sequences_consumed=int(consumed),
        replay_batch_sizes=list(batch_sizes),
        replay_sequence_lengths=[list(x) for x in seq_lens],
        eligible_count_by_outer_update=[4] * n,
        sample_ids_by_outer_update=[[0, 1, 2, 3]] * replay_updates,
        start_offsets_by_outer_update=[[0, 0, 0, 0]] * replay_updates)
    if drop_field is not None:
        cert.pop(drop_field, None)
    labels = CONTRACT.replay_protocol_labels(
        "original_vtrace", seq_length, batch_size,
        **({"learner": learner} if learner is not None else {}),
        **({"rng_rule": rng_rule} if rng_rule is not None else {}))
    protocol = labels["protocol_definition"]
    if drop_protocol_field is not None:
        protocol.pop(drop_protocol_field, None)  # simulate an incomplete protocol definition
    if extra_protocol_field is not None:
        protocol[extra_protocol_field[0]] = extra_protocol_field[1]  # unknown extra field
    return dict(
        arm=arm,
        phase4a_v2=dict(replay_labels=labels),
        exposure_certificate=cert)


def self_test():
    results = []

    def check(name, ok, detail=""):
        results.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""),
              flush=True)

    print("phase4a_v2_exposure_validator --self-test", flush=True)

    # (1) identical exposure on both arms -> PASS at all claimable levels
    sa = _synthetic_summary("persistent", replay_updates=3, consumed=12,
                            batch_sizes=[4, 4, 4], seq_lens=[[129] * 4] * 3,
                            attempt_mask=[False, True, True, True, True],
                            not_ready_updates=[0])
    sb = _synthetic_summary("reset128", replay_updates=3, consumed=12,
                            batch_sizes=[4, 4, 4], seq_lens=[[129] * 4] * 3,
                            attempt_mask=[False, True, True, True, True],
                            not_ready_updates=[0])
    rep = validate_two_arm(sa, sb)
    check("identical_exposure -> MATCHED_REPLAY_EXPOSURE=PASS",
          rep["MATCHED_REPLAY_EXPOSURE"] == "PASS" and rep["PROTOCOL_MATCH"] == "PASS"
          and rep["EXPOSURE_COUNT_MATCH"] == "PASS", str(rep["EXPOSURE_DIFFERING_FIELDS"]))
    check("identical_exposure -> CONTENT never PASS",
          rep["CONTENT_MATCH"] == "NOT_APPLICABLE_ENDOGENOUS_BUFFERS"
          and rep["MATCHED_REPLAY_CONTENT"] == "NOT_CLAIMED")

    # (2) differing replay_update_count -> FAIL, differing field reported
    sb2 = _synthetic_summary("reset128", replay_updates=2, consumed=8,
                             batch_sizes=[4, 4], seq_lens=[[129] * 4] * 2,
                             attempt_mask=[False, True, True, True, True],
                             not_ready_updates=[0])
    rep2 = validate_two_arm(sa, sb2)
    check("count_mismatch -> MATCHED_REPLAY_EXPOSURE=FAIL",
          rep2["MATCHED_REPLAY_EXPOSURE"] == "FAIL"
          and "replay_update_count" in rep2["EXPOSURE_DIFFERING_FIELDS"],
          str(rep2["EXPOSURE_DIFFERING_FIELDS"]))
    check("count_mismatch -> screening still READY_AFTER_SMOKE (not discarded)",
          rep2["ENDOGENOUS_REPLAY_SCREENING"] == "READY_AFTER_SMOKE")

    # (3) protocol mismatch -> PROTOCOL_MATCH=FAIL even with equal counts
    sb3 = _synthetic_summary("reset128", replay_updates=3, consumed=12,
                             batch_sizes=[4, 4, 4], seq_lens=[[129] * 4] * 3,
                             attempt_mask=[False, True, True, True, True],
                             not_ready_updates=[0], seq_length=104)
    rep3 = validate_two_arm(sa, sb3)
    check("protocol_mismatch -> PROTOCOL_MATCH=FAIL & exposure FAIL",
          rep3["PROTOCOL_MATCH"] == "FAIL"
          and "sequence_length" in rep3["PROTOCOL_DIFFERING_FIELDS"]
          and rep3["MATCHED_REPLAY_EXPOSURE"] == "FAIL")

    # (4) one arm NOT_READY (zero replay updates) while the other replays -> FAIL + flag
    sb4 = _synthetic_summary("reset128", replay_updates=0, consumed=0,
                             batch_sizes=[], seq_lens=[],
                             attempt_mask=[True, True, True, True, True],
                             not_ready_updates=[0, 1, 2, 3, 4])
    rep4 = validate_two_arm(sa, sb4)
    check("not_ready_vs_replayed -> FAIL with flag, screening retained",
          rep4["MATCHED_REPLAY_EXPOSURE"] == "FAIL"
          and rep4["one_arm_not_ready_vs_replayed"] is True
          and rep4["ENDOGENOUS_REPLAY_SCREENING"] == "READY_AFTER_SMOKE")

    # (5) missing certificate field -> fail-closed MATCHED_REPLAY_CERTIFICATE_REQUIRED
    sb5 = _synthetic_summary("reset128", replay_updates=3, consumed=12,
                             batch_sizes=[4, 4, 4], seq_lens=[[129] * 4] * 3,
                             attempt_mask=[False, True, True, True, True],
                             not_ready_updates=[0], drop_field="replay_sequences_consumed")
    try:
        validate_two_arm(sa, sb5)
        check("missing_field -> MATCHED_REPLAY_CERTIFICATE_REQUIRED raised", False, "no raise")
    except ValueError as e:
        check("missing_field -> MATCHED_REPLAY_CERTIFICATE_REQUIRED raised",
              "MATCHED_REPLAY_CERTIFICATE_REQUIRED" in str(e))

    # (5b) no exposure_certificate block at all -> fail-closed
    try:
        validate_two_arm(sa, dict(arm="reset128"))
        check("missing_certificate_block -> raised", False, "no raise")
    except ValueError as e:
        check("missing_certificate_block -> raised",
              "MATCHED_REPLAY_CERTIFICATE_REQUIRED" in str(e))

    # (6) content-match claim -> fail-closed ENDOGENOUS_REPLAY_CONTENT_CANNOT_BE_CLAIMED_MATCHED
    try:
        CONTRACT.assert_content_match_not_claimed(buffer_kind="endogenous")
        check("content_claim -> ENDOGENOUS_REPLAY_CONTENT_CANNOT_BE_CLAIMED_MATCHED raised",
              False, "no raise")
    except ValueError as e:
        check("content_claim -> ENDOGENOUS_REPLAY_CONTENT_CANNOT_BE_CLAIMED_MATCHED raised",
              "ENDOGENOUS_REPLAY_CONTENT_CANNOT_BE_CLAIMED_MATCHED" in str(e))

    # (7) PASS-claim guard: assert_matched_exposure_pass_allowed refuses a FAIL comparison
    try:
        CONTRACT.assert_matched_exposure_pass_allowed(
            extract_certificate(sa, "persistent"), extract_certificate(sb2, "reset128"))
        check("pass_claim_guard -> raised on FAIL comparison", False, "no raise")
    except ValueError as e:
        check("pass_claim_guard -> raised on FAIL comparison",
              "MATCHED_REPLAY_CERTIFICATE_REQUIRED" in str(e))

    # (8) an input certificate claiming MATCHED_REPLAY_CONTENT=PASS -> fail-closed
    sb8 = _synthetic_summary("reset128", replay_updates=3, consumed=12,
                             batch_sizes=[4, 4, 4], seq_lens=[[129] * 4] * 3,
                             attempt_mask=[False, True, True, True, True],
                             not_ready_updates=[0])
    sb8["phase4a_v2"]["replay_labels"]["MATCHED_REPLAY_CONTENT"] = "PASS"
    try:
        validate_two_arm(sa, sb8)
        check("input_content_claim=PASS -> raised", False, "no raise")
    except ValueError as e:
        check("input_content_claim=PASS -> raised",
              "ENDOGENOUS_REPLAY_CONTENT_CANNOT_BE_CLAIMED_MATCHED" in str(e))

    # --- Phase4A-v2.2 §二/§八: FULL canonical protocol identity ------------------
    # (9) identical protocol -> protocol SHA256 emitted and equal on both arms
    check("protocol_sha_emitted_and_equal",
          rep["PROTOCOL_DEFINITION_SHA256_ARM_A"] is not None
          and rep["PROTOCOL_DEFINITION_SHA256_ARM_A"] == rep["PROTOCOL_DEFINITION_SHA256_ARM_B"]
          and rep["PROTOCOL_MISSING_FIELDS_ARM_A"] == []
          and rep["PROTOCOL_MISSING_FIELDS_ARM_B"] == [])

    # (10) different LEARNER, identical exposure -> PROTOCOL_MATCH=FAIL (old whitelist missed it)
    sb10 = _synthetic_summary("reset128", replay_updates=3, consumed=12,
                              batch_sizes=[4, 4, 4], seq_lens=[[129] * 4] * 3,
                              attempt_mask=[False, True, True, True, True],
                              not_ready_updates=[0],
                              learner="full_p2_legacy_update_rmt")
    rep10 = validate_two_arm(sa, sb10)
    check("different_learner -> PROTOCOL_MATCH=FAIL & exposure FAIL",
          rep10["PROTOCOL_MATCH"] == "FAIL"
          and "learner" in rep10["PROTOCOL_DIFFERING_FIELDS"]
          and rep10["MATCHED_REPLAY_EXPOSURE"] == "FAIL",
          str(rep10["PROTOCOL_DIFFERING_FIELDS"]))

    # (11) different RNG RULE, identical exposure -> PROTOCOL_MATCH=FAIL
    sb11 = _synthetic_summary("reset128", replay_updates=3, consumed=12,
                              batch_sizes=[4, 4, 4], seq_lens=[[129] * 4] * 3,
                              attempt_mask=[False, True, True, True, True],
                              not_ready_updates=[0],
                              rng_rule="np.random.default_rng(seed)")
    rep11 = validate_two_arm(sa, sb11)
    check("different_rng_rule -> PROTOCOL_MATCH=FAIL & exposure FAIL",
          rep11["PROTOCOL_MATCH"] == "FAIL"
          and "rng_rule" in rep11["PROTOCOL_DIFFERING_FIELDS"]
          and rep11["MATCHED_REPLAY_EXPOSURE"] == "FAIL",
          str(rep11["PROTOCOL_DIFFERING_FIELDS"]))

    # (12) missing LEARNER field -> fail closed PROTOCOL_IDENTITY_INCOMPLETE
    sb12 = _synthetic_summary("reset128", replay_updates=3, consumed=12,
                              batch_sizes=[4, 4, 4], seq_lens=[[129] * 4] * 3,
                              attempt_mask=[False, True, True, True, True],
                              not_ready_updates=[0], drop_protocol_field="learner")
    try:
        validate_two_arm(sa, sb12)
        check("missing_learner -> PROTOCOL_IDENTITY_INCOMPLETE raised", False, "no raise")
    except ValueError as e:
        check("missing_learner -> PROTOCOL_IDENTITY_INCOMPLETE raised",
              "PROTOCOL_IDENTITY_INCOMPLETE" in str(e) and "learner" in str(e))

    # (13) missing RNG_RULE field -> fail closed
    sb13 = _synthetic_summary("reset128", replay_updates=3, consumed=12,
                              batch_sizes=[4, 4, 4], seq_lens=[[129] * 4] * 3,
                              attempt_mask=[False, True, True, True, True],
                              not_ready_updates=[0], drop_protocol_field="rng_rule")
    try:
        validate_two_arm(sa, sb13)
        check("missing_rng_rule -> PROTOCOL_IDENTITY_INCOMPLETE raised", False, "no raise")
    except ValueError as e:
        check("missing_rng_rule -> PROTOCOL_IDENTITY_INCOMPLETE raised",
              "PROTOCOL_IDENTITY_INCOMPLETE" in str(e) and "rng_rule" in str(e))

    # (14) extra UNKNOWN protocol field on one arm only -> FAIL via keyset mismatch
    sb14 = _synthetic_summary("reset128", replay_updates=3, consumed=12,
                              batch_sizes=[4, 4, 4], seq_lens=[[129] * 4] * 3,
                              attempt_mask=[False, True, True, True, True],
                              not_ready_updates=[0],
                              extra_protocol_field=("unregistered_field", 1))
    rep14 = validate_two_arm(sa, sb14)
    check("extra_protocol_field_one_arm -> PROTOCOL_MATCH=FAIL (keyset mismatch)",
          rep14["PROTOCOL_MATCH"] == "FAIL"
          and "unregistered_field" in rep14["PROTOCOL_KEYSET_MISMATCH"]
          and rep14["MATCHED_REPLAY_EXPOSURE"] == "FAIL",
          str(rep14["PROTOCOL_KEYSET_MISMATCH"]))

    # (15) same keys/values, different INSERTION ORDER -> PASS (canonical comparison)
    sb15 = _synthetic_summary("reset128", replay_updates=3, consumed=12,
                              batch_sizes=[4, 4, 4], seq_lens=[[129] * 4] * 3,
                              attempt_mask=[False, True, True, True, True],
                              not_ready_updates=[0])
    proto15 = sb15["phase4a_v2"]["replay_labels"]["protocol_definition"]
    sb15["phase4a_v2"]["replay_labels"]["protocol_definition"] = dict(
        reversed(list(proto15.items())))
    rep15 = validate_two_arm(sa, sb15)
    check("same_protocol_different_key_order -> PASS",
          rep15["PROTOCOL_MATCH"] == "PASS" and rep15["MATCHED_REPLAY_EXPOSURE"] == "PASS"
          and rep15["PROTOCOL_DEFINITION_SHA256_ARM_A"]
          == rep15["PROTOCOL_DEFINITION_SHA256_ARM_B"])

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"SELF_TEST_SUMMARY total={len(results)} pass={len(results) - n_fail} fail={n_fail}",
          flush=True)
    return 1 if n_fail else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--persistent", help="path to persistent arm *_train_summary.json")
    ap.add_argument("--reset128", help="path to reset128 arm *_train_summary.json")
    ap.add_argument("--out", help="optional path to write the comparison report JSON")
    ap.add_argument("--self-test", action="store_true",
                    help="run synthetic certificate self-tests (no files, no training)")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if not (args.persistent and args.reset128):
        ap.error("--persistent and --reset128 are required unless --self-test is given")

    with open(args.persistent, encoding="utf-8") as f:
        persistent_summary = json.load(f)
    with open(args.reset128, encoding="utf-8") as f:
        reset128_summary = json.load(f)

    try:
        report = validate_two_arm(persistent_summary, reset128_summary)
    except ValueError as e:
        print(f"EXPOSURE_VALIDATOR_BLOCKED={e}", flush=True)
        return 1

    print(json.dumps({k: report[k] for k in (
        "PROTOCOL_MATCH", "PROTOCOL_MISSING_FIELDS_ARM_A", "PROTOCOL_MISSING_FIELDS_ARM_B",
        "PROTOCOL_KEYSET_MISMATCH", "PROTOCOL_DIFFERING_FIELDS",
        "PROTOCOL_DEFINITION_SHA256_ARM_A", "PROTOCOL_DEFINITION_SHA256_ARM_B",
        "EXPOSURE_COUNT_MATCH", "EXPOSURE_DIFFERING_FIELDS", "MATCHED_REPLAY_EXPOSURE",
        "CONTENT_MATCH", "MATCHED_REPLAY_CONTENT", "ENDOGENOUS_REPLAY_SCREENING",
        "one_arm_not_ready_vs_replayed")}, indent=2), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"EXPOSURE_REPORT_WRITTEN={args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
