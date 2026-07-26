"""Phase4A-v2.1 replay contract (CC2 directive §三/§四/§五). PURE Python — no JAX/numpy.

Centralizes three things that were previously implicit, unused, or mis-stated, so that config,
runtime manifest, summary, reports and gates all speak with ONE vocabulary:

  (§三) the policy-lag GATE identity of each replay mode, plus a fail-closed config validator;
  (§四) the four-way split of the previously-conflated "matched replay protocol" concept;
  (§五) the per-arm exposure-certificate field spec and the fail-closed claim gates.

It is imported by the launcher (manifest + summary), by tests/test_phase4a_v2_gates.py and by
tests/phase4a_v2_exposure_validator.py.
"""

# ===========================================================================
# §三 — policy-lag gate identity
# ===========================================================================
# original_vtrace performs V-trace importance-sampling off-policy correction using each
# transition's STORED behavior log_prob, clipped by rho_bar/c_bar. It deliberately applies NO
# additional hard trajectory-level policy-lag gate: filtering per-transition with the imprecise
# EPISODE scalar version would be pseudo-precise (the episode may span many policy versions).
# full_p2_legacy keeps its legacy lag gate (legacy-scoped only, default-forbidden).

POLICY_LAG_MODE_ORIGINAL = "not_applicable_original_vtrace"
POLICY_LAG_MODE_OFF = "off_no_replay"
POLICY_LAG_MODE_LEGACY = "legacy_full_p2"

OFF_POLICY_CORRECTION_ORIGINAL = "vtrace_importance_sampling"


def policy_lag_runtime_manifest(replay_mode, legacy_max_policy_lag=16,
                                rho_bar=1.0, c_bar=1.0):
    """The policy-lag identity recorded in checkpoint/summary/run manifest (CC2 §三.2)."""
    if replay_mode == "original_vtrace":
        return dict(
            policy_lag_gate_active=False,
            policy_lag_gate_mode=POLICY_LAG_MODE_ORIGINAL,
            max_policy_lag=None,
            off_policy_correction=OFF_POLICY_CORRECTION_ORIGINAL,
            rho_bar=float(rho_bar), c_bar=float(c_bar))
    if replay_mode == "off":
        return dict(
            policy_lag_gate_active=False,
            policy_lag_gate_mode=POLICY_LAG_MODE_OFF,
            max_policy_lag=None,
            off_policy_correction="none",
            rho_bar=None, c_bar=None)
    if replay_mode == "full_p2_legacy":
        return dict(
            policy_lag_gate_active=True,
            policy_lag_gate_mode=POLICY_LAG_MODE_LEGACY,
            max_policy_lag=int(legacy_max_policy_lag),
            off_policy_correction="vtrace+awr_legacy",
            rho_bar=float(rho_bar), c_bar=float(c_bar))
    raise ValueError(f"policy_lag_runtime_manifest: unknown replay_mode {replay_mode!r}")


def validate_policy_lag_config(scientific_config):
    """Fail closed (CC2 §三.1): original_vtrace MUST NOT declare an active policy-lag gate.

    Accepts the normalized `policy_lag` block:
        policy_lag: {active: false, mode: not_applicable_original_vtrace,
                     max_policy_lag: null, correction: {method, rho_bar, c_bar}}
    Raises ValueError('ORIGINAL_VTRACE_POLICY_LAG_CONFIG_CONFLICT...') when:
      * replay_mode == original_vtrace AND policy_lag.active is true, OR
      * replay_mode == original_vtrace AND a stray top-level max_policy_lag is present.
    Returns the (possibly empty) policy_lag block on success.
    """
    replay_mode = scientific_config.get("replay_mode")
    pl = scientific_config.get("policy_lag") or {}
    active = bool(pl.get("active", False))
    stray_top_level = scientific_config.get("max_policy_lag", None)
    if replay_mode == "original_vtrace":
        if active:
            raise ValueError(
                "ORIGINAL_VTRACE_POLICY_LAG_CONFIG_CONFLICT: replay_mode=original_vtrace "
                "forbids policy_lag.active=true (V-trace importance sampling is the ONLY "
                "off-policy correction; no additional hard lag gate).")
        if stray_top_level is not None:
            raise ValueError(
                "ORIGINAL_VTRACE_POLICY_LAG_CONFIG_CONFLICT: replay_mode=original_vtrace has a "
                "top-level max_policy_lag; legacy lag belongs under legacy_full_p2_only with "
                "policy_lag.active=false for original_vtrace.")
    return pl


# ===========================================================================
# §四 — replay protocol / exposure / content label split
# ===========================================================================
# These four labels are MUTUALLY DISTINCT and must never be collapsed into a single
# "MATCHED_REPLAY_PROTOCOL_READY=true".

SAME_REPLAY_PROTOCOL_READY = "READY"
MATCHED_REPLAY_EXPOSURE_NOT_RUN = "NOT_RUN"
MATCHED_REPLAY_CONTENT_NOT_CLAIMED = "NOT_CLAIMED"
ENDOGENOUS_REPLAY_SCREENING_READY = "READY_AFTER_SMOKE"


def replay_protocol_labels(replay_mode, sequence_length, batch_size,
                           sampler="eligible_only",
                           learner="original_vtrace_update_rmt",
                           loss="vtrace_original_goal",
                           rng_rule="np.random.RandomState(seed+7)"):
    """The four-way label block (CC2 §四). SAME_REPLAY_PROTOCOL=READY asserts only that both
    arms share an IDENTICAL protocol definition; it does NOT assert matched exposure/content."""
    protocol_definition = dict(
        sequence_length=int(sequence_length), batch_size=int(batch_size),
        replay_mode=replay_mode, sampler=sampler, learner=learner, loss=loss, rng_rule=rng_rule)
    return dict(
        SAME_REPLAY_PROTOCOL=SAME_REPLAY_PROTOCOL_READY,
        MATCHED_REPLAY_EXPOSURE=MATCHED_REPLAY_EXPOSURE_NOT_RUN,
        MATCHED_REPLAY_CONTENT=MATCHED_REPLAY_CONTENT_NOT_CLAIMED,
        MATCHED_REPLAY_CONTENT_reason=(
            "endogenous per-arm trajectory buffers: trajectory IDs, episode content, eligible "
            "sets and start offsets differ across arms; content match is NOT claimed"),
        ENDOGENOUS_REPLAY_SCREENING=ENDOGENOUS_REPLAY_SCREENING_READY,
        protocol_definition=protocol_definition,
    )


# ===========================================================================
# §五 — exposure certificate spec + fail-closed claim gates
# ===========================================================================
# Per-arm FINAL summary MUST emit at least these fields (CC2 §五). sample_ids/start_offsets are
# per-arm INTERNAL provenance only — they are NOT compared across arms (endogenous buffers have
# no shared trajectory identity).
EXPOSURE_CERTIFICATE_FIELDS = [
    "outer_update_count",
    "replay_attempt_mask",
    "replay_attempt_outer_updates",
    "replay_not_ready_outer_updates",
    "replay_update_outer_updates",
    "replay_update_count",
    "accepted_replay_policy_update_count",
    "kl_rejected_replay_update_count",
    "replay_sequences_consumed",
    "replay_batch_sizes",
    "replay_sequence_lengths",
    "eligible_count_by_outer_update",
    "sample_ids_by_outer_update",
    "start_offsets_by_outer_update",
]

# The fields that must be EQUAL across both arms for MATCHED_REPLAY_EXPOSURE=PASS (CC2 §五.1).
EXPOSURE_MATCH_FIELDS = [
    "replay_attempt_mask",
    "replay_update_outer_updates",
    "replay_update_count",
    "replay_sequences_consumed",
    "replay_batch_sizes",
    "replay_sequence_lengths",
]

# Fields the two-arm PROTOCOL comparison checks (level 1, CC2 §五).
PROTOCOL_MATCH_FIELDS = [
    "sequence_length", "batch_size", "replay_mode", "sampler", "loss",
]


def check_certificate_complete(summary, arm_name="arm"):
    """Return the list of missing EXPOSURE_CERTIFICATE_FIELDS (empty == complete)."""
    if not summary:
        return list(EXPOSURE_CERTIFICATE_FIELDS)
    return [f for f in EXPOSURE_CERTIFICATE_FIELDS if f not in summary]


def compare_exposure(arm_a_summary, arm_b_summary):
    """Three-level two-arm comparison (CC2 §五). Fail-closed.

    Level 1 PROTOCOL_MATCH     : protocol_definition fields equal.
    Level 2 EXPOSURE_COUNT_MATCH: EXPOSURE_MATCH_FIELDS equal -> MATCHED_REPLAY_EXPOSURE=PASS,
                                  else FAIL (still usable as ENDOGENOUS_REPLAY_SCREENING).
    Level 3 CONTENT_MATCH      : always NOT_APPLICABLE_ENDOGENOUS_BUFFERS (never PASS).

    Raises ValueError('MATCHED_REPLAY_CERTIFICATE_REQUIRED...') if either certificate is
    missing/incomplete.
    """
    for name, s in (("arm_a", arm_a_summary), ("arm_b", arm_b_summary)):
        missing = check_certificate_complete(s, name)
        if missing:
            raise ValueError(
                f"MATCHED_REPLAY_CERTIFICATE_REQUIRED: {name} exposure certificate missing "
                f"fields {missing}; cannot assess MATCHED_REPLAY_EXPOSURE.")

    # Level 1 — protocol
    pa = arm_a_summary.get("protocol_definition", {})
    pb = arm_b_summary.get("protocol_definition", {})
    protocol_diff = [f for f in PROTOCOL_MATCH_FIELDS if pa.get(f) != pb.get(f)]
    protocol_match = "PASS" if not protocol_diff else "FAIL"

    # Level 2 — exposure counts/timing
    exposure_diff = [f for f in EXPOSURE_MATCH_FIELDS
                     if arm_a_summary.get(f) != arm_b_summary.get(f)]
    exposure_match = "PASS" if not exposure_diff else "FAIL"

    # Level 3 — content (endogenous buffers: never PASS)
    return dict(
        PROTOCOL_MATCH=protocol_match,
        PROTOCOL_DIFFERING_FIELDS=protocol_diff,
        EXPOSURE_COUNT_MATCH=exposure_match,
        EXPOSURE_DIFFERING_FIELDS=exposure_diff,
        MATCHED_REPLAY_EXPOSURE=("PASS" if (protocol_match == "PASS" and exposure_match == "PASS")
                                 else "FAIL"),
        CONTENT_MATCH="NOT_APPLICABLE_ENDOGENOUS_BUFFERS",
        MATCHED_REPLAY_CONTENT="NOT_CLAIMED",
        ENDOGENOUS_REPLAY_SCREENING="READY_AFTER_SMOKE",
    )


def assert_matched_exposure_pass_allowed(arm_a_summary, arm_b_summary):
    """Fail-closed gate (CC2 §五.2): a report may write MATCHED_REPLAY_EXPOSURE=PASS ONLY if both
    complete arm certificates exist AND compare_exposure yields PASS. Otherwise raises
    ValueError('MATCHED_REPLAY_CERTIFICATE_REQUIRED...'). Returns the comparison dict."""
    result = compare_exposure(arm_a_summary, arm_b_summary)  # raises if a certificate is missing
    if result["MATCHED_REPLAY_EXPOSURE"] != "PASS":
        raise ValueError(
            "MATCHED_REPLAY_CERTIFICATE_REQUIRED: cannot claim MATCHED_REPLAY_EXPOSURE=PASS; "
            f"exposure comparison = {result['MATCHED_REPLAY_EXPOSURE']} "
            f"(differing: {result['EXPOSURE_DIFFERING_FIELDS']}).")
    return result


def assert_content_match_not_claimed(buffer_kind="endogenous"):
    """Fail-closed gate (CC2 §五.2): MATCHED_REPLAY_CONTENT=PASS is impossible with endogenous
    buffers. Raises ValueError('ENDOGENOUS_REPLAY_CONTENT_CANNOT_BE_CLAIMED_MATCHED')."""
    if buffer_kind == "endogenous":
        raise ValueError(
            "ENDOGENOUS_REPLAY_CONTENT_CANNOT_BE_CLAIMED_MATCHED: endogenous per-arm buffers "
            "have no shared trajectory identity; MATCHED_REPLAY_CONTENT=PASS is forbidden.")
    return dict(MATCHED_REPLAY_CONTENT="NOT_CLAIMED")
