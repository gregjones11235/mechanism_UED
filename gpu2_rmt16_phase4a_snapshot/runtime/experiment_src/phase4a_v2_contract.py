"""Phase4A-v2.1 replay contract (CC2 directive §三/§四/§五). PURE Python — no JAX/numpy.

Centralizes three things that were previously implicit, unused, or mis-stated, so that config,
runtime manifest, summary, reports and gates all speak with ONE vocabulary:

  (§三) the policy-lag GATE identity of each replay mode, plus a fail-closed config validator;
  (§四) the four-way split of the previously-conflated "matched replay protocol" concept;
  (§五) the per-arm exposure-certificate field spec and the fail-closed claim gates.

It is imported by the launcher (manifest + summary), by tests/test_phase4a_v2_gates.py and by
tests/phase4a_v2_exposure_validator.py.

Phase4A-v2.2 (§二): the two-arm protocol comparison was upgraded from an INCOMPLETE field
whitelist (missing learner and rng_rule) to FULL canonical protocol identity: required-field
completeness, identical key sets, identical canonical dicts and identical protocol SHA256.

Phase4A-v2.3 (§八): the replay protocol's learner/sampler/RNG were STRING DECLARATIONS only —
nothing bound them to the code that actually EXECUTES. v2.3 adds executed-protocol source
identity: after import, the driver binds the REAL learner function, sampler function and sampler
RNG via inspect (module / qualname / source SHA256), fail-closed, and checks them against the
declared protocol_definition. stdlib `inspect` only — still no JAX/numpy at import time.
"""
import hashlib
import json

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


def active_replay_config_manifest(replay_mode, rho_bar=1.0, c_bar=1.0):
    """Phase4A-v2.2 (§三.1): the ACTIVE replay-configuration block recorded in the summary,
    the checkpoint manifests and the run certificate.

    For replay_mode=original_vtrace NO field in this block is a numeric max_policy_lag
    (max_policy_lag=null, policy_lag_gate_active=false, correction=vtrace_importance_sampling).
    The legacy numeric lag (16) lives ONLY in the inactive legacy_full_p2_only scope. Built from
    policy_lag_runtime_manifest so the identity has a single source."""
    return dict(replay_mode=replay_mode,
                **policy_lag_runtime_manifest(replay_mode, rho_bar=rho_bar, c_bar=c_bar))


def legacy_full_p2_manifest(active=False, max_policy_lag=16):
    """Phase4A-v2.2 (§三.1): the ONLY place the numeric legacy lag 16 may appear in an
    original_vtrace run — an explicitly INACTIVE legacy scope. For original_vtrace runs
    `active` MUST be false; the 16 is a documentary legacy reference, not a live gate."""
    return dict(active=bool(active), max_policy_lag=int(max_policy_lag))


# Block paths considered ACTIVE for the §三.3 leak scan (relative to the scanned document).
_ACTIVE_POLICY_LAG_SCAN_PATHS = (
    (),                                    # a stray top-level max_policy_lag
    ("phase4a_v2",),
    ("phase4a_v2", "active_replay_config"),
    ("active_replay_config",),
    ("p2_frozen",),
    ("scientific_config", "policy_lag"),
    ("policy_lag",),
    ("run_manifest",),
    ("manifest",),
)
_LEGACY_POLICY_LAG_SCAN_PATHS = (
    ("legacy_full_p2_only",),
    ("scientific_config", "legacy_full_p2_only"),
    ("phase4a_v2", "legacy_full_p2_only"),
)


def _collect_replay_modes(doc):
    """Best-effort collection of replay_mode declarations anywhere in a document."""
    modes = []

    def take(node):
        if isinstance(node, dict):
            m = node.get("replay_mode")
            if isinstance(m, str):
                modes.append(m)

    if isinstance(doc, dict):
        take(doc)
        for key in ("scientific_config", "active_replay_config", "phase4a_v2",
                    "run_manifest", "manifest"):
            take(doc.get(key))
        v = doc.get("phase4a_v2")
        if isinstance(v, dict):
            take(v.get("active_replay_config"))
    return modes


def assert_no_active_policy_lag_leak(doc):
    """Phase4A-v2.2 (§三.3): fail-closed leak scan over a summary / manifest / certificate doc.

    When replay_mode=original_vtrace, NO active block may carry a numeric max_policy_lag or an
    active policy-lag gate. Scanned active blocks: top level, phase4a_v2,
    phase4a_v2.active_replay_config, active_replay_config, p2_frozen,
    scientific_config.policy_lag, policy_lag, run_manifest / manifest. The legacy 16 is legal
    ONLY under legacy_full_p2_only AND requires legacy_full_p2_only.active == false (explicit).

    Raises ValueError('ORIGINAL_VTRACE_ACTIVE_POLICY_LAG_LEAK...') on any violation.
    Returns a scan record on a clean document (no-op when replay_mode is not original_vtrace)."""
    modes = _collect_replay_modes(doc)
    if "original_vtrace" not in modes:
        return dict(replay_modes_seen=modes, scan="not_applicable_not_original_vtrace")
    leaks = []
    root = doc if isinstance(doc, dict) else {}
    for path in _ACTIVE_POLICY_LAG_SCAN_PATHS:
        node = root
        for part in path:
            node = node.get(part) if isinstance(node, dict) else None
        if not isinstance(node, dict):
            continue
        label = ".".join(path) if path else "(top_level)"
        v = node.get("max_policy_lag")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            leaks.append((f"{label}.max_policy_lag", v))
        if node.get("policy_lag_gate_active") is True:
            leaks.append((f"{label}.policy_lag_gate_active", True))
    for path in _LEGACY_POLICY_LAG_SCAN_PATHS:
        node = root
        for part in path:
            node = node.get(part) if isinstance(node, dict) else None
        if (isinstance(node, dict)
                and isinstance(node.get("max_policy_lag"), (int, float))
                and not isinstance(node.get("max_policy_lag"), bool)
                and node.get("active") is not False):
            leaks.append((f"{'.'.join(path)}.active_not_explicitly_false", node.get("active")))
    if leaks:
        raise ValueError(
            "ORIGINAL_VTRACE_ACTIVE_POLICY_LAG_LEAK: replay_mode=original_vtrace but an active "
            f"policy-lag leak was detected: {leaks}. The numeric legacy lag 16 is legal ONLY "
            "under legacy_full_p2_only with active=false; every active block must read "
            "policy_lag_gate_active=false / max_policy_lag=null / "
            "off_policy_correction=vtrace_importance_sampling.")
    return dict(replay_modes_seen=modes, scan="clean", leaks=[])


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
                           rng_rule="np.random.RandomState(seed+7)",
                           rng_engine="np.random.RandomState",
                           rng_seed_derivation="run_seed_plus_7",
                           rng_stream="dedicated_replay_sampler",
                           hidden_buffer_rng_used=False):
    """The four-way label block (CC2 §四). SAME_REPLAY_PROTOCOL=READY asserts only that both
    arms share an IDENTICAL protocol definition; it does NOT assert matched exposure/content.

    Phase4A-v2.2 (§二): the protocol_definition is the COMPLETE canonical protocol identity.
    rng_rule keeps the human-readable rule string AND carries explicit auditable sub-fields
    (rng_engine / rng_seed_derivation / rng_stream / hidden_buffer_rng_used) so that two arms
    using a different learner or a different RNG rule can NEVER be judged PROTOCOL_MATCH=PASS."""
    protocol_definition = dict(
        sequence_length=int(sequence_length), batch_size=int(batch_size),
        replay_mode=replay_mode, sampler=sampler, learner=learner, loss=loss,
        rng_rule=rng_rule,
        # Phase4A-v2.2 (§二.2): explicit, auditable RNG identity of the replay sampler.
        rng_engine=rng_engine,
        rng_seed_derivation=rng_seed_derivation,
        rng_stream=rng_stream,
        hidden_buffer_rng_used=bool(hidden_buffer_rng_used))
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

# Phase4A-v2.2 (§二): the COMPLETE protocol identity spec. The old partial whitelist
# (sequence_length/batch_size/replay_mode/sampler/loss) is DELETED on purpose — it silently
# ignored learner and rng_rule, so two arms with a different learner or a different RNG rule
# could still be judged PROTOCOL_MATCH=PASS. Every protocol_definition MUST now carry ALL
# required fields (missing -> fail closed), and the comparison is FULL canonical-dict identity
# (all keys, all values, canonical SHA) — not a subset whitelist.
REQUIRED_PROTOCOL_FIELDS = {
    "sequence_length",
    "batch_size",
    "replay_mode",
    "sampler",
    "learner",
    "loss",
    "rng_rule",
}


def canonical_protocol_json(protocol_definition):
    """Canonical JSON serialization of a protocol definition (CC2 §二.1).

    Key order is IRRELEVANT for identity: dicts with identical keys/values but different
    insertion order produce identical canonical JSON. Used for both the byte-exact comparison
    and the protocol SHA256."""
    if not isinstance(protocol_definition, dict):
        raise ValueError(
            "PROTOCOL_IDENTITY_INCOMPLETE: protocol_definition must be a dict, got "
            f"{type(protocol_definition).__name__}")
    return json.dumps(protocol_definition, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def protocol_definition_sha256(protocol_definition):
    """SHA256 of the canonical protocol JSON (CC2 §二.1 emitted comparison field)."""
    return hashlib.sha256(
        canonical_protocol_json(protocol_definition).encode("utf-8")).hexdigest()


def missing_required_protocol_fields(protocol_definition):
    """Sorted list of REQUIRED_PROTOCOL_FIELDS absent from a protocol definition."""
    if not isinstance(protocol_definition, dict):
        return sorted(REQUIRED_PROTOCOL_FIELDS)
    return sorted(f for f in REQUIRED_PROTOCOL_FIELDS if f not in protocol_definition)


def compare_protocols(arm_a_protocol, arm_b_protocol):
    """FULL canonical protocol identity comparison (CC2 §二.1).

    PROTOCOL_MATCH=PASS requires ALL of:
      * both arms carry a dict protocol_definition;
      * both are complete w.r.t. REQUIRED_PROTOCOL_FIELDS (missing -> fail closed);
      * identical key sets (an extra/unknown field on one arm only -> FAIL);
      * identical canonical dicts (values equal; key order irrelevant);
      * identical protocol SHA256.

    Emits at least: PROTOCOL_MATCH, PROTOCOL_MISSING_FIELDS_ARM_A, PROTOCOL_MISSING_FIELDS_ARM_B,
    PROTOCOL_DIFFERING_FIELDS, PROTOCOL_DEFINITION_SHA256_ARM_A, PROTOCOL_DEFINITION_SHA256_ARM_B.
    Raises ValueError('PROTOCOL_IDENTITY_INCOMPLETE...') when either protocol is absent/not a
    dict or misses required fields (fail closed)."""
    result = dict(
        PROTOCOL_MATCH="FAIL",
        PROTOCOL_MISSING_FIELDS_ARM_A=missing_required_protocol_fields(arm_a_protocol),
        PROTOCOL_MISSING_FIELDS_ARM_B=missing_required_protocol_fields(arm_b_protocol),
        PROTOCOL_KEYSET_MISMATCH=[],
        PROTOCOL_DIFFERING_FIELDS=[],
        PROTOCOL_DEFINITION_SHA256_ARM_A=None,
        PROTOCOL_DEFINITION_SHA256_ARM_B=None,
    )
    if not isinstance(arm_a_protocol, dict) or not isinstance(arm_b_protocol, dict):
        raise ValueError(
            "PROTOCOL_IDENTITY_INCOMPLETE: protocol_definition missing or not a dict on at least "
            "one arm; cannot assess PROTOCOL_MATCH.")
    if result["PROTOCOL_MISSING_FIELDS_ARM_A"] or result["PROTOCOL_MISSING_FIELDS_ARM_B"]:
        raise ValueError(
            "PROTOCOL_IDENTITY_INCOMPLETE: protocol_definition missing required fields "
            f"(arm_a={result['PROTOCOL_MISSING_FIELDS_ARM_A']}, "
            f"arm_b={result['PROTOCOL_MISSING_FIELDS_ARM_B']}); fail closed.")

    keys_a = set(arm_a_protocol)
    keys_b = set(arm_b_protocol)
    result["PROTOCOL_KEYSET_MISMATCH"] = sorted(keys_a ^ keys_b)
    result["PROTOCOL_DIFFERING_FIELDS"] = sorted(
        k for k in keys_a & keys_b if arm_a_protocol[k] != arm_b_protocol[k])
    result["PROTOCOL_DEFINITION_SHA256_ARM_A"] = protocol_definition_sha256(arm_a_protocol)
    result["PROTOCOL_DEFINITION_SHA256_ARM_B"] = protocol_definition_sha256(arm_b_protocol)

    if (not result["PROTOCOL_KEYSET_MISMATCH"]
            and not result["PROTOCOL_DIFFERING_FIELDS"]
            and canonical_protocol_json(arm_a_protocol) == canonical_protocol_json(arm_b_protocol)
            and result["PROTOCOL_DEFINITION_SHA256_ARM_A"]
            == result["PROTOCOL_DEFINITION_SHA256_ARM_B"]):
        result["PROTOCOL_MATCH"] = "PASS"
    return result


def check_certificate_complete(summary, arm_name="arm"):
    """Return the list of missing EXPOSURE_CERTIFICATE_FIELDS (empty == complete)."""
    if not summary:
        return list(EXPOSURE_CERTIFICATE_FIELDS)
    return [f for f in EXPOSURE_CERTIFICATE_FIELDS if f not in summary]


def compare_exposure(arm_a_summary, arm_b_summary):
    """Three-level two-arm comparison (CC2 §五, upgraded by v2.2 §二/§八 and v2.4 §九).
    Fail-closed.

    Level 1 PROTOCOL_MATCH      : the EFFECTIVE protocol must match (Phase4A-v2.4 §九) — i.e.
                                  ALL THREE of:
                                    DECLARED_PROTOCOL_MATCH   full canonical declared protocol
                                        identity (compare_protocols: required fields complete
                                        incl. learner and rng_rule, identical key sets, identical
                                        canonical dicts, identical protocol SHA256);
                                    EXECUTED_PROTOCOL_MATCH   the executed learner/sampler/RNG
                                        source identities complete and equal across arms
                                        (module + module_file_sha256 + function_source_sha256 +
                                        RNG class + numpy version);
                                    EFFECTIVE_PROTOCOL_MATCH  identical effective key sets,
                                        identical canonical effective dicts, identical
                                        effective_protocol_sha256.
                                  Even identical exposure counts with ANY declared, executed or
                                  effective difference -> PROTOCOL_MATCH=FAIL.
    Level 2 EXPOSURE_COUNT_MATCH: EXPOSURE_MATCH_FIELDS equal.
    Level 3 CONTENT_MATCH       : always NOT_APPLICABLE_ENDOGENOUS_BUFFERS (never PASS).

    MATCHED_REPLAY_EXPOSURE=PASS requires EFFECTIVE_PROTOCOL_MATCH=PASS AND
    EXPOSURE_COUNT_MATCH=PASS (Phase4A-v2.4 §九).

    Raises ValueError('MATCHED_REPLAY_CERTIFICATE_REQUIRED...') if either certificate is
    missing/incomplete, ValueError('PROTOCOL_IDENTITY_INCOMPLETE...') if either declared
    protocol_definition is missing/incomplete, and ValueError('EXECUTED_PROTOCOL_IDENTITY_
    REQUIRED...') if either arm lacks a complete executed/effective identity (fail closed; NO
    fallback to declared-string-only comparison).
    """
    for name, s in (("arm_a", arm_a_summary), ("arm_b", arm_b_summary)):
        missing = check_certificate_complete(s, name)
        if missing:
            raise ValueError(
                f"MATCHED_REPLAY_CERTIFICATE_REQUIRED: {name} exposure certificate missing "
                f"fields {missing}; cannot assess MATCHED_REPLAY_EXPOSURE.")

    # Level 1a — DECLARED protocol (v2.2 §二: FULL canonical identity, not a field whitelist)
    protocol = compare_protocols(
        arm_a_summary.get("protocol_definition"),
        arm_b_summary.get("protocol_definition"))
    declared_match = protocol["PROTOCOL_MATCH"]

    # Level 1b — EXECUTED + EFFECTIVE protocol (v2.4 §九; fail closed on missing identity)
    effective = compare_effective_protocols(arm_a_summary, arm_b_summary)
    executed_match = effective["EXECUTED_PROTOCOL_MATCH"]
    effective_match = effective["EFFECTIVE_PROTOCOL_MATCH"]
    protocol_match = ("PASS" if (declared_match == "PASS" and executed_match == "PASS"
                                 and effective_match == "PASS") else "FAIL")

    # Level 2 — exposure counts/timing
    exposure_diff = [f for f in EXPOSURE_MATCH_FIELDS
                     if arm_a_summary.get(f) != arm_b_summary.get(f)]
    exposure_match = "PASS" if not exposure_diff else "FAIL"

    # Level 3 — content (endogenous buffers: never PASS)
    return dict(
        PROTOCOL_MATCH=protocol_match,
        DECLARED_PROTOCOL_MATCH=declared_match,
        EXECUTED_PROTOCOL_MATCH=executed_match,
        EFFECTIVE_PROTOCOL_MATCH=effective_match,
        EFFECTIVE_PROTOCOL_SHA256_ARM_A=effective["EFFECTIVE_PROTOCOL_SHA256_ARM_A"],
        EFFECTIVE_PROTOCOL_SHA256_ARM_B=effective["EFFECTIVE_PROTOCOL_SHA256_ARM_B"],
        EFFECTIVE_PROTOCOL_KEYSET_MISMATCH=effective["EFFECTIVE_PROTOCOL_KEYSET_MISMATCH"],
        EXECUTED_PROTOCOL_DIFFERING_FIELDS=effective["EXECUTED_PROTOCOL_DIFFERING_FIELDS"],
        PROTOCOL_MISSING_FIELDS_ARM_A=protocol["PROTOCOL_MISSING_FIELDS_ARM_A"],
        PROTOCOL_MISSING_FIELDS_ARM_B=protocol["PROTOCOL_MISSING_FIELDS_ARM_B"],
        PROTOCOL_KEYSET_MISMATCH=protocol["PROTOCOL_KEYSET_MISMATCH"],
        PROTOCOL_DIFFERING_FIELDS=protocol["PROTOCOL_DIFFERING_FIELDS"],
        PROTOCOL_DEFINITION_SHA256_ARM_A=protocol["PROTOCOL_DEFINITION_SHA256_ARM_A"],
        PROTOCOL_DEFINITION_SHA256_ARM_B=protocol["PROTOCOL_DEFINITION_SHA256_ARM_B"],
        EXPOSURE_COUNT_MATCH=exposure_match,
        EXPOSURE_DIFFERING_FIELDS=exposure_diff,
        MATCHED_REPLAY_EXPOSURE=("PASS" if (effective_match == "PASS"
                                            and exposure_match == "PASS") else "FAIL"),
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


# ===========================================================================
# Phase4A-v2.3 (§八) — EXECUTED protocol source identity (bound to running code)
# ===========================================================================
# v2.2's protocol_definition carried the learner/sampler/rng as STRING LABELS. Nothing proved
# the code that actually EXECUTES during training is what the labels say: a driver could declare
# learner="original_vtrace_update_rmt" while importing/calling something else. v2.3 binds the
# EXECUTING objects by inspecting them (module / qualname / source SHA256) AFTER import, fails
# closed when the bound objects are not the expected ones, and then checks the bound identity
# against the declared protocol_definition (two-phase: declare, then bind + reconcile).
#
# Phase4A-v2.4 (§七/§八): the binding was still too weak in three ways: (a) a SAME-NAME impostor
# defined in a DIFFERENT module passed the name-only check; (b) the identity carried no MODULE
# file binding (two modules with the same function source but different module files were
# indistinguishable); (c) the RNG identity carried no numpy version and the bound identity never
# entered the two-arm comparison. v2.4 therefore binds, per function: module / qualname / name /
# module_realpath / module_file_sha256 / function_source_sha256 / source_lines — fail closed
# (EXECUTED_PROTOCOL_SOURCE_UNAVAILABLE) when ANY of module/sourcefile/existence/readability/
# source-inspectability fails — and requires the executed objects to come from the DECLARED
# module + qualname (a same-name impostor from another module is rejected with
# EXECUTED_PROTOCOL_SOURCE_MISMATCH). The RNG identity now binds numpy_version +
# seed_derivation + hidden_buffer_rng_used, and build_effective_protocol_definition folds the
# declared protocol + executed learner/sampler + executed RNG into one effective protocol with
# a stable SHA, compared across arms by compare_effective_protocols (§九).

DECLARED_PROTOCOL_LEARNER = "original_vtrace_update_rmt"      # RL.original_vtrace_update_rmt
DECLARED_PROTOCOL_LEARNER_MODULE = "rmt_replay_learner"       # the module it MUST execute from
DECLARED_PROTOCOL_SAMPLER_FUNCTION = "sample_eligible"        # RMTReplayBuffer.sample_eligible
DECLARED_PROTOCOL_SAMPLER_QUALNAME = "RMTReplayBuffer.sample_eligible"
DECLARED_PROTOCOL_SAMPLER_MODULE = "rmt_replay_buffer"        # the module it MUST execute from
DECLARED_PROTOCOL_SAMPLER_LABEL = "eligible_only"             # protocol_definition["sampler"]
DECLARED_PROTOCOL_RNG_ENGINE = "np.random.RandomState"        # protocol_definition["rng_engine"]
DECLARED_PROTOCOL_RNG_SEED_DERIVATION = "run_seed_plus_7"     # RandomState(args.seed + 7)
DECLARED_PROTOCOL_HIDDEN_BUFFER_RNG_USED = False              # dedicated RNG, never the buffer's
_RNG_RANDOMSTATE_MODULES = ("numpy.random", "numpy.random.mtrand")


def _source_identity(fn):
    """The FULL source identity of a callable. Phase4A-v2.4 (§七): module / qualname / name /
    module_realpath / module_file_sha256 / function_source_sha256 / source_lines.

    Fail closed (EXECUTED_PROTOCOL_SOURCE_UNAVAILABLE) if the source cannot be inspected OR the
    defining module file cannot be located / does not exist / has no realpath / is unreadable /
    its SHA cannot be computed — an un-inspectable or module-file-unbound learner/sampler can
    never be proven to be the declared one (a same-name impostor, or code exec'd from a deleted
    file, must be rejected, not passed)."""
    import inspect
    import os
    qualname = getattr(fn, "__qualname__", None)
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError) as e:
        raise ValueError(
            f"EXECUTED_PROTOCOL_SOURCE_UNAVAILABLE: cannot inspect source of "
            f"{qualname or repr(fn)}: {e}")
    module = inspect.getmodule(fn)
    source_file = None
    try:
        source_file = inspect.getsourcefile(fn)
    except TypeError:
        source_file = None
    if module is None:
        raise ValueError(
            f"EXECUTED_PROTOCOL_SOURCE_UNAVAILABLE: no module resolvable for {qualname or repr(fn)}")
    if not source_file:
        raise ValueError(
            f"EXECUTED_PROTOCOL_SOURCE_UNAVAILABLE: no source file resolvable for "
            f"{qualname or repr(fn)} (module={getattr(module, '__name__', None)!r})")
    if not os.path.exists(source_file):
        raise ValueError(
            f"EXECUTED_PROTOCOL_SOURCE_UNAVAILABLE: source file does not exist: {source_file!r} "
            f"(function {qualname or repr(fn)})")
    try:
        module_realpath = os.path.realpath(source_file)
    except OSError as e:
        raise ValueError(
            f"EXECUTED_PROTOCOL_SOURCE_UNAVAILABLE: cannot resolve realpath of {source_file!r}: "
            f"{e}")
    try:
        with open(module_realpath, "rb") as f:
            module_file_sha256 = hashlib.sha256(f.read()).hexdigest()
    except OSError as e:
        raise ValueError(
            f"EXECUTED_PROTOCOL_SOURCE_UNAVAILABLE: cannot read/hash module file "
            f"{module_realpath!r}: {e}")
    return dict(module=getattr(fn, "__module__", None),
                qualname=qualname,
                name=getattr(fn, "__name__", None),
                module_realpath=module_realpath,
                module_file_sha256=module_file_sha256,
                function_source_sha256=hashlib.sha256(src.encode("utf-8")).hexdigest(),
                source_lines=len(src.splitlines()))


def executed_function_source_identity(learner_fn, sampler_fn):
    """§八 phase 2a: bind the ACTUALLY EXECUTING replay learner + sampler — not their labels.

    learner_fn : the function the training loop actually calls for the replay update
                 (expected: rmt_replay_learner.original_vtrace_update_rmt)
    sampler_fn : the function the loop actually samples with (expected:
                 RMTReplayBuffer.sample_eligible — pass the unbound method or a bound one)

    Fail closed (EXECUTED_PROTOCOL_SOURCE_MISMATCH) if either object's source/module file is
    uninspectable, its name is not the expected one, OR it executes from the wrong module /
    qualname. Phase4A-v2.4 (§七.1): the module + qualname expectation is what rejects a
    SAME-NAME impostor — a function called `original_vtrace_update_rmt` defined in any module
    other than `rmt_replay_learner` (or a `sample_eligible` not qualified as
    `RMTReplayBuffer.sample_eligible` of `rmt_replay_buffer`) is rejected even though its name
    matches. The per-function module_file_sha256 + function_source_sha256 then let the two-arm
    validator (§九) reject a same-name function whose SOURCE or MODULE FILE differs. Returns a
    record with the full per-function source identity."""
    errors = []
    learner_id = _source_identity(learner_fn)
    sampler_id = _source_identity(sampler_fn)
    if learner_id["name"] != DECLARED_PROTOCOL_LEARNER:
        errors.append(
            f"executed learner={learner_id['name']!r} ({learner_id['module']}) != declared "
            f"{DECLARED_PROTOCOL_LEARNER!r}")
    if learner_id["module"] != DECLARED_PROTOCOL_LEARNER_MODULE:
        errors.append(
            f"executed learner module={learner_id['module']!r} != declared module "
            f"{DECLARED_PROTOCOL_LEARNER_MODULE!r} (same-name impostor rejected)")
    if sampler_id["name"] != DECLARED_PROTOCOL_SAMPLER_FUNCTION:
        errors.append(
            f"executed sampler={sampler_id['name']!r} ({sampler_id['module']}) != declared "
            f"{DECLARED_PROTOCOL_SAMPLER_FUNCTION!r}")
    if sampler_id["module"] != DECLARED_PROTOCOL_SAMPLER_MODULE:
        errors.append(
            f"executed sampler module={sampler_id['module']!r} != declared module "
            f"{DECLARED_PROTOCOL_SAMPLER_MODULE!r} (same-name impostor rejected)")
    if sampler_id["qualname"] != DECLARED_PROTOCOL_SAMPLER_QUALNAME:
        errors.append(
            f"executed sampler qualname={sampler_id['qualname']!r} != declared qualname "
            f"{DECLARED_PROTOCOL_SAMPLER_QUALNAME!r}")
    if errors:
        raise ValueError("EXECUTED_PROTOCOL_SOURCE_MISMATCH: " + " | ".join(errors))
    return dict(executed_function_binding="PASS",
                learner=learner_id, sampler=sampler_id)


def verify_rng_instance_identity(rng_instance, *, numpy_version=None,
                                 seed_derivation=DECLARED_PROTOCOL_RNG_SEED_DERIVATION,
                                 hidden_buffer_rng_used=DECLARED_PROTOCOL_HIDDEN_BUFFER_RNG_USED):
    """§八 phase 2b: bind the ACTUAL replay-sampler RNG instance. It MUST be a
    numpy.random.RandomState (the declared rng_engine np.random.RandomState) — not the global
    legacy generator, not PCG64, not a hidden buffer RNG. Fail closed (EXECUTED_PROTOCOL_RNG_
    MISMATCH). Returns a class-identity record.

    Phase4A-v2.4 (§八): the record ALSO binds numpy_version (the exact numpy the RandomState
    class comes from), seed_derivation (run_seed_plus_7) and hidden_buffer_rng_used (false) —
    these enter the effective protocol definition and the two-arm comparison, so an arm that
    silently switches numpy or seed rule can never be judged protocol-matched. This function
    only INSPECTS the instance class; it draws NO random numbers (the RNG state is not
    consumed before training)."""
    cls = type(rng_instance)
    identity = dict(class_module=getattr(cls, "__module__", None),
                    class_name=getattr(cls, "__name__", None))
    if not (identity["class_name"] == "RandomState"
            and identity["class_module"] in _RNG_RANDOMSTATE_MODULES):
        raise ValueError(
            f"EXECUTED_PROTOCOL_RNG_MISMATCH: executed replay-sampler RNG is "
            f"{identity['class_module']}.{identity['class_name']}, not numpy.random.RandomState "
            f"(declared rng_engine={DECLARED_PROTOCOL_RNG_ENGINE!r}).")
    if numpy_version is None:
        try:
            import numpy as _np
            numpy_version = getattr(_np, "__version__", None)
        except ImportError:
            numpy_version = None
    if not numpy_version:
        raise ValueError(
            "EXECUTED_PROTOCOL_RNG_MISMATCH: numpy version unavailable; the executed RNG "
            "identity must bind the exact numpy version (fail closed).")
    identity["numpy_version"] = str(numpy_version)
    identity["seed_derivation"] = seed_derivation
    identity["hidden_buffer_rng_used"] = bool(hidden_buffer_rng_used)
    identity["rng_binding"] = "PASS"
    return identity


def verify_executed_protocol_matches_declared(executed_identity, protocol_definition):
    """§八 two-phase reconciliation: the EXECUTED source identity must correspond to the DECLARED
    protocol_definition labels (replay_protocol_labels(...).protocol_definition). The declared
    sampler LABEL 'eligible_only' maps to the executed FUNCTION 'sample_eligible'; learner and
    rng_engine map by name. Fail closed (EXECUTED_PROTOCOL_DECLARATION_MISMATCH)."""
    pd = protocol_definition or {}
    errors = []
    learner = (executed_identity or {}).get("learner") or {}
    sampler = (executed_identity or {}).get("sampler") or {}
    if pd.get("learner") != learner.get("name"):
        errors.append(
            f"declared learner={pd.get('learner')!r} != executed {learner.get('name')!r}")
    if pd.get("sampler") != DECLARED_PROTOCOL_SAMPLER_LABEL:
        errors.append(
            f"declared sampler label={pd.get('sampler')!r} != "
            f"{DECLARED_PROTOCOL_SAMPLER_LABEL!r}")
    if sampler.get("name") != DECLARED_PROTOCOL_SAMPLER_FUNCTION:
        errors.append(
            f"executed sampler={sampler.get('name')!r} != "
            f"{DECLARED_PROTOCOL_SAMPLER_FUNCTION!r}")
    if pd.get("rng_engine") != DECLARED_PROTOCOL_RNG_ENGINE:
        errors.append(
            f"declared rng_engine={pd.get('rng_engine')!r} != {DECLARED_PROTOCOL_RNG_ENGINE!r}")
    if errors:
        raise ValueError("EXECUTED_PROTOCOL_DECLARATION_MISMATCH: " + " | ".join(errors))
    return dict(executed_protocol_declaration_match="PASS",
                declared_learner=pd.get("learner"), executed_learner=learner.get("name"),
                declared_sampler_label=pd.get("sampler"),
                executed_sampler_function=sampler.get("name"),
                declared_rng_engine=pd.get("rng_engine"))


# ===========================================================================
# Phase4A-v2.4 (§八) — EFFECTIVE protocol definition (declared + executed + RNG)
# ===========================================================================
# The declared protocol_definition is a PROMISE; the executed source identity is what RUNS. v2.4
# folds both — plus the executed RNG's class + numpy version + seed rule — into ONE effective
# protocol definition with a stable canonical SHA. Both arms emit it in their summaries, and the
# two-arm validator (§九) compares the EFFECTIVE protocol, not the declared strings alone.

EFFECTIVE_PROTOCOL_LEARNER_FIELDS = (
    "module", "qualname", "module_realpath", "module_file_sha256", "function_source_sha256")
EFFECTIVE_PROTOCOL_SAMPLER_FIELDS = EFFECTIVE_PROTOCOL_LEARNER_FIELDS
EFFECTIVE_PROTOCOL_RNG_FIELDS = (
    "class_module", "class_name", "numpy_version", "seed_derivation", "hidden_buffer_rng_used")
EXECUTED_SOURCE_IDENTITY_FIELDS = (
    "module", "qualname", "name", "module_realpath", "module_file_sha256",
    "function_source_sha256", "source_lines")


def _project_source_identity(source_id, fields):
    """Project a _source_identity record onto the effective-protocol field set (fail closed on
    any missing/null field — an incomplete binding can never enter the effective protocol)."""
    src = source_id or {}
    missing = [f for f in fields if src.get(f) in (None, "")]
    if missing:
        raise ValueError(
            f"EXECUTED_PROTOCOL_IDENTITY_REQUIRED: source identity is incomplete; missing "
            f"field(s) {missing}")
    return {f: src[f] for f in fields}


def build_effective_protocol_definition(declared_protocol_definition, executed_identity,
                                        rng_identity, *, numpy_version=None):
    """§八: assemble the EFFECTIVE protocol definition from the declared protocol + the executed
    learner/sampler source identity + the executed RNG identity, and compute its stable SHA256.

    effective_protocol_definition = {
        declared_protocol : the full declared protocol_definition (all required fields),
        executed_learner  : {module, qualname, module_realpath, module_file_sha256,
                             function_source_sha256},
        executed_sampler  : {same five fields},
        executed_rng      : {class_module, class_name, numpy_version, seed_derivation,
                             hidden_buffer_rng_used}}

    effective_protocol_sha256 = SHA256 of the canonical JSON of that dict (key-order invariant,
    stable for identical bindings). Fails closed (EXECUTED_PROTOCOL_IDENTITY_REQUIRED /
    PROTOCOL_IDENTITY_INCOMPLETE) on any missing/incomplete input. Returns
    (effective_protocol_definition, effective_protocol_sha256)."""
    if not isinstance(declared_protocol_definition, dict):
        raise ValueError(
            "PROTOCOL_IDENTITY_INCOMPLETE: declared protocol_definition missing or not a dict")
    missing_declared = missing_required_protocol_fields(declared_protocol_definition)
    if missing_declared:
        raise ValueError(
            f"PROTOCOL_IDENTITY_INCOMPLETE: declared protocol_definition missing required "
            f"fields {missing_declared}")
    if not isinstance(executed_identity, dict):
        raise ValueError(
            "EXECUTED_PROTOCOL_IDENTITY_REQUIRED: no executed protocol identity supplied")
    learner = _project_source_identity(executed_identity.get("learner"),
                                       EFFECTIVE_PROTOCOL_LEARNER_FIELDS)
    sampler = _project_source_identity(executed_identity.get("sampler"),
                                       EFFECTIVE_PROTOCOL_SAMPLER_FIELDS)
    rng = dict(rng_identity or {})
    rng_missing = [f for f in EFFECTIVE_PROTOCOL_RNG_FIELDS if rng.get(f) in (None, "")]
    if rng_missing:
        raise ValueError(
            f"EXECUTED_PROTOCOL_IDENTITY_REQUIRED: executed RNG identity incomplete; missing "
            f"field(s) {rng_missing}")
    if numpy_version is not None and str(numpy_version) != str(rng.get("numpy_version")):
        raise ValueError(
            f"EXECUTED_PROTOCOL_RNG_MISMATCH: rng_identity numpy_version="
            f"{rng.get('numpy_version')!r} != supplied numpy_version={numpy_version!r}")
    effective = dict(
        declared_protocol=dict(declared_protocol_definition),
        executed_learner=learner,
        executed_sampler=sampler,
        executed_rng={f: rng[f] for f in EFFECTIVE_PROTOCOL_RNG_FIELDS})
    return effective, protocol_definition_sha256(effective)


# ===========================================================================
# Phase4A-v2.4 (§九) — EFFECTIVE protocol cross-arm comparison (fail closed)
# ===========================================================================
# PROTOCOL_MATCH=PASS across the two arms now requires the EFFECTIVE protocol to match — the
# declared strings alone are insufficient. Missing executed identity on either arm fails closed
# with EXECUTED_PROTOCOL_IDENTITY_REQUIRED (NO fallback to declared-only comparison).

def compare_effective_protocols(arm_a_summary, arm_b_summary):
    """§九: compare the EFFECTIVE replay protocol of two arm summaries. Fail closed.

    PROTOCOL identity (effective) PASS requires ALL of:
      * both arms carry a complete executed_protocol_identity (learner + sampler each with
        module / qualname / name / module_realpath / module_file_sha256 /
        function_source_sha256 / source_lines; rng with class_module / class_name /
        numpy_version / seed_derivation / hidden_buffer_rng_used) — missing =>
        EXECUTED_PROTOCOL_IDENTITY_REQUIRED (never a declared-only fallback);
      * both arms carry an effective_protocol_definition dict + effective_protocol_sha256;
      * identical effective key sets;
      * identical learner module + module_file_sha256 + function_source_sha256;
      * identical sampler module + module_file_sha256 + function_source_sha256;
      * identical RNG class_module + class_name + numpy_version;
      * identical effective_protocol_sha256.

    Returns DECLARED-independent fields: EFFECTIVE_PROTOCOL_MATCH, EXECUTED_PROTOCOL_MATCH,
    EFFECTIVE_PROTOCOL_SHA256_ARM_A/_ARM_B, EFFECTIVE_PROTOCOL_KEYSET_MISMATCH,
    EXECUTED_PROTOCOL_DIFFERING_FIELDS."""
    def _executed(summary, arm_name):
        ex = (summary or {}).get("executed_protocol_identity")
        if not isinstance(ex, dict):
            raise ValueError(
                f"EXECUTED_PROTOCOL_IDENTITY_REQUIRED: {arm_name} summary has no "
                "executed_protocol_identity; effective protocol comparison is fail-closed "
                "(no declared-string-only fallback).")
        return ex

    def _require_complete(ex, arm_name):
        missing = []
        for part in ("learner", "sampler"):
            pid = ex.get(part) or {}
            for f in EXECUTED_SOURCE_IDENTITY_FIELDS:
                if pid.get(f) in (None, ""):
                    missing.append(f"{part}.{f}")
        rng = ex.get("rng_instance") or {}
        for f in EFFECTIVE_PROTOCOL_RNG_FIELDS:
            if rng.get(f) in (None, ""):
                missing.append(f"rng_instance.{f}")
        if missing:
            raise ValueError(
                f"EXECUTED_PROTOCOL_IDENTITY_REQUIRED: {arm_name} executed identity incomplete; "
                f"missing field(s) {missing}")

    ex_a = _executed(arm_a_summary, "arm_a")
    ex_b = _executed(arm_b_summary, "arm_b")
    _require_complete(ex_a, "arm_a")
    _require_complete(ex_b, "arm_b")
    eff_a = (arm_a_summary or {}).get("effective_protocol_definition")
    eff_b = (arm_b_summary or {}).get("effective_protocol_definition")
    sha_a = (arm_a_summary or {}).get("effective_protocol_sha256")
    sha_b = (arm_b_summary or {}).get("effective_protocol_sha256")
    if not isinstance(eff_a, dict) or not isinstance(eff_b, dict) or not sha_a or not sha_b:
        raise ValueError(
            "EXECUTED_PROTOCOL_IDENTITY_REQUIRED: effective_protocol_definition / "
            "effective_protocol_sha256 missing on at least one arm; fail closed.")

    result = dict(
        EFFECTIVE_PROTOCOL_MATCH="FAIL",
        EXECUTED_PROTOCOL_MATCH="FAIL",
        EFFECTIVE_PROTOCOL_SHA256_ARM_A=sha_a,
        EFFECTIVE_PROTOCOL_SHA256_ARM_B=sha_b,
        EFFECTIVE_PROTOCOL_KEYSET_MISMATCH=sorted(set(eff_a) ^ set(eff_b)),
        EXECUTED_PROTOCOL_DIFFERING_FIELDS=[])

    differing = []
    la, lb = ex_a["learner"], ex_b["learner"]
    sa, sb = ex_a["sampler"], ex_b["sampler"]
    ra, rb = ex_a["rng_instance"], ex_b["rng_instance"]
    for f in ("module", "module_realpath", "module_file_sha256", "function_source_sha256"):
        if la.get(f) != lb.get(f):
            differing.append(f"learner.{f}")
        if sa.get(f) != sb.get(f):
            differing.append(f"sampler.{f}")
    for f in EFFECTIVE_PROTOCOL_RNG_FIELDS:
        if ra.get(f) != rb.get(f):
            differing.append(f"rng_instance.{f}")
    result["EXECUTED_PROTOCOL_DIFFERING_FIELDS"] = sorted(differing)
    executed_match = not differing
    result["EXECUTED_PROTOCOL_MATCH"] = "PASS" if executed_match else "FAIL"
    if (executed_match
            and not result["EFFECTIVE_PROTOCOL_KEYSET_MISMATCH"]
            and canonical_protocol_json(eff_a) == canonical_protocol_json(eff_b)
            and sha_a == sha_b):
        result["EFFECTIVE_PROTOCOL_MATCH"] = "PASS"
    return result
