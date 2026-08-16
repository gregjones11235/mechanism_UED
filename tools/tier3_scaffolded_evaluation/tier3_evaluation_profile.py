#!/usr/bin/env python3
"""CC4 Tier3 — FROZEN PUBLIC EVALUATION PROFILE (closing contract §2).

Loads and verifies ``configs/tier3_evaluation_profile_v1.json`` — the UNIQUE
scientific protocol for the unified Student pool. Every evaluator / runner /
certificate consumes this ONE document; there is no second protocol, no per-arm
schedule, no ad-hoc seed list.

Verification (fail closed on ANY violation):

  * schema / profile_version constants;
  * self-hash: ``evaluation_profile_sha256`` reproduces from the canonical JSON of
    the remaining fields (the same self-checksum pattern as the checkpoint
    contract — a single tampered byte fails closed);
  * common invariants cross-checked against the LIVE committed constants:
    max_timesteps == tier3_evaluator.MAX_TIMESTEPS (4096), action_dim ==
    tier3_source_audit.CRAFTAX_RUNTIME_BINDINGS["action_count"] (43),
    observation_shape == [8335], action_mode == greedy_argmax;
  * FRONT / BACK scenarios: bank content SHA == the frozen
    tier3_state_bank_materializer.FROZEN_BANK_HASH, full bank indices 0..n-1,
    seeds == materializer.fixed_seed_schedule(...), primary / dense metric names
    == tier3_metrics constants, BACK identity / N/A metrics / no-boss-search;
  * FULL scenario: seeds == tier3_evaluator.PERF_FULL_SEEDS (the committed held-out
    canonical reset seeds 200000..200063 — this set was frozen BEFORE this profile
    and is backed by NEG41 + boundary design §7.5; this module NEVER invents new
    seeds), and FULL_PROFILE_READY must be EXACTLY the truth value of that match:
    if the FULL world/seed set is not frozen and evidence-backed, this field is
    false and no formal FULL performance run is authorized.

PURE STDLIB at import time: runs on the base interpreter (no JAX). The heavy
modules (evaluator / materializer) are imported lazily inside verify_profile; they
are themselves pure at import time, so the full cross-check still needs no JAX.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

# Runnable-as-script AND importable-as-package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_source_audit as audit        # noqa: E402

SCHEMA = "mechanism_UED.tier3_evaluation_profile/v1"
PROFILE_VERSION = "tier3_evaluation_profile/v1"
SELF_SHA_FIELD = "evaluation_profile_sha256"
FROZEN_OBSERVATION_SHAPE = [8335]
FROZEN_ACTION_MODE = "greedy_argmax"

DEFAULT_PROFILE_PATH = str(audit.repo_root() / "configs"
                           / "tier3_evaluation_profile_v1.json")


class FailClosed(Exception):
    """Hard stop on any profile / protocol violation."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


def _is_sha256_hex(value) -> bool:
    return isinstance(value, str) and len(value) == 64 \
        and all(c in "0123456789abcdef" for c in value)


def compute_profile_sha256(profile: dict) -> str:
    """SHA256 of the canonical JSON bytes of the profile with the self-hash field
    removed (sorted keys, compact separators) — identical pattern to
    tier3_checkpoint_contract.contract_sha256."""
    body = {k: v for k, v in profile.items() if k != SELF_SHA_FIELD}
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canon).hexdigest()


def load_profile(path: str = DEFAULT_PROFILE_PATH) -> dict:
    """Load + structurally validate + self-checksum-verify the frozen profile.

    Any structural problem or a wrong recorded ``evaluation_profile_sha256`` fails
    closed. Does NOT yet cross-check against live module constants — call
    verify_profile() for the full gate."""
    require(os.path.isfile(path),
            "FAIL CLOSED: evaluation profile not found at %r" % path)
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    require(isinstance(doc, dict), "FAIL CLOSED: profile is not a JSON object")
    require(doc.get("schema") == SCHEMA,
            "FAIL CLOSED: profile schema %r != %r" % (doc.get("schema"), SCHEMA))
    require(doc.get("profile_version") == PROFILE_VERSION,
            "FAIL CLOSED: profile_version %r != %r"
            % (doc.get("profile_version"), PROFILE_VERSION))
    recorded = doc.get(SELF_SHA_FIELD)
    require(_is_sha256_hex(recorded),
            "FAIL CLOSED: profile %s %r is not a 64-hex value"
            % (SELF_SHA_FIELD, recorded))
    recomputed = compute_profile_sha256(doc)
    require(recomputed == recorded,
            "FAIL CLOSED: profile %s %s does not reproduce (recomputed %s) — the "
            "profile file was tampered with"
            % (SELF_SHA_FIELD, recorded[:16], recomputed[:16]))
    return doc


def verify_profile(profile: dict) -> dict:
    """Full frozen-protocol gate (closing contract §2). Returns the profile on
    success; ANY deviation from the committed canonical evidence raises FailClosed.
    """
    import tier3_metrics as metrics
    import tier3_state_bank_materializer as mat
    import tier3_evaluator as ev

    inv = profile.get("common_evaluation_invariants")
    require(isinstance(inv, dict),
            "FAIL CLOSED: profile has no common_evaluation_invariants object")
    require(int(inv.get("max_timesteps", -1)) == ev.MAX_TIMESTEPS == 4096,
            "FAIL CLOSED: profile max_timesteps %r != frozen evaluator MAX_TIMESTEPS %d"
            % (inv.get("max_timesteps"), ev.MAX_TIMESTEPS))
    require(inv.get("action_mode") == FROZEN_ACTION_MODE,
            "FAIL CLOSED: profile action_mode %r != %r"
            % (inv.get("action_mode"), FROZEN_ACTION_MODE))
    require(list(inv.get("observation_shape", [])) == FROZEN_OBSERVATION_SHAPE,
            "FAIL CLOSED: profile observation_shape %r != frozen %r"
            % (inv.get("observation_shape"), FROZEN_OBSERVATION_SHAPE))
    require(int(inv.get("action_dim", -1))
            == int(audit.CRAFTAX_RUNTIME_BINDINGS["action_count"]) == 43,
            "FAIL CLOSED: profile action_dim %r != frozen runtime binding %d"
            % (inv.get("action_dim"),
               audit.CRAFTAX_RUNTIME_BINDINGS["action_count"]))
    require(inv.get("single_training_seed") is True,
            "FAIL CLOSED: profile must bind single_training_seed=true")
    require(inv.get("scientific_claim_authorized") is False,
            "FAIL CLOSED: profile must bind scientific_claim_authorized=false")
    require(inv.get("scaffolded_results_can_replace_full_task") is False,
            "FAIL CLOSED: scaffolded results can NEVER replace the full task")

    scope = profile.get("candidate_scope")
    require(isinstance(scope, dict), "FAIL CLOSED: profile has no candidate_scope")
    require(scope.get("CANDIDATE_SCOPE") == "RMT16_ORIGINAL_VTRACE_PAIR_ONLY",
            "FAIL CLOSED: CANDIDATE_SCOPE %r != RMT16_ORIGINAL_VTRACE_PAIR_ONLY"
            % scope.get("CANDIDATE_SCOPE"))
    require(sorted(scope.get("arms", [])) == ["persistent", "reset128"],
            "FAIL CLOSED: profile arms %r != [persistent, reset128]"
            % scope.get("arms"))
    require(scope.get("OVERALL_STRONG_STUDENT_SELECTION_AUTHORIZED") is False,
            "FAIL CLOSED: overall strong-student selection is NOT authorized")
    require(scope.get("SCIENTIFIC_SUPERIORITY_CLAIM") is False,
            "FAIL CLOSED: no scientific-superiority claim this round")

    scenarios = profile.get("scenarios")
    require(isinstance(scenarios, dict)
            and set(scenarios) == {metrics.FULL, metrics.FRONT, metrics.BACK},
            "FAIL CLOSED: profile scenarios %r != {full, front_l2, back_l2}"
            % sorted(scenarios or {}))

    # ---- FRONT / BACK frozen scaffold banks --------------------------------
    for scen, hash_label in ((metrics.FRONT, "FRONT_SCAFFOLD_STATE_BANK_HASH"),
                             (metrics.BACK, "BACK_SCAFFOLD_STATE_BANK_HASH")):
        s = scenarios[scen]
        require(s.get("kind") == "frozen_scaffold_bank",
                "FAIL CLOSED: %s kind %r != frozen_scaffold_bank"
                % (scen, s.get("kind")))
        require(_is_sha256_hex(s.get("bank_content_sha256")),
                "FAIL CLOSED: %s bank_content_sha256 %r is not 64-hex"
                % (scen, s.get("bank_content_sha256")))
        require(s["bank_content_sha256"] == mat.FROZEN_BANK_HASH[scen],
                "FAIL CLOSED: %s bank_content_sha256 %s != frozen %s"
                % (scen, s["bank_content_sha256"][:16],
                   mat.FROZEN_BANK_HASH[scen][:16]))
        require(s.get("bank_hash_label") == hash_label,
                "FAIL CLOSED: %s bank_hash_label %r != %r"
                % (scen, s.get("bank_hash_label"), hash_label))
        require(int(s.get("n", -1)) == mat.FROZEN_BANK_N,
                "FAIL CLOSED: %s n %r != frozen FROZEN_BANK_N %d"
                % (scen, s.get("n"), mat.FROZEN_BANK_N))
        require(int(s.get("seed_base", -1)) == mat.FROZEN_SEED_BASE
                and int(s.get("seed_stride", -1)) == mat.FROZEN_SEED_STRIDE,
                "FAIL CLOSED: %s seed_base/stride (%r, %r) != frozen (%d, %d)"
                % (scen, s.get("seed_base"), s.get("seed_stride"),
                   mat.FROZEN_SEED_BASE, mat.FROZEN_SEED_STRIDE))
        require(list(s.get("bank_indices", [])) == list(range(mat.FROZEN_BANK_N)),
                "FAIL CLOSED: %s bank_indices %r != full indices 0..%d"
                % (scen, s.get("bank_indices"), mat.FROZEN_BANK_N - 1))
        want_seeds = mat.fixed_seed_schedule(scen, mat.FROZEN_BANK_N,
                                             mat.FROZEN_SEED_BASE,
                                             mat.FROZEN_SEED_STRIDE)
        require([int(x) for x in s.get("seeds", [])] == want_seeds,
                "FAIL CLOSED: %s seeds %r != frozen schedule %r"
                % (scen, s.get("seeds"), want_seeds))
        require(s.get("can_replace_full_task") is False,
                "FAIL CLOSED: %s scaffold can NEVER replace the full task" % scen)

    fr = scenarios[metrics.FRONT]
    require(fr.get("primary_metric") == metrics.FRONT_PRIMARY_METRIC,
            "FAIL CLOSED: front primary %r != frozen %r"
            % (fr.get("primary_metric"), metrics.FRONT_PRIMARY_METRIC))
    require(fr.get("dense_metric") == metrics.FRONT_DENSE_METRIC,
            "FAIL CLOSED: front dense %r != frozen %r"
            % (fr.get("dense_metric"), metrics.FRONT_DENSE_METRIC))

    bk = scenarios[metrics.BACK]
    require(bk.get("primary_metric") == metrics.BACK_PRIMARY_METRIC,
            "FAIL CLOSED: back primary %r != frozen %r"
            % (bk.get("primary_metric"), metrics.BACK_PRIMARY_METRIC))
    require(bk.get("identity_class") == metrics.BACK_IDENTITY,
            "FAIL CLOSED: back identity %r != frozen %r"
            % (bk.get("identity_class"), metrics.BACK_IDENTITY))
    require(list(bk.get("na_metrics", [])) == list(metrics.BACK_NA_METRICS),
            "FAIL CLOSED: back na_metrics %r != frozen %r"
            % (bk.get("na_metrics"), metrics.BACK_NA_METRICS))
    require(bk.get("boss_search_claimed") is False,
            "FAIL CLOSED: BACK makes NO boss-search claim (conditional defeat only)")

    # ---- FULL canonical world / seed set -----------------------------------
    full = scenarios[metrics.FULL]
    require(full.get("kind") == "canonical_full_task",
            "FAIL CLOSED: full kind %r != canonical_full_task" % full.get("kind"))
    require(full.get("primary_metric") == metrics.FULL_PRIMARY_METRIC,
            "FAIL CLOSED: full primary %r != frozen %r"
            % (full.get("primary_metric"), metrics.FULL_PRIMARY_METRIC))
    require(full.get("dense_metric") is None,
            "FAIL CLOSED: FULL has no dense substitute metric")
    wss = full.get("world_seed_set")
    require(isinstance(wss, dict), "FAIL CLOSED: full has no world_seed_set object")
    require(wss.get("kind") == "canonical_reset_seeds_held_out",
            "FAIL CLOSED: full seed set kind %r != canonical_reset_seeds_held_out"
            % wss.get("kind"))
    require(int(wss.get("base", -1)) == ev.PERF_FULL_SEED_BASE
            and int(wss.get("count", -1)) == ev.PERF_FULL_N
            and int(wss.get("stride", -1)) == 1,
            "FAIL CLOSED: full seed set (base,count,stride) (%r,%r,%r) != frozen "
            "(%d,%d,1)" % (wss.get("base"), wss.get("count"), wss.get("stride"),
                           ev.PERF_FULL_SEED_BASE, ev.PERF_FULL_N))
    seeds_frozen = [int(x) for x in wss.get("seeds", [])] == list(ev.PERF_FULL_SEEDS)
    require(seeds_frozen,
            "FAIL CLOSED: full seeds != the COMMITTED held-out canonical reset seeds "
            "(tier3_evaluator.PERF_FULL_SEEDS 200000..200063). This profile may NOT "
            "carry seeds that lack existing canonical evidence.")
    # FULL_PROFILE_READY must be EXACTLY the truth value of the frozen-seed match:
    # never true without evidence, never a stale true after the schedule moves.
    require(full.get("FULL_PROFILE_READY") is seeds_frozen,
            "FAIL CLOSED: FULL_PROFILE_READY %r inconsistent with the frozen-seed "
            "match %r — before the FULL world/seed set is frozen and evidence-backed "
            "this field must be false and no formal FULL run is authorized"
            % (full.get("FULL_PROFILE_READY"), seeds_frozen))
    return profile


def load_and_verify(path: str = DEFAULT_PROFILE_PATH) -> dict:
    return verify_profile(load_profile(path))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def self_test() -> int:
    problems = []

    def check(name, cond):
        if not cond:
            problems.append(name)

    profile = load_and_verify()
    check("profile_loads_and_verifies", isinstance(profile, dict))
    check("self_hash_binds",
          profile[SELF_SHA_FIELD] == compute_profile_sha256(profile))
    check("full_profile_ready_true_with_evidence",
          profile["scenarios"]["full"]["FULL_PROFILE_READY"] is True)

    def tamper(mut):
        import copy
        bad = copy.deepcopy(profile)
        mut(bad)
        try:
            verify_profile(bad)
            return False
        except FailClosed:
            return True

    # Self-hash gate: a recomputed-hash mismatch fails closed at load time.
    import copy, json as _json, tempfile
    bad = copy.deepcopy(profile)
    bad["common_evaluation_invariants"]["max_timesteps"] = 1024
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8") as fh:
            _json.dump(bad, fh)
            tmppath = fh.name
        load_profile(tmppath)
        check("tampered_self_hash_rejected_at_load", False)
    except FailClosed:
        check("tampered_self_hash_rejected_at_load", True)
    finally:
        os.unlink(tmppath)

    # Protocol gates (each on a self-hash-consistent re-stamped copy).
    def stamped(mut):
        import copy
        bad = copy.deepcopy(profile)
        mut(bad)
        bad[SELF_SHA_FIELD] = compute_profile_sha256(bad)
        return bad

    check("shortened_horizon_rejected",
          tamper(lambda p: p["common_evaluation_invariants"]
                 .__setitem__("max_timesteps", 1024)))
    check("non_greedy_action_mode_rejected",
          tamper(lambda p: p["common_evaluation_invariants"]
                 .__setitem__("action_mode", "sample")))
    check("observation_shape_rejected",
          tamper(lambda p: p["common_evaluation_invariants"]
                 .__setitem__("observation_shape", [9999])))
    check("action_dim_rejected",
          tamper(lambda p: p["common_evaluation_invariants"]
                 .__setitem__("action_dim", 42)))
    check("front_bank_hash_rejected",
          tamper(lambda p: p["scenarios"]["front_l2"]
                 .__setitem__("bank_content_sha256", "a" * 64)))
    check("back_bank_hash_rejected",
          tamper(lambda p: p["scenarios"]["back_l2"]
                 .__setitem__("bank_content_sha256", "b" * 64)))
    check("dropped_bank_index_rejected",
          tamper(lambda p: p["scenarios"]["front_l2"]
                 .__setitem__("bank_indices", [0, 1, 2, 3, 4, 5, 6])))
    check("shifted_front_seed_rejected",
          tamper(lambda p: p["scenarios"]["front_l2"]
                 .__setitem__("seeds", [10001, 10002, 10003, 10004, 10005, 10006,
                                        10007, 10008])))
    check("back_primary_metric_rejected",
          tamper(lambda p: p["scenarios"]["back_l2"]
                 .__setitem__("primary_metric", "P_BOSS_AREA_REACHED")))
    check("back_boss_search_claim_rejected",
          tamper(lambda p: p["scenarios"]["back_l2"]
                 .__setitem__("boss_search_claimed", True)))
    check("full_new_seeds_rejected",
          tamper(lambda p: p["scenarios"]["full"]["world_seed_set"]
                 .__setitem__("seeds", list(range(300000, 300064)))))
    check("full_profile_ready_false_inconsistent",
          tamper(lambda p: p["scenarios"]["full"]
                 .__setitem__("FULL_PROFILE_READY", False)))
    check("selection_authorization_rejected",
          tamper(lambda p: p["candidate_scope"]
                 .__setitem__("OVERALL_STRONG_STUDENT_SELECTION_AUTHORIZED", True)))
    check("scientific_superiority_claim_rejected",
          tamper(lambda p: p["candidate_scope"]
                 .__setitem__("SCIENTIFIC_SUPERIORITY_CLAIM", True)))
    check("extra_arm_rejected",
          tamper(lambda p: p["candidate_scope"]["arms"].append("gtrxl_base")))

    # A stamped (self-hash-consistent) copy with a protocol violation must be
    # rejected by verify_profile, not by the self-hash gate.
    st = stamped(lambda p: p["common_evaluation_invariants"]
                 .__setitem__("max_timesteps", 1024))
    try:
        verify_profile(st)
        check("stamped_short_horizon_rejected", False)
    except FailClosed as exc:
        check("stamped_short_horizon_rejected", "max_timesteps" in str(exc))

    if problems:
        print("TIER3_EVALUATION_PROFILE_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_EVALUATION_PROFILE_SELF_TEST_PASS "
          "(profile frozen; FULL_PROFILE_READY=true; all tamper gates closed)")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return self_test()
    if "--verify" in argv:
        path = DEFAULT_PROFILE_PATH
        if "--profile" in argv:
            path = argv[argv.index("--profile") + 1]
        profile = load_and_verify(path)
        print("EVALUATION_PROFILE_VERIFIED sha256=%s FULL_PROFILE_READY=%s"
              % (profile[SELF_SHA_FIELD],
                 profile["scenarios"]["full"]["FULL_PROFILE_READY"]))
        return 0
    print("usage: tier3_evaluation_profile.py --self-test | --verify [--profile PATH]")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
