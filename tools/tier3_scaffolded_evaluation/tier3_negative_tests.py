#!/usr/bin/env python3
"""CC4 Tier3 — negative tests (§十七 NEG01–NEG29).

Each negative test constructs an INVALID input and asserts the corresponding guard
REJECTS it (fail-closed). A test PASSES when the rejection is correctly detected;
the suite requirement is FAIL=0 (no negative test silently accepts a violation).
BLOCKED is allowed only with a documented environment-capability absence — never a
fake PASS.

Coverage (all 29 implemented; FAIL=0 required):
  NEG01-NEG18  boundary / builder / state-bank / predicate level
  NEG19        episode missing valid_start (evaluator)
  NEG20        ambiguous termination silently labelled (failure taxonomy)
  NEG21-NEG23  checkpoint params SHA / observation shape / params update (adapter)
  NEG24        scaffold hash must never be the GLOBAL_WORLD_SET_HASH (materializer/cert)
  NEG25        scaffold result claims full-task success (certificate)
  NEG26        state/sample selection must be blind to Student performance (materializer)
  NEG27        certificate eval_binding must carry real VALUES, never labels (certificate)
  NEG28        tampered frozen bank manifest fails closed (materializer, pure compare)
  NEG29        certificate provenance (pid/argv/times/exit code/driver SHA) missing or
               invalid (certificate)
"""
from __future__ import annotations

import json
import os
import sys

# Runnable-as-script AND importable-as-package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_source_audit as audit                 # noqa: E402
import tier3_event_predicates as pred             # noqa: E402
import tier3_state_serializer as ser              # noqa: E402
import tier3_scaffold_builder as builder          # noqa: E402
import tier3_state_bank_materializer as mat       # noqa: E402
import tier3_checkpoint_adapter as ckpt          # noqa: E402
import tier3_metrics as metrics                   # noqa: E402
import tier3_failure_taxonomy as taxonomy         # noqa: E402
import tier3_evaluator as evaluator              # noqa: E402
import tier3_evaluation_certificate as certmod    # noqa: E402

# Every guard may raise its own module's FailClosed (or the reused V3 one).
FAILCLOSED = (audit.FailClosed, pred.FailClosed, ser.FailClosed, builder.FailClosed,
              mat.FailClosed, ckpt.FailClosed, metrics.FailClosed, taxonomy.FailClosed,
              evaluator.FailClosed, certmod.FailClosed, ser.v3mat.FailClosed)

KOBOLD = mat.SYNTHETIC_KOBOLD_TYPE_ID   # == resolved craftax==1.4.5 binding (RANGED type_id 3)


def rejects(fn) -> bool:
    """True iff fn() raises a FailClosed (the guard correctly rejected the input)."""
    try:
        fn()
        return False
    except FAILCLOSED:
        return True


# ---------------------------------------------------------------------------
# Synthetic states (identical to the materializer's; clearly test-only)
# ---------------------------------------------------------------------------
def front_state(**over):
    s = mat.synthesize_front_start(0)
    s.update(over)
    return s


def back_state(**over):
    s = mat.synthesize_back_start(0, KOBOLD)
    s.update(over)
    return s


# ---------------------------------------------------------------------------
# NEG01–NEG18, NEG24, NEG26
# ---------------------------------------------------------------------------
def neg01():
    """Boundary source SHA mismatch vs the real on-disk source -> fail."""
    role = "world_builder"
    real_sha = audit.sha256_file(audit.resolve_source_path(role))

    def check(claimed):
        if claimed != real_sha:
            raise audit.FailClosed(
                "FAIL CLOSED (NEG01): boundary source_file_sha256 %s != on-disk %s"
                % (claimed[:16], real_sha[:16]))
    # correct SHA passes; tampered SHA rejected
    check(real_sha)
    return rejects(lambda: check("0" * 64))


def neg02():
    """Builder source realpath/SHA mismatch -> fail (V3 executed-source identity)."""
    other = str(audit.resolve_source_path("game_mechanics"))  # a DIFFERENT real file
    return rejects(lambda: builder.bind_builder_source_identity(imported_file=other))


def neg03():
    """Canonical task source SHA mismatch -> fail."""
    return rejects(lambda: builder.verify_canonical_task_source(expected_sha256="0" * 64))


def neg04():
    """Missing required EnvState field -> fail."""
    bad = front_state()
    del bad["player_level"]
    return rejects(lambda: ser.assert_required_envstate_fields(bad))


def neg05():
    """State payload hash tampered -> fail (hash compare; bytes are opaque)."""
    sha, payload = ser.normalized_payload_hash(front_state())
    return rejects(lambda: ser.verify_payload_hash("0" * 64, payload))


def neg06():
    """State bank order changed -> hash changes (order-sensitive)."""
    m = mat.process_a_materialize(mat.FRONT, 5)
    hashes = [e["state_payload_hash"] for e in m["entries"]]
    src = mat.source_shas_for_bank()
    h_fwd = mat.state_bank_hash(hashes, mat.FRONT, src)
    h_rev = mat.state_bank_hash(list(reversed(hashes)), mat.FRONT, src)
    return h_fwd != h_rev


def neg07():
    """Persistent vs Reset128 using different (per-arm) state banks -> fail."""
    m = mat.process_a_materialize(mat.FRONT, 3)
    m["per_arm"] = {"persistent": "bankA", "reset128": "bankB"}
    return rejects(lambda: mat.assert_no_arm_partition(m))


def neg08():
    """Arm-specific scaffold metadata -> fail."""
    spec = builder.build_front_spec()
    spec["arm_id"] = "persistent"
    return rejects(lambda: builder.assert_no_arm_specific_metadata(spec))


def neg09():
    """Extra observation field / changed observation schema -> fail."""
    spec = builder.build_front_spec()
    spec["observation_schema"] = "canonical_craftax_symbolic + exit_direction_arrow"
    return rejects(lambda: builder.validate_scaffold_legality(spec))


def neg10():
    """Action space changed -> fail."""
    spec = builder.build_back_spec()
    spec["action_space"] = "reduced_discrete_8"
    return rejects(lambda: builder.validate_scaffold_legality(spec))


def neg11():
    """Hidden boss direction injected into observation -> fail."""
    spec = builder.build_back_spec()
    spec["legality"]["no_hidden_boss_direction"] = False
    return rejects(lambda: builder.validate_scaffold_legality(spec))


def neg12():
    """Invalid inventory value (negative / non-int) -> fail."""
    return (rejects(lambda: builder.validate_inventory({"wood": -1}))
            and rejects(lambda: builder.validate_inventory({"wood": 1.5}))
            and rejects(lambda: builder.validate_inventory({"": 3})))


def neg13():
    """Invalid player position (negative / outside grid) -> fail."""
    return (rejects(lambda: builder.validate_player_position((-1, 3), 48, 48))
            and rejects(lambda: builder.validate_player_position((5, 99), 48, 48)))


def neg14():
    """Front start already beyond the corridor exit -> invalid scaffold."""
    beyond = front_state(player_level=audit.CORRIDOR_EXIT_FLOOR)
    return pred.valid_front_scaffold_start(beyond) is False


def neg15():
    """Back start already has DEFEAT_KOBOLD -> invalid scaffold."""
    solved = back_state(achieved={pred.DEFEAT_KOBOLD})
    return pred.valid_back_scaffold_start(solved, KOBOLD) is False


def neg16():
    """Back state with no live Kobold (kill task requires one) -> invalid scaffold."""
    no_kobold = back_state(mobs=[])
    return pred.valid_back_scaffold_start(no_kobold, KOBOLD) is False


def neg17():
    """Progress must always lie in [0,1] (sweep every reachable cell)."""
    walk = [[False] * 5 for _ in range(5)]
    for c in range(5):
        walk[2][c] = True
    walk[1][2] = True                      # a dead-end spur
    start, exit_pos = (2, 0), (2, 4)
    ok = True
    for r in range(5):
        for c in range(5):
            if not walk[r][c]:
                continue
            p = pred.normalized_corridor_progress(
                front_state(player_position=(r, c)), walk, start, exit_pos)
            ok = ok and (0.0 <= p <= 1.0)
    return ok


def neg18():
    """Unreachable exit without an explicit blocked label -> fail."""
    walk = [[False] * 5 for _ in range(5)]
    walk[2][0] = True
    walk[2][1] = True                       # exit (2,4) isolated
    return rejects(lambda: pred.normalized_corridor_progress(
        front_state(player_position=(2, 0)), walk, (2, 0), (2, 4)))


def neg24():
    """Scaffold bank hash labelled as GLOBAL_WORLD_SET_HASH -> fail."""
    m = mat.process_a_materialize(mat.FRONT, 3)
    m["hash_label"] = mat.GLOBAL_HASH_LABEL
    return rejects(lambda: mat.assert_not_global_world_set_hash(m))


def neg26():
    """State/sample selection based on Student performance -> fail."""
    m = mat.process_a_materialize(mat.FRONT, 3)
    m["seeds"] = [s + 1 for s in m["seeds"]]   # no longer reproducible from schedule
    return rejects(lambda: mat.assert_selection_is_result_blind(m))


def neg19():
    """Episode record missing the valid_start flag -> fail."""
    ep = {"episode_id": "e", "scenario": mat.FRONT, "terminal_label": "",
          "corridor_exit_reached": True, "defeat_kobold": False, "timesteps": 5}
    # no valid_start key
    return rejects(lambda: evaluator.validate_episode_record(ep))


def neg20():
    """Ambiguous/contradictory termination silently assigned one label -> fail."""
    ep = {"scenario": mat.FRONT, "valid_start": True, "defeat_kobold": True,
          "player_died": True, "timed_out": False, "corridor_exit_reached": True,
          "kobold_engaged": False, "boss_area_reached": False}
    return rejects(lambda: taxonomy.classify_episode(ep))


def neg21():
    """Checkpoint params SHA mismatch -> fail."""
    rec = ckpt.make_checkpoint_record({"w": [1, 2]}, (67, 7, 7), "canonical_craftax_action_set")
    return rejects(lambda: ckpt.assert_params_identity(rec, "0" * 64))


def neg22():
    """Checkpoint observation shape mismatch -> fail."""
    rec = ckpt.make_checkpoint_record({"w": [1, 2]}, (67, 7, 7), "canonical_craftax_action_set")
    return rejects(lambda: ckpt.assert_observation_shape(rec, (68, 7, 7)))


def neg23():
    """Evaluation tries to update params -> fail."""
    rec = ckpt.make_checkpoint_record({"w": [1, 2]}, (67, 7, 7), "canonical_craftax_action_set")
    mutated = dict(rec)
    mutated["params_sha256"] = ckpt.params_sha256({"w": [9, 9]})
    return rejects(lambda: ckpt.assert_evaluation_does_not_update_params(rec, mutated))


def neg25():
    """Scaffold result claims full-task success / breakthrough -> fail."""
    claims = ["TIER3_FRONT_HALF_BREAKTHROUGH"]
    result = {
        "scenario": mat.FRONT,
        "contract": {"observation_schema": "canonical_craftax_symbolic"},
        "metrics": {"primary": {"metric": metrics.FRONT_PRIMARY_METRIC, "value": 0.5,
                                "valid_starts": 4}},
        "failure_rule_version": taxonomy.FAILURE_RULE_VERSION,
        "terminal_label_counts": {},
        "rollout_status": "BLOCKED_ENVIRONMENT",
    }
    return rejects(lambda: certmod.build_certificate(result, claims=claims))


def neg27():
    """Certificate eval_binding with a hash LABEL / missing value instead of a real
    64-hex SHA value (or wrong interface / params changed) -> fail."""
    result = {
        "scenario": mat.FRONT,
        "contract": {"observation_schema": "canonical_craftax_symbolic"},
        "metrics": {"primary": {"metric": metrics.FRONT_PRIMARY_METRIC, "value": 0.5,
                                "valid_starts": 4}},
        "failure_rule_version": taxonomy.FAILURE_RULE_VERSION,
        "terminal_label_counts": {},
        "rollout_status": "BLOCKED_ENVIRONMENT",
    }
    binding = {
        "state_bank_hash": "FRONT_SCAFFOLD_STATE_BANK_HASH",   # a LABEL, not a SHA value
        "state_payload_hashes": ["a" * 64],
        "checkpoint_file_sha256": "b" * 64,
        "cc2_params_sha256": "c" * 64,
        "checkpoint_step": 4096,
        "carry_mode": "persistent",
        "run_class": "INTERFACE_SMOKE",
        "episode_records_sha256": "d" * 64,
        "cc2_policy_source_sha256": "e" * 64,
        "evaluator_source_sha256": "f" * 64,
        "predicate_code_sha256": "0" * 64,
        "observation_shape": [8335],
        "action_dim": 43,
        "params_unchanged": True,
        "performance_claim_authorized": False,
    }
    label_rejected = rejects(
        lambda: certmod.build_certificate(result, eval_binding=dict(binding)))
    empty = dict(binding)
    empty["state_bank_hash"] = "a" * 64
    empty["checkpoint_file_sha256"] = None                       # missing value
    missing_rejected = rejects(
        lambda: certmod.build_certificate(result, eval_binding=empty))
    changed = dict(binding)
    changed["state_bank_hash"] = "a" * 64
    changed["params_unchanged"] = False                          # params mutated
    changed_rejected = rejects(
        lambda: certmod.build_certificate(result, eval_binding=changed))
    return label_rejected and missing_rejected and changed_rejected


def neg29():
    """Certificate eval_binding with missing / invalid PROCESS PROVENANCE (actual pid /
    argv / start-end UTC / exit code) or driver-source SHA -> fail closed.

    A complete provenance binding is accepted; ten tamper paths (bad/missing pid,
    empty argv, empty argv element, unparseable/empty timestamps, non-zero/missing
    exit code, non-hex/missing driver SHA) are each rejected."""
    result = {
        "scenario": mat.FRONT,
        "contract": {"observation_schema": "canonical_craftax_symbolic"},
        "metrics": {"primary": {"metric": metrics.FRONT_PRIMARY_METRIC, "value": 0.5,
                                "valid_starts": 4}},
        "failure_rule_version": taxonomy.FAILURE_RULE_VERSION,
        "terminal_label_counts": {},
        "rollout_status": "BLOCKED_ENVIRONMENT",
    }
    binding = {
        "state_bank_hash": "a" * 64,
        "state_payload_hashes": ["a" * 64],
        "checkpoint_file_sha256": "b" * 64,
        "cc2_params_sha256": "c" * 64,
        "checkpoint_step": 98304,
        "carry_mode": "persistent",
        "run_class": "INTERFACE_SMOKE",
        "episode_records_sha256": "d" * 64,
        "cc2_policy_source_sha256": "e" * 64,
        "evaluator_source_sha256": "f" * 64,
        "predicate_code_sha256": "0" * 64,
        "driver_source_sha256": "9" * 64,
        "process_pid": 4242,
        "process_argv": ["python", "tier3_evaluator.py", "--interface-smoke"],
        "run_start_utc": "2026-07-30T00:00:00+00:00",
        "run_end_utc": "2026-07-30T00:05:00+00:00",
        "run_exit_code": 0,
        "observation_shape": [8335],
        "action_dim": 43,
        "params_unchanged": True,
        "performance_claim_authorized": False,
    }
    complete_accepted = not rejects(
        lambda: certmod.build_certificate(result, eval_binding=dict(binding)))
    tamper_results = []
    for over in ({"process_pid": None}, {"process_pid": 0},
                 {"process_argv": []}, {"process_argv": ["python", ""]},
                 {"run_start_utc": "yesterday"}, {"run_end_utc": None},
                 {"run_exit_code": 137}, {"run_exit_code": None},
                 {"driver_source_sha256": "not-a-sha"},
                 {"driver_source_sha256": None}):
        b = dict(binding)
        b.update(over)
        tamper_results.append(rejects(
            lambda b=b: certmod.build_certificate(result, eval_binding=b)))
    return complete_accepted and all(tamper_results)


def _fake_real_manifest(scenario, n=8):
    """A manifest shaped EXACTLY like a REAL bank with every frozen binding correct
    EXCEPT the per-entry payload hashes (which are fabricated 64-hex values, so the
    order-sensitive bank-hash recomputation cannot match the frozen hash). Pure /
    host-independent; used by NEG28."""
    seeds = mat.fixed_seed_schedule(scenario, n, mat.FROZEN_SEED_BASE, mat.FROZEN_SEED_STRIDE)
    entries = [{"index": i, "seed": int(seeds[i]),
                "state_payload_hash": ("%064x" % (i + 1)),
                "field_manifest_sha256": mat.FROZEN_FIELD_MANIFEST_SHA256,
                "synthetic": False} for i in range(n)]
    return {
        "schema": mat.SCHEMA,
        "scenario": scenario,
        "hash_label": mat.HASH_LABELS[scenario],
        "hash_status": "MATERIALIZED",
        "states_are": "REAL_ENVSTATE",
        "state_count": n,
        "seed_schedule_params": {"scenario": scenario, "n": n,
                                 "seed_base": mat.FROZEN_SEED_BASE,
                                 "stride": mat.FROZEN_SEED_STRIDE},
        "seeds": seeds,
        "source_shas": mat.source_shas_for_bank(),
        "boundary_predicate_version": pred.PREDICATE_VERSION,
        "field_manifest_sha256": mat.FROZEN_FIELD_MANIFEST_SHA256,
        "state_bank_hash": mat.FROZEN_BANK_HASH[scenario],
        "entries": entries,
    }


def neg28():
    """Tampered frozen bank manifest -> fail closed (pure comparison; any host).

    Three independent tamper paths: (a) fabricated per-entry payload hashes cannot
    recompute the frozen bank hash (order-sensitive), (b) a declared state_bank_hash
    different from the frozen value is rejected, (c) a shifted seed schedule breaks
    the result-blind binding."""
    m = _fake_real_manifest(mat.FRONT)
    payload_tamper_rejected = rejects(lambda: mat.check_frozen_manifest_bindings(mat.FRONT, m))
    m2 = _fake_real_manifest(mat.FRONT)
    m2["state_bank_hash"] = "0" * 64
    bankhash_tamper_rejected = rejects(lambda: mat.check_frozen_manifest_bindings(mat.FRONT, m2))
    m3 = _fake_real_manifest(mat.BACK)
    m3["seeds"] = [s + 1 for s in m3["seeds"]]
    seed_tamper_rejected = rejects(lambda: mat.check_frozen_manifest_bindings(mat.BACK, m3))
    return payload_tamper_rejected and bankhash_tamper_rejected and seed_tamper_rejected


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
NEG_TESTS = [
    ("NEG01", "boundary source SHA mismatch", neg01),
    ("NEG02", "builder source realpath/SHA mismatch", neg02),
    ("NEG03", "canonical task source SHA mismatch", neg03),
    ("NEG04", "missing required EnvState field", neg04),
    ("NEG05", "state payload hash tampered", neg05),
    ("NEG06", "state bank order changed -> hash changes", neg06),
    ("NEG07", "per-arm (Persistent vs Reset128) state bank", neg07),
    ("NEG08", "arm-specific scaffold metadata", neg08),
    ("NEG09", "extra observation field / schema change", neg09),
    ("NEG10", "action space changed", neg10),
    ("NEG11", "hidden boss direction injected", neg11),
    ("NEG12", "invalid inventory value", neg12),
    ("NEG13", "invalid player position", neg13),
    ("NEG14", "front start already beyond exit", neg14),
    ("NEG15", "back start already DEFEAT_KOBOLD", neg15),
    ("NEG16", "back state has no live Kobold", neg16),
    ("NEG17", "progress always within [0,1]", neg17),
    ("NEG18", "unreachable exit without blocked label", neg18),
    ("NEG19", "episode missing valid_start", neg19),
    ("NEG20", "ambiguous termination silently labelled", neg20),
    ("NEG21", "checkpoint params SHA mismatch", neg21),
    ("NEG22", "checkpoint observation shape mismatch", neg22),
    ("NEG23", "evaluation tries to update params", neg23),
    ("NEG24", "scaffold hash used as GLOBAL_WORLD_SET_HASH", neg24),
    ("NEG25", "scaffold result claims full-task success", neg25),
    ("NEG26", "result-based state/sample selection", neg26),
    ("NEG27", "certificate eval_binding label / missing value / params changed", neg27),
    ("NEG28", "tampered frozen bank manifest (payload / hash / seeds)", neg28),
    ("NEG29", "certificate provenance missing/invalid (pid/argv/times/exit/driver SHA)",
     neg29),
]

# All 29 NEG tests are implemented (NEG19-23/25 landed with the Commit-3 modules).
PENDING_COMMIT_3 = []


def run_all():
    results = []
    n_fail = 0
    for neg_id, desc, fn in NEG_TESTS:
        try:
            ok = bool(fn())
        except Exception as exc:           # unexpected error == a real failure
            ok = False
            desc = "%s (unexpected exception: %r)" % (desc, exc)
        if not ok:
            n_fail += 1
        results.append({"id": neg_id, "description": desc, "rejected_correctly": ok,
                        "status": "PASS" if ok else "FAIL"})
    return results, n_fail


def self_test() -> int:
    results, n_fail = run_all()
    implemented = len(NEG_TESTS)
    pending = len(PENDING_COMMIT_3)
    for r in results:
        print("  [%s] %s - %s" % (r["status"], r["id"], r["description"]))
    for neg_id, desc, owner in PENDING_COMMIT_3:
        print("  [PENDING] %s - %s (lands with %s in Commit 3)" % (neg_id, desc, owner))
    if n_fail != 0:
        print("TIER3_NEGATIVE_TESTS_FAIL (FAIL=%d/%d implemented)" % (n_fail, implemented))
        return 1
    print("TIER3_NEGATIVE_TESTS_PASS (FAIL=0; implemented=%d/29, pending_commit3=%d)"
          % (implemented, pending))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # On a JAX host the REAL materializer path mints live EnvStates, which imports
    # minicraftax from the audited source tree (repo-relative, no absolute paths).
    _src = str(audit.repo_root() / "dicode_src" / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)
    if "--json" in argv:
        results, n_fail = run_all()
        print(json.dumps({"fail": n_fail, "results": results,
                          "pending_commit3": [p[0] for p in PENDING_COMMIT_3]},
                         indent=2, ensure_ascii=False, sort_keys=True))
        return 0 if n_fail == 0 else 1
    return self_test()


if __name__ == "__main__":
    raise SystemExit(main())
