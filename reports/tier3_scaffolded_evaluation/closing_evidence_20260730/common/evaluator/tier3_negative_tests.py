#!/usr/bin/env python3
"""CC4 Tier3 — negative tests (§十七 NEG01–NEG49).

Each negative test constructs an INVALID input and asserts the corresponding guard
REJECTS it (fail-closed). A test PASSES when the rejection is correctly detected;
the suite requirement is FAIL=0 (no negative test silently accepts a violation).
BLOCKED is allowed only with a documented environment-capability absence — never a
fake PASS.

Coverage (all 49 implemented; FAIL=0 required):
  NEG01-NEG18  boundary / builder / state-bank / predicate level
  NEG19        episode missing valid_start (evaluator)
  NEG20        ambiguous termination silently labelled (failure taxonomy)
  NEG21-NEG23  checkpoint params SHA / observation shape / params update (adapter)
  NEG24        scaffold hash must never be the GLOBAL_WORLD_SET_HASH (materializer/cert)
  NEG25        scaffold result claims full-task success (certificate)
  NEG26        state/sample selection must be blind to Student performance (materializer)
  NEG27        certificate eval_binding must carry real VALUES, never labels (certificate)
  NEG28        tampered frozen bank manifest fails closed (materializer, pure compare)
  NEG29        finalized certificate runner provenance (child pid/argv/times/literal
               exit code/exit_source/runner SHA) missing or invalid (certificate)
  NEG30        wrong final checkpoint FILE SHA vs the frozen contract (contract)
  NEG31        wrong final PARAMS SHA vs the frozen contract (contract)
  NEG32        step-8192 checkpoint impersonating final 98304 (contract)
  NEG33        wrong arm name / carry_mode vs the contract arm (contract)
  NEG34        wrong replay_mode vs the frozen contract (contract)
  NEG35        wrong seed / run_class vs the frozen contract (contract)
  NEG36        wrong base_checkpoint_params_sha256 vs the frozen contract (contract)
  NEG37        self-declared / legacy exit code never accepted (certificate/runner)
  NEG38        engine cert without runner literal provenance can never finalize (runner)
  NEG39        non-empty / file output dir rejected; fresh dir accepted (evaluator)
  NEG40        split Student status labels correct; old ambiguous key gone (certificate)
  NEG41        both arms' frozen start schedules identical; drift rejected (evaluator)
  NEG42        carry-mode comparison rule fixed/recomputable + fixed scope fields + forbidden selection vocabulary rejected (selection)
  NEG43        ROUND1 screening schedule must be a declared subset of the frozen
               schedule (FULL prefix / FRONT/BACK bank indices); shifts rejected (总控 §四)
  NEG44        ROUND1 screening must embed the frozen PARENT schedule it screens from;
               dropped / tampered parent rejected (总控 §四)
  NEG45        ROUND1 bank_indices strictly ascending / unique / in [0,8); seeds must
               be the frozen bank states at exactly those indices (总控 §四)
  NEG46        scaffold bank provenance honesty: formal class with an in-memory
               regenerated bank rejected; loaded content != frozen bank hash rejected;
               regeneration=true + artifact source rejected; unknown bank_source
               rejected (总控 §一.7/§一.8)
  NEG47        bank_kind / bank_source cross-binding: canonical reset seeds may not
               masquerade as an artifact source (or vice versa); unknown bank_kind
               rejected (总控 §一.8)
  NEG48        artifact SHAs must be 64-hex where required and null where forbidden
               (canonical reset seeds / in-memory banks carry none) (总控 §一.8)
  NEG49        bank_source / device_provenance must be bound; empty provenance or a
               missing 'mint'/'load' identity dict rejected (总控 §一.8)
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
import tier3_checkpoint_contract as contractmod   # noqa: E402
import tier3_evaluation_runner as runnermod       # noqa: E402
import tier3_provisional_selection as selection   # noqa: E402

# Every guard may raise its own module's FailClosed (or the reused V3 one).
FAILCLOSED = (audit.FailClosed, pred.FailClosed, ser.FailClosed, builder.FailClosed,
              mat.FailClosed, ckpt.FailClosed, metrics.FailClosed, taxonomy.FailClosed,
              evaluator.FailClosed, certmod.FailClosed, ser.v3mat.FailClosed,
              contractmod.FailClosed, runnermod.FailClosed, selection.FailClosed)

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


def _front_result():
    return {
        "scenario": mat.FRONT,
        "contract": {"observation_schema": "canonical_craftax_symbolic"},
        "metrics": {"primary": {"metric": metrics.FRONT_PRIMARY_METRIC, "value": 0.5,
                                "valid_starts": 4}},
        "failure_rule_version": taxonomy.FAILURE_RULE_VERSION,
        "terminal_label_counts": {},
        "rollout_status": "BLOCKED_ENVIRONMENT",
    }


def _engine_binding(**over):
    """A complete ENGINE-STAGE binding (task §一/§五): every frozen field present,
    NO exit provenance (the engine cannot know its own literal exit code)."""
    b = {
        "state_bank_hash": "2" + "a" * 63,
        "state_payload_hashes": ["b" * 64, "c" * 64],
        "checkpoint_file_sha256": "d" * 64,
        "cc2_params_sha256": "e" * 64,
        "checkpoint_step": 98304,
        "carry_mode": "persistent",
        "run_class": "INTERFACE_SMOKE",
        "episode_records_sha256": "f" * 64,
        "cc2_policy_source_sha256": "0" * 64,
        "evaluator_source_sha256": "1" * 64,
        "predicate_code_sha256": "a4fba86b054d20412fc1df2c79e7000d66b0525d"
                                 "ecb1801fa474ee7fb0d25b4c",
        "observation_shape": [8335],
        "action_dim": 43,
        "params_unchanged": True,
        "performance_claim_authorized": False,
        "driver_source_sha256": "9" * 64,
        "checkpoint_contract_sha256": "7" * 64,
        "checkpoint_contract_arm": "persistent",
        "action_mode": "greedy_argmax",
        "max_timesteps": 32,
        "evaluation_seed_schedule": {
            metrics.FULL: {"kind": "canonical_reset_seeds_smoke", "base": 42,
                           "count": 2, "seeds": [42, 43]},
            metrics.FRONT: {"kind": "frozen_bank_state_smoke", "seed_base": 10000,
                            "stride": 1, "count": 2, "seeds": [10000, 10001]},
            metrics.BACK: {"kind": "frozen_bank_state_smoke", "seed_base": 10000,
                           "stride": 1, "count": 2, "seeds": [1010000, 1010001]},
        },
        "state_entry_ids": {metrics.FULL: ["full-seed42", "full-seed43"],
                            metrics.FRONT: ["front_l2-bank0", "front_l2-bank1"],
                            metrics.BACK: ["back_l2-bank0", "back_l2-bank1"]},
        "python_version": "3.11.9",
        "jax_version": "0.4.30",
        "jaxlib_version": "0.4.30",
        "numpy_version": "1.26.4",
        "flax_version": "0.8.5",
        "craftax_version": "1.4.5",
        "evaluator_git_commit": "f67675b87ad98b391f82678bc2f937ab30578145",
        "scientific_claim_authorized": False,
        "single_training_seed": True,
        "provisional_selection_only": True,
        # 总控 §一.8 bank provenance. The default helper is an INTERFACE_SMOKE
        # binding, whose scaffold bank may (only here) be re-minted in memory —
        # every FORMAL-class NEG binding overrides this with _ARTIFACT_PROV.
        "bank_kind": "FROZEN_SCAFFOLD_BANK",
        "bank_source": "REGENERATED_IN_MEMORY",
        "bank_regenerated_on_eval_device": True,
        "artifact_file_sha256": None,
        "loaded_content_sha256": None,
        "device_provenance": {
            "mint": {"python_version": "3.11.9", "jax_version": "0.4.30",
                     "jaxlib_version": "0.4.30", "numpy_version": "1.26.4",
                     "flax_version": "0.8.5", "craftax_version": "1.4.5",
                     "jax_default_backend": "cpu"},
            "load": {"python_version": "3.11.9", "jax_version": "0.4.30",
                     "jaxlib_version": "0.4.30", "numpy_version": "1.26.4",
                     "flax_version": "0.8.5", "craftax_version": "1.4.5",
                     "jax_default_backend": "cpu"},
        },
    }
    b.update(over)
    return b


# 总控 §一.8: the FROZEN_SERIALIZED_ARTIFACT provenance every formal-class NEG
# binding must carry. loaded_content_sha256 equals the helper's bound
# state_bank_hash ("2" + "a"*63) — exactly what the artifact loader produces
# when the loaded states canonicalize to the frozen bank hash.
_ARTIFACT_PROV = {
    "bank_source": "FROZEN_SERIALIZED_ARTIFACT",
    "bank_regenerated_on_eval_device": False,
    "artifact_file_sha256": "a" * 64,
    "loaded_content_sha256": "2" + "a" * 63,
    "device_provenance": {
        "mint": {"python_version": "3.11.15", "jax_version": "0.4.30",
                 "jaxlib_version": "0.4.30", "numpy_version": "1.26.4",
                 "flax_version": "0.8.5", "craftax_version": "1.4.5",
                 "jax_default_backend": "cpu"},
        "load": {"python_version": "3.11.15", "jax_version": "0.4.30",
                 "jaxlib_version": "0.4.30", "numpy_version": "1.26.4",
                 "flax_version": "0.8.5", "craftax_version": "1.4.5",
                 "jax_default_backend": "gpu"},
    },
}


def _exit_binding(**over):
    """The RUNNER-SUPPLIED provenance (task §二) — literal wait() exit code only."""
    p = {
        "child_process_pid": 4242,
        "child_process_argv": ["python", "-u",
                               "tools/tier3_scaffolded_evaluation/tier3_evaluator.py",
                               "--performance-evaluation"],
        "actual_started_at_utc": "2026-07-30T00:00:00+00:00",
        "actual_finished_at_utc": "2026-07-30T01:00:00+00:00",
        "literal_exit_code": 0,
        "exit_source": "wait_pid",
        "inferred_from_log": False,
        "evaluation_runner_source_sha256": "8" * 64,
    }
    p.update(over)
    return p


def _prov_entry_ids():
    return {metrics.FULL: ["full-seed%d" % (200000 + i) for i in range(64)],
            metrics.FRONT: ["front_l2-bank%d" % i for i in range(8)],
            metrics.BACK: ["back_l2-bank%d" % i for i in range(8)]}


def neg27():
    """Certificate eval_binding with a hash LABEL / missing value instead of a real
    64-hex SHA value (or wrong interface / params changed / wrong frozen action
    identity) -> fail."""
    label_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_engine_binding(state_bank_hash="FRONT_SCAFFOLD_STATE_BANK_HASH")))
    missing_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_engine_binding(checkpoint_file_sha256=None)))
    changed_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_engine_binding(params_unchanged=False)))
    contract_label_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_engine_binding(checkpoint_contract_sha256="contract-sha-label")))
    action_mode_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_engine_binding(action_mode="sampling")))
    complete_ok = not rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_engine_binding()))
    return (label_rejected and missing_rejected and changed_rejected
            and contract_label_rejected and action_mode_rejected and complete_ok)


def neg29():
    """FINALIZED certificate with missing / invalid RUNNER PROVENANCE (child pid /
    argv / actual start-finish UTC / literal exit code / exit_source / runner source
    SHA) -> fail closed (task §二).

    A complete runner-finalized binding is accepted; every tamper path (non-zero or
    missing literal exit code, non-wait_pid exit_source, inferred-from-log, bad pid,
    empty argv, unparseable/empty timestamps, non-hex runner SHA) is rejected."""
    def build(**over):
        b = _engine_binding()
        b.update(_exit_binding())
        b.update(over)
        return certmod.build_certificate(
            _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
            eval_binding=b, finalized=True)
    complete_accepted = not rejects(build)
    tamper_results = [
        rejects(lambda: build(literal_exit_code=137)),
        rejects(lambda: build(literal_exit_code=None)),
        rejects(lambda: build(exit_source="self_declared")),
        rejects(lambda: build(exit_source="log_inferred")),
        rejects(lambda: build(inferred_from_log=True)),
        rejects(lambda: build(child_process_pid=0)),
        rejects(lambda: build(child_process_pid=None)),
        rejects(lambda: build(child_process_argv=[])),
        rejects(lambda: build(child_process_argv=["python", ""])),
        rejects(lambda: build(actual_started_at_utc="yesterday")),
        rejects(lambda: build(actual_finished_at_utc="")),
        rejects(lambda: build(evaluation_runner_source_sha256="not-a-sha")),
        rejects(lambda: build(evaluation_runner_source_sha256=None)),
    ]
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
# NEG30–NEG42 (task §一/§二/§四/§七/§九/§十二): frozen final-98304 checkpoint
# contract, literal exit provenance, output freshness, split Student labels,
# identical arm schedules, and the recomputable provisional selection rule.
# ---------------------------------------------------------------------------
def _contract_mismatch_rejects(fn) -> bool:
    """True iff fn() raises the stable FINAL_98304_CHECKPOINT_CONTRACT_MISMATCH id."""
    try:
        fn()
        return False
    except contractmod.FailClosed as exc:
        return "FINAL_98304_CHECKPOINT_CONTRACT_MISMATCH" in str(exc)


def _contract_run(**kw):
    """Verify a (possibly tampered) loaded checkpoint against the committed frozen
    contract's persistent arm."""
    file_sha = kw.get("file_sha", contractmod.FROZEN_CHECKPOINT_FILE_SHA256["persistent"])
    params_sha = kw.get("params_sha", contractmod.FROZEN_PARAMS_SHA256["persistent"])
    manifest = kw.get("manifest", contractmod._synthetic_manifest())
    driver = kw.get("driver_sha", contractmod.FROZEN_DRIVER_SOURCE_SHA256)
    policy = kw.get("policy_sha", contractmod.FROZEN_CC2_POLICY_SOURCE_SHA256)
    return contractmod.verify_checkpoint_against_contract(
        kw.get("arm", "persistent"), file_sha, params_sha, manifest, driver, policy)


def neg30():
    """Wrong final checkpoint FILE SHA vs the frozen contract -> mismatch id."""
    return _contract_mismatch_rejects(lambda: _contract_run(file_sha="0" * 64))


def neg31():
    """Wrong final PARAMS SHA vs the frozen contract -> mismatch id."""
    return _contract_mismatch_rejects(lambda: _contract_run(params_sha="1" * 64))


def neg32():
    """A step-8192 checkpoint impersonating the final 98304 -> mismatch id."""
    return _contract_mismatch_rejects(
        lambda: _contract_run(manifest=contractmod._synthetic_manifest(step=8192)))


def neg33():
    """Wrong arm name OR wrong carry_mode vs the contract arm -> mismatch id."""
    arm_rejected = _contract_mismatch_rejects(
        lambda: _contract_run(manifest=contractmod._synthetic_manifest(
            arm="RMT16-Evil-Arm")))
    carry_rejected = _contract_mismatch_rejects(
        lambda: _contract_run(manifest=contractmod._synthetic_manifest(
            carry_mode="reset128")))
    # the reset128 file SHA must never verify against the persistent arm entry
    cross_arm_rejected = _contract_mismatch_rejects(
        lambda: _contract_run(
            file_sha=contractmod.FROZEN_CHECKPOINT_FILE_SHA256["reset128"]))
    return arm_rejected and carry_rejected and cross_arm_rejected


def neg34():
    """Wrong replay_mode vs the frozen contract -> mismatch id."""
    return _contract_mismatch_rejects(
        lambda: _contract_run(manifest=contractmod._synthetic_manifest(
            replay_mode="replay")))


def neg35():
    """Wrong seed OR wrong run_class vs the frozen contract -> mismatch id."""
    seed_rejected = _contract_mismatch_rejects(
        lambda: _contract_run(manifest=contractmod._synthetic_manifest(seed=43)))
    p4 = dict(contractmod._synthetic_manifest()["phase4a_v2"], run_class="smoke")
    run_class_rejected = _contract_mismatch_rejects(
        lambda: _contract_run(manifest=contractmod._synthetic_manifest(phase4a_v2=p4)))
    return seed_rejected and run_class_rejected


def neg36():
    """Wrong base_checkpoint_params_sha256 vs the frozen contract -> mismatch id."""
    p4 = dict(contractmod._synthetic_manifest()["phase4a_v2"],
              base_checkpoint_params_sha256="2" * 64)
    return _contract_mismatch_rejects(
        lambda: _contract_run(manifest=contractmod._synthetic_manifest(phase4a_v2=p4)))


def neg37():
    """Self-declared / legacy exit provenance is NEVER accepted (task §二/§十二-8).

    Legacy fields in an engine binding, an engine certificate that already carries
    runner provenance, and a finalized binding with a self-declared / log-inferred
    exit source are each rejected."""
    legacy_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_engine_binding(run_exit_code=0)))
    legacy_pid_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_engine_binding(process_pid=4242)))
    # an engine certificate pre-loaded with runner provenance fails the ENGINE stage
    pre = _engine_binding()
    pre.update(_exit_binding())
    engine_stage_rejected = rejects(
        lambda: certmod.assert_engine_binding_complete({"eval_binding": pre}))
    self_declared_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=dict(pre, exit_source="self_declared"), finalized=True))
    inferred_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=dict(pre, inferred_from_log=True), finalized=True))
    return (legacy_rejected and legacy_pid_rejected and engine_stage_rejected
            and self_declared_rejected and inferred_rejected)


def neg38():
    """An engine certificate WITHOUT the runner's literal wait() provenance can never
    pass FINAL verification (task §二/§十二-9): when the child fails there is no PASS
    certificate, because only the parent runner can bind literal_exit_code=0."""
    engine_cert = certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_engine_binding(), finalized=False)
    return rejects(lambda: certmod.assert_eval_binding_complete(engine_cert))


def neg39():
    """Non-empty output dir / file path rejected; missing or empty dir fresh
    (task §四/§十二-10). Never rm -rf, never overwrite."""
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as td:
        missing = os.path.join(td, "new_dir")
        evaluator.assert_output_dir_fresh(missing)          # must NOT raise
        os.makedirs(missing)
        evaluator.assert_output_dir_fresh(missing)          # empty -> fresh
        with open(os.path.join(missing, "stale.jsonl"), "w") as fh:
            fh.write("{}\n")
        ok = ok and rejects(lambda: evaluator.assert_output_dir_fresh(missing))
        fpath = os.path.join(td, "afile")
        with open(fpath, "w") as fh:
            fh.write("x")
        ok = ok and rejects(lambda: evaluator.assert_output_dir_fresh(fpath))
    return ok


def neg40():
    """Split Student status labels (task §三/§十二-11,12): smoke vs performance modes
    label exactly the executed activity; the old ambiguous REAL_STUDENT_EVALUATION
    key is gone from EVERY mode; the scientific claim stays unauthorized."""
    ss_smoke = {"student_checkpoint_loaded": True,
                "student_policy_rollout_executed": True,
                "performance_evaluation_executed": False,
                "scientific_claim_authorized": False}
    ss_perf = dict(ss_smoke, performance_evaluation_executed=True)
    smoke = certmod.honest_status_labels(True, ss_smoke, "interface_smoke")
    perf = certmod.honest_status_labels(True, ss_perf, "performance_evaluation")
    ok = (smoke["REAL_STUDENT_INTERFACE_SMOKE"] == "EXECUTED"
          and smoke["REAL_STUDENT_PERFORMANCE_EVALUATION"] == "NOT_RUN"
          and perf["REAL_STUDENT_PERFORMANCE_EVALUATION"] == "EXECUTED"
          and perf["REAL_STUDENT_INTERFACE_SMOKE"] == "NOT_RUN"
          and smoke["FORMAL_SCIENTIFIC_CLAIM"] == "NOT_AUTHORIZED_SINGLE_TRAINING_SEED"
          and perf["FORMAL_SCIENTIFIC_CLAIM"] == "NOT_AUTHORIZED_SINGLE_TRAINING_SEED")
    # the ambiguous key must never reappear, in any mode
    for mode in certmod.CERT_MODES:
        ok = ok and ("REAL_STUDENT_EVALUATION"
                     not in certmod.honest_status_labels(False, ss_smoke, mode))
    return ok


def neg41():
    """Both arms run the IDENTICAL frozen start schedule (task §七/§十二-13):
    performance_start_schedule() is pure and reproduces 64 held-out seeds
    200000..200063 plus all 8 FRONT/BACK bank states; a drifted schedule is rejected
    by the certificate binding for PROVISIONAL runs."""
    s1 = evaluator.performance_start_schedule()
    s2 = evaluator.performance_start_schedule()
    identical = (s1 == s2
                 and s1[metrics.FULL]["seeds"] == [200000 + i for i in range(64)]
                 and s1[metrics.FULL]["count"] == 64
                 and s1[metrics.FRONT]["seeds"] == mat.fixed_seed_schedule(
                     mat.FRONT, mat.FROZEN_BANK_N, mat.FROZEN_SEED_BASE,
                     mat.FROZEN_SEED_STRIDE)
                 and s1[metrics.BACK]["seeds"] == mat.fixed_seed_schedule(
                     mat.BACK, mat.FROZEN_BANK_N, mat.FROZEN_SEED_BASE,
                     mat.FROZEN_SEED_STRIDE))
    shifted = json.loads(json.dumps(s1))
    shifted[metrics.FULL]["seeds"] = [200001 + i for i in range(64)]
    prov = _engine_binding(run_class="PROVISIONAL_STRONG_STUDENT_SELECTION",
                           max_timesteps=4096,
                           evaluation_seed_schedule=shifted,
                           state_entry_ids=_prov_entry_ids(),
                           **_ARTIFACT_PROV)   # 总控 §一.8 formal provenance
    drift_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=prov))
    return identical and drift_rejected


def neg42():
    """The carry-mode comparison rule is FIXED, machine-readable and strictly
    RECOMPUTABLE (task §九/§十二-14 + 总控范围修正): identical metrics -> identical
    RMT16_CARRY_MODE_WINNER; the rule constant is embedded verbatim; EVERY output
    carries the fixed scope fields (no overall strong-student selection is
    authorized this round); the forbidden strong-student / bakeoff vocabulary is
    rejected by the overclaim gate; a run that did not complete the frozen 64/8/8
    counts is not even extractable."""
    pm = selection.extract_arm_metrics(selection._result_doc(5, 4, 0.6, 2))
    rm = selection.extract_arm_metrics(selection._result_doc(3, 4, 0.6, 2, "reset128"))
    a = selection.select_provisional(pm, rm)
    b = selection.select_provisional(pm, rm)
    scope_ok = all(a.get(k) == v
                   for k, v in selection.FIXED_SCOPE_FIELDS.items())
    recomputable = (a == b
                    and a["RMT16_CARRY_MODE_WINNER"] == "PERSISTENT"
                    and a["decided_at_level"] == 1
                    and a["rule"] is selection.SELECTION_RULE
                    and scope_ok
                    and a["OVERALL_STRONG_STUDENT_SELECTION_AUTHORIZED"] is False
                    and a["STRONG_STUDENT_V1"] == "NOT_SELECTED"
                    and a["EXISTING_STUDENT_BAKEOFF_REQUIRED"] is True
                    and a["SCIENTIFIC_SUPERIORITY_CLAIM"] is False
                    and a["REQUIRES_MULTI_SEED_CONFIRMATION"] is True
                    and "PROVISIONAL_STRONG_STUDENT_RECOMMENDATION" not in a)
    # 总控 §二: the overclaim gate rejects the forbidden selection vocabulary
    # outright (existing gate, no new NEG number).
    forbidden_rejected = all(
        rejects(lambda claim=c: selection.assert_no_forbidden_claims(
            {"note": claim}))
        for c in ("PROVISIONAL_STRONG_STUDENT_RECOMMENDATION",
                  "STRONG_STUDENT_V1=PERSISTENT",
                  "STRONG_STUDENT_V1=RESET128",
                  "BEST_OVERALL_STUDENT",
                  "ALL_STUDENT_BAKEOFF_WINNER"))
    wrong = selection._result_doc(5, 4, 0.6, 2)
    wrong["results"][metrics.FULL]["metrics"]["primary"]["valid_starts"] = 63
    count_gate = rejects(lambda: selection.extract_arm_metrics(wrong))
    return recomputable and forbidden_rejected and count_gate


# ---------------------------------------------------------------------------
# NEG43–NEG49 (总控 ruling: frozen-bank artifacts + Round 1 screening)
# ---------------------------------------------------------------------------
def _round1_sched_ids():
    """The ROUND1 screening schedule exactly as the evaluator declares it
    (FULL 8-seed prefix of the frozen 64; FRONT/BACK bank indices 0,5) plus the
    matching state_entry_ids (总控 §四)."""
    sched = evaluator.screening_start_schedule(8, (0, 5))
    ids = {metrics.FULL: ["full-seed%d" % s for s in sched[metrics.FULL]["seeds"]],
           metrics.FRONT: ["front_l2-bank0", "front_l2-bank5"],
           metrics.BACK: ["back_l2-bank0", "back_l2-bank5"]}
    return sched, ids


def _round1_binding(sched, ids, **over):
    return _engine_binding(run_class="ROUND1_SCREENING", max_timesteps=4096,
                           evaluation_seed_schedule=sched,
                           state_entry_ids=ids, **over)


def neg43():
    """ROUND1 screening schedule purity (总控 §四): a declared subset of the frozen
    schedule (FULL 8-seed prefix; FRONT/BACK bank indices 0,5) is accepted by the
    certificate; a shifted FULL prefix or a dropped scenario is rejected."""
    sched, ids = _round1_sched_ids()
    accepted = not rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_round1_binding(sched, ids, **_ARTIFACT_PROV)))
    shifted = json.loads(json.dumps(sched))
    shifted[metrics.FULL]["seeds"] = [200001 + i for i in range(8)]
    shifted_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_round1_binding(shifted, ids, **_ARTIFACT_PROV)))
    missing = json.loads(json.dumps(sched))
    del missing[metrics.BACK]
    missing_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_round1_binding(missing, ids, **_ARTIFACT_PROV)))
    return accepted and shifted_rejected and missing_rejected


def neg44():
    """ROUND1 screening must embed the frozen PARENT schedule it screens from
    (screening_of_frozen_schedule); a subset with the parent dropped or tampered
    is rejected (总控 §四)."""
    sched, ids = _round1_sched_ids()
    no_parent = json.loads(json.dumps(sched))
    del no_parent[metrics.BACK]["screening_of_frozen_schedule"]
    dropped_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_round1_binding(no_parent, ids, **_ARTIFACT_PROV)))
    tampered = json.loads(json.dumps(sched))
    tampered[metrics.FRONT]["screening_of_frozen_schedule"] = [10001 + i
                                                               for i in range(8)]
    tampered_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_round1_binding(tampered, ids, **_ARTIFACT_PROV)))
    return dropped_rejected and tampered_rejected


def neg45():
    """ROUND1 FRONT/BACK bank_indices must be strictly ascending, unique and
    within [0,8), and the seeds the frozen bank states at exactly those indices
    (总控 §四)."""
    for bad in ((5, 0), (0, 0), (-1, 5), (0, 8)):
        if not rejects(lambda bad=bad: evaluator.screening_start_schedule(8, bad)):
            return False
    sched, ids = _round1_sched_ids()
    bad = json.loads(json.dumps(sched))
    bad[metrics.FRONT]["bank_indices"] = [5, 0]
    bad[metrics.FRONT]["seeds"] = [10005, 10000]   # consistent with the bad indices
    return rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_round1_binding(bad, ids, **_ARTIFACT_PROV)))


def neg46():
    """Scaffold bank provenance honesty (总控 §一.7/§一.8): a formal run class whose
    scaffold bank was regenerated in memory is rejected; artifact provenance whose
    loaded content does not canonicalize to the bound frozen bank hash is
    rejected; regeneration=true together with an artifact source is rejected; an
    unknown bank_source value is rejected."""
    sched, ids = _round1_sched_ids()
    inmemory_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_round1_binding(sched, ids)))   # default REGENERATED_IN_MEMORY
    mismatch = dict(_ARTIFACT_PROV)
    mismatch["loaded_content_sha256"] = "3" + "a" * 63
    mismatch_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_round1_binding(sched, ids, **mismatch)))
    regen = dict(_ARTIFACT_PROV)
    regen["bank_regenerated_on_eval_device"] = True
    regen_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_round1_binding(sched, ids, **regen)))
    bad_source_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_engine_binding(bank_source="GPU_REGENERATED")))
    return (inmemory_rejected and mismatch_rejected and regen_rejected
            and bad_source_rejected)


def neg47():
    """bank_kind / bank_source cross-binding (总控 §一.8): canonical reset seeds may
    not masquerade as an artifact source; canonical reset seeds must declare
    regeneration on the eval device; an unknown bank_kind is rejected."""
    cross_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_engine_binding(bank_kind="CANONICAL_RESET_SEEDS",
                                     **_ARTIFACT_PROV)))
    regen_false_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_engine_binding(
            bank_kind="CANONICAL_RESET_SEEDS",
            bank_source="CANONICAL_RESET_SEEDS",
            bank_regenerated_on_eval_device=False,
            artifact_file_sha256=None,
            loaded_content_sha256=None,
            device_provenance={"eval_device": {"jax_default_backend": "gpu"}})))
    bad_kind_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_engine_binding(bank_kind="SOMETHING_ELSE")))
    return cross_rejected and regen_false_rejected and bad_kind_rejected


def neg48():
    """Artifact SHAs must be 64-hex values exactly where required, and null
    exactly where forbidden (总控 §一.8): non-hex artifact SHAs are rejected;
    canonical reset seeds and in-memory regenerated banks may carry no artifact
    SHAs at all."""
    sched, ids = _round1_sched_ids()
    bad_file = dict(_ARTIFACT_PROV)
    bad_file["artifact_file_sha256"] = "xyz"
    file_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_round1_binding(sched, ids, **bad_file)))
    bad_loaded = dict(_ARTIFACT_PROV)
    bad_loaded["loaded_content_sha256"] = "z" * 64
    loaded_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_round1_binding(sched, ids, **bad_loaded)))
    canonical_shas_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_engine_binding(
            bank_kind="CANONICAL_RESET_SEEDS",
            bank_source="CANONICAL_RESET_SEEDS",
            bank_regenerated_on_eval_device=True,
            artifact_file_sha256="a" * 64,
            loaded_content_sha256="2" + "a" * 63,
            device_provenance={"eval_device": {"jax_default_backend": "gpu"}})))
    inmemory_shas_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_engine_binding(artifact_file_sha256="a" * 64)))
    return (file_rejected and loaded_rejected and canonical_shas_rejected
            and inmemory_shas_rejected)


def neg49():
    """bank_source / device_provenance must be bound (总控 §一.8): a missing
    bank_source, an empty device_provenance, and an artifact binding missing the
    'mint' or 'load' identity dict are all rejected."""
    sched, ids = _round1_sched_ids()
    no_source = _engine_binding()
    del no_source["bank_source"]
    source_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=no_source))
    empty_dev_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_engine_binding(device_provenance={})))
    no_mint = dict(_ARTIFACT_PROV)
    no_mint["device_provenance"] = {"load": {"jax_default_backend": "gpu"}}
    no_mint_rejected = rejects(lambda: certmod.build_certificate(
        _front_result(), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        eval_binding=_round1_binding(sched, ids, **no_mint)))
    return source_rejected and empty_dev_rejected and no_mint_rejected


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
    ("NEG29", "finalized certificate runner provenance missing/invalid", neg29),
    ("NEG30", "wrong final checkpoint file SHA vs frozen contract", neg30),
    ("NEG31", "wrong final params SHA vs frozen contract", neg31),
    ("NEG32", "step-8192 checkpoint impersonating final 98304", neg32),
    ("NEG33", "wrong arm name / carry_mode / cross-arm file SHA", neg33),
    ("NEG34", "wrong replay_mode vs frozen contract", neg34),
    ("NEG35", "wrong seed / run_class vs frozen contract", neg35),
    ("NEG36", "wrong base_checkpoint_params_sha256 vs frozen contract", neg36),
    ("NEG37", "self-declared / legacy exit code never accepted", neg37),
    ("NEG38", "engine cert without literal runner provenance never finalizes", neg38),
    ("NEG39", "non-empty / file output dir rejected (freshness gate)", neg39),
    ("NEG40", "split Student status labels; ambiguous key removed", neg40),
    ("NEG41", "both arms' frozen start schedules identical; drift rejected", neg41),
    ("NEG42", "comparison rule fixed + scope fields + forbidden vocab rejected", neg42),
    ("NEG43", "ROUND1 screening schedule subset purity (shift/drop rejected)", neg43),
    ("NEG44", "ROUND1 embedded frozen parent schedule required", neg44),
    ("NEG45", "ROUND1 bank_indices ascending/unique/in-range + seed match", neg45),
    ("NEG46", "scaffold bank provenance honesty (in-memory formal rejected)", neg46),
    ("NEG47", "bank_kind / bank_source cross-binding", neg47),
    ("NEG48", "artifact SHA 64-hex where required / null where forbidden", neg48),
    ("NEG49", "bank_source / device_provenance must be bound", neg49),
]

# All 49 NEG tests are implemented (NEG30-42 landed with the frozen final-98304
# checkpoint contract + runner provenance + provisional selection work; NEG43-49
# land with the frozen-bank artifact protocol + Round 1 screening, 总控 ruling).
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
    print("TIER3_NEGATIVE_TESTS_PASS (FAIL=0; implemented=%d/49, pending_commit3=%d)"
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
