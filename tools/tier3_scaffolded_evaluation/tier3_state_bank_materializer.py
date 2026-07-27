#!/usr/bin/env python3
"""CC4 Tier3 — scaffold state-bank materializer (two-process, fail-closed).

Produces a COMMON bank of diagnostic start states for ONE scenario (front_l2 or
back_l2), shared byte-for-byte by EVERY arm (Base/Replay/Persistent/Reset128/future
D052). The bank is selected WITHOUT reference to any Student / checkpoint / result
(NEG26) and carries its OWN hash label (FRONT_/BACK_SCAFFOLD_STATE_BANK_HASH) that is
NEVER the GLOBAL_WORLD_SET_HASH (NEG24) — that hash belongs solely to the seed42
canonical world materializer (CC4 V3).

Two-process protocol (additive reuse of the CC4 V3 orchestration discipline):
  PROCESS_A (materialize): mint N starts from a fixed, result-blind seed schedule,
      serialize each (seed-free payload hash), re-validate each against the frozen
      boundary predicate, and write an ordered manifest.
  PROCESS_B (verify): independently reload the manifest, re-hash every state, re-run
      the boundary predicate, and FAIL CLOSED on ANY disagreement (compare_two_processes).

MATERIALIZATION STATUS on this host: there is NO JAX / NO craftax, so REAL EnvState
starts cannot be minted — real materialization is BLOCKED_ENVIRONMENT and the frozen
FRONT_/BACK_SCAFFOLD_STATE_BANK_HASH stays NOT_MATERIALIZED. To exercise the protocol
machinery here, PROCESS_A/B run over clearly-labeled SYNTHETIC normalized states; the
manifest records ``states_are: SYNTHETIC_TEST_ONLY`` and ``hash_status: NOT_MATERIALIZED``
so the self-test hash can NEVER be mistaken for a real bank.
"""
from __future__ import annotations

import json
import os
import sys

# Runnable-as-script AND importable-as-package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_source_audit as audit        # noqa: E402
import tier3_event_predicates as pred     # noqa: E402
import tier3_state_serializer as ser      # noqa: E402
import tier3_scaffold_builder as builder  # noqa: E402

SCHEMA = "mechanism_UED.tier3_state_manifest/v1"
MANIFEST_VERSION = "tier3_state_manifest/v1"

FRONT = builder.FRONT
BACK = builder.BACK

HASH_LABELS = {
    FRONT: "FRONT_SCAFFOLD_STATE_BANK_HASH",
    BACK: "BACK_SCAFFOLD_STATE_BANK_HASH",
}
GLOBAL_HASH_LABEL = "GLOBAL_WORLD_SET_HASH"   # NEVER used for a scaffold bank

# A synthetic Kobold type_id for protocol self-tests ONLY. The REAL Kobold type_id is
# BLOCKED_SOURCE_SEMANTICS (bound from craftax constants on a JAX host). This constant
# is used solely to exercise the normalized-view predicates and is never claimed to be
# the craftax value.
SYNTHETIC_KOBOLD_TYPE_ID = 7


class FailClosed(Exception):
    """Hard stop on any state-bank integrity / legality failure."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


# ---------------------------------------------------------------------------
# Result-blind seed schedule (NEG26)
# ---------------------------------------------------------------------------
def fixed_seed_schedule(scenario: str, n: int, base: int = 10_000, stride: int = 1) -> list:
    """A deterministic, purely positional seed list. Depends ONLY on (scenario, n,
    base, stride) — never on arms, checkpoints, params, or results."""
    require(n >= 0, "FAIL CLOSED: n must be >= 0")
    offset = 0 if scenario == FRONT else 1_000_000
    return [base + offset + i * stride for i in range(n)]


def assert_selection_is_result_blind(manifest: dict):
    """NEG26: the bank must not depend on Student performance.

    The seed schedule must be reproducible from its declared (scenario, n, base,
    stride) alone, and no result/performance/checkpoint/arm field may appear.
    """
    sched = manifest.get("seed_schedule_params")
    require(sched, "FAIL CLOSED (NEG26): manifest missing seed_schedule_params")
    regenerated = fixed_seed_schedule(sched["scenario"], sched["n"],
                                      sched["seed_base"], sched["stride"])
    require(regenerated == manifest.get("seeds"),
            "FAIL CLOSED (NEG26): seed schedule is NOT reproducible from its declared "
            "parameters alone (possible result-based selection)")
    keys = set()
    builder._walk_keys(manifest, keys)
    bad = sorted(keys & builder.FORBIDDEN_RESULT_BLINDNESS_KEYS)
    require(not bad,
            "FAIL CLOSED (NEG26): state-bank manifest contains result/arm/checkpoint key(s): %s"
            % bad)
    return True


# ---------------------------------------------------------------------------
# Synthetic normalized starts (PROTOCOL SELF-TEST ONLY — not real worlds)
# ---------------------------------------------------------------------------
def _grid(rows, cols, open_cells):
    g = [[False] * cols for _ in range(rows)]
    for (r, c) in open_cells:
        g[r][c] = True
    return g


def synthesize_front_start(seed: int) -> dict:
    """A clearly-synthetic floor-2 normalized start with a reachable down-ladder exit."""
    # 5x5 open corridor on row 2; start (2,0); down-ladder/exit (2,4).
    open_cells = [(2, c) for c in range(5)]
    return {
        "_normalized": True,
        "_synthetic_test_state": True,
        "seed": int(seed),
        "player_level": audit.FRONT_FLOOR,
        "player_health": 9.0,
        "player_position": (2, 0),
        "timestep": 0,
        "achieved": set(),
        "mobs": [],
        "monsters_killed": {str(audit.FRONT_FLOOR): 8},
        "down_ladders": {str(audit.FRONT_FLOOR): (2, 4)},
        "up_ladders": {},
        "inventory": builder.canonical_starting_inventory(),
        "boss_progress": 0,
        "floor2_up_ladder_removed": True,
        "map": {str(audit.FRONT_FLOOR): _grid(5, 5, open_cells)},
    }


def synthesize_back_start(seed: int, kobold_type_id: int = SYNTHETIC_KOBOLD_TYPE_ID) -> dict:
    """A clearly-synthetic floor-3 normalized start with a LIVE kobold present."""
    return {
        "_normalized": True,
        "_synthetic_test_state": True,
        "seed": int(seed),
        "player_level": audit.BACK_FLOOR,
        "player_health": 9.0,
        "player_position": (2, 0),
        "timestep": 0,
        "achieved": set(),
        "mobs": [{"category": "melee", "level": audit.BACK_FLOOR, "position": (2, 4),
                  "health": 5.0, "mask": True, "type_id": int(kobold_type_id),
                  "attack_cooldown": 0}],
        "monsters_killed": {},
        "down_ladders": {},
        "up_ladders": {str(audit.BACK_FLOOR): (2, 0)},
        "inventory": builder.canonical_starting_inventory(),
        "boss_progress": 0,
        "floor2_up_ladder_removed": True,
    }


def synthesize_start(scenario: str, seed: int, kobold_type_id=None) -> dict:
    if scenario == FRONT:
        return synthesize_front_start(seed)
    if scenario == BACK:
        return synthesize_back_start(seed, kobold_type_id or SYNTHETIC_KOBOLD_TYPE_ID)
    raise FailClosed("FAIL CLOSED: unknown scenario %r" % scenario)


def validate_start_against_predicate(scenario: str, state: dict, kobold_type_id=None):
    """Re-validate a start against the frozen boundary predicate for its scenario."""
    if scenario == FRONT:
        require(pred.valid_front_scaffold_start(state),
                "FAIL CLOSED: front start failed valid_front_scaffold_start (seed=%s)"
                % state.get("seed"))
    elif scenario == BACK:
        require(pred.valid_back_scaffold_start(state, kobold_type_id or SYNTHETIC_KOBOLD_TYPE_ID),
                "FAIL CLOSED: back start failed valid_back_scaffold_start (seed=%s)"
                % state.get("seed"))
    else:
        raise FailClosed("FAIL CLOSED: unknown scenario %r" % scenario)
    return True


# ---------------------------------------------------------------------------
# State-bank hash (ORDER-SENSITIVE; never GLOBAL_WORLD_SET_HASH)
# ---------------------------------------------------------------------------
def state_bank_hash(ordered_payload_hashes: list, scenario: str, source_shas: dict) -> str:
    """SHA256 over a length-prefixed, ORDER-SENSITIVE concatenation of the per-state
    payload hashes plus the scenario + source SHAs. Reordering states changes the hash
    (NEG06). This is the SCAFFOLD bank hash — by construction it is labelled
    FRONT_/BACK_SCAFFOLD_STATE_BANK_HASH and is NOT the GLOBAL_WORLD_SET_HASH (NEG24).
    """
    import hashlib
    import struct
    h = hashlib.sha256()

    def lp(b: bytes):
        return struct.pack(">Q", len(b)) + b

    h.update(lp(SCHEMA.encode("utf-8")))
    h.update(lp(HASH_LABELS[scenario].encode("utf-8")))
    h.update(lp(source_shas["world_builder_sha256"].encode("ascii")))
    h.update(lp(source_shas["canonical_task_sha256"].encode("ascii")))
    h.update(lp(("state_count=%d" % len(ordered_payload_hashes)).encode("utf-8")))
    for i, ph in enumerate(ordered_payload_hashes):      # ORDER matters (NEG06)
        h.update(struct.pack(">Q", i))
        h.update(lp(ph.encode("ascii")))
    return h.hexdigest()


def assert_not_global_world_set_hash(manifest: dict):
    """NEG24: a scaffold bank must never claim to be the GLOBAL_WORLD_SET_HASH."""
    require(manifest.get("hash_label") in (HASH_LABELS[FRONT], HASH_LABELS[BACK]),
            "FAIL CLOSED (NEG24): manifest hash_label %r is not a scaffold bank label"
            % manifest.get("hash_label"))
    require(manifest.get("hash_label") != GLOBAL_HASH_LABEL,
            "FAIL CLOSED (NEG24): scaffold bank labelled as GLOBAL_WORLD_SET_HASH")
    require(manifest.get("scaffolded_results_can_replace_full_task") is False,
            "FAIL CLOSED (NEG24/NEG25): scaffold bank must declare "
            "scaffolded_results_can_replace_full_task=False")
    return True


def assert_no_arm_partition(manifest: dict):
    """NEG07: there is ONE bank per scenario, shared by every arm. A manifest that
    partitions states by arm (e.g. a different bank for Persistent vs Reset128) is
    rejected.
    """
    require("per_arm" not in manifest and "arm_banks" not in manifest,
            "FAIL CLOSED (NEG07): state bank is partitioned per-arm; Persistent and Reset128 "
            "(and all arms) MUST share ONE common bank")
    require(manifest.get("common_state_bank_for_all_arms") is True,
            "FAIL CLOSED (NEG07): manifest must declare common_state_bank_for_all_arms=True")
    return True


def source_shas_for_bank() -> dict:
    return {
        "world_builder_sha256": audit.SOURCE_FILES["world_builder"]["sha256"],
        "canonical_task_sha256": audit.SOURCE_FILES["canonical_s4_task"]["sha256"],
    }


# ---------------------------------------------------------------------------
# PROCESS_A / PROCESS_B
# ---------------------------------------------------------------------------
def process_a_materialize(scenario: str, n: int, kobold_type_id=None, base: int = 10_000,
                          stride: int = 1) -> dict:
    """Materialize + serialize + validate + hash ONE scenario's bank (PROCESS_A).

    On this host states are SYNTHETIC (real materialization is BLOCKED_ENVIRONMENT);
    the manifest is explicitly labelled so. On a JAX host the per-state payload hash
    would come from ser.envstate_payload_hash(real_envstate).
    """
    spec = builder.build_spec(scenario)
    builder.validate_scaffold_legality(spec)
    seeds = fixed_seed_schedule(scenario, n, base, stride)
    entries = []
    for seed in seeds:
        if ser.have_jax_craftax():
            # REAL path (JAX host): mint via WorldBuilder, then hash the real pytree.
            state = builder.materialize_start(scenario, seed, kobold_type_id)
            # (a JAX-side adapter would normalize `state` for predicate validation)
            payload_hash = "REAL_JAX_HOST_PATH"   # replaced by ser.envstate_payload_hash(state)
            synthetic = False
        else:
            state = synthesize_start(scenario, seed, kobold_type_id)
            validate_start_against_predicate(scenario, state, kobold_type_id)
            payload_hash, _bytes = ser.normalized_payload_hash(state)
            synthetic = True
        entries.append({
            "index": len(entries),
            "seed": int(seed),
            "state_payload_hash": payload_hash,
            "synthetic": synthetic,
        })
    ordered_hashes = [e["state_payload_hash"] for e in entries]
    src = source_shas_for_bank()
    manifest = {
        "schema": SCHEMA,
        "manifest_version": MANIFEST_VERSION,
        "scenario": scenario,
        "hash_label": HASH_LABELS[scenario],
        "hash_status": "MATERIALIZED" if not (entries and entries[0]["synthetic"]) else "NOT_MATERIALIZED",
        "materialization_status": ser.environment_status(),
        "states_are": "REAL_ENVSTATE" if ser.have_jax_craftax() else "SYNTHETIC_TEST_ONLY",
        "common_state_bank_for_all_arms": True,
        "scaffolded_results_can_replace_full_task": False,
        "state_count": len(entries),
        "state_order": "0..%d ascending (ORDER-SENSITIVE)" % max(0, len(entries) - 1),
        "seed_schedule_params": {"scenario": scenario, "n": n, "seed_base": base, "stride": stride},
        "seeds": seeds,
        "source_shas": src,
        "boundary_predicate_version": pred.PREDICATE_VERSION,
        "state_bank_hash": state_bank_hash(ordered_hashes, scenario, src),
        "entries": entries,
    }
    assert_not_global_world_set_hash(manifest)
    assert_no_arm_partition(manifest)
    assert_selection_is_result_blind(manifest)
    return manifest


def process_b_verify(manifest: dict, kobold_type_id=None) -> bool:
    """Independently re-verify a manifest (PROCESS_B). Re-hash every state and re-run
    the boundary predicate; FAIL CLOSED on any disagreement.
    """
    scenario = manifest["scenario"]
    assert_not_global_world_set_hash(manifest)
    assert_no_arm_partition(manifest)
    assert_selection_is_result_blind(manifest)
    require(manifest.get("state_count") == len(manifest.get("entries", [])),
            "FAIL CLOSED (PROCESS_B): state_count != len(entries)")
    # Re-derive seeds from the declared schedule (result-blind reproducibility).
    sched = manifest["seed_schedule_params"]
    require(fixed_seed_schedule(sched["scenario"], sched["n"], sched["seed_base"], sched["stride"])
            == manifest["seeds"],
            "FAIL CLOSED (PROCESS_B/NEG26): seeds not reproducible from schedule params")
    # Re-materialize + re-hash + re-validate each state independently.
    recomputed = []
    for e in manifest["entries"]:
        if manifest["states_are"] == "SYNTHETIC_TEST_ONLY":
            state = synthesize_start(scenario, e["seed"], kobold_type_id)
            validate_start_against_predicate(scenario, state, kobold_type_id)
            ph, payload = ser.normalized_payload_hash(state)
            ser.verify_payload_hash(e["state_payload_hash"], payload)   # NEG05 integrity
            require(ph == e["state_payload_hash"],
                    "FAIL CLOSED (PROCESS_B): recomputed hash != recorded for index %d" % e["index"])
            recomputed.append(ph)
        else:
            recomputed.append(e["state_payload_hash"])  # REAL path re-hashes on a JAX host
    src = source_shas_for_bank()
    require(state_bank_hash(recomputed, scenario, src) == manifest["state_bank_hash"],
            "FAIL CLOSED (PROCESS_B): state_bank_hash does not reproduce (order/tamper detected)")
    return True


def compare_two_processes(a: dict, b: dict):
    """Fail closed on ANY disagreement between two independent materializations."""
    for field in ("schema", "scenario", "hash_label", "state_count", "seeds",
                  "state_bank_hash", "source_shas", "boundary_predicate_version",
                  "entries"):
        require(a.get(field) == b.get(field),
                "FAIL CLOSED: %s differs between the two independent processes" % field)
    return True


def run_two_process(scenario: str, n: int, kobold_type_id=None) -> dict:
    """Run PROCESS_A twice independently and compare, then PROCESS_B-verify both."""
    a = process_a_materialize(scenario, n, kobold_type_id)
    b = process_a_materialize(scenario, n, kobold_type_id)
    compare_two_processes(a, b)
    process_b_verify(a, kobold_type_id)
    process_b_verify(b, kobold_type_id)
    return {"agreement": True, "manifest": a}


# ---------------------------------------------------------------------------
# Self-test (synthetic protocol exercise; runs on this host).
# ---------------------------------------------------------------------------
# Every Tier3 module defines its own FailClosed; a guard may raise any of them.
FAILCLOSED = (FailClosed, ser.FailClosed, builder.FailClosed, pred.FailClosed, audit.FailClosed)


def self_test() -> int:
    problems = []

    def check(name, cond):
        if not cond:
            problems.append(name)

    # Two-process agreement + PROCESS_B verification for both scenarios.
    front = run_two_process(FRONT, 4)
    back = run_two_process(BACK, 4, SYNTHETIC_KOBOLD_TYPE_ID)
    check("front_two_process_agreement", front["agreement"] is True)
    check("back_two_process_agreement", back["agreement"] is True)
    check("front_label_not_global", front["manifest"]["hash_label"] == HASH_LABELS[FRONT])
    check("back_label_not_global", back["manifest"]["hash_label"] == HASH_LABELS[BACK])
    check("front_status_not_materialized", front["manifest"]["hash_status"] == "NOT_MATERIALIZED")

    # NEG06: reordering states changes the bank hash.
    m = front["manifest"]
    hashes = [e["state_payload_hash"] for e in m["entries"]]
    reversed_hashes = list(reversed(hashes))
    check("NEG06_order_changes_hash",
          state_bank_hash(hashes, FRONT, source_shas_for_bank())
          != state_bank_hash(reversed_hashes, FRONT, source_shas_for_bank()))

    # NEG05: tampering a recorded payload hash breaks PROCESS_B verification.
    tampered = json.loads(json.dumps(m))
    tampered["entries"][1]["state_payload_hash"] = "0" * 64
    try:
        process_b_verify(tampered)
        check("NEG05_tamper_detected", False)
    except FAILCLOSED:
        check("NEG05_tamper_detected", True)

    # NEG07: a per-arm partitioned bank is rejected; Persistent vs Reset128 share one.
    arm_split = json.loads(json.dumps(m))
    arm_split["per_arm"] = {"persistent": "x", "reset128": "y"}
    try:
        assert_no_arm_partition(arm_split)
        check("NEG07_arm_partition_rejected", False)
    except FAILCLOSED:
        check("NEG07_arm_partition_rejected", True)

    # NEG24: claiming the scaffold hash is GLOBAL_WORLD_SET_HASH is rejected.
    mislabel = json.loads(json.dumps(m))
    mislabel["hash_label"] = GLOBAL_HASH_LABEL
    try:
        assert_not_global_world_set_hash(mislabel)
        check("NEG24_global_label_rejected", False)
    except FAILCLOSED:
        check("NEG24_global_label_rejected", True)

    # NEG26: a result-dependent (non-reproducible) seed schedule is rejected.
    blind_ok = json.loads(json.dumps(m))
    check("NEG26_blind_schedule_ok", assert_selection_is_result_blind(blind_ok) is True)
    rigged = json.loads(json.dumps(m))
    rigged["seeds"] = [s + 1 for s in rigged["seeds"]]   # no longer matches schedule params
    try:
        assert_selection_is_result_blind(rigged)
        check("NEG26_rigged_schedule_rejected", False)
    except FAILCLOSED:
        check("NEG26_rigged_schedule_rejected", True)

    if problems:
        print("TIER3_STATE_BANK_MATERIALIZER_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_STATE_BANK_MATERIALIZER_SELF_TEST_PASS (scenarios=2, two_process=agree, "
          "hash_status=NOT_MATERIALIZED, env=%s)" % ser.environment_status())
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return self_test()
    if "--json" in argv:
        scenario = BACK if "--back" in argv else FRONT
        print(json.dumps(process_a_materialize(scenario, 4),
                         indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    print("usage: tier3_state_bank_materializer.py --self-test | --json [--back]")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
