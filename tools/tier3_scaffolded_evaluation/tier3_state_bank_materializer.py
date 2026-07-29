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

MATERIALIZATION MODES:
  * REAL (JAX + craftax==1.4.5 host): PROCESS_A mints real EnvState pytrees via
    builder.materialize_start (canonical WorldBuilder rng sequence), hashes each with
    ser.envstate_payload_hash (V3 seed-free serializer), normalizes each with
    builder.normalize_envstate and re-validates it against the frozen boundary
    predicate. The manifest records ``states_are: REAL_ENVSTATE``,
    ``hash_status: MATERIALIZED`` and a bank-level field manifest. PROCESS_B
    re-materializes + re-hashes + re-validates every state independently.
  * SYNTHETIC (no JAX): real materialization is BLOCKED_ENVIRONMENT and the frozen
    FRONT_/BACK_SCAFFOLD_STATE_BANK_HASH stays NOT_MATERIALIZED. To exercise the
    protocol machinery, PROCESS_A/B run over clearly-labeled SYNTHETIC normalized
    states; the manifest records ``states_are: SYNTHETIC_TEST_ONLY`` and
    ``hash_status: NOT_MATERIALIZED`` so the self-test hash can NEVER be mistaken
    for a real bank.

TWO INDEPENDENT OS PROCESSES: ``run_two_process_real`` spawns two fresh interpreters
(``--materialize-real``), each minting the bank from the declared result-blind seed
schedule, then compares ordered IDs / per-state payload hash / field manifest /
state-bank hash and PROCESS_B-verifies both. Any disagreement -> FailClosed.
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

# Kobold type_id used for the SYNTHETIC protocol self-test states. It equals the
# RESOLVED craftax==1.4.5 binding (RANGED category, ranged type_id 3 — see
# tier3_source_audit.CRAFTAX_RUNTIME_BINDINGS) so the synthetic path exercises the
# exact same predicate semantics as the real path; the REAL path resolves the value
# live from craftax constants (builder.resolve_kobold_type_id).
SYNTHETIC_KOBOLD_TYPE_ID = 3


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
    """A clearly-synthetic floor-3 normalized start with a LIVE Kobold present
    (RANGED category, canonical max health 8.0 — mirrors the resolved binding)."""
    return {
        "_normalized": True,
        "_synthetic_test_state": True,
        "seed": int(seed),
        "player_level": audit.BACK_FLOOR,
        "player_health": 9.0,
        "player_position": (2, 0),
        "timestep": 0,
        "achieved": set(),
        "mobs": [{"category": pred.KOBOLD_CATEGORY, "level": audit.BACK_FLOOR, "position": (2, 4),
                  "health": float(audit.CRAFTAX_RUNTIME_BINDINGS["kobold_canonical_max_health"]),
                  "mask": True, "type_id": int(kobold_type_id),
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
# REAL per-state carrier (JAX host): mint -> V3 payload hash -> normalize -> validate
# ---------------------------------------------------------------------------
def _field_manifest_sha(paths: list) -> str:
    return ser.sha256_bytes(("\n".join(paths)).encode("utf-8"))


def real_state_entry(scenario: str, seed: int, kobold_type_id=None) -> dict:
    """Mint ONE real start and return its evidence carrier:
      state_payload_hash — V3 seed-free canonical serialization SHA256
      field_manifest     — sorted V3 pytree leaf paths (the serialized field set)
      field_manifest_sha256
      normalized_view    — the JAX-free predicate view (for validation; not hashed here)
    """
    require(ser.have_jax_craftax(),
            "FAIL CLOSED (BLOCKED_ENVIRONMENT): real state entry requires JAX+craftax")
    state = builder.materialize_start(scenario, seed, kobold_type_id)
    payload_hash, _payload, v3manifest = ser.envstate_payload_hash(state)
    view = builder.normalize_envstate(state)
    if scenario == FRONT:
        view["floor2_up_ladder_removed"] = True   # materialize_start performs the removal
    type_id = builder.resolve_kobold_type_id(kobold_type_id) if scenario == BACK else None
    validate_start_against_predicate(scenario, view, type_id)
    ser.assert_required_envstate_fields(view)
    paths = sorted(str(m["path"]) for m in v3manifest)
    return {
        "state_payload_hash": payload_hash,
        "field_manifest": paths,
        "field_manifest_sha256": _field_manifest_sha(paths),
        "normalized_view": view,
    }


def _synthetic_field_manifest(state: dict) -> list:
    return sorted(str(k) for k in ser._canonicalize(state).keys())


# ---------------------------------------------------------------------------
# PROCESS_A / PROCESS_B
# ---------------------------------------------------------------------------
def process_a_materialize(scenario: str, n: int, kobold_type_id=None, base: int = 10_000,
                          stride: int = 1) -> dict:
    """Materialize + serialize + validate + hash ONE scenario's bank (PROCESS_A).

    REAL on a JAX+craftax host (real EnvState pytrees, V3 payload hashes, predicate
    re-validation); SYNTHETIC and explicitly labelled otherwise.
    """
    spec = builder.build_spec(scenario)
    builder.validate_scaffold_legality(spec)
    real = ser.have_jax_craftax()
    seeds = fixed_seed_schedule(scenario, n, base, stride)
    entries = []
    field_manifest = None
    for seed in seeds:
        if real:
            carrier = real_state_entry(scenario, seed, kobold_type_id)
            if field_manifest is None:
                field_manifest = carrier["field_manifest"]
            entries.append({
                "index": len(entries),
                "seed": int(seed),
                "state_payload_hash": carrier["state_payload_hash"],
                "field_manifest_sha256": carrier["field_manifest_sha256"],
                "synthetic": False,
            })
        else:
            state = synthesize_start(scenario, seed, kobold_type_id)
            validate_start_against_predicate(scenario, state, kobold_type_id)
            payload_hash, _bytes = ser.normalized_payload_hash(state)
            fm = _synthetic_field_manifest(state)
            entries.append({
                "index": len(entries),
                "seed": int(seed),
                "state_payload_hash": payload_hash,
                "field_manifest_sha256": _field_manifest_sha(fm),
                "synthetic": True,
            })
    # The serialized field set MUST be identical for every state in the bank.
    fm_shas = {e["field_manifest_sha256"] for e in entries}
    require(len(fm_shas) <= 1,
            "FAIL CLOSED: field manifest differs across states in the same bank: %s"
            % sorted(fm_shas))
    ordered_hashes = [e["state_payload_hash"] for e in entries]
    src = source_shas_for_bank()
    manifest = {
        "schema": SCHEMA,
        "manifest_version": MANIFEST_VERSION,
        "scenario": scenario,
        "hash_label": HASH_LABELS[scenario],
        "hash_status": "MATERIALIZED" if real else "NOT_MATERIALIZED",
        "materialization_status": ser.environment_status(),
        "states_are": "REAL_ENVSTATE" if real else "SYNTHETIC_TEST_ONLY",
        "common_state_bank_for_all_arms": True,
        "scaffolded_results_can_replace_full_task": False,
        "state_count": len(entries),
        "state_order": "0..%d ascending (ORDER-SENSITIVE)" % max(0, len(entries) - 1),
        "seed_schedule_params": {"scenario": scenario, "n": n, "seed_base": base, "stride": stride},
        "seeds": seeds,
        "source_shas": src,
        "boundary_predicate_version": pred.PREDICATE_VERSION,
        "field_manifest": field_manifest,
        "field_manifest_sha256": entries[0]["field_manifest_sha256"] if entries else None,
        "state_bank_hash": state_bank_hash(ordered_hashes, scenario, src),
        "entries": entries,
    }
    if scenario == BACK:
        manifest["resolved_kobold_type_id"] = (builder.resolve_kobold_type_id(kobold_type_id)
                                               if real else SYNTHETIC_KOBOLD_TYPE_ID)
    assert_not_global_world_set_hash(manifest)
    assert_no_arm_partition(manifest)
    assert_selection_is_result_blind(manifest)
    return manifest


def process_b_verify(manifest: dict, kobold_type_id=None) -> bool:
    """Independently re-verify a manifest (PROCESS_B). Re-materialize + re-hash +
    re-validate EVERY state (real on a JAX host, synthetic otherwise); FAIL CLOSED on
    any disagreement.
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
    real = manifest["states_are"] == "REAL_ENVSTATE"
    if real:
        require(ser.have_jax_craftax(),
                "FAIL CLOSED (BLOCKED_ENVIRONMENT): manifest claims REAL_ENVSTATE but this "
                "host cannot re-materialize (no JAX+craftax)")
    # Re-materialize + re-hash + re-validate each state independently.
    recomputed = []
    for e in manifest["entries"]:
        if real:
            carrier = real_state_entry(scenario, e["seed"], kobold_type_id)
            require(carrier["state_payload_hash"] == e["state_payload_hash"],
                    "FAIL CLOSED (PROCESS_B): recomputed REAL payload hash != recorded "
                    "for index %d (seed %d)" % (e["index"], e["seed"]))
            require(carrier["field_manifest_sha256"] == e.get("field_manifest_sha256"),
                    "FAIL CLOSED (PROCESS_B): field manifest drifted for index %d" % e["index"])
            recomputed.append(carrier["state_payload_hash"])
        else:
            state = synthesize_start(scenario, e["seed"], kobold_type_id)
            validate_start_against_predicate(scenario, state, kobold_type_id)
            ph, payload = ser.normalized_payload_hash(state)
            ser.verify_payload_hash(e["state_payload_hash"], payload)   # NEG05 integrity
            require(ph == e["state_payload_hash"],
                    "FAIL CLOSED (PROCESS_B): recomputed hash != recorded for index %d" % e["index"])
            recomputed.append(ph)
    src = source_shas_for_bank()
    require(state_bank_hash(recomputed, scenario, src) == manifest["state_bank_hash"],
            "FAIL CLOSED (PROCESS_B): state_bank_hash does not reproduce (order/tamper detected)")
    if manifest.get("field_manifest_sha256") is not None:
        require(all(e.get("field_manifest_sha256") == manifest["field_manifest_sha256"]
                    for e in manifest["entries"]),
                "FAIL CLOSED (PROCESS_B): per-entry field_manifest_sha256 != bank-level value")
    return True


def compare_two_processes(a: dict, b: dict):
    """Fail closed on ANY disagreement between two independent materializations:
    ordered IDs (seeds/entries), per-state payload hashes, field manifest and the
    state-bank hash itself."""
    for field in ("schema", "scenario", "hash_label", "state_count", "seeds",
                  "state_bank_hash", "source_shas", "boundary_predicate_version",
                  "field_manifest", "field_manifest_sha256", "entries"):
        require(a.get(field) == b.get(field),
                "FAIL CLOSED: %s differs between the two independent processes" % field)
    return True


def run_two_process(scenario: str, n: int, kobold_type_id=None) -> dict:
    """In-process two-run agreement (PROCESS_A twice + PROCESS_B both). Used by the
    self-test; real deployment uses run_two_process_real (separate OS processes)."""
    a = process_a_materialize(scenario, n, kobold_type_id)
    b = process_a_materialize(scenario, n, kobold_type_id)
    compare_two_processes(a, b)
    process_b_verify(a, kobold_type_id)
    process_b_verify(b, kobold_type_id)
    return {"agreement": True, "manifest": a}


def run_two_process_real(scenario: str, n: int, outdir: str, kobold_type_id=None) -> dict:
    """TWO INDEPENDENT OS PROCESSES: spawn fresh interpreters that each mint the bank
    via --materialize-real, then compare + PROCESS_B-verify both manifests here."""
    import subprocess
    require(ser.have_jax_craftax(),
            "FAIL CLOSED (BLOCKED_ENVIRONMENT): real two-process materialization requires "
            "JAX+craftax (jax=%s, craftax=%s)" % (ser.have_jax(), ser.have_craftax()))
    os.makedirs(outdir, exist_ok=True)
    script = os.path.abspath(__file__)
    manifest_paths = []
    for tag in ("a", "b"):
        out = os.path.join(outdir, "manifest_%s_%s.json" % (tag, scenario))
        cmd = [sys.executable, script, "--materialize-real", "--scenario", scenario,
               "--n", str(n), "--out", out]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        require(proc.returncode == 0,
                "FAIL CLOSED: --materialize-real process %s exited %d; stderr tail: %s"
                % (tag, proc.returncode, (proc.stderr or "")[-2000:]))
        manifest_paths.append(out)
    with open(manifest_paths[0], "r", encoding="utf-8") as fh:
        a = json.load(fh)
    with open(manifest_paths[1], "r", encoding="utf-8") as fh:
        b = json.load(fh)
    compare_two_processes(a, b)
    process_b_verify(a, kobold_type_id)
    process_b_verify(b, kobold_type_id)
    return {
        "two_process_agreement": True,
        "scenario": scenario,
        "hash_label": a["hash_label"],
        "state_bank_hash": a["state_bank_hash"],
        "field_manifest_sha256": a["field_manifest_sha256"],
        "state_count": a["state_count"],
        "manifest_paths": manifest_paths,
    }


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
    # On a JAX host this exercises the REAL path (n=2 to bound mint cost ≈16 mints);
    # without JAX it exercises the labelled SYNTHETIC protocol path (n=4).
    real = ser.have_jax_craftax()
    n_self = 2 if real else 4
    front = run_two_process(FRONT, n_self)
    back = run_two_process(BACK, n_self, SYNTHETIC_KOBOLD_TYPE_ID)
    check("front_two_process_agreement", front["agreement"] is True)
    check("back_two_process_agreement", back["agreement"] is True)
    check("front_label_not_global", front["manifest"]["hash_label"] == HASH_LABELS[FRONT])
    check("back_label_not_global", back["manifest"]["hash_label"] == HASH_LABELS[BACK])
    if real:
        check("front_hash_status_materialized", front["manifest"]["hash_status"] == "MATERIALIZED")
        check("front_states_are_real", front["manifest"]["states_are"] == "REAL_ENVSTATE")
        check("back_states_are_real", back["manifest"]["states_are"] == "REAL_ENVSTATE")
        check("bank_field_manifest_present",
              isinstance(front["manifest"]["field_manifest"], list)
              and len(front["manifest"]["field_manifest"]) > 0)
        check("back_resolved_kobold_type_id",
              back["manifest"].get("resolved_kobold_type_id")
              == audit.CRAFTAX_RUNTIME_BINDINGS["kobold"]["type_id"])
    else:
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
    status = "MATERIALIZED" if ser.have_jax_craftax() else "NOT_MATERIALIZED"
    print("TIER3_STATE_BANK_MATERIALIZER_SELF_TEST_PASS (scenarios=2, two_process=agree, "
          "hash_status=%s, env=%s)" % (status, ser.environment_status()))
    return 0


def _opt_value(argv, flag, default=None):
    if flag in argv:
        return argv[argv.index(flag) + 1]
    return default


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Real materialization imports minicraftax, which lives under <repo>/dicode_src/src
    # (audited relpaths). Make it importable for fresh subprocess interpreters too.
    _src = str(audit.repo_root() / "dicode_src" / "src")
    if os.path.isdir(_src) and _src not in sys.path:
        sys.path.insert(0, _src)
    if "--self-test" in argv:
        return self_test()
    if "--materialize-real" in argv:
        scenario = _opt_value(argv, "--scenario", FRONT)
        n = int(_opt_value(argv, "--n", "4"))
        out = _opt_value(argv, "--out")
        require(out is not None, "FAIL CLOSED: --materialize-real requires --out <path>")
        manifest = process_a_materialize(scenario, n)
        require(manifest["states_are"] == "REAL_ENVSTATE",
                "FAIL CLOSED (BLOCKED_ENVIRONMENT): --materialize-real requires JAX+craftax")
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False, sort_keys=True)
            fh.write("\n")
        print("materialized REAL %s bank (n=%d) -> %s | %s=%s"
              % (scenario, n, out, manifest["hash_label"], manifest["state_bank_hash"]))
        return 0
    if "--two-process-real" in argv:
        scenario = _opt_value(argv, "--scenario", FRONT)
        n = int(_opt_value(argv, "--n", "4"))
        outdir = _opt_value(argv, "--outdir")
        require(outdir is not None, "FAIL CLOSED: --two-process-real requires --outdir <dir>")
        summary = run_two_process_real(scenario, n, outdir)
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    if "--json" in argv:
        scenario = BACK if "--back" in argv else FRONT
        print(json.dumps(process_a_materialize(scenario, 4),
                         indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    print("usage: tier3_state_bank_materializer.py --self-test | --json [--back] | "
          "--materialize-real --scenario <front_l2|back_l2> --n <N> --out <path> | "
          "--two-process-real --scenario <front_l2|back_l2> --n <N> --outdir <dir>")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
