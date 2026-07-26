"""GATE 4 — shared frozen candidate pool integrity.

Verifies: deterministic pool_hash; frozen=True; empty rejected; duplicate task_id
rejected; no-overwrite persistence; load re-validates + re-hashes; tamper
detection; selectors hard-fail on a pool_hash mismatch (shared-pool invariant)."""
import json

import pytest

from d052.achievements import AchievementError
from d052.generation.pool import (
    PoolError,
    SharedFrozenPoolStore,
    build_pool,
)
from d052.schemas.candidate import CandidatePool


def raw(task_id, names=("collect_wood",)):
    return {"task_id": task_id,
            "task_params": {"passive_spawn_multiplier": 1.0,
                            "melee_spawn_multiplier": 1.0,
                            "mob_health_multiplier": 1.0,
                            "mob_damage_multiplier": 1.0},
            "target_achievements": list(names)}


def test_build_pool_valid_and_frozen():
    pool = build_pool("p", [raw("t1"), raw("t2", ["defeat_archer"])])
    assert pool.frozen is True
    assert pool.candidate_count == 2
    assert len(pool.pool_hash) == 64


def test_build_pool_deterministic_hash():
    a = build_pool("p", [raw("t1"), raw("t2", ["defeat_archer"])])
    b = build_pool("p", [raw("t1"), raw("t2", ["defeat_archer"])])
    assert a.pool_hash == b.pool_hash


def test_build_pool_order_sensitive_hash():
    # pool_hash is over the ORDERED candidate chashes -> order matters
    a = build_pool("p", [raw("t1"), raw("t2")])
    b = build_pool("p", [raw("t2"), raw("t1")])
    assert a.pool_hash != b.pool_hash


def test_build_pool_empty_rejected():
    with pytest.raises(PoolError) as ei:
        build_pool("p", [])
    assert ei.value.code == PoolError.EMPTY_POOL


def test_build_pool_duplicate_task_id_rejected():
    with pytest.raises(Exception) as ei:
        build_pool("p", [raw("t1"), raw("t1", ["defeat_archer"])])
    assert "DUPLICATE_TASK_ID" in str(ei.value)


def test_build_pool_unknown_target_propagates():
    with pytest.raises(AchievementError) as ei:
        build_pool("p", [raw("t1", ["defeat_dragon"])])
    assert ei.value.code == AchievementError.UNKNOWN_ACHIEVEMENT


# --- store: no-overwrite + roundtrip + tamper -------------------------------

def test_store_write_load_roundtrip(tmp_path):
    store = SharedFrozenPoolStore(str(tmp_path))
    pool = build_pool("p", [raw("t1"), raw("t2", ["defeat_archer"])])
    path = store.write(pool)
    assert path.endswith("pool.json")
    loaded = store.load("p")
    assert loaded.pool_hash == pool.pool_hash
    assert loaded.candidate_count == 2


def test_store_refuses_overwrite(tmp_path):
    store = SharedFrozenPoolStore(str(tmp_path))
    pool = build_pool("p", [raw("t1")])
    store.write(pool)
    with pytest.raises(PoolError) as ei:
        store.write(pool)  # second write -> NO_LEGACY_ARTIFACT_OVERWRITE
    assert ei.value.code == PoolError.EXISTS_NO_OVERWRITE


def test_store_load_missing(tmp_path):
    store = SharedFrozenPoolStore(str(tmp_path))
    with pytest.raises(PoolError) as ei:
        store.load("does_not_exist")
    assert ei.value.code == PoolError.NOT_FOUND


def test_store_tamper_detection(tmp_path):
    store = SharedFrozenPoolStore(str(tmp_path))
    pool = build_pool("p", [raw("t1"), raw("t2")])
    store.write(pool)
    # tamper: change a candidate's task_id without fixing pool_hash
    pool_path = tmp_path / "p" / "pool.json"
    data = json.loads(pool_path.read_text(encoding="utf-8"))
    data["candidates"][0]["task_id"] = "TAMPERED"
    pool_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(Exception) as ei:
        store.load("p")
    assert "POOL_HASH_MISMATCH" in str(ei.value) or "HASH_MISMATCH" in str(ei.value)


def test_shared_pool_invariant_selectors_must_match(tmp_path):
    store = SharedFrozenPoolStore(str(tmp_path))
    pool = build_pool("p", [raw("t1")])
    store.write(pool)
    loaded = store.load("p")
    # matching hash passes
    SharedFrozenPoolStore.assert_pool_matches(loaded, pool.pool_hash)
    # mismatched hash hard-fails (a selector handed a different pool)
    with pytest.raises(PoolError) as ei:
        SharedFrozenPoolStore.assert_pool_matches(loaded, "f" * 64)
    assert ei.value.code == PoolError.POOL_HASH_MISMATCH


def test_store_never_writes_into_existing_dir(tmp_path):
    # simulate a pre-existing (legacy) experiment dir with the same pool_id
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / "legacy_artifact.json").write_text("{}", encoding="utf-8")
    store = SharedFrozenPoolStore(str(tmp_path))
    pool = build_pool("p", [raw("t1")])
    with pytest.raises(PoolError) as ei:
        store.write(pool)
    assert ei.value.code == PoolError.EXISTS_NO_OVERWRITE
    # legacy file untouched
    assert (tmp_path / "p" / "legacy_artifact.json").read_text() == "{}"
