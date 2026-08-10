import pytest

from dicode.session_boundary import (
    BoundaryIntegrityError,
    BoundaryStore,
    verify_reference_digests,
)


def test_boundary_round_trip_and_latest(tmp_path):
    store = BoundaryStore(tmp_path / "boundaries")
    manifest = store.write(
        session_idx=3,
        global_update_step=120,
        global_env_steps=1_572_864,
        state={"rng": [1, 2], "task_ids": ["a", "b"], "original_return": 0.25},
        references={"train_state": "abc", "task_graph": "def"},
        provenance={"commit": "base", "gpu_uuid": "GPU-test"},
    )

    loaded_manifest, state = store.read(3)
    assert loaded_manifest == manifest
    assert state["task_ids"] == ["a", "b"]
    assert store.latest()[0] == manifest


def test_boundary_is_immutable_and_fail_closed(tmp_path):
    store = BoundaryStore(tmp_path / "boundaries")
    store.write(
        session_idx=0,
        global_update_step=1,
        global_env_steps=2,
        state={"rng": 7},
        references={"train_state": "abc"},
        provenance={},
    )
    with pytest.raises(BoundaryIntegrityError):
        store.write(
            session_idx=0,
            global_update_step=2,
            global_env_steps=3,
            state={"rng": 8},
            references={"train_state": "abc"},
            provenance={},
        )

    payload = tmp_path / "boundaries" / "session_000000" / "payload.pkl"
    payload.write_bytes(payload.read_bytes() + b"tamper")
    with pytest.raises(BoundaryIntegrityError):
        store.read(0)


def test_reference_mapping_must_match_exactly(tmp_path):
    store = BoundaryStore(tmp_path / "boundaries")
    manifest = store.write(
        session_idx=1,
        global_update_step=1,
        global_env_steps=2,
        state={},
        references={"train_state": "abc", "task_graph": "def"},
        provenance={},
    )
    verify_reference_digests(manifest, {"train_state": "abc", "task_graph": "def"})
    with pytest.raises(BoundaryIntegrityError):
        verify_reference_digests(manifest, {"train_state": "abc"})
