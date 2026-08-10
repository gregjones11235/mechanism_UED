# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

BUG-E3-07/08/09: the canonical FULL RunState codec (params + opt_state +
step + RNG + session + archive + plan + bundle hash) must round-trip and
restore in an INDEPENDENT process.  A params-only snapshot is never a full
run state.
"""

import os
import pickle

import pytest

from dicode.simulator_frontier.runstate_codec import (
    REQUIRED_RUNSTATE_FIELDS,
    RunStateCheckpointManager,
    RunStateError,
    build_full_run_state,
    fresh_process_restore,
    next_policy_step_hash,
    runstate_content_hash,
)

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True


def _mini_train_state():
    import numpy as np
    from types import SimpleNamespace
    return SimpleNamespace(
        params={"w": np.arange(6, dtype=np.float32).reshape(2, 3)},
        opt_state={"m": np.zeros((2, 3), dtype=np.float32)},
        step=7)


def _run_state(ts=None):
    ts = ts or _mini_train_state()
    return build_full_run_state(
        rl_train_state=ts,
        training_rng=123,
        env_rng=456,
        global_update_step=10,
        global_env_steps=200,
        current_session_idx=2,
        task_archive_identity="a" * 64,
        plan_hash="b" * 64,
        runtime_bundle_hash="c" * 64,
        config_hash="d" * 64,
        source_commit="src-sha256:test")


class TestBuildFullRunState:
    def test_all_required_fields_present(self):
        run_state = _run_state()
        for name in REQUIRED_RUNSTATE_FIELDS:
            assert name in run_state, name

    def test_incomplete_source_refused(self):
        from types import SimpleNamespace
        with pytest.raises(RunStateError):
            build_full_run_state(
                rl_train_state=SimpleNamespace(params={"w": 1}, opt_state=None),
                training_rng=1, env_rng=2, global_update_step=0,
                global_env_steps=0, current_session_idx=0,
                task_archive_identity="a" * 64, plan_hash="b" * 64,
                runtime_bundle_hash="c" * 64, config_hash="d" * 64,
                source_commit="test")


class TestSaveRestore:
    def test_save_refuses_incomplete_state(self, tmp_path):
        manager = RunStateCheckpointManager()
        with pytest.raises(RunStateError):
            manager.save({"params": {"w": 1}}, str(tmp_path / "ckpt"))

    def test_round_trip_content_hash(self, tmp_path):
        manager = RunStateCheckpointManager()
        run_state = _run_state()
        report = manager.save(run_state, str(tmp_path / "ckpt"),
                              idempotency_token="tok")
        assert report["checkpoint_hash"]
        restored = manager.restore(str(tmp_path / "ckpt"))
        assert restored["metadata"]["idempotency_token"] == "tok"
        assert runstate_content_hash(restored["run_state"]) \
            == runstate_content_hash(run_state)

    def test_tampered_state_rejected(self, tmp_path):
        manager = RunStateCheckpointManager()
        run_state = _run_state()
        manager.save(run_state, str(tmp_path / "ckpt"))
        state_path = str(tmp_path / "ckpt") + ".state.pkl"
        with open(state_path, "rb") as handle:
            payload = pickle.load(handle)
        payload["params"] = {"w": "TAMPERED"}
        with open(state_path, "wb") as handle:
            pickle.dump(payload, handle)
        with pytest.raises(RunStateError):
            manager.restore(str(tmp_path / "ckpt"))

    def test_meta_publish_failure_removes_new_state_and_allows_retry(self, tmp_path, monkeypatch):
        manager = RunStateCheckpointManager()
        run_state = _run_state()
        real_replace = os.replace
        calls = []

        def fail_meta(src, dst):
            calls.append((src, dst))
            if str(dst).endswith(".meta.json"):
                raise OSError("injected metadata replace failure")
            return real_replace(src, dst)

        monkeypatch.setattr(os, "replace", fail_meta)
        with pytest.raises(OSError):
            manager.save(run_state, str(tmp_path / "ckpt"), idempotency_token="retry")
        assert not list(tmp_path.glob("ckpt.*"))
        monkeypatch.setattr(os, "replace", real_replace)
        report = manager.save(run_state, str(tmp_path / "ckpt"), idempotency_token="retry")
        assert report["checkpoint_hash"]

    def test_existing_checkpoint_idempotent_and_conflict_refused(self, tmp_path):
        manager = RunStateCheckpointManager()
        run_state = _run_state()
        first = manager.save(run_state, str(tmp_path / "ckpt"), idempotency_token="same")
        assert manager.save(run_state, str(tmp_path / "ckpt"), idempotency_token="same") == first
        with pytest.raises(RunStateError):
            manager.save(run_state, str(tmp_path / "ckpt"), idempotency_token="different")


class TestFreshProcessRestore:
    def test_fresh_process_restore_equivalence(self, tmp_path):
        manager = RunStateCheckpointManager()
        run_state = _run_state()
        report = manager.save(run_state, str(tmp_path / "ckpt"))
        local_hash = runstate_content_hash(run_state)
        restored = fresh_process_restore(
            str(tmp_path / "ckpt"),
            extra_pythonpath=os.environ.get("E3_SRC_DIR", ""))
        assert restored["restored"] is True
        assert restored["content_hash"] == local_hash
        assert restored["checkpoint_hash"] == report["checkpoint_hash"]


class TestPolicyStateHash:
    def test_next_policy_step_hash_deterministic(self):
        ts = _mini_train_state()
        assert next_policy_step_hash(ts) == next_policy_step_hash(ts)
        assert len(next_policy_step_hash(ts)) == 64

    def test_next_policy_step_hash_changes_with_params(self):
        ts = _mini_train_state()
        ts2 = _mini_train_state()
        import numpy as np
        ts2.params["w"] = ts2.params["w"] + 1.0
        assert next_policy_step_hash(ts) != next_policy_step_hash(ts2)
