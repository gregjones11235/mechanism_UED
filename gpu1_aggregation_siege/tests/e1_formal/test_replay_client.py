"""C5 tests: replay-only LLM client + 7-field immutable keys.

Offline discipline: a cache miss is a HARD FAIL (no fallback, no live
call), record mode is disabled this round (keeps REAL_ENVCODER_USED ==
false provable), and answers depend ONLY on the cache key.
"""
import pytest

from dicode.mechanisms.immutable_cache import compute_immutable_cache_key
from dicode.teachers.e1_formal import llm_client as LC
from dicode.teachers.e1_formal.manifest import (
    E1_REPLAY_MODEL_ID,
    E1_REPLAY_PROVIDER,
)


def _key(**overrides):
    base = dict(
        role="student_modeler",
        evidence_hash="e" * 64,
        prompt_envelope_hash="p" * 64,
        prompt_version="e1-board-prompt-v1",
        schema_version="e1-role-output-v1",
    )
    base.update(overrides)
    return LC.make_replay_key(**base)


class TestMakeReplayKey:
    def test_key_is_deterministic(self):
        assert _key() == _key()

    def test_key_matches_shared_7field_helper(self):
        # make_replay_key must be a thin wrapper over the shared helper
        expected = compute_immutable_cache_key(
            task_code_hash="p" * 64,
            student_stage_id="e" * 64,
            role="student_modeler",
            provider=E1_REPLAY_PROVIDER,
            exact_model_id=E1_REPLAY_MODEL_ID,
            prompt_version="e1-board-prompt-v1",
            schema_version="e1-role-output-v1",
        )
        assert _key() == expected

    def test_pinned_provider_and_model_id_appear_in_key(self):
        key = _key()
        assert E1_REPLAY_PROVIDER in key
        assert E1_REPLAY_MODEL_ID in key

    @pytest.mark.parametrize(
        "field",
        ["role", "evidence_hash", "prompt_envelope_hash",
         "prompt_version", "schema_version"],
    )
    def test_every_identity_field_changes_the_key(self, field):
        assert _key(**{field: "x" * 64}) != _key()

    @pytest.mark.parametrize(
        "field",
        ["role", "evidence_hash", "prompt_envelope_hash",
         "prompt_version", "schema_version"],
    )
    def test_empty_fields_rejected_fail_closed(self, field):
        with pytest.raises(ValueError):
            _key(**{field: ""})

    @pytest.mark.parametrize("alias", ["latest", "auto", "latest-v2", "my_auto"])
    def test_latest_auto_aliases_rejected(self, alias):
        with pytest.raises(ValueError):
            _key(prompt_version=alias)


class TestReplayClientQuery:
    def test_hit_returns_content_list(self):
        key = _key()
        client = LC.ReplayLLMClient({key: '{"ok": true}'}, "t")
        out = client.query(
            "system", ["user"], cache_key=key, role="student_modeler"
        )
        assert out == [{"content": '{"ok": true}'}]

    def test_answer_depends_only_on_cache_key(self):
        # Different prompts with the same key must give the same answer:
        # determinism is keyed, not prompt-dependent.
        key = _key()
        client = LC.ReplayLLMClient({key: "answer-A"}, "t")
        a = client.query("s1", ["u1"], cache_key=key, role="explorer")
        b = client.query("TOTALLY DIFFERENT", ["other"], cache_key=key, role="explorer")
        assert a == b == [{"content": "answer-A"}]

    def test_miss_is_hard_fail_with_role_and_key(self):
        client = LC.ReplayLLMClient({}, "t")
        with pytest.raises(RuntimeError) as excinfo:
            client.query("s", ["u"], cache_key="missing_key", role="critic")
        msg = str(excinfo.value)
        assert msg.startswith(LC.HARD_FAIL_PREFIX)
        assert "role=critic" in msg
        assert "key=missing_key" in msg

    def test_empty_cache_key_rejected(self):
        client = LC.ReplayLLMClient({"k": "v"}, "t")
        with pytest.raises(ValueError):
            client.query("s", ["u"], cache_key="", role="critic")

    def test_double_run_equality(self):
        key = _key()
        c1 = LC.ReplayLLMClient({key: "payload"}, "t1")
        c2 = LC.ReplayLLMClient({key: "payload"}, "t2")
        assert c1.query("s", ["u"], cache_key=key, role="explorer") == \
            c2.query("s", ["u"], cache_key=key, role="explorer")


class TestReplayClientDiscipline:
    def test_record_mode_is_disabled_this_round(self):
        assert LC.E1_RECORD_MODE_DISABLED_THIS_ROUND is True
        client = LC.ReplayLLMClient({}, "t")
        with pytest.raises(RuntimeError) as excinfo:
            client.record("anything", key="k")
        msg = str(excinfo.value)
        assert "HARD FAIL" in msg
        assert "record mode is disabled" in msg
        assert "REAL_ENVCODER_USED" in msg

    def test_store_is_defensively_copied(self):
        key = _key()
        store = {key: "original"}
        client = LC.ReplayLLMClient(store, "t")
        store[key] = "tampered-after-construction"
        out = client.query("s", ["u"], cache_key=key, role="explorer")
        assert out == [{"content": "original"}]

    def test_store_must_be_mapping(self):
        with pytest.raises(TypeError):
            LC.ReplayLLMClient([("k", "v")], "t")

    @pytest.mark.parametrize("bad_store", [{1: "v"}, {"k": 1}])
    def test_store_must_map_str_to_str(self, bad_store):
        with pytest.raises(TypeError):
            LC.ReplayLLMClient(bad_store, "t")
