import asyncio
import json
from collections import OrderedDict
from types import SimpleNamespace

import pytest


def _llm(monkeypatch, enabled=True, limit=2):
    mod = pytest.importorskip("dicode.dreaming.llm")
    obj = mod.LLM.__new__(mod.LLM)
    obj.provider, obj.base_url, obj.model, obj.embedding_size = "local", "url", "m", 3
    obj._embedding_cache_enabled = enabled
    obj._embedding_cache_max_entries = limit
    obj._embedding_cache = OrderedDict()
    import threading
    obj._embedding_cache_lock = threading.RLock()
    calls = []
    async def fake(texts, instruction=None):
        calls.append((list(texts), instruction))
        return [{"input_text": t, "embedding": [len(t)], "embedding_dim": 1, "error": None} for t in texts]
    obj._query_local_embed = fake
    return obj, calls


def test_embedding_hit_mixed_duplicate_order_and_copy():
    obj, calls = _llm(None)
    first = obj.get_embedding(["a", "a", "b"], instruction="i")
    assert [x["input_text"] for x in first] == ["a", "a", "b"]
    assert calls == [(["a", "b"], "i")]
    first[0]["embedding"][0] = 99
    second = obj.get_embedding(["b", "a"], instruction="i")
    assert [x["input_text"] for x in second] == ["b", "a"]
    assert second[1]["embedding"][0] == 1
    assert len(calls) == 1
    assert isinstance(obj.get_embedding("c"), list)
    assert isinstance(obj.get_embedding(("d",)), list)


def test_embedding_key_changes_and_errors_not_cached():
    obj, calls = _llm(None)
    async def fail_once(texts, instruction=None):
        calls.append((list(texts), instruction))
        if len(calls) == 1:
            return [{"input_text": t, "embedding": None, "embedding_dim": 0, "error": "bad"} for t in texts]
        return [{"input_text": t, "embedding": [1], "embedding_dim": 1, "error": None} for t in texts]
    obj._query_local_embed = fail_once
    obj.get_embedding("a", instruction="i")
    obj.get_embedding("a", instruction="i")
    obj.get_embedding("a", instruction="j")
    obj.model = "other"
    obj.get_embedding("a", instruction="j")
    assert len(calls) == 4


def test_embedding_lru_eviction():
    obj, calls = _llm(None, limit=1)
    obj.get_embedding("a"); obj.get_embedding("b"); obj.get_embedding("a")
    assert len(calls) == 3


def test_embedding_disabled_uses_original_path():
    obj, calls = _llm(None, enabled=False)
    result = obj.get_embedding("a")
    assert calls == [(["a"], None)]
    assert result[0]["input_text"] == "a"


def test_validation_cache_success_failure_lru_and_disabled():
    mod = pytest.importorskip("dicode.dreaming.gen_manager")
    obj = mod.EnvGenerator.__new__(mod.EnvGenerator)
    import threading
    obj.performance = {"validation_cache": True, "validation_cache_max_entries": 2, "validation_static_lint": False}
    obj._validation_cache = OrderedDict()
    obj._validation_cache_lock = threading.RLock()
    obj._validation_inflight = {}
    obj._validation_source_sha = "test"
    calls = []
    obj._check_compilation_uncached = lambda code: (calls.append(code) or (False, "bad"))
    assert obj.check_compilation("a\r\nb") == (False, "bad")
    assert obj.check_compilation("a\nb") == (False, "bad")
    assert len(calls) == 1
    obj.check_compilation("b"); obj.check_compilation("c"); obj.check_compilation("a\r\nb")
    assert len(calls) == 4
    obj.performance["validation_cache"] = False
    obj.check_compilation("a\r\nb")
    assert len(calls) == 5


def test_validation_key_includes_version_source_and_lf_normalization():
    mod = pytest.importorskip("dicode.dreaming.gen_manager")
    obj = mod.EnvGenerator.__new__(mod.EnvGenerator)
    obj._validation_source_sha = "sha1"
    a = obj._validation_key("a\r\nb")
    b = obj._validation_key("a\nb")
    assert a == b
    assert a[1] == mod.VALIDATOR_CACHE_VERSION
    obj._validation_source_sha = "sha2"
    assert a != obj._validation_key("x\ny")


def test_validation_singleflight_and_static_lint():
    mod = pytest.importorskip("dicode.dreaming.gen_manager")
    obj = mod.EnvGenerator.__new__(mod.EnvGenerator)
    import threading, time
    obj.performance = {"validation_cache": True, "validation_cache_max_entries": 4, "validation_static_lint": False}
    obj._validation_cache = OrderedDict(); obj._validation_cache_lock = threading.RLock(); obj._validation_inflight = {}
    count = [0]
    def compile_once(code):
        count[0] += 1; time.sleep(0.03); return True, ""
    obj._check_compilation_uncached = compile_once
    threads = [threading.Thread(target=lambda: obj.check_compilation("same")) for _ in range(4)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert count[0] == 1
    obj.performance["validation_static_lint"] = True
    assert obj._static_lint("from craftax.craftax.constants import BlockType as B\nB.NOT_REAL")[0] is False
    assert obj._static_lint("from craftax.craftax.constants import BlockType as B\nB.STONE")[0] is True
    assert obj._static_lint("from craftax.craftax.craftax_state import Inventory as I\nI(bad_kw=1)")[0] is False
    from craftax.craftax.craftax_state import Inventory
    from dataclasses import fields
    valid = fields(Inventory)[0].name
    assert obj._static_lint(f"from craftax.craftax.craftax_state import Inventory as I\nI({valid}=1)")[0] is True


def test_llm_request_events_have_duration_status_and_no_prompt(tmp_path):
    mod = pytest.importorskip("dicode.dreaming.llm")
    from dicode.runtime_analysis import tracker
    tracker.configure(enabled=True, output_jsonl=tmp_path / "events.jsonl", run_id="llm", reset=True)
    obj = mod.LLM.__new__(mod.LLM)
    obj.provider = "local"; obj.base_url = ""; obj.model = "m"; obj.embedding_size = 2
    obj.max_tokens = obj.temperature = obj.top_p = None; obj.think = False
    class Chat:
        class completions:
            @staticmethod
            async def create(**kwargs):
                await asyncio.sleep(0.002)
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    class Emb:
        @staticmethod
        async def create(**kwargs):
            await asyncio.sleep(0.002)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[1, 2]) for _ in kwargs["input"]])
    obj.client = SimpleNamespace(chat=Chat, embeddings=Emb)
    result = asyncio.run(obj._query_local_gen("SECRET_PROMPT", "SECRET_USER"))
    assert result["content"] == "ok"
    asyncio.run(obj._query_local_embed(["x"]))
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert {row["phase"] for row in rows} == {"chat_request", "embedding_request"}
    assert all(row["duration_s"] >= 0 and row["status"] == "ok" for row in rows)
    assert all("SECRET_PROMPT" not in json.dumps(row) and "SECRET_USER" not in json.dumps(row) for row in rows)


def test_validation_profile_phases_include_static_lint_and_wait(tmp_path):
    mod = pytest.importorskip("dicode.dreaming.gen_manager")
    from dicode.runtime_analysis import tracker
    tracker.configure(enabled=True, output_jsonl=tmp_path / "events.jsonl", run_id="validation", reset=True)
    obj = mod.EnvGenerator.__new__(mod.EnvGenerator)
    import threading, time
    obj.performance = {"validation_cache": True, "validation_cache_max_entries": 4, "validation_static_lint": True}
    obj._validation_cache = OrderedDict(); obj._validation_cache_lock = threading.RLock(); obj._validation_inflight = {}
    obj._check_compilation_uncached = lambda code: (time.sleep(0.01) or (True, ""))
    # Static-lint rejection is recorded under its dedicated phase.
    obj.check_compilation("from craftax.craftax.constants import BlockType as B\nB.NOT_REAL")
    threads = [threading.Thread(target=lambda: obj.check_compilation("same")) for _ in range(2)]
    [thread.start() for thread in threads]; [thread.join() for thread in threads]
    phases = [json.loads(line)["phase"] for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert "candidate_validation_static_lint" in phases
    assert "candidate_validation_wait" in phases
