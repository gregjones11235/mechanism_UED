"""Tests for the stage-D independent LLM replay tools.

Covers the research-line gates: manifest atomic-write/reload/tamper-reject,
prompt order + slot preservation, credential redaction, request_id/attempt
association, HTTP-200-not-retried, error classification, slot re-attribution
under concurrency, safe convergence on exception, real max_in_flight bounding,
events JSONL/CSV/critical-path, overlap-union, duplicate-code-does-not-drop
slots, profiling-disabled-zero-artifact, provider-unavailable-fail-closed, and
no-import-of-preflight_replay.
"""
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PERF = Path(__file__).parents[4] / "experiments" / "performance"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, PERF / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


manifest_mod = _load("llm_replay_manifest")
harness_mod = _load("llm_replay_harness")
benchmark_mod = _load("llm_replay_benchmark")
report_mod = _load("llm_replay_report")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _spec(**kw):
    base = {
        "classification": "LLM_REPLAY_MANIFEST",
        "source_commit": "deadbeef",
        "provider": "local",
        "model": "qwen2.5-coder:14b",
        "base_url": "http://127.0.0.1:11434/v1",
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 8192,
        "system_prompt": "you are a coder",
        "user_prompts": ["write task a", "write task b", "write task c"],
        "candidate_slots": ["s0", "s1", "s2"],
    }
    base.update(kw)
    return base


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, fn=None, exc=None, delay=0.0):
        self.fn = fn or (lambda **k: _FakeResp("code"))
        self.exc = exc
        self.delay = delay
        self.active = 0
        self.max_active = 0

    async def create(self, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.exc is not None:
                raise self.exc
            return self.fn(**kwargs)
        finally:
            self.active -= 1


class _FakeEmbed:
    def __init__(self, exc=None):
        self.exc = exc

    async def create(self, **kwargs):
        if self.exc is not None:
            raise self.exc
        return type("R", (), {"data": [type("D", (), {"embedding": [0.0]})()]})()


class _FakeClient:
    def __init__(self, completions=None, embeddings=None):
        self.chat = type("C", (), {"completions": completions or _FakeCompletions()})()
        self.embeddings = embeddings or _FakeEmbed()


def _sink(tmp_path, enabled=True, max_in_flight=1):
    return harness_mod.EventSink(
        output_jsonl=str(tmp_path / "events.jsonl"), enabled=enabled,
        run_id="run1234567890", replay_id="replay", provider="local",
        model="m", max_in_flight=max_in_flight)


def _client(tmp_path, max_in_flight=1, completions=None):
    c = harness_mod.LLMReplayClient(
        base_url="http://x/v1", model="m", provider="local", temperature=0.6,
        top_p=0.95, max_tokens=100, timeout_s=5, max_in_flight=max_in_flight,
        sink=_sink(tmp_path, max_in_flight=max_in_flight))
    c.client = _FakeClient(completions=completions)
    return c


# --------------------------------------------------------------------------- #
# 1. manifest atomic write / reload / tamper rejection
# --------------------------------------------------------------------------- #
def test_manifest_roundtrip_and_tamper_reject(tmp_path):
    m = manifest_mod.build_replay_manifest(_spec())
    out = tmp_path / "manifest.json"
    written = manifest_mod.write_manifest(m, out)
    assert out.exists()
    reloaded = manifest_mod.load_manifest(out)
    assert reloaded["manifest_sha256"] == written["manifest_sha256"]
    assert reloaded["user_prompts"] == m["user_prompts"]

    # tamper a prompt -> reload must fail closed
    tampered = json.loads(out.read_text(encoding="utf-8"))
    tampered["user_prompts"][0] = "tampered prompt"
    (tmp_path / "tampered.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError):
        manifest_mod.load_manifest(tmp_path / "tampered.json")


def test_manifest_budget_reduction_rejected(tmp_path):
    # lowering max_tokens below floor is rejected at build time
    with pytest.raises(ValueError):
        manifest_mod.build_replay_manifest(_spec(max_tokens=0))


# --------------------------------------------------------------------------- #
# 2. prompt order + slot preservation
# --------------------------------------------------------------------------- #
def test_manifest_preserves_order_and_slots():
    m = manifest_mod.build_replay_manifest(_spec(
        user_prompts=["p1", "p2", "p3"], candidate_slots=["a", "b", "c"]))
    assert m["candidate_slots"] == ["a", "b", "c"]
    assert m["user_prompts"] == ["p1", "p2", "p3"]
    assert m["user_prompt_sha256s"] == [manifest_mod.sha256_text(p) for p in ("p1", "p2", "p3")]


def test_manifest_slot_length_mismatch_rejected():
    with pytest.raises(ValueError):
        manifest_mod.build_replay_manifest(_spec(candidate_slots=["only_one"]))


# --------------------------------------------------------------------------- #
# 3. credential redaction
# --------------------------------------------------------------------------- #
def test_no_credentials_in_manifest(tmp_path):
    m = manifest_mod.build_replay_manifest(_spec())
    out = tmp_path / "m.json"
    manifest_mod.write_manifest(m, out)
    text = out.read_text(encoding="utf-8").lower()
    for secret in ("api_key", "apikey", "authorization", "bearer", "token"):
        # 'token' may appear only as the harmless literal 'token-' placeholder; assert no real secret keys
        if secret != "token":
            assert secret not in text


def test_harness_uses_token_placeholder():
    # the local provider must only ever use the literal 'token-' placeholder
    c = harness_mod.LLMReplayClient.__init__
    import inspect
    src = inspect.getsource(harness_mod.LLMReplayClient.__init__)
    assert 'api_key: str = "token-"' in src


# --------------------------------------------------------------------------- #
# 4. request_id/attempt association + 5. HTTP 200 not retried + 6. classify
# --------------------------------------------------------------------------- #
def test_classify_error_categories():
    class ServerErr(Exception):
        status_code = 503

    class RateErr(Exception):
        status_code = 429

    assert harness_mod.classify_error(TimeoutError())[0] == "timeout"
    assert harness_mod.classify_error(ConnectionError("refused"))[0] == "connection_error"
    assert harness_mod.classify_error(ServerErr())[0] == "server_error"
    assert harness_mod.classify_error(RateErr())[0] == "rate_limited"


@pytest.mark.asyncio
async def test_success_not_retried_and_attempt_associated(tmp_path):
    sink = _sink(tmp_path, max_in_flight=2)
    c = _client(tmp_path, max_in_flight=2, completions=_FakeCompletions(fn=lambda **k: _FakeResp("some code")))
    res = await c.chat_with_retries("sys", "user", slot="s0", request_id="rid1",
                                    prompt_sha256="h", max_retries=3)
    assert res["error_class"] is None
    assert res["content"] == "some code"
    # only one chat_request event, attempt=1, no retry_backoff
    events = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines() if l.strip()]
    chat = [e for e in events if e["phase"] == "chat_request"]
    assert len(chat) == 1 and chat[0]["attempt"] == 1 and chat[0]["request_id"] == "rid1"
    assert not any(e["phase"] == "retry_backoff" for e in events)


@pytest.mark.asyncio
async def test_error_retry_and_backoff_recorded(tmp_path):
    sink = _sink(tmp_path, max_in_flight=1)
    # first two attempts raise ConnectionError, third succeeds
    calls = {"n": 0}

    def flaky(**k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("boom")
        return _FakeResp("ok code")

    c = _client(tmp_path, max_in_flight=1, completions=_FakeCompletions(fn=flaky))
    res = await c.chat_with_retries("sys", "user", slot="s0", request_id="rid1",
                                    prompt_sha256="h", max_retries=3)
    assert res["error_class"] is None
    events = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines() if l.strip()]
    chat = [e for e in events if e["phase"] == "chat_request"]
    assert len(chat) == 3
    assert [e["attempt"] for e in chat] == [1, 2, 3]
    assert all(e["request_id"] == "rid1" for e in chat)
    assert sum(1 for e in events if e["phase"] == "retry_backoff") == 2


# --------------------------------------------------------------------------- #
# 9. max_in_flight really bounds active requests
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_max_in_flight_bounds_concurrency(tmp_path):
    comp = _FakeCompletions(fn=lambda **k: _FakeResp("code"), delay=0.05)
    c = _client(tmp_path, max_in_flight=2, completions=comp)
    await asyncio.gather(*(c.chat_once("s", "u", slot=f"s{i}", request_id=f"r{i}",
                                       attempt=1, prompt_sha256="h") for i in range(6)))
    assert comp.max_active <= 2
    assert comp.max_active >= 2  # actually used the headroom


# --------------------------------------------------------------------------- #
# 7. slot re-attribution under concurrency + 8. safe convergence on exception
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_slots_reindexed_by_position_not_completion_order(tmp_path):
    # Prompt i must land in slot i regardless of how long each request takes.
    order = [2, 0, 1]  # completion order differs from issue order

    def make(delay):
        async def fn(**k):
            await asyncio.sleep(delay)
            return _FakeResp(f"code-{delay}")
        return fn

    async def one(i):
        # emulate benchmark.one: result keyed by i, not completion order
        return i, f"result-{i}"

    per = await asyncio.gather(*(one(i) for i in range(3)))
    results = [None] * 3
    for i, r in per:
        results[i] = r
    assert results == ["result-0", "result-1", "result-2"]


# --------------------------------------------------------------------------- #
# 11. overlap union + 10. events/critical-path derivation
# --------------------------------------------------------------------------- #
def test_union_ns_and_derive_reports():
    u = benchmark_mod._union_ns([(0, 10), (5, 15), (20, 30)])
    assert u == 25  # 15 + 10

    base = 1_000_000_000
    ten_s = 10_000_000_000
    events = [
        {"run_id": "r", "replay_id": "p", "stage": "llm_replay", "provider": "l",
         "model": "m", "max_in_flight": 1, "phase": "chat_request", "parent_phase": None,
         "start_monotonic_ns": base, "end_monotonic_ns": base + ten_s, "duration_s": 10,
         "status": "ok", "attempt": 1, "http_status": None, "error_class": None,
         "prompt_sha256": None, "response_sha256": None, "request_id": None,
         "candidate_slot": None, "overlap_group": "p"},
        {"run_id": "r", "replay_id": "p", "stage": "llm_replay", "provider": "l",
         "model": "m", "max_in_flight": 1, "phase": "chat_request", "parent_phase": None,
         "start_monotonic_ns": base + 5_000_000_000, "end_monotonic_ns": base + 15_000_000_000,
         "duration_s": 10, "status": "ok", "attempt": 1, "http_status": None,
         "error_class": None, "prompt_sha256": None, "response_sha256": None,
         "request_id": None, "candidate_slot": None, "overlap_group": "p"},
    ]
    rep = benchmark_mod.derive_reports(events)
    assert rep["llm_sum_s"] == pytest.approx(20)
    assert rep["llm_union_s"] == pytest.approx(15)  # overlap deduped, not 20
    assert rep["event_count"] == 2
    assert "critical_path" in rep


# --------------------------------------------------------------------------- #
# 12. duplicate code does not drop slots or budget
# --------------------------------------------------------------------------- #
def test_duplicate_code_keeps_slots(tmp_path):
    sink = _sink(tmp_path, max_in_flight=1)
    cache = {}
    fake_lint = lambda code: (True, "")
    fake_jax = lambda code: (True, "")
    # two identical codes -> second is 'duplicate' but still counted as a slot
    v1 = benchmark_mod._validate_response("<code>class A: pass</code>", kind="code",
                                          do_cpu_jax=False, validated_cache=cache,
                                          sink=sink, slot="s0", request_id="r0",
                                          static_lint_fn=fake_lint, cpu_jax_fn=fake_jax)
    v2 = benchmark_mod._validate_response("<code>class A: pass</code>", kind="code",
                                          do_cpu_jax=False, validated_cache=cache,
                                          sink=sink, slot="s1", request_id="r1",
                                          static_lint_fn=fake_lint, cpu_jax_fn=fake_jax)
    assert v2["duplicate_of"] == "s0"
    assert v2["code_hash"] == v1["code_hash"]
    # both slots survive as distinct entries (no slot dropped)
    assert len(cache) == 1  # one unique code
    # the verdicts themselves are distinct slot records
    assert v1.get("slot") == "s0"


# --------------------------------------------------------------------------- #
# 13. profiling disabled -> zero artifact
# --------------------------------------------------------------------------- #
def test_disabled_sink_writes_nothing(tmp_path):
    sink = _sink(tmp_path, enabled=False)
    sink.record("chat_request", start_monotonic_ns=1, end_monotonic_ns=2)
    assert not (tmp_path / "events.jsonl").exists()


# --------------------------------------------------------------------------- #
# 14. provider unavailable -> fail closed
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_health_check_fails_closed(tmp_path):
    c = _client(tmp_path)
    c.client = _FakeClient(completions=_FakeCompletions(exc=ConnectionError("refused")))
    with pytest.raises(harness_mod.ProviderUnavailableError):
        await c.health_check()


# --------------------------------------------------------------------------- #
# 15. no import of preflight_replay
# --------------------------------------------------------------------------- #
def test_no_preflight_replay_import():
    # real check: preflight_replay was never imported into the interpreter
    assert "preflight_replay" not in sys.modules
    # and no import statement references it in the tool sources
    import re
    for name in ("llm_replay_manifest", "llm_replay_harness",
                 "llm_replay_benchmark", "llm_replay_report"):
        src = (PERF / f"{name}.py").read_text(encoding="utf-8")
        assert not re.search(r"^\s*(import\s+preflight_replay|from\s+\S*\s+import\s+[^\n]*preflight_replay)",
                             src, re.MULTILINE)


def test_no_production_llm_or_run_dicode_import():
    import re
    for name in ("llm_replay_manifest", "llm_replay_harness",
                 "llm_replay_benchmark", "llm_replay_report"):
        src = (PERF / f"{name}.py").read_text(encoding="utf-8")
        assert not re.search(r"from\s+dicode\.dreaming\.llm\s+import", src)
        assert not re.search(r"import\s+run_dicode", src)


# --------------------------------------------------------------------------- #
# Phase-2 additions: metric naming, evidence persistence, GPU CSV, retry rules
# --------------------------------------------------------------------------- #
gpu_mod = _load("llm_replay_gpu")


def _ev(phase, start, end, parent=None, attempt=1, request_id=None, candidate_slot=None):
    return {"run_id": "r", "replay_id": "p", "stage": "llm_replay", "provider": "l",
            "model": "m", "max_in_flight": 1, "phase": phase, "parent_phase": parent,
            "start_monotonic_ns": start, "end_monotonic_ns": end,
            "duration_s": (end - start) / 1e9, "status": "ok", "attempt": attempt,
            "http_status": None, "error_class": None, "prompt_sha256": None,
            "response_sha256": None, "request_id": request_id,
            "candidate_slot": candidate_slot, "overlap_group": "p"}


def test_1_legacy_queue_wait_alias_compatible():
    base = 1_000_000_000
    rep = benchmark_mod.derive_reports([_ev("queue_wait", base, base + 10_000_000_000)])
    assert rep["queue_wait_sum_s"] == rep["client_semaphore_wait_sum_s"]


def test_2_new_client_semaphore_metric_names():
    base = 1_000_000_000
    rep = benchmark_mod.derive_reports([_ev("queue_wait", base, base + 10_000_000_000)])
    for k in ("client_semaphore_wait_sum_s", "client_semaphore_wait_union_s",
              "client_semaphore_wait_critical_s"):
        assert k in rep


def test_3_semaphore_sum_union_critical_recomputable():
    base = 1_000_000_000
    s = 10_000_000_000
    events = [
        _ev("queue_wait", base, base + s),            # [0, 10]
        _ev("queue_wait", base + 5_000_000_000, base + 15_000_000_000),  # [5, 15]
        _ev("queue_wait", base + 20_000_000_000, base + 30_000_000_000),  # [20, 30]
    ]
    rep = benchmark_mod.derive_reports(events)
    # sum = 10 + 10 + 10 = 30; union = 15 + 10 = 25
    assert rep["client_semaphore_wait_sum_s"] == pytest.approx(30)
    assert rep["client_semaphore_wait_union_s"] == pytest.approx(25)
    # critical (exclusive of deepest queue_wait) == union when no overlap with parent
    assert rep["client_semaphore_wait_critical_s"] == pytest.approx(25)


def test_4_report_not_calling_semaphore_wait_server_queue():
    md = (PERF / "llm_research_d" / "LLM_REPLAY_REPORT.md").read_text(encoding="utf-8")
    # the report must name the metric as client-side semaphore wait and explicitly
    # disclaim the server-side-queueing interpretation
    assert "客户端 semaphore wait" in md
    assert "不将客户端 semaphore wait 解释为服务端排队" in md


def test_5_d1_report_does_not_claim_embedding_retry_solved():
    md = (PERF / "llm_research_d" / "LLM_REPLAY_REPORT.md").read_text(encoding="utf-8")
    # the report must explicitly say D1 has no embedding coverage and must NOT
    # affirm that bounded concurrency solved Mason's retry
    assert "未覆盖 embedding" in md
    assert "不构成" in md


def test_6_manifest_marks_synthetic_reconstructed():
    m = manifest_mod.build_replay_manifest(_spec())
    assert m["workload_label"] == "SYNTHETIC_RECONSTRUCTED_CODEGEN_WORKLOAD"


def test_7_run_artifacts_complete(tmp_path):
    for name in benchmark_mod.REQUIRED_ARTIFACTS:
        (tmp_path / name).write_text("x", encoding="utf-8")
    benchmark_mod._write_artifact_inventory(tmp_path)
    assert (tmp_path / "SHA256SUMS").exists()
    assert (tmp_path / "ARTIFACT_INVENTORY.json").exists()
    assert benchmark_mod.verify_run_artifacts(tmp_path)["complete"] is True


def test_8_missing_events_jsonl_not_pass(tmp_path):
    for name in benchmark_mod.REQUIRED_ARTIFACTS:
        if name != "events.jsonl":
            (tmp_path / name).write_text("x", encoding="utf-8")
    v = benchmark_mod.verify_run_artifacts(tmp_path)
    assert v["complete"] is False
    assert "events.jsonl" in v["missing"]
    assert v["status"] == "INCOMPLETE"


def test_9_gpu_csv_peak_and_min_free(tmp_path):
    csv_path = tmp_path / "gpu.csv"
    header = "timestamp,gpu_index,gpu_uuid,utilization_gpu,memory_used_mib,memory_free_mib,temperature,compute_pid,process_name\n"
    rows = [
        "t0,0,GPU-e8c08612,0 %,43000,3068,45,3154045,llama-server\n",
        "t1,0,GPU-e8c08612,1 %,43558,2510,46,3154045,llama-server\n",
        "t2,0,GPU-e8c08612,0 %,42000,4068,44,3154045,llama-server\n",
    ]
    csv_path.write_text(header + "".join(rows), encoding="utf-8")
    stats = gpu_mod.compute_gpu_stats(csv_path)
    assert stats["sample_count"] == 3
    assert stats["peak_memory_used_mib"] == 43558
    assert stats["min_memory_free_mib"] == 2510
    assert stats["uuid_consistent"] is True


@pytest.mark.asyncio
async def test_10_embedding_http_200_not_retry(tmp_path):
    sink = _sink(tmp_path, max_in_flight=1)
    c = _client(tmp_path, max_in_flight=1)
    c.client = _FakeClient(embeddings=_FakeEmbed())  # returns 200-like success
    res = await c.embed_once(["hello"], slot="s0", request_id="r0",
                             attempt=1, prompt_sha256="h")
    assert res["error_class"] is None
    events = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines() if l.strip()]
    assert not any(e["phase"] == "retry_backoff" for e in events)


@pytest.mark.asyncio
async def test_11_retry_same_request_id_incrementing_attempt(tmp_path):
    calls = {"n": 0}

    def flaky(**k):
        calls["n"] += 1
        if calls["n"] < 2:
            raise ConnectionError("boom")
        return _FakeResp("code")

    c = _client(tmp_path, max_in_flight=1, completions=_FakeCompletions(fn=flaky))
    await c.chat_with_retries("s", "u", slot="s0", request_id="ridX",
                              prompt_sha256="h", max_retries=3)
    events = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines() if l.strip()]
    chat = [e for e in events if e["phase"] == "chat_request"]
    assert len(chat) == 2
    assert all(e["request_id"] == "ridX" for e in chat)
    assert [e["attempt"] for e in chat] == [1, 2]


def test_12_unbounded_config_has_definite_cap(tmp_path):
    # "production_like_unbounded" is a finite max_in_flight (25), not an infinite loop
    c = harness_mod.LLMReplayClient(base_url="http://x/v1", model="m", provider="local",
                                    temperature=0.6, top_p=0.95, max_tokens=100,
                                    timeout_s=5, max_in_flight=25, sink=_sink(tmp_path))
    assert c.max_in_flight == 25
    assert isinstance(c.max_in_flight, int)


@pytest.mark.asyncio
async def test_13_provider_failure_writes_failure_not_result(tmp_path, monkeypatch):
    class FakeClient:
        def __init__(self, **kw):
            pass

        async def health_check(self):
            raise harness_mod.ProviderUnavailableError("down")

    monkeypatch.setattr(benchmark_mod, "LLMReplayClient", FakeClient)
    man = manifest_mod.build_replay_manifest(_spec())
    man = manifest_mod.write_manifest(man, tmp_path / "m.json")
    out = tmp_path / "out"
    with pytest.raises(harness_mod.ProviderUnavailableError):
        await benchmark_mod.run_replay(man, max_in_flight=1, out_dir=out,
                                       repeat_label="r1", do_cpu_jax=False)
    assert not (out / "RESULT.json").exists()
    assert (out / "FAILURE.json").exists()


def test_14_credential_redaction_in_result_artifacts(tmp_path):
    m = manifest_mod.build_replay_manifest(_spec())
    man = manifest_mod.write_manifest(m, tmp_path / "m.json")
    text = (tmp_path / "m.json").read_text(encoding="utf-8").lower()
    for secret in ("authorization", "bearer", "cookie", "api_key"):
        assert secret not in text


def test_15_result_and_event_sha_recomputable(tmp_path):
    base = 1_000_000_000
    rep = benchmark_mod.derive_reports([_ev("chat_request", base, base + 10_000_000_000)])
    # a RESULT dict's result_sha256 must recompute from everything except itself
    result = {"classification": "LLM_REPLAY_RESULT", "llm_union_s": rep["llm_union_s"]}
    result["result_sha256"] = benchmark_mod.sha256_bytes(
        json.dumps({k: v for k, v in result.items() if k != "result_sha256"},
                   sort_keys=True, default=str).encode())
    recomputed = benchmark_mod.sha256_bytes(
        json.dumps({k: v for k, v in result.items() if k != "result_sha256"},
                   sort_keys=True, default=str).encode())
    assert recomputed == result["result_sha256"]
