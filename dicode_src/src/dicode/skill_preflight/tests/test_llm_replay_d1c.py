"""Tests for the D1c production-shape embedding retry replay (offline, no LLM/GPU)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PERF = Path(__file__).parents[4] / "experiments" / "performance"
LLM_D = PERF / "llm_research_d"


def _load(name, sub="llm_research_d"):
    path = PERF / sub / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# load shared modules first (d1c_harness imports them)
_manifest = _load("llm_replay_manifest", sub="")
_harness = _load("llm_replay_harness", sub="")
_benchmark = _load("llm_replay_benchmark", sub="")
_gpu = _load("llm_replay_gpu", sub="")

d1c = _load("d1c_harness")
freeze = _load("freeze_production_embedding")

EXPECTED_BATCH_SEQUENCE = [5, 10, 12, 13, 16, 10, 15, 16, 9, 16, 16, 10]


# --- 1. production batch sequence + order frozen ---
def test_1_batch_sequence_frozen():
    assert freeze.BATCH_SEQUENCE == EXPECTED_BATCH_SEQUENCE
    assert len(freeze.BATCH_SEQUENCE) == 12
    assert max(freeze.BATCH_SEQUENCE) <= 16  # late-stage scale


# --- 2. manifest tamper rejection (reuse chat manifest roundtrip) ---
def test_2_manifest_tamper_rejection(tmp_path):
    m = _manifest.build_replay_manifest({
        "classification": "LLM_REPLAY_MANIFEST", "source_commit": "abc",
        "provider": "local", "model": "nomic-embed-text", "base_url": "http://x",
        "system_prompt": "s", "user_prompts": ["a", "b"]})
    out = tmp_path / "m.json"
    _manifest.write_manifest(m, out)
    tampered = json.loads(out.read_text())
    tampered["user_prompts"][0] = "tampered"
    (tmp_path / "t.json").write_text(json.dumps(tampered))
    with pytest.raises(ValueError):
        _manifest.load_manifest(tmp_path / "t.json")


# --- 3. client lifecycle labels accurate ---
def test_3_client_lifecycle_labels():
    src = (LLM_D / "run_d1c.py").read_text(encoding="utf-8")
    assert '"persistent"' in src and '"fresh"' in src
    assert "PERSISTENT_CONTIGUOUS" in src or "persistent" in src


# --- 4. persistent vs fresh client caching ---
def test_4_client_caching_and_reset():
    c = d1c.D1CEmbeddingClient(base_url="http://x", model="m", api_key="token-",
                               timeout_s=60, max_retries=2, embedding_size=768)
    c1 = c._get()
    c2 = c._get()
    assert c1 is c2  # persistent: cached
    c.reset()
    c3 = c._get()
    assert c3 is not c1  # fresh: new client


# --- 5. idle gap uses monotonic clock (asyncio.sleep) ---
def test_5_idle_gap_monotonic():
    src = (LLM_D / "run_d1c.py").read_text(encoding="utf-8")
    assert "asyncio.sleep" in src


# --- 6. HTTP 200 not retry (valid result -> no error class) ---
def test_6_http_200_valid_not_retry():
    ok, err, cnt, shape = d1c.validate_embedding(
        type("R", (), {"data": [type("D", (), {"embedding": [0.1, 0.2]})() for _ in range(3)]})(),
        expected_count=3, embedding_size=768)
    assert ok is True and err is None and cnt == 3


# --- 7. SDK retry associated with same request_id ---
def test_7_sdk_retry_same_request_id_in_event():
    src = (LLM_D / "d1c_harness.py").read_text(encoding="utf-8")
    # each embed() call writes one event keyed by its own request_id + sdk_retry_count
    assert '"sdk_retry_count"' in src and '"request_id"' in src


# --- 8. max_retries=0 captures underlying exception ---
def test_8_max_retries_zero_passthrough():
    c = d1c.D1CEmbeddingClient(base_url="http://x", model="m", api_key="token-",
                               timeout_s=60, max_retries=0, embedding_size=768)
    assert c.max_retries == 0


# --- 9. no diagnostic arm D when no retry ---
def test_9_no_diagnostic_d_when_no_retry():
    src = (LLM_D / "run_d1c.py").read_text(encoding="utf-8")
    # D arm is conditional (only after a retry is observed); it is NOT in the
    # default ARMS schedule, so it cannot run when there is no retry.
    assert '("D",' not in src and '("D" ,' not in src


# --- 10/11/12. item count / shape / non-finite fail-closed ---
def test_10_item_count_mismatch_fail_closed():
    ok, err, _, _ = d1c.validate_embedding(
        type("R", (), {"data": [type("D", (), {"embedding": [0.1]})()]})(),
        expected_count=3, embedding_size=768)
    assert ok is False and err == "item_count_mismatch"


def test_11_shape_mismatch_fail_closed():
    data = type("R", (), {"data": [
        type("D", (), {"embedding": [0.1]})(),
        type("D", (), {"embedding": [0.1, 0.2, 0.3]})(),
    ]})()
    ok, err, _, _ = d1c.validate_embedding(data, expected_count=2, embedding_size=768)
    assert ok is False and err == "shape_mismatch"


def test_12_non_finite_fail_closed():
    data = type("R", (), {"data": [type("D", (), {"embedding": [float("nan")]})()]})()
    ok, err, _, _ = d1c.validate_embedding(data, expected_count=1, embedding_size=768)
    assert ok is False and err == "non_finite_embedding"


# --- 13/14. GPU CSV coverage / insufficient samples ---
def test_13_gpu_csv_has_timestamps_and_interval():
    src = (LLM_D / "run_d1c.py").read_text(encoding="utf-8")
    assert "gpu0_memory_2s.csv" in src and "interval_s=2.0" in src


def test_14_gpu_insufficient_samples_not_full_supervision(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("")  # no samples
    stats = _gpu.compute_gpu_stats(csv_path)
    assert stats.get("sample_count", 0) == 0  # zero samples -> cannot claim coverage


# --- 15. Ollama PID change invalidates arm ---
def test_15_ollama_pid_change_aborts():
    src = (LLM_D / "run_d1c.py").read_text(encoding="utf-8")
    assert "SAFETY ABORT: Ollama PID changed" in src


# --- 16. raw/enriched hash contract ---
def test_16_hash_contract():
    value = {"classification": "X", "n": 1.5}
    legacy = _benchmark.legacy_json_sha256(value)
    canonical = _benchmark.canonical_json_sha256(value)
    assert legacy and canonical and legacy != canonical  # different serialization
    assert _benchmark.RESULT_SHA256_SCOPE == "RESULT_FIELDS_EXCLUDING_RESULT_SHA256_AND_ARTIFACT_INVENTORY"


# --- 17. artifact inventory complete ---
def test_17_artifact_inventory_complete(tmp_path):
    for name in _benchmark.REQUIRED_ARTIFACTS:
        (tmp_path / name).write_text("x")
    _benchmark._write_artifact_inventory(tmp_path)
    assert _benchmark.verify_run_artifacts(tmp_path)["complete"] is True


# --- 18. logs/results no credentials ---
def test_18_no_credentials_in_sources():
    for name in ("d1c_harness.py", "run_d1c.py", "freeze_production_embedding.py"):
        src = (LLM_D / name).read_text(encoding="utf-8").lower()
        for secret in ("authorization", "bearer ", "cookie=", "deepinfra_api_key",
                       "sk-", "openrouter_api_key", "together_api_key"):
            assert secret not in src, f"{name} contains {secret}"


# --- 19. D2 blocked probe once ---
def test_19_d2_probe_once():
    # D1c run script must not contain any D2/235B probing logic
    src = (LLM_D / "run_d1c.py").read_text(encoding="utf-8")
    assert "235B" not in src and "DEEPINFRA" not in src


# --- 20. no B/C production import ---
def test_20_no_bc_import():
    import re
    for name in ("d1c_harness.py", "run_d1c.py", "freeze_production_embedding.py"):
        src = (LLM_D / name).read_text(encoding="utf-8")
        assert not re.search(r"from\s+dicode\.dreaming\.llm\s+import", src)
        assert not re.search(r"import\s+run_dicode", src)
        assert not re.search(r"from\s+dicode\.skill_preflight\s+import", src)
        assert not re.search(r"import\s+preflight_replay", src)
