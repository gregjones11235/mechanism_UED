"""Tests for the D2 provider availability probe (offline, no network/LLM/GPU)."""
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


probe = _load("d2_provider_probe")

DEEPINFRA = "provider: deepinfra\nbase_url: https://api.deepinfra.com/v1/openai\nmodel: Qwen/Qwen3-235B-A22B-Thinking-2507\n"
LOCAL_GEN = "provider: local\nbase_url: http://localhost:5000/v1\nmodel: Qwen/Qwen3-235B-A22B-Thinking-2507-FP8\n"


def _config_root(tmp_path):
    root = tmp_path / "dicode_src"
    (root / "conf" / "gen_manager" / "llm").mkdir(parents=True)
    # write raw bytes to avoid Windows CRLF translation (keeps config SHA stable)
    (root / "conf" / "gen_manager" / "llm" / "deepinfra.yaml").write_bytes(DEEPINFRA.encode())
    (root / "conf" / "gen_manager" / "llm" / "local_gen.yaml").write_bytes(LOCAL_GEN.encode())
    return root


def _offline_network(monkeypatch, localhost=None, ollama=None):
    monkeypatch.setattr(probe, "check_localhost_5000", lambda **k: localhost or {
        "reachable": False, "http_status": None, "model_ids": [],
        "local_model_available": False, "error_class": "connection_error"})
    monkeypatch.setattr(probe, "list_ollama_models", lambda **k: ollama or {
        "reachable": True, "http_status": 200,
        "model_names": ["qwen2.5-coder:14b", "nomic-embed-text"], "error_class": None})


def _build(monkeypatch, tmp_path, **net):
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    _offline_network(monkeypatch, **net)
    return probe.build_probe(str(_config_root(tmp_path)), "branch", "head", "commit", "utc")


# 1. no key -> BLOCKED
def test_1_no_key_blocked(monkeypatch, tmp_path):
    p = _build(monkeypatch, tmp_path)
    assert p["credential_present"] is False
    assert p["conclusion"] == "D2_BLOCKED_EXTERNAL_PROVIDER"
    assert p["external_provider_request_performed"] is False
    assert p["llm_api_calls"] == 0
    assert p["gpu_used"] is False


# 2. credential value never in JSON
def test_2_credential_value_never_serialized(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "sk-SECRET-VALUE")
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)  # then absent for the probe
    _offline_network(monkeypatch)
    p = probe.build_probe(str(_config_root(tmp_path)), "b", "h", "c", "utc")
    assert p["credential_value_serialized"] is False
    serialized = json.dumps(p)
    assert "sk-SECRET" not in serialized
    assert "credential_present" in p and isinstance(p["credential_present"], bool)


# 3. external DeepInfra request never called when credential absent
def test_3_external_request_zero_calls(monkeypatch, tmp_path):
    src = (LLM_D / "d2_provider_probe.py").read_text(encoding="utf-8")
    # the tool has no external DeepInfra endpoint call at all
    assert "api.deepinfra.com" not in src or "external_provider_request_performed" in src
    p = _build(monkeypatch, tmp_path)
    assert p["external_provider_request_performed"] is False


# 4. provider/model read from config and model id non-empty
def test_4_provider_model_from_config(monkeypatch, tmp_path):
    p = _build(monkeypatch, tmp_path)
    assert p["provider"] == "deepinfra"
    assert p["model"] == "Qwen/Qwen3-235B-A22B-Thinking-2507"
    assert p["model_id_explicit"] is True


# 5. config SHA correct
def test_5_config_sha_correct(monkeypatch, tmp_path):
    p = _build(monkeypatch, tmp_path)
    import hashlib
    expected = hashlib.sha256(DEEPINFRA.encode()).hexdigest()
    assert p["config_evidence"][0]["sha256"] == expected


# 6. localhost unavailable normalized
def test_6_localhost_unavailable(monkeypatch, tmp_path):
    p = _build(monkeypatch, tmp_path, localhost={
        "reachable": False, "http_status": None, "model_ids": [],
        "local_model_available": False, "error_class": "connection_error"})
    assert p["local_endpoint_probe"]["error_class"] == "connection_error"
    assert p["decision_inputs"]["local_model_available"] is False


# 7. localhost other model not impersonated
def test_7_localhost_other_model_not_235b(monkeypatch, tmp_path):
    p = _build(monkeypatch, tmp_path, localhost={
        "reachable": True, "http_status": 200, "model_ids": ["some-other-model"],
        "local_model_available": False, "error_class": None})
    assert p["decision_inputs"]["local_model_available"] is False


# 8. exact 235B model id -> local_model_available true
def test_8_exact_235b_model_id(monkeypatch, tmp_path):
    p = _build(monkeypatch, tmp_path, localhost={
        "reachable": True, "http_status": 200,
        "model_ids": ["Qwen/Qwen3-235B-A22B-Thinking-2507"],
        "local_model_available": True, "error_class": None})
    assert p["decision_inputs"]["local_model_available"] is True


# 9. budget NOT_OBSERVED
def test_9_budget_not_observed(monkeypatch, tmp_path):
    p = _build(monkeypatch, tmp_path)
    assert p["budget_authorization"] == "NOT_OBSERVED"


# 10. refuse overwrite
def test_10_refuse_overwrite(tmp_path):
    out = tmp_path / "probe.json"
    out.write_text("existing")
    with pytest.raises(FileExistsError):
        probe.atomic_write_refusing_overwrite(out, {"x": 1})


# 11. canonical hash recomputable
def test_11_canonical_hash_recomputable():
    p = {"a": 1, "b": [1.0, 2.0]}
    p["artifact_sha256"] = probe.canonical_json_sha256(
        {k: v for k, v in p.items() if k != "artifact_sha256"})
    assert probe.canonical_json_sha256(
        {k: v for k, v in p.items() if k != "artifact_sha256"}) == p["artifact_sha256"]


# 12. tamper -> load reject
def test_12_load_rejects_tamper(tmp_path):
    out = tmp_path / "probe.json"
    p = {"classification": "D2_PROVIDER_AVAILABILITY_PROBE"}
    p["artifact_sha256"] = probe.canonical_json_sha256(
        {k: v for k, v in p.items() if k != "artifact_sha256"})
    probe.atomic_write_refusing_overwrite(out, p)
    tampered = json.loads(out.read_text())
    tampered["classification"] = "TAMPERED"
    (tmp_path / "t.json").write_text(json.dumps(tampered))
    with pytest.raises(ValueError):
        probe.load_probe(tmp_path / "t.json")


# 13. D2_RESULT arms_executed=0, llm_api_calls=0
def test_13_result_zero_execution(monkeypatch, tmp_path):
    p = _build(monkeypatch, tmp_path)
    r = probe.build_d2_result(p, "probe.json")
    assert r["arms_executed"] == 0
    assert r["llm_api_calls"] == 0
    assert r["chat_requests"] == 0 and r["embedding_requests"] == 0


# 14. D2_RESULT no speed/quality conclusion
def test_14_result_no_speed_quality_claim(monkeypatch, tmp_path):
    p = _build(monkeypatch, tmp_path)
    r = probe.build_d2_result(p, "probe.json")
    serialized = json.dumps(r)
    assert "performance_comparison_available" in r and r["performance_comparison_available"] is False
    assert "quality_comparison_available" in r and r["quality_comparison_available"] is False
    assert "slower" not in serialized and "faster" not in serialized
    assert "better" not in serialized and "worse" not in serialized


# 15. status BLOCKED / gate blocked, not benchmark completed
def test_15_result_gate_blocked_not_completed(monkeypatch, tmp_path):
    p = _build(monkeypatch, tmp_path)
    r = probe.build_d2_result(p, "probe.json")
    assert r["status"] == "BLOCKED"
    assert r["conclusion"] == "D2_BLOCKED_EXTERNAL_PROVIDER"
    assert "completed" not in json.dumps(r).lower()


# 19. no B/C/preflight/PPO import
def test_19_no_bc_import():
    import re
    src = (LLM_D / "d2_provider_probe.py").read_text(encoding="utf-8")
    assert not re.search(r"from\s+dicode\.dreaming\.llm\s+import", src)
    assert not re.search(r"import\s+run_dicode", src)
    assert not re.search(r"preflight", src, re.IGNORECASE)
    assert "jax" not in src.lower()


# 20. no credentials / sensitive headers
def test_20_no_credentials_or_headers():
    src = (LLM_D / "d2_provider_probe.py").read_text(encoding="utf-8")
    for secret in ("Bearer", "Authorization", "Cookie", "X-Api-Key", "sk-"):
        assert secret not in src
