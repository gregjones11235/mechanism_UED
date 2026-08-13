"""CPU-only and metadata-only tests for the D3 contract."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import d3_metadata_gate as gate  # noqa: E402
import d3_runner as runner  # noqa: E402


def test_manifest_and_contract_are_frozen():
    manifest = runner.load_frozen_manifest(HERE / "FROZEN_MANIFEST.json")
    contract = runner.d3_contract(manifest)
    assert contract["prompt_count"] == 12
    assert contract["prompt_stages"] == {"early": 4, "mid": 4, "late": 4}
    assert contract["arm_order"] == [list(x) for x in runner.ARM_ORDER]
    assert contract["post_limit_per_provider"] == 108
    assert contract["large_model_size"] == "UNKNOWN"


def test_shared_post_budget_includes_retry_and_repair():
    budget = runner.ProviderPostBudget(3)
    assert [budget.reserve("ollama", kind=k) for k in ("generation", "transport_retry", "repair")] == [1, 2, 3]
    with pytest.raises(RuntimeError):
        budget.reserve("ollama", kind="repair")


def test_metadata_gate_uses_only_metadata_endpoints(monkeypatch):
    calls = []

    class Response:
        status = 200

        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(self.payload).encode()

    def fake_urlopen(request, timeout=0):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        calls.append(url)
        if url.endswith("/api/tags"):
            return Response({"models": [{"name": gate.OLLAMA_MODEL}]})
        if url.endswith("/models"):
            return Response({"data": [{"id": gate.DEEPSEEK_MODEL}]})
        raise AssertionError(url)

    monkeypatch.setattr(gate.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret")
    ollama = gate.check_ollama()
    deepseek = gate.check_deepseek()
    assert ollama["exact_model_available"] is True
    assert deepseek["exact_model_available"] is True
    assert all(url.endswith("/api/tags") or url.endswith("/models") for url in calls)
    assert not any("completions" in url or "embeddings" in url for url in calls)
    assert ollama["completion_requests"] == 0
    assert deepseek["embedding_requests"] == 0


def test_deepseek_credential_is_redacted(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unique-secret-value")

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"data": []}'

    monkeypatch.setattr(gate.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    result = gate.check_deepseek()
    assert result["credential_present"] is True
    assert result["credential_value_serialized"] is False
    assert "unique-secret-value" not in json.dumps(result)


def test_gate_artifacts_are_no_clobber(tmp_path):
    result = {
        "gate_status": "BLOCKED_METADATA_GATE",
        "metadata_requests_total": 2,
        "completion_requests_total": 0,
        "embedding_requests_total": 0,
        "gpu2_smoke_allowed": False,
        "blocked_reasons": ["test"],
    }
    paths = gate.write_gate_artifacts(result, tmp_path / "out")
    assert all(Path(p).exists() for p in paths.values())
    with pytest.raises(FileExistsError):
        gate.write_gate_artifacts(result, tmp_path / "out")
