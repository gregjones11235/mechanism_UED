"""Offline tests for the audited phase-D final disposition."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest


PERF = Path(__file__).parents[4] / "experiments" / "performance"
LLM_D = PERF / "llm_research_d"
SCRIPT = LLM_D / "d_phase_finalize.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("d_phase_finalize", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


finalize = _load_module()


def _copy_inputs(target: Path) -> dict[str, str]:
    target.mkdir()
    before = {}
    for name in finalize.INPUT_RAW_SHA256:
        raw = (LLM_D / name).read_bytes()
        (target / name).write_bytes(raw)
        before[name] = hashlib.sha256(raw).hexdigest()
    return before


def _generate(tmp_path: Path):
    evidence = tmp_path / "evidence"
    before = _copy_inputs(evidence)
    result_path = tmp_path / "out" / "D_PHASE_FINAL_RESULT.json"
    report_path = tmp_path / "out" / "D_PHASE_FINAL_REPORT.md"
    result = finalize.finalize(evidence, result_path, report_path)
    return evidence, before, result_path, report_path, result


def test_generate_and_load_final_result(tmp_path):
    _, _, result_path, report_path, generated = _generate(tmp_path)
    loaded = finalize.load_result(result_path)
    assert loaded == generated
    assert loaded["phase"] == "D"
    assert loaded["phase_execution_status"] == "COMPLETE"
    assert loaded["review_status"] == "PASS_WITH_CONCERNS"
    assert loaded["conclusion"] == "D_PHASE_CLOSED_NO_PRODUCTION_OPTIMIZATION"
    assert report_path.is_file()
    assert report_path.read_text(encoding="utf-8").startswith(
        "# D 阶段最终收口报告\n\n## 结论"
    )


def test_input_raw_hashes_are_bound_and_internal_hashes_verified(tmp_path):
    _, before, result_path, _, _ = _generate(tmp_path)
    result = finalize.load_result(result_path)
    assert {name: item["raw_file_sha256"]
            for name, item in result["input_artifacts"].items()} == before
    for name in ("D2_RESULT.json", "D2_EVIDENCE_FINAL.json"):
        binding = result["input_artifacts"][name]
        assert binding["internal_hash_algorithm"] == "canonical_json_sha256"
        assert len(binding["internal_artifact_sha256"]) == 64
        assert "EXCLUDING_ARTIFACT_SHA256" in binding["internal_hash_scope"]


def test_final_internal_and_raw_hashes_recompute(tmp_path):
    _, _, result_path, report_path, _ = _generate(tmp_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in result.items()
               if key != "artifact_sha256"}
    assert finalize.canonical_json_sha256(payload) == result["artifact_sha256"]
    raw = hashlib.sha256(result_path.read_bytes()).hexdigest()
    assert raw in report_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("field,value", [
    ("conclusion", "D_PHASE_PASS"),
    ("review_status", "PASS"),
    ("eligible_for_mainline_combination", ["chat_concurrency"]),
])
def test_load_rejects_top_level_tamper(tmp_path, field, value):
    _, _, result_path, _, _ = _generate(tmp_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result[field] = value
    tampered = tmp_path / f"tampered-{field}.json"
    tampered.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(finalize.EvidenceError):
        finalize.load_result(tampered)


def test_load_rejects_nested_tamper_even_if_semantically_plausible(tmp_path):
    _, _, result_path, _, _ = _generate(tmp_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["findings"][0]["reason"] = "three repeats"
    tampered = tmp_path / "nested.json"
    tampered.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(finalize.EvidenceError, match="tampered"):
        finalize.load_result(tampered)


def test_input_tamper_fails_closed(tmp_path):
    evidence = tmp_path / "evidence"
    _copy_inputs(evidence)
    path = evidence / "D1B_ALL_RESULTS.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["results"][0]["wall_clock_s"] += 1
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(finalize.EvidenceError, match="raw SHA256 mismatch"):
        finalize.build_result(evidence)


def test_d2_blocked_is_not_promoted_to_benchmark(tmp_path):
    _, _, result_path, report_path, _ = _generate(tmp_path)
    result = finalize.load_result(result_path)
    d2 = next(item for item in result["findings"] if item["stage"] == "D2")
    assert d2["conclusion"] == "D2_BLOCKED_EXTERNAL_PROVIDER_EVIDENCE_COMPLETE"
    assert d2["disposition"] == "NO_BENCHMARK_CONCLUSION"
    assert result["speed_or_quality_conclusion_for_235b_vs_14b"] is False
    report = report_path.read_text(encoding="utf-8")
    assert "benchmark 执行臂为 0" in report
    assert "不存在 235B 与 14B 的速度或质量结论" in report


def test_no_mainline_combination_candidate(tmp_path):
    _, _, result_path, _, _ = _generate(tmp_path)
    result = finalize.load_result(result_path)
    assert result["eligible_for_mainline_combination"] == []
    assert all(item["disposition"] != "ELIGIBLE_FOR_MAINLINE_COMBINATION"
               for item in result["findings"])


def test_script_has_no_network_api_gpu_process_dependency():
    source = SCRIPT.read_text(encoding="utf-8").lower()
    banned = (
        "requests", "httpx", "openai", "socket", "subprocess", "jax",
        "nvidia-smi", "urllib", "urlopen", "cuda", "ollama",
    )
    for token in banned:
        assert not re.search(rf"\b{re.escape(token)}\b", source), token


def test_atomic_refuses_overwrite_without_changing_files(tmp_path):
    evidence, _, result_path, report_path, _ = _generate(tmp_path)
    before_result = result_path.read_bytes()
    before_report = report_path.read_bytes()
    with pytest.raises(FileExistsError):
        finalize.finalize(evidence, result_path, report_path)
    assert result_path.read_bytes() == before_result
    assert report_path.read_bytes() == before_report


def test_historical_artifacts_remain_byte_identical(tmp_path):
    evidence, before, _, _, _ = _generate(tmp_path)
    after = {name: hashlib.sha256((evidence / name).read_bytes()).hexdigest()
             for name in before}
    assert after == before == finalize.INPUT_RAW_SHA256


def test_real_historical_artifacts_unchanged_by_offline_validation():
    before = {name: hashlib.sha256((LLM_D / name).read_bytes()).hexdigest()
              for name in finalize.INPUT_RAW_SHA256}
    finalize.build_result(LLM_D)
    after = {name: hashlib.sha256((LLM_D / name).read_bytes()).hexdigest()
             for name in finalize.INPUT_RAW_SHA256}
    assert before == after == finalize.INPUT_RAW_SHA256


def test_committed_result_loads_and_is_bound_to_committed_inputs():
    result = finalize.load_result(LLM_D / "D_PHASE_FINAL_RESULT.json")
    assert result["branch"] == "perf/llm-production-shape-d1c"
    assert result["base_commit"] == finalize.BASE_COMMIT
    assert result["d2_head"] == finalize.D2_HEAD
    assert {name: item["raw_file_sha256"]
            for name, item in result["input_artifacts"].items()} == \
        finalize.INPUT_RAW_SHA256
