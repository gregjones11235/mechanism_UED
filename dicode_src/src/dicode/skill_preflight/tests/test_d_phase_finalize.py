"""Offline tests for the audited, fail-closed phase-D disposition."""
from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import re
import sys
import threading
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


def _copy_inputs(target: Path, *, newline: str | None = None) -> dict[str, bytes]:
    target.mkdir()
    before = {}
    for name in finalize.INPUT_NORMALIZED_TEXT_SHA256:
        raw = (LLM_D / name).read_bytes()
        if newline is not None:
            text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
            raw = text.replace("\n", newline).encode("utf-8")
        (target / name).write_bytes(raw)
        before[name] = raw
    return before


def _generate(tmp_path: Path):
    evidence = tmp_path / "evidence"
    before = _copy_inputs(evidence)
    result_path = tmp_path / "out" / "D_PHASE_FINAL_RESULT.json"
    report_path = tmp_path / "out" / "D_PHASE_FINAL_REPORT.md"
    result = finalize.finalize(evidence, result_path, report_path)
    return evidence, before, result_path, report_path, result


def _rehash(result: dict) -> None:
    result["artifact_sha256"] = finalize.canonical_json_sha256(
        {key: value for key, value in result.items() if key != "artifact_sha256"})


def _write_rehashed_tamper(tmp_path: Path, result: dict, name: str) -> Path:
    _rehash(result)
    path = tmp_path / name
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


def _rewrite_canonical_artifact(path: Path, mutate) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    data["artifact_sha256"] = finalize.canonical_json_sha256(
        {key: value for key, value in data.items() if key != "artifact_sha256"})
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_generate_and_load_final_result(tmp_path):
    evidence, _, result_path, report_path, generated = _generate(tmp_path)
    loaded = finalize.load_result(result_path, evidence)
    assert loaded == generated
    assert loaded["phase_execution_status"] == "COMPLETE"
    assert loaded["review_status"] == "PASS_WITH_CONCERNS"
    assert loaded["conclusion"] == "D_PHASE_CLOSED_NO_PRODUCTION_OPTIMIZATION"
    assert report_path.read_text(encoding="utf-8").startswith(
        "# D 阶段最终收口报告\n\n## 结论")


def test_cross_platform_lf_crlf_gate_hashes_are_identical(tmp_path):
    lf = tmp_path / "lf"
    crlf = tmp_path / "crlf"
    _copy_inputs(lf, newline="\n")
    _copy_inputs(crlf, newline="\r\n")
    for name, expected in finalize.INPUT_NORMALIZED_TEXT_SHA256.items():
        assert finalize.normalized_utf8_text_sha256(lf / name) == expected
        assert finalize.normalized_utf8_text_sha256(crlf / name) == expected
    assert finalize.build_result(lf) == finalize.build_result(crlf)


def test_content_change_still_fails_cross_platform_gate(tmp_path):
    evidence = tmp_path / "evidence"
    _copy_inputs(evidence, newline="\r\n")
    path = evidence / "D1B_ALL_RESULTS.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["results"][0]["wall_clock_s"] += 1
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(finalize.EvidenceError, match="normalized UTF-8 LF"):
        finalize.build_result(evidence)


def test_bindings_have_explicit_cross_platform_contract(tmp_path):
    evidence, _, result_path, _, _ = _generate(tmp_path)
    result = finalize.load_result(result_path, evidence)
    for name, binding in result["input_artifacts"].items():
        assert binding["content_gate_sha256"] == \
            finalize.INPUT_NORMALIZED_TEXT_SHA256[name]
        assert binding["content_gate_algorithm"] == "sha256_utf8_text_newlines_lf"
        assert binding["content_gate_scope"] == \
            "FULL_UTF8_TEXT_AFTER_CRLF_AND_CR_TO_LF"
    assert result["input_artifacts"]["D2_RESULT.json"][
        "internal_artifact_sha256"] == \
        "54b1e01d6afa01a98f8fa0396ad8e9ccfec6ca79d87f8e78c55e1ecf557acaa6"
    assert result["input_artifacts"]["D2_EVIDENCE_FINAL.json"][
        "internal_artifact_sha256"] == \
        "234b766a8c494d6fa0f3afd875270ae8486a96c4a72b747f81deb92f99c5e037"


def test_final_internal_and_raw_hashes_recompute(tmp_path):
    _, _, result_path, report_path, _ = _generate(tmp_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    saved = result.pop("artifact_sha256")
    assert finalize.canonical_json_sha256(result) == saved
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() in \
        report_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("kind", ["nested_conclusion", "reason", "metric", "binding"])
def test_rehashed_semantic_tamper_is_rejected_by_reconstruction(tmp_path, kind):
    evidence, _, result_path, _, _ = _generate(tmp_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if kind == "nested_conclusion":
        result["findings"][0]["conclusion"] = "D1_CHAT_PASS"
    elif kind == "reason":
        result["findings"][0]["reason"] = "three repeats"
    elif kind == "metric":
        result["derived_metrics"]["chat_wall_gain_pct"] = 99
    else:
        result["input_artifacts"]["D1_ALL_RESULTS.json"]["content_gate_sha256"] = "0" * 64
    tampered = _write_rehashed_tamper(tmp_path, result, f"{kind}.json")
    with pytest.raises(finalize.EvidenceError, match="reconstruction"):
        finalize.load_result(tampered, evidence)


def test_missing_any_input_fails_closed(tmp_path):
    for missing in finalize.INPUT_NORMALIZED_TEXT_SHA256:
        evidence = tmp_path / missing.replace(".", "_")
        _copy_inputs(evidence)
        (evidence / missing).unlink()
        with pytest.raises(finalize.EvidenceError, match="missing input"):
            finalize.build_result(evidence)


@pytest.mark.parametrize("name", ["D2_RESULT.json", "D2_EVIDENCE_FINAL.json"])
def test_d2_internal_hash_mismatch_fails_closed(tmp_path, name):
    evidence = tmp_path / "evidence"
    _copy_inputs(evidence)
    data = json.loads((evidence / name).read_text(encoding="utf-8"))
    data["artifact_sha256"] = "0" * 64
    # Keep the outer content gate current so the internal verifier is exercised.
    path = evidence / name
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    original = finalize.INPUT_NORMALIZED_TEXT_SHA256[name]
    finalize.INPUT_NORMALIZED_TEXT_SHA256[name] = finalize.normalized_utf8_text_sha256(path)
    try:
        with pytest.raises(finalize.EvidenceError, match="artifact_sha256 mismatch"):
            finalize.build_result(evidence)
    finally:
        finalize.INPUT_NORMALIZED_TEXT_SHA256[name] = original


@pytest.mark.parametrize("mutation", ["status", "arms", "conclusion"])
def test_d2_blocked_semantics_change_fails_closed(tmp_path, mutation):
    evidence = tmp_path / "evidence"
    _copy_inputs(evidence)
    path = evidence / "D2_RESULT.json"
    def change(data):
        if mutation == "status": data["status"] = "COMPLETE"
        if mutation == "arms": data["arms_executed"] = 2
        if mutation == "conclusion": data["conclusion"] = "D2_PASS"
    _rewrite_canonical_artifact(path, change)
    original = finalize.INPUT_NORMALIZED_TEXT_SHA256[path.name]
    finalize.INPUT_NORMALIZED_TEXT_SHA256[path.name] = finalize.normalized_utf8_text_sha256(path)
    try:
        with pytest.raises(finalize.EvidenceError, match="improperly represents"):
            finalize.build_result(evidence)
    finally:
        finalize.INPUT_NORMALIZED_TEXT_SHA256[path.name] = original


@pytest.mark.parametrize("field,value", [
    ("conclusion", "D2_EVIDENCE_INCOMPLETE"),
    ("d2_benchmark_executed", True),
    ("speed_or_quality_conclusion_available", True),
])
def test_d2_evidence_complete_semantics_change_fails_closed(tmp_path, field, value):
    evidence = tmp_path / "evidence"
    _copy_inputs(evidence)
    path = evidence / "D2_EVIDENCE_FINAL.json"
    _rewrite_canonical_artifact(path, lambda data: data.__setitem__(field, value))
    original = finalize.INPUT_NORMALIZED_TEXT_SHA256[path.name]
    finalize.INPUT_NORMALIZED_TEXT_SHA256[path.name] = finalize.normalized_utf8_text_sha256(path)
    try:
        with pytest.raises(finalize.EvidenceError, match="conclusion mismatch"):
            finalize.build_result(evidence)
    finally:
        finalize.INPUT_NORMALIZED_TEXT_SHA256[path.name] = original


def test_d2_blocked_not_promoted_and_no_combination(tmp_path):
    evidence, _, result_path, report_path, _ = _generate(tmp_path)
    result = finalize.load_result(result_path, evidence)
    d2 = next(item for item in result["findings"] if item["stage"] == "D2")
    assert d2["disposition"] == "NO_BENCHMARK_CONCLUSION"
    assert result["speed_or_quality_conclusion_for_235b_vs_14b"] is False
    assert result["eligible_for_mainline_combination"] == []
    report = report_path.read_text(encoding="utf-8")
    assert "benchmark 执行臂为 0" in report
    assert "不存在 235B 与 14B 的速度或质量结论" in report


def test_script_has_no_network_api_gpu_process_dependency():
    source = SCRIPT.read_text(encoding="utf-8").lower()
    for token in ("requests", "httpx", "openai", "socket", "subprocess", "jax",
                  "nvidia-smi", "urllib", "urlopen", "cuda", "ollama"):
        assert not re.search(rf"\b{re.escape(token)}\b", source), token


def test_atomic_refuses_sequential_overwrite(tmp_path):
    target = tmp_path / "target"
    finalize.atomic_write_refusing_overwrite(target, "first")
    with pytest.raises(FileExistsError):
        finalize.atomic_write_refusing_overwrite(target, "second")
    assert target.read_text() == "first"


def test_atomic_real_concurrent_no_clobber(tmp_path):
    target = tmp_path / "race"
    barrier = threading.Barrier(2)
    payloads = ["A" * 100_000, "B" * 100_000]
    def writer(payload):
        barrier.wait()
        try:
            finalize.atomic_write_refusing_overwrite(target, payload)
            return "success"
        except FileExistsError:
            return "exists"
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(writer, payloads))
    assert sorted(outcomes) == ["exists", "success"]
    assert target.read_text() in payloads
    assert not list(tmp_path.glob(".race.*"))


def test_historical_artifacts_byte_identical_before_after(tmp_path):
    evidence = tmp_path / "evidence"
    before = _copy_inputs(evidence)
    finalize.build_result(evidence)
    assert {name: (evidence / name).read_bytes() for name in before} == before


def test_real_historical_artifacts_unchanged_and_head_content_bound():
    before = {name: (LLM_D / name).read_bytes()
              for name in finalize.INPUT_NORMALIZED_TEXT_SHA256}
    result = finalize.build_result(LLM_D)
    after = {name: (LLM_D / name).read_bytes() for name in before}
    assert before == after
    assert {name: item["content_gate_sha256"]
            for name, item in result["input_artifacts"].items()} == \
        finalize.INPUT_NORMALIZED_TEXT_SHA256


def test_committed_result_loads_by_reconstructing_from_committed_inputs():
    result = finalize.load_result(LLM_D / "D_PHASE_FINAL_RESULT.json", LLM_D)
    assert result["branch"] == "perf/llm-production-shape-d1c"
    assert result["base_commit"] == finalize.BASE_COMMIT
    assert result["d2_head"] == finalize.D2_HEAD
