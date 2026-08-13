#!/usr/bin/env python3
"""Close research phase D from immutable, local evidence only.

The finalizer performs no provider, model, GPU, or process operation.  It
validates the saved D1/D1b/D1c/D2 artifacts, derives the deliberately narrow
phase disposition, and writes new result/report files with refuse-overwrite
semantics.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from statistics import median
from typing import Any


CANONICAL_ALGORITHM = "canonical_json_sha256"
RESULT_SCOPE = "D_PHASE_FINAL_RESULT_FIELDS_EXCLUDING_ARTIFACT_SHA256"
CONCLUSION = "D_PHASE_CLOSED_NO_PRODUCTION_OPTIMIZATION"
REVIEW_STATUS = "PASS_WITH_CONCERNS"
BASE_COMMIT = "453dc356d29dce783dfb7c6e915f5195dc272fe1"
D2_HEAD = "4fa39478ef02d68ff528155bcfcef429562f7de4"
BRANCH = "perf/llm-production-shape-d1c"

INPUT_RAW_SHA256 = {
    "D1_ALL_RESULTS.json": "0b009e1b02d161af78a39a54dcf23925ff418b44bb943fcf5db9f5ca67e60310",
    "D1B_ALL_RESULTS.json": "726012108ce8fcb67656727521ab39794bf55eaf9b9991de4ae97eb6ec8d1cae",
    "D1B_BATCH_RESULTS.json": "08e4a0adc8e3af7327e75cebabbc1ff6b39d53b9a4431f63c1cce3b4faacbae0",
    "CHAT_UNBOUNDED_RESULTS_AUDITED.json": "08b6a15301cc58fc8a19abbcdd9c45c99c5ccf07e4c76125ee7fff04118331c4",
    "D1C_ALL_RESULTS.json": "06574f6a264bb2ffdd0da243d1ca5a0fb699fcbb2b5f2c17dce995fb880551ed",
    "D2_RESULT.json": "45e1b0b35d2cbc1cde984685f00981bda055d52f2c368c9cef048d51324a1f0c",
    "D2_EVIDENCE_FINAL.json": "1356d9cfb4ad8ecb2783b7019f3745583c2e5031a3c7d57e831363dfb0797dd2",
}


class EvidenceError(ValueError):
    """The frozen evidence does not support the declared phase result."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical(value: Any) -> Any:
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "Inf"
        if value == float("-inf"):
            return "-Inf"
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items(), key=str)}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    return value


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        canonical(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def legacy_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode()
    return sha256_bytes(encoded)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot load evidence {path.name}: {exc}") from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _verify_legacy_entries(entries: list[dict[str, Any]], label: str) -> None:
    _require(bool(entries), f"{label}: no result entries")
    for index, entry in enumerate(entries):
        payload = {
            key: value for key, value in entry.items()
            if key not in ("result_sha256", "artifact_inventory")
        }
        _require(
            legacy_json_sha256(payload) == entry.get("result_sha256"),
            f"{label}[{index}]: result_sha256 mismatch",
        )


def _verify_canonical_artifact(
    artifact: dict[str, Any], *, label: str, algorithm: str, scope: str,
) -> None:
    _require(artifact.get("artifact_sha256_algorithm") == algorithm,
             f"{label}: hash algorithm mismatch")
    _require(artifact.get("artifact_sha256_scope") == scope,
             f"{label}: hash scope mismatch")
    payload = {key: value for key, value in artifact.items()
               if key != "artifact_sha256"}
    _require(canonical_json_sha256(payload) == artifact.get("artifact_sha256"),
             f"{label}: artifact_sha256 mismatch")


def validate_inputs(evidence_dir: Path) -> dict[str, Any]:
    evidence_dir = Path(evidence_dir)
    loaded: dict[str, Any] = {}
    bindings: dict[str, Any] = {}
    for name, expected_sha in INPUT_RAW_SHA256.items():
        path = evidence_dir / name
        _require(path.is_file(), f"missing input artifact: {name}")
        actual_sha = file_sha256(path)
        _require(actual_sha == expected_sha,
                 f"{name}: raw SHA256 mismatch")
        loaded[name] = _load_json(path)
        bindings[name] = {"raw_file_sha256": actual_sha}

    d1_entries = loaded["D1_ALL_RESULTS.json"].get("results", [])
    d1b_entries = loaded["D1B_ALL_RESULTS.json"].get("results", [])
    batch = loaded["D1B_BATCH_RESULTS.json"]
    _verify_legacy_entries(d1_entries, "D1")
    _verify_legacy_entries(d1b_entries, "D1b")
    _verify_legacy_entries(batch, "D1b-batch")

    chat = loaded["CHAT_UNBOUNDED_RESULTS_AUDITED.json"]
    _require(isinstance(chat, list) and len(chat) == 4,
             "audited chat: expected four entries")
    for index, entry in enumerate(chat):
        _require(entry.get("enriched_summary_sha256_algorithm") == CANONICAL_ALGORITHM,
                 f"audited chat[{index}]: algorithm mismatch")
        _require(entry.get("enriched_summary_sha256_scope") ==
                 "AUDITED_SUMMARY_FIELDS_EXCLUDING_ENRICHED_SUMMARY_SHA256",
                 f"audited chat[{index}]: scope mismatch")
        payload = {key: value for key, value in entry.items()
                   if key != "enriched_summary_sha256"}
        _require(canonical_json_sha256(payload) ==
                 entry.get("enriched_summary_sha256"),
                 f"audited chat[{index}]: canonical hash mismatch")
        _require(entry.get("raw_result_sha256_verified") is True,
                 f"audited chat[{index}]: raw result not verified")

    d1c = loaded["D1C_ALL_RESULTS.json"].get("results", [])
    _require(len(d1c) == 8, "D1c: expected eight entries")
    for index, entry in enumerate(d1c):
        _require(entry.get("enriched_summary_sha256_algorithm") ==
                 CANONICAL_ALGORITHM, f"D1c[{index}]: algorithm mismatch")
        _require(entry.get("enriched_summary_sha256_scope") ==
                 "ENRICHED_SUMMARY_FIELDS_EXCLUDING_ENRICHED_SUMMARY_SHA256",
                 f"D1c[{index}]: scope mismatch")
        payload = {key: value for key, value in entry.items()
                   if key != "enriched_summary_sha256"}
        _require(canonical_json_sha256(payload) ==
                 entry.get("enriched_summary_sha256"),
                 f"D1c[{index}]: canonical hash mismatch")

    d2_result = loaded["D2_RESULT.json"]
    _verify_canonical_artifact(
        d2_result, label="D2 result", algorithm=CANONICAL_ALGORITHM,
        scope="D2_RESULT_FIELDS_EXCLUDING_ARTIFACT_SHA256",
    )
    d2_final = loaded["D2_EVIDENCE_FINAL.json"]
    _verify_canonical_artifact(
        d2_final, label="D2 final evidence", algorithm=CANONICAL_ALGORITHM,
        scope="D2_EVIDENCE_FINAL_FIELDS_EXCLUDING_ARTIFACT_SHA256",
    )
    bindings["D2_RESULT.json"].update({
        "internal_artifact_sha256": d2_result["artifact_sha256"],
        "internal_hash_algorithm": d2_result["artifact_sha256_algorithm"],
        "internal_hash_scope": d2_result["artifact_sha256_scope"],
    })
    bindings["D2_EVIDENCE_FINAL.json"].update({
        "internal_artifact_sha256": d2_final["artifact_sha256"],
        "internal_hash_algorithm": d2_final["artifact_sha256_algorithm"],
        "internal_hash_scope": d2_final["artifact_sha256_scope"],
    })

    # Derive and gate the narrow conclusions from evidence, not report prose.
    chat_by_mif = {
        mif: [item for item in chat if item.get("max_in_flight") == mif]
        for mif in (1, 12)
    }
    _require(all(len(items) == 2 for items in chat_by_mif.values()),
             "chat: expected two repeats for mif=1 and mif=12")
    chat_wall_1 = median(item["wall_clock_s"] for item in chat_by_mif[1])
    chat_wall_12 = median(item["wall_clock_s"] for item in chat_by_mif[12])
    chat_gain_pct = (chat_wall_1 - chat_wall_12) / chat_wall_1 * 100.0
    _require(0.0 < chat_gain_pct < 10.0,
             "chat: evidence no longer supports limited, non-large gain")

    embed_by_mif = {
        mif: [item for item in d1b_entries if item.get("max_in_flight") == mif]
        for mif in (1, 25)
    }
    _require(all(len(items) == 2 for items in embed_by_mif.values()),
             "embedding: expected two repeats for mif=1 and mif=25")
    embed_1 = median(item["wall_clock_s"] for item in embed_by_mif[1])
    embed_25 = median(item["wall_clock_s"] for item in embed_by_mif[25])
    embed_gain_pct = (embed_1 - embed_25) / embed_1 * 100.0
    _require(70.0 < embed_gain_pct < 80.0,
             "embedding: synthetic gain outside audited range")
    _require(all(item.get("not_end_to_end_ued") is True for item in d1b_entries),
             "embedding: synthetic/non-end-to-end limitation missing")

    batch_by_label = {item["label"]: item for item in batch}
    _require({"1x25", "4x25", "25x25"}.issubset(batch_by_label),
             "batch embedding: expected stress arms missing")
    batch_base = batch_by_label["1x25"]["wall_clock_s"]
    batch_gain_4 = (batch_base - batch_by_label["4x25"]["wall_clock_s"]) / batch_base * 100.0
    batch_gain_25 = (batch_base - batch_by_label["25x25"]["wall_clock_s"]) / batch_base * 100.0
    _require(25.0 < batch_gain_4 < 35.0 and 25.0 < batch_gain_25 < 35.0,
             "batch embedding: stress gains outside audited range")

    _require({(item["arm"], item["repeat"]) for item in d1c} == {
        (arm, repeat) for arm in ("A", "B", "C30", "C120")
        for repeat in ("r1", "r2")
    }, "D1c: arm/repeat matrix mismatch")
    _require(all(item.get("total_sdk_retries") == 0 and
                 item.get("error_request_count") == 0 for item in d1c),
             "D1c: retry/error evidence changed")

    _require(d2_result.get("status") == "BLOCKED" and
             d2_result.get("arms_executed") == 0 and
             d2_result.get("performance_comparison_available") is False and
             d2_result.get("quality_comparison_available") is False,
             "D2 result improperly represents a completed benchmark")
    _require(d2_final.get("conclusion") ==
             "D2_BLOCKED_EXTERNAL_PROVIDER_EVIDENCE_COMPLETE" and
             d2_final.get("d2_benchmark_executed") is False and
             d2_final.get("speed_or_quality_conclusion_available") is False,
             "D2 final evidence conclusion mismatch")

    return {
        "bindings": bindings,
        "metrics": {
            "chat_wall_gain_pct": round(chat_gain_pct, 6),
            "chat_repeats_per_arm": 2,
            "synthetic_single_text_embedding_gain_pct": round(embed_gain_pct, 6),
            "stress_batch_embedding_gain_mif4_pct": round(batch_gain_4, 6),
            "stress_batch_embedding_gain_mif25_pct": round(batch_gain_25, 6),
            "d1c_total_requests": sum(item["batch_count"] for item in d1c),
            "d1c_sdk_retries": sum(item["total_sdk_retries"] for item in d1c),
        },
    }


def build_result(evidence_dir: Path) -> dict[str, Any]:
    validated = validate_inputs(evidence_dir)
    result: dict[str, Any] = {
        "classification": "D_PHASE_FINAL_RESULT",
        "phase": "D",
        "phase_execution_status": "COMPLETE",
        "review_status": REVIEW_STATUS,
        "conclusion": CONCLUSION,
        "branch": BRANCH,
        "base_commit": BASE_COMMIT,
        "d2_head": D2_HEAD,
        "eligible_for_mainline_combination": [],
        "production_code_modified_by_phase_close": False,
        "speed_or_quality_conclusion_for_235b_vs_14b": False,
        "findings": [
            {
                "stage": "D1",
                "conclusion": "D1_CHAT_CONCURRENCY_NO_LARGE_STABLE_GAIN",
                "disposition": "NOT_ELIGIBLE_FOR_MAINLINE_COMBINATION",
                "reason": "approximately 3.3% observed gain with only two repeats per arm",
            },
            {
                "stage": "D1b-single-text",
                "conclusion": "EMBEDDING_CONCURRENCY_SPEEDUP_OBSERVED_ON_SYNTHETIC_WORKLOAD",
                "disposition": "NOT_ELIGIBLE_FOR_MAINLINE_COMBINATION",
                "reason": "synthetic workload is not production-equivalent",
            },
            {
                "stage": "D1b-batch",
                "conclusion": "BATCH_EMBEDDING_CONCURRENCY_SPEEDUP_OBSERVED_IN_STRESS_REPLAY",
                "disposition": "NOT_ELIGIBLE_FOR_MAINLINE_COMBINATION",
                "reason": "stress replay is not production-equivalent",
            },
            {
                "stage": "D1c",
                "conclusion": "D1C_RETRY_NOT_REPRODUCED",
                "disposition": "NO_PRODUCTION_CHANGE",
                "reason": "Mason retry trigger remains unconfirmed",
            },
            {
                "stage": "D2",
                "conclusion": "D2_BLOCKED_EXTERNAL_PROVIDER_EVIDENCE_COMPLETE",
                "disposition": "NO_BENCHMARK_CONCLUSION",
                "reason": "zero benchmark arms executed; no 235B-vs-14B comparison",
            },
        ],
        "limitations": [
            "chat concurrency has only two repeats per arm",
            "embedding gains were measured only in synthetic or stress replay workloads",
            "Mason transport retry was not reproduced",
            "D2 provider gate was blocked and its benchmark did not run",
        ],
        "derived_metrics": validated["metrics"],
        "input_artifacts": validated["bindings"],
        "artifact_sha256_algorithm": CANONICAL_ALGORITHM,
        "artifact_sha256_scope": RESULT_SCOPE,
    }
    result["artifact_sha256"] = canonical_json_sha256(result)
    return result


def atomic_write_refusing_overwrite(path: Path, text: str) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_result(path: Path, result: dict[str, Any]) -> None:
    atomic_write_refusing_overwrite(
        path, json.dumps(canonical(result), indent=2, sort_keys=True,
                         ensure_ascii=False) + "\n",
    )


def load_result(path: Path) -> dict[str, Any]:
    result = _load_json(Path(path))
    _require(isinstance(result, dict), "final result must be a JSON object")
    _require(result.get("artifact_sha256_algorithm") == CANONICAL_ALGORITHM,
             "final result hash algorithm mismatch")
    _require(result.get("artifact_sha256_scope") == RESULT_SCOPE,
             "final result hash scope mismatch")
    payload = {key: value for key, value in result.items()
               if key != "artifact_sha256"}
    _require(canonical_json_sha256(payload) == result.get("artifact_sha256"),
             "final result artifact_sha256 mismatch (tampered)")
    _require(result.get("phase") == "D" and
             result.get("phase_execution_status") == "COMPLETE",
             "final phase state mismatch")
    _require(result.get("review_status") == REVIEW_STATUS and
             result.get("conclusion") == CONCLUSION,
             "final review conclusion mismatch")
    _require(result.get("eligible_for_mainline_combination") == [],
             "D phase cannot introduce a mainline combination candidate")
    d2 = next((item for item in result.get("findings", [])
               if item.get("stage") == "D2"), None)
    _require(d2 is not None and d2.get("disposition") == "NO_BENCHMARK_CONCLUSION",
             "D2 finding improperly represents a benchmark conclusion")
    return result


def render_report(result: dict[str, Any], result_raw_sha256: str,
                  finalizer_raw_sha256: str) -> str:
    inputs = "\n".join(
        f"- `{name}` raw SHA256: `{binding['raw_file_sha256']}`"
        for name, binding in sorted(result["input_artifacts"].items())
    )
    metrics = result["derived_metrics"]
    return f"""# D 阶段最终收口报告

## 结论

**`{CONCLUSION}`**

- 阶段状态：`COMPLETE`
- 审核状态：`{REVIEW_STATUS}`
- 可纳入主线组合的 D 阶段优化：`[]`
- 本收口没有修改生产代码，也没有执行新的 LLM、API、GPU 或 provider 实验。

这意味着 D 阶段研究工作已结束，但当前证据没有放行任何生产优化。它不等于“所有方向均无收益”：只表示已有收益没有达到可安全外推至生产主线的证据门槛。

## Git 边界

- branch：`{BRANCH}`
- base：`{BASE_COMMIT}`
- D2 审计修复 HEAD：`{D2_HEAD}`
- 本任务不 push、不 merge。

## 分项判定

1. **D1 Chat 并发**：观察到约 `{metrics['chat_wall_gain_pct']:.2f}%` 的有限改善，但每臂只有 2 次重复，不能证明大幅稳定收益，不进入组合。
2. **D1b 非批处理 embedding**：观察到约 `{metrics['synthetic_single_text_embedding_gain_pct']:.2f}%` 改善；这是合成 workload，不能外推生产。
3. **D1b 批处理 embedding**：压力 replay 中 mif=4 / mif=25 分别观察到约 `{metrics['stress_batch_embedding_gain_mif4_pct']:.2f}%` / `{metrics['stress_batch_embedding_gain_mif25_pct']:.2f}%` 改善；压力 replay 不等于 Mason 生产路径。
4. **D1c Mason retry**：96 个受控 batched 请求的 SDK retry 为 0，历史 retry 触发条件未复现，故不修改生产调度。
5. **D2 235B 对照**：仅完成阻塞证据审计；benchmark 执行臂为 0，不存在 235B 与 14B 的速度或质量结论。

## 证据绑定

{inputs}

- `D_PHASE_FINAL_RESULT.json` internal canonical SHA256: `{result['artifact_sha256']}`
- `D_PHASE_FINAL_RESULT.json` raw file SHA256: `{result_raw_sha256}`
- `d_phase_finalize.py` raw file SHA256: `{finalizer_raw_sha256}`

JSON 内部哈希使用 `canonical_json_sha256`，作用域为 `{RESULT_SCOPE}`。输出报告自身 raw SHA 不写入自身，以避免自引用循环。

## 遗留关注项

- Chat 每臂仅 2 次重复，缺少预设置信区间。
- embedding 收益只来自合成/压力 replay，未在生产 Mason session 重现。
- Mason 574/575 transport retry 根因仍未定位。
- D2 因 provider 可用性与授权证据阻塞，没有运行双模型 benchmark。

## 后续主线含义

D 阶段不向当前 B/C 组合优化追加变量。未来若重新研究 D，必须作为新的独立研究线，以生产形状 replay、预设重复次数和 provider 授权门禁重新开始，不能把本阶段的合成收益直接当作生产收益。
"""


def finalize(evidence_dir: Path, result_path: Path, report_path: Path) -> dict[str, Any]:
    if Path(result_path).exists() or Path(report_path).exists():
        existing = result_path if Path(result_path).exists() else report_path
        raise FileExistsError(f"refusing to overwrite existing {existing}")
    result = build_result(evidence_dir)
    write_result(result_path, result)
    # A failure after the result write remains fail-closed: a rerun refuses to
    # overwrite, preserving the exact first artifact for diagnosis.
    loaded = load_result(result_path)
    report = render_report(
        loaded, file_sha256(Path(result_path)), file_sha256(Path(__file__).resolve()),
    )
    atomic_write_refusing_overwrite(report_path, report)
    return loaded


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) not in (0, 3):
        raise SystemExit(
            "usage: d_phase_finalize.py [<evidence_dir> <result_json> <report_md>]"
        )
    here = Path(__file__).resolve().parent
    evidence_dir, result_path, report_path = (
        (here, here / "D_PHASE_FINAL_RESULT.json", here / "D_PHASE_FINAL_REPORT.md")
        if not args else tuple(Path(arg) for arg in args)
    )
    result = finalize(evidence_dir, result_path, report_path)
    print(json.dumps({
        "phase": result["phase"],
        "phase_execution_status": result["phase_execution_status"],
        "review_status": result["review_status"],
        "conclusion": result["conclusion"],
        "artifact_sha256": result["artifact_sha256"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
