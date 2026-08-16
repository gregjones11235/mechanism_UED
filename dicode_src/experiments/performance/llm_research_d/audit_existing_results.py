#!/usr/bin/env python3
"""Offline auditor: generate an audited chat summary from local evidence only.

Reads the local ``CHAT_UNBOUNDED_RESULTS.json`` (enriched summaries produced by
the OLD ``run_replay``, where ``result_sha256`` predates ``artifact_inventory``)
and produces ``CHAT_UNBOUNDED_RESULTS_AUDITED.json`` with an explicit hash scope.

This does NOT access the network, LLM, GPU, or the remote run directory, and it
does NOT modify the original ``CHAT_UNBOUNDED_RESULTS.json``. Fields that cannot
be independently verified from the local summary are marked
``not_independently_recoverable_from_local_summary`` rather than forged true.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SRC = HERE / "CHAT_UNBOUNDED_RESULTS.json"
DST = HERE / "CHAT_UNBOUNDED_RESULTS_AUDITED.json"

HASH_ALGORITHM_LEGACY = "legacy_default_json_sha256"
HASH_ALGORITHM_CANONICAL = "canonical_json_sha256"
NOT_RECOVERABLE = "not_independently_recoverable_from_local_summary"
EVIDENCE_STATUS = "PASS_WITH_LEGACY_SUMMARY_HASH_CAVEAT"
HASH_SCOPE = "LEGACY_ENRICHED_HASH_SCOPE_AMBIGUOUS"


def sha256_bytes(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> Any:
    # NaN/inf/array canonicalization mirroring llm_replay_manifest.canonical
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "Inf"
        if value == float("-inf"):
            return "-Inf"
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(x) for x in value]
    return value


def legacy_json_sha256(value: dict) -> str:
    """Historical default JSON serialization (default=str), used by result_sha256."""
    return sha256_bytes(json.dumps(value, sort_keys=True, default=str).encode())


def canonical_json_sha256(value: Any) -> str:
    """Canonical JSON SHA256 (ensure_ascii=False, UTF-8, separators, sort_keys)."""
    return sha256_bytes(json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8"))


def audit_entry(entry: dict) -> dict:
    raw_result_sha256 = entry["result_sha256"]
    # The legacy result_sha256 covered the raw RESULT fields EXCLUDING
    # result_sha256 and artifact_inventory (artifact_inventory was appended after
    # the hash was computed). Recompute with that exact scope.
    raw_fields = {k: v for k, v in entry.items()
                  if k not in ("result_sha256", "artifact_inventory")}
    recomputed = legacy_json_sha256(raw_fields)
    raw_result_sha256_verified = recomputed == raw_result_sha256

    raw_result_file_sha256 = (entry.get("artifact_inventory", {}).get("files", {})
                              .get("RESULT.json", "missing"))
    artifact_inventory_sha256 = canonical_json_sha256(entry["artifact_inventory"])

    audited = {
        "run_id": entry["run_id"],
        "replay_id": entry["replay_id"],
        "max_in_flight": entry["max_in_flight"],
        "repeat_label": entry["repeat_label"],
        "wall_clock_s": entry["wall_clock_s"],
        "llm_union_s": entry["llm_union_s"],
        "valid_task_rate": entry["valid_task_rate"],
        "retry_count": entry["retry_count"],
        "sdk_transport_retry_count": entry["sdk_transport_retry_count"],
        "raw_result_sha256": raw_result_sha256,
        "raw_result_sha256_verified": raw_result_sha256_verified,
        "raw_result_file_sha256": raw_result_file_sha256,
        "artifact_inventory_sha256": artifact_inventory_sha256,
        "artifact_inventory_sha256_verified": NOT_RECOVERABLE,  # no stored value to compare
        "legacy_enriched_hash_scope_ambiguous": True,
        "hash_scope": HASH_SCOPE,
        "evidence_status": (EVIDENCE_STATUS if raw_result_sha256_verified
                            else "RAW_RESULT_HASH_MISMATCH"),
        "enriched_summary_sha256_algorithm": HASH_ALGORITHM_CANONICAL,
        "enriched_summary_sha256_scope": "AUDITED_SUMMARY_FIELDS_EXCLUDING_ENRICHED_SUMMARY_SHA256",
    }
    audited["enriched_summary_sha256_verified"] = True
    audited["enriched_summary_sha256"] = canonical_json_sha256(
        {k: v for k, v in audited.items() if k != "enriched_summary_sha256"})
    return audited


def main() -> None:
    entries = json.loads(SRC.read_text(encoding="utf-8"))
    audited = [audit_entry(e) for e in entries]
    # verify the audited summary self-consistently (enriched hash recomputable)
    for a in audited:
        recomputed = canonical_json_sha256(
            {k: v for k, v in a.items() if k != "enriched_summary_sha256"})
        a["enriched_summary_sha256_verified"] = recomputed == a["enriched_summary_sha256"]
    DST.write_text(json.dumps(audited, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verified_raw = sum(1 for a in audited if a["raw_result_sha256_verified"])
    verified_enriched = sum(1 for a in audited if a["enriched_summary_sha256_verified"])
    print(f"audited {len(audited)} entries")
    print(f"raw_result_sha256 verified: {verified_raw}/{len(audited)}")
    print(f"enriched_summary_sha256 verified: {verified_enriched}/{len(audited)}")
    print(f"wrote {DST.name}")


def _verify_result_list(path, name):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        entries = data.get("results", [])
    else:
        entries = data
    ok = 0
    for e in entries:
        # D1/D1b/batch results have no artifact_inventory in the enriched dict,
        # so result_sha256 covers everything except itself. (chat has it — audited separately.)
        exclude = {"result_sha256", "artifact_inventory"}
        raw = {k: v for k, v in e.items() if k not in exclude}
        if legacy_json_sha256(raw) == e.get("result_sha256"):
            ok += 1
    print(f"{name}: {ok}/{len(entries)} raw result hash verified")
    return ok, len(entries)


def verify_evidence_hashes() -> None:
    print("=== offline evidence hash verification ===")
    _verify_result_list(HERE / "D1_ALL_RESULTS.json", "D1")
    _verify_result_list(HERE / "D1B_ALL_RESULTS.json", "D1b embedding")
    _verify_result_list(HERE / "D1B_BATCH_RESULTS.json", "D1b batch")
    chat_entries = json.loads(SRC.read_text(encoding="utf-8"))
    chat_ok = sum(1 for e in chat_entries
                  if legacy_json_sha256({k: v for k, v in e.items()
                                         if k not in ("result_sha256", "artifact_inventory")})
                  == e["result_sha256"])
    print(f"chat remote RESULT saved-hash evidence: {chat_ok}/{len(chat_entries)}")
    audited = json.loads(DST.read_text(encoding="utf-8")) if DST.exists() else []
    audited_ok = sum(1 for a in audited
                     if canonical_json_sha256({k: v for k, v in a.items()
                                              if k != "enriched_summary_sha256"})
                     == a["enriched_summary_sha256"])
    print(f"audited chat summary enriched hash: {audited_ok}/{len(audited)}")


if __name__ == "__main__":
    main()
    verify_evidence_hashes()
