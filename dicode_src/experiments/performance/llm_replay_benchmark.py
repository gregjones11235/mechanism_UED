#!/usr/bin/env python3
"""LLM replay benchmark orchestrator (stage D research line).

Runs a frozen prompt replay against one model at one ``max_in_flight``, records
the timing events, runs post-generation validation (parse + static lint +
CPU JAX), dedups identical returned code by hash (without dropping any slot or
changing order), and derives union/critical-path/queue-wait metrics. Concurrency
durations are never naively summed: the report computes the overlap-union and
critical path so the real overlap benefit is measurable.

Independent research tool: does NOT import production ``llm.py``,
``gen_manager`` orchestration, or ``preflight_replay``.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from llm_replay_harness import (
    EventSink,
    LLMReplayClient,
    ProviderUnavailableError,
    extract_code,
    static_lint,
    cpu_jax_validation,
    enable_sdk_retry_counting,
)
from llm_replay_manifest import CLASSIFICATION as MANIFEST_CLASSIFICATION, canonical

CLASSIFICATION = "LLM_REPLAY_RESULT"

EVENT_FIELDS = (
    "run_id", "replay_id", "stage", "provider", "model", "max_in_flight",
    "request_id", "candidate_slot", "phase", "parent_phase",
    "start_monotonic_ns", "end_monotonic_ns", "duration_s", "status",
    "attempt", "http_status", "error_class", "prompt_sha256",
    "response_sha256", "overlap_group",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# Hash-contract constants. ``result_sha256`` uses the LEGACY default-JSON
# serialization (historical hashes must stay valid); the enriched-summary and
# artifact-inventory hashes use the CANONICAL serialization.
HASH_ALGORITHM_LEGACY = "legacy_default_json_sha256"
HASH_ALGORITHM_CANONICAL = "canonical_json_sha256"
RESULT_SHA256_SCOPE = "RESULT_FIELDS_EXCLUDING_RESULT_SHA256_AND_ARTIFACT_INVENTORY"
ENRICHED_SHA256_SCOPE = "ENRICHED_SUMMARY_FIELDS_EXCLUDING_ENRICHED_SUMMARY_SHA256"


def legacy_json_sha256(value: Mapping[str, Any]) -> str:
    """Historical default JSON serialization (default=str) SHA256 — the
    algorithm that produced all existing ``result_sha256`` values."""
    return sha256_bytes(json.dumps(value, sort_keys=True, default=str).encode())


def canonical_json_sha256(value: Any) -> str:
    """Canonical JSON SHA256: json.dumps sort_keys=True separators=(',',':')
    ensure_ascii=False, UTF-8, SHA256, with NaN/inf/array canonicalization."""
    return sha256_bytes(
        json.dumps(canonical(value), sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8")
    )


def _union_ns(intervals: list[tuple[int, int]]) -> int:
    merged: list[tuple[int, int]] = []
    for start, end in sorted((a, b) for a, b in intervals if b >= a):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, sort_keys=True, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EVENT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows({k: row.get(k) for k in EVENT_FIELDS} for row in rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def derive_reports(events: list[dict]) -> dict[str, Any]:
    """Compute overlap-union, critical path, and per-phase exclusive totals."""
    if not events:
        return {"event_count": 0, "phase_totals": {}, "exclusive_phase_totals": {},
                "llm_sum_s": 0.0, "llm_union_s": 0.0, "queue_wait_sum_s": 0.0,
                "retry_backoff_sum_s": 0.0, "wall_clock_s": 0.0}
    wall_start = min(int(e["start_monotonic_ns"]) for e in events)
    wall_end = max(int(e["end_monotonic_ns"]) for e in events)
    work = []
    for e in events:
        start = max(wall_start, int(e["start_monotonic_ns"]))
        end = min(wall_end, int(e["end_monotonic_ns"]))
        if end >= start:
            work.append((e, start, end))
    phase_totals: dict[str, float] = {}
    for phase in sorted({e.get("phase") for e in events}):
        intervals = [(s, en) for e, s, en in work if e.get("phase") == phase]
        phase_totals[phase] = _union_ns(intervals) / 1e9

    # deepest-active attribution for exclusive/critical-path
    from collections import defaultdict
    exclusive_ns: dict[str, int] = defaultdict(int)
    boundaries = sorted({p for _, s, e in work for p in (s, e)})
    def depth(row: dict, seen=None) -> int:
        seen = set() if seen is None else seen
        ph = row.get("phase")
        if ph in seen or not row.get("parent_phase"):
            return 0
        seen.add(ph)
        parent = next((c for c, _, _ in work if c.get("phase") == row.get("parent_phase")), None)
        return 1 + depth(parent, seen) if parent else 0
    for left, right in zip(boundaries, boundaries[1:]):
        active = [r for r, s, e in work if s <= left and e >= right]
        if active and right > left:
            chosen = max(active, key=lambda r: (depth(r), int(r.get("start_monotonic_ns", 0))))
            exclusive_ns[str(chosen.get("phase"))] += right - left
    exclusive = {ph: v / 1e9 for ph, v in exclusive_ns.items()}

    def _sum_phase(phase: str) -> float:
        return sum((int(e["end_monotonic_ns"]) - int(e["start_monotonic_ns"])) / 1e9
                   for e in events if e.get("phase") == phase)

    chat = [e for e in events if e.get("phase") == "chat_request"]
    sem_wait_sum = _sum_phase("queue_wait")
    return {
        "event_count": len(events),
        "wall_clock_s": (wall_end - wall_start) / 1e9,
        "phase_totals": phase_totals,
        "exclusive_phase_totals": exclusive,
        "llm_sum_s": _sum_phase("chat_request"),
        "llm_union_s": _union_ns([(int(e["start_monotonic_ns"]), int(e["end_monotonic_ns"]))
                                  for e in chat]) / 1e9 if chat else 0.0,
        # client-side semaphore wait, precisely named; queue_wait_sum_s is a
        # legacy alias retained for historical-file compatibility only.
        "client_semaphore_wait_sum_s": sem_wait_sum,
        "client_semaphore_wait_union_s": phase_totals.get("queue_wait", 0.0),
        "client_semaphore_wait_critical_s": exclusive.get("queue_wait", 0.0),
        "queue_wait_sum_s": sem_wait_sum,
        "legacy_alias": {"queue_wait_sum_s": True},
        "retry_backoff_sum_s": _sum_phase("retry_backoff"),
        "critical_path": sorted(
            ({"phase": ph, "duration_s": d} for ph, d in exclusive.items()),
            key=lambda x: x["duration_s"], reverse=True),
    }


def _validate_response(content: str | None, *, kind: str, do_cpu_jax: bool,
                       validated_cache: dict[str, dict], sink: EventSink,
                       slot: str, request_id: str,
                       static_lint_fn=static_lint,
                       cpu_jax_fn=cpu_jax_validation) -> dict[str, Any]:
    """Parse + static + CPU-JAX validation for one response, deduping by code hash.

    ``static_lint_fn`` / ``cpu_jax_fn`` are injectable so the dedup/ordering logic
    can be unit-tested without the craftax/jax environment installed.
    """
    if content is None:
        return {"valid": False, "reason": "empty", "code_hash": None,
                "static_ok": None, "jax_ok": None, "duplicate_of": None}
    p_start = time.monotonic_ns()
    code = extract_code(content)
    p_end = time.monotonic_ns()
    sink.record("response_parse", start_monotonic_ns=p_start, end_monotonic_ns=p_end,
                status="ok", request_id=request_id, candidate_slot=slot)
    if code is None or not code.strip():
        return {"valid": False, "reason": "no_code", "code_hash": None,
                "static_ok": None, "jax_ok": None, "duplicate_of": None}
    code_hash = sha256_bytes(code.replace("\r\n", "\n").replace("\r", "\n").encode())

    # dedup: identical code -> reuse prior verdict, never drop the slot
    if code_hash in validated_cache:
        prev = validated_cache[code_hash]
        with sink.span("candidate_finalize", status="ok" if prev["valid"] else "error",
                       request_id=request_id, candidate_slot=slot,
                       error_class=None if prev["valid"] else "static_invalid"):
            pass
        return {"valid": prev["valid"], "reason": "duplicate", "code_hash": code_hash,
                "static_ok": prev["static_ok"], "jax_ok": prev["jax_ok"],
                "duplicate_of": prev["slot"]}

    s_start = time.monotonic_ns()
    static_ok, static_msg = static_lint_fn(code)
    s_end = time.monotonic_ns()
    sink.record("static_validation", start_monotonic_ns=s_start, end_monotonic_ns=s_end,
                status="ok" if static_ok else "error", request_id=request_id,
                candidate_slot=slot,
                error_class=None if static_ok else "static_invalid")

    jax_ok = None
    if static_ok and do_cpu_jax:
        j_start = time.monotonic_ns()
        jax_ok, jax_msg = cpu_jax_fn(code)
        j_end = time.monotonic_ns()
        sink.record("cpu_jax_validation", start_monotonic_ns=j_start, end_monotonic_ns=j_end,
                    status="ok" if jax_ok else "error", request_id=request_id,
                    candidate_slot=slot,
                    error_class=None if jax_ok else "jax_validation_failed")
    valid = bool(static_ok and (jax_ok if do_cpu_jax else True))
    verdict = {"valid": valid, "reason": "compiled" if valid else
               ("static_invalid" if not static_ok else "jax_failed"),
               "code_hash": code_hash, "static_ok": static_ok, "jax_ok": jax_ok,
               "duplicate_of": None, "slot": slot}
    validated_cache[code_hash] = verdict
    with sink.span("candidate_finalize", status="ok" if valid else "error",
                   request_id=request_id, candidate_slot=slot,
                   error_class=None if valid else
                   ("static_invalid" if not static_ok else "jax_validation_failed")):
        pass
    return verdict


REQUIRED_ARTIFACTS = (
    "RESULT.json", "events.jsonl", "events.csv", "critical_path.json",
    "frozen_manifest.json", "SHA256SUMS", "ARTIFACT_INVENTORY.json",
)

_SDK_RETRY_COUNTER = None


def _sdk_retry_counter():
    global _SDK_RETRY_COUNTER
    if _SDK_RETRY_COUNTER is None:
        _SDK_RETRY_COUNTER = enable_sdk_retry_counting()
    return _SDK_RETRY_COUNTER


def verify_run_artifacts(out: Path) -> dict[str, Any]:
    """Check a run directory has every required raw artifact. A run missing
    ``events.jsonl`` (or any other required file) is NOT a complete PASS."""
    missing = [name for name in REQUIRED_ARTIFACTS if not (out / name).is_file()]
    return {"complete": not missing, "missing": missing,
            "status": "PASS" if not missing else "INCOMPLETE",
            "required": list(REQUIRED_ARTIFACTS)}


def _write_artifact_inventory(out: Path) -> dict[str, Any]:
    """Persist a per-run artifact inventory + SHA256SUMS (excluding the
    inventory/sums files themselves). Returns the inventory dict."""
    files = sorted(p for p in out.rglob("*") if p.is_file()
                   and p.name not in ("SHA256SUMS", "ARTIFACT_INVENTORY.json"))
    entries: dict[str, str] = {}
    for p in files:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        entries[str(p.relative_to(out))] = h.hexdigest()
    inv = {"classification": "ARTIFACT_INVENTORY", "artifact_count": len(entries),
           "out_dir": str(out), "files": entries}
    _atomic_json(out / "ARTIFACT_INVENTORY.json", inv)
    with open(out / "SHA256SUMS", "w", encoding="utf-8", newline="\n") as f:
        for name, h in sorted(entries.items()):
            f.write(f"{h}  {name}\n")
    return inv


async def run_replay(manifest: Mapping[str, Any], *, max_in_flight: int,
                     out_dir: str | Path, repeat_label: str = "r1",
                     do_cpu_jax: bool = True, enabled_events: bool = True,
                     run_id: str | None = None) -> dict[str, Any]:
    """Run one frozen replay at one concurrency level. Returns RESULT dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    replay_id = f"{manifest['provider']}-{manifest['model'].replace('/', '_')}-mif{max_in_flight}-{repeat_label}"
    run_id = run_id or uuid.uuid4().hex

    sink = EventSink(output_jsonl=str(out / "events.jsonl"), enabled=enabled_events,
                     run_id=run_id, replay_id=replay_id, provider=manifest["provider"],
                     model=manifest["model"], max_in_flight=max_in_flight)
    client = LLMReplayClient(
        base_url=manifest["base_url"], model=manifest["model"],
        provider=manifest["provider"], temperature=manifest["temperature"],
        top_p=manifest["top_p"], max_tokens=manifest["max_tokens"],
        timeout_s=manifest["timeout_s"], max_in_flight=max_in_flight, sink=sink,
    )

    # Fail closed before any measurement: on provider outage write FAILURE.json
    # (never a success RESULT) and re-raise.
    try:
        await client.health_check()
    except Exception as e:
        _atomic_json(out / "FAILURE.json", {"error_class": "provider_unavailable",
                                            "error": f"{type(e).__name__}: {e}"})
        raise

    _sdk_retry_counter().reset()

    system_prompt = manifest["system_prompt"]
    prompts = manifest["user_prompts"]
    slots = manifest["candidate_slots"]
    order = manifest["request_order"]
    prompt_hashes = manifest["user_prompt_sha256s"]

    wall_start = time.monotonic_ns()
    with sink.span("replay_wall", status="ok"):
        # Phase 1: concurrent generation, bounded by the client semaphore.
        async def one(i: int) -> tuple[int, dict]:
            slot = slots[i]
            kind = order[i].get("kind", "generation") if i < len(order) else "generation"
            rid = client.next_request_id()
            res = await client.chat_with_retries(
                system_prompt, prompts[i], slot=slot, request_id=rid,
                prompt_sha256=prompt_hashes[i], max_retries=manifest["max_retries"])
            return i, {"result": res, "kind": kind, "slot": slot, "request_id": rid}

        results = [None] * len(prompts)
        per = await asyncio.gather(*(one(i) for i in range(len(prompts))))
        for i, r in per:
            results[i] = r

        # Phase 2: post-generation validation (parse/static/CPU-JAX), dedup by hash.
        validated_cache: dict[str, dict] = {}
        verdicts = []
        for i, r in enumerate(results):
            content = r["result"]["content"]
            verdict = _validate_response(
                content, kind=r["kind"], do_cpu_jax=do_cpu_jax,
                validated_cache=validated_cache, sink=sink, slot=r["slot"],
                request_id=r["request_id"])
            verdicts.append(verdict)
    wall_end = time.monotonic_ns()

    # Load events for derived reports (from the JSONL we just wrote).
    events: list[dict] = []
    if enabled_events:
        with open(out / "events.jsonl", "r", encoding="utf-8") as f:
            events = [json.loads(line) for line in f if line.strip()]
    derived = derive_reports(events)

    # Quality + error aggregation.
    error_counts: dict[str, int] = {}
    retry_count = 0
    empty_count = 0
    for r in results:
        ec = r["result"].get("error_class")
        if ec is None:
            continue
        error_counts[ec] = error_counts.get(ec, 0) + 1
        if ec == "empty_response":
            empty_count += 1
    for e in events:
        if e.get("phase") == "retry_backoff":
            retry_count += 1

    valid = [v for v in verdicts if v.get("valid")]
    valid_count = len(valid)
    unique_codes = len({v["code_hash"] for v in verdicts if v.get("code_hash")})
    static_invalid = sum(1 for v in verdicts if v.get("static_ok") is False)
    jax_failed = sum(1 for v in verdicts if v.get("jax_ok") is False)

    llm_union_s = derived.get("llm_union_s", 0.0)
    result = {
        "classification": CLASSIFICATION,
        "not_end_to_end_ued": True,
        "run_id": run_id,
        "replay_id": replay_id,
        "repeat_label": repeat_label,
        "manifest_sha256": manifest.get("manifest_sha256"),
        "source_commit": manifest.get("source_commit"),
        "provider": manifest["provider"],
        "model": manifest["model"],
        "max_in_flight": max_in_flight,
        "temperature": manifest["temperature"],
        "max_tokens": manifest["max_tokens"],
        "request_count": len(prompts),
        "candidate_slots": len(slots),
        "wall_clock_s": derived.get("wall_clock_s", (wall_end - wall_start) / 1e9),
        "llm_sum_s": derived.get("llm_sum_s", 0.0),
        "llm_union_s": llm_union_s,
        "client_semaphore_wait_sum_s": derived.get("client_semaphore_wait_sum_s", 0.0),
        "client_semaphore_wait_union_s": derived.get("client_semaphore_wait_union_s", 0.0),
        "client_semaphore_wait_critical_s": derived.get("client_semaphore_wait_critical_s", 0.0),
        "queue_wait_sum_s": derived.get("queue_wait_sum_s", 0.0),
        "legacy_alias": {"queue_wait_sum_s": True},
        "retry_backoff_sum_s": derived.get("retry_backoff_sum_s", 0.0),
        "retry_count": retry_count,
        "sdk_transport_retry_count": _sdk_retry_counter().count(),
        "empty_response_count": empty_count,
        "error_counts": error_counts,
        "valid_tasks": valid_count,
        "invalid_tasks": len(verdicts) - valid_count,
        "unique_code_hashes": unique_codes,
        "static_invalid": static_invalid,
        "jax_failed": jax_failed,
        "valid_task_rate": (valid_count / len(verdicts)) if verdicts else 0.0,
        "llm_seconds_per_valid_task": (llm_union_s / valid_count) if valid_count else None,
        "critical_path": derived.get("critical_path", []),
        "do_cpu_jax": do_cpu_jax,
        "enabled_events": enabled_events,
        # hash-contract metadata (constant; part of the raw RESULT content)
        "result_sha256_scope": RESULT_SHA256_SCOPE,
        "result_sha256_algorithm": HASH_ALGORITHM_LEGACY,
        "legacy_result_hash_algorithm": HASH_ALGORITHM_LEGACY,
        "canonical_summary_hash_algorithm": HASH_ALGORITHM_CANONICAL,
    }
    result["result_sha256"] = legacy_json_sha256(
        {k: v for k, v in result.items() if k != "result_sha256"})
    with sink.span("result_write", status="ok"):
        _atomic_json(out / "RESULT.json", result)
    if enabled_events:
        _atomic_csv(out / "events.csv", events)
        _atomic_json(out / "critical_path.json", derived)
    # Persist the frozen manifest + a per-run artifact inventory/SHA256SUMS so
    # every raw artifact is reproducible after the run (audit evidence gate).
    from llm_replay_manifest import write_manifest
    write_manifest(manifest, out / "frozen_manifest.json")
    inv = _write_artifact_inventory(out)
    result["artifact_inventory"] = inv
    result["artifact_inventory_sha256"] = canonical_json_sha256(inv)
    result["enriched_summary_sha256_scope"] = ENRICHED_SHA256_SCOPE
    result["enriched_summary_sha256_algorithm"] = HASH_ALGORITHM_CANONICAL
    result["enriched_summary_sha256"] = canonical_json_sha256(
        {k: v for k, v in result.items() if k != "enriched_summary_sha256"})
    return result


def main(argv: list[str] | None = None) -> int:
    from llm_replay_manifest import load_manifest
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-in-flight", type=int, default=1)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repeat-label", default="r1")
    parser.add_argument("--no-cpu-jax", action="store_true")
    parser.add_argument("--no-events", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    try:
        result = asyncio.run(run_replay(
            manifest, max_in_flight=args.max_in_flight, out_dir=args.out_dir,
            repeat_label=args.repeat_label, do_cpu_jax=not args.no_cpu_jax,
            enabled_events=not args.no_events))
    except ProviderUnavailableError as e:
        _atomic_json(Path(args.out_dir) / "FAILURE.json",
                     {"error_class": "provider_unavailable", "error": str(e)})
        raise
    print(json.dumps({k: result[k] for k in
                      ("run_id", "replay_id", "max_in_flight", "wall_clock_s",
                       "llm_union_s", "llm_sum_s", "queue_wait_sum_s", "retry_count",
                       "valid_tasks", "valid_task_rate",
                       "llm_seconds_per_valid_task")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
