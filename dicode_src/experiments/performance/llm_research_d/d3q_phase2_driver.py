"""D3Q phase-2 chunk driver: experiment-global ledger + launcher orchestration.

The phase-1 slot launcher keeps a per-run ledger on the remote host and
deletes it during cleanup, so the experiment-global POST budget (3 POSTs per
slot, 108 POSTs per provider, frozen) must be maintained by this driver.

Responsibilities:

* ``seed``: verify the frozen phase-1 smoke artifact byte-for-byte and seed
  the global ledger with the two POSTs it already performed.
* ``run-chunk``: validate the requested slots against the frozen order and
  the global budget, invoke ``d3q_slot_launcher`` for one chunk, then merge
  the chunk's per-POST metadata back into the global ledger (fail closed on
  any shape/identity mismatch, non-PASS chunk, or secret-shaped text).
* ``status``: report used/remaining budget state from the ledger.

This module performs no network access itself and imports no constants except
from the frozen phase-1 modules.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from d3q_budget import D3QLedger  # noqa: E402
import d3q_slot_runner as runner_mod  # noqa: E402

SEED_RUN_ID = "d3q_p1_smoke_20260815T022644Z"
ARTIFACTS_DIR = HERE / "d3q_artifacts"
DEFAULT_SEED_DIR = ARTIFACTS_DIR / SEED_RUN_ID
DEFAULT_LEDGER_PATH = ARTIFACTS_DIR / "D3Q_REQUEST_LEDGER.jsonl"
PROMPT_COUNT = 12
SEED_SLOTS = ("slot_r1_small_p00", "slot_r1_large_p00")
EXPECTED_SEED_COUNTS = {"ollama": 1, "deepseek_official": 1}
EXPECTED_REMAINING_AFTER_SEED = 70
REPEATS = ("r1", "r2", "r3")
RECONCILIATION_FILENAME = "D3Q_BUDGET_RECONCILIATION.json"
RECOVERY_FILENAME = "D3Q_CHUNK_RECOVERY.json"
ALLOWED_RECOVERY_REASONS = frozenset(
    {"gpu2_external_app", "ollama_pid_changed", "ollama_digest_changed"}
)

EVENT_FIELDS = (
    "ts_utc",
    "slot_id",
    "model",
    "provider",
    "kind",
    "attempt_index",
    "post_index_in_slot",
    "post_index_for_provider",
)


class Phase2Error(Exception):
    """Fail-closed driver error carrying a machine-readable reason."""

    def __init__(self, reason: str, detail: Any = None) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(reason if detail is None else f"{reason}: {detail}")


# ---------------------------------------------------------------------------
# Frozen slot order.
# ---------------------------------------------------------------------------


def all_slots_ordered() -> List[str]:
    """All 72 slot ids in the frozen ARM_ORDER x prompt-index order."""
    slots: List[str] = []
    for arm, repeat in runner_mod.ARM_ORDER:
        for prompt_index in range(PROMPT_COUNT):
            slots.append(f"slot_{repeat}_{arm}_p{prompt_index:02d}")
    return slots


def remaining_slots(ledger: D3QLedger) -> List[str]:
    """Slots with zero recorded POSTs, in frozen order."""
    return [slot for slot in all_slots_ordered() if ledger.slot_post_count(slot) == 0]


def remaining_slots_effective(
    ledger: D3QLedger, excluded: Sequence[str] = ()
) -> List[str]:
    """Slots with zero ledger POSTs and not excluded, in frozen order."""
    excluded_set = set(excluded)
    return [
        slot
        for slot in all_slots_ordered()
        if ledger.slot_post_count(slot) == 0 and slot not in excluded_set
    ]


def chunk_slots_for_repeat(
    repeat: str, ledger: D3QLedger, excluded: Sequence[str] = ()
) -> List[str]:
    """Untouched, non-excluded slots of one repeat, in frozen order."""
    if repeat not in REPEATS:
        raise Phase2Error("invalid_repeat", repeat)
    prefix = f"slot_{repeat}_"
    excluded_set = set(excluded)
    return [
        slot
        for slot in all_slots_ordered()
        if slot.startswith(prefix)
        and ledger.slot_post_count(slot) == 0
        and slot not in excluded_set
    ]


# ---------------------------------------------------------------------------
# Seed verification.
# ---------------------------------------------------------------------------


def _parse_sha256sums(seed_dir: Path) -> Dict[str, str]:
    sums_path = seed_dir / "SHA256SUMS"
    if not sums_path.is_file():
        raise Phase2Error("seed_sha256sums_missing", str(sums_path))
    digests: Dict[str, str] = {}
    for lineno, raw in enumerate(sums_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.rstrip()
        if not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise Phase2Error("seed_sha256sums_malformed", f"line {lineno}")
        digest, rel = parts[0].strip(), parts[1].strip().lstrip("*").replace("\\", "/")
        if not runner_mod.is_sha256(digest):
            raise Phase2Error("seed_sha256sums_malformed", f"line {lineno}: bad digest")
        if rel in digests:
            raise Phase2Error("seed_sha256sums_duplicate_entry", rel)
        digests[rel] = digest
    if not digests:
        raise Phase2Error("seed_sha256sums_empty", str(sums_path))
    return digests


def verify_seed_source(seed_dir: Path) -> Dict[str, Any]:
    """Verify listed file set and per-file hashes of the seed artifact."""
    seed_dir = Path(seed_dir)
    digests = _parse_sha256sums(seed_dir)
    actual = {
        path.relative_to(seed_dir).as_posix()
        for path in seed_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    listed = set(digests)
    if actual != listed:
        raise Phase2Error(
            "seed_file_set_mismatch",
            {
                "missing_from_sums": sorted(actual - listed),
                "listed_but_absent": sorted(listed - actual),
            },
        )
    for rel, expected in sorted(digests.items()):
        got = runner_mod.sha256_file(seed_dir / rel)
        if got != expected:
            raise Phase2Error(
                "seed_hash_mismatch", {"file": rel, "expected": expected, "got": got}
            )
    return {"files_checked": len(digests), "seed_dir": str(seed_dir)}


def _expected_identity(slot_id: str) -> tuple:
    _repeat, arm, _prompt = runner_mod.parse_slot_id(slot_id)
    provider, model, _base_url = runner_mod.arm_to_provider_model(arm)
    return provider, model


def extract_seed_events(seed_dir: Path) -> List[Dict[str, Any]]:
    """Extract the 6-field reserve events of the two seed slots (fail closed)."""
    seed_dir = Path(seed_dir)
    events: List[Dict[str, Any]] = []
    for slot_id in SEED_SLOTS:
        slot_dir = seed_dir / "slots" / slot_id
        meta_path = slot_dir / f"request_{slot_id}_a1.json"
        result_path = slot_dir / f"{slot_id}.result.json"
        if not meta_path.is_file():
            raise Phase2Error("seed_request_metadata_missing", str(meta_path))
        if not result_path.is_file():
            raise Phase2Error("seed_result_missing", str(result_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("classification") != "D3Q_REQUEST_METADATA":
            raise Phase2Error("seed_metadata_classification", str(meta_path))
        provider, model = _expected_identity(slot_id)
        if (
            meta.get("slot_id") != slot_id
            or meta.get("provider") != provider
            or meta.get("model") != model
        ):
            raise Phase2Error("seed_metadata_identity_mismatch", slot_id)
        event = meta.get("ledger_event")
        if not isinstance(event, dict):
            raise Phase2Error("seed_ledger_event_missing", str(meta_path))
        for field in EVENT_FIELDS:
            if field not in event:
                raise Phase2Error("seed_event_shape_unexpected", {"slot": slot_id, "field": field})
        if (
            not isinstance(event["ts_utc"], str)
            or not event["ts_utc"].strip()
            or event["slot_id"] != slot_id
            or event["provider"] != provider
            or event["model"] != model
            or not isinstance(event["attempt_index"], int)
            or not isinstance(event["post_index_in_slot"], int)
            or not isinstance(event["post_index_for_provider"], int)
        ):
            raise Phase2Error("seed_event_shape_unexpected", slot_id)
        if (
            event["kind"] != "initial"
            or event["attempt_index"] != 1
            or event["post_index_in_slot"] != 1
            or event["post_index_for_provider"] != 1
        ):
            raise Phase2Error("seed_event_indices_unexpected", slot_id)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("ledger_counts") != {"slot": 1, "provider": 1}:
            raise Phase2Error("seed_result_ledger_counts_unexpected", slot_id)
        if result.get("attempts") != 1 or len(result.get("attempts_detail") or []) != 1:
            raise Phase2Error("seed_result_attempts_unexpected", slot_id)
        events.append(
            {field: event[field] for field in ("ts_utc", "slot_id", "model", "provider", "kind", "attempt_index")}
        )
    return events


# ---------------------------------------------------------------------------
# Secret guard.
# ---------------------------------------------------------------------------


def _fail_on_secret_text(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    if runner_mod.contains_secret(text):
        raise Phase2Error("secret_detected", path.name)


# ---------------------------------------------------------------------------
# Budget reconciliation (incident D3Q_PHASE2_INCIDENT_01).
# ---------------------------------------------------------------------------


def _load_reconciliation(artifacts_dir: Path) -> Dict[str, Any]:
    """Load and fail-closed-validate the budget reconciliation record.

    Reconciliation accounts for POSTs that really reached a provider but whose
    per-POST metadata was lost to a tool failure.  Such slots are permanently
    barred from re-dispatch (their consumed budget counts against the frozen
    provider limits).
    """
    artifacts_dir = Path(artifacts_dir)
    path = artifacts_dir / RECONCILIATION_FILENAME
    if not path.is_file():
        return {"provider_consumed": {}, "slot_consumed": {}, "path": None}
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Phase2Error("reconciliation_malformed", str(exc))
    if not isinstance(record, dict):
        raise Phase2Error("reconciliation_shape", str(path))
    if record.get("classification") != "D3Q_BUDGET_RECONCILIATION":
        raise Phase2Error("reconciliation_classification", str(path))
    if record.get("schema_version") != 1:
        raise Phase2Error("reconciliation_schema_version", str(path))
    slot_consumed = record.get("slot_consumed")
    provider_consumed = record.get("provider_consumed")
    if not isinstance(slot_consumed, dict) or not isinstance(provider_consumed, dict):
        raise Phase2Error("reconciliation_shape", str(path))
    derived: Dict[str, int] = {}
    for slot_id, count in slot_consumed.items():
        if not isinstance(count, int) or not 1 <= count <= runner_mod.MAX_POSTS_PER_SLOT:
            raise Phase2Error("reconciliation_slot_count", slot_id)
        try:
            _repeat, arm, _prompt = runner_mod.parse_slot_id(slot_id)
        except ValueError:
            raise Phase2Error("reconciliation_slot_id", slot_id)
        provider, _model, _url = runner_mod.arm_to_provider_model(arm)
        derived[provider] = derived.get(provider, 0) + count
    if derived != dict(provider_consumed):
        raise Phase2Error(
            "reconciliation_provider_mismatch",
            {"derived": derived, "recorded": dict(provider_consumed)},
        )
    incident_sha = record.get("incident_result_sha256")
    incident_rel = record.get("incident_artifact")
    if incident_sha or incident_rel:
        incident_result = (
            artifacts_dir.parent / str(incident_rel) / "D3Q_SLOT_LAUNCHER_RESULT.json"
        )
        if (
            not isinstance(incident_sha, str)
            or not incident_result.is_file()
            or runner_mod.sha256_file(incident_result) != incident_sha
        ):
            raise Phase2Error("reconciliation_evidence_mismatch", str(incident_rel))
    _fail_on_secret_text(path)
    return {
        "provider_consumed": dict(provider_consumed),
        "slot_consumed": dict(slot_consumed),
        "path": str(path),
    }


# ---------------------------------------------------------------------------
# Commands.
# ---------------------------------------------------------------------------


def cmd_seed(
    ledger_path: Path, seed_dir: Path, force: bool = False
) -> Dict[str, Any]:
    ledger_path = Path(ledger_path)
    seed_dir = Path(seed_dir)
    if ledger_path.exists():
        if not force:
            raise Phase2Error("ledger_exists", str(ledger_path))
        ledger_path.unlink()
    verification = verify_seed_source(seed_dir)
    events = extract_seed_events(seed_dir)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = D3QLedger(ledger_path)
    for event in events:
        ledger.reserve(**event)
    provider_counts = {
        provider: ledger.provider_post_count(provider)
        for provider in sorted(EXPECTED_SEED_COUNTS)
    }
    if provider_counts != EXPECTED_SEED_COUNTS:
        raise Phase2Error("seed_provider_counts_unexpected", provider_counts)
    remaining = remaining_slots_effective(ledger)
    if len(remaining) != EXPECTED_REMAINING_AFTER_SEED:
        raise Phase2Error("seed_remaining_unexpected", len(remaining))
    _fail_on_secret_text(ledger_path)
    return {
        "status": "PASS",
        "ledger": str(ledger_path),
        "verification": verification,
        "seeded_events": events,
        "provider_counts": provider_counts,
        "remaining_slots": len(remaining),
    }


def merge_chunk_events(
    ledger: D3QLedger, artifact_dir: Path, slots: Sequence[str]
) -> List[Dict[str, Any]]:
    """Merge every POST of a published chunk into the global ledger."""
    artifact_dir = Path(artifact_dir)
    merged: List[Dict[str, Any]] = []
    for slot_id in slots:
        slot_dir = artifact_dir / "slots" / slot_id
        result_path = slot_dir / f"{slot_id}.result.json"
        if not result_path.is_file():
            raise Phase2Error("slot_result_missing", str(result_path))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        detail = result.get("attempts_detail")
        if (
            not isinstance(detail, list)
            or len(detail) < 1
            or result.get("attempts") != len(detail)
        ):
            raise Phase2Error("slot_attempts_shape_unexpected", slot_id)
        provider, model = _expected_identity(slot_id)
        posts = 0
        for index in range(1, len(detail) + 1):
            meta_path = slot_dir / f"request_{slot_id}_a{index}.json"
            if not meta_path.is_file():
                raise Phase2Error("request_metadata_missing", str(meta_path))
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("classification") != "D3Q_REQUEST_METADATA":
                raise Phase2Error("request_metadata_classification", str(meta_path))
            event = meta.get("ledger_event")
            if not isinstance(event, dict):
                raise Phase2Error("event_identity_mismatch", slot_id)
            if (
                event.get("slot_id") != slot_id
                or event.get("provider") != provider
                or event.get("model") != model
            ):
                raise Phase2Error("event_identity_mismatch", slot_id)
            if event.get("attempt_index") != index:
                raise Phase2Error("event_attempt_index_mismatch", slot_id)
            if event.get("post_index_in_slot") != index:
                raise Phase2Error("event_slot_index_mismatch", slot_id)
            meta_kind = meta.get("kind")
            detail_kind = detail[index - 1].get("kind") if isinstance(detail[index - 1], dict) else None
            if not (event.get("kind") == meta_kind == detail_kind):
                raise Phase2Error("event_kind_mismatch", slot_id)
            ledger.reserve(
                ts_utc=str(event.get("ts_utc")),
                slot_id=slot_id,
                model=model,
                provider=provider,
                kind=str(event.get("kind")),
                attempt_index=index,
            )
            posts += 1
        if (result.get("ledger_counts") or {}).get("slot") != posts:
            raise Phase2Error("slot_ledger_counts_mismatch", slot_id)
        merged.append(
            {"slot_id": slot_id, "posts": posts, "final_valid": result.get("final_valid")}
        )
    return merged


def _launcher_argv(run_id: str, slots: Sequence[str], artifacts_dir: Path) -> List[str]:
    return ["--run-id", run_id, "--slots", ",".join(slots), "--artifacts-dir", str(artifacts_dir)]


def _default_launcher(argv: List[str]) -> int:
    import d3q_slot_launcher  # noqa: E402

    return d3q_slot_launcher.main(argv)


def cmd_run_chunk(
    run_id: str,
    slots: Sequence[str],
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
    launcher: Optional[Callable[[List[str]], int]] = None,
) -> Dict[str, Any]:
    if not slots:
        raise Phase2Error("no_slots")
    slots = list(slots)
    if len(set(slots)) != len(slots):
        raise Phase2Error("duplicate_slots")
    ledger_path = Path(ledger_path)
    artifacts_dir = Path(artifacts_dir)
    if not ledger_path.is_file():
        raise Phase2Error("ledger_missing", str(ledger_path))
    ledger = D3QLedger(ledger_path)
    reconciliation = _load_reconciliation(artifacts_dir)
    order_index = {slot: index for index, slot in enumerate(all_slots_ordered())}
    for slot_id in slots:
        if slot_id not in order_index:
            raise Phase2Error("unknown_slot", slot_id)
    if [order_index[slot] for slot in slots] != sorted(order_index[slot] for slot in slots):
        raise Phase2Error("slots_out_of_frozen_order", slots)
    for slot_id in slots:
        if slot_id in reconciliation["slot_consumed"]:
            raise Phase2Error("slot_exhausted_reconciled", slot_id)
        if ledger.slot_post_count(slot_id) != 0:
            raise Phase2Error("slot_already_in_ledger", slot_id)
    provider_need: Dict[str, int] = {}
    for slot_id in slots:
        _repeat, arm, _prompt = runner_mod.parse_slot_id(slot_id)
        provider, _model, _url = runner_mod.arm_to_provider_model(arm)
        provider_need[provider] = provider_need.get(provider, 0) + runner_mod.MAX_POSTS_PER_SLOT
    for provider in sorted(provider_need):
        need = provider_need[provider]
        ledger_used = ledger.provider_post_count(provider)
        reconciled = int(reconciliation["provider_consumed"].get(provider, 0))
        if ledger_used + reconciled + need > runner_mod.MAX_PROVIDER_POSTS:
            raise Phase2Error(
                "provider_budget_exhausted",
                {
                    "provider": provider,
                    "need": need,
                    "used": ledger_used + reconciled,
                    "ledger_used": ledger_used,
                    "reconciled": reconciled,
                    "limit": runner_mod.MAX_PROVIDER_POSTS,
                },
            )
    chunk_dir = artifacts_dir / run_id
    if chunk_dir.exists():
        raise Phase2Error("local_output_exists", str(chunk_dir))
    invoke = launcher if launcher is not None else _default_launcher
    rc = invoke(_launcher_argv(run_id, slots, artifacts_dir))
    result_path = chunk_dir / "D3Q_SLOT_LAUNCHER_RESULT.json"
    if not result_path.is_file():
        raise Phase2Error("launcher_result_missing", str(result_path))
    launcher_result = json.loads(result_path.read_text(encoding="utf-8"))
    if launcher_result.get("status") != "PASS" or rc != 0:
        raise Phase2Error(
            "chunk_not_pass", {"rc": rc, "reason": launcher_result.get("reason")}
        )
    merged = merge_chunk_events(ledger, chunk_dir, slots)
    reloaded = D3QLedger(ledger_path)
    for slot_id in slots:
        if reloaded.slot_post_count(slot_id) != ledger.slot_post_count(slot_id):
            raise Phase2Error("ledger_reload_inconsistent", slot_id)
    for provider in provider_need:
        if reloaded.provider_post_count(provider) != ledger.provider_post_count(provider):
            raise Phase2Error("ledger_reload_inconsistent", provider)
    _fail_on_secret_text(ledger_path)
    return {
        "status": "PASS",
        "run_id": run_id,
        "slots": slots,
        "merged": merged,
        "provider_counts": {
            provider: ledger.provider_post_count(provider)
            for provider in (runner_mod.SMALL_PROVIDER, runner_mod.LARGE_PROVIDER)
        },
        "remaining_slots": len(
            remaining_slots_effective(ledger, reconciliation["slot_consumed"])
        ),
    }


def cmd_status(ledger_path: Path) -> Dict[str, Any]:
    ledger_path = Path(ledger_path)
    if not ledger_path.is_file():
        raise Phase2Error("ledger_missing", str(ledger_path))
    ledger = D3QLedger(ledger_path)
    reconciliation = _load_reconciliation(ledger_path.parent)
    excluded = sorted(reconciliation["slot_consumed"])
    remaining = remaining_slots_effective(ledger, excluded)
    providers = (runner_mod.SMALL_PROVIDER, runner_mod.LARGE_PROVIDER)
    return {
        "status": "PASS",
        "ledger": str(ledger_path),
        "slot_limit": ledger.slot_limit,
        "provider_limits": {provider: ledger.provider_limit for provider in providers},
        "provider_counts": {
            provider: ledger.provider_post_count(provider)
            + int(reconciliation["provider_consumed"].get(provider, 0))
            for provider in providers
        },
        "provider_counts_ledger_only": {
            provider: ledger.provider_post_count(provider) for provider in providers
        },
        "reconciled_slots": excluded,
        "touched_slots": [slot for slot in all_slots_ordered() if ledger.slot_post_count(slot) > 0],
        "remaining_count": len(remaining),
        "remaining_next": remaining[:6],
        "total_slots": len(all_slots_ordered()),
    }


def _verify_chunk_sha256sums(chunk_dir: Path) -> int:
    sums_path = chunk_dir / "SHA256SUMS"
    if not sums_path.is_file():
        raise Phase2Error("chunk_sha256sums_missing", str(sums_path))
    entries: Dict[str, str] = {}
    for lineno, raw in enumerate(sums_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.rstrip()
        if not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or not runner_mod.is_sha256(parts[0].strip()):
            raise Phase2Error("chunk_sha256sums_malformed", f"line {lineno}")
        rel = parts[1].strip().lstrip("*").replace("\\", "/")
        if rel in entries:
            raise Phase2Error("chunk_sha256sums_duplicate_entry", rel)
        entries[rel] = parts[0].strip()
    if not entries:
        raise Phase2Error("chunk_sha256sums_empty", str(sums_path))
    actual = {
        path.relative_to(chunk_dir).as_posix()
        for path in chunk_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if actual != set(entries):
        raise Phase2Error(
            "chunk_sha256sums_set_mismatch",
            {
                "missing_from_sums": sorted(actual - set(entries)),
                "listed_but_absent": sorted(set(entries) - actual),
            },
        )
    for rel, expected in sorted(entries.items()):
        got = runner_mod.sha256_file(chunk_dir / rel)
        if got != expected:
            raise Phase2Error("chunk_sha256sums_hash_mismatch", rel)
    return len(entries)


def cmd_recover_completed_chunk(
    run_id: str,
    incident_id: str,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> Dict[str, Any]:
    """Merge a chunk that completed all slots but was blocked only by a
    post-run environment gate (incident 02 semantics).  Fail closed unless
    every integrity condition holds."""
    if not incident_id.strip():
        raise Phase2Error("incident_id_missing")
    ledger_path = Path(ledger_path)
    artifacts_dir = Path(artifacts_dir)
    if not ledger_path.is_file():
        raise Phase2Error("ledger_missing", str(ledger_path))
    chunk_dir = artifacts_dir / run_id
    result_path = chunk_dir / "D3Q_SLOT_LAUNCHER_RESULT.json"
    if not result_path.is_file():
        raise Phase2Error("chunk_result_missing", str(result_path))
    launcher_result = json.loads(result_path.read_text(encoding="utf-8"))
    if launcher_result.get("classification") != "D3Q_SLOT_LAUNCHER":
        raise Phase2Error("chunk_result_classification", str(result_path))
    if launcher_result.get("status") != "BLOCKED":
        raise Phase2Error("recovery_not_blocked", launcher_result.get("status"))
    reason = launcher_result.get("reason")
    if reason not in ALLOWED_RECOVERY_REASONS:
        raise Phase2Error("recovery_reason_not_allowed", reason)
    slots = launcher_result.get("slots")
    if not isinstance(slots, list) or not slots or not all(isinstance(s, str) for s in slots):
        raise Phase2Error("recovery_slots_shape", str(result_path))
    ledger = D3QLedger(ledger_path)
    reconciliation = _load_reconciliation(artifacts_dir)
    order_index = {slot: index for index, slot in enumerate(all_slots_ordered())}
    for slot_id in slots:
        if slot_id not in order_index:
            raise Phase2Error("unknown_slot", slot_id)
        if slot_id in reconciliation["slot_consumed"]:
            raise Phase2Error("slot_exhausted_reconciled", slot_id)
        if ledger.slot_post_count(slot_id) != 0:
            raise Phase2Error("slot_already_in_ledger", slot_id)
    if [order_index[s] for s in slots] != sorted(order_index[s] for s in slots):
        raise Phase2Error("slots_out_of_frozen_order", slots)
    files_checked = _verify_chunk_sha256sums(chunk_dir)
    merged = merge_chunk_events(ledger, chunk_dir, slots)
    reloaded = D3QLedger(ledger_path)
    for slot_id in slots:
        if reloaded.slot_post_count(slot_id) != ledger.slot_post_count(slot_id):
            raise Phase2Error("ledger_reload_inconsistent", slot_id)
    _fail_on_secret_text(ledger_path)
    recovery = {
        "classification": "D3Q_CHUNK_RECOVERY",
        "schema_version": 1,
        "run_id": run_id,
        "incident_id": incident_id.strip(),
        "blocked_reason": reason,
        "launcher_result_sha256": runner_mod.sha256_file(result_path),
        "chunk_sha256sums_files_checked": files_checked,
        "slots": slots,
        "merged": merged,
        "provider_counts_after": {
            provider: ledger.provider_post_count(provider)
            for provider in (runner_mod.SMALL_PROVIDER, runner_mod.LARGE_PROVIDER)
        },
        "remaining_slots_after": len(
            remaining_slots_effective(ledger, reconciliation["slot_consumed"])
        ),
    }
    recovery_path = artifacts_dir / RECOVERY_FILENAME
    existing = []
    if recovery_path.is_file():
        loaded = json.loads(recovery_path.read_text(encoding="utf-8"))
        existing = loaded.get("recoveries", []) if isinstance(loaded, dict) else []
        if any(item.get("run_id") == run_id for item in existing):
            raise Phase2Error("recovery_already_recorded", run_id)
    document = {
        "classification": "D3Q_CHUNK_RECOVERY",
        "schema_version": 1,
        "recoveries": existing + [recovery],
    }
    recovery_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return {"status": "PASS", "recovery": recovery}


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="d3q_phase2_driver")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed_parser = subparsers.add_parser("seed", help="seed the global ledger from the frozen smoke artifact")
    seed_parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    seed_parser.add_argument("--seed-dir", default=str(DEFAULT_SEED_DIR))
    seed_parser.add_argument("--force", action="store_true")

    status_parser = subparsers.add_parser("status", help="report ledger budget state")
    status_parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))

    chunk_parser = subparsers.add_parser("run-chunk", help="run one chunk of untouched slots")
    chunk_parser.add_argument("--run-id", required=True)
    slot_source = chunk_parser.add_mutually_exclusive_group(required=True)
    slot_source.add_argument("--slots", help="comma-separated slot ids in frozen order")
    slot_source.add_argument("--repeat", help="run every untouched slot of this repeat")
    chunk_parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    chunk_parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))

    recover_parser = subparsers.add_parser(
        "recover-completed-chunk",
        help="merge a completed chunk blocked only by a post-run environment gate",
    )
    recover_parser.add_argument("--run-id", required=True)
    recover_parser.add_argument("--incident-id", required=True)
    recover_parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    recover_parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))

    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "seed":
            result = cmd_seed(Path(args.ledger), Path(args.seed_dir), force=args.force)
        elif args.command == "status":
            result = cmd_status(Path(args.ledger))
        elif args.command == "recover-completed-chunk":
            result = cmd_recover_completed_chunk(
                args.run_id,
                args.incident_id,
                ledger_path=Path(args.ledger),
                artifacts_dir=Path(args.artifacts_dir),
            )
        else:
            ledger_path = Path(args.ledger)
            artifacts_dir = Path(args.artifacts_dir)
            if args.slots is not None:
                slots = [item.strip() for item in args.slots.split(",") if item.strip()]
            else:
                if args.repeat not in REPEATS:
                    raise Phase2Error("invalid_repeat", args.repeat)
                if not ledger_path.is_file():
                    raise Phase2Error("ledger_missing", str(ledger_path))
                reconciliation = _load_reconciliation(artifacts_dir)
                slots = chunk_slots_for_repeat(
                    args.repeat,
                    D3QLedger(ledger_path),
                    excluded=sorted(reconciliation["slot_consumed"]),
                )
            result = cmd_run_chunk(
                args.run_id,
                slots,
                ledger_path=ledger_path,
                artifacts_dir=artifacts_dir,
            )
    except Phase2Error as exc:
        print(
            json.dumps(
                {"status": "BLOCKED", "reason": exc.reason, "detail": exc.detail},
                sort_keys=True,
                default=str,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
