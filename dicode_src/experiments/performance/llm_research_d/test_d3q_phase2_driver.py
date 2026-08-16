"""Tests for the D3Q phase-2 chunk driver (ledger seeding + chunk orchestration)."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import d3q_phase2_driver as drv  # noqa: E402
import d3q_slot_runner as runner  # noqa: E402
from d3q_budget import D3QLedger  # noqa: E402

SEED_SRC = drv.ARTIFACTS_DIR / drv.SEED_RUN_ID


@pytest.fixture()
def seed_copy(tmp_path):
    seed = tmp_path / "seed"
    shutil.copytree(SEED_SRC, seed)
    return seed


@pytest.fixture()
def seeded(tmp_path, seed_copy):
    ledger_path = tmp_path / "ledger.jsonl"
    result = drv.cmd_seed(ledger_path, seed_copy)
    assert result["status"] == "PASS"
    return ledger_path, seed_copy


def _fake_chunk_artifact(artifacts_dir, run_id, spec, status="PASS"):
    chunk_dir = Path(artifacts_dir) / run_id
    slots_dir = chunk_dir / "slots"
    for slot_id, kinds in spec:
        slot_dir = slots_dir / slot_id
        slot_dir.mkdir(parents=True, exist_ok=True)
        provider, model, _url = runner.arm_to_provider_model(runner.parse_slot_id(slot_id)[1])
        detail = []
        for index, kind in enumerate(kinds, start=1):
            event = {
                "ts_utc": "2026-08-16T00:00:00Z",
                "slot_id": slot_id,
                "model": model,
                "provider": provider,
                "kind": kind,
                "attempt_index": index,
                "post_index_in_slot": index,
                "post_index_for_provider": index,
            }
            meta = {
                "classification": "D3Q_REQUEST_METADATA",
                "slot_id": slot_id,
                "provider": provider,
                "model": model,
                "kind": kind,
                "attempt_index": index,
                "ledger_event": event,
            }
            (slot_dir / f"request_{slot_id}_a{index}.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            detail.append({"attempt_index": index, "kind": kind})
        result = {
            "attempts": len(kinds),
            "attempts_detail": detail,
            "ledger_counts": {"slot": len(kinds), "provider": len(kinds)},
            "final_valid": True,
        }
        (slot_dir / f"{slot_id}.result.json").write_text(json.dumps(result), encoding="utf-8")
    reason = None if status == "PASS" else "forced_failure"
    (chunk_dir / "D3Q_SLOT_LAUNCHER_RESULT.json").write_text(
        json.dumps({"status": status, "run_id": run_id, "reason": reason}), encoding="utf-8"
    )
    return chunk_dir


def _make_fake_launcher(calls, status="PASS"):
    def fake_launcher(argv):
        calls.append(list(argv))
        options = {}
        iterator = iter(argv)
        for token in iterator:
            if token == "--run-id":
                options["run_id"] = next(iterator)
            elif token == "--slots":
                options["slots"] = next(iterator).split(",")
            elif token == "--artifacts-dir":
                options["artifacts_dir"] = next(iterator)
        spec = [(slot, ["initial"]) for slot in options["slots"]]
        _fake_chunk_artifact(options["artifacts_dir"], options["run_id"], spec, status=status)
        return 0 if status == "PASS" else 3

    return fake_launcher


def _read_ledger_lines(ledger_path):
    return [json.loads(line) for line in Path(ledger_path).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_all_slots_ordered_72_and_boundaries():
    slots = drv.all_slots_ordered()
    assert len(slots) == 72
    assert len(set(slots)) == 72
    assert slots[0] == "slot_r1_small_p00"
    assert slots[11] == "slot_r1_small_p11"
    assert slots[12] == "slot_r1_large_p00"
    assert slots[24] == "slot_r2_large_p00"
    assert slots[36] == "slot_r2_small_p00"
    assert slots[48] == "slot_r3_small_p00"
    assert slots[60] == "slot_r3_large_p00"


def test_seed_success_counts_and_lines(seeded):
    ledger_path, _seed = seeded
    lines = _read_ledger_lines(ledger_path)
    assert len(lines) == 2
    assert lines[0]["provider"] == "ollama"
    assert lines[0]["slot_id"] == "slot_r1_small_p00"
    assert lines[0]["post_index_in_slot"] == 1
    assert lines[0]["post_index_for_provider"] == 1
    assert lines[1]["provider"] == "deepseek_official"
    assert lines[1]["slot_id"] == "slot_r1_large_p00"
    assert lines[1]["post_index_for_provider"] == 1


def test_seed_rejects_existing_ledger(tmp_path, seed_copy):
    ledger_path = tmp_path / "ledger.jsonl"
    assert drv.cmd_seed(ledger_path, seed_copy)["status"] == "PASS"
    with pytest.raises(drv.Phase2Error) as exc:
        drv.cmd_seed(ledger_path, seed_copy)
    assert exc.value.reason == "ledger_exists"
    again = drv.cmd_seed(ledger_path, seed_copy, force=True)
    assert again["status"] == "PASS"
    assert len(_read_ledger_lines(ledger_path)) == 2


def test_seed_hash_tamper_rejected(seed_copy, tmp_path):
    target = seed_copy / "slots" / "slot_r1_small_p00" / "candidate_attempt_1.py"
    target.write_bytes(target.read_bytes() + b"# tampered\n")
    with pytest.raises(drv.Phase2Error) as exc:
        drv.cmd_seed(tmp_path / "ledger.jsonl", seed_copy)
    assert exc.value.reason == "seed_hash_mismatch"


def test_seed_stray_file_rejected(seed_copy, tmp_path):
    (seed_copy / "slots" / "extra.txt").write_text("stray", encoding="utf-8")
    with pytest.raises(drv.Phase2Error) as exc:
        drv.cmd_seed(tmp_path / "ledger.jsonl", seed_copy)
    assert exc.value.reason == "seed_file_set_mismatch"
    assert "slots/extra.txt" in exc.value.detail["missing_from_sums"]


def test_seed_missing_listed_file_rejected(seed_copy, tmp_path):
    (seed_copy / "slots" / "slot_r1_small_p00" / "final_code.py").unlink()
    with pytest.raises(drv.Phase2Error) as exc:
        drv.cmd_seed(tmp_path / "ledger.jsonl", seed_copy)
    assert exc.value.reason == "seed_file_set_mismatch"
    assert "slots/slot_r1_small_p00/final_code.py" in exc.value.detail["listed_but_absent"]


def test_remaining_excludes_seed_slots(seeded):
    ledger_path, _seed = seeded
    ledger = D3QLedger(ledger_path)
    remaining = drv.remaining_slots(ledger)
    assert len(remaining) == 70
    assert remaining[0] == "slot_r1_small_p01"
    assert "slot_r1_small_p00" not in remaining
    assert "slot_r1_large_p00" not in remaining


def test_budget_counts_seeded(seeded):
    ledger_path, _seed = seeded
    ledger = D3QLedger(ledger_path)
    assert ledger.provider_post_count("ollama") == 1
    assert ledger.provider_post_count("deepseek_official") == 1
    assert ledger.slot_post_count("slot_r1_small_p00") == 1
    assert ledger.slot_post_count("slot_r1_large_p00") == 1
    assert ledger.slot_post_count("slot_r1_small_p01") == 0


def test_merge_renumbers_global_provider_indices(seeded, tmp_path):
    ledger_path, _seed = seeded
    ledger = D3QLedger(ledger_path)
    chunk_dir = _fake_chunk_artifact(
        tmp_path / "artifacts",
        "d3q_p2_fake_merge",
        [("slot_r1_small_p01", ["initial", "semantic_repair"])],
    )
    merged = drv.merge_chunk_events(ledger, chunk_dir, ["slot_r1_small_p01"])
    assert merged == [{"slot_id": "slot_r1_small_p01", "posts": 2, "final_valid": True}]
    lines = _read_ledger_lines(ledger_path)
    ollama_indices = [
        line["post_index_for_provider"] for line in lines if line["provider"] == "ollama"
    ]
    assert ollama_indices == [1, 2, 3]
    assert lines[-1]["kind"] == "semantic_repair"
    assert lines[-1]["attempt_index"] == 2
    reloaded = D3QLedger(ledger_path)
    assert reloaded.slot_post_count("slot_r1_small_p01") == 2
    assert reloaded.provider_post_count("ollama") == 3


def test_merge_tampered_event_provider_rejected(seeded, tmp_path):
    ledger_path, _seed = seeded
    ledger = D3QLedger(ledger_path)
    chunk_dir = _fake_chunk_artifact(
        tmp_path / "artifacts", "d3q_p2_fake_tamper", [("slot_r1_small_p01", ["initial"])]
    )
    meta_path = chunk_dir / "slots" / "slot_r1_small_p01" / "request_slot_r1_small_p01_a1.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["ledger_event"]["provider"] = "deepseek_official"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(drv.Phase2Error) as exc:
        drv.merge_chunk_events(ledger, chunk_dir, ["slot_r1_small_p01"])
    assert exc.value.reason == "event_identity_mismatch"


def test_run_chunk_rejects_touched_slot_without_launcher(seeded, tmp_path):
    ledger_path, _seed = seeded
    calls = []
    with pytest.raises(drv.Phase2Error) as exc:
        drv.cmd_run_chunk(
            "d3q_p2_fake_touched",
            ["slot_r1_small_p00"],
            ledger_path=ledger_path,
            artifacts_dir=tmp_path / "artifacts",
            launcher=_make_fake_launcher(calls),
        )
    assert exc.value.reason == "slot_already_in_ledger"
    assert calls == []


def test_run_chunk_out_of_order_rejected(seeded, tmp_path):
    ledger_path, _seed = seeded
    calls = []
    with pytest.raises(drv.Phase2Error) as exc:
        drv.cmd_run_chunk(
            "d3q_p2_fake_order",
            ["slot_r1_large_p00", "slot_r1_small_p01"],
            ledger_path=ledger_path,
            artifacts_dir=tmp_path / "artifacts",
            launcher=_make_fake_launcher(calls),
        )
    assert exc.value.reason == "slots_out_of_frozen_order"
    assert calls == []


def test_chunk_slots_for_repeat(seeded):
    ledger_path, _seed = seeded
    ledger = D3QLedger(ledger_path)
    r1 = drv.chunk_slots_for_repeat("r1", ledger)
    assert len(r1) == 22
    assert r1[0] == "slot_r1_small_p01"
    # Both r1 seed slots (small_p00 AND large_p00) are excluded, so the 12th
    # untouched r1 slot is large_p01.
    assert r1[10] == "slot_r1_small_p11"
    assert r1[11] == "slot_r1_large_p01"
    r2 = drv.chunk_slots_for_repeat("r2", D3QLedger(ledger_path))
    assert len(r2) == 24
    assert r2[0] == "slot_r2_large_p00"


def test_provider_budget_exhausted(seeded, tmp_path, monkeypatch):
    ledger_path, _seed = seeded
    monkeypatch.setattr(drv.runner_mod, "MAX_PROVIDER_POSTS", 2)
    with pytest.raises(drv.Phase2Error) as exc:
        drv.cmd_run_chunk(
            "d3q_p2_fake_exhaust",
            ["slot_r1_small_p01"],
            ledger_path=ledger_path,
            artifacts_dir=tmp_path / "artifacts",
            launcher=_make_fake_launcher([]),
        )
    assert exc.value.reason == "provider_budget_exhausted"


def test_run_chunk_end_to_end_fake_launcher(seeded, tmp_path):
    ledger_path, _seed = seeded
    calls = []
    artifacts_dir = tmp_path / "artifacts"
    result = drv.cmd_run_chunk(
        "d3q_p2_fake_chunk",
        ["slot_r1_small_p01", "slot_r1_large_p01"],
        ledger_path=ledger_path,
        artifacts_dir=artifacts_dir,
        launcher=_make_fake_launcher(calls),
    )
    assert result["status"] == "PASS"
    assert result["provider_counts"] == {"ollama": 2, "deepseek_official": 2}
    assert result["remaining_slots"] == 68
    assert [item["posts"] for item in result["merged"]] == [1, 1]
    assert len(calls) == 1


def test_chunk_not_pass_leaves_ledger_unchanged(seeded, tmp_path):
    ledger_path, _seed = seeded
    before = _read_ledger_lines(ledger_path)
    artifacts_dir = tmp_path / "artifacts"
    with pytest.raises(drv.Phase2Error) as exc:
        drv.cmd_run_chunk(
            "d3q_p2_fake_blocked",
            ["slot_r1_small_p01"],
            ledger_path=ledger_path,
            artifacts_dir=artifacts_dir,
            launcher=_make_fake_launcher([], status="BLOCKED"),
        )
    assert exc.value.reason == "chunk_not_pass"
    assert _read_ledger_lines(ledger_path) == before


def test_cli_seed_and_status(seed_copy, tmp_path, capsys):
    ledger_path = tmp_path / "ledger.jsonl"
    rc = drv.main(["seed", "--ledger", str(ledger_path), "--seed-dir", str(seed_copy)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"
    assert payload["remaining_slots"] == 70
    rc = drv.main(["status", "--ledger", str(ledger_path)])
    assert rc == 0
    status = json.loads(capsys.readouterr().out)
    assert status["remaining_count"] == 70
    assert status["total_slots"] == 72
    assert status["provider_counts"] == {"ollama": 1, "deepseek_official": 1}
    assert status["remaining_next"][0] == "slot_r1_small_p01"


def test_cli_seed_tamper_rc2(seed_copy, tmp_path, capsys):
    target = seed_copy / "slots" / "slot_r1_large_p00" / "raw_slot_r1_large_p00_a1.txt"
    target.write_bytes(target.read_bytes() + b"tamper")
    rc = drv.main(["seed", "--ledger", str(tmp_path / "ledger.jsonl"), "--seed-dir", str(seed_copy)])
    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "BLOCKED"
    assert payload["reason"] == "seed_hash_mismatch"


# ---------------------------------------------------------------------------
# Budget reconciliation (incident D3Q_PHASE2_INCIDENT_01 semantics).
# ---------------------------------------------------------------------------


def _write_reconciliation(artifacts_dir, slot_consumed=None, provider_consumed=None):
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if slot_consumed is None:
        slot_consumed = {
            "slot_r1_small_p01": 1,
            "slot_r1_small_p02": 1,
            "slot_r1_small_p03": 1,
            "slot_r1_small_p04": 1,
            "slot_r1_small_p05": 3,
        }
    if provider_consumed is None:
        provider_consumed = {"ollama": sum(slot_consumed.values())}
    incident_dir = artifacts_dir.parent / "fake_incident"
    incident_dir.mkdir(parents=True, exist_ok=True)
    incident_result = incident_dir / "D3Q_SLOT_LAUNCHER_RESULT.json"
    incident_result.write_text(
        json.dumps({"status": "BLOCKED", "reason": "budget_exceeded"}), encoding="utf-8"
    )
    record = {
        "classification": "D3Q_BUDGET_RECONCILIATION",
        "schema_version": 1,
        "recorded_utc": "2026-08-16T04:40:00Z",
        "incident_id": "FAKE_INCIDENT",
        "incident_run_id": "d3q_fake",
        "incident_artifact": "fake_incident",
        "incident_result_sha256": runner.sha256_file(incident_result),
        "disposition": "attrition_no_rerun",
        "provider_consumed": provider_consumed,
        "slot_consumed": slot_consumed,
    }
    (artifacts_dir / drv.RECONCILIATION_FILENAME).write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )


def test_reconciliation_blocks_reconciled_slot(seeded, tmp_path):
    ledger_path, _seed = seeded
    artifacts_dir = tmp_path / "artifacts"
    _write_reconciliation(artifacts_dir)
    with pytest.raises(drv.Phase2Error) as exc:
        drv.cmd_run_chunk(
            "d3q_p2_fake_recon",
            ["slot_r1_small_p05"],
            ledger_path=ledger_path,
            artifacts_dir=artifacts_dir,
            launcher=_make_fake_launcher([]),
        )
    assert exc.value.reason == "slot_exhausted_reconciled"
    ledger = D3QLedger(ledger_path)
    excluded = sorted(drv._load_reconciliation(artifacts_dir)["slot_consumed"])
    r1 = drv.chunk_slots_for_repeat("r1", ledger, excluded=excluded)
    assert len(r1) == 17
    assert "slot_r1_small_p05" not in r1
    assert r1[0] == "slot_r1_small_p06"


def test_reconciliation_counts_against_provider_budget(seeded, tmp_path, monkeypatch):
    ledger_path, _seed = seeded
    artifacts_dir = tmp_path / "artifacts"
    _write_reconciliation(artifacts_dir)
    monkeypatch.setattr(drv.runner_mod, "MAX_PROVIDER_POSTS", 10)
    with pytest.raises(drv.Phase2Error) as exc:
        drv.cmd_run_chunk(
            "d3q_p2_fake_recon_budget",
            ["slot_r1_small_p06"],
            ledger_path=ledger_path,
            artifacts_dir=artifacts_dir,
            launcher=_make_fake_launcher([]),
        )
    assert exc.value.reason == "provider_budget_exhausted"
    assert exc.value.detail["reconciled"] == 7
    # Control: identical chunk without reconciliation fits under the same limit.
    control_dir = tmp_path / "artifacts_control"
    control_dir.mkdir()
    result = drv.cmd_run_chunk(
        "d3q_p2_fake_control",
        ["slot_r1_small_p06"],
        ledger_path=ledger_path,
        artifacts_dir=control_dir,
        launcher=_make_fake_launcher([]),
    )
    assert result["status"] == "PASS"


def test_reconciliation_evidence_mismatch(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    _write_reconciliation(artifacts_dir)
    incident_result = tmp_path / "fake_incident" / "D3Q_SLOT_LAUNCHER_RESULT.json"
    incident_result.write_text("tampered", encoding="utf-8")
    with pytest.raises(drv.Phase2Error) as exc:
        drv._load_reconciliation(artifacts_dir)
    assert exc.value.reason == "reconciliation_evidence_mismatch"


def test_reconciliation_provider_mismatch(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    _write_reconciliation(artifacts_dir, provider_consumed={"ollama": 99})
    with pytest.raises(drv.Phase2Error) as exc:
        drv._load_reconciliation(artifacts_dir)
    assert exc.value.reason == "reconciliation_provider_mismatch"


def test_status_includes_reconciliation(seeded, tmp_path):
    ledger_path, _seed = seeded
    art = tmp_path / "art"
    art.mkdir()
    shutil.copy(ledger_path, art / "ledger.jsonl")
    _write_reconciliation(art)
    status = drv.cmd_status(art / "ledger.jsonl")
    assert status["provider_counts"]["ollama"] == 8
    assert status["provider_counts_ledger_only"]["ollama"] == 1
    assert len(status["reconciled_slots"]) == 5
    assert status["remaining_count"] == 72 - 2 - 5
    assert status["remaining_next"][0] == "slot_r1_small_p06"
