"""Local tests for the D3Q preflight orchestrator (prepare) and remote driver
helpers (archive injection, conditioning table)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import d3q_preflight_orchestrator as orch  # noqa: E402
import d3q_preflight_remote as remote  # noqa: E402


def _make_slot(artifact_dir, slot_id, final_valid=True, code=b"class Env:\n    pass\n"):
    slot_dir = Path(artifact_dir) / "slots" / slot_id
    slot_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "final_valid": final_valid,
        "attempts": 1,
        "attempts_detail": [{"kind": "initial"}],
        "ledger_counts": {"slot": 1, "provider": 1},
    }
    (slot_dir / f"{slot_id}.result.json").write_text(json.dumps(result), encoding="utf-8")
    if final_valid:
        (slot_dir / "final_code.py").write_bytes(code)
    return slot_dir


def test_prepare_aggregates_dedup_and_aliases(tmp_path, monkeypatch):
    art = tmp_path / "art"
    code_a = b"class Env:\n    A = 1\n"
    _make_slot(art, "slot_r1_small_p00", True, code_a)
    _make_slot(art, "slot_r1_small_p01", True, code_a)  # duplicate code -> alias
    _make_slot(art, "slot_r1_large_p00", True, b"class Env:\n    B = 1\n")
    _make_slot(art, "slot_r1_large_p01", False)
    monkeypatch.setattr(
        orch,
        "all_slots_ordered",
        lambda: [
            "slot_r1_small_p00", "slot_r1_small_p01",
            "slot_r1_large_p00", "slot_r1_large_p01",
        ],
    )
    staging = tmp_path / "staging"
    plan = orch.cmd_prepare([art], staging, tmp_path / "no_reconciliation")
    arms = {a["arm_id"]: a for a in plan["arms"]}
    assert arms["small_r1"]["candidate_count"] == 1
    assert arms["small_r1"]["final_valid_slots"] == 2
    assert arms["large_r1"]["candidate_count"] == 1
    assert arms["large_r1"]["final_invalid_slots"] == 1
    assert arms["small_r2"]["candidate_count"] == 0
    meta = json.loads((staging / "arms" / "small_r1" / "ARM_CANDIDATES.json").read_text(encoding="utf-8"))
    assert len(meta["candidates"]) == 1
    cand = meta["candidates"][0]
    assert cand["id"] == "slot_r1_small_p00"
    assert cand["aliased_slots"] == ["slot_r1_small_p01"]
    assert (staging / "arms" / "small_r1" / "candidates" / "slot_r1_small_p00.py").read_bytes() == code_a
    assert plan["execution_order"] == [a["arm_id"] for a in plan["arms"]]


def test_prepare_missing_slot_fails_closed(tmp_path, monkeypatch):
    art = tmp_path / "art"
    _make_slot(art, "slot_r1_small_p00")
    monkeypatch.setattr(orch, "all_slots_ordered", lambda: ["slot_r1_small_p00", "slot_r1_small_p01"])
    with pytest.raises(orch.OrchestratorError) as exc:
        orch.cmd_prepare([art], tmp_path / "staging", tmp_path / "no_reconciliation")
    assert exc.value.reason == "slot_artifact_missing"


def test_prepare_reconciled_loss_not_required(tmp_path, monkeypatch):
    art = tmp_path / "art"
    _make_slot(art, "slot_r1_small_p00")
    monkeypatch.setattr(orch, "all_slots_ordered", lambda: ["slot_r1_small_p00", "slot_r1_small_p05"])
    root = tmp_path / "artifacts_root"
    root.mkdir()
    recon = {
        "classification": "D3Q_BUDGET_RECONCILIATION",
        "schema_version": 1,
        "provider_consumed": {"ollama": 3},
        "slot_consumed": {"slot_r1_small_p05": 3},
    }
    (root / orch.RECONCILIATION_FILENAME).write_text(json.dumps(recon), encoding="utf-8")
    plan = orch.cmd_prepare([art], tmp_path / "staging", root)
    small = next(a for a in plan["arms"] if a["arm_id"] == "small_r1")
    assert small["lost_slots"] == 1
    assert small["candidate_count"] == 1
    meta = json.loads((tmp_path / "staging" / "arms" / "small_r1" / "ARM_CANDIDATES.json").read_text(encoding="utf-8"))
    assert meta["slots_lost_reconciled"] == ["slot_r1_small_p05"]


def test_prepare_final_code_missing_fails_closed(tmp_path, monkeypatch):
    art = tmp_path / "art"
    slot_dir = _make_slot(art, "slot_r1_small_p00")
    (slot_dir / "final_code.py").unlink()
    monkeypatch.setattr(orch, "all_slots_ordered", lambda: ["slot_r1_small_p00"])
    with pytest.raises(orch.OrchestratorError) as exc:
        orch.cmd_prepare([art], tmp_path / "staging", tmp_path / "no_reconciliation")
    assert exc.value.reason == "final_code_missing"


def test_remote_inject_candidates_roundtrip(tmp_path):
    nx = pytest.importorskip("networkx")
    graph = nx.DiGraph()
    graph.add_node("task_1", code="class Env:\n    pass\n", status="seed")
    src = tmp_path / "src" / "task_graph.graphml"
    src.parent.mkdir()
    nx.write_graphml(graph, src)
    code_file = tmp_path / "cand.py"
    code_text = "class Env:\n    X = 42\n"
    code_file.write_text(code_text, encoding="utf-8")
    out_dir = tmp_path / "archive_copy"
    remote.inject_candidates(src, out_dir, [{"id": "slot_r1_small_p00", "path": str(code_file)}])
    loaded = nx.read_graphml(out_dir / "task_graph.graphml")
    assert loaded.nodes["slot_r1_small_p00"]["code"] == code_text
    assert loaded.nodes["slot_r1_small_p00"]["is_active"] == "false"
    assert loaded.nodes["task_1"]["status"] == "seed"


def test_remote_conditioning_shape(tmp_path):
    np = pytest.importorskip("numpy")
    path = remote.write_conditioning(tmp_path, 4)
    table = np.load(path, allow_pickle=False)
    assert table.shape == (5, remote.CONDITIONING_DIM)
    assert table.dtype == np.float32
    assert (table == 0).all()
