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
    remote.inject_candidates(src, out_dir, [{"id": "slot_r1_small_p00", "path": str(code_file)}], tmp_path)
    loaded = nx.read_graphml(out_dir / "task_graph.graphml")
    assert loaded.nodes["slot_r1_small_p00"]["code"] == code_text
    assert loaded.nodes["slot_r1_small_p00"]["is_active"] == "false"
    assert loaded.nodes["task_1"]["status"] == "seed"


def test_remote_inject_candidates_relative_path(tmp_path):
    nx = pytest.importorskip("networkx")
    graph = nx.DiGraph()
    graph.add_node("task_1", code="class Env:\n    pass\n", status="seed")
    src = tmp_path / "src" / "task_graph.graphml"
    src.parent.mkdir()
    nx.write_graphml(graph, src)
    arm_dir = tmp_path / "arm"
    (arm_dir / "candidates").mkdir(parents=True)
    code_text = "class Env:\n    Y = 7\n"
    (arm_dir / "candidates" / "slot_r1_small_p00.py").write_text(code_text, encoding="utf-8")
    out_dir = arm_dir / "archive_copy"
    remote.inject_candidates(
        src, out_dir,
        [{"id": "slot_r1_small_p00", "path": "candidates/slot_r1_small_p00.py"}],
        arm_dir,
    )
    loaded = nx.read_graphml(out_dir / "task_graph.graphml")
    assert loaded.nodes["slot_r1_small_p00"]["code"] == code_text


def test_remote_conditioning_shape(tmp_path):
    np = pytest.importorskip("numpy")
    path = remote.write_conditioning(tmp_path, 4)
    table = np.load(path, allow_pickle=False)
    assert table.shape == (5, remote.CONDITIONING_DIM)
    assert table.dtype == np.float32
    assert (table == 0).all()


def test_provenance_constants_match_frozen_b1_snapshot():
    # Regression guard for the 20260816T063130Z failure: replay must run
    # against the frozen B1 dicode_src snapshot (source_commit 4d1f54f),
    # not the baseline 91a75e5 worktree (which lacks preflight_route.py).
    import d3q_preflight_orchestrator as orch

    frozen_src = "/home/oseasy/e2_data_disk2/skill_preflight_runs/perf48_b1r2_gpu2_20260813T032611Z/dicode_src/src"
    assert orch.MASON_SRC == frozen_src
    assert remote.SOURCE_COMMIT == "4d1f54fd32223ec0d51b38a64a3e6902d334c3c3"
    labels = dict(remote.SOURCE_FILES)
    assert labels["preflight_route.py"] == "skill_preflight/preflight_route.py"
    mapping = {name: f"{orch.MASON_SRC}/dicode/{rel}" for name, rel in remote.SOURCE_FILES}
    assert mapping["preflight_route.py"].endswith(
        "perf48_b1r2_gpu2_20260813T032611Z/dicode_src/src/dicode/skill_preflight/preflight_route.py"
    )


def test_classify_poll_states():
    assert orch._classify_poll("DONE rc=0") == ("DONE", 0)
    assert orch._classify_poll("DONE rc=3\n") == ("DONE", 3)
    assert orch._classify_poll("DONE rc=2") == ("DONE", 2)
    assert orch._classify_poll("RUNNING\n") == ("RUNNING", None)
    assert orch._classify_poll("DEAD") == ("DEAD", None)
    assert orch._classify_poll("")[0] == "UNKNOWN"
    assert orch._classify_poll("weird output")[0] == "UNKNOWN"
    assert orch._classify_poll("DONE rc=")[0] == "UNKNOWN"


def test_poll_probe_self_match_guard():
    import re

    exec_root = "/tmp/d3q_preflight_20260816T000000Z"
    cmd = orch._poll_probe_cmd(exec_root)
    pattern = f"d3q_preflight_remote[.]py --exec-root {exec_root}"
    # a real driver command line must match...
    real = f"/home/x/python d3q_preflight_remote.py --exec-root {exec_root} --mason-src /y"
    assert re.search(pattern, real) is not None
    # ...but the probe's own command line (bracketed) must not.
    assert re.search(pattern, cmd) is None


# ---------------------------------------------------------------------------
# incident-05 recovery
# ---------------------------------------------------------------------------


def _mk_recover_arm(root, arm_id, n_cands, execute_s):
    import subprocess as _sp

    arm = root / "arms" / arm_id
    (arm / "run").mkdir(parents=True)
    (arm / "ARM_CANDIDATES.json").write_text(
        json.dumps({"arm_id": arm_id, "candidates": [{"id": f"c{i}", "path": f"c{i}.py"} for i in range(n_cands)]}),
        encoding="utf-8",
    )
    (arm / "spec.json").write_text("{}", encoding="utf-8")
    (arm / "manifest.json").write_text("{}", encoding="utf-8")
    (arm / "run" / "RESULT.json").write_text(json.dumps({"accepted_ids": ["c0"], "rejected_ids": []}), encoding="utf-8")
    (arm / "run" / "replay_summary.json").write_text(json.dumps({"session_wall_s": 100.0}), encoding="utf-8")
    (arm / "run" / "critical_path.json").write_text(
        json.dumps({"critical_path": [{"phase": "preflight_eval_execute", "duration_s": execute_s}]}), encoding="utf-8"
    )
    return arm


def _mk_recover_dir(tmp_path, executes=(100.0, 105.0)):
    root = tmp_path / "d3q_preflight_20260816T000000Z"
    root.mkdir()
    _mk_recover_arm(root, "arm_a", 10, executes[0])
    _mk_recover_arm(root, "arm_b", 10, executes[1])
    (root / "D3Q_PREFLIGHT_REMOTE_SUMMARY.json").write_text(
        json.dumps({"status": "PASS", "arms": [
            {"arm_id": "arm_a", "status": "PASS"}, {"arm_id": "arm_b", "status": "PASS"}]}),
        encoding="utf-8",
    )
    (root / "remote_run_rc.txt").write_text("0\n", encoding="utf-8")
    (root / "driver.rc").write_text("0\n", encoding="utf-8")
    return root


def _patch_remote_gates(monkeypatch):
    monkeypatch.setattr(orch, "_gpu_gate_remote", lambda t, k: {"gpu2": {"uuid": "GPU-x"}, "external": []})
    monkeypatch.setattr(orch, "_ollama_gate_remote", lambda t, k: {"qwen_digest": "9ec8897f747e"})


def test_recovery_happy_path(tmp_path, monkeypatch):
    _patch_remote_gates(monkeypatch)
    root = _mk_recover_dir(tmp_path)
    result = orch.cmd_recover_completed_run(root, "gpu2_external_app", None, "test detail", "t", "k")
    assert result["status"] == "PASS"
    assert result["recovery"]["reason"] == "gpu2_external_app"
    assert (root / "D3Q_PREFLIGHT_ORCHESTRATOR_RESULT.json").is_file()
    assert (root / "D3Q_PREFLIGHT_RECOVERY.json").is_file()


def test_recovery_reason_whitelist(tmp_path, monkeypatch):
    _patch_remote_gates(monkeypatch)
    root = _mk_recover_dir(tmp_path)
    with pytest.raises(orch.OrchestratorError):
        orch.cmd_recover_completed_run(root, "something_else", None, "", "t", "k")


def test_recovery_rejects_non_pass_summary(tmp_path, monkeypatch):
    _patch_remote_gates(monkeypatch)
    root = _mk_recover_dir(tmp_path)
    summary = json.loads((root / "D3Q_PREFLIGHT_REMOTE_SUMMARY.json").read_text(encoding="utf-8"))
    summary["status"] = "FAILED"
    (root / "D3Q_PREFLIGHT_REMOTE_SUMMARY.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(orch.OrchestratorError):
        orch.cmd_recover_completed_run(root, "gpu2_external_app", None, "", "t", "k")


def test_recovery_rejects_nonzero_rc(tmp_path, monkeypatch):
    _patch_remote_gates(monkeypatch)
    root = _mk_recover_dir(tmp_path)
    (root / "driver.rc").write_text("3\n", encoding="utf-8")
    with pytest.raises(orch.OrchestratorError):
        orch.cmd_recover_completed_run(root, "gpu2_external_app", None, "", "t", "k")


def test_recovery_rejects_existing_result(tmp_path, monkeypatch):
    _patch_remote_gates(monkeypatch)
    root = _mk_recover_dir(tmp_path)
    (root / "D3Q_PREFLIGHT_ORCHESTRATOR_RESULT.json").write_text("{}", encoding="utf-8")
    with pytest.raises(orch.OrchestratorError):
        orch.cmd_recover_completed_run(root, "gpu2_external_app", None, "", "t", "k")


def test_recovery_interference_ratio_fail_closed(tmp_path, monkeypatch):
    _patch_remote_gates(monkeypatch)
    root = _mk_recover_dir(tmp_path, executes=(100.0, 400.0))  # arm_b 4x slower per candidate
    with pytest.raises(orch.OrchestratorError):
        orch.cmd_recover_completed_run(root, "gpu2_external_app", None, "", "t", "k")


def test_recovery_rejects_live_external_pid(tmp_path, monkeypatch):
    import subprocess as _sp

    _patch_remote_gates(monkeypatch)
    root = _mk_recover_dir(tmp_path)
    monkeypatch.setattr(
        orch, "_run_local",
        lambda argv, timeout=600: _sp.CompletedProcess(argv, 0, stdout="ALIVE\n", stderr=""),
    )
    with pytest.raises(orch.OrchestratorError):
        orch.cmd_recover_completed_run(root, "gpu2_external_app", 12345, "", "t", "k")
