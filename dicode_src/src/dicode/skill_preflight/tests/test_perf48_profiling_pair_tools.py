import importlib.util
import json
from pathlib import Path

import pytest
import numpy as np
import networkx as nx

PERF = Path(__file__).parents[4] / "experiments" / "performance"


def load(name):
    spec = importlib.util.spec_from_file_location(name, PERF / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _manifest_spec(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = {}
    config = {}
    for label in ("e0", "perf48"):
        path = tmp_path / f"{label}.py"; path.write_text(f"# {label}\n")
        source[label] = str(path)
    for label in ("e0", "perf48_off", "perf48_profile"):
        path = tmp_path / f"{label}.yaml"; path.write_text(f"arm: {label}\n")
        config[label] = str(path)
    stages = {}
    for index, stage in enumerate(("early", "mid", "late"), 1):
        graph = tmp_path / f"{stage}.graphml"
        g = nx.DiGraph(); g.add_node("task_1", code="class Env: pass", description=stage)
        nx.write_graphml(g, graph)
        checkpoint = tmp_path / f"checkpoint_{index}"; checkpoint.mkdir(); (checkpoint / "_CHECKPOINT_METADATA").write_text("ok")
        cond = tmp_path / f"{stage}.npy"; np.save(cond, np.zeros((2, 67), dtype=np.float32))
        stages[stage] = {"graph": str(graph), "checkpoint": str(checkpoint), "task_ids": ["task_1"], "global_step": index, "initial_env_steps": index * 1024 * 128, "archive_reconstruction_limit": "all", "conditioning_path": str(cond)}
    return {"base_dir": str(tmp_path), "budget": {"timesteps": 2_000_000_000, "num_envs": 1024, "num_steps": 128, "updates": 100}, "conditioning_type": "one_hot", "source": source, "config": config, "stages": stages}


def test_manifest_contract_and_tamper(tmp_path):
    m = load("perf48_replay_manifest")
    assert m.ARMS == ("E0", "P0_OFF", "P0_PROFILE")
    assert m.BUDGET == {"timesteps": 2_000_000_000, "num_envs": 1024, "num_steps": 128, "updates": 100}
    manifest = m.build_manifest(_manifest_spec(tmp_path))
    assert [stage["orders"] for stage in manifest["stages"]] == [[ ["P0_OFF", "P0_PROFILE"], ["P0_PROFILE", "P0_OFF"] ]] * 3
    assert manifest["source_gate"] == {"stage": "early", "repeat": 0, "order": ["E0", "P0_OFF"]}
    out = tmp_path / "manifest.json"; m.write_manifest(manifest, out); assert m.load_manifest(out)["pair_count"] == 6
    for mutate in ("graph", "checkpoint", "conditioning", "source", "config", "manifest"):
        spec = _manifest_spec(tmp_path / mutate); built = m.build_manifest(spec); target = tmp_path / mutate / "manifest.json"; m.write_manifest(built, target)
        if mutate == "graph": Path(spec["stages"]["early"]["graph"]).write_text("tamper")
        elif mutate == "checkpoint": (Path(spec["stages"]["early"]["checkpoint"]) / "extra").write_text("tamper")
        elif mutate == "conditioning": np.save(spec["stages"]["early"]["conditioning_path"], np.ones((2, 67), dtype=np.float32))
        elif mutate == "source": Path(spec["source"]["e0"]).write_text("tamper")
        elif mutate == "config": Path(spec["config"]["e0"]).write_text("tamper")
        else:
            data = json.loads(target.read_text()); data["manifest_sha256"] = "bad"; target.write_text(json.dumps(data))
        with pytest.raises(ValueError): m.load_manifest(target)


def test_pair_aggregate_boundaries_and_conclusion():
    p = load("perf48_pair_benchmark")
    base = {"measured_session_wall_s": 100, "env_steps": 100, "gpu_peak_memory_mib": 100, "gpu_min_free_mib": 5000,
            "checkpoint_loadable": True, "compact_scoring_payload": False}
    pairs = [{"off": {**base}, "profile": {**base, "train_wall_s": 100}} for _ in range(6)]
    out = p.aggregate(pairs)
    assert out["conclusion"] == "P0_PROFILING_PASS"
    for pair in pairs:
        pair["profile"]["measured_session_wall_s"] = 102
    assert p.aggregate(pairs)["conclusion"] == "REJECTED_PROFILING_OVERHEAD"
    for pair in pairs:
        pair["profile"]["measured_session_wall_s"] = 100.99
    assert p.aggregate(pairs)["conclusion"] == "P0_PROFILING_PASS"
    for pair in pairs:
        pair["profile"]["measured_session_wall_s"] = 101.0
    assert p.aggregate(pairs)["conclusion"] == "REJECTED_PROFILING_OVERHEAD"


def test_aggregate_memory_and_runtime_markers_reject():
    p = load("perf48_pair_benchmark")
    base = {"measured_session_wall_s": 100, "env_steps": 100, "gpu_peak_memory_mib": 100, "gpu_min_free_mib": 5000, "checkpoint_loadable": True}
    pairs = [{"off": {**base}, "profile": {**base}} for _ in range(6)]
    pairs[0]["profile"]["gpu_peak_memory_mib"] = 700
    assert p.aggregate(pairs)["conclusion"] == "REJECTED_RUNTIME_FAILURE"
    pairs[0]["profile"]["gpu_peak_memory_mib"] = 613
    assert p.aggregate(pairs)["conclusion"] == "REJECTED_RUNTIME_FAILURE"
    pairs[0]["profile"]["gpu_peak_memory_mib"] = 100; pairs[0]["profile"]["runtime_failure"] = True
    assert p.aggregate(pairs)["conclusion"] == "REJECTED_RUNTIME_FAILURE"
    pairs[0]["profile"].pop("runtime_failure"); pairs[0]["profile"]["gpu_min_free_mib"] = 4095
    assert p.aggregate(pairs)["conclusion"] == "REJECTED_RUNTIME_FAILURE"


def test_fatal_and_gpu_external_fail_closed():
    p = load("perf48_pair_benchmark")
    assert p.FATAL_RE.search("CUDA Xid 79")
    assert not p.FATAL_RE.search("/tmp/checkpoint.txt")
    rows, bad = p.classify_gpu_apps("1,trainer,0,GPU-1", 999999, "GPU-1")
    assert bad
    import os
    rows, bad = p.classify_gpu_apps(f"{os.getpid()},trainer,0,GPU-1", os.getpid(), "GPU-1")
    assert rows and rows[0]["classification"] in {"owned_descendant", "stale_transient"}


def test_supervisor_tree_stop_is_owned_only(monkeypatch):
    p = load("perf48_pair_benchmark")
    monkeypatch.setattr(p, "descendants", lambda pid: [pid, pid + 1])
    calls = []; monkeypatch.setattr(p.os, "kill", lambda pid, sig: calls.append((pid, sig)))
    monkeypatch.setattr(p.subprocess, "check_output", lambda *args, **kwargs: "")
    owned = p.stop_owned(10, term_timeout=0)
    assert set(owned) == {10, 11} and all(pid in {10, 11} for pid, _ in calls)


def test_compare_semantic_and_checkpoint_gates():
    p = load("perf48_pair_benchmark")
    fields = {name: "same" for name in p.SEMANTIC_FIELDS}; fields.update({"gpu_uuid": "GPU-1", "classification": "TRAINING_KERNEL_BENCHMARK", "llm_api_calls": 0, "checkpoint_loadable": True, "compact_scoring_payload": False, "input_rng_sha256": "same", "train_rng_sha256": "same", "rng_sha256_after": "same"})
    assert p.compare_pair(fields, dict(fields)) == "SEMANTIC_PASS"
    for name in p.SEMANTIC_FIELDS:
        bad = dict(fields); bad[name] = "different"
        assert p.compare_pair(fields, bad) == "REJECTED_SEMANTIC_MISMATCH"
    bad = dict(fields); bad["checkpoint_loadable"] = False
    assert p.compare_pair(fields, bad) == "REJECTED_RUNTIME_FAILURE"


def test_supervisor_conclusion_contract_and_atomic(tmp_path):
    s = load("perf48_supervisor")
    s.atomic_json(tmp_path / "x.json", {"conclusion": "P0_PROFILING_PASS"})
    assert json.loads((tmp_path / "x.json").read_text())["conclusion"] == "P0_PROFILING_PASS"
    for conclusion in ("P0_PROFILING_PASS", "REJECTED_PROFILING_OVERHEAD", "REJECTED_RUNTIME_FAILURE", "REJECTED_SEMANTIC_MISMATCH"):
        assert s.validate_conclusion(conclusion)
    assert not s.validate_conclusion("REJECTED_NO_SPEEDUP")


@pytest.mark.parametrize("field", ["event_count", "jsonl", "events_csv_sha256", "critical_path_sha256", "enabled"])
def test_validate_rejects_disabled_artifacts(field):
    p = load("perf48_pair_benchmark")
    profiling = {"enabled": False, "event_count": 0, "jsonl": None, "events_csv_sha256": None, "critical_path_sha256": None}; profiling[field] = True if field in {"enabled", "jsonl", "events_csv_sha256", "critical_path_sha256"} else 1
    doc = {"manifest_sha256": "m", "stage": "early", "repeat": 0, "arm": "P0_OFF", "classification": "TRAINING_KERNEL_BENCHMARK", "llm_api_calls": 0, "gpu_uuid": "GPU-1", "profiling": profiling, "runtime_source_evidence": {"verified": True, "paths": {k: "p" for k in ("run_training_session", "calculate_scores_from_snapshot", "wrappers_cl", "TaskArchive", "load_tasks_from_env_codes", "_load_agent_state", "_calculate_task_distribution", "_create_achievement_masks")}, "hashes": {k: "h" for k in ("run_training_session", "calculate_scores_from_snapshot", "wrappers_cl", "TaskArchive", "load_tasks_from_env_codes", "_load_agent_state", "_calculate_task_distribution", "_create_achievement_masks")}}, "checkpoint_exists": True, "checkpoint_loadable": True, "checkpoint_path": __file__}
    with pytest.raises(RuntimeError):
        p.validate_result(doc, manifest_sha256="m", stage="early", repeat=0, arm="P0_OFF", gpu_uuid="GPU-1")


def test_validate_profile_requires_schema_phases_and_hashes():
    p = load("perf48_pair_benchmark")
    names = ("run_training_session", "calculate_scores_from_snapshot", "wrappers_cl", "TaskArchive", "load_tasks_from_env_codes", "_load_agent_state", "_calculate_task_distribution", "_create_achievement_masks")
    evidence = {"verified": True, "paths": {k: "p" for k in names}, "hashes": {k: "h" for k in names}}
    doc = {"manifest_sha256": "m", "stage": "early", "repeat": 0, "arm": "P0_PROFILE", "classification": "TRAINING_KERNEL_BENCHMARK", "llm_api_calls": 0, "gpu_uuid": "GPU-1", "profiling": {"enabled": True, "event_count": 1, "events_csv_sha256": "x", "critical_path_sha256": "y", "events": []}, "runtime_source_evidence": evidence, "checkpoint_exists": True, "checkpoint_loadable": True, "checkpoint_path": __file__}
    with pytest.raises(RuntimeError):
        p.validate_result(doc, manifest_sha256="m", stage="early", repeat=0, arm="P0_PROFILE", gpu_uuid="GPU-1")


def test_harness_arm_cli_contains_three_arms():
    h = load("perf48_training_kernel_harness")
    assert h._arm_values({"performance": {}, "training": {"compact_scoring_payload": False}})[0] is False


def test_runtime_source_evidence_requires_all_bindings(tmp_path, monkeypatch):
    h = load("perf48_training_kernel_harness")
    names = ("run_training_session", "calculate_scores_from_snapshot", "wrappers_cl", "TaskArchive", "load_tasks_from_env_codes", "_load_agent_state", "_calculate_task_distribution", "_create_achievement_masks")
    rel = {"run_training_session": "src/dicode/ppo_tr.py", "calculate_scores_from_snapshot": "src/dicode/scoring.py", "wrappers_cl": "src/dicode/wrappers_cl.py", "TaskArchive": "src/dicode/dreaming/gen_manager.py", "load_tasks_from_env_codes": "src/dicode/task_utils.py", "_load_agent_state": "src/dicode/setup.py", "_calculate_task_distribution": "src/dicode/training.py", "_create_achievement_masks": "src/dicode/training.py"}
    files = {}; rt = {}
    for name in names:
        path = tmp_path / f"{name}.py"; path.write_text(name); files[name] = path; rt[name] = lambda: None
    files["_create_achievement_masks"] = files["_calculate_task_distribution"]
    entries = {}
    for name in names:
        entries.setdefault(f"perf48/{rel[name]}", {"path": str(files[name]), "sha256": h.sha256_file(files[name])})
    loaded = {"source_config": {"source": entries}}
    monkeypatch.setattr(h.inspect, "getsourcefile", lambda obj: str(files[next(k for k, v in rt.items() if v is obj)]))
    evidence = h._runtime_source_evidence(rt, loaded, "P0_PROFILE")
    assert evidence["verified"] and set(evidence["paths"]) == set(names)
    for missing in names:
        bad = dict(rt); bad.pop(missing)
        with pytest.raises(RuntimeError): h._runtime_source_evidence(bad, loaded, "P0_PROFILE")
    wrong = {**loaded}; wrong["source_config"] = {"source": {**entries, "perf48/src/dicode/scoring.py": {"path": str(files["calculate_scores_from_snapshot"]), "sha256": "bad"}}}
    with pytest.raises(RuntimeError): h._runtime_source_evidence(rt, wrong, "P0_PROFILE")
