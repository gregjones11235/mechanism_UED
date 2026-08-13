"""Local tests for the BC combo experiment tooling (B2 + C).

These are pure-logic / CPU-only tests: config overlay generation, manifest
build/hash/order/RNG contract, pair comparison gates, mechanism verification,
and supervisor conclusion validation. The JAX/GPU-facing behavior is validated
on the server (real-JAX tests + the combo harness itself).
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import numpy as np
import networkx as nx

PERF = Path(__file__).parents[4] / "experiments" / "performance"


def load(name):
    spec = importlib.util.spec_from_file_location(name, PERF / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


# ---- synthetic base config mirroring configs/perf48_off.yaml structure ----
BASE_CONFIG = {
    "dicode_manager": {"score_function": "learnability", "max_updates_per_session": 100},
    "training": {"num_envs": 1024, "num_steps": 128, "total_timesteps": 2000000000,
                 "conditioning_type": "one_hot", "condition_on_task": True,
                 "compact_scoring_payload": False},
    "performance": {"eval_compile_cache": False, "train_compile_cache": False,
                    "compiled_cache_max_entries": 8, "embedding_cache": False,
                    "validation_cache": False},
    "runtime_profiling": {"enabled": False, "output_jsonl": "runtime_analysis/off_events.jsonl"},
}

SOURCE_FILES = (
    "src/dicode/dreaming/gen_manager.py",
    "src/dicode/task_utils.py",
    "src/dicode/evaluation/online_evaluation.py",
    "src/dicode/ppo_tr.py",
    "src/dicode/scoring.py",
    "src/dicode/setup.py",
    "src/dicode/skill_preflight/preflight.py",
    "src/dicode/skill_preflight/preflight_route.py",
    "src/dicode/skill_preflight/reuse_loaded_tasks.py",
    "src/dicode/craftax_evaluation.py",
    "src/dicode/wrappers_cl.py",
)


def _manifest_spec(tmp_path):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = {}
    for rel in SOURCE_FILES:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {rel}\n")
        source[rel] = str(path)
    base_cfg = tmp_path / "configs" / "perf48_off.yaml"
    base_cfg.parent.mkdir(parents=True, exist_ok=True)
    base_cfg.write_text("arm: base\nperformance:\n  eval_compile_cache: false\n")
    config = {"base": str(base_cfg), "perf48_off": str(base_cfg), "perf48_on": str(base_cfg)}
    stages = {}
    for index, stage in enumerate(("early", "mid", "late"), 1):
        graph = tmp_path / f"{stage}.graphml"
        g = nx.DiGraph()
        g.add_node("task_1", code="class Env: pass", description=stage)
        nx.write_graphml(g, graph)
        checkpoint = tmp_path / f"checkpoint_{index}"
        checkpoint.mkdir(exist_ok=True)
        (checkpoint / "_CHECKPOINT_METADATA").write_text("ok")
        cond = tmp_path / f"{stage}.npy"
        np.save(cond, np.zeros((2, 67), dtype=np.float32))
        stages[stage] = {
            "graph": str(graph), "checkpoint": str(checkpoint),
            "task_ids": ["task_1"], "global_step": index,
            "initial_env_steps": index * 1024 * 128,
            "archive_reconstruction_limit": "all",
            "conditioning_path": str(cond),
        }
    return {
        "base_dir": str(tmp_path),
        "budget": {"timesteps": 2_000_000_000, "num_envs": 1024, "num_steps": 128, "updates": 100},
        "conditioning_type": "one_hot",
        "source": source, "config": config,
        "stages": stages,
    }


# ---- config overlay ----------------------------------------------------------
def test_overlay_diff_only_b2_c_switches():
    cfg = load("perf48_combo_config")
    off = cfg.build_overlay_dict(BASE_CONFIG, arm="BC_OFF")
    on = cfg.build_overlay_dict(BASE_CONFIG, arm="BC_ON")
    diff = cfg.normalized_diff(off, on)
    assert diff == ["performance.eval_compile_cache", "performance.preflight_reuse_loaded_tasks"]
    verdict = cfg.verify_overlay_pair(off, on)
    assert verdict["valid"]
    assert verdict["unexpected_diff_paths"] == []
    assert off["performance"]["preflight_reuse_loaded_tasks"] is False
    assert on["performance"]["preflight_reuse_loaded_tasks"] is True
    assert off["performance"]["eval_compile_cache"] is False
    assert on["performance"]["eval_compile_cache"] is True


def test_overlay_forces_b3_train_embedding_validation_false():
    cfg = load("perf48_combo_config")
    base = {**BASE_CONFIG, "performance": {
        **BASE_CONFIG["performance"],
        "compact_preflight_payload": True, "train_compile_cache": True,
        "embedding_cache": True, "validation_cache": True,
    }}
    for arm in ("BC_OFF", "BC_ON"):
        data = cfg.build_overlay_dict(base, arm=arm)
        for key in cfg.FORCED_FALSE:
            assert data["performance"][key] is False, f"{arm} must force {key}=false"
        assert data["runtime_profiling"]["enabled"] is True


def test_overlay_both_arms_profile_enabled_and_diff_detected():
    cfg = load("perf48_combo_config")
    off = cfg.build_overlay_dict(BASE_CONFIG, arm="BC_OFF")
    on = cfg.build_overlay_dict(BASE_CONFIG, arm="BC_ON")
    assert off["runtime_profiling"]["enabled"] is True
    assert on["runtime_profiling"]["enabled"] is True
    # introducing a third difference must trip the gate
    bad = cfg.build_overlay_dict(BASE_CONFIG, arm="BC_ON")
    bad["performance"]["train_compile_cache"] = True
    with pytest.raises(ValueError):
        cfg.verify_overlay_pair(off, bad)
    bad2 = cfg.build_overlay_dict(BASE_CONFIG, arm="BC_ON")
    bad2["runtime_profiling"]["enabled"] = False
    with pytest.raises(ValueError):
        cfg.verify_overlay_pair(off, bad2)


def test_overlay_write_yaml_roundtrip(tmp_path):
    import yaml
    cfg = load("perf48_combo_config")
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(BASE_CONFIG), encoding="utf-8")
    off_path = tmp_path / "config_off.yaml"
    on_path = tmp_path / "config_on.yaml"
    cfg.write_overlay_yaml(base, arm="BC_OFF", out_path=off_path)
    cfg.write_overlay_yaml(base, arm="BC_ON", out_path=on_path)
    off = cfg.load_overlay(off_path)
    on = cfg.load_overlay(on_path)
    verdict = cfg.verify_overlay_pair(off, on)
    assert verdict["valid"]
    assert set(verdict["diff_paths"]) == {"performance.eval_compile_cache",
                                          "performance.preflight_reuse_loaded_tasks"}
    # yaml round-trip preserved the unrelated keys
    assert off["performance"]["compiled_cache_max_entries"] == 8
    assert off["dicode_manager"]["score_function"] == "learnability"


def test_overlay_rejects_unknown_arm():
    cfg = load("perf48_combo_config")
    with pytest.raises(ValueError):
        cfg.build_overlay_dict(BASE_CONFIG, arm="BC_THIRD")


# ---- manifest -----------------------------------------------------------------
def test_manifest_contract_and_orders(tmp_path):
    m = load("perf48_combo_manifest")
    assert m.ARMS == ("BC_OFF", "BC_ON")
    assert m.STAGES == ("early", "mid", "late")
    manifest = m.build_combo_manifest(_manifest_spec(tmp_path))
    assert [stage["orders"] for stage in manifest["stages"]] == [[["BC_OFF", "BC_ON"], ["BC_ON", "BC_OFF"]]] * 3
    assert manifest["pair_count"] == 6
    assert manifest["stage_count"] == 3
    assert manifest["classification"] == "PERF48_COMBO_BENCHMARK"
    out = tmp_path / "manifest.json"
    m.write_manifest(manifest, out)
    loaded = m.load_manifest(out)
    assert loaded["pair_count"] == 6


def test_manifest_six_group_alternation(tmp_path):
    m = load("perf48_combo_manifest")
    manifest = m.build_combo_manifest(_manifest_spec(tmp_path))
    seq = []
    for stage in manifest["stages"]:
        for repeat in stage["repeats"]:
            seq.extend(repeat["order"])
    expected = ["BC_OFF", "BC_ON", "BC_ON", "BC_OFF"] * 3
    assert seq == expected
    # each stage/repeat has the frozen RNG
    for stage in manifest["stages"]:
        assert stage["rng"] == [m._u32_rng(stage["name"], 0), m._u32_rng(stage["name"], 1)]


def test_manifest_rng_matches_frozen_p0_values():
    m = load("perf48_combo_manifest")
    expected = {
        ("early", 0): [249215833, 2283659212],
        ("early", 1): [2573928946, 2637418465],
        ("mid", 0): [898125105, 1275040781],
        ("mid", 1): [1803701838, 866904108],
        ("late", 0): [97867067, 1233137050],
        ("late", 1): [3902662763, 299068710],
    }
    for (stage, repeat), value in expected.items():
        assert m._u32_rng(stage, repeat) == value, f"{stage} repeat {repeat} RNG mismatch"


def test_manifest_tamper_detection(tmp_path):
    m = load("perf48_combo_manifest")
    for mutate in ("graph", "checkpoint", "conditioning", "source", "config", "manifest", "candidate"):
        spec = _manifest_spec(tmp_path / mutate)
        built = m.build_combo_manifest(spec)
        target = tmp_path / mutate / "manifest.json"
        m.write_manifest(built, target)
        if mutate == "graph":
            Path(spec["stages"]["early"]["graph"]).write_text("tamper", encoding="utf-8")
        elif mutate == "checkpoint":
            (Path(spec["stages"]["early"]["checkpoint"]) / "extra").write_text("tamper")
        elif mutate == "conditioning":
            np.save(spec["stages"]["early"]["conditioning_path"], np.ones((2, 67), dtype=np.float32))
        elif mutate == "source":
            Path(spec["source"]["src/dicode/task_utils.py"]).write_text("tamper", encoding="utf-8")
        elif mutate == "config":
            Path(spec["config"]["base"]).write_text("tamper", encoding="utf-8")
        elif mutate == "candidate":
            code_path = Path(built["stages"][0]["candidate_codes"]["task_1"]["path"])
            code_path.write_text("tampered code", encoding="utf-8")
        else:
            data = json.loads(target.read_text())
            data["manifest_sha256"] = "bad"
            target.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            m.load_manifest(target)


# ---- benchmark pair gates ------------------------------------------------------
def test_compare_combo_pair_semantic_and_gates():
    b = load("perf48_combo_benchmark")
    fields = {name: "same" for name in b.SEMANTIC_FIELDS}
    fields.update({"gpu_uuid": "GPU-1", "classification": "PERF48_COMBO_BENCHMARK",
                   "llm_api_calls": 0, "checkpoint_loadable": True, "compact_scoring_payload": False})
    assert b.compare_combo_pair(fields, dict(fields)) == "SEMANTIC_PASS"
    for name in b.SEMANTIC_FIELDS:
        bad = dict(fields)
        bad[name] = "different"
        assert b.compare_combo_pair(fields, bad) == "REJECTED_SEMANTIC_MISMATCH"
    bad = dict(fields)
    bad["checkpoint_loadable"] = False
    assert b.compare_combo_pair(fields, bad) == "REJECTED_RUNTIME_FAILURE"


def test_verify_mechanisms_b2_c():
    b = load("perf48_combo_benchmark")
    off = {"preflight_task_reload_occurred": True, "preflight_task_reload_explicit_absent": False,
           "eval_compile_span_count": 0, "eval_first_cache_miss": False, "eval_cache_hit_count": 0}
    on = {"preflight_task_reload_occurred": False, "preflight_task_reload_explicit_absent": True,
          "eval_compile_span_count": 1, "eval_first_cache_miss": True, "eval_cache_hit_count": 1}
    mech = b.verify_mechanisms(off, on)
    assert mech["ok"]
    # C broken: on still compiles twice
    bad_on = dict(on)
    bad_on["eval_compile_span_count"] = 2
    assert not b.verify_mechanisms(off, bad_on)["ok"]
    # B2 broken: on still reloads
    bad_on2 = dict(on)
    bad_on2["preflight_task_reload_occurred"] = True
    assert not b.verify_mechanisms(off, bad_on2)["ok"]
    # B2 broken: off does not reload
    bad_off = dict(off)
    bad_off["preflight_task_reload_occurred"] = False
    assert not b.verify_mechanisms(bad_off, on)["ok"]


def test_validate_combo_result_contract(tmp_path):
    b = load("perf48_combo_benchmark")
    doc = {name: "same" for name in b.SEMANTIC_FIELDS}
    doc.update({name: 0 for name in b.PERF_FIELDS})
    doc.update({
        "classification": "PERF48_COMBO_BENCHMARK", "manifest_sha256": "m",
        "stage": "early", "repeat": 0, "arm": "BC_OFF", "llm_api_calls": 0,
        "gpu_uuid": "GPU-1", "source_commit": "abc",
        "checkpoint_loadable": True, "compact_scoring_payload": False,
        "runtime_source_evidence": {"verified": True},
        "profiling": {"enabled": True, "event_count": 1, "events_csv_sha256": "x",
                      "critical_path_sha256": "y"},
        "env_evidence": {"jax_version": "0.6.0"},
    })
    validated = b.validate_combo_result(doc, manifest_sha256="m", stage="early", repeat=0,
                                        arm="BC_OFF", gpu_uuid="GPU-1", source_commit="abc")
    assert validated["arm"] == "BC_OFF"
    bad = dict(doc)
    bad["compact_scoring_payload"] = True
    with pytest.raises(RuntimeError):
        b.validate_combo_result(bad, manifest_sha256="m", stage="early", repeat=0,
                                arm="BC_OFF", gpu_uuid="GPU-1", source_commit="abc")
    bad2 = dict(doc)
    del bad2["params_sha256_before"]
    with pytest.raises(RuntimeError):
        b.validate_combo_result(bad2, manifest_sha256="m", stage="early", repeat=0,
                                arm="BC_OFF", gpu_uuid="GPU-1", source_commit="abc")


def test_append_xla_flag_and_arm_env(tmp_path):
    b = load("perf48_combo_benchmark")
    assert b.append_xla_flag(None) == b._pair.DETERMINISTIC_FLAG
    import types
    import os
    src = tmp_path / "src"
    (src / "src").mkdir(parents=True)
    args = types.SimpleNamespace(gpu_uuid="GPU-1", deterministic_xla=True)
    env = b._arm_env(args, "BC_ON", str(src), tmp_path / "out")
    assert env["XLA_FLAGS"].count(b._pair.DETERMINISTIC_FLAG) == 1
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-1"
    assert env["PYTHONPATH"] == str(src / "src")


def test_parse_only_pairs():
    b = load("perf48_combo_benchmark")
    assert b._parse_only_pairs(None) is None
    assert b._parse_only_pairs("") is None
    assert b._parse_only_pairs("early:0") == {("early", 0)}
    assert b._parse_only_pairs("early:0,mid:1,late:0") == {("early", 0), ("mid", 1), ("late", 0)}
    # trailing comma is tolerated (empty tokens skipped)
    assert b._parse_only_pairs("early:0,") == {("early", 0)}
    for bad in ("early", "early:2", "foo:0"):
        try:
            b._parse_only_pairs(bad)
            raise AssertionError(f"expected ValueError for {bad!r}")
        except ValueError:
            pass


def test_reconstruct_archive_signature_matches_call_site():
    """Regression: _reconstruct_archive must accept exactly the graph path
    (a previous version required an unused 'config' arg, breaking the runtime
    binding -- caught by the GPU2 smoke fail-closed)."""
    import inspect
    h = load("perf48_combo_harness")
    sig = inspect.signature(h._reconstruct_archive)
    assert len(sig.parameters) == 1, f"got {sig}"
    # the runtime binding calls it with exactly one positional argument
    rt_names = set()
    spec = importlib.util.spec_from_file_location(
        "perf48_combo_harness", PERF / "perf48_combo_harness.py")
    src = Path(spec.origin).read_text(encoding="utf-8")
    assert 'rt["reconstruct_archive"](stage["graph"]["path"])' in src


def test_reconstruct_archive_real_jax(tmp_path):
    pytest.importorskip("jax")
    h = load("perf48_combo_harness")
    g = nx.DiGraph()
    g.add_node("task_1", code="class Env:\n    pass\n")
    graph = tmp_path / "g.graphml"
    nx.write_graphml(g, graph)
    archive = h._reconstruct_archive(str(graph))
    codes = archive.get_task_codes(["task_1"])
    assert "task_1" in codes and "class Env" in codes["task_1"]


def test_harness_verify_gpu(monkeypatch):
    import types
    h = load("perf48_combo_harness")
    fake_jax = types.SimpleNamespace(default_backend=lambda: "gpu", devices=lambda: [1])
    monkeypatch.setattr(h.os, "environ", {"CUDA_VISIBLE_DEVICES": "GPU-1"})
    # inject fake jax via the import inside _verify_gpu
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "jax":
            return fake_jax
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    args = types.SimpleNamespace(required_gpu_uuid="GPU-1")
    h._verify_gpu(args)  # must not raise
    args_bad = types.SimpleNamespace(required_gpu_uuid="GPU-2")
    try:
        h._verify_gpu(args_bad)
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass
    # CPU backend must fail when a GPU UUID is required
    fake_jax_cpu = types.SimpleNamespace(default_backend=lambda: "cpu", devices=lambda: [])
    fake_import_cpu = lambda name, *a, **k: fake_jax_cpu if name == "jax" else real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import_cpu)
    try:
        h._verify_gpu(args)
        raise AssertionError("expected RuntimeError for cpu backend")
    except RuntimeError:
        pass


# ---- supervisor ----------------------------------------------------------------
def test_supervisor_conclusion_contract_and_atomic(tmp_path):
    s = load("perf48_combo_supervisor")
    s.atomic_json(tmp_path / "x.json", {"conclusion": "COMBO_PASS"})
    assert json.loads((tmp_path / "x.json").read_text())["conclusion"] == "COMBO_PASS"
    for conclusion in ("COMBO_PASS", "REJECTED_SEMANTIC_MISMATCH", "REJECTED_RUNTIME_FAILURE", "REJECTED_MECHANISM"):
        assert s.validate_conclusion(conclusion)
    assert not s.validate_conclusion("COMBO_FAIL")
    assert not s.validate_conclusion("COMBO_PASS", returncode=1)


def test_supervisor_tree_stop_is_owned_only(monkeypatch):
    s = load("perf48_combo_supervisor")
    monkeypatch.setattr(s, "descendants", lambda pid: [pid, pid + 1])
    calls = []
    monkeypatch.setattr(s.os, "kill", lambda pid, sig: calls.append((pid, sig)))
    monkeypatch.setattr(s.subprocess, "check_output", lambda *args, **kwargs: "")
    owned = s.stop_tree(10, term_timeout=0)
    assert set(owned) == {10, 11} and all(pid in {10, 11} for pid, _ in calls)


# ---- deploy helper --------------------------------------------------------------
def test_deploy_build_stage_spec_from_frozen_manifest(tmp_path):
    d = load("perf48_combo_deploy")
    run = tmp_path / "frozen_run"
    run.mkdir(parents=True)
    manifest = {"stages": [
        {"name": name, "task_ids": ["task_1", "task_2"], "global_step": step,
         "initial_env_steps": step * 1024 * 128, "archive_reconstruction_limit": "all"}
        for name, step in zip(("early", "mid", "late"), (600, 2100, 3500))
    ]}
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name in ("early", "mid", "late"):
        (run / "stages" / name).mkdir(parents=True, exist_ok=True)
    out = tmp_path / "deploy"
    base_cfg = tmp_path / "perf48_off.yaml"
    base_cfg.write_text("arm: base\n", encoding="utf-8")
    spec = d.build_stage_spec(run, base_cfg, out)
    assert set(spec["stages"]) == {"early", "mid", "late"}
    assert spec["stages"]["mid"]["global_step"] == 2100
    assert spec["stages"]["mid"]["task_ids"] == ["task_1", "task_2"]
    assert spec["stages"]["early"]["initial_env_steps"] == 600 * 1024 * 128
    assert spec["config"]["base"] == str(base_cfg)
    assert "task_graph.graphml" in spec["stages"]["late"]["graph"]


def test_deploy_rejects_missing_stage():
    d = load("perf48_combo_deploy")
    run = Path(__import__("tempfile").mkdtemp()) / "frozen_run"
    run.mkdir(parents=True)
    manifest = {"stages": [{"name": "early", "task_ids": ["t"], "global_step": 600,
                            "initial_env_steps": 600 * 1024 * 128,
                            "archive_reconstruction_limit": "all"}]}
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    base_cfg = run / "perf48_off.yaml"
    base_cfg.write_text("arm: base\n", encoding="utf-8")
    with pytest.raises(ValueError):
        d.build_stage_spec(run, base_cfg, run / "deploy")


# ---- real-JAX / real-config tests (run on the server CPU; skipped locally) ----
def test_combo_env_evidence_real_jax():
    pytest.importorskip("jax")
    h = load("perf48_combo_harness")
    ev = h.env_evidence()
    assert ev.get("jax_version")
    assert ev.get("jaxlib_version")
    assert ev.get("jax_backend") in ("cpu", "gpu")


def test_combo_verify_conditioning_real_numpy(tmp_path):
    h = load("perf48_combo_harness")
    arr = np.zeros((2, 67), dtype=np.float32)
    path = tmp_path / "conditioning.npy"
    np.save(path, arr)
    content = hashlib.sha256()
    content.update(repr((2, 67)).encode())
    content.update(str(np.dtype("float32")).encode())
    content.update(np.ascontiguousarray(arr).tobytes())
    stage = {
        "conditioning": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "content_sha256": content.hexdigest(), "shape": [2, 67]},
        "embedding": {"hash": content.hexdigest()},
    }
    values, ch = h._verify_conditioning(stage)
    assert ch == content.hexdigest()
    assert list(values.shape) == [2, 67]
    # tampered content must fail closed
    np.save(path, np.ones((2, 67), dtype=np.float32))
    with pytest.raises(RuntimeError):
        h._verify_conditioning(stage)


def test_combo_arm_contract_real_config():
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf

    h = load("perf48_combo_harness")
    base = {
        "dicode_manager": {"score_function": "learnability", "max_updates_per_session": 100},
        "training": {"num_envs": 1024, "num_steps": 128, "total_timesteps": 2000000000,
                     "conditioning_type": "one_hot", "condition_on_task": True,
                     "compact_scoring_payload": False},
        "performance": {"preflight_reuse_loaded_tasks": True, "eval_compile_cache": True,
                        "compact_preflight_payload": False, "train_compile_cache": False,
                        "embedding_cache": False, "validation_cache": False},
        "runtime_profiling": {"enabled": True},
    }
    cfg_on = OmegaConf.create(base)
    h._arm_contract(cfg_on, "BC_ON")  # must not raise
    cfg_off = OmegaConf.create({**base, "performance": {**base["performance"],
                                                       "preflight_reuse_loaded_tasks": False,
                                                       "eval_compile_cache": False}})
    h._arm_contract(cfg_off, "BC_OFF")  # must not raise
    with pytest.raises(RuntimeError):
        h._arm_contract(cfg_on, "BC_OFF")  # BC_OFF cannot have B2/C on
    bad = OmegaConf.create({**base, "performance": {**base["performance"],
                                                    "compact_preflight_payload": True}})
    with pytest.raises(RuntimeError):
        h._arm_contract(bad, "BC_ON")
    no_profile = OmegaConf.create({**base, "runtime_profiling": {"enabled": False}})
    with pytest.raises(RuntimeError):
        h._arm_contract(no_profile, "BC_ON")
