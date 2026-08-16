"""CPU and fail-closed coverage for validation_replay."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest


PERF = Path(__file__).parents[4] / "experiments" / "performance"
SOURCE_ROOT = PERF.parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, PERF / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def _manifest(tmp_path: Path):
    manifest_module = load("perf48_combo_manifest")
    tmp_path.mkdir(parents=True, exist_ok=True)
    dummy_source = tmp_path / "source_evidence.py"
    dummy_source.write_text("EVIDENCE = True\n", encoding="utf-8")
    config = tmp_path / "base.yaml"
    config.write_text("performance:\n  validation_cache: false\n", encoding="utf-8")
    stages = {}
    for step, stage in enumerate(("early", "mid", "late"), 1):
        graph_path = tmp_path / f"{stage}.graphml"
        graph = nx.DiGraph()
        graph.add_node(
            "candidate_1",
            code=f"class Env:\n    pass\n# {stage}\n",
            description=stage,
        )
        nx.write_graphml(graph, graph_path)
        checkpoint = tmp_path / f"checkpoint_{step}"
        checkpoint.mkdir()
        (checkpoint / "_CHECKPOINT_METADATA").write_text("ok", encoding="utf-8")
        conditioning = tmp_path / f"{stage}.npy"
        np.save(conditioning, np.zeros((2, 67), dtype=np.float32))
        stages[stage] = {
            "graph": str(graph_path),
            "checkpoint": str(checkpoint),
            "task_ids": ["candidate_1"],
            "global_step": step,
            "initial_env_steps": step * 1024 * 128,
            "archive_reconstruction_limit": "all",
            "conditioning_path": str(conditioning),
        }
    gen_manager = SOURCE_ROOT / "src" / "dicode" / "dreaming" / "gen_manager.py"
    spec = {
        "base_dir": str(tmp_path),
        "budget": {
            "timesteps": 2_000_000_000,
            "num_envs": 1024,
            "num_steps": 128,
            "updates": 100,
        },
        "conditioning_type": "one_hot",
        "source": {
            "gen_manager": str(gen_manager),
            "extra": str(dummy_source),
        },
        "config": {"base": str(config)},
        "stages": stages,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_module.write_manifest(
        manifest_module.build_combo_manifest(spec), manifest_path
    )
    return manifest_path, dummy_source


@pytest.mark.parametrize("target", ("manifest", "source", "archive", "code"))
def test_frozen_material_tamper_fails_closed(tmp_path, target):
    replay = load("validation_replay")
    manifest_path, dummy_source = _manifest(tmp_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if target == "manifest":
        document["classification"] = "tampered"
        manifest_path.write_text(json.dumps(document), encoding="utf-8")
    elif target == "source":
        dummy_source.write_text("EVIDENCE = False\n", encoding="utf-8")
    elif target == "archive":
        Path(document["stages"][0]["graph"]["path"]).write_text(
            "tampered graph", encoding="utf-8"
        )
    else:
        entry = document["stages"][0]["candidate_codes"]["candidate_1"]
        Path(entry["path"]).write_text("class Env: pass\n# tampered\n", encoding="utf-8")
    with pytest.raises((ValueError, KeyError)):
        replay.load_replay_inputs(manifest_path, "early")


def test_atomic_result_hash_and_source_has_no_eager_jax_or_dicode_import(tmp_path):
    replay = load("validation_replay")
    result = replay._atomic_json(
        tmp_path / "result.json",
        {"classification": replay.CLASSIFICATION, "full_code_recorded": False},
    )
    assert replay._load_hashed_json(tmp_path / "result.json") == result
    tree = ast.parse((PERF / "validation_replay.py").read_text(encoding="utf-8"))
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({"jax", "dicode", "torch", "cupy", "requests", "httpx"})


def _jax_and_runtime_available():
    try:
        import jax  # noqa: F401
        import dicode.dreaming.gen_manager  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _jax_and_runtime_available(), reason="JAX/Craftax runtime absent")
def test_real_method_binding_failure_cache_counts_and_source_key_miss(monkeypatch):
    replay = load("validation_replay")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    candidate = {
        "id": "bad",
        "code": "class Env:\n    pass\n",
    }
    candidate["code_sha256"] = replay._sha256_bytes(candidate["code"].encode())

    off, off_class, off_evidence = replay._new_real_env_generator(False, 1, "source-v1")
    assert off_evidence["real_method_bound"] is True
    assert off._check_compilation_uncached.__func__ is off_class._check_compilation_uncached
    with replay._RealMethodCounter(off_class._check_compilation_uncached) as counter:
        off_rows = replay._run_phase(off, [candidate], "worker", counter)
        off_rows += replay._run_phase(off, [candidate], "main", counter, 1)
    assert counter.count == 2
    assert all(row["error_class"] == "CompilationError" for row in off_rows)
    assert not any(row["cache_hit"] for row in off_rows)

    on, on_class, on_evidence = replay._new_real_env_generator(True, 1, "source-v1")
    assert on_evidence["real_method_bound"] is True
    assert on._check_compilation_uncached.__func__ is on_class._check_compilation_uncached
    with replay._RealMethodCounter(on_class._check_compilation_uncached) as counter:
        worker = replay._run_phase(on, [candidate], "worker", counter)
        main = replay._run_phase(on, [candidate], "main", counter, 1)
        on._validation_source_sha = "source-v2"
        miss = replay._run_phase(on, [candidate], "main", counter, 2)
    assert counter.count == 2
    assert worker[0]["cache_hit"] is False
    assert main[0]["cache_hit"] is True
    assert main[0]["success"] is False  # failed validations are cached too
    assert miss[0]["cache_hit"] is False
    assert worker[0]["validator_key"]["fingerprint"] != miss[0]["validator_key"]["fingerprint"]


@pytest.mark.skipif(not _jax_and_runtime_available(), reason="JAX/Craftax runtime absent")
def test_two_real_cpu_subprocess_arms_preserve_order_and_never_record_code(tmp_path):
    replay = load("validation_replay")
    manifest_path, _ = _manifest(tmp_path / "frozen")
    out = tmp_path / "run"
    env = dict(os.environ)
    source = str(SOURCE_ROOT / "src")
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            str(PERF / "validation_replay.py"),
            "--manifest", str(manifest_path),
            "--stage", "early",
            "--out", str(out),
            "--python", sys.executable,
        ],
        env=env,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    result = replay._load_hashed_json(out / "RESULT.json")
    off = replay._load_hashed_json(out / "off.json")
    on = replay._load_hashed_json(out / "on.json")
    assert result["separate_processes"] is True
    assert len({off["pid"], on["pid"], os.getpid()}) == 3
    assert off["uncached_call_count"] == 2
    assert on["uncached_call_count"] == 1
    assert [row["phase"] for row in off["requests"]] == ["worker", "main"]
    assert [row["phase"] for row in on["requests"]] == ["worker", "main"]
    assert on["requests"][0]["cache_hit"] is False
    assert on["requests"][1]["cache_hit"] is True
    assert on["requests"][1]["error_class"] == "CompilationError"
    for arm in (off, on):
        worker_wall_s = sum(
            row["wall_s"] for row in arm["requests"] if row["phase"] == "worker"
        )
        main_wall_s = sum(
            row["wall_s"] for row in arm["requests"] if row["phase"] == "main"
        )
        assert arm["worker_wall_s"] == worker_wall_s
        assert arm["main_wall_s"] == main_wall_s
        assert arm["total_wall_s"] == worker_wall_s + main_wall_s
        assert result["arms"][arm["arm"]]["worker_wall_s"] == worker_wall_s
        assert result["arms"][arm["arm"]]["main_wall_s"] == main_wall_s
        assert result["arms"][arm["arm"]]["total_wall_s"] == worker_wall_s + main_wall_s
    effect = result["validation_cache_effect"]
    off_wall = {
        key: off[key] for key in ("worker_wall_s", "main_wall_s", "total_wall_s")
    }
    on_wall = {
        key: on[key] for key in ("worker_wall_s", "main_wall_s", "total_wall_s")
    }
    assert effect["off"] == off_wall
    assert effect["on"] == on_wall
    assert effect["main_wall_s_avoided"] == (
        off["main_wall_s"] - on["main_wall_s"]
    )
    assert effect["main_speedup"] == (
        effect["main_wall_s_avoided"] / off["main_wall_s"]
    )
    assert effect["total_wall_s_avoided"] == (
        off["total_wall_s"] - on["total_wall_s"]
    )
    assert effect["total_speedup"] == (
        effect["total_wall_s_avoided"] / off["total_wall_s"]
    )
    assert replay._load_hashed_json(out / "RESULT.json") == result
    assert all(replay._load_hashed_json(out / f"{arm}.json") for arm in ("off", "on"))
    assert all(doc["jax"]["backend"] == "cpu" for doc in (off, on))
    assert all(doc["jax"]["CUDA_VISIBLE_DEVICES"] == "" for doc in (off, on))
    assert all(doc["network_guard"] and doc["llm_api_calls"] == 0 for doc in (off, on))
    artifacts = "".join(path.read_text(encoding="utf-8") for path in out.glob("*.json"))
    assert "class Env" not in artifacts
