from __future__ import annotations

import json
from pathlib import Path
import threading

import networkx as nx
import numpy as np
from omegaconf import OmegaConf
import pytest

from dicode.skill_preflight import async_preflight as ap


SOURCE_ROOT = Path(__file__).resolve().parents[4]


def _config(tmp_path: Path, **overrides):
    values = {
        "performance": {
            "async_preflight_pipeline": True,
            "async_preflight_gpu_uuid": "GPU-test-exact",
            "async_preflight_root": str(tmp_path / "async"),
            "async_preflight_result_timeout_s": 0,
            "async_preflight_shutdown_timeout_s": 0,
            "learnability_fused_preflight_summary": True,
            "preflight_reuse_loaded_tasks": True,
        },
        "skill_preflight": {"use_preflight": True},
        "training": {"conditioning_type": "one_hot"},
        "dicode_manager": {"score_function": "learnability"},
    }
    for dotted, value in overrides.items():
        first, second = dotted.split("__", 1)
        values[first][second] = value
    return OmegaConf.create(values)


class FakeArchive:
    def __init__(self, graph=None):
        self.graph = graph if graph is not None else nx.DiGraph()
        self._lock = threading.Lock()
        self.calls = []

    def update_node_learnability(self, task_id, value):
        self.calls.append(("learnability", task_id, value))
        self.graph.nodes[task_id]["learnability_score"] = value

    def update_node_status(self, task_id, value):
        self.calls.append(("status", task_id, value))
        self.graph.nodes[task_id]["status"] = value

    def set_task_active_status(self, task_id, value):
        self.calls.append(("active", task_id, value))
        self.graph.nodes[task_id]["is_active"] = value


def _archive():
    graph = nx.DiGraph()
    graph.add_node(
        "fresh_a", code="class SecretCandidateA: pass", status="compiled",
        is_active=False, priority_score=0.0, performance_history=[],
    )
    graph.add_node(
        "fresh_b", code="class SecretCandidateB: pass", status="compiled",
        is_active=False, priority_score=0.0, performance_history=[],
    )
    return FakeArchive(graph)


class FakeCheckpointManager:
    def __init__(self):
        self.wait_calls = 0
        self.error_checks = 0
        self.metadata_steps = []

    def wait_until_finished(self):
        self.wait_calls += 1

    def check_for_errors(self):
        self.error_checks += 1

    def item_metadata(self, step):
        self.metadata_steps.append(step)
        return {"step": step}


class FakeProcess:
    next_pid = 41000

    def __init__(self, command, kwargs):
        self.command = list(command)
        self.kwargs = kwargs
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.returncode = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if self.returncode is None:
            if timeout == 0:
                raise ap.subprocess.TimeoutExpired(self.command, timeout)
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1
        self.returncode = 0

    def kill(self):
        self.kill_calls += 1
        self.returncode = -9


class ProcessFactory:
    def __init__(self):
        self.processes = []

    def __call__(self, command, **kwargs):
        process = FakeProcess(command, kwargs)
        self.processes.append(process)
        return process


def _setup(tmp_path: Path):
    config = _config(tmp_path)
    archive = _archive()
    checkpoint_root = tmp_path / "checkpoints"
    exact = checkpoint_root / "600"
    exact.mkdir(parents=True)
    (exact / "state.bin").write_bytes(b"immutable-checkpoint")
    checkpoint_manager = FakeCheckpointManager()
    factory = ProcessFactory()
    atexit_callbacks = []
    manager = ap.AsyncPreflightManager(
        config,
        source_root=SOURCE_ROOT,
        process_factory=factory,
        register_atexit=atexit_callbacks.append,
    )
    return config, archive, checkpoint_root, checkpoint_manager, factory, manager, atexit_callbacks


class FakeTaskArchive(FakeArchive):
    def __init__(self, config):
        super().__init__(nx.read_graphml(config.graph_path))


class FakeJax:
    @staticmethod
    def device_get(value):
        return value


class FakeJnp:
    @staticmethod
    def asarray(value, dtype=None):
        return np.asarray(value, dtype=dtype)


class FakeOmegaConf:
    create = staticmethod(OmegaConf.create)


def _worker_runtime(captured=None):
    captured = captured if captured is not None else {}

    def evaluate(config, rng, train_state, ids, archive, embedding_model, **kwargs):
        captured.update(
            ids=list(ids),
            embedding_model=embedding_model,
            preloaded_task_classes=kwargs.get("preloaded_task_classes"),
            preloaded_task_ids=kwargs.get("preloaded_task_ids"),
            rng=np.asarray(rng).tolist(),
        )
        return {
            "learnability_summary": {
                "finished_counts": np.asarray([2, 0], dtype=np.int32),
                "success_counts": np.asarray([1, 0], dtype=np.int32),
            }
        }

    return {
        "backend": "gpu",
        "device_count": 1,
        "jax": FakeJax,
        "jnp": FakeJnp,
        "OmegaConf": FakeOmegaConf,
        "TaskArchive": FakeTaskArchive,
        "load_agent_state": lambda config, path: {"loaded": path},
        "load_tasks": lambda archive, ids: ([object() for _ in ids], list(ids)),
        "evaluate_new_tasks": evaluate,
    }


def _launch(setup):
    config, archive, ckpt, ckpt_manager, factory, manager, callbacks = setup
    job_dir = manager.launch(
        session_idx=8,
        global_update_step=600,
        task_ids=["fresh_a", "fresh_b"],
        pf_rng=np.asarray([11, 29], dtype=np.uint32),
        archive=archive,
        rl_ckpt_manager=ckpt_manager,
        rl_ckpt_path=ckpt,
    )
    return job_dir, factory.processes[-1]


def _run_fake_worker(monkeypatch, job_dir, captured=None):
    job = ap.load_hashed_json(job_dir / "JOB.json")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", job["gpu_uuid"])
    return ap.run_worker_job(job_dir / "JOB.json", runtime=_worker_runtime(captured))


def test_default_off_and_pure_session_plan_have_zero_async_work():
    config_text = (SOURCE_ROOT / "conf/config.yaml").read_text(encoding="utf-8")
    assert "async_preflight_pipeline: false" in config_text
    assert ap.async_pipeline_enabled({"performance": {}}) is False
    assert ap.plan_async_session(
        async_enabled=False, delayed_ids=["old"], fresh_ids=["fresh"], pending=True
    ) == {"training_new_ids": ["fresh"], "launch_ids": []}
    source = (SOURCE_ROOT / "experiments/training/run_dicode.py").read_text(encoding="utf-8")
    flag = source.index("if _async_pf_enabled:")
    assert source.index("from dicode.skill_preflight.async_preflight import") > flag
    assert "new_task_ids = fresh_task_ids" in source


@pytest.mark.parametrize(
    "override",
    [
        {"skill_preflight__use_preflight": False},
        {"training__conditioning_type": "embedding"},
        {"dicode_manager__score_function": "pvl"},
        {"performance__learnability_fused_preflight_summary": False},
        {"performance__preflight_reuse_loaded_tasks": False},
        {"performance__async_preflight_gpu_uuid": None},
    ],
)
def test_contract_fails_before_root_or_launch(tmp_path, override):
    config = _config(tmp_path, **override)
    with pytest.raises(ap.AsyncPreflightError, match="before launch"):
        ap.AsyncPreflightManager(config, source_root=SOURCE_ROOT)
    assert not (tmp_path / "async").exists()


def test_launch_writes_minimal_hashed_receipts_and_exact_gpu_env(tmp_path):
    setup = _setup(tmp_path)
    config, archive, ckpt, ckpt_manager, factory, manager, callbacks = setup
    job_dir, process = _launch(setup)
    job = ap.load_hashed_json(job_dir / "JOB.json")
    running = ap.load_hashed_json(job_dir / "RUNNING.json")
    assert ckpt_manager.wait_calls == 1
    assert ckpt_manager.error_checks == 1
    assert ckpt_manager.metadata_steps == [600]
    assert job["checkpoint_metadata_verified"] is True
    assert job["checkpoint_path"] == str((ckpt / "600").resolve())
    assert job["task_ids"] == ["fresh_a", "fresh_b"]
    assert [row["task_id"] for row in job["task_code_hashes"]] == job["task_ids"]
    assert "SecretCandidate" not in (job_dir / "JOB.json").read_text(encoding="utf-8")
    assert process.command[1:4] == ["-m", "dicode.skill_preflight.async_preflight", "--job"]
    assert process.kwargs["start_new_session"] is True
    env = process.kwargs["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-test-exact"
    assert env["WANDB_MODE"] == "offline"
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["DICODE_ASYNC_PREFLIGHT_NO_NETWORK"] == "1"
    assert running["pid"] == process.pid and running["owned_by_pid"] == ap.os.getpid()
    assert callbacks == [manager.shutdown]


def test_worker_fused_receipt_then_next_session_routes_exactly_once(tmp_path, monkeypatch):
    setup = _setup(tmp_path)
    _, archive, _, _, _, manager, _ = setup
    job_dir, process = _launch(setup)
    assert manager.poll_and_apply(
        archive=archive, current_session_idx=8, route_apply_fn=lambda *args: None,
        route_fn=lambda *args: None,
    ) is None
    captured = {}
    result = _run_fake_worker(monkeypatch, job_dir, captured)
    process.returncode = 0
    route_calls = []

    def route_apply(scores, ok_ids, kept, target_archive, route_fn):
        route_calls.append((scores, list(ok_ids)))
        for index, task_id in enumerate(ok_ids):
            if scores[str(index)]["sr"] == 0.5:
                kept.append(task_id)

    kept = manager.poll_and_apply(
        archive=archive,
        current_session_idx=9,
        route_apply_fn=route_apply,
        route_fn=object(),
    )
    assert kept == ["fresh_a"]
    assert len(route_calls) == 1
    assert result["route_calls"] == 0 and result["archive_mutations"] == 0
    assert result["jax_backend"] == "gpu" and result["jax_device_count"] == 1
    assert captured["embedding_model"] is None
    assert captured["preloaded_task_ids"] == ["fresh_a", "fresh_b"]
    assert len(captured["preloaded_task_classes"]) == 2
    applied = ap.load_hashed_json(job_dir / "APPLIED.json")
    assert applied["route_calls"] == 1 and applied["kept_ids"] == ["fresh_a"]
    assert manager.poll_and_apply(
        archive=archive, current_session_idx=10, route_apply_fn=route_apply,
        route_fn=object(),
    ) == []
    assert len(route_calls) == 1


def test_session_n_excludes_fresh_and_n_plus_one_includes_delayed():
    launched = ap.plan_async_session(
        async_enabled=True, delayed_ids=[], fresh_ids=["fresh"], pending=False
    )
    assert launched == {"training_new_ids": [], "launch_ids": ["fresh"]}
    applied = ap.plan_async_session(
        async_enabled=True, delayed_ids=["fresh"], fresh_ids=[], pending=False
    )
    assert applied == {"training_new_ids": ["fresh"], "launch_ids": []}
    with pytest.raises(ap.AsyncPreflightError, match="fresh batch"):
        ap.plan_async_session(
            async_enabled=True, delayed_ids=[], fresh_ids=["next"], pending=True
        )


def test_busy_launch_and_failed_worker_fail_closed(tmp_path):
    setup = _setup(tmp_path)
    _, archive, ckpt, ckpt_manager, _, manager, _ = setup
    job_dir, process = _launch(setup)
    with pytest.raises(ap.AsyncPreflightError, match="pending job"):
        manager.launch(
            session_idx=9, global_update_step=700, task_ids=["fresh_a"],
            pf_rng=[1, 2], archive=archive, rl_ckpt_manager=ckpt_manager,
            rl_ckpt_path=ckpt,
        )
    ap.atomic_json(
        job_dir / "FAILURE.json",
        {"classification": ap.CLASSIFICATION, "error_class": "Boom", "error": "failed", "llm_api_calls": 0},
    )
    process.returncode = 1
    with pytest.raises(ap.AsyncPreflightError, match="Boom failed"):
        manager.poll_and_apply(
            archive=archive, current_session_idx=9,
            route_apply_fn=lambda *args: None, route_fn=lambda *args: None,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("session_idx", 99),
        ("global_update_step", 601),
        ("pf_rng_sha256", "bad-rng"),
        ("source_sha256", "bad-source"),
        ("checkpoint_sha256", "bad-checkpoint"),
    ],
)
def test_result_receipt_mismatch_fails_closed(tmp_path, monkeypatch, field, value):
    setup = _setup(tmp_path)
    _, archive, _, _, _, manager, _ = setup
    job_dir, process = _launch(setup)
    result = _run_fake_worker(monkeypatch, job_dir)
    result[field] = value
    ap.atomic_json(job_dir / "RESULT.json", result)
    process.returncode = 0
    with pytest.raises(ap.AsyncPreflightError, match="receipt mismatch"):
        manager.poll_and_apply(
            archive=archive, current_session_idx=9,
            route_apply_fn=lambda *args: None, route_fn=lambda *args: None,
        )


def test_live_code_and_checkpoint_mismatch_fail_closed(tmp_path, monkeypatch):
    setup = _setup(tmp_path)
    _, archive, ckpt, _, _, manager, _ = setup
    job_dir, process = _launch(setup)
    _run_fake_worker(monkeypatch, job_dir)
    process.returncode = 0
    archive.graph.nodes["fresh_a"]["code"] = "changed"
    with pytest.raises(ap.AsyncPreflightError, match="candidate order/code"):
        manager.poll_and_apply(
            archive=archive, current_session_idx=9,
            route_apply_fn=lambda *args: None, route_fn=lambda *args: None,
        )
    archive.graph.nodes["fresh_a"]["code"] = "class SecretCandidateA: pass"
    (ckpt / "600/state.bin").write_bytes(b"changed")
    with pytest.raises(ap.AsyncPreflightError, match="checkpoint changed"):
        manager.poll_and_apply(
            archive=archive, current_session_idx=9,
            route_apply_fn=lambda *args: None, route_fn=lambda *args: None,
        )


def test_apply_same_launch_session_is_rejected(tmp_path, monkeypatch):
    setup = _setup(tmp_path)
    _, archive, _, _, _, manager, _ = setup
    job_dir, process = _launch(setup)
    _run_fake_worker(monkeypatch, job_dir)
    process.returncode = 0
    with pytest.raises(ap.AsyncPreflightError, match="launch session"):
        manager.poll_and_apply(
            archive=archive, current_session_idx=8,
            route_apply_fn=lambda *args: None, route_fn=lambda *args: None,
        )


def test_job_and_result_selfhash_tamper(tmp_path, monkeypatch):
    setup = _setup(tmp_path)
    _, _, _, _, _, _, _ = setup
    job_dir, _ = _launch(setup)
    job_path = job_dir / "JOB.json"
    raw = json.loads(job_path.read_text(encoding="utf-8"))
    raw["session_idx"] = 100
    job_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ap.AsyncPreflightError, match="self-hash"):
        ap.load_hashed_json(job_path)


def test_result_and_applied_selfhash_tamper_fail_recovery(tmp_path, monkeypatch):
    setup = _setup(tmp_path)
    config, archive, _, _, _, manager, _ = setup
    job_dir, process = _launch(setup)
    _run_fake_worker(monkeypatch, job_dir)
    result_path = job_dir / "RESULT.json"
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    raw["score_projection"][0]["sr"] = 0.9
    result_path.write_text(json.dumps(raw), encoding="utf-8")
    process.returncode = 0
    with pytest.raises(ap.AsyncPreflightError, match="self-hash"):
        manager.poll_and_apply(
            archive=archive, current_session_idx=9,
            route_apply_fn=lambda *args: None, route_fn=object(),
        )

    setup2 = _setup(tmp_path / "applied")
    config2, archive2, _, _, _, manager2, _ = setup2
    job_dir2, process2 = _launch(setup2)
    _run_fake_worker(monkeypatch, job_dir2)
    process2.returncode = 0

    def apply(scores, ids, kept, archive, route):
        kept.extend(ids)

    manager2.poll_and_apply(
        archive=archive2, current_session_idx=9,
        route_apply_fn=apply, route_fn=object(),
    )
    applied_path = job_dir2 / "APPLIED.json"
    applied_raw = json.loads(applied_path.read_text(encoding="utf-8"))
    applied_raw["route_calls"] = 2
    applied_path.write_text(json.dumps(applied_raw), encoding="utf-8")
    with pytest.raises(ap.AsyncPreflightError, match="self-hash"):
        ap.AsyncPreflightManager(
            config2, source_root=SOURCE_ROOT, process_factory=ProcessFactory(),
            register_atexit=lambda fn: None,
        )


def test_missing_exact_checkpoint_and_snapshot_tamper_fail_closed(tmp_path, monkeypatch):
    setup = _setup(tmp_path)
    _, archive, ckpt, ckpt_manager, _, manager, _ = setup
    with pytest.raises(ap.AsyncPreflightError, match="checkpoint path does not exist"):
        manager.launch(
            session_idx=8, global_update_step=601,
            task_ids=["fresh_a"], pf_rng=[1, 2], archive=archive,
            rl_ckpt_manager=ckpt_manager, rl_ckpt_path=ckpt,
        )
    job_dir, process = _launch(setup)
    _run_fake_worker(monkeypatch, job_dir)
    process.returncode = 0
    graph_path = job_dir / "archive.graphml"
    graph_path.write_bytes(graph_path.read_bytes() + b"\n")
    with pytest.raises(ap.AsyncPreflightError, match="snapshot changed"):
        manager.poll_and_apply(
            archive=archive, current_session_idx=9,
            route_apply_fn=lambda *args: None, route_fn=object(),
        )


def test_resume_completed_result_and_reject_unowned_running(tmp_path, monkeypatch):
    setup = _setup(tmp_path)
    config, archive, _, _, _, first, _ = setup
    job_dir, process = _launch(setup)
    _run_fake_worker(monkeypatch, job_dir)
    process.returncode = 0
    # Simulate process loss after a durable result; a new manager may apply it.
    resumed = ap.AsyncPreflightManager(
        config, source_root=SOURCE_ROOT, process_factory=ProcessFactory(),
        register_atexit=lambda fn: None,
    )
    calls = []

    def apply(scores, ids, kept, archive, route):
        calls.append(1)
        kept.extend(ids)

    assert resumed.poll_and_apply(
        archive=archive, current_session_idx=9, route_apply_fn=apply, route_fn=object()
    ) == ["fresh_a", "fresh_b"]
    assert calls == [1]

    other_root = tmp_path / "other"
    other_config = _config(other_root)
    other_config.performance.async_preflight_root = str(other_root / "async")
    other_setup = _setup(other_root)
    _, _, _, _, _, running_manager, _ = other_setup
    _, external_process = _launch(other_setup)
    with pytest.raises(ap.AsyncPreflightError, match="not owned"):
        ap.AsyncPreflightManager(
            other_setup[0], source_root=SOURCE_ROOT,
            process_factory=ProcessFactory(), register_atexit=lambda fn: None,
        )
    assert external_process.terminate_calls == external_process.kill_calls == 0


def test_applying_receipt_is_unrecoverable_and_never_routes_twice(tmp_path, monkeypatch):
    setup = _setup(tmp_path)
    config, _, _, _, _, manager, _ = setup
    job_dir, process = _launch(setup)
    result = _run_fake_worker(monkeypatch, job_dir)
    job = ap.load_hashed_json(job_dir / "JOB.json")
    process.returncode = 0
    ap.atomic_json(
        job_dir / "APPLYING.json",
        {
            "classification": ap.CLASSIFICATION,
            "job_sha256": job["result_sha256"],
            "result_receipt_sha256": result["result_sha256"],
            "session_applied": 9,
            "llm_api_calls": 0,
        },
    )
    with pytest.raises(ap.AsyncPreflightError, match="already begun"):
        manager.poll_and_apply(
            archive=setup[1], current_session_idx=9,
            route_apply_fn=lambda *args: pytest.fail("route must not run"),
            route_fn=object(),
        )
    with pytest.raises(ap.AsyncPreflightError, match="route may be partial"):
        ap.AsyncPreflightManager(
            config, source_root=SOURCE_ROOT, process_factory=ProcessFactory(),
            register_atexit=lambda fn: None,
        )


def test_shutdown_signals_only_owned_process_object(tmp_path):
    setup = _setup(tmp_path)
    _, _, _, _, _, manager, _ = setup
    _, process = _launch(setup)
    manager.shutdown()
    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert manager._process is None


def test_shutdown_escalates_owned_process_only_after_timeout(tmp_path):
    class StuckProcess(FakeProcess):
        def terminate(self):
            self.terminate_calls += 1

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            if self.kill_calls == 0:
                raise ap.subprocess.TimeoutExpired(self.command, timeout)
            self.returncode = -9
            return self.returncode

    class StuckFactory(ProcessFactory):
        def __call__(self, command, **kwargs):
            process = StuckProcess(command, kwargs)
            self.processes.append(process)
            return process

    config = _config(tmp_path)
    archive = _archive()
    ckpt = tmp_path / "checkpoints/600"
    ckpt.mkdir(parents=True)
    (ckpt / "state").write_bytes(b"x")
    factory = StuckFactory()
    manager = ap.AsyncPreflightManager(
        config, source_root=SOURCE_ROOT, process_factory=factory,
        register_atexit=lambda fn: None,
    )
    manager.launch(
        session_idx=8, global_update_step=600, task_ids=["fresh_a"],
        pf_rng=[1, 2], archive=archive,
        rl_ckpt_manager=FakeCheckpointManager(), rl_ckpt_path=ckpt.parent,
    )
    process = factory.processes[0]
    manager.shutdown()
    assert process.terminate_calls == 1 and process.kill_calls == 1


def test_worker_rejects_wrong_gpu_env_and_never_stores_full_code(tmp_path, monkeypatch):
    setup = _setup(tmp_path)
    job_dir, _ = _launch(setup)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-wrong")
    with pytest.raises(ap.AsyncPreflightError, match="CUDA_VISIBLE_DEVICES"):
        ap.run_worker_job(job_dir / "JOB.json", runtime=_worker_runtime())
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in job_dir.iterdir()
        if path.suffix == ".json"
    )
    assert "SecretCandidate" not in text


def test_source_audit_has_no_client_or_prompt_path_and_wiring_is_delayed():
    module = (SOURCE_ROOT / ap.SOURCE_RELATIVES["async_preflight"]).read_text(encoding="utf-8")
    assert "openai" not in module.lower()
    assert "requests." not in module and "urllib.request" not in module
    assert "prompt" not in module.lower()
    driver = (SOURCE_ROOT / "experiments/training/run_dicode.py").read_text(encoding="utf-8")
    assert driver.index("poll_and_apply(") < driver.index("# --- Step 1:")
    launch = driver.index("_async_pf_manager.launch(")
    assert driver.rfind("rng, _async_pf_rng = jax.random.split(rng)", 0, launch) != -1
    assert "and new_task_ids and _async_pf_manager is None" in driver
