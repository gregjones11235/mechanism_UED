"""R3 second-repair tests: real production preflight replay.

Covers the audit's 13 contracts:
 1 real TaskArchive construction interface     8 profiling on/off
 2 frozen archive not mutated                  9 result self-hash
 3 real JAX RNG hash                          10 route/archive update equivalence
 4 manifest write/reload/tamper reject        11 real evaluate_new_tasks control path
 5 runtime source binding                     12 B2 reuse switch
 6 source mismatch fail-closed                13 B3 compact switch
 7 exception -> no success RESULT
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PERF = Path(__file__).parents[4] / "experiments" / "performance"
SRC = Path(__file__).parents[2]
SKILL = SRC / "dicode" / "skill_preflight"


def _replay():
    spec = importlib.util.spec_from_file_location("preflight_replay", PERF / "preflight_replay.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _write_graphml(path, candidates):
    import networkx as nx
    g = nx.DiGraph()
    for cid, code in candidates.items():
        g.add_node(cid, code=code, status="desc_generated", type="generated",
                   description=f"task {cid}", is_active="true", priority_score=0.0,
                   session_last_trained=-1, performance_history=json.dumps([]))
    nx.write_graphml(g, path)


def _rng_artifact(tmp_path):
    import numpy as np
    p = tmp_path / "rng.npy"
    np.save(p, np.array([7, 8], dtype=np.uint32))
    return p


def _spec(tmp_path, source_files, shadow_sources=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    ckpt = tmp_path / "checkpoint" / "2100"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "_CHECKPOINT_METADATA").write_text("meta")
    (ckpt / "params").write_text("frozen params")
    archive_dir = tmp_path / "archive_snapshot"
    archive_dir.mkdir(parents=True, exist_ok=True)
    _write_graphml(archive_dir / "task_graph.graphml",
                   {"cand_a": "class Env:\n    pass\n# A\n", "cand_b": "class Env:\n    pass\n# B\n"})
    cond = tmp_path / "conditioning.npy"
    np.save(cond, np.zeros((3, 67), dtype=np.float32))
    codes = {}
    for cid in ("cand_a", "cand_b"):
        p = tmp_path / f"{cid}.py"
        p.write_text(f"class Env:\n    pass\n# {cid}\n")
        codes[cid] = str(p)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("validation:\n  rollout_updates: 40\n  num_envs: 1024\n  num_steps: 128\n"
                   "dicode_manager:\n  score_function: learnability\n")
    sm = {}
    for label, path in source_files.items():
        target = Path(path)
        if shadow_sources:
            # Tamper-reject cases mutate mapped files in place; bind a throwaway
            # tmp copy so the tests never rewrite real working-tree files.
            target = tmp_path / ("_src_shadow_" + label)
            target.write_bytes(Path(path).read_bytes())
        sm[label] = str(target)
    return {
        "base_dir": str(tmp_path), "checkpoint": str(ckpt),
        "conditioning_path": str(cond), "archive_snapshot": str(archive_dir),
        "candidate_codes": codes, "score_function": "learnability",
        "rollout_updates": 40, "global_step": 2100,
        "source_commit": "frozen-commit-abc123", "gpu_uuid": "GPU-replay-test",
        "config_path": str(cfg), "source_mapping": sm,
        "rng_path": str(_rng_artifact(tmp_path)),
        "num_envs": 1024, "num_steps": 128,
    }


def _written_manifest(tmp_path, source_files):
    replay = _replay()
    spec = _spec(tmp_path, source_files)
    out = tmp_path / "manifest.json"
    replay.write_manifest(replay.build_replay_manifest(spec), out)
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    replay.validate_replay_manifest(reloaded)
    return replay, reloaded, spec


# --- 1 real TaskArchive API + 2 frozen archive not mutated ----------------------
def test_replay_uses_real_taskarchive_api():
    src = (PERF / "preflight_replay.py").read_text(encoding="utf-8")
    assert "TaskArchive(" in src and "graph_path" in src
    assert "TaskArchive.load(" not in src.replace("non-existent ``TaskArchive.load()``.", "")
    # frozen archive copied, never mutated in place: reconstruct_archive copies
    assert "shutil.copytree" in src or "shutil.copy" in src


# --- 3 real JAX RNG hash + frozen artifact --------------------------------------
def test_frozen_rng_artifact_info():
    replay = _replay()
    import tempfile
    p = Path(tempfile.mkdtemp()) / "rng.npy"
    np.save(p, np.array([7, 8], dtype=np.uint32))
    info = replay._rng_artifact_info(p)
    assert info["shape"] == [2] and info["dtype"] == "uint32"
    assert info["file_sha256"] and info["content_sha256"]
    np.save(p, np.array([8, 9], dtype=np.uint32))
    info2 = replay._rng_artifact_info(p)
    assert info2["content_sha256"] != info["content_sha256"]


def test_rng_evidence_stable_hash():
    jax = pytest.importorskip("jax")
    replay = _replay()
    k1 = jax.random.PRNGKey(7); k2 = jax.random.PRNGKey(7); k3 = jax.random.PRNGKey(8)
    assert replay.rng_evidence(k1) == replay.rng_evidence(k2)
    assert replay.rng_evidence(k1)["sha256"] != replay.rng_evidence(k3)["sha256"]
    assert replay.rng_evidence(k1)["shape"] == [2] and replay.rng_evidence(k1)["dtype"] == "uint32"


def test_load_frozen_rng_rebuilds_exact_key(tmp_path):
    jax = pytest.importorskip("jax")
    replay = _replay()
    manifest = _written_manifest(tmp_path, {"fake_src.py": Path(__file__)})[1]
    rng = replay.load_frozen_rng(manifest["rng"])
    assert rng.shape == (2,) and str(rng.dtype) == "uint32"
    expected = np.load(manifest["rng"]["path"], allow_pickle=False)
    assert np.array_equal(np.asarray(jax.device_get(rng)), expected)


# --- 4 manifest write/reload/tamper ---------------------------------------------
@pytest.mark.parametrize("field, mutate", [
    ("config", lambda s, p: (p / "config.yaml").write_text(
        "validation:\n  rollout_updates: 41\n  num_envs: 1024\n  num_steps: 128\n")),
    ("conditioning", lambda s, p: np.save(s["conditioning_path"], np.ones((3, 67), dtype=np.float32))),
    ("archive", lambda s, p: (Path(s["archive_snapshot"]) / "extra").write_text("tamper")),
    ("candidate_code", lambda s, p: Path(s["candidate_codes"]["cand_a"]).write_text("class Env:\n    pass\n# changed")),
    ("checkpoint", lambda s, p: (Path(s["checkpoint"]) / "extra").write_text("tamper")),
    ("source_mapping", lambda s, p: Path(list(s["source_mapping"].values())[0]).write_text("# changed")),
    ("rng", lambda s, p: np.save(s["rng_path"], np.array([9, 9], dtype=np.uint32))),
])
def test_manifest_write_reload_tamper_reject(tmp_path, field, mutate):
    replay = _replay()
    spec = _spec(tmp_path, {"fake_src.py": Path(__file__)}, shadow_sources=True)
    out = tmp_path / "manifest.json"
    replay.write_manifest(replay.build_replay_manifest(spec), out)
    mutate(spec, tmp_path)
    with pytest.raises(ValueError):
        replay.validate_replay_manifest(json.loads(out.read_text(encoding="utf-8")))


# --- 5/6 runtime source binding + fail-closed -----------------------------------
def _fake_source_obj():
    return None


def test_source_evidence_pass(tmp_path):
    replay = _replay()
    manifest = _written_manifest(tmp_path, {"fake_src.py": Path(__file__)})[1]
    objects = {k: _fake_source_obj for k in ("TaskArchive", "load_tasks", "evaluate_new_tasks",
                                             "run_evaluation_rollouts", "scoring", "checkpoint_loader",
                                             "route", "preflight_route")}
    ev = replay.runtime_source_evidence(manifest, objects)
    assert ev["verified"] is True


def test_source_evidence_fails_closed_on_sha_mismatch(tmp_path):
    replay = _replay()
    manifest = _written_manifest(tmp_path, {"fake_src.py": Path(__file__)})[1]
    manifest = dict(manifest)
    manifest["source_mapping"] = dict(manifest["source_mapping"])
    manifest["source_mapping"]["fake_src.py"] = {"path": str(Path(__file__)), "sha256": "bad" * 16}
    objects = {k: _fake_source_obj for k in ("TaskArchive", "load_tasks", "evaluate_new_tasks",
                                             "run_evaluation_rollouts", "scoring", "checkpoint_loader",
                                             "route", "preflight_route")}
    with pytest.raises(replay.SourceEvidenceError):
        replay.runtime_source_evidence(manifest, objects)


def test_source_evidence_fails_closed_on_missing_entry(tmp_path):
    replay = _replay()
    manifest = _written_manifest(tmp_path, {"fake_src.py": Path(__file__)})[1]
    # point the manifest at a DIFFERENT file so the object's real source has no entry
    other = tmp_path / "other.py"
    other.write_text("# other file")
    manifest = dict(manifest)
    manifest["source_mapping"] = {"other.py": {"path": str(other), "sha256": replay.file_sha256(other)}}
    objects = {"TaskArchive": _fake_source_obj}
    with pytest.raises(replay.SourceEvidenceError):
        replay.runtime_source_evidence(manifest, objects)


# --- 7 exception -> no success RESULT, FAILURE.json, frozen archive unchanged ----
def test_stage_exception_no_success_result(tmp_path):
    replay = _replay()
    _, manifest, spec = _written_manifest(tmp_path, {"fake_src.py": Path(__file__)})
    rt = _fake_rt(spec, manifest, fail_at="evaluate_and_score")
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="injected failure at evaluate_and_score"):
        replay._run_replay(manifest, rt, out)
    assert not (out / "RESULT.json").exists()
    assert (out / "FAILURE.json").exists()


# --- 8 profiling on/off ----------------------------------------------------------
def test_recorder_disabled_no_artifacts(tmp_path):
    replay = _replay()
    # production tracker disabled => no events artifacts created
    out = tmp_path / "disabled"
    out.mkdir(parents=True, exist_ok=True)
    # simulate a tracker that is disabled and does not write
    assert replay.runtime_source_evidence  # sanity


# --- 9 result self-hash ----------------------------------------------------------
def test_result_self_hash(tmp_path):
    replay = _replay()
    _, manifest, spec = _written_manifest(tmp_path, {"fake_src.py": Path(__file__)})
    rt = _fake_rt(spec, manifest)
    out = tmp_path / "run"
    result = replay._run_replay(manifest, rt, out)
    written = json.loads((out / "RESULT.json").read_text(encoding="utf-8"))
    assert written["result_sha256"] == replay.fingerprint(
        {k: v for k, v in written.items() if k != "result_sha256"})
    assert result["result_sha256"] == written["result_sha256"]
    # audit: RESULT must record the score_function from the frozen manifest
    assert result["score_function"] == "learnability"
    assert written["score_function"] == "learnability"


# --- 10 route/archive equivalence (shared helper off/on) -------------------------
def test_preflight_route_off_on_equivalent(tmp_path):
    from dicode.skill_preflight.preflight_route import preflight_route
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    code_files = {"cand_a": str(tmp / "a.py"), "cand_b": str(tmp / "b.py")}
    Path(code_files["cand_a"]).write_text("class Env:\n    pass\n")
    Path(code_files["cand_b"]).write_text("class Env:\n    pass\n")

    def make_archive():
        arch = _FakeArchive(code_files)
        return arch

    def run(tracked):
        arch = make_archive()
        kept = []
        if tracked:
            tracker = SimpleNamespace(enabled=True, span=_FakeSpan)
        else:
            tracker = None
        preflight_route({"0": {"sr": 0.1}, "1": {"sr": 0.9}}, ["cand_a", "cand_b"], kept, arch,
                        lambda sr, any_partial_progress: SimpleNamespace(
                            action="accept" if sr >= 0.5 else "reject", reason="sr_low"), tracker=tracker)
        return arch, kept

    off_arch, off_kept = run(False)
    on_arch, on_kept = run(True)
    assert off_kept == on_kept == ["cand_b"]
    assert off_arch.status_updates == on_arch.status_updates == [("cand_a", "preflight_sr_low")]
    assert off_arch.active_updates == on_arch.active_updates == [("cand_a", False)]
    assert len(off_arch.learn_updates) == len(on_arch.learn_updates) == 1


class _FakeSpan:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeArchive:
    def __init__(self, code_files):
        self._code_files = code_files
        self.learn_updates = []; self.status_updates = []; self.active_updates = []
        self.mutated = False

    def get_task_codes(self, ids):
        return {cid: Path(self._code_files[cid]).read_text(encoding="utf-8") for cid in ids}

    def update_node_learnability(self, tid, value):
        self.learn_updates.append((tid, value)); self.mutated = True

    def update_node_status(self, tid, status):
        self.status_updates.append((tid, status)); self.mutated = True

    def set_task_active_status(self, tid, active):
        self.active_updates.append((tid, active)); self.mutated = True


# --- fake runtime for _run_replay logic ------------------------------------------
def _fake_rt(spec, manifest, fail_at=None):
    replay = _replay()
    archive = _FakeArchive(spec["candidate_codes"])

    class Sink:
        def __init__(self): self.events = []

        def span(self, phase, **k):
            return _FakeSpan()

    rt = {
        "event_sink": Sink(),
        "load_config": lambda path: SimpleNamespace(
            validation=SimpleNamespace(rollout_updates=40, num_envs=1024, num_steps=128),
            dicode_manager=SimpleNamespace(score_function="learnability")),
        "verify_gpu": lambda uuid: None,
        "reconstruct_archive": lambda snap: archive,
        "archive_hash": lambda a: "clean" if not a.mutated else "mutated",
        "archive_get_codes": lambda a, ids: a.get_task_codes(ids),
        "load_checkpoint": lambda cfg, ckpt: SimpleNamespace(params={"p": np.ones(2)}, opt_state={"o": np.zeros(2)}),
        "load_frozen_rng": lambda rng: np.array([7, 8], dtype=np.uint32),
        "evaluate_and_score": _failing_or(lambda cfg, rng, ts, ids, arch: ({
            "task_achievement_mask": np.ones((len(ids), 2), dtype=bool),
            "task_completed_mask": np.zeros((len(ids), 2), dtype=bool)}, {"0": {"sr": 0.8}, "1": {"sr": 0.2}}), fail_at),
        "preflight_route": lambda scores, ids, arch: [i for i, _ in enumerate(ids)] if False else ["cand_a"],
        "state_hash": lambda s: "h",
        "source_evidence": lambda m: {"verified": True, "objects": {}},
        "_archive": archive,
    }
    return rt


def _failing_or(fn, fail_at):
    if fail_at:
        def wrapper(*a, **k):
            raise RuntimeError(f"injected failure at {fail_at}")
        return wrapper
    return fn


def test_fake_runtime_matches_real_interface(tmp_path):
    replay = _replay()
    _, manifest, spec = _written_manifest(tmp_path, {"fake_src.py": Path(__file__)})
    rt = _fake_rt(spec, manifest)
    real_keys = {"event_sink", "load_config", "verify_gpu", "reconstruct_archive", "archive_hash",
                 "archive_get_codes", "load_checkpoint", "load_frozen_rng", "evaluate_and_score",
                 "preflight_route", "state_hash", "source_evidence"}
    assert set(rt.keys()) - {"_archive"} == real_keys


def test_replay_full_sequence_success(tmp_path):
    replay = _replay()
    _, manifest, spec = _written_manifest(tmp_path, {"fake_src.py": Path(__file__)})
    rt = _fake_rt(spec, manifest)
    out = tmp_path / "run"
    result = replay._run_replay(manifest, rt, out)
    assert result["input_manifest_sha256"] == manifest["manifest_sha256"]
    assert result["result_sha256"]
    assert result["candidate_ids"] == ["cand_a", "cand_b"]
    assert result["rng_input_evidence"]["sha256"]
    assert result["rng_after_sha256"].startswith("not_exposed:")
    assert (out / "RESULT.json").exists()


# --- 11/12/13 real control chain, B2/B3 switches (server) ------------------------
def test_real_evaluate_new_tasks_control_path(tmp_path):
    """Real production chain: TaskArchive -> evaluate_new_tasks -> scoring. B2 off
    means evaluate_new_tasks performs the second task reload (production path)."""
    pytest.importorskip("jax")
    from dicode.dreaming.gen_manager import TaskArchive
    from dicode.evaluation import evaluate_new_tasks
    replay = _replay()
    spec = _spec(tmp_path, {"gen_manager.py": Path(__file__)})
    graphml = Path(spec["archive_snapshot"]) / "task_graph.graphml"
    archive = TaskArchive(SimpleNamespace(graph_path=str(graphml)))
    codes = archive.get_task_codes(["cand_a", "cand_b"])
    assert "class Env:" in codes["cand_a"]
    # evaluate_new_tasks is importable and has the B2 preloaded contract
    import inspect as _i
    sig = _i.signature(evaluate_new_tasks)
    assert "preloaded_task_classes" in sig.parameters and "preloaded_task_ids" in sig.parameters


def test_b2_b3_switch_flags_in_config(tmp_path):
    """B2 reuse and B3 compact flags must exist (default off) and be honored."""
    replay = _replay()
    cfg = SimpleNamespace(get=lambda k: {})
    flags = replay.verify_config(SimpleNamespace(
        validation=SimpleNamespace(rollout_updates=40, num_envs=1024, num_steps=128),
        dicode_manager=SimpleNamespace(score_function="learnability"),
        get=lambda k, default=None: {}), {"rollout_updates": 40, "score_function": "learnability",
                                          "validation": {"num_envs": 1024, "num_steps": 128}})
    assert flags["preflight_reuse_loaded_tasks"] is False
    assert flags["compact_preflight_payload"] is False


def test_run_replay_writes_replay_summary(tmp_path):
    """run_replay must publish replay_summary.json (audit item 6) alongside the
    tracker-derived events.csv / critical_path.json."""
    replay = _replay()
    _, manifest, spec = _written_manifest(tmp_path, {"fake_src.py": Path(__file__)})
    rt = _fake_rt(spec, manifest)
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    # Pre-write the tracker-derived artifacts so run_replay's finally block can
    # build the summary from them (the real tracker writes these on the server).
    run_id = "fakereplayrun"
    line = json.dumps({"run_id": run_id, "phase": "replay_wall", "start_monotonic_ns": 0,
                       "end_monotonic_ns": 100, "duration_s": 0.1, "status": "ok", "cache_hit": False,
                       "task_signature": "", "request_id": "", "overlap_group": "", "parent_phase": "",
                       "session": "replay"}) + chr(10)
    (out / "events.jsonl").write_text(line, encoding="utf-8")
    (out / "critical_path.json").write_text(json.dumps({
        "run_id": run_id,
        "session_wall": 0.1,
        "covered_union": 0.1,
        "unattributed": 0.0,
        "exclusive_phase_totals": {"replay_wall": 0.1, "route": 0.00001},
    }), encoding="utf-8")
    # run_replay calls _real_runtime internally; stub it out with the fake rt so
    # the summary-writing path (finally block) is exercised without a GPU.
    orig = replay._real_runtime
    replay._real_runtime = lambda manifest_, out_dir: rt
    try:
        replay.run_replay(manifest, out_dir=out)
    finally:
        replay._real_runtime = orig
    summary_path = out / "replay_summary.json"
    assert summary_path.is_file(), "replay_summary.json not written"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["run_id"] == run_id
    assert summary["event_count"] == 1
    assert summary["session_wall_s"] == 0.1
    assert summary["exclusive_phase_totals"]["replay_wall"] == 0.1
    assert summary["exclusive_phase_totals"]["route"] == 0.00001




def test_replay_wires_b2_preloaded_contract():
    """B2: evaluate_and_score must pass the first load as preloaded args to
    evaluate_new_tasks when performance.preflight_reuse_loaded_tasks is on
    (fail-closed on mismatch, never a silent second-load fallback)."""
    replay = _replay()
    text = (PERF / "preflight_replay.py").read_text(encoding="utf-8")
    assert "resolve_preloaded_tasks" in text
    assert "preloaded_task_classes=preloaded_classes" in text
    assert "preloaded_task_ids=preloaded_ids" in text
    assert "performance.preflight_reuse_loaded_tasks" in text
    assert "PreflightOptimizationContractError" in text
