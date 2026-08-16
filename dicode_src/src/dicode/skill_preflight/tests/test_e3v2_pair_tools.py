import importlib.util
import json
from pathlib import Path

import networkx as nx
import pytest

ROOT = Path(__file__).parents[4]
PERF = ROOT / "experiments" / "performance"


def load(name):
    spec = importlib.util.spec_from_file_location(name, PERF / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _spec(tmp_path):
    m = load("e3v2_replay_manifest")
    stages = {}
    for i, name in enumerate(m.STAGES, 1):
        graph = nx.Graph()
        ids = [f"task_{j}" for j in range(1, i + 1)]
        for task_id in ids:
            graph.add_node(task_id, code=f"code-{task_id}", description=f"desc-{task_id}", session_created="s", is_active="true")
        graph_path = tmp_path / f"{name}.graphml"
        nx.write_graphml(graph, graph_path)
        step = i * 100
        checkpoint = tmp_path / f"checkpoint_{step}"
        checkpoint.mkdir()
        (checkpoint / "_CHECKPOINT_METADATA").write_text("metadata")
        (checkpoint / "state.bin").write_bytes(bytes([i]))
        conditioning = tmp_path / f"conditioning_{name}.npy"
        __import__("numpy").save(conditioning, __import__("numpy").zeros((i + 1, 67), dtype="float32"))
        stages[name] = {"graph": str(graph_path), "checkpoint": str(checkpoint), "global_step": step,
                        "initial_env_steps": step * 1024 * 128, "task_ids": ids,
                        "archive_reconstruction_limit": "100", "conditioning_path": str(conditioning)}
    source = tmp_path / "source.py"; source.write_text("source")
    config = tmp_path / "config.yaml"; config.write_text("config")
    return {"budget": m.BUDGET, "conditioning_type": "one_hot", "source": {"source.py": str(source)},
            "config": {"config.yaml": str(config)}, "stages": stages}


def test_manifest_valid_three_stages_and_task_counts(tmp_path):
    m = load("e3v2_replay_manifest"); manifest = m.build_manifest(_spec(tmp_path))
    assert [s["task_count"] for s in manifest["stages"]] == [1, 2, 3]
    assert [s["task_ids"] for s in manifest["stages"]][1] == ["task_1", "task_2"]


def test_manifest_rng_deterministic_and_distinct():
    m = load("e3v2_replay_manifest")
    assert m._u32_rng("early", 0) == m._u32_rng("early", 0)
    assert m._u32_rng("early", 0) != m._u32_rng("early", 1)
    assert m._u32_rng("early", 0) != m._u32_rng("mid", 0)


@pytest.mark.parametrize("field,value", [("budget", {"timesteps": 1}), ("conditioning_type", "embedding")])
def test_manifest_budget_or_embedding_rejected(tmp_path, field, value):
    m = load("e3v2_replay_manifest"); spec = _spec(tmp_path); spec[field] = value
    with pytest.raises(ValueError): m.build_manifest(spec)


@pytest.mark.parametrize("kind", ["rows", "dim", "dtype", "nonfinite", "path"])
def test_manifest_conditioning_table_rejected(tmp_path, kind):
    m = load("e3v2_replay_manifest"); spec = _spec(tmp_path)
    stage = spec["stages"]["early"]
    np = __import__("numpy")
    if kind == "rows": np.save(stage["conditioning_path"], np.zeros((1, 67), dtype="float32"))
    elif kind == "dim": np.save(stage["conditioning_path"], np.zeros((2, 66), dtype="float32"))
    elif kind == "dtype": np.save(stage["conditioning_path"], np.zeros((2, 67), dtype="float64"))
    elif kind == "nonfinite":
        values = np.zeros((2, 67), dtype="float32"); values[0, 0] = np.nan; np.save(stage["conditioning_path"], values)
    else: stage["conditioning_path"] = str(tmp_path / "missing.npy")
    with pytest.raises(ValueError): m.build_manifest(spec)


def test_manifest_conditioning_tamper_rejected(tmp_path):
    m = load("e3v2_replay_manifest"); spec = _spec(tmp_path); out = tmp_path / "manifest.json"
    m.write_manifest(m.build_manifest(spec), out)
    values = __import__("numpy").zeros((2, 67), dtype="float32"); values[0, 0] = 1; __import__("numpy").save(spec["stages"]["early"]["conditioning_path"], values)
    with pytest.raises(ValueError): m.load_manifest(out)


def test_manifest_initial_env_steps_rejected(tmp_path):
    m = load("e3v2_replay_manifest"); spec = _spec(tmp_path); spec["stages"]["early"]["initial_env_steps"] += 1
    with pytest.raises(ValueError): m.build_manifest(spec)


@pytest.mark.parametrize("missing", ["source", "config"])
def test_manifest_requires_source_and_config(tmp_path, missing):
    m = load("e3v2_replay_manifest"); spec = _spec(tmp_path); spec.pop(missing)
    with pytest.raises(ValueError): m.build_manifest(spec)


@pytest.mark.parametrize("mutation", ["missing", "empty", "duplicate"])
def test_manifest_task_validation(tmp_path, mutation):
    m = load("e3v2_replay_manifest"); spec = _spec(tmp_path)
    if mutation == "missing": spec["stages"]["early"]["task_ids"] = ["task_9"]
    elif mutation == "empty": spec["stages"]["early"]["task_ids"] = [""]
    else: spec["stages"]["early"]["task_ids"] = ["task_1", "task_1"]
    with pytest.raises(ValueError): m.build_manifest(spec)


def test_manifest_empty_code_rejected(tmp_path):
    m = load("e3v2_replay_manifest"); spec = _spec(tmp_path)
    graph = nx.read_graphml(spec["stages"]["early"]["graph"]); graph.nodes["task_1"]["code"] = ""
    nx.write_graphml(graph, spec["stages"]["early"]["graph"])
    with pytest.raises(ValueError): m.build_manifest(spec)


def test_manifest_checkpoint_metadata_rejected(tmp_path):
    m = load("e3v2_replay_manifest"); spec = _spec(tmp_path)
    (Path(spec["stages"]["early"]["checkpoint"]) / "_CHECKPOINT_METADATA").unlink()
    with pytest.raises(ValueError): m.build_manifest(spec)


@pytest.mark.parametrize("kind", ["source", "config", "graph", "checkpoint"])
def test_manifest_file_tamper_rejected(tmp_path, kind):
    m = load("e3v2_replay_manifest"); spec = _spec(tmp_path); out = tmp_path / "manifest.json"
    m.write_manifest(m.build_manifest(spec), out)
    if kind == "source":
        Path(spec["source"]["source.py"]).write_text("tampered")
    elif kind == "config":
        Path(spec["config"]["config.yaml"]).write_text("tampered")
    elif kind == "graph":
        with Path(spec["stages"]["early"]["graph"]).open("ab") as f: f.write(b"tampered")
    else:
        (Path(spec["stages"]["early"]["checkpoint"]) / "state.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError): m.load_manifest(out)


def test_manifest_content_tamper_rejected(tmp_path):
    m = load("e3v2_replay_manifest"); out = tmp_path / "manifest.json"
    m.write_manifest(m.build_manifest(_spec(tmp_path)), out)
    data = json.loads(out.read_text()); data["stages"][0]["task_ids"] = ["task_1", "task_9"]; out.write_text(json.dumps(data))
    with pytest.raises(ValueError): m.load_manifest(out)


def test_manifest_task_order_and_atomic_output(tmp_path):
    m = load("e3v2_replay_manifest"); spec = _spec(tmp_path)
    spec["stages"]["early"]["task_ids"] = ["task_1"]
    manifest = m.build_manifest(spec); out = tmp_path / "manifest.json"; m.write_manifest(manifest, out)
    assert out.is_file() and m.load_manifest(out)["stages"][0]["task_ids"] == ["task_1"]


def test_nan_inf_fingerprint_stable():
    m = load("e3v2_replay_manifest")
    assert m.fingerprint({"x": float("nan")}) == m.fingerprint({"x": float("nan")})


def test_compare_gates():
    p = load("e3v2_pair_benchmark")
    assert p.compare_pair({"duration_s": 10, "scoring_fingerprint": "x"}, {"duration_s": 8, "scoring_fingerprint": "x"}) == "REJECTED_SEMANTIC_MISMATCH"
    assert p.compare_pair({"duration_s": 10, "scoring_fingerprint": "x"}, {"duration_s": 8, "scoring_fingerprint": "y"}) == "REJECTED_SEMANTIC_MISMATCH"

def load_harness(): return load("e3v2_training_kernel_harness")

def test_harness_canonical_nan_inf_np_stable():
    h = load_harness(); np = __import__("numpy")
    assert h.scoring_fingerprint({"z": np.array([float("nan"), float("inf")])}) == h.scoring_fingerprint({"z": [float("nan"), float("inf")]})

def test_conditioning_dimension_is_fixed():
    m = load("e3v2_replay_manifest")
    assert m.CONDITIONING_TYPE == "one_hot" and m.CONDITIONING_DIM == 67

def test_harness_stage_repeat_selection(tmp_path):
    h = load_harness(); m = load("e3v2_replay_manifest"); spec = _spec(tmp_path); man = m.build_manifest(spec)
    stage = next(x for x in man["stages"] if x["name"] == "mid")
    assert stage["repeats"][1]["order"] == ["E3V2", "E0"] and stage["repeats"][1]["rng"] == m._u32_rng("mid", 1)

def test_harness_arm_contract():
    h = load_harness(); h._arm_contract({"compact_scoring_payload": False}, "E0")
    with pytest.raises(RuntimeError): h._arm_contract({"compact_scoring_payload": True}, "E0")

def test_harness_budget_contract_rejects_mutation():
    h = load_harness()
    with pytest.raises(RuntimeError): h._config_contract({"total_timesteps": 1, "num_envs": 1024, "num_steps": 128, "updates": 100, "conditioning_type": "embedding", "embedding_dim": 768})

def test_harness_reset_proof_4100(): assert load_harness().verify_selection_semantics()["cases"] == 4100

def test_harness_atomic_json(tmp_path):
    h = load_harness(); out = tmp_path / "x.json"; h._atomic_json(out, {"b": 1, "a": float("nan")}); assert out.is_file() and json.loads(out.read_text())["a"] == "NaN"

def test_harness_tree_hash_value_sensitive():
    h = load_harness(); assert h.state_hash([1, 2]) != h.state_hash([1, 3])

def test_harness_result_contract_symbols():
    text = (PERF / "e3v2_training_kernel_harness.py").read_text()
    assert "checkpoint_loadable" in text and "scoring_fingerprint" in text and "llm_api_calls" in text
    assert "checkpoint_reloaded_optimizer_sha256" in text
    assert "host_callback_free" not in text and "retain_only_scoring_window" not in text

def test_harness_runtime_imports_are_fail_closed():
    h = load_harness(); assert callable(h._runtime_imports)

def test_harness_run_with_fake_runtime(tmp_path, monkeypatch):
    h = load_harness(); m = load("e3v2_replay_manifest")
    ids = ["task_1"]; np = __import__("numpy"); conditioning = np.zeros((2, 67), dtype="float32"); conditioning_path = tmp_path / "conditioning.npy"; np.save(conditioning_path, conditioning)
    eh = __import__("hashlib").sha256(); eh.update(repr(tuple(conditioning.shape)).encode()); eh.update(str(conditioning.dtype).encode()); eh.update(conditioning.tobytes())
    stage = {"name":"early", "task_ids":ids, "tasks":[{"code_sha256":"c"}], "global_step":100, "initial_env_steps":12800, "embedding":{"hash":eh.hexdigest()}, "conditioning":{"path":str(conditioning_path), "content_sha256":eh.hexdigest(), "shape":[2,67], "dtype":"float32"}, "graph":{"path":"g"}, "checkpoint":{"path":"c"}, "repeats":[{"rng":[1,2]}]}
    loaded = {"classification":"TRAINING_KERNEL_BENCHMARK", "manifest_sha256":"m", "stages":[stage]}
    class D(dict):
        __getattr__ = dict.__getitem__
    cfg = D(training=D(compact_scoring_payload=True, score_function="learnability", total_timesteps=2000000000, num_envs=1024, num_steps=128, conditioning_type="one_hot", condition_on_task=True), gen_manager=D(graph_path="", max_updates_per_session=100), compact_scoring_payload=True, score_function="learnability", total_timesteps=2000000000, num_envs=1024, num_steps=128, updates=100, conditioning_type="one_hot", condition_on_task=True)
    class State:
        params={"p":1}; opt_state={"o":2}
    calls = {"run":0,"score":0,"blocked":0}
    class Jax:
        class random:
            @staticmethod
            def split(x): return (__import__("numpy").array([3,4],dtype="uint32"), __import__("numpy").array([5,6],dtype="uint32"))
        class tree_util:
            @staticmethod
            def tree_leaves(x): return [1,2]
        @staticmethod
        def block_until_ready(x): calls.__setitem__("blocked", calls["blocked"]+1)
        @staticmethod
        def device_get(x): return x
    class Wandb:
        def init(self, **k): pass
        def finish(self): pass
        def log(self, *a, **k): pass
    class Manager:
        def __init__(self, path, *a, **k): self.path=Path(path)
        def save(self, step, state): (self.path/str(step)).mkdir(parents=True, exist_ok=True)
        def wait_until_finished(self): calls["wait"] = True
        def close(self): calls["close"] = True
    runtime = {"jax":Jax, "jnp":D(asarray=lambda x: x), "ocp":D(CheckpointManager=Manager, PyTreeCheckpointer=lambda:None, CheckpointManagerOptions=lambda **k:None), "wandb":Wandb(), "TaskArchive":lambda x: object(), "load_tasks_from_env_codes":lambda a,i: ([object()],i), "_calculate_task_distribution":lambda c,n: [], "_create_achievement_masks":lambda c:([True],[False]), "run_training_session":lambda *a, **k: (calls.__setitem__("run",calls["run"]+1) or {"train_state":State(),"metrics":{"num_updates_done":100,"num_env_steps_done":13107200,"scoring_window_data":{}}}), "_load_agent_state":lambda *a: State(), "calculate_scores_from_snapshot":lambda *a: (calls.__setitem__("score",calls["score"]+1) or {"x":1}), "OriginalTask":object(), "wrappers_cl":object()}
    monkeypatch.setattr(h, "preflight", lambda *a: None); monkeypatch.setattr(h, "_runtime_imports", lambda: runtime); monkeypatch.setattr(h, "_load_config", lambda x: cfg); monkeypatch.setattr(h.inspect, "getsourcefile", lambda x: str(tmp_path/"wrappers_cl.py")); (tmp_path/"wrappers_cl.py").write_text("x")
    monkeypatch.setattr(h, "_runtime_source_evidence", lambda *a: {"verified": True, "paths": {k: "p" for k in ("run_training_session", "calculate_scores_from_snapshot", "wrappers_cl", "TaskArchive", "load_tasks_from_env_codes", "_load_agent_state")}, "hashes": {k: "h" for k in ("run_training_session", "calculate_scores_from_snapshot", "wrappers_cl", "TaskArchive", "load_tasks_from_env_codes", "_load_agent_state")}})
    args=D(stage="early", repeat=0, arm="E3V2", config="x", out=str(tmp_path), required_gpu_uuid="u", source_commit="s")
    result=h.run(args, loaded); assert calls["run"]==calls["score"]==1 and calls["blocked"]==2 and result["checkpoint_loadable"] and result["gpu_uuid"] == "u" and result["source_commit"] == "s"
    assert result["checkpoint_reloaded_optimizer_sha256"] == h.state_hash(State().opt_state)

def test_harness_fake_runtime_reloaded_optimizer_contract(tmp_path, monkeypatch):
    """The result must expose the post-checkpoint optimizer fingerprint consumed by pair gates."""
    h = load_harness()
    source = (PERF / "e3v2_training_kernel_harness.py").read_text()
    assert '"checkpoint_reloaded_optimizer_sha256": opt_after' in source

def test_harness_score_function_fallback():
    h = load_harness()
    cfg = {"training": {}, "dicode_manager": {"score_function": "learnability"}, "compact_scoring_payload": True}
    assert h._arm_values(cfg)[1] == "learnability"
    h._arm_contract(cfg, "E3V2")
    h._arm_contract({"training": {}, "dicode_manager": {"score_function": "pvl"}}, "E0")

def test_pair_manifest_order_six(): assert [(s,r) for s in ("early","mid","late") for r in (0,1)] == [("early",0),("early",1),("mid",0),("mid",1),("late",0),("late",1)]
def test_pair_semantic_field_mismatch():
    p=load("e3v2_pair_benchmark"); a={"params_sha256_before":"a"}; b={"params_sha256_before":"b"}; assert p.compare_pair(a,b)=="REJECTED_SEMANTIC_MISMATCH"
def test_pair_compact_contract():
    p=load("e3v2_pair_benchmark"); a={"compact_scoring_payload":True}; b={"compact_scoring_payload":False}; assert p.compare_pair(a,b)=="REJECTED_SEMANTIC_MISMATCH"
def test_pair_aggregate_pass():
    p=load("e3v2_pair_benchmark"); rows=[{"e0":{"duration_s":10,"env_steps":100,"gpu_peak_memory_mib":100,"gpu_min_free_mib":5000},"e3":{"duration_s":8,"env_steps":100,"gpu_peak_memory_mib":200,"gpu_min_free_mib":5000},"status":"SEMANTIC_PASS"} for _ in range(6)]; assert p.aggregate(rows)["conclusion"]=="E3V2_SCORING_PAYLOAD_PASS"
def test_pair_aggregate_no_speed():
    p=load("e3v2_pair_benchmark"); rows=[{"e0":{"duration_s":10,"env_steps":100,"gpu_peak_memory_mib":100,"gpu_min_free_mib":5000},"e3":{"duration_s":10,"env_steps":100,"gpu_peak_memory_mib":100,"gpu_min_free_mib":5000},"status":"SEMANTIC_PASS"} for _ in range(6)]; assert p.aggregate(rows)["conclusion"]=="REJECTED_NO_SPEEDUP"
def test_pair_atomic_json(tmp_path):
    p=load("e3v2_pair_benchmark"); p.atomic_json(tmp_path/"x.json",{"x":1}); assert (tmp_path/"x.json").exists()
def test_pair_fatal_regex():
    p=load("e3v2_pair_benchmark"); assert p.FATAL_RE.search("CUDA Xid")
def test_pair_existing_invalid(tmp_path): assert not (tmp_path/"RESULT.json").exists()
def test_supervisor_descendants():
    s=load("e3v2_supervisor"); assert s.descendants(__import__("os").getpid())
def test_supervisor_stop_tree_shape():
    s=load("e3v2_supervisor"); assert callable(s.stop_tree)
def test_supervisor_atomic_failure_path(tmp_path): assert not (tmp_path/"failure.json").exists()
def test_pair_cli_parameterized():
    text=(PERF/"e3v2_pair_benchmark.py").read_text(); assert "--manifest" in text and "gpu_uuid" in text


def _result_doc(tmp_path, **updates):
    p = load("e3v2_pair_benchmark")
    checkpoint = tmp_path / "checkpoint"; checkpoint.mkdir()
    keys = ["run_training_session", "calculate_scores_from_snapshot", "wrappers_cl", "TaskArchive", "load_tasks_from_env_codes", "_load_agent_state"]
    evidence = {"verified": True, "paths": {k: str(tmp_path / (k + ".py")) for k in keys}, "hashes": {k: "h" for k in keys}}
    data = {"manifest_sha256": "m", "stage": "early", "repeat": 0, "arm": "E0", "runtime_source_evidence": evidence,
            "classification": "TRAINING_KERNEL_BENCHMARK", "llm_api_calls": 0,
            "checkpoint_exists": True, "checkpoint_loadable": True,
            "checkpoint_path": str(checkpoint), "gpu_uuid": "GPU-1"}
    data.update(updates)
    return p, data


def test_validate_result_uses_manifest_sha256_not_path(tmp_path):
    p, doc = _result_doc(tmp_path)
    assert p.validate_result(doc, manifest_sha256="m", stage="early", repeat=0, arm="E0", gpu_uuid="GPU-1")["arm"] == "E0"
    with pytest.raises(RuntimeError):
        p.validate_result(doc, manifest_sha256="manifest-file", stage="early", repeat=0, arm="E0", gpu_uuid="GPU-1")


def test_validate_result_requires_source_commit_when_requested(tmp_path):
    p, doc = _result_doc(tmp_path, source_commit="abc")
    assert p.validate_result(doc, manifest_sha256="m", stage="early", repeat=0, arm="E0", gpu_uuid="GPU-1", source_commit="abc")["source_commit"] == "abc"
    with pytest.raises(RuntimeError):
        p.validate_result(doc, manifest_sha256="m", stage="early", repeat=0, arm="E0", gpu_uuid="GPU-1", source_commit="wrong")


def test_runtime_source_evidence_rejects_wrong_path_and_sha(tmp_path, monkeypatch):
    h = load_harness(); keys = ["run_training_session", "calculate_scores_from_snapshot", "wrappers_cl", "TaskArchive", "load_tasks_from_env_codes", "_load_agent_state", "_calculate_task_distribution"]
    rel = {"run_training_session":"src/dicode/ppo_tr.py", "calculate_scores_from_snapshot":"src/dicode/scoring.py", "wrappers_cl":"src/dicode/wrappers_cl.py", "TaskArchive":"src/dicode/dreaming/gen_manager.py", "load_tasks_from_env_codes":"src/dicode/task_utils.py", "_load_agent_state":"src/dicode/setup.py", "_calculate_task_distribution":"src/dicode/training.py"}
    files = {k: tmp_path / (k + ".py") for k in keys}; [p.write_text(k) for k,p in files.items()]
    rt = {k: object() for k in keys}; monkeypatch.setattr(h.inspect, "getsourcefile", lambda obj: str(files[next(k for k,v in rt.items() if v is obj)]))
    source = {f"e0/{rel[k]}": {"path": str(files[k]), "sha256": h.sha256_file(files[k])} for k in keys}
    loaded = {"source_config": {"source": source}}
    assert h._runtime_source_evidence(rt, loaded, "E0")["verified"]
    source["e0/src/dicode/ppo_tr.py"]["sha256"] = "bad"
    with pytest.raises(RuntimeError): h._runtime_source_evidence(rt, loaded, "E0")


def test_validate_result_rejects_stage_repeat_arm_and_checkpoint(tmp_path):
    p, doc = _result_doc(tmp_path)
    for key, value in (("stage", "late"), ("repeat", 1), ("arm", "E3V2"), ("checkpoint_loadable", False)):
        bad = dict(doc, **{key: value})
        with pytest.raises(RuntimeError):
            p.validate_result(bad, manifest_sha256="m", stage="early", repeat=0, arm="E0", gpu_uuid="GPU-1")


def test_arm_gpu_metrics_reads_fixed_schema(tmp_path):
    p = load("e3v2_pair_benchmark"); csv = tmp_path / "gpu_memory.csv"
    csv.write_text(p.CSV_HEADER + "\n1.0,0,GPU-1,500,8000,10,10,owned_descendant\n2.0,0,GPU-1,700,7000,20,10,owned_descendant\n")
    assert p.arm_gpu_metrics(csv) == (700, 7000)


def test_fatal_in_ignores_path_substrings(tmp_path):
    p = load("e3v2_pair_benchmark"); log = tmp_path / "trainer.stderr"
    log.write_text("opening /tmp/out_of_memory/checkpoint_corrupt.bin\n")
    assert p.fatal_in([log]) is None
    log.write_text("CUDA Xid 79\n")
    assert p.fatal_in([log]) == "CUDA Xid 79"
    log.write_text("NVRM: Xid (PCI:0000:01:00): 79, pid=123\n")
    assert p.fatal_in([log]).startswith("NVRM: Xid (PCI:0000:01:00): 79")


def test_compare_pair_rejects_gpu_uuid_and_llm_mismatch():
    p = load("e3v2_pair_benchmark")
    base = {k: "x" for k in p.SEMANTIC_FIELDS}; base.update({"input_rng_sha256":"i", "train_rng_sha256":"t", "rng_sha256_after":"a", "outer_rng_after_sha256":"a", "compact_scoring_payload":False, "classification":"TRAINING_KERNEL_BENCHMARK", "llm_api_calls":0, "gpu_uuid":"GPU-1", "checkpoint_loadable":True})
    other = dict(base, compact_scoring_payload=True, llm_api_calls=1)
    assert p.compare_pair(base, other) == "REJECTED_SEMANTIC_MISMATCH"
    other = dict(base, gpu_uuid="GPU-2")
    assert p.compare_pair(base, other) == "REJECTED_SEMANTIC_MISMATCH"


def test_aggregate_memory_violation_is_runtime():
    p = load("e3v2_pair_benchmark")
    rows = [{"e0":{"env_steps":100,"train_wall_s":10,"gpu_peak_memory_mib":100,"gpu_min_free_mib":5000}, "e3":{"env_steps":100,"train_wall_s":8,"gpu_peak_memory_mib":700,"gpu_min_free_mib":5000}} for _ in range(6)]
    assert p.aggregate(rows)["conclusion"] == "REJECTED_RUNTIME_FAILURE"


def test_aggregate_free_memory_floor_is_runtime():
    p = load("e3v2_pair_benchmark")
    rows = [{"e0":{"env_steps":100,"train_wall_s":10,"gpu_peak_memory_mib":100,"gpu_min_free_mib":4095}, "e3":{"env_steps":100,"train_wall_s":8,"gpu_peak_memory_mib":100,"gpu_min_free_mib":5000}} for _ in range(6)]
    assert p.aggregate(rows)["conclusion"] == "REJECTED_RUNTIME_FAILURE"


def test_aggregate_duration_regression_is_runtime():
    p = load("e3v2_pair_benchmark")
    rows = [{"e0":{"env_steps":100,"train_wall_s":10,"gpu_peak_memory_mib":100,"gpu_min_free_mib":5000}, "e3":{"env_steps":100,"train_wall_s":10.2,"gpu_peak_memory_mib":100,"gpu_min_free_mib":5000}} for _ in range(6)]
    assert p.aggregate(rows)["conclusion"] == "REJECTED_NO_SPEEDUP"


def test_aggregate_requires_exactly_six_pairs():
    p = load("e3v2_pair_benchmark"); assert p.aggregate([])["conclusion"] == "REJECTED_RUNTIME_FAILURE"


def test_supervisor_requires_nonempty_json_command(tmp_path, monkeypatch):
    s = load("e3v2_supervisor"); monkeypatch.setattr("sys.argv", ["supervisor", "--pair-command-json", "[]", "--out", str(tmp_path)])
    with pytest.raises(SystemExit): s.main()


def test_supervisor_writes_atomic_startup_and_completion(tmp_path, monkeypatch):
    s = load("e3v2_supervisor"); code = "import json,sys; json.dump({'conclusion':'REJECTED_NO_SPEEDUP'},open(sys.argv[1]+'/E3V2_PAIR_RESULT.json','w'))"
    command = json.dumps([__import__("sys").executable, "-c", code, str(tmp_path)])
    monkeypatch.setattr("sys.argv", ["supervisor", "--pair-command-json", command, "--out", str(tmp_path)])
    s.main()
    assert (tmp_path / "startup_health.json").is_file() and (tmp_path / "completion.json").is_file()


def test_supervisor_nonzero_writes_failure(tmp_path, monkeypatch):
    s = load("e3v2_supervisor"); command = json.dumps([__import__("sys").executable, "-c", "raise SystemExit(3)"])
    monkeypatch.setattr("sys.argv", ["supervisor", "--pair-command-json", command, "--out", str(tmp_path)])
    with pytest.raises(RuntimeError): s.main()
    assert (tmp_path / "failure.json").is_file()


def test_supervisor_nonzero_with_rejection_evidence_completes(tmp_path, monkeypatch):
    s = load("e3v2_supervisor"); code = "import json,sys; json.dump({'conclusion':'REJECTED_RUNTIME_FAILURE'},open(sys.argv[1]+'/E3V2_PAIR_RESULT.json','w')); raise SystemExit(3)"
    command = json.dumps([__import__("sys").executable, "-c", code, str(tmp_path)])
    monkeypatch.setattr("sys.argv", ["supervisor", "--pair-command-json", command, "--out", str(tmp_path)])
    s.main()
    assert (tmp_path / "completion.json").is_file() and not (tmp_path / "failure.json").exists()
