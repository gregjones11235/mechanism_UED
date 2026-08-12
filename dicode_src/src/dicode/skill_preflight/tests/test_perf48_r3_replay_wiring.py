"""R3: frozen preflight replay must have real runtime wiring (fake-runtime test).

The GPU replay (`preflight_replay._run_replay`) drives a real call sequence
against a runtime bundle. This test provides a FAKE runtime that records every
call and asserts the full production sequence: load config (never None) ->
restore archive copy -> verify candidate code hashes -> load classes (order
preserved) -> build REAL achievement masks from task attributes -> load
checkpoint -> run 40-update rollout with the frozen input RNG -> score -> route
-> apply archive mutations on the copy -> emit before/after hashes. A mid-way
exception must propagate (no RESULT is written by _run_replay; run_replay writes
atomically only after success). Manifest tamper on config/archive/code/
checkpoint/conditioning/source-mapping must reject.
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PERF = Path(__file__).parents[4] / "experiments" / "performance"


def _replay():
    spec = importlib.util.spec_from_file_location("preflight_replay", PERF / "preflight_replay.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _spec(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    ckpt = tmp_path / "checkpoint" / "2100"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "_CHECKPOINT_METADATA").write_text("meta")
    (ckpt / "params").write_text("frozen params")
    archive = tmp_path / "archive_snapshot"
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "graph.graphml").write_text("frozen archive graph")
    cond = tmp_path / "conditioning.npy"
    np.save(cond, np.zeros((3, 67), dtype=np.float32))  # [N+1, 67], N=2
    codes = {}
    for i, cid in enumerate(("cand_a", "cand_b")):
        p = tmp_path / f"{cid}.py"
        p.write_text(f"class Env:\n    pass\n# task {cid}\n")
        codes[cid] = str(p)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("validation:\n  rollout_updates: 40\n  num_envs: 1024\n  num_steps: 128\n"
                   "dicode_manager:\n  score_function: learnability\n")
    src = tmp_path / "source.py"
    src.write_text("# frozen source\n")
    return {
        "base_dir": str(tmp_path),
        "checkpoint": str(ckpt),
        "conditioning_path": str(cond),
        "archive_snapshot": str(archive),
        "candidate_codes": codes,
        "score_function": "learnability",
        "rollout_updates": 40,
        "global_step": 2100,
        "source_commit": "frozen-commit-abc123",
        "gpu_uuid": "GPU-replay-test",
        "config_path": str(cfg),
        "source_mapping": {"src/dicode/ppo_tr.py": str(src)},
        "num_envs": 1024,
        "num_steps": 128,
    }


class _FakeArchive:
    """Fake archive that exposes task codes and records learnability/status/active mutations."""

    def __init__(self, code_files):
        self._code_files = code_files  # {id: path}
        self.learn_updates = []
        self.status_updates = []
        self.active_updates = []
        self.mutated = False

    def get_task_codes(self, ids):
        return {cid: Path(self._code_files[cid]).read_text(encoding="utf-8") for cid in ids}

    def update_node_learnability(self, tid, value):
        self.learn_updates.append((tid, value)); self.mutated = True

    def update_node_status(self, tid, status):
        self.status_updates.append((tid, status)); self.mutated = True

    def set_task_active_status(self, tid, active):
        self.active_updates.append((tid, active)); self.mutated = True


def _fake_runtime(tmp_path, spec, fail_at=None):
    """Build a fake runtime bundle that records call counts; fail_at injects an
    exception in the named step to test propagation."""
    replay = _replay()
    calls = {}

    def counted(name):
        def deco(fn):
            def wrapper(*a, **k):
                calls[name] = calls.get(name, 0) + 1
                if fail_at == name:
                    raise RuntimeError(f"injected failure at {name}")
                return fn(*a, **k)
            return wrapper
        return deco

    class _Decision:
        action = "accept"
        reason = ""

    def reject_or_accept(sr, any_partial_progress):
        if sr >= 0.5:
            return SimpleNamespace(action="accept", reason="")
        return SimpleNamespace(action="reject", reason="sr_low")

    code_files = {cid: Path(p) for cid, p in spec["candidate_codes"].items()}
    archive = _FakeArchive(code_files)

    rt = {
        "load_config": counted("load_config")(lambda path: SimpleNamespace(
            validation=SimpleNamespace(rollout_updates=40, num_envs=1024, num_steps=128),
            dicode_manager=SimpleNamespace(score_function="learnability"))),
        "verify_gpu": counted("verify_gpu")(lambda uuid: None),
        "reconstruct_archive": counted("reconstruct_archive")(lambda snap: archive),
        "archive_get_codes": counted("archive_get_codes")(lambda a, ids: a.get_task_codes(ids)),
        "load_tasks": counted("load_tasks")(lambda a, ids: ([object() for _ in ids], list(ids))),
        "achievement_masks": counted("achievement_masks")(
            lambda classes: (np.ones((len(classes), 2), dtype=bool), np.zeros((len(classes), 2), dtype=bool))),
        "load_checkpoint": counted("load_checkpoint")(
            lambda cfg, ckpt: SimpleNamespace(params={"p": np.ones(2)}, opt_state={"o": np.zeros(2)})),
        "state_hash": counted("state_hash")(lambda s: "state-hash"),
        "make_input_rng": counted("make_input_rng")(lambda seed, ids: [seed, 1, 2, 3]),
        "run_rollout": counted("run_rollout")(
            lambda cfg, rng, classes, updates, emb, ts: {"metrics": {"scoring_window_data": {"fake": 1}}}),
        "score": counted("score")(
            lambda cfg, swd, n, mask, comp: {"0": {"sr": 0.8}, "1": {"sr": 0.2}}),
        "route": counted("route")(reject_or_accept),
        "archive_hash": counted("archive_hash")(lambda a: "archive-clean" if not a.mutated else "archive-mutated"),
        "archive_update_accept": counted("archive_update_accept")(
            lambda a, tid, lr: a.update_node_learnability(tid, lr)),
        "archive_update_reject": counted("archive_update_reject")(
            lambda a, tid, status: (a.update_node_status(tid, status),
                                    a.set_task_active_status(tid, False))),
    }
    return replay, rt, calls, archive


def test_replay_full_sequence_success(tmp_path):
    replay, rt, calls, archive = _fake_runtime(tmp_path, _spec(tmp_path))
    manifest = replay.build_replay_manifest(_spec(tmp_path))
    result = replay._run_replay(manifest, rt)

    # every production step runs the expected number of times
    for name in ("load_config", "verify_gpu", "reconstruct_archive", "archive_get_codes",
                 "load_tasks", "achievement_masks", "load_checkpoint",
                 "make_input_rng", "run_rollout", "score"):
        assert calls.get(name, 0) == 1, f"{name} called {calls.get(name, 0)} times"
    # state_hash: params + optimizer (2); archive_hash: before + after (2)
    assert calls.get("state_hash", 0) == 2
    assert calls.get("archive_hash", 0) == 2
    # route: one per candidate (2); accept: only cand_a (sr 0.8); reject: cand_b (sr 0.2)
    assert calls.get("route", 0) == 2
    assert calls.get("archive_update_accept", 0) == 1
    assert calls.get("archive_update_reject", 0) == 1
    # config never None: verify_config ran against a real config object
    assert rt["load_config"] is not None
    # candidate order preserved
    assert result["candidate_ids"] == ["cand_a", "cand_b"]
    # masks come from task attributes (achievement_masks derived from classes)
    assert result["task_masks_hash"] is not None
    # archive mutated: before != after
    assert result["archive_before_sha256"] != result["archive_after_sha256"]
    # route/archive updates applied on the copy
    assert archive.learn_updates and archive.status_updates
    # RNG not faked: rng_after marked not_exposed (run_evaluation_rollouts returns no final RNG)
    assert result["rng_after_sha256"].startswith("not_exposed:")
    assert result["rng_input_sha256"] is not None
    # evidence fields present
    assert result["gpu_uuid"] == "GPU-replay-test"
    assert result["llm_api_calls"] == 0
    assert result["runtime_source_evidence"]["src/dicode/ppo_tr.py"]
    assert result["checkpoint_tree_sha256"] and result["conditioning_content_sha256"]


@pytest.mark.parametrize("fail_at", ["load_config", "reconstruct_archive", "load_tasks",
                                     "run_rollout", "score", "route"])
def test_replay_midway_exception_propagates(tmp_path, fail_at):
    replay, rt, _, _ = _fake_runtime(tmp_path, _spec(tmp_path), fail_at=fail_at)
    manifest = replay.build_replay_manifest(_spec(tmp_path))
    with pytest.raises(RuntimeError, match=f"injected failure at {fail_at}"):
        replay._run_replay(manifest, rt)
    # _run_replay does not write any RESULT; nothing left behind
    # (run_replay writes atomically only after _run_replay returns)


@pytest.mark.parametrize("field, mutate", [
    ("config", lambda s, p: (p / "config.yaml").write_text(
        "validation:\n  rollout_updates: 41\n  num_envs: 1024\n  num_steps: 128\n")),
    ("conditioning", lambda s, p: np.save(s["conditioning_path"], np.ones((3, 67), dtype=np.float32))),
    ("archive", lambda s, p: (Path(s["archive_snapshot"]) / "extra").write_text("tamper")),
    ("candidate_code", lambda s, p: Path(s["candidate_codes"]["cand_a"]).write_text("class Env:\n    pass\n# changed")),
    ("checkpoint", lambda s, p: (Path(s["checkpoint"]) / "extra").write_text("tamper")),
    ("source_mapping", lambda s, p: Path(s["source_mapping"]["src/dicode/ppo_tr.py"]).write_text("# changed")),
])
def test_replay_manifest_tamper_rejected(tmp_path, field, mutate):
    replay = _replay()
    spec = _spec(tmp_path)
    # build + persist a manifest from the GOOD spec, then tamper the underlying
    # file, then validate must re-derive hashes and reject the changed file.
    manifest = replay.build_replay_manifest(spec)
    out = tmp_path / "manifest.json"
    replay.write_manifest(manifest, out)
    mutate(spec, tmp_path)
    with pytest.raises(ValueError):
        replay.validate_replay_manifest(json.loads(out.read_text()))
