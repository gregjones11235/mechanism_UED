"""B1 local tests: preflight profiling phase contract + fixed replay manifest.

All tests here are pure logic (no jax / no craftax / no omegaconf) so they run on
the CPU-only box. The GPU replay itself is `# requires-jax-server`.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PERF = Path(__file__).parents[4] / "experiments" / "performance"
DICODE = Path(__file__).parents[2]
RUN_DICODE = Path(__file__).parents[4] / "experiments" / "training" / "run_dicode.py"


def load(name):
    spec = importlib.util.spec_from_file_location(name, PERF / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EXPECTED_PHASES = (
    "candidate_code_load",
    "candidate_cpu_validation_build",
    "candidate_cpu_validation_compile",
    "candidate_cpu_validation_execute",
    "preflight_task_reload",
    "preflight_eval_build",
    "preflight_eval_lower_compile",
    "preflight_eval_execute",
    "route",
    "archive_update",
    "preflight_wall",
)


def test_b1_phase_name_set_matches_expected():
    replay = load("preflight_replay")
    assert replay.PREFLIGHT_PHASES == EXPECTED_PHASES
    assert len(set(replay.PREFLIGHT_PHASES)) == len(replay.PREFLIGHT_PHASES)
    assert replay.MID_CHECKPOINT_STEP == 2100
    assert replay.ROLLOUT_UPDATES == 40


@pytest.mark.parametrize(("phase", "path"), [
    ("candidate_code_load", DICODE / "task_utils.py"),
    ("candidate_cpu_validation_build", DICODE / "dreaming" / "gen_manager.py"),
    ("candidate_cpu_validation_compile", DICODE / "dreaming" / "gen_manager.py"),
    ("candidate_cpu_validation_execute", DICODE / "dreaming" / "gen_manager.py"),
    ("preflight_task_reload", RUN_DICODE),
    ("preflight_task_reload", DICODE / "evaluation" / "online_evaluation.py"),
    ("preflight_eval_build", DICODE / "ppo_tr.py"),
    ("preflight_eval_lower_compile", DICODE / "ppo_tr.py"),
    ("preflight_eval_execute", DICODE / "ppo_tr.py"),
    ("route", RUN_DICODE),
    ("archive_update", RUN_DICODE),
    ("preflight_wall", RUN_DICODE),
])
def test_b1_instrumentation_site_present(phase, path):
    """Static audit: the exact phase string is emitted at its instrumentation site."""
    assert path.is_file(), f"missing source file: {path}"
    assert phase in path.read_text(encoding="utf-8"), f"{phase} not found in {path}"


def test_b1_instrumentation_default_off_invariants():
    """Default-off contract: sites that change call structure are gated behind
    `if tracker.enabled:`; the rest are pure tracker.span/record calls which are
    no-ops (zero I/O) when profiling is disabled."""
    gen = (DICODE / "dreaming" / "gen_manager.py").read_text(encoding="utf-8")
    ppo = (DICODE / "ppo_tr.py").read_text(encoding="utf-8")
    run = RUN_DICODE.read_text(encoding="utf-8")
    online = (DICODE / "evaluation" / "online_evaluation.py").read_text(encoding="utf-8")
    # The two paths that change the fused call structure (validation split, eval
    # split) must branch on tracker.enabled so the disabled path is byte-identical.
    assert "if tracker.enabled:" in gen
    assert "if tracker.enabled:" in ppo
    # The remaining sites are pure tracker.span / tracker.record calls.
    assert 'tracker.span("preflight_task_reload")' in run
    assert 'tracker.span("preflight_task_reload")' in online
    assert 'tracker.span("route")' in run and 'tracker.span("archive_update")' in run
    assert 'tracker.record("preflight_wall"' in run
    # The single I/O funnel returns immediately when profiling is disabled.
    rt_src = (DICODE / "runtime_analysis.py").read_text(encoding="utf-8")
    assert "if not self.enabled:" in rt_src


def _replay_spec(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    ckpt = tmp_path / "checkpoint_2100"
    ckpt.mkdir()
    (ckpt / "_CHECKPOINT_METADATA").write_text("step 2100")
    (ckpt / "params").write_text("frozen params")
    archive = tmp_path / "archive_snapshot"
    archive.mkdir()
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


def test_replay_manifest_build_validate_and_tamper(tmp_path):
    replay = load("preflight_replay")
    spec = _replay_spec(tmp_path / "base")
    manifest = replay.build_replay_manifest(spec)
    assert manifest["classification"] == "PREFLIGHT_CANDIDATE_REPLAY"
    assert manifest["mid_checkpoint_step"] == 2100
    assert manifest["candidate_ids"] == ["cand_a", "cand_b"]
    assert manifest["rng"] == {
        "cand_a": replay.derive_rng(42, "cand_a", 0),
        "cand_b": replay.derive_rng(42, "cand_b", 0),
    }
    # validate (recompute hashes) passes on a well-formed manifest
    replay.validate_replay_manifest(dict(manifest))

    # every tamper type must fail closed
    out = tmp_path / "m.json"
    replay.write_manifest(manifest, out)
    for mutate in ("checkpoint", "conditioning", "archive", "candidate_code", "manifest"):
        spec2 = _replay_spec(tmp_path / mutate)
        built = replay.build_replay_manifest(spec2)
        target = tmp_path / mutate / "m.json"
        replay.write_manifest(built, target)
        if mutate == "checkpoint":
            (Path(spec2["checkpoint"]) / "extra").write_text("tamper")
        elif mutate == "conditioning":
            np.save(spec2["conditioning_path"], np.ones((3, 67), dtype=np.float32))
        elif mutate == "archive":
            (Path(spec2["archive_snapshot"]) / "extra").write_text("tamper")
        elif mutate == "candidate_code":
            Path(spec2["candidate_codes"]["cand_a"]).write_text("class Env:\n    pass\n# changed")
        else:
            data = json.loads(target.read_text())
            data["manifest_sha256"] = "bad"
            target.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            replay.validate_replay_manifest(json.loads(target.read_text()))


def test_replay_manifest_rejects_bad_specs(tmp_path):
    replay = load("preflight_replay")
    spec = _replay_spec(tmp_path)
    for mutate in ("score_function", "rollout_updates", "candidate_ids", "global_step"):
        bad = dict(spec)
        if mutate == "score_function":
            bad["score_function"] = "bogus"
        elif mutate == "rollout_updates":
            bad["rollout_updates"] = 41
        elif mutate == "candidate_ids":
            bad["candidate_codes"] = dict(spec["candidate_codes"])
            bad["candidate_codes"]["dup"] = bad["candidate_codes"]["cand_a"]
        else:
            bad["global_step"] = 2099
        with pytest.raises(ValueError):
            replay.build_replay_manifest(bad)


def test_replay_rng_derivation_deterministic():
    replay = load("preflight_replay")
    assert replay.derive_rng(42, "cand_a", 0) == replay.derive_rng(42, "cand_a", 0)
    assert replay.derive_rng(42, "cand_a", 0) != replay.derive_rng(43, "cand_a", 0)
    assert replay.derive_rng(42, "cand_a", 0) != replay.derive_rng(42, "cand_b", 0)
    assert all(0 <= x < 2**32 for x in replay.derive_rng(42, "cand_a", 0))
