#!/usr/bin/env python3
"""R4b read-only Student mount smoke driver (Stage 3).

Mounts one Student candidate (primary: PERSISTENT_RMT16_ORIGINAL_VTRACE_98304)
strictly READ-ONLY:

    registry resolve -> identity gates -> REAL_CHECKPOINT_LOADED gate chain
    -> no-update forward smoke -> fresh-process params-hash recheck
    -> JSON evidence report.

Usage (venv python, PYTHONPATH=<repo>/gpu1_aggregation_siege/src,
JAX_PLATFORMS=cpu):

    python run_student_mount_smoke.py \
        student.profile=rmt16_persistent_98304 \
        student.checkpoint_path=<PATH> \
        [student.expected_params_sha256=<SHA>] \
        [student.expected_source_commit=<SRC>] \
        [--driver-source=<PATH>] [--driver-sha=<SHA>] \
        [--steps=N] [--out=<DIR>]

Exit codes: 0 PASS, 4 FAIL, 5 BLOCKED.

DISCLAIMERS (also recorded in the JSON report):
  * forward smoke is NOT a performance evaluation;
  * the R4a env-side PASS and this R4b checkpoint-side PASS together do NOT
    constitute the R4c joint fresh-process proof;
    COMBINED_FRESH_PROCESS_RESTORE stays false until that combined run is
    executed in one fresh process;
  * contract/fake tests are NOT the real closed loop.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "gpu1_aggregation_siege" / "src"
DEFAULT_OUT_DIR = REPO_ROOT / "gpu1_aggregation_siege" / "reports" / "simulator_frontier_foundation"

# Frozen driver binding (see student_adapters.architectures.rmt16_provenance):
# the staging/deploy copy whose SHA the frozen tier3 checkpoint contract binds.
DEFAULT_DRIVER_SOURCE = (
    "D:/Projects/dicode-codex-director/orchestration/control/_cc2_stage/"
    "train_rmt16_p2replay.py")

PASS, FAIL, BLOCKED = 0, 4, 5


def _log(msg: str) -> None:
    print(f"[mount-smoke] {msg}", flush=True)


def parse_args(argv):
    driver_source = DEFAULT_DRIVER_SOURCE
    driver_sha = None  # resolved from provenance below unless overridden
    steps = 8
    out_dir = str(DEFAULT_OUT_DIR)
    rest = []
    for arg in argv:
        if arg.startswith("--driver-source="):
            driver_source = arg.split("=", 1)[1]
        elif arg.startswith("--driver-sha="):
            driver_sha = arg.split("=", 1)[1]
        elif arg.startswith("--steps="):
            steps = int(arg.split("=", 1)[1])
        elif arg.startswith("--out="):
            out_dir = arg.split("=", 1)[1]
        else:
            rest.append(arg)
    if steps < 1 or steps > 256:
        raise ValueError(f"--steps out of range [1,256]: {steps}")
    return rest, driver_source, driver_sha, steps, out_dir


def environment_info() -> dict:
    import jax
    import numpy

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jax": jax.__version__,
        "numpy": numpy.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
    }
    try:
        import flax
        info["flax"] = flax.__version__
    except Exception:
        info["flax"] = "unavailable"
    try:
        import craftax
        info["craftax"] = getattr(craftax, "__version__", "installed")
    except Exception:
        info["craftax"] = "unavailable"
    return info


def observation_source(obs_dim: int):
    """Prefer a real MiniCraftaxTrain reset observation; fall back to a
    labeled structured-synthetic vector.  Returns (obs, label, info).

    Honest labeling: the trained 8335-dim contract is 8268 MiniCraftaxTrain
    obs + 67 multitask embedding as produced by the TRAINING wrapper; a bare
    env reset only yields the env part, so unless the full 8335 vector is
    obtainable we use the established seeded-normal structured synthetic
    input (non-zero, deterministic) and say so in the report.
    """
    import numpy as np

    info: dict = {"obs_dim": obs_dim}
    try:
        import jax
        from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
        from minicraftax.envs.base import MiniCraftaxTrain
        from minicraftax.tasks.seed_tasks import survive

        # SAME convention as simulator_frontier.craftax_checks.build_core_setup:
        # one EnvParams object for construction and stepping.
        params = EnvParams(max_timesteps=64)
        static_params = StaticEnvParams()
        task = survive.Env(static_params, params)
        env = MiniCraftaxTrain(task, static_env_params=static_params)
        obs0, _state0 = env.reset_env(jax.random.PRNGKey(20260803), params)
        flat = np.asarray(obs0).reshape(-1)
        info["real_craftax_reset_obs_size"] = int(flat.size)
        if flat.size == obs_dim:
            return flat.astype(np.float32), "REAL_CRAFTAX_RESET_OBS", info
    except Exception as exc:
        info["real_craftax_error"] = repr(exc)
    rng = np.random.default_rng(20260803)
    vec = rng.normal(size=obs_dim).astype(np.float32)
    info["note"] = ("structured synthetic (seeded normal, non-zero, deterministic); "
                    "the trained obs is 8268 MiniCraftaxTrain + 67 multitask embedding "
                    "assembled by the training wrapper, which a bare env reset cannot "
                    "reproduce — labeled, never disguised as real")
    return vec, "STRUCTURED_SYNTHETIC_SEEDED_NORMAL", info


def fresh_process_recheck(checkpoint_path: str, expected_sha: str) -> dict:
    """Independent fresh process re-loads the pkl and recomputes the params
    tree hash (R4b cross-process evidence)."""
    code = (
        "from dicode.student_adapters.checkpoint_codec import load_cc2_pkl;"
        f"loaded = load_cc2_pkl(r'''{checkpoint_path}''');"
        "print('RESHASH ' + loaded.params_sha256)"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("JAX_PLATFORMS", "cpu")
    started = time.time()
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=str(REPO_ROOT), env=env,
        capture_output=True, text=True, timeout=600)
    out = {
        "returncode": proc.returncode,
        "elapsed_s": round(time.time() - started, 3),
        "executable": sys.executable,
        "in_process_sha256": expected_sha,
        "fresh_process_sha256": None,
        "match": False,
    }
    for line in (proc.stdout or "").splitlines():
        if line.startswith("RESHASH "):
            out["fresh_process_sha256"] = line.split(" ", 1)[1].strip()
    out["match"] = (proc.returncode == 0
                    and out["fresh_process_sha256"] == expected_sha)
    if not out["match"]:
        out["stderr_tail"] = (proc.stderr or "")[-800:]
    return out


def run_smoke(adapter, params, steps: int, obs_vec, obs_source_label: str) -> dict:
    """No-update forward smoke. ZERO parameter updates; assertions only."""
    import numpy as np
    from dicode.student_adapters.checkpoint_codec import cc2_params_sha256

    obs_dim = adapter.observation_spec().shape[0]
    action_count = adapter.action_spec().count
    sha_before = cc2_params_sha256(params)
    results: dict = {"obs_source": obs_source_label, "obs_dim": obs_dim,
                     "action_count": action_count, "batches": {}}
    for batch in (1, 4):
        mem = adapter.initial_memory(batch)
        obs_batch = np.broadcast_to(obs_vec, (batch, obs_dim)).copy()
        entry: dict = {}
        # deterministic reproducibility
        out1 = adapter.policy_step(params, obs_batch, mem, None, None, None, True)
        out2 = adapter.policy_step(params, obs_batch, mem, None, None, None, True)
        entry["deterministic_reproducible"] = bool(
            np.array_equal(out1["action"], out2["action"])
            and np.array_equal(out1["logits"], out2["logits"]))
        entry["action_shape"] = list(np.asarray(out1["action"]).shape)
        entry["logits_shape"] = list(out1["logits"].shape)
        entry["actions_in_range"] = bool(
            ((np.asarray(out1["action"]) >= 0) & (np.asarray(out1["action"]) < action_count)).all())
        entry["logits_finite"] = bool(np.isfinite(out1["logits"]).all())
        entry["value_finite"] = bool(np.isfinite(np.asarray(out1["value"])).all())
        # stochastic seeded reproducibility
        s1 = adapter.policy_step(params, obs_batch, mem, None, None,
                                 np.random.default_rng(777), False)
        s2 = adapter.policy_step(params, obs_batch, mem, None, None,
                                 np.random.default_rng(777), False)
        entry["stochastic_seeded_reproducible"] = bool(np.array_equal(s1["action"], s2["action"]))
        entry["stochastic_actions_in_range"] = bool(
            ((np.asarray(s1["action"]) >= 0) & (np.asarray(s1["action"]) < action_count)).all())
        # memory progression over `steps` deterministic steps
        mem_walk = adapter.initial_memory(batch)
        for _ in range(steps):
            out_walk = adapter.policy_step(params, obs_batch, mem_walk, None, None, None, True)
            mem_walk = out_walk["memory"]
        seg_count = int(np.asarray(mem_walk["rmt.seg_count"]).max())
        entry["memory_progression"] = {
            "steps": steps,
            "seg_count_after": seg_count,
            "seg_count_increments_correct": seg_count == min(steps, 128),
            "mem_tokens_untouched_before_boundary": bool(
                np.all(np.asarray(mem_walk["rmt.mem_tokens"]) == 0.0)) if steps < 128 else None,
        }
        results["batches"][f"batch_{batch}"] = entry

    # 128-step segment-boundary probe (carry semantics of the mounted arm)
    boundary: dict = {"ran": False}
    if steps >= 1:
        batch = 1
        mem_b = adapter.initial_memory(batch)
        obs_batch = np.broadcast_to(obs_vec, (batch, obs_dim)).copy()
        for _ in range(128):
            out_b = adapter.policy_step(params, obs_batch, mem_b, None, None, None, True)
            mem_b = out_b["memory"]
        tokens = np.asarray(mem_b["rmt.mem_tokens"])
        seg_after_boundary = int(np.asarray(mem_b["rmt.seg_count"])[0])
        boundary = {
            "ran": True,
            "seg_count_after_128_steps": seg_after_boundary,
            "segment_boundary_crossed": seg_after_boundary == 0,
            "mem_tokens_nonzero_after_boundary": bool(np.any(tokens != 0.0)),
            "carry_mode": adapter._carry_mode,  # noqa: SLF001 - report evidence
            "persistent_carry_verified": (
                adapter._carry_mode == "persistent" and bool(np.any(tokens != 0.0))  # noqa: SLF001
                and seg_after_boundary == 0),
        }
    results["segment_boundary_probe"] = boundary

    sha_after = cc2_params_sha256(params)
    results["zero_update_params_bit_identical"] = sha_before == sha_after
    results["params_sha256_before"] = sha_before
    results["params_sha256_after"] = sha_after
    return results


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    started = time.time()
    report: dict = {
        "schema": "simulator_frontier.student_mount_smoke/v1",
        "r4_scope": ("R4b checkpoint-side proof only; R4a env-side PASS + R4b PASS "
                     "!= R4c joint fresh-process proof"),
        "COMBINED_FRESH_PROCESS_RESTORE": False,
        "disclaimers": [
            "forward smoke != performance evaluation",
            "contract/fake tests != real closed loop",
            "zero parameter updates; zero training this round",
            "optimizer/train-rng/policy-memory are ABSENT in the CC2 pkl; the R4c "
            "joint proof is unavailable for those components this round",
        ],
    }
    try:
        rest, driver_source, driver_sha, steps, out_dir = parse_args(argv)
    except Exception as exc:
        report["verdict"] = "BLOCKED"
        report["reason"] = f"argv parse error: {exc!r}"
        _finish(report, str(DEFAULT_OUT_DIR))
        return BLOCKED

    try:
        from dicode.student_adapters.architectures.rmt16_provenance import (
            FROZEN_DRIVER_SOURCE_SHA256,
        )
        from dicode.student_adapters.rmt16_adapter import RMT16StudentAdapter
        from dicode.student_adapters.registry import (
            default_profile_dir,
            load_student_profile,
            resolve_runtime_overrides,
        )
    except Exception as exc:
        report["verdict"] = "BLOCKED"
        report["reason"] = f"BLOCKED_ENVIRONMENT: {exc!r}"
        _finish(report, out_dir)
        return BLOCKED

    driver_sha = driver_sha or FROZEN_DRIVER_SOURCE_SHA256
    try:
        overrides = resolve_runtime_overrides(rest)
    except Exception as exc:
        report["verdict"] = "FAIL"
        report["reason"] = f"runtime override violation: {exc!r}"
        _finish(report, out_dir)
        return FAIL

    profile_name = overrides.get("student.profile")
    checkpoint_path = overrides.get("student.checkpoint_path")
    if not profile_name or not checkpoint_path:
        report["verdict"] = "BLOCKED"
        report["reason"] = ("student.profile and student.checkpoint_path are both "
                            "required (never defaulted, never guessed)")
        _finish(report, out_dir)
        return BLOCKED
    if not Path(checkpoint_path).is_file():
        report["verdict"] = "BLOCKED"
        report["reason"] = f"BLOCKED_ARTIFACT: checkpoint missing: {checkpoint_path}"
        _finish(report, out_dir)
        return BLOCKED
    if not Path(driver_source).is_file():
        report["verdict"] = "BLOCKED"
        report["reason"] = f"BLOCKED_ARTIFACT: driver source missing: {driver_source}"
        _finish(report, out_dir)
        return BLOCKED

    report["profile_name"] = profile_name
    report["checkpoint_path"] = checkpoint_path
    report["driver_source_path"] = driver_source
    report["driver_source_sha256_expected"] = driver_sha

    try:
        profile = load_student_profile(default_profile_dir() / f"{profile_name}.yaml")
    except Exception as exc:
        report["verdict"] = "FAIL"
        report["reason"] = f"profile load failed (fail closed): {exc!r}"
        _finish(report, out_dir)
        return FAIL

    # caller-side assertions (overrides must AGREE with the profile, never replace it)
    expected_sha_ovr = overrides.get("student.expected_params_sha256")
    if expected_sha_ovr and expected_sha_ovr != profile.params_sha256:
        report["verdict"] = "FAIL"
        report["reason"] = ("student.expected_params_sha256 override disagrees with the "
                            "profile (fail closed)")
        _finish(report, out_dir)
        return FAIL
    expected_src_ovr = overrides.get("student.expected_source_commit")
    if expected_src_ovr and expected_src_ovr != profile.source_commit:
        report["verdict"] = "FAIL"
        report["reason"] = ("student.expected_source_commit override disagrees with the "
                            "profile (fail closed)")
        _finish(report, out_dir)
        return FAIL

    report["candidate_id"] = profile.candidate_id
    report["identity"] = {
        "candidate_id": profile.candidate_id,
        "architecture_family": profile.architecture_family,
        "checkpoint_format": profile.checkpoint_format,
        "global_step": profile.global_step,
        "total_env_steps": profile.total_env_steps,
        "params_sha256": profile.params_sha256,
        "source_commit": profile.source_commit,
        "observation_shape": list(profile.observation_shape),
        "action_count": profile.action_count,
        "memory_spec_hash": profile.memory_spec().spec_hash(),
    }

    if profile.architecture_family != "RMT16":
        report["verdict"] = "BLOCKED"
        report["reason"] = (f"architecture_family {profile.architecture_family} has no "
                            "Stage-3 adapter yet (RMT16 only this stage; other families "
                            "= Stage-4 read-only probes)")
        _finish(report, out_dir)
        return BLOCKED

    adapter = RMT16StudentAdapter(
        profile, driver_source_path=driver_source, expected_driver_sha256=driver_sha)

    try:
        loaded = adapter.load_full_state(checkpoint_path, profile.expected_identity())
    except Exception as exc:
        report["verdict"] = "FAIL"
        report["REAL_CHECKPOINT_LOADED"] = False
        report["reason"] = f"REAL_CHECKPOINT_LOADED gate chain failed: {exc!r}"
        _finish(report, out_dir)
        return FAIL

    report["REAL_CHECKPOINT_LOADED"] = True
    report["gates"] = loaded["gates"]
    report["contains_optimizer"] = loaded["contains_optimizer"]
    report["contains_rng"] = loaded["contains_rng"]
    report["contains_policy_memory"] = loaded["contains_policy_memory"]
    report["r4c_joint_proof_status"] = loaded["r4c_joint_proof_status"]
    report["file_sha256"] = loaded["file_sha256"]
    report["params_sha256"] = loaded["params_sha256"]
    report["driver_source_sha256_actual"] = loaded["driver_source_sha256"]
    report["manifest"] = {k: loaded["manifest"].get(k) for k in
                          ("step", "arm", "carry_mode", "replay_mode", "seed")}
    _log(f"REAL_CHECKPOINT_LOADED gates={sorted(loaded['gates'])}")

    # forward smoke
    try:
        obs_vec, obs_label, obs_info = observation_source(int(profile.observation_shape[0]))
        report["observation_source"] = {"label": obs_label, **obs_info}
        smoke = run_smoke(adapter, loaded["params"], steps, obs_vec, obs_label)
    except Exception as exc:
        report["verdict"] = "FAIL"
        report["REAL_FORWARD_SMOKE_PASS"] = False
        report["reason"] = f"forward smoke failed: {exc!r}"
        _finish(report, out_dir)
        return FAIL

    ok_smoke = (
        smoke["zero_update_params_bit_identical"]
        and all(e["deterministic_reproducible"] and e["actions_in_range"]
                and e["logits_finite"] and e["value_finite"]
                and e["stochastic_seeded_reproducible"]
                and e["memory_progression"]["seg_count_increments_correct"]
                for e in smoke["batches"].values())
        and smoke["segment_boundary_probe"]["persistent_carry_verified"]
    )
    report["REAL_FORWARD_SMOKE_PASS"] = bool(ok_smoke)
    report["smoke"] = smoke
    if not ok_smoke:
        report["verdict"] = "FAIL"
        report["reason"] = "forward smoke assertions failed"
        _finish(report, out_dir)
        return FAIL

    # fresh-process recheck
    try:
        recheck = fresh_process_recheck(checkpoint_path, loaded["params_sha256"])
    except Exception as exc:
        report["verdict"] = "FAIL"
        report["reason"] = f"fresh-process recheck crashed: {exc!r}"
        _finish(report, out_dir)
        return FAIL
    report["fresh_process_recheck"] = recheck
    if not recheck["match"]:
        report["verdict"] = "FAIL"
        report["reason"] = "fresh-process params sha mismatch"
        _finish(report, out_dir)
        return FAIL

    report["verdict"] = "PASS"
    report["elapsed_s"] = round(time.time() - started, 2)
    report["environment"] = environment_info()
    _finish(report, out_dir)
    _log(f"PASS REAL_CHECKPOINT_LOADED + REAL_FORWARD_SMOKE_PASS "
         f"({report['elapsed_s']}s) — R4b only, NOT R4c joint proof")
    return PASS


def _finish(report: dict, out_dir: str) -> None:
    try:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        candidate = report.get("candidate_id", "UNKNOWN_CANDIDATE")
        path = out / f"student_compatibility_{candidate}.json"
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=False)
        os.replace(tmp, path)
        size = path.stat().st_size
        assert size < 512 * 1024, f"report too large: {size}"
        print(f"[mount-smoke] report: {path} ({size} B)", flush=True)
    except Exception as exc:
        print(f"[mount-smoke] REPORT_WRITE_FAILED: {exc!r}", flush=True)
    print(f"[mount-smoke] verdict={report.get('verdict')} "
          f"reason={report.get('reason', '-')}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
