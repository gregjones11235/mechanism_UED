"""20-check READ-ONLY student compatibility probe (Stage 4, roadmap §二十三).

Entry point (never imported by ``simulator_frontier.__init__``):

    python -m dicode.simulator_frontier.probes.student_compatibility \
        student.profile=<name> student.checkpoint_path=<PATH> \
        [student.expected_params_sha256=<SHA>] [--steps=N] [--out=<DIR>]

Verdict semantics (fail closed):
  * RMT16 family with local artifact -> all 20 checks run; PASS only if ALL pass.
  * checkpoint artifact missing      -> ARTIFACT_HANDOFF_REQUIRED, never PASS.
  * family without an adapter yet    -> ADAPTER_PENDING (identity/inventory
    checks only), never PASS.
Zero parameter updates everywhere; forward checks are smoke only and never a
performance evaluation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PASS, FAIL, BLOCKED = 0, 4, 5

CHECK_NAMES = (
    "profile_loads",
    "identity_hash_stable",
    "override_agreement",
    "checkpoint_artifact_present",
    "driver_source_present_and_sha_gated",
    "frozen_cfg_recovered_from_driver_source",
    "file_sha256_gate",
    "params_sha256_recomputed_gate",
    "manifest_gate",
    "tree_structure_gate",
    "observation_dim_gate",
    "action_count_gate",
    "memory_spec_gate",
    "initial_memory_contract",
    "validate_memory_rejects_bad_memory",
    "deterministic_forward_reproducible",
    "stochastic_forward_seeded_reproducible",
    "zero_update_params_bit_identical",
    "memory_progression_contract",
    "fresh_process_params_recheck",
)
assert len(CHECK_NAMES) == 20

ADAPTER_PENDING_FAMILIES = ("GTRXL128", "TEACHER_REFERENCE", "SLOWGRU")

DEFAULT_DRIVER_SOURCE = (
    "D:/Projects/dicode-codex-director/orchestration/control/_cc2_stage/"
    "train_rmt16_p2replay.py")


def _log(msg: str) -> None:
    print(f"[probe] {msg}", flush=True)


def parse_args(argv):
    driver_source = DEFAULT_DRIVER_SOURCE
    driver_sha = None
    steps = 8
    out_dir = None  # default resolved after repo root is known
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


def _check(name: str, status: str, detail: str = "") -> dict:
    return {"check": name, "status": status, "detail": detail}


def _seeded_obs(obs_dim: int):
    import numpy as np
    rng = np.random.default_rng(20260803)
    return rng.normal(size=obs_dim).astype(np.float32)


def _fresh_process_sha(checkpoint_path: str) -> tuple[int, str | None]:
    repo_src = Path(__file__).resolve().parents[3]  # .../gpu1_aggregation_siege/src
    code = (
        "from dicode.student_adapters.checkpoint_codec import load_cc2_pkl;"
        f"loaded = load_cc2_pkl(r'''{checkpoint_path}''');"
        "print('RESHASH ' + loaded.params_sha256)"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_src) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("JAX_PLATFORMS", "cpu")
    proc = subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, timeout=600)
    sha = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("RESHASH "):
            sha = line.split(" ", 1)[1].strip()
    return proc.returncode, sha


def probe_rmt16(profile, overrides: dict, driver_source: str, driver_sha: str,
                steps: int) -> dict:
    """All 20 checks against the real read-only RMT16 adapter chain."""
    import numpy as np
    from dicode.student_adapters.architectures.rmt16_provenance import (
        load_rmt16_cfg_from_driver_source,
        sha256_lf_file,
        verify_frozen_cfg,
    )
    from dicode.student_adapters.checkpoint_codec import cc2_params_sha256
    from dicode.student_adapters.rmt16_adapter import RMT16StudentAdapter

    checks: dict[str, dict] = {}
    # 1-3 identity/override checks
    h1 = profile.expected_identity().identity_hash()
    h2 = profile.expected_identity().identity_hash()
    checks["profile_loads"] = _check("profile_loads", "PASS",
                                     f"candidate_id={profile.candidate_id}")
    checks["identity_hash_stable"] = _check(
        "identity_hash_stable", "PASS" if h1 == h2 else "FAIL", h1)
    ovr_sha = overrides.get("student.expected_params_sha256")
    ovr_src = overrides.get("student.expected_source_commit")
    agree = ((ovr_sha is None or ovr_sha == profile.params_sha256)
             and (ovr_src is None or ovr_src == profile.source_commit))
    checks["override_agreement"] = _check(
        "override_agreement", "PASS" if agree else "FAIL",
        "overrides agree with the profile" if agree else "override disagrees (fail closed)")

    checkpoint_path = overrides.get("student.checkpoint_path", "")
    # 4-5 artifact presence
    if not checkpoint_path or not Path(checkpoint_path).is_file():
        checks["checkpoint_artifact_present"] = _check(
            "checkpoint_artifact_present", "BLOCKED",
            f"ARTIFACT_HANDOFF_REQUIRED: {checkpoint_path or '<no path given>'}")
        for name in CHECK_NAMES[4:]:
            checks[name] = _check(name, "NOT_RUN", "blocked by missing artifact")
        return {"checks": checks, "status": "ARTIFACT_HANDOFF_REQUIRED"}
    checks["checkpoint_artifact_present"] = _check(
        "checkpoint_artifact_present", "PASS",
        f"{checkpoint_path} ({Path(checkpoint_path).stat().st_size} B)")

    if not Path(driver_source).is_file():
        checks["driver_source_present_and_sha_gated"] = _check(
            "driver_source_present_and_sha_gated", "BLOCKED",
            f"ARTIFACT_HANDOFF_REQUIRED: driver source missing {driver_source}")
        for name in CHECK_NAMES[5:]:
            checks[name] = _check(name, "NOT_RUN", "blocked by missing driver source")
        return {"checks": checks, "status": "ARTIFACT_HANDOFF_REQUIRED"}
    actual_driver_sha = sha256_lf_file(driver_source)
    driver_ok = actual_driver_sha == driver_sha
    checks["driver_source_present_and_sha_gated"] = _check(
        "driver_source_present_and_sha_gated", "PASS" if driver_ok else "FAIL",
        f"LF-SHA {actual_driver_sha[:16]}… expected {driver_sha[:16]}…")

    # 6 frozen cfg recovery (AST-literal, never executed)
    try:
        cfg, _ = load_rmt16_cfg_from_driver_source(driver_source, driver_sha)
        verify_frozen_cfg(cfg)
        checks["frozen_cfg_recovered_from_driver_source"] = _check(
            "frozen_cfg_recovered_from_driver_source", "PASS",
            f"{len(cfg)} Cfg fields recovered")
    except Exception as exc:
        checks["frozen_cfg_recovered_from_driver_source"] = _check(
            "frozen_cfg_recovered_from_driver_source", "FAIL", repr(exc))

    # 7-13 via the adapter's G2-G5 gate chain (single source of truth)
    try:
        adapter = RMT16StudentAdapter(
            profile, driver_source_path=driver_source, expected_driver_sha256=driver_sha)
        loaded = adapter.load_full_state(checkpoint_path, profile.expected_identity())
        gates = loaded["gates"]
        gate_map = {
            "file_sha256_gate": ("G2_file_sha256", loaded["file_sha256"]),
            "params_sha256_recomputed_gate": ("G3_params_sha256", loaded["params_sha256"]),
            "manifest_gate": ("G4_manifest", json.dumps(gates.get("G4_manifest", {}),
                                                         default=str)),
            "tree_structure_gate": ("G5_structure", json.dumps(gates.get("G5_structure", {}),
                                                                default=str)),
        }
        for name, (gk, detail) in gate_map.items():
            checks[name] = _check(name, "PASS" if gk in gates else "FAIL", str(detail)[:300])
        g5 = gates.get("G5_structure", {})
        obs_ok = g5.get("observation_dim") == int(profile.observation_shape[0])
        act_ok = (g5.get("action_count") == int(profile.action_count)
                  and g5.get("craftax_len_action") == int(profile.action_count))
        checks["observation_dim_gate"] = _check(
            "observation_dim_gate", "PASS" if obs_ok else "FAIL",
            f"encoder kernel in-dim {g5.get('observation_dim')} vs "
            f"profile {profile.observation_shape[0]}")
        checks["action_count_gate"] = _check(
            "action_count_gate", "PASS" if act_ok else "FAIL",
            f"head {g5.get('action_count')} / env enum {g5.get('craftax_len_action')} vs "
            f"profile {profile.action_count}")
        mem_ok = adapter.memory_spec().spec_hash() == profile.memory_spec().spec_hash()
        checks["memory_spec_gate"] = _check(
            "memory_spec_gate", "PASS" if mem_ok else "FAIL",
            adapter.memory_spec().spec_hash())
    except Exception as exc:
        for name in CHECK_NAMES[6:13]:
            if name not in checks:
                checks[name] = _check(name, "FAIL", f"gate chain failed: {exc!r}")

    # 14-15 memory contract
    try:
        mem = adapter.initial_memory(2)
        ok14 = adapter.validate_memory(mem, 2)["ok"]
        checks["initial_memory_contract"] = _check(
            "initial_memory_contract", "PASS" if ok14 else "FAIL",
            "shapes/dtypes/mem_idx=window_mem per the CC2 reset convention")
        bad = {k: v for k, v in mem.items()}
        bad["rmt.seg_count"] = np.full((2,), 10 ** 6, dtype=np.int32)  # out of range
        bad_ok = adapter.validate_memory(bad, 2)["ok"]
        missing_ok = adapter.validate_memory({k: v for k, v in mem.items() if k != "memories"}, 2)["ok"]
        checks["validate_memory_rejects_bad_memory"] = _check(
            "validate_memory_rejects_bad_memory",
            "PASS" if (not bad_ok and not missing_ok) else "FAIL",
            "out-of-range seg_count and missing field both rejected")
    except Exception as exc:
        for name in ("initial_memory_contract", "validate_memory_rejects_bad_memory"):
            if name not in checks:
                checks[name] = _check(name, "FAIL", repr(exc))

    # 16-19 forward smoke (zero updates)
    try:
        params = loaded["params"]
        obs_dim = int(profile.observation_shape[0])
        obs_vec = _seeded_obs(obs_dim)
        obs_batch = np.broadcast_to(obs_vec, (2, obs_dim)).copy()
        sha_before = cc2_params_sha256(params)

        m0 = adapter.initial_memory(2)
        d1 = adapter.policy_step(params, obs_batch, m0, None, None, None, True)
        d2 = adapter.policy_step(params, obs_batch, m0, None, None, None, True)
        checks["deterministic_forward_reproducible"] = _check(
            "deterministic_forward_reproducible",
            "PASS" if (np.array_equal(d1["action"], d2["action"])
                       and np.array_equal(d1["logits"], d2["logits"])
                       and bool(np.isfinite(d1["logits"]).all())) else "FAIL")

        s1 = adapter.policy_step(params, obs_batch, m0, None, None,
                                 np.random.default_rng(777), False)
        s2 = adapter.policy_step(params, obs_batch, m0, None, None,
                                 np.random.default_rng(777), False)
        checks["stochastic_forward_seeded_reproducible"] = _check(
            "stochastic_forward_seeded_reproducible",
            "PASS" if np.array_equal(s1["action"], s2["action"]) else "FAIL")

        mem_walk = adapter.initial_memory(2)
        for _ in range(steps):
            out = adapter.policy_step(params, obs_batch, mem_walk, None, None, None, True)
            mem_walk = out["memory"]
        seg_after = int(np.asarray(mem_walk["rmt.seg_count"]).max())
        checks["memory_progression_contract"] = _check(
            "memory_progression_contract",
            "PASS" if seg_after == min(steps, 128) else "FAIL",
            f"seg_count after {steps} steps = {seg_after}")

        sha_after = cc2_params_sha256(params)
        checks["zero_update_params_bit_identical"] = _check(
            "zero_update_params_bit_identical",
            "PASS" if sha_before == sha_after else "FAIL", sha_before)
    except Exception as exc:
        for name in CHECK_NAMES[15:19]:
            if name not in checks:
                checks[name] = _check(name, "FAIL", repr(exc))

    # 20 fresh-process recheck
    try:
        rc, fresh_sha = _fresh_process_sha(checkpoint_path)
        checks["fresh_process_params_recheck"] = _check(
            "fresh_process_params_recheck",
            "PASS" if (rc == 0 and fresh_sha == profile.params_sha256) else "FAIL",
            f"fresh-process sha {str(fresh_sha)[:16]}… rc={rc}")
    except Exception as exc:
        checks["fresh_process_params_recheck"] = _check(
            "fresh_process_params_recheck", "FAIL", repr(exc))

    ordered = [checks.get(name) or _check(name, "NOT_RUN", "internal error: check missing")
               for name in CHECK_NAMES]
    all_pass = all(c["status"] == "PASS" for c in ordered)
    return {"checks": {c["check"]: c for c in ordered},
            "status": "PASS" if all_pass else "FAIL"}


def probe_pending_family(profile, overrides: dict) -> dict:
    """Identity/inventory checks only for families without an adapter yet.

    NEVER yields PASS: absence of an adapter (or of the local artifact) is an
    explicit ARTIFACT_HANDOFF_REQUIRED / ADAPTER_PENDING record, not a result.
    """
    checks: dict[str, dict] = {}
    checks["profile_loads"] = _check("profile_loads", "PASS",
                                     f"candidate_id={profile.candidate_id}")
    h1 = profile.expected_identity().identity_hash()
    checks["identity_hash_stable"] = _check("identity_hash_stable", "PASS", h1)
    ovr_sha = overrides.get("student.expected_params_sha256")
    agree = ovr_sha is None or ovr_sha == profile.params_sha256
    checks["override_agreement"] = _check(
        "override_agreement", "PASS" if agree else "FAIL")
    checkpoint_path = overrides.get("student.checkpoint_path", "")
    artifact = Path(checkpoint_path) if checkpoint_path else None
    # orbax checkpoints are DIRECTORIES; CC2 pkls are files -> existence check
    if artifact is not None and artifact.exists():
        kind = "dir" if artifact.is_dir() else f"file ({artifact.stat().st_size} B)"
        checks["checkpoint_artifact_present"] = _check(
            "checkpoint_artifact_present", "PASS", f"{checkpoint_path} ({kind})")
        status = "ADAPTER_PENDING"
    else:
        checks["checkpoint_artifact_present"] = _check(
            "checkpoint_artifact_present", "BLOCKED",
            "ARTIFACT_HANDOFF_REQUIRED (server-only or missing artifact)")
        status = "ARTIFACT_HANDOFF_REQUIRED"
    for name in CHECK_NAMES[4:]:
        checks[name] = _check(
            name, "NOT_APPLICABLE",
            f"family {profile.architecture_family} has no Stage-4 adapter yet "
            "(read-only probe refuses to fake a PASS)")
    return {"checks": checks, "status": status}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    repo_root = Path(__file__).resolve().parents[5]
    default_out = repo_root / "gpu1_aggregation_siege" / "reports" / \
        "simulator_frontier_foundation" / "probes"
    report: dict = {
        "schema": "simulator_frontier.student_compatibility_probe/v1",
        "disclaimers": [
            "zero parameter updates; read-only probe",
            "forward checks are smoke only, never a performance evaluation",
            "missing artifact or adapter is never disguised as PASS",
        ],
    }
    try:
        rest, driver_source, driver_sha, steps, out_dir = parse_args(argv)
    except Exception as exc:
        report["status"] = "BLOCKED"
        report["reason"] = f"argv parse error: {exc!r}"
        _finish(report, str(default_out))
        return BLOCKED
    out_dir = out_dir or str(default_out)

    try:
        from dicode.student_adapters.architectures.rmt16_provenance import (
            FROZEN_DRIVER_SOURCE_SHA256,
        )
        from dicode.student_adapters.registry import (
            default_profile_dir,
            load_student_profile,
            resolve_runtime_overrides,
        )
    except Exception as exc:
        report["status"] = "BLOCKED"
        report["reason"] = f"BLOCKED_ENVIRONMENT: {exc!r}"
        _finish(report, out_dir)
        return BLOCKED
    driver_sha = driver_sha or FROZEN_DRIVER_SOURCE_SHA256

    try:
        overrides = resolve_runtime_overrides(rest)
    except Exception as exc:
        report["status"] = "FAIL"
        report["reason"] = f"runtime override violation: {exc!r}"
        _finish(report, out_dir)
        return FAIL
    profile_name = overrides.get("student.profile")
    if not profile_name:
        report["status"] = "BLOCKED"
        report["reason"] = "student.profile is required (never defaulted, never guessed)"
        _finish(report, out_dir)
        return BLOCKED

    try:
        profile = load_student_profile(default_profile_dir() / f"{profile_name}.yaml")
    except Exception as exc:
        report["status"] = "FAIL"
        report["reason"] = f"profile load failed (fail closed): {exc!r}"
        _finish(report, out_dir)
        return FAIL

    report["profile_name"] = profile_name
    report["candidate_id"] = profile.candidate_id
    report["architecture_family"] = profile.architecture_family

    started = time.time()
    if profile.architecture_family == "RMT16":
        try:
            outcome = probe_rmt16(profile, overrides, driver_source, driver_sha, steps)
        except Exception as exc:
            report["status"] = "FAIL"
            report["reason"] = f"probe crashed (fail closed): {exc!r}"
            _finish(report, out_dir)
            return FAIL
    else:
        outcome = probe_pending_family(profile, overrides)

    report["checks"] = outcome["checks"]
    report["status"] = outcome["status"]
    report["elapsed_s"] = round(time.time() - started, 2)
    exit_code = PASS if outcome["status"] == "PASS" else (
        FAIL if outcome["status"] == "FAIL" else BLOCKED)
    _finish(report, out_dir)
    n_pass = sum(1 for c in outcome["checks"].values() if c["status"] == "PASS")
    _log(f"candidate={profile.candidate_id} status={outcome['status']} "
         f"checks_pass={n_pass}/20 ({report['elapsed_s']}s)")
    return exit_code


def _finish(report: dict, out_dir: str) -> None:
    try:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        candidate = report.get("candidate_id", "UNKNOWN_CANDIDATE")
        path = out / f"student_compatibility_probe_{candidate}.json"
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        print(f"[probe] report: {path} ({path.stat().st_size} B)", flush=True)
    except Exception as exc:
        print(f"[probe] REPORT_WRITE_FAILED: {exc!r}", flush=True)
    print(f"[probe] status={report.get('status')} reason={report.get('reason', '-')}",
          flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
