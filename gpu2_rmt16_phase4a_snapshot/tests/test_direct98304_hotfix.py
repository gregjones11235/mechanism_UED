#!/usr/bin/env python3
"""Phase4A-direct-98304 hotfix — CPU regression tests (director §四). NO TRAINING.

Run on the server under the dicode310 env with:
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" python tests/test_direct98304_hotfix.py

Covers, at minimum, the ten prescribed cases:
  1. formal_vtrace 12/2  -> post-JAX imported binding PASS   (real driver, CPU self-test hook)
  2. smoke 2/2           -> post-JAX imported binding PASS   (both arms)
  3. long98304 48/4      -> post-JAX imported binding PASS   (both arms)
  4. smoke total_updates=3 -> FAIL  (budget drift rejected)
  5. long save_every=2     -> FAIL  (budget drift rejected)
  6. non-budget constant drift -> FAIL (deep_diff is NOT filtered/ignored)
  7. engineering YAML comment-only edit -> file SHA FAIL (ENGINEERING_CONFIG_CONTENT_IDENTITY_MISMATCH)
  8. engineering YAML value edit (CLI synced) -> scientific SHA FAIL
  9. non-empty output directory -> launcher preflight REJECT (RUN_OUTPUT_DIRECTORY_NOT_FRESH)
 10. empty/nonexistent output directory -> launcher preflight PASS

The driver runs are the REAL driver under RMT16_POSTJAX_BINDING_SELFTEST=1, which exits immediately
AFTER the post-JAX binding and BEFORE env build / ckpt load / training (so: no training, CPU only).
"""
import os
import sys
import subprocess
import tempfile
import shutil

# Force CPU BEFORE anything imports jax (subprocesses inherit this environment too).
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.normpath(os.path.join(HERE, ".."))
EXP = os.path.join(SNAP, "runtime", "experiment_src")
FROZEN = os.path.join(SNAP, "runtime", "frozen_modules")
CFG_DIR = os.path.join(SNAP, "configs")
sys.path.insert(0, EXP)
sys.path.insert(0, FROZEN)

import phase4a_v2_runtime_config as RTC          # noqa: E402  (pure)
import phase4a_v2_frozen_spec as FSPEC           # noqa: E402  (pure)
import phase4a_v2_formal_identity as FID         # noqa: E402  (pure)

PYTHON = os.environ.get("PYTHON", sys.executable)
DRIVER = os.path.join(EXP, "train_rmt16_p2replay.py")
# A base-checkpoint path that textually references 17500 (never READ pre-JAX; the self-test hook
# exits before checkpoint load). The real path is used if present, else a synthetic reference.
CKPT17500 = os.environ.get(
    "CKPT17500",
    "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/"
    "base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500")

CONFIG_FOR = {
    ("formal_vtrace", "persistent"): "rmt16_phase4a_v2_persistent.yaml",
    ("formal_vtrace", "reset128"): "rmt16_phase4a_v2_reset128.yaml",
    ("engineering_smoke", "persistent"): "rmt16_phase4a_smoke_persistent.yaml",
    ("engineering_smoke", "reset128"): "rmt16_phase4a_smoke_reset128.yaml",
    ("long_run_98304", "persistent"): "rmt16_phase4a_long98304_persistent.yaml",
    ("long_run_98304", "reset128"): "rmt16_phase4a_long98304_reset128.yaml",
}

_results = []


def check(name, ok, detail=""):
    _results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail else ""), flush=True)


def _config_assignment(run_class, arm):
    cfg_path = os.path.join(CFG_DIR, CONFIG_FOR[(run_class, arm)])
    rec = RTC.load_formal_config(cfg_path)
    ra = rec["config"]["runtime_assignment"]
    return cfg_path, str(ra["gpu_uuid"]), str(ra["out_dir"])


def run_driver_binding(run_class, arm, total_updates, save_every, seed=42, seqlen=129):
    """Run the REAL driver under the CPU self-test hook; return (returncode, combined_output).
    Exits before env build / ckpt load / training, so this never trains."""
    cfg_path, gpu_uuid, out_dir = _config_assignment(run_class, arm)
    run_root = tempfile.mkdtemp(prefix="p4a_hotfix_selftest_")
    out = os.path.join(run_root, *out_dir.replace("\\", "/").split("/"))
    os.makedirs(out, exist_ok=True)
    argv = [
        PYTHON, DRIVER,
        "--carry_mode", arm,
        "--replay_mode", "original_vtrace",
        "--run_class", run_class,
        "--sequence_length", str(seqlen),
        "--ckpt17500", CKPT17500,
        "--out", out,
        "--gpu_uuid", gpu_uuid,
        "--formal_config", cfg_path,
        "--snapshot_root", SNAP,
        "--run_root", run_root,
        "--total_updates", str(total_updates),
        "--seed", str(seed),
        "--save_every", str(save_every),
    ]
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["RMT16_POSTJAX_BINDING_SELFTEST"] = "1"
    env["PYTHONPATH"] = FROZEN + os.pathsep + EXP + os.pathsep + env.get("PYTHONPATH", "")
    try:
        p = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=600)
        return p.returncode, (p.stdout or "") + "\n" + (p.stderr or "")
    finally:
        shutil.rmtree(run_root, ignore_errors=True)


def test_1_formal_pass():
    rc, out = run_driver_binding("formal_vtrace", "persistent", 12, 2)
    check("(1) formal_vtrace 12/2 -> post-JAX binding PASS",
          rc == 0 and "POSTJAX_BINDING_SELFTEST=PASS" in out,
          f"rc={rc}" + ("" if rc == 0 else " tail=" + out.strip().splitlines()[-1][:140]))


def test_2_smoke_pass():
    for arm in ("persistent", "reset128"):
        rc, out = run_driver_binding("engineering_smoke", arm, 2, 2)
        check(f"(2) smoke 2/2 [{arm}] -> post-JAX binding PASS",
              rc == 0 and "POSTJAX_BINDING_SELFTEST=PASS" in out,
              f"rc={rc}" + ("" if rc == 0 else " tail=" + out.strip().splitlines()[-1][:140]))


def test_3_long_pass():
    for arm in ("persistent", "reset128"):
        rc, out = run_driver_binding("long_run_98304", arm, 48, 4)
        check(f"(3) long98304 48/4 [{arm}] -> post-JAX binding PASS",
              rc == 0 and "POSTJAX_BINDING_SELFTEST=PASS" in out,
              f"rc={rc}" + ("" if rc == 0 else " tail=" + out.strip().splitlines()[-1][:140]))


def test_4_smoke_wrong_total_updates_fail():
    # smoke YAML binds total_updates=2; CLI total_updates=3 must be rejected (budget drift).
    rc, out = run_driver_binding("engineering_smoke", "persistent", 3, 2)
    rejected = (rc != 0) and ("MISMATCH" in out)
    check("(4) smoke total_updates=3 -> FAIL (budget drift rejected)", rejected,
          f"rc={rc}")


def test_5_long_wrong_save_every_fail():
    # long YAML binds save_every=4; CLI save_every=2 must be rejected (budget drift).
    rc, out = run_driver_binding("long_run_98304", "persistent", 48, 2)
    rejected = (rc != 0) and ("MISMATCH" in out)
    check("(5) long save_every=2 -> FAIL (budget drift rejected)", rejected,
          f"rc={rc}")


def test_6_nonbudget_drift_fail():
    # The post-JAX comparison must NOT filter/ignore non-budget fields: a drift in ANY constant
    # (here ema_tau, a non-budget PPO/EMA constant) must be reported by deep_diff.
    base = FSPEC.build_kwargs("persistent")
    expected = RTC.build_runtime_scientific_config(**base)
    mutated = dict(base)
    mutated["ema_tau"] = 0.5                      # non-budget constant drift
    imported = RTC.build_runtime_scientific_config(**mutated)
    drift = RTC.deep_diff(RTC.canonical_scientific_config(expected),
                          RTC.canonical_scientific_config(imported))
    paths = {d["path"] for d in drift}
    check("(6) non-budget constant drift (ema_tau) -> FAIL (not filtered/ignored)",
          len(drift) >= 1 and "scientific_config.ema_tau" in paths,
          f"drift_paths={sorted(paths)[:4]}")


def _edited_record(run_class, arm, mutate):
    cfg_path = os.path.join(CFG_DIR, CONFIG_FOR[(run_class, arm)])
    text = open(cfg_path, encoding="utf-8").read()
    tmp = tempfile.mkdtemp(prefix="p4a_hotfix_cfg_")
    edited_path = os.path.join(tmp, "edited.yaml")
    with open(edited_path, "w", encoding="utf-8") as f:
        f.write(mutate(text))
    rec = RTC.load_formal_config(edited_path)
    return tmp, rec


def test_7_comment_edit_file_sha_fail():
    tmp, rec = _edited_record(
        "engineering_smoke", "persistent",
        lambda t: "# comment that changes file bytes only\n" + t)
    try:
        FID.verify_engineering_config_content_identity(rec, "engineering_smoke", "persistent")
        check("(7) engineering YAML comment-only edit -> file SHA FAIL", False, "no raise")
    except ValueError as e:
        check("(7) engineering YAML comment-only edit -> file SHA FAIL",
              "ENGINEERING_CONFIG_CONTENT_IDENTITY_MISMATCH" in str(e)
              and "file_sha256" in str(e), str(e)[:90])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_8_value_edit_scientific_sha_fail():
    # Edit a scientific value AND imagine the CLI synced to it: the scientific_config canonical
    # SHA changes, so the frozen content identity must still FAIL (not merely self-consistent).
    tmp, rec = _edited_record(
        "long_run_98304", "persistent",
        lambda t: t.replace("total_updates: 48", "total_updates: 49"))
    try:
        FID.verify_engineering_config_content_identity(rec, "long_run_98304", "persistent")
        check("(8) engineering YAML value edit (CLI synced) -> scientific SHA FAIL",
              False, "no raise")
    except ValueError as e:
        check("(8) engineering YAML value edit (CLI synced) -> scientific SHA FAIL",
              "ENGINEERING_CONFIG_CONTENT_IDENTITY_MISMATCH" in str(e), str(e)[:90])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _launcher_preflight(launcher, run_root):
    env = dict(os.environ)
    env["RUN_ROOT"] = run_root
    env["PYTHON"] = PYTHON
    env["PYTHONPATH"] = FROZEN + os.pathsep + EXP + os.pathsep + env.get("PYTHONPATH", "")
    p = subprocess.run(["bash", launcher, "--preflight-only"],
                       env=env, capture_output=True, text=True, timeout=120)
    return p.returncode, (p.stdout or "") + "\n" + (p.stderr or "")


def test_9_nonempty_outdir_reject():
    launcher = os.path.join(EXP, "launch_smoke_pair.sh")
    run_root = tempfile.mkdtemp(prefix="p4a_hotfix_outdir_")
    try:
        stale = os.path.join(run_root, "runs", "RMT16-SMOKE-PERSISTENT-4096")
        os.makedirs(stale, exist_ok=True)
        with open(os.path.join(stale, "full_state.pkl"), "w") as f:
            f.write("stale")
        rc, out = _launcher_preflight(launcher, run_root)
        check("(9) non-empty output directory -> launcher REJECT",
              rc != 0 and "RUN_OUTPUT_DIRECTORY_NOT_FRESH" in out
              and "FRESHNESS_GATE=FAIL" in out, f"rc={rc}")
    finally:
        shutil.rmtree(run_root, ignore_errors=True)


def test_10_empty_outdir_pass():
    launcher = os.path.join(EXP, "launch_smoke_pair.sh")
    run_root = tempfile.mkdtemp(prefix="p4a_hotfix_outdir_")
    try:
        rc, out = _launcher_preflight(launcher, run_root)   # runs/ does not exist yet -> fresh
        check("(10) empty/nonexistent output directory -> launcher preflight PASS",
              rc == 0 and "FRESHNESS_GATE=PASS" in out, f"rc={rc}")
    finally:
        shutil.rmtree(run_root, ignore_errors=True)


def main():
    print("Phase4A-direct-98304 hotfix — CPU regression tests (§四); JAX_PLATFORMS="
          + os.environ.get("JAX_PLATFORMS", "?") + " CUDA_VISIBLE_DEVICES="
          + repr(os.environ.get("CUDA_VISIBLE_DEVICES", "?")), flush=True)
    print(f"  snapshot_root={SNAP}", flush=True)
    print(f"  python={PYTHON}", flush=True)
    test_1_formal_pass()
    test_2_smoke_pass()
    test_3_long_pass()
    test_4_smoke_wrong_total_updates_fail()
    test_5_long_wrong_save_every_fail()
    test_6_nonbudget_drift_fail()
    test_7_comment_edit_file_sha_fail()
    test_8_value_edit_scientific_sha_fail()
    test_9_nonempty_outdir_reject()
    test_10_empty_outdir_pass()
    n = len(_results); n_pass = sum(_results)
    print(f"SELFTEST_SUMMARY total={n} pass={n_pass} fail={n - n_pass}", flush=True)
    return 0 if n_pass == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
