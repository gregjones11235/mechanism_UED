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
    # CC2 §二 BASE_GTRXL third arm (engineering smoke / long98304 ONLY; no formal_vtrace profile).
    ("engineering_smoke", "base_gtrxl"): "rmt16_phase4a_smoke_base_gtrxl.yaml",
    ("long_run_98304", "base_gtrxl"): "rmt16_phase4a_long98304_base_gtrxl.yaml",
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


# ============================================================================
# CC2 §二 BASE_GTRXL_ORIGINAL_VTRACE_98304 — third-arm regression (NO TRAINING; CPU self-test hook).
# ============================================================================

def test_2b_base_smoke_pass():
    rc, out = run_driver_binding("engineering_smoke", "base_gtrxl", 2, 2)
    check("(2b) base_gtrxl smoke 2/2 -> post-JAX binding PASS",
          rc == 0 and "POSTJAX_BINDING_SELFTEST=PASS" in out,
          f"rc={rc}" + ("" if rc == 0 else " tail=" + out.strip().splitlines()[-1][:140]))


def test_3b_base_long_pass():
    rc, out = run_driver_binding("long_run_98304", "base_gtrxl", 48, 4)
    check("(3b) base_gtrxl long98304 48/4 -> post-JAX binding PASS",
          rc == 0 and "POSTJAX_BINDING_SELFTEST=PASS" in out,
          f"rc={rc}" + ("" if rc == 0 else " tail=" + out.strip().splitlines()[-1][:140]))


def test_4b_base_smoke_wrong_total_updates_fail():
    # base_gtrxl smoke YAML binds total_updates=2; CLI total_updates=3 must be rejected (budget drift).
    rc, out = run_driver_binding("engineering_smoke", "base_gtrxl", 3, 2)
    rejected = (rc != 0) and ("MISMATCH" in out)
    check("(4b) base_gtrxl smoke total_updates=3 -> FAIL (budget drift rejected)", rejected,
          f"rc={rc}")


def test_7b_base_comment_edit_file_sha_fail():
    tmp, rec = _edited_record(
        "engineering_smoke", "base_gtrxl",
        lambda t: "# comment that changes base_gtrxl file bytes only\n" + t)
    try:
        FID.verify_engineering_config_content_identity(rec, "engineering_smoke", "base_gtrxl")
        check("(7b) base_gtrxl YAML comment-only edit -> file SHA FAIL", False, "no raise")
    except ValueError as e:
        check("(7b) base_gtrxl YAML comment-only edit -> file SHA FAIL",
              "ENGINEERING_CONFIG_CONTENT_IDENTITY_MISMATCH" in str(e)
              and "file_sha256" in str(e), str(e)[:90])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_8b_base_value_edit_scientific_sha_fail():
    tmp, rec = _edited_record(
        "long_run_98304", "base_gtrxl",
        lambda t: t.replace("total_updates: 48", "total_updates: 49"))
    try:
        FID.verify_engineering_config_content_identity(rec, "long_run_98304", "base_gtrxl")
        check("(8b) base_gtrxl YAML value edit (CLI synced) -> scientific SHA FAIL", False, "no raise")
    except ValueError as e:
        check("(8b) base_gtrxl YAML value edit (CLI synced) -> scientific SHA FAIL",
              "ENGINEERING_CONFIG_CONTENT_IDENTITY_MISMATCH" in str(e), str(e)[:90])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _base_launcher_preflight(run_root):
    launcher = os.path.join(EXP, "launch_base_gtrxl.sh")
    return _launcher_preflight(launcher, run_root)


def test_9b_base_nonempty_outdir_reject():
    run_root = tempfile.mkdtemp(prefix="p4a_base_outdir_")
    try:
        stale = os.path.join(run_root, "runs", "BASEGTRXL-SMOKE-4096")
        os.makedirs(stale, exist_ok=True)
        with open(os.path.join(stale, "full_state.pkl"), "w") as f:
            f.write("stale")
        rc, out = _base_launcher_preflight(run_root)
        check("(9b) base_gtrxl non-empty output directory -> launcher REJECT",
              rc != 0 and "RUN_OUTPUT_DIRECTORY_NOT_FRESH" in out and "FRESHNESS_GATE=FAIL" in out,
              f"rc={rc}")
    finally:
        shutil.rmtree(run_root, ignore_errors=True)


def test_10b_base_empty_outdir_pass():
    run_root = tempfile.mkdtemp(prefix="p4a_base_outdir_")
    try:
        rc, out = _base_launcher_preflight(run_root)   # runs/ does not exist yet -> fresh
        check("(10b) base_gtrxl empty/nonexistent output directory -> launcher preflight PASS",
              rc == 0 and "FRESHNESS_GATE=PASS" in out, f"rc={rc}")
    finally:
        shutil.rmtree(run_root, ignore_errors=True)


def test_11_base_read_skip_unit():
    """Numerical proof that carry_mode=base_gtrxl SKIPS the RMT read path (pure GTrXL backbone).

    Real frozen-architecture network on CPU. (a) the shared helper returns None for base_gtrxl;
    (b) base forward == explicit mem_tokens=None (bit-exact); (c) at init the zero gate makes a
    zero-token read a no-op (== backbone); (d) opening the gate proves the read path is LIVE
    (changes the output with non-zero tokens) yet base STILL skips it (== None)."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from network_rmt16 import ActorCriticTransformerRMT16
    import rmt_memory_anchor as RA
    B, WM, NL, E, NH, NT, NS, AD, OD = 2, 128, 2, 256, 8, 16, 128, 6, 16
    net = ActorCriticTransformerRMT16(
        action_dim=AD, activation="relu", hidden_layers=256, encoder_size=E, num_heads=NH,
        qkv_features=256, num_layers=NL, gating=True, gating_bias=2.0, rmt_num_tokens=NT)
    memories = jnp.zeros((B, WM, NL, E))
    mask = jnp.zeros((B, NH, 1, WM + 1), jnp.bool_)
    obs = jnp.zeros((B, OD))
    tok0 = jnp.zeros((B, NT, E))
    seg = jnp.zeros((B, NS, E))
    variables = net.init(jax.random.PRNGKey(0), memories, obs, mask, tok0, seg,
                         method=net.init_all)
    params = variables["params"]
    apply = RA.make_apply_eval_rmt(net)
    rmt_st = {"mem_tokens": tok0, "seg_buf": seg, "seg_count": jnp.zeros((B,), jnp.int32)}
    nz = {"mem_tokens": jnp.ones((B, NT, E)), "seg_buf": seg,
          "seg_count": jnp.zeros((B,), jnp.int32)}
    # (a) shared helper: base -> None; persistent/reset128 -> the carried tokens
    check("(11a) entering_read_tokens(base_gtrxl) is None",
          RA.entering_read_tokens(rmt_st, "base_gtrxl") is None)
    check("(11a) entering_read_tokens(persistent/reset128) is mem_tokens",
          RA.entering_read_tokens(rmt_st, "persistent") is rmt_st["mem_tokens"]
          and RA.entering_read_tokens(rmt_st, "reset128") is rmt_st["mem_tokens"])
    # (b) base forward == explicit None forward (read path skipped, bit-exact)
    lg_base = np.asarray(apply(params, memories, obs, mask,
                               RA.entering_read_tokens(rmt_st, "base_gtrxl"))[0])
    lg_none = np.asarray(apply(params, memories, obs, mask, None)[0])
    check("(11b) base_gtrxl logits == mem_tokens=None (pure backbone, bit-exact)",
          np.array_equal(lg_base, lg_none))
    # (c) at init (gate=0) reading ZERO tokens is a no-op -> equals the backbone
    lg_read0 = np.asarray(apply(params, memories, obs, mask, tok0)[0])
    check("(11c) gate=0 read-with-zero-tokens == backbone (zero-init no-op)",
          np.allclose(lg_base, lg_read0))
    # (d) open the gate: the read path is LIVE but base STILL skips it
    params_open = dict(params)
    params_open["rmt_gate"] = jnp.ones((1,))
    lg_base_open = np.asarray(apply(params_open, memories, obs, mask,
                                    RA.entering_read_tokens(nz, "base_gtrxl"))[0])
    lg_read_open = np.asarray(apply(params_open, memories, obs, mask, nz["mem_tokens"])[0])
    lg_none_open = np.asarray(apply(params_open, memories, obs, mask, None)[0])
    check("(11d) gate!=0: base (skipped) != read-with-nonzero-tokens (read path is real)",
          not np.allclose(lg_base_open, lg_read_open, atol=1e-6))
    check("(11d) gate!=0: base still == explicit None (skip independent of gate)",
          np.array_equal(lg_base_open, lg_none_open))


def test_12_pr_protection_regression():
    """Appending base_gtrxl MUST NOT alter the frozen P/R identity (CC2 §二 P/R protection)."""
    # (a) FROZEN_SPEC_SHA256 unchanged (VALID_CARRY_MODES is NOT hashed; FROZEN_SPEC dict untouched).
    check("(12a) FROZEN_SPEC_SHA256 unchanged (== 722b9971...)",
          FSPEC.FROZEN_SPEC_SHA256
          == "722b99716215c6393d25c40c3baba93f73cb4fc7e84974de343ea9dbbc769bf8",
          FSPEC.FROZEN_SPEC_SHA256[:16])
    # (b) build_kwargs(base_gtrxl) differs from persistent ONLY by carry_mode.
    kw_b = FSPEC.build_kwargs("base_gtrxl")
    kw_p = FSPEC.build_kwargs("persistent")
    diff = {k for k in set(kw_b) | set(kw_p) if kw_b.get(k) != kw_p.get(k)}
    check("(12b) build_kwargs(base_gtrxl) differs from persistent ONLY in carry_mode",
          diff == {"carry_mode"}, f"diff={sorted(diff)}")
    # (c) the 4 frozen P/R ENGINEERING_CONFIG_IDENTITIES SHAs are UNCHANGED.
    expected_pr = {
        ("engineering_smoke", "persistent"): (
            "c35627fc09ae9062e52add7ff0befd2d752361030fcf8bad6d1aef01dc000202",
            "02207e60d2dc8dc509fd236e8869fd944ff46e9c82f7b85901e505cbc38eef6a"),
        ("engineering_smoke", "reset128"): (
            "b07aad0c91e4c88c0ef4918d21163fcbbaff40887154b68a5b4653ecb8d546ac",
            "50bbcf49074b2c2260522e0a1f3dae346db9db762e8bd43e34d660edbf0d53e4"),
        ("long_run_98304", "persistent"): (
            "992445eefa30042dc043ed8c5403568e016c6db3f416eaa0564b12a01ba9109b",
            "1dffdf09cb741c6a3f933071f1d83e58a4ec40c49dbad4fa8c4c616c3fb03092"),
        ("long_run_98304", "reset128"): (
            "923940cf2a87b41149bc6275e27741fd45aea5f548cc7d464c912e10921f1e92",
            "140a535c786f02c0a2bb0124629f04558bd821fe80c6d5939a95083efe412693"),
    }
    pr_ok = True
    for k, (fsha, ssha) in expected_pr.items():
        ident = FID.ENGINEERING_CONFIG_IDENTITIES[k]
        if ident["file_sha256"] != fsha or ident["scientific_config_sha256"] != ssha:
            pr_ok = False
            print(f"    [{k}] P/R engineering identity drift!", flush=True)
    check("(12c) 4 P/R ENGINEERING_CONFIG_IDENTITIES SHAs unchanged", pr_ok)
    # (d) the two base_gtrxl identities are present (additive).
    check("(12d) base_gtrxl engineering identities present",
          ("engineering_smoke", "base_gtrxl") in FID.ENGINEERING_CONFIG_IDENTITIES
          and ("long_run_98304", "base_gtrxl") in FID.ENGINEERING_CONFIG_IDENTITIES)


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
    # ---- CC2 §二 BASE_GTRXL third arm (+ P/R protection regression) ----
    test_2b_base_smoke_pass()
    test_3b_base_long_pass()
    test_4b_base_smoke_wrong_total_updates_fail()
    test_7b_base_comment_edit_file_sha_fail()
    test_8b_base_value_edit_scientific_sha_fail()
    test_9b_base_nonempty_outdir_reject()
    test_10b_base_empty_outdir_pass()
    test_11_base_read_skip_unit()
    test_12_pr_protection_regression()
    n = len(_results); n_pass = sum(_results)
    print(f"SELFTEST_SUMMARY total={n} pass={n_pass} fail={n - n_pass}", flush=True)
    return 0 if n_pass == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
