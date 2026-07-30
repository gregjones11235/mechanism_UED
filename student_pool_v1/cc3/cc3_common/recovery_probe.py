#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CC3 student-pool checkpoint recovery probe (READ-ONLY identity verification).

Task: CC3_OWN_SLOWGRU_AND_EVENTMEM_STUDENTS_END_TO_END, section 三.
Flow per candidate:
  1. registry 已知路径存在性;
  2. 完整文件 SHA256 重算并核对;
  3. 该家族唯一正确 loader (训练驱动同款 sys.path / pickle 布局) 反序列化;
  4. params SHA 重算 —— 与训练驱动 `_params_sha` 完全一致 (tree_leaves 顺序,
     np.ascontiguousarray(np.asarray(v)).tobytes() 串联);
  5. 核对 manifest 内嵌 params_sha256 / opt_state_leaf_hash / longstate_leaf_hash;
  6. 核对训练步数、seed、opt_step、obs shape、memory 布局、code SHA、params 有限性。
Classification: RECOVERED / NOT_FOUND / IDENTITY_CONFLICT / CHECKPOINT_CORRUPTED /
                CONSTRUCTOR_CONFIG_MISSING.

Hard rules honored: 不依据历史性能表虚构可用性 (一切以重算 SHA 为准);
不做无边界目录扫描 (只访问 registry 登记的固定路径); 只读, 不写任何 checkpoint;
CPU-only (JAX_PLATFORMS=cpu) —— 身份核验不触碰 GPU。
任一身份门禁失败 → status != RECOVERED, 调用方必须停止汇报。
"""
import os

os.environ["JAX_PLATFORMS"] = "cpu"          # identity work only; never touch GPU here
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import hashlib
import json
import pickle
import platform
import sys
import traceback

import numpy as np

V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB"
V7_SRC = V7 + "/src"

GPU2_UUID = "GPU-8df11537-ab79-722d-606f-411966196c4c"

# Canonical frozen protocol (must equal pkl config; from FROZEN trainer docstring)
FROZEN_CFG = dict(
    num_envs=16, num_steps=128, window_mem=128, window_grad=64, num_heads=8,
    num_layers=2, embed_size=256, qkv_features=256, hidden_layers=256,
    gating=True, gating_bias=2.0, activation="relu", gamma=0.999, gae_lambda=0.8,
    clip_eps=0.2, vf_coef=0.5, ent_coef=0.002, max_grad_norm=1.0, lr=2e-5,
    anneal_lr=False, num_minibatches=2, update_epochs=1, optimistic_reset_ratio=16,
    condition_on_task=True,
)
OBS_DIM = 8335
SLOW_INTERVAL = 32
SLOW_DIM = 256
STEPS_PER_UPDATE = 2048
MASTER_SEED = 42

REQUIRED_PKL_KEYS = {
    "params", "opt_state", "opt_step", "env_state", "memories", "memories_mask",
    "memories_mask_idx", "obs", "done", "true_done", "longstate",
    "step_env_currentloop", "update_step", "rng", "global_step", "update_count",
    "config", "code_sha256", "manifest",
}

CANDIDATES = {
    "SLOWGRU_RESET128_LONGRUN_98304": dict(
        family="SLOWGRU",
        arm="LC_SLOWGRU_RESET128_LONGRUN",
        carry_mode="RESET128",
        arm_src="/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_slowgru_reset128_longrun/src",
        ckpt="/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_slowgru_reset128_longrun/train/ckpt/98304/full_state.pkl",
        expected_global_step=98304,
        expected_update_step=48,
        expected_opt_step=96,               # 48 updates x 2 minibatches (driver table)
        expected_file_sha="2c065fa88bcc8cfcb193deda6ef599522238b99bf7151f5eeab0b70e4420f2de",
        expected_params_sha="9d92c5b9e2e2148b2375c59f7f595d53b95f924d62436ebdccf8bf9ea3d59247",
        expected_network_src_sha="b265210597d003218e303ef458ff697b5c9c6a14bfccca098ccce8014bf3eb0b",
        expected_launcher_src_sha="ab32f362b97e10fe4ec3e8947a1ab90892542d2e5b8c10d0b343a4f86a1a517a",
        expected_teacher_init_sha="d4e85af58b7f87d689fadea12eec70c852fa098a09f5ea8907448684b3bf60f5",
        # longrun driver writes carry_mode into the pkl manifest
        expected_manifest_carry_mode="RESET128",
        driver_file="run_slowgru_reset128_longrun.py",
        expect_clear_block=True,
    ),
    "SLOWGRU_PERSISTENT_24576": dict(
        family="SLOWGRU",
        arm="LC_SLOWGRU",
        carry_mode="PERSISTENT",
        arm_src="/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_lc_slowgru/src",
        ckpt="/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_lc_slowgru/train_24576/ckpt/24576/full_state.pkl",
        expected_global_step=24576,
        expected_update_step=12,
        expected_opt_step=24,               # 12 updates x 2 minibatches (driver table)
        expected_file_sha="d4e7008da7d2a78a7765b32379704719165288015d031856850ad7c8a0e7495e",
        expected_params_sha="1bd4fbfe91ab4da44c274ef20f372e04bf6a7e8367869e39e0b65f044c85e9f2",
        expected_network_src_sha="b265210597d003218e303ef458ff697b5c9c6a14bfccca098ccce8014bf3eb0b",
        expected_launcher_src_sha="86bb12c4b7591b0a41057c131cf97848eb0c4914f00d938746fd9a6c1c1d135a",
        expected_teacher_init_sha="d4e85af58b7f87d689fadea12eec70c852fa098a09f5ea8907448684b3bf60f5",
        # ORIGINAL LC_SLOWGRU driver predates the manifest carry_mode field -> key absent (None).
        # Persistent semantics are pinned by launcher SHA + absence of the boundary-clear block
        # + live smoke carry verification (never by a performance table).
        expected_manifest_carry_mode=None,
        driver_file="run_slowgru_24576.py",
        expect_clear_block=False,
    ),
}

# the Reset128 clear block as written in the longrun driver; absent in the persistent driver
CLEAR_BLOCK_MARKER = "_ls_cleared = init_longstate(config.num_envs)"
CARRY_MARKER = "longstate_previous = runner_state[8]"


# ---------------------------------------------------------------- hashing (driver-exact)
def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def params_sha_packed(packed):
    """SHA over the stored np leaves, in flatten order. `_pack` stored
    [np.asarray(l) for l in tree_leaves(params)], so this is byte-identical to the
    driver's _params_sha(params)."""
    leaves, _treedef = packed
    h = hashlib.sha256()
    for v in leaves:
        h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    return h.hexdigest()


def leaf_hash_packed(packed):
    """Driver `_leaf_hash` over stored leaves (shape-prefixed)."""
    leaves, _treedef = packed
    h = hashlib.sha256()
    for l in leaves:
        a = np.asarray(l)
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def leaves_finite(packed):
    leaves, _ = packed
    return bool(all(np.all(np.isfinite(np.asarray(v))) for v in leaves
                    if np.asarray(v).dtype.kind in "fi"))


# ---------------------------------------------------------------- probe
def probe(candidate_id):
    spec = CANDIDATES[candidate_id]
    checks = []            # list of (name, passed, detail)
    recovered_facts = {}

    def ck(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)))
        return ok

    # ---- 1. registry 已知路径存在性 (定向, 无扫描) ----
    if not os.path.isfile(spec["ckpt"]):
        ck("checkpoint_path_exists", False, spec["ckpt"])
        return _result(candidate_id, spec, checks, "NOT_FOUND", recovered_facts)
    size = os.path.getsize(spec["ckpt"])
    ck("checkpoint_path_exists", True, "%s (%d bytes)" % (spec["ckpt"], size))
    recovered_facts["checkpoint_bytes"] = size

    # ---- 2. 完整文件 SHA 重算 ----
    file_sha = sha_file(spec["ckpt"])
    recovered_facts["file_sha256_recomputed"] = file_sha
    if not ck("file_sha256_match_registry", file_sha == spec["expected_file_sha"],
              "recomputed=%s expected=%s" % (file_sha, spec["expected_file_sha"])):
        return _result(candidate_id, spec, checks, "CHECKPOINT_CORRUPTED", recovered_facts)

    # ---- 3. 家族唯一正确 loader: 训练驱动同款 sys.path 后反序列化 ----
    for p in (spec["arm_src"], V7_SRC, V7):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        with open(spec["ckpt"], "rb") as f:
            rd = pickle.load(f)
    except Exception as e:
        ck("pickle_load", False, repr(e)[:400])
        return _result(candidate_id, spec, checks, "CHECKPOINT_CORRUPTED", recovered_facts)
    ck("pickle_load", True, "keys=%d" % len(rd))

    missing = sorted(REQUIRED_PKL_KEYS - set(rd.keys()))
    if not ck("pkl_key_layout", not missing, "missing=%s" % missing):
        return _result(candidate_id, spec, checks, "CHECKPOINT_CORRUPTED", recovered_facts)

    # ---- 4. params SHA 重算 (驱动算法) 并与 registry / 内嵌 manifest 三方核对 ----
    try:
        params_sha = params_sha_packed(rd["params"])
    except Exception as e:
        ck("params_unpack", False, repr(e)[:400])
        return _result(candidate_id, spec, checks, "CHECKPOINT_CORRUPTED", recovered_facts)
    recovered_facts["params_sha256_recomputed"] = params_sha
    manifest = rd.get("manifest", {})
    embedded_params_sha = manifest.get("params_sha256")
    recovered_facts["params_sha256_embedded_manifest"] = embedded_params_sha
    ck("params_sha_match_registry", params_sha == spec["expected_params_sha"],
       "recomputed=%s expected=%s" % (params_sha, spec["expected_params_sha"]))
    ck("params_sha_match_embedded_manifest", params_sha == embedded_params_sha,
       "recomputed=%s embedded=%s" % (params_sha, embedded_params_sha))

    # ---- 5. 身份/构造门禁 ----
    ck("manifest_arm", manifest.get("arm") == spec["arm"],
       "arm=%s expected=%s" % (manifest.get("arm"), spec["arm"]))
    ck("manifest_carry_mode", manifest.get("carry_mode") == spec["expected_manifest_carry_mode"],
       "embedded=%r expected=%r (candidate carry_mode=%s)" % (
           manifest.get("carry_mode"), spec["expected_manifest_carry_mode"], spec["carry_mode"]))
    ck("manifest_rng_seed", int(manifest.get("rng_seed", -1)) == MASTER_SEED,
       "seed=%s" % manifest.get("rng_seed"))
    ck("manifest_gpu_uuid", manifest.get("gpu_uuid") == GPU2_UUID,
       "gpu=%s" % manifest.get("gpu_uuid"))
    ck("manifest_slow_interval", int(manifest.get("slow_interval", -1)) == SLOW_INTERVAL,
       "slow_interval=%s" % manifest.get("slow_interval"))
    ck("manifest_slow_dim", int(manifest.get("slow_dim", -1)) == SLOW_DIM,
       "slow_dim=%s" % manifest.get("slow_dim"))
    ck("manifest_teacher_init_sha",
       manifest.get("teacher_init_sha256") == spec["expected_teacher_init_sha"],
       "teacher=%s" % str(manifest.get("teacher_init_sha256"))[:16])

    # 步数 / opt 一致性 (global = update*2048; opt = update*2 minibatches)
    gstep = int(rd.get("global_step", -1))
    ustep = int(rd.get("update_step", -1))
    ucount = int(rd.get("update_count", -1))
    ostep = int(rd.get("opt_step", -1))
    recovered_facts.update(global_step=gstep, update_step=ustep,
                           update_count=ucount, opt_step=ostep,
                           step_env_currentloop=int(rd.get("step_env_currentloop", -1)))
    ck("global_step_expected", gstep == spec["expected_global_step"], "global_step=%d" % gstep)
    ck("update_step_expected", ustep == spec["expected_update_step"], "update_step=%d" % ustep)
    ck("update_count_eq_update_step", ucount == ustep, "count=%d step=%d" % (ucount, ustep))
    ck("global_eq_update_x_2048", gstep == ustep * STEPS_PER_UPDATE,
       "%d vs %d*2048" % (gstep, ustep))
    ck("opt_step_expected", ostep == spec["expected_opt_step"], "opt_step=%d" % ostep)
    ck("step_env_currentloop_zero_at_save_node", int(rd.get("step_env_currentloop", -1)) == 0,
       "loop=%s" % rd.get("step_env_currentloop"))

    # frozen config 核对
    cfg = rd.get("config", {})
    cfg_mismatch = {k: (cfg.get(k), v) for k, v in FROZEN_CFG.items() if cfg.get(k) != v}
    if not ck("frozen_config_match", not cfg_mismatch, str(cfg_mismatch)[:300]):
        return _result(candidate_id, spec, checks, "CONSTRUCTOR_CONFIG_MISSING", recovered_facts)

    # code SHA: 内嵌记录 vs registry; 再与服务器当前 src 文件重算三方核对
    code = rd.get("code_sha256", {})
    ck("code_network_sha_embedded", code.get("network") == spec["expected_network_src_sha"],
       "network=%s" % str(code.get("network"))[:16])
    ck("code_launcher_sha_embedded", code.get("launcher") == spec["expected_launcher_src_sha"],
       "launcher=%s" % str(code.get("launcher"))[:16])
    net_path = os.path.join(spec["arm_src"], "slowgru_network.py")
    if os.path.isfile(net_path):
        net_disk_sha = sha_file(net_path)
        recovered_facts["network_src_sha256_on_disk"] = net_disk_sha
        ck("network_src_on_disk_sha_match", net_disk_sha == spec["expected_network_src_sha"],
           "disk=%s" % net_disk_sha[:16])
    else:
        ck("network_src_on_disk_present", False, net_path)

    # 驱动器源码 (服务器上 SHA 权威) 固定 carry/reset 语义: launcher SHA + clear-block 存在性 + carry 链
    drv_path = os.path.join(spec["arm_src"], spec["driver_file"])
    if os.path.isfile(drv_path):
        drv_sha = sha_file(drv_path)
        recovered_facts["driver_src_sha256_on_disk"] = drv_sha
        ck("driver_src_on_disk_sha_match", drv_sha == spec["expected_launcher_src_sha"],
           "disk=%s expected=%s" % (drv_sha[:16], spec["expected_launcher_src_sha"][:16]))
        with open(drv_path, "r", encoding="utf-8", errors="replace") as f:
            drv_src = f.read()
        has_clear = CLEAR_BLOCK_MARKER in drv_src
        has_carry = CARRY_MARKER in drv_src
        recovered_facts["driver_has_boundary_clear_block"] = has_clear
        recovered_facts["driver_has_longstate_carry"] = has_carry
        ck("driver_clear_block_matches_carry_mode", has_clear == spec["expect_clear_block"],
           "has_clear=%s expected=%s (carry_mode=%s)" % (has_clear, spec["expect_clear_block"],
                                                         spec["carry_mode"]))
        ck("driver_longstate_carry_present", has_carry)
    else:
        ck("driver_src_on_disk_present", False, drv_path)

    # ---- 6. 张量布局 / 有限性 (obs 8335; memory ABI 形状) ----
    obs = np.asarray(rd["obs"])
    recovered_facts["obs_shape"] = list(obs.shape)
    ck("obs_shape_E_8335", obs.shape == (FROZEN_CFG["num_envs"], OBS_DIM), str(obs.shape))
    mem = np.asarray(rd["memories"])
    ck("memories_shape", mem.shape == (16, 128, 2, 256), str(mem.shape))
    mask = np.asarray(rd["memories_mask"])
    ck("memories_mask_shape_bool", mask.shape == (16, 8, 1, 129) and mask.dtype == np.bool_,
       "%s %s" % (mask.shape, mask.dtype))
    midx = np.asarray(rd["memories_mask_idx"])
    ck("memories_mask_idx_shape", midx.shape == (16,), str(midx.shape))
    ls_leaves, _ = rd["longstate"]
    ls_shapes = sorted(tuple(np.asarray(l).shape) for l in ls_leaves)
    ck("longstate_layout", ls_shapes == sorted([(16,), (16, 256), (16, 32, 256)]),
       str(ls_shapes))
    rng = np.asarray(rd["rng"])
    ck("rng_shape", rng.shape == (2,), str(rng.shape))
    ck("params_finite", leaves_finite(rd["params"]))
    ck("opt_state_finite", leaves_finite(rd["opt_state"]))

    # opt_state / longstate leaf-hash vs embedded manifest
    if manifest.get("opt_state_leaf_hash"):
        oh = leaf_hash_packed(rd["opt_state"])
        recovered_facts["opt_state_leaf_hash_recomputed"] = oh
        ck("opt_state_leaf_hash_match", oh == manifest["opt_state_leaf_hash"],
           "recomputed=%s embedded=%s" % (oh[:16], str(manifest["opt_state_leaf_hash"])[:16]))
    if manifest.get("longstate_leaf_hash"):
        lh = leaf_hash_packed(rd["longstate"])
        recovered_facts["longstate_leaf_hash_recomputed"] = lh
        ck("longstate_leaf_hash_match", lh == manifest["longstate_leaf_hash"],
           "recomputed=%s embedded=%s" % (lh[:16], str(manifest["longstate_leaf_hash"])[:16]))

    status = "RECOVERED" if all(p for _, p, _ in checks) else "IDENTITY_CONFLICT"
    return _result(candidate_id, spec, checks, status, recovered_facts)


def _result(candidate_id, spec, checks, status, facts):
    import jax
    env = {}
    try:
        import craftax, minicraftax
        env = {"craftax_version": getattr(craftax, "__version__", "unknown"),
               "minicraftax_version": getattr(minicraftax, "__version__", "unknown")}
    except Exception:
        pass
    n_pass = sum(1 for _, p, _ in checks if p)
    result = dict(
        candidate_id=candidate_id,
        family=spec["family"],
        arm=spec["arm"],
        carry_mode=spec["carry_mode"],
        recovery_status=status,
        checks=dict(total=len(checks), passed=n_pass, failed=len(checks) - n_pass),
        check_details=[dict(check=n, passed=p, detail=d) for n, p, d in checks],
        recovered_facts=facts,
        probe_environment=dict(
            python=platform.python_version(),
            jax=getattr(jax, "__version__", "unknown"),
            platform_node=platform.node(),
            jax_platforms="cpu",
            gpu_used_by_probe="NONE",
            **env),
        notes=[
            "params_sha algorithm == trainer driver _params_sha (tree_leaves order, "
            "np.ascontiguousarray(np.asarray(v)).tobytes()).",
            "action_dim=43 not stored in pkl; verified at interface smoke via S4_dark env "
            "construction assert (driver line: action_dim==43 and obs_dim==8335).",
        ],
    )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, choices=sorted(CANDIDATES))
    ap.add_argument("--out", required=True, help="output JSON path (capsule recovery_result.json)")
    args = ap.parse_args()
    try:
        result = probe(args.candidate)
    except Exception:
        result = dict(candidate_id=args.candidate, recovery_status="CHECKPOINT_CORRUPTED",
                      probe_exception=traceback.format_exc()[-2000:])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    print("CANDIDATE=%s STATUS=%s CHECKS=%s/%s OUT=%s" % (
        args.candidate, result.get("recovery_status"),
        result.get("checks", {}).get("passed", "?"),
        result.get("checks", {}).get("total", "?"), args.out))
    failed = [c["check"] for c in result.get("check_details", []) if not c["passed"]]
    if failed:
        print("FAILED_CHECKS=%s" % ",".join(failed))


if __name__ == "__main__":
    main()
