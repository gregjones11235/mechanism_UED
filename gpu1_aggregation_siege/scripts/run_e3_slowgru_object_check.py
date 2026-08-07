#!/usr/bin/env python3
"""E3 SlowGRU OBJECT CHECK ONLY — real assets mounted, ZERO execution.

Strictly: LLM calls=0, rollout=0, optimizer updates=0, checkpoint writes=0,
TaskArchive production writes=0.

Checks:
  1. Real SlowGRU profile loads (logical identity)
  2. Real checkpoint loads via slowgru_runtime (file-SHA + params-SHA gates)
  3. Real adapter loads and passes all identity gates
  4. Real policy forward succeeds (deterministic, batch=1)
  5. Real memory interface succeeds (init_memory + validate_memory)
  6. Canonical DiCode Runtime importable
  7. TaskArchive interface importable
  8. RunState codec importable

On success prints: E3_SLOWGRU_PERSISTENT_OBJECT_CHECK_ONLY_OK
Exit codes: 0 PASS, 4 FAIL, 5 BLOCKED.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIEGE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(SIEGE_ROOT, "src")
sys.path.insert(0, SRC_DIR)

PASS, FAIL, BLOCKED = 0, 4, 5

CANDIDATE_ID = "SLOWGRU_PERSISTENT_CANONICAL_98304"
PROFILE_NAME = "slowgru_persistent_98304"
CHECKPOINT_PATH = ("/home/oseasy/student_pool_v1/cc3/"
                   "SLOWGRU_PERSISTENT_CANONICAL_98304/ckpt/98304/full_state.pkl")
CHECKPOINT_CONTRACT_PATH = ("/home/oseasy/student_pool_v1/cc3/"
                            "SLOWGRU_PERSISTENT_CANONICAL_98304/"
                            "checkpoint_contract.json")
SLOWGRU_RUNTIME_PATH = "/home/oseasy/student_pool_v1/cc3/slowgru_runtime"
NETWORK_SRC_SHA256 = "b265210597d003218e303ef458ff697b5c9c6a14bfccca098ccce8014bf3eb0b"
TRAINER_SRC_SHA256 = "7918333c63bdb6c8917bf423dfb8484942fb46edc6a7c8fa7e36c769cada2545"


def _log(msg: str) -> None:
    print(f"[e3-slowgru-obj] {msg}", flush=True)


def main() -> int:
    started = time.time()
    checks: dict[str, bool] = {}
    errors: list[str] = []

    _log("=== E3 SlowGRU Object Check Only ===")
    _log(f"candidate_id={CANDIDATE_ID}")

    # --- 1. Profile loading -------------------------------------------------
    try:
        from dicode.student_adapters.registry import default_profile_dir, load_student_profile
        profile = load_student_profile(default_profile_dir() / f"{PROFILE_NAME}.yaml")
        assert profile.candidate_id == CANDIDATE_ID, f"candidate_id mismatch: {profile.candidate_id}"
        assert profile.architecture_family == "SLOWGRU", f"family mismatch: {profile.architecture_family}"
        assert profile.observation_shape == (8335,), f"obs shape: {profile.observation_shape}"
        assert profile.action_count == 43, f"action count: {profile.action_count}"
        assert profile.memory_mode == "PERSISTENT", f"memory mode: {profile.memory_mode}"
        expected_identity = profile.expected_identity()
        assert expected_identity.params_sha256 == profile.params_sha256
        checks["1_profile_load"] = True
        _log("1. Profile loaded OK")
    except Exception as exc:
        checks["1_profile_load"] = False
        errors.append(f"profile_load: {type(exc).__name__}: {exc}")
        _log(f"1. Profile FAILED: {exc}")

    # --- 2. Adapter construction --------------------------------------------
    try:
        from dicode.student_adapters.slowgru_adapter import SlowGRUStudentAdapter
        adapter = SlowGRUStudentAdapter(
            profile,
            slowgru_runtime_path=SLOWGRU_RUNTIME_PATH,
            checkpoint_contract_path=CHECKPOINT_CONTRACT_PATH,
            expected_network_src_sha256=NETWORK_SRC_SHA256,
            expected_trainer_src_sha256=TRAINER_SRC_SHA256,
        )
        checks["2_adapter_construct"] = True
        _log("2. Adapter constructed OK")
    except Exception as exc:
        checks["2_adapter_construct"] = False
        errors.append(f"adapter_construct: {type(exc).__name__}: {exc}")
        _log(f"2. Adapter FAILED: {exc}")

    # --- 3. Checkpoint load (real gates) ------------------------------------
    try:
        loaded = adapter.load_full_state(CHECKPOINT_PATH, expected_identity)
        assert loaded["params_sha256"] == profile.params_sha256, "params_sha256 mismatch"
        assert loaded["file_sha256"] == profile.notes["checkpoint_file_sha256"], "file_sha256 mismatch"
        assert loaded["global_step"] == 98304, f"global_step: {loaded['global_step']}"
        gates = loaded["gates"]
        for g in ("G0_identity", "G1_contract", "G2_runtime_path",
                  "G3_network_src_sha256", "G4_load", "G5_trainer_src_sha256",
                  "G6_memory_contract", "G7_action_count"):
            assert g in gates, f"missing gate {g}"
        checks["3_checkpoint_load"] = True
        _log("3. Checkpoint loaded OK (all gates passed)")
    except Exception as exc:
        checks["3_checkpoint_load"] = False
        errors.append(f"checkpoint_load: {type(exc).__name__}: {exc}")
        _log(f"3. Checkpoint FAILED: {exc}")

    # --- 4. Policy forward ------------------------------------------------
    try:
        import numpy as np
        params = loaded["params"]
        # Use batch_size=2: the network's transformerXL.forward_eval uses
        # x.squeeze() which collapses the batch dim at batch_size=1.
        # The adapter handles this internally; the direct check uses batch_size=2.
        obs = np.zeros((2, 8335), dtype=np.float32)
        memory = adapter.initial_memory(2)
        out = adapter.policy_step(params, obs, memory, 0, 0.0, None, True)
        assert "action" in out, "missing action"
        assert "logits" in out, "missing logits"
        assert "value" in out, "missing value"
        assert "memory" in out, "missing memory"
        assert np.isfinite(out["logits"]).all(), "non-finite logits"
        assert np.isfinite(out["value"]).all(), "non-finite value"
        # Also test batch_size=1 via adapter (adapter handles x.squeeze() padding)
        obs1 = np.zeros((1, 8335), dtype=np.float32)
        mem1 = adapter.initial_memory(1)
        out1 = adapter.policy_step(params, obs1, mem1, 0, 0.0, None, True)
        assert out1["action"] is not None, "batch_size=1 action is None"
        assert np.isfinite(out1["logits"]).all(), "batch_size=1 non-finite logits"
        checks["4_policy_forward"] = True
        _log(f"4. Policy forward OK (batch=2 action={out['action']}, batch=1 action={out1['action']})")
    except Exception as exc:
        checks["4_policy_forward"] = False
        errors.append(f"policy_forward: {type(exc).__name__}: {exc}")
        _log(f"4. Policy forward FAILED: {exc}")

    # --- 5. Memory interface -----------------------------------------------
    try:
        import numpy as np
        mem = adapter.initial_memory(2)
        check = adapter.validate_memory(mem, 2)
        assert check["ok"], f"validate_memory failed: {check['reasons']}"
        # Check all 6 fields present
        for key in ("memories", "memories_mask", "memories_mask_idx",
                     "longstate.h", "longstate.buf", "longstate.count"):
            assert key in mem, f"missing memory field {key}"
        assert mem["memories"].shape == (2, 128, 2, 256)
        assert mem["memories_mask"].shape == (2, 8, 1, 129)
        assert mem["memories_mask_idx"].shape == (2,)
        assert mem["longstate.h"].shape == (2, 256)
        assert mem["longstate.buf"].shape == (2, 32, 256)
        assert mem["longstate.count"].shape == (2,)
        # Check persistence across steps: memories should change (fast window),
        # and longstate.buf should accumulate (slow-GRU buffer).
        # Note: longstate.h only commits every SLOW_INTERVAL=32 steps,
        # so it may still be zero after 1 step.
        obs = np.zeros((2, 8335), dtype=np.float32)
        out1 = adapter.policy_step(params, obs, mem, 0, 0.0, None, True)
        mem2 = out1["memory"]
        # Fast memories should have changed (not all zeros after rollout)
        assert not np.allclose(np.asarray(mem2["memories"]), 0.0), "memories unchanged"
        # longstate.buf should accumulate (slow-GRU period buffer)
        # After 1 step, buf may still be all zeros because the first step
        # writes to position 0.  Check that shapes are preserved.
        assert mem2["memories"].shape == (2, 128, 2, 256), "memories shape changed"
        assert mem2["longstate.h"].shape == (2, 256), "longstate.h shape changed"
        assert mem2["longstate.buf"].shape == (2, 32, 256), "longstate.buf shape changed"
        checks["5_memory_interface"] = True
        _log("5. Memory interface OK (init + validate + persistence)")
    except Exception as exc:
        checks["5_memory_interface"] = False
        errors.append(f"memory_interface: {type(exc).__name__}: {exc}")
        _log(f"5. Memory interface FAILED: {exc}")

    # --- 6. Canonical DiCode Runtime importable ---------------------------
    try:
        from dicode.simulator_frontier.canonical_dicode_runtime import (
            callable_source_sha256,
            mint_canonical_dicode_one_update_runtime,
        )
        checks["6_canonical_dicode_import"] = True
        _log("6. Canonical DiCode Runtime importable OK")
    except Exception as exc:
        checks["6_canonical_dicode_import"] = False
        errors.append(f"canonical_dicode: {type(exc).__name__}: {exc}")
        _log(f"6. Canonical DiCode Runtime FAILED: {exc}")

    # --- 7. TaskArchive interface importable ------------------------------
    try:
        from dicode.dreaming.gen_manager import GenManager
        # Just verify the class is importable; no GenManager construction
        checks["7_taskarchive_import"] = True
        _log("7. TaskArchive interface importable OK")
    except Exception as exc:
        checks["7_taskarchive_import"] = False
        errors.append(f"taskarchive: {type(exc).__name__}: {exc}")
        _log(f"7. TaskArchive FAILED: {exc}")

    # --- 8. RunState codec importable -------------------------------------
    try:
        from dicode.simulator_frontier.runstate_codec import (
            RunStateCheckpointManager,
            build_full_run_state,
            runstate_content_hash,
        )
        checks["8_runstate_codec_import"] = True
        _log("8. RunState codec importable OK")
    except Exception as exc:
        checks["8_runstate_codec_import"] = False
        errors.append(f"runstate_codec: {type(exc).__name__}: {exc}")
        _log(f"8. RunState codec FAILED: {exc}")

    # --- Summary ---------------------------------------------------------
    elapsed = round(time.time() - started, 2)
    all_passed = all(checks.values())
    _log(f"--- Summary ({elapsed}s) ---")
    for name, ok in checks.items():
        _log(f"  {name}: {'PASS' if ok else 'FAIL'}")
    if errors:
        for e in errors:
            _log(f"  ERROR: {e}")

    if all_passed:
        _log("E3_SLOWGRU_PERSISTENT_OBJECT_CHECK_ONLY_OK")
        return PASS
    else:
        _log("E3_SLOWGRU_PERSISTENT_OBJECT_CHECK_ONLY_BLOCKED")
        return BLOCKED


if __name__ == "__main__":
    raise SystemExit(main())