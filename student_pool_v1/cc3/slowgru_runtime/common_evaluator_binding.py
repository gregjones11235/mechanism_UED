#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CC3 SlowGRU candidate <-> CC4 COMMON EVALUATOR binding smoke (task section 6).

Contract:
  * waits for and reads /home/oseasy/student_pool_v1/common/COMMON_EVALUATOR_READY.json
    (published by CC4); polls a few times because CC4 may publish while this runs;
  * uses ONLY the CC4 common runner ABI declared in that READY file — no local
    reimplementation of the evaluator, no FRONT/BACK/FULL scientific predicates
    inside CC3 code (the runtime only supplies greedy policy_step);
  * FRONT / BACK / FULL 32-step binding smokes, run_class=INTERFACE_SMOKE,
    performance_claim_authorized=false — smoke reward/success labels must NOT be
    used for ranking (formal ranking is the common evaluator's full job).

Honest states:
  * COMMON_EVALUATOR_READY.json absent after polling -> status
    WAITING_CC4_COMMON_CONTRACT, every profile NOT_EXECUTED_BINDING_ABSENT.
    This is ACCESS_BLOCKED, NOT FAIL, and NOT PASS.
  * ABI import/call error -> status BINDING_ERROR with the literal error recorded;
    nothing is faked and no profile is marked PASS.
"""
import argparse
import hashlib
import json
import os
import sys
import time


def _parse_gpu():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--gpu-uuid", required=True)
    known, _ = ap.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = known.gpu_uuid
    os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_SILENT"] = "true"
    return known.gpu_uuid


GPU_UUID = _parse_gpu()

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON_READY_DEFAULT = "/home/oseasy/student_pool_v1/common/COMMON_EVALUATOR_READY.json"
PROFILES = ("FRONT", "BACK", "FULL")


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def poll_common_ready(path, times, interval):
    attempts = []
    for i in range(max(1, times)):
        present = os.path.isfile(path)
        attempts.append(dict(attempt=i + 1,
                             at_utc8=time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
                             present=present))
        if present:
            return attempts, True
        if i < times - 1:
            time.sleep(interval)
    return attempts, False


def waiting_result(candidate_id, contract, ready_path, attempts, reason):
    return dict(
        record_version="cc3_common_evaluator_binding/v1",
        candidate_id=candidate_id,
        owner="CC3",
        binding_status="WAITING_CC4_COMMON_CONTRACT",
        run_class="INTERFACE_SMOKE",
        performance_claim_authorized=False,
        gpu_uuid=GPU_UUID,
        common_ready_path=ready_path,
        common_ready_present=False,
        wait_reason=reason,
        poll_attempts=attempts,
        profiles={p: dict(status="NOT_EXECUTED_BINDING_ABSENT", steps=0,
                          reward_reference_only=None)
                  for p in PROFILES},
        note="ACCESS_BLOCKED != FAIL: CC4 公共 evaluator 合同尚未发布; 候选内部身份/"
             "interface/memory 门禁独立判定; 公共绑定在 CC4 发布后补跑",
        ranking_use="PROHIBITED — smoke 结果不得用于排名",
    )


def run_binding_with_cc4(candidate_id, contract_path, ready, ready_path):
    """Best-effort adapter to the CC4-declared runner ABI. Never fakes PASS."""
    profiles = {}
    status = "PASS"
    abi_note = ""
    try:
        runner_path = ready["runner_module_path"]
        declared_sha = ready.get("runner_sha256")
        if declared_sha:
            disk_sha = sha_file(runner_path)
            if disk_sha != declared_sha:
                raise RuntimeError("RUNNER_SHA_MISMATCH disk=%s declared=%s"
                                   % (disk_sha, declared_sha))
        abi_note = "abi_version=%s runner=%s sha_pinned=%s" % (
            ready.get("abi_version"), runner_path, bool(declared_sha))
        mod_dir = os.path.dirname(os.path.abspath(runner_path))
        mod_name = os.path.splitext(os.path.basename(runner_path))[0]
        if mod_dir not in sys.path:
            sys.path.insert(0, mod_dir)
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        runner = __import__(mod_name)
        if not hasattr(runner, "run_binding_smoke"):
            raise RuntimeError("RUNNER_ABI_MISSING_ENTRYPOINT run_binding_smoke")
        for p in PROFILES:
            try:
                res = runner.run_binding_smoke(
                    candidate_runtime_dir=HERE,
                    checkpoint_contract_path=contract_path,
                    profile=p, steps=32, gpu_uuid=GPU_UUID)
                ok = bool(res.get("ok"))
                profiles[p] = dict(status="PASS" if ok else "FAIL", steps=32,
                                   abi_result=res)
                if not ok:
                    status = "FAIL"
            except Exception as e:   # noqa: BLE001 — record literally, never fake
                profiles[p] = dict(status="BINDING_ERROR", steps=0,
                                   error=repr(e))
                status = "BINDING_ERROR"
    except Exception as e:   # noqa: BLE001
        status = "BINDING_ERROR"
        abi_note = "%s | adapter_error=%r" % (abi_note, e)
        for p in PROFILES:
            profiles.setdefault(p, dict(status="NOT_EXECUTED_ADAPTER_ERROR", steps=0))
    return status, profiles, abi_note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--common-ready", default=COMMON_READY_DEFAULT)
    ap.add_argument("--poll-times", type=int, default=4)
    ap.add_argument("--poll-interval", type=int, default=30)
    args, _ = ap.parse_known_args()

    with open(args.contract, encoding="utf-8") as f:
        contract = json.load(f)
    candidate_id = contract["candidate_id"]

    attempts, present = poll_common_ready(args.common_ready, args.poll_times,
                                          args.poll_interval)
    if not present:
        result = waiting_result(candidate_id, contract, args.common_ready, attempts,
                                "COMMON_EVALUATOR_READY.json 不存在于共享目录 (CC4 尚未发布)")
    else:
        with open(args.common_ready, encoding="utf-8") as f:
            ready = json.load(f)
        status, profiles, abi_note = run_binding_with_cc4(
            candidate_id, os.path.abspath(args.contract), ready, args.common_ready)
        result = dict(
            record_version="cc3_common_evaluator_binding/v1",
            candidate_id=candidate_id,
            owner="CC3",
            binding_status=status,
            run_class="INTERFACE_SMOKE",
            performance_claim_authorized=False,
            gpu_uuid=GPU_UUID,
            common_ready_path=args.common_ready,
            common_ready_present=True,
            common_ready_contents=ready,
            abi_note=abi_note,
            poll_attempts=attempts,
            profiles=profiles,
            ranking_use="PROHIBITED — 32-step binding smoke 不得用于排名; 正式排名由 "
                        "CC4 公共 evaluator 的完整评估固定",
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    print("BINDING candidate=%s status=%s ready_present=%s OUT=%s" % (
        candidate_id, result["binding_status"], result["common_ready_present"], args.out))


if __name__ == "__main__":
    main()
