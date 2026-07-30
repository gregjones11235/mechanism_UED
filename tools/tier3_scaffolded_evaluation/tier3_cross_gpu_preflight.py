#!/usr/bin/env python3
"""CC4 Tier3 — CROSS-GPU DETERMINISM PREFLIGHT (closing contract §8 / 总控 §三).

Before any formal pool evaluation, the SAME Persistent RMT16 checkpoint rolls out
the SAME episodes on two confirmed-idle, non-conflicting server GPUs; the canonical
results must agree BIT-FOR-BIT. This module never regenerates a bank to dodge a
difference and never loosens the comparison.

Two phases:

  --run --gpu N ...      roll out FULL/FRONT/BACK, one episode each, max_steps=32,
                         greedy_argmax, through the common candidate runtime ABI +
                         the evaluator's own rollout_episode / canonical env; write
                         a per-GPU result JSON (action sequences, terminal labels,
                         metric payloads, episode canonical SHAs, checkpoint / params
                         / bank content SHAs, environment identity).
  --compare A B --out C  compare two per-GPU results field-by-field per the frozen
                         bit_agreement_policy (schemas/tier3_metric_schema_v1.json):
                         canonical fields must be bit-identical; the FIRST canonical
                         mismatch stops everything with
                         CROSS_GPU_DETERMINISM_PREFLIGHT=FAIL and is reported; timing
                         / device-counter fields may differ and are never predicates.
                         Writes cross_gpu_preflight_certificate.json.

FULL starts from the FIRST frozen held-out canonical reset seed (200000 — the
profile's canonical set, never an ad-hoc seed); FRONT/BACK start from frozen bank
index 0 loaded READ-ONLY from the serialized artifact. GPU selection is via
CUDA_VISIBLE_DEVICES BEFORE any JAX import (set it in the environment when calling
--run; this tool verifies and records it).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

# Runnable-as-script AND importable-as-package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_source_audit as audit        # noqa: E402

SCHEMA = "mechanism_UED.cross_gpu_preflight/v1"
PREFLIGHT_MAX_STEPS = 32
DEFAULT_FULL_SEED = 200000          # FIRST of the frozen held-out canonical seeds
DEFAULT_BANK_INDEX = 0
FRONT_FROZEN_CONTENT_SHA256 = (
    "21aeb7dcdcb4ffccfb1eedc80f2c6daec1995c242822a0efbec9b947e275d687")
BACK_FROZEN_CONTENT_SHA256 = (
    "c632e30dcabea7ff812fa07ce855809ec54e7cbca87747fa2d5ab775431f2566")

# Canonical comparison fields (closing contract §8 + metric schema
# bit_agreement_policy). ANY mismatch on these is a HARD FAIL.
CANONICAL_FIELDS = (
    "action_sequence", "terminal_label", "timesteps", "valid_start",
    "metric_payload", "episode_record_sha256",
    "checkpoint_file_sha256", "params_sha256",
    "front_bank_content_sha256", "back_bank_content_sha256")


class FailClosed(Exception):
    """Hard stop on any preflight violation."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: str, doc: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def run_preflight_episodes(checkpoint_path: str, arm: str, contract_path: str,
                           frozen_bank_artifacts: str, full_seed: int,
                           bank_index: int, max_steps: int) -> dict:
    """Roll out the three preflight episodes on THIS process's visible GPU and
    return the canonical result document (no timing field is a predicate)."""
    # dicode source tree for CC2 module import (same setup as every entry point).
    _src = str(audit.repo_root() / "dicode_src" / "src")
    if os.path.isdir(_src) and _src not in sys.path:
        sys.path.insert(0, _src)

    import jax
    import tier3_evaluator as ev
    import tier3_candidate_runtime as abi
    import tier3_frozen_bank_artifacts as art
    import tier3_metrics as metrics

    require(1 <= int(max_steps) <= ev.MAX_TIMESTEPS,
            "FAIL CLOSED: preflight max_steps %r outside [1, %d]"
            % (max_steps, ev.MAX_TIMESTEPS))
    require(arm in abi.ARMS, "FAIL CLOSED: arm %r not in %s" % (arm, abi.ARMS))

    runtime = abi.load_candidate({"runtime_family": "rmt16_gtrxl_cc2", "arm": arm,
                                  "checkpoint_path": checkpoint_path,
                                  "checkpoint_contract_path": contract_path})
    meta = runtime.candidate_metadata()

    front = art.load_bank(metrics.FRONT, frozen_bank_artifacts)
    back = art.load_bank(metrics.BACK, frozen_bank_artifacts)
    require(front["loaded_content_sha256"] == FRONT_FROZEN_CONTENT_SHA256,
            "FAIL CLOSED: FRONT artifact content SHA %s != frozen %s"
            % (front["loaded_content_sha256"][:16],
               FRONT_FROZEN_CONTENT_SHA256[:16]))
    require(back["loaded_content_sha256"] == BACK_FROZEN_CONTENT_SHA256,
            "FAIL CLOSED: BACK artifact content SHA %s != frozen %s"
            % (back["loaded_content_sha256"][:16],
               BACK_FROZEN_CONTENT_SHA256[:16]))
    require(0 <= int(bank_index) < len(front["states"]),
            "FAIL CLOSED: bank_index %r outside the frozen bank" % bank_index)

    entry = ev.make_canonical_env()
    reset_fn = ev._jit_reset(entry)
    devices = [str(d) for d in jax.devices()]

    episodes = {}
    specs = ((metrics.FULL, None), (metrics.FRONT, metrics.FRONT),
             (metrics.BACK, metrics.BACK))
    for sc, bank_sc in specs:
        memory = runtime.init_memory(1)
        holder = {"m": memory}

        def policy_fn(obs, env_state, _h=holder):
            out = runtime.policy_step(obs, _h["m"])
            _h["m"] = out["memory_state"]
            return out["action"]

        if sc == metrics.FULL:
            seed = int(full_seed)
            _obs0, start_state = reset_fn(jax.random.PRNGKey(seed))
            episode_id = "preflight-full-seed%d" % seed
        else:
            seed = int(front["seeds"][bank_index] if sc == metrics.FRONT
                       else back["seeds"][bank_index])
            import jax.numpy as jnp
            states = front["states"] if sc == metrics.FRONT else back["states"]
            start_state = jax.tree.map(jnp.asarray, states[int(bank_index)])
            episode_id = "preflight-%s-bank%d" % (sc, int(bank_index))

        rec = ev.rollout_episode(entry, start_state, sc, policy_fn, episode_id,
                                 seed, max_steps=int(max_steps))
        rec["episode_record_sha256"] = hashlib.sha256(
            json.dumps(rec, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")).encode("utf-8")).hexdigest()
        metric_payload = ev.evaluate(sc, [rec])
        episodes[sc] = {
            "episode_id": episode_id,
            "seed": seed,
            "bank_index": (None if sc == metrics.FULL else int(bank_index)),
            "action_sequence": [int(a) for a in rec["action_sequence"]],
            "terminal_label": rec["terminal_label"],
            "timesteps": int(rec["timesteps"]),
            "valid_start": bool(rec["valid_start"]),
            "episode_record_sha256": rec["episode_record_sha256"],
            "metric_payload": metric_payload,
        }
        print("  [%s] steps=%d terminal=%s record_sha=%s"
              % (sc, rec["timesteps"], rec["terminal_label"],
                 rec["episode_record_sha256"][:16]), flush=True)

    return {
        "schema": SCHEMA,
        "phase": "run",
        "generated_at_utc": _utc_now(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jax_devices": devices,
        "arm": arm,
        "runtime_family": meta["runtime_family"],
        "max_steps": int(max_steps),
        "full_seed": int(full_seed),
        "bank_index": int(bank_index),
        "checkpoint_file_sha256": meta["checkpoint_file_sha256"],
        "params_sha256": meta["params_sha256"],
        "driver_source_sha256": meta["driver_source_sha256"],
        "cc2_policy_source_sha256": meta["cc2_policy_source_sha256"],
        "front_bank_content_sha256": front["loaded_content_sha256"],
        "front_bank_artifact_file_sha256": front["artifact_file_sha256"],
        "back_bank_content_sha256": back["loaded_content_sha256"],
        "back_bank_artifact_file_sha256": back["artifact_file_sha256"],
        "evaluator_source_sha256": _sha256_file(ev.__file__.replace(".pyc", ".py")),
        "abi_source_sha256": _sha256_file(abi.__file__.replace(".pyc", ".py")),
        "evaluator_git_commit": ev._git_commit_head(),
        "python_version": sys.version.split()[0],
        "jax_version": getattr(__import__("jax"), "__version__", "unknown"),
        "episodes": episodes,
    }


def compare_preflight_runs(run_a: dict, run_b: dict) -> dict:
    """Field-by-field canonical comparison of two per-GPU runs. Returns the
    certificate document; CROSS_GPU_DETERMINISM_PREFLIGHT is PASS iff every
    canonical field agrees. The first canonical difference (if any) is recorded —
    never swallowed, never loosened."""
    require(run_a.get("schema") == SCHEMA and run_b.get("schema") == SCHEMA,
            "FAIL CLOSED: both inputs must be preflight run documents (schema %s)"
            % SCHEMA)
    require(run_a.get("arm") == run_b.get("arm") == "persistent",
            "FAIL CLOSED: the preflight uses the Persistent checkpoint on BOTH GPUs "
            "(got %r / %r)" % (run_a.get("arm"), run_b.get("arm")))
    require(run_a.get("max_steps") == run_b.get("max_steps") == PREFLIGHT_MAX_STEPS,
            "FAIL CLOSED: preflight max_steps must be %d on both GPUs"
            % PREFLIGHT_MAX_STEPS)
    require(run_a.get("full_seed") == run_b.get("full_seed")
            and run_a.get("bank_index") == run_b.get("bank_index"),
            "FAIL CLOSED: the two runs must use the identical start schedule")

    mismatches = []
    for field in ("checkpoint_file_sha256", "params_sha256",
                  "front_bank_content_sha256", "back_bank_content_sha256"):
        if run_a.get(field) != run_b.get(field):
            mismatches.append({"field": field, "gpu_a": run_a.get(field),
                               "gpu_b": run_b.get(field)})

    eps_a, eps_b = run_a.get("episodes", {}), run_b.get("episodes", {})
    require(set(eps_a) == set(eps_b) and len(eps_a) == 3,
            "FAIL CLOSED: both runs must carry the FULL/FRONT/BACK episodes")
    for sc in sorted(eps_a):
        ea, eb = eps_a[sc], eps_b[sc]
        for field in ("action_sequence", "terminal_label", "timesteps",
                      "valid_start", "metric_payload", "episode_record_sha256"):
            if ea.get(field) != eb.get(field):
                mismatches.append({"scenario": sc, "field": field,
                                   "gpu_a": ea.get(field), "gpu_b": eb.get(field)})

    passed = not mismatches
    return {
        "schema": SCHEMA,
        "phase": "certificate",
        "generated_at_utc": _utc_now(),
        "CROSS_GPU_DETERMINISM_PREFLIGHT": "PASS" if passed else "FAIL",
        "compared_canonical_fields": list(CANONICAL_FIELDS),
        "non_canonical_fields_may_differ": ["wall_clock", "device_counters",
                                            "timestamps", "jax_devices"],
        "gpu_a": {"cuda_visible_devices": run_a.get("cuda_visible_devices"),
                  "jax_devices": run_a.get("jax_devices")},
        "gpu_b": {"cuda_visible_devices": run_b.get("cuda_visible_devices"),
                  "jax_devices": run_b.get("jax_devices")},
        "checkpoint_file_sha256": run_a.get("checkpoint_file_sha256"),
        "params_sha256": run_a.get("params_sha256"),
        "front_bank_content_sha256": run_a.get("front_bank_content_sha256"),
        "back_bank_content_sha256": run_a.get("back_bank_content_sha256"),
        "episode_record_sha256_by_scenario": {
            sc: eps_a[sc]["episode_record_sha256"] for sc in sorted(eps_a)},
        "first_difference": mismatches[0] if mismatches else None,
        "all_mismatches": mismatches,
        "bank_regeneration_to_dodge_difference": False,
        "comparison_loosened": False,
        "run_a_source_sha256": hashlib.sha256(
            json.dumps(run_a, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode("utf-8")).hexdigest(),
        "run_b_source_sha256": hashlib.sha256(
            json.dumps(run_b, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode("utf-8")).hexdigest(),
    }


def self_test() -> int:
    problems = []

    def check(name, cond):
        if not cond:
            problems.append(name)

    def fake_run(gpu, record_sha="r" * 64, actions=(1, 2, 3), terminal="TIMEOUT",
                 ckpt="c" * 64):
        eps = {}
        for sc in ("full", "front_l2", "back_l2"):
            eps[sc] = {"episode_id": "preflight-%s" % sc, "seed": 200000,
                       "bank_index": None if sc == "full" else 0,
                       "action_sequence": list(actions), "terminal_label": terminal,
                       "timesteps": 32, "valid_start": True,
                       "episode_record_sha256": record_sha,
                       "metric_payload": {"value": 0.0}}
        return {"schema": SCHEMA, "phase": "run", "arm": "persistent",
                "max_steps": 32, "full_seed": 200000, "bank_index": 0,
                "cuda_visible_devices": gpu, "jax_devices": [gpu],
                "checkpoint_file_sha256": ckpt, "params_sha256": "p" * 64,
                "front_bank_content_sha256": FRONT_FROZEN_CONTENT_SHA256,
                "back_bank_content_sha256": BACK_FROZEN_CONTENT_SHA256,
                "episodes": eps}

    a, b = fake_run("2"), fake_run("3")
    cert = compare_preflight_runs(a, b)
    check("identical_runs_pass",
          cert["CROSS_GPU_DETERMINISM_PREFLIGHT"] == "PASS"
          and cert["first_difference"] is None)

    # A canonical action-sequence difference on ONE scenario is a HARD FAIL and is
    # reported as the first difference.
    b2 = fake_run("3", actions=(1, 2, 4))
    cert2 = compare_preflight_runs(a, b2)
    check("action_mismatch_fails",
          cert2["CROSS_GPU_DETERMINISM_PREFLIGHT"] == "FAIL"
          and cert2["first_difference"]["field"] == "action_sequence")

    b3 = fake_run("3", terminal="DIED")
    check("terminal_label_mismatch_fails",
          compare_preflight_runs(a, b3)["CROSS_GPU_DETERMINISM_PREFLIGHT"] == "FAIL")
    b4 = fake_run("3", record_sha="x" * 64)
    check("episode_sha_mismatch_fails",
          compare_preflight_runs(a, b4)["CROSS_GPU_DETERMINISM_PREFLIGHT"] == "FAIL")
    b5 = fake_run("3", ckpt="d" * 64)
    check("checkpoint_sha_mismatch_fails",
          compare_preflight_runs(a, b5)["CROSS_GPU_DETERMINISM_PREFLIGHT"] == "FAIL")

    # Non-canonical differences (device strings) must NOT fail the preflight.
    b6 = fake_run("3")
    b6["jax_devices"] = ["cuda:id3"]
    check("device_string_difference_not_a_predicate",
          compare_preflight_runs(a, b6)["CROSS_GPU_DETERMINISM_PREFLIGHT"] == "PASS")

    # Structural gates.
    try:
        bad = fake_run("3")
        bad["arm"] = "reset128"
        compare_preflight_runs(a, bad)
        check("arm_mismatch_rejected", False)
    except FailClosed:
        check("arm_mismatch_rejected", True)
    try:
        bad = fake_run("3")
        bad["max_steps"] = 64
        compare_preflight_runs(a, bad)
        check("max_steps_mismatch_rejected", False)
    except FailClosed:
        check("max_steps_mismatch_rejected", True)
    try:
        bad = fake_run("3")
        del bad["episodes"]["back_l2"]
        compare_preflight_runs(a, bad)
        check("missing_scenario_rejected", False)
    except FailClosed:
        check("missing_scenario_rejected", True)

    if problems:
        print("TIER3_CROSS_GPU_PREFLIGHT_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_CROSS_GPU_PREFLIGHT_SELF_TEST_PASS "
          "(canonical fields bit-exact; first difference reported; no loosening)")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return self_test()
    if "--run" in argv:
        def opt(flag, default=None):
            if flag in argv:
                return argv[argv.index(flag) + 1]
            return default
        require(opt("--checkpoint") and opt("--frozen-bank-artifacts")
                and opt("--out"),
                "usage: tier3_cross_gpu_preflight.py --run --checkpoint PKL "
                "--frozen-bank-artifacts DIR --out RESULT.json "
                "[--arm persistent] [--contract PATH] [--full-seed 200000] "
                "[--bank-index 0] [--max-steps 32]  (set CUDA_VISIBLE_DEVICES=N in "
                "the environment BEFORE calling)")
        doc = run_preflight_episodes(
            opt("--checkpoint"), opt("--arm", "persistent"),
            opt("--contract"), opt("--frozen-bank-artifacts"),
            int(opt("--full-seed", DEFAULT_FULL_SEED)),
            int(opt("--bank-index", DEFAULT_BANK_INDEX)),
            int(opt("--max-steps", PREFLIGHT_MAX_STEPS)))
        _atomic_json(opt("--out"), doc)
        print("PREFLIGHT_RUN_WRITTEN %s gpu=%s" % (opt("--out"),
                                                   doc["cuda_visible_devices"]))
        return 0
    if "--compare" in argv:
        i = argv.index("--compare")
        path_a, path_b = argv[i + 1], argv[i + 2]
        out = argv[argv.index("--out") + 1] if "--out" in argv else None
        require(out, "usage: --compare A.json B.json --out CERT.json")
        with open(path_a, encoding="utf-8") as fh:
            run_a = json.load(fh)
        with open(path_b, encoding="utf-8") as fh:
            run_b = json.load(fh)
        cert = compare_preflight_runs(run_a, run_b)
        _atomic_json(out, cert)
        print("CROSS_GPU_DETERMINISM_PREFLIGHT=%s cert=%s"
              % (cert["CROSS_GPU_DETERMINISM_PREFLIGHT"], out))
        return 0 if cert["CROSS_GPU_DETERMINISM_PREFLIGHT"] == "PASS" else 1
    print("usage: tier3_cross_gpu_preflight.py --self-test | --run ... | --compare ...")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
