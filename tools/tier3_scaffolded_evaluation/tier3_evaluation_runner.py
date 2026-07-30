#!/usr/bin/env python3
"""CC4 Tier3 — evaluation RUNNER: real process-exit provenance + atomic finalize.

task §二: an evaluator may NOT self-declare its own exit code. This runner is the
PARENT process:

    runner (this file) ──spawn──▶ tier3_evaluator.py (engine CHILD)
        │                              │
        │ capture pid / argv /         │ engine-stage artifacts ONLY
        │ actual_started_at_utc        │ (episode_records.jsonl,
        │                              │  evaluation_result.json,
        │                              │  evaluation_certificate.json with
        │                              │  ENGINE certificates — no exit
        │                              │  provenance, no SHA256SUMS)
        │◀────────── wait() ───────────┘
        │ literal_exit_code = rc
        ▼
    rc != 0  → run_status.json(ENGINE_FAILED) into the FINAL dir; the temp dir is
               NEVER promoted; no PASS certificate exists (the engine certificates
               carry no exit provenance and fail full verification).
    rc == 0  → re-assert the ENGINE-stage binding of every scenario certificate
               (clean: no provenance yet), inject the RUNNER-SUPPLIED provenance
               (child_process_pid / child_process_argv / actual_started_at_utc /
               actual_finished_at_utc / literal_exit_code=0 / exit_source=wait_pid /
               inferred_from_log=False / evaluation_runner_source_sha256), run the
               FULL finalized verification (assert_eval_binding_complete + NEG24 +
               NEG25), then write run_status.json(FINALIZED_PASS) + SHA256SUMS into
               the temp dir, fsync, and ATOMICALLY rename temp → final.

task §四: the final output dir must be fresh (missing or empty) and the temp dir
must not exist — never rm -rf, never overwrite, never append, never auto-rename.
Self-declared / log-inferred exit codes are never accepted (NEG37); a finalized
certificate is only ever emitted when wait() returned the literal exit code 0.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

# Runnable-as-script AND importable-as-package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_source_audit as audit                  # noqa: E402
import tier3_evaluation_certificate as certmod      # noqa: E402
import tier3_checkpoint_contract as contractmod     # noqa: E402
import tier3_evaluator as evaluator                 # noqa: E402 (pure at import)

SCHEMA = "mechanism_UED.tier3_run_status/v1"

# Engine-stage artifacts (written by the child engine; all must exist, non-empty).
ENGINE_ARTIFACTS = ("episode_records.jsonl", "evaluation_result.json",
                    "evaluation_certificate.json")
# Files covered by the finalized SHA256SUMS (engine artifacts + runner status).
SUMMED_FILES = ENGINE_ARTIFACTS + ("run_status.json",)

RUN_STATUS_FINALIZED_PASS = "FINALIZED_PASS"
RUN_STATUS_ENGINE_FAILED = "ENGINE_FAILED"
RUN_STATUS_FINALIZE_FAILED = "FINALIZE_FAILED"
RUN_CLASSES = {
    "performance": "PROVISIONAL_STRONG_STUDENT_SELECTION",
    "smoke": "INTERFACE_SMOKE",
}


class FailClosed(Exception):
    """Hard stop on any provenance / freshness / finalize violation."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


def _now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _lf_sha256_file(path: str) -> str:
    """LF-normalized source SHA (EOL-independent; same canonical form repo-wide)."""
    import hashlib
    with open(path, "rb") as fh:
        data = fh.read().decode("utf-8")
    return hashlib.sha256(data.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def runner_source_sha256() -> str:
    return _lf_sha256_file(os.path.abspath(__file__))


def evaluator_source_sha256() -> str:
    """LF-SHA of the ACTUAL evaluator source bound by this run (总控 §三)."""
    return _lf_sha256_file(os.path.abspath(evaluator.__file__))


def _git_or_none(*git_args) -> str:
    """Best-effort `git rev-parse` → 40-hex, or 'UNAVAILABLE' (never fail the run
    for a metadata probe; the authoritative gate is the certificate's frozen
    evaluator_git_commit binding)."""
    import subprocess as _sp
    try:
        out = _sp.run(["git", "rev-parse"] + list(git_args),
                      cwd=str(audit.repo_root()), capture_output=True, text=True)
        v = (out.stdout or "").strip()
        if out.returncode == 0 and len(v) == 40 and all(
                c in "0123456789abcdef" for c in v):
            return v
    except OSError:
        pass
    return "UNAVAILABLE"


def local_commit_sha() -> str:
    return _git_or_none("HEAD")


def local_tree_sha() -> str:
    return _git_or_none("HEAD^{tree}")


# ---------------------------------------------------------------------------
# Parent/child process handling (task §二)
# ---------------------------------------------------------------------------
def run_child(engine_argv: list, cwd: str) -> dict:
    """Spawn the engine child, wait() on it, and return the LITERAL exit provenance.

    stdio is inherited so the child's per-episode progress is monitored live. The
    exit code is the raw return value of wait() (proc.wait()) — never parsed from a
    log, never self-declared (exit_source=wait_pid, inferred_from_log=False).
    """
    argv = [str(a) for a in engine_argv]
    require(argv and argv[0],
            "FAIL CLOSED: run_child needs a non-empty engine argv")
    started = _now_iso()
    proc = subprocess.Popen(argv, cwd=cwd)          # inherited stdio (monitoring)
    rc = proc.wait()                                # the LITERAL exit code
    finished = _now_iso()
    return {
        "child_process_pid": int(proc.pid),
        "child_process_argv": argv,
        "actual_started_at_utc": started,
        "actual_finished_at_utc": finished,
        "literal_exit_code": int(rc),
        "exit_source": "wait_pid",
        "inferred_from_log": False,
    }


def build_engine_argv(mode: str, arm: str, checkpoint: str, contract: str,
                      out_tmp: str, scenario: str = "all", episodes: int = None,
                      max_steps: int = None, cc2_snapshot_root: str = None,
                      cc2_driver_source: str = None, python: str = None) -> list:
    """The child argv, with paths RELATIVE to the repo root (cwd=repo root at spawn
    time) so committed provenance never rests on local absolute D: paths."""
    require(mode in RUN_CLASSES, "FAIL CLOSED: runner mode %r not in %s"
            % (mode, sorted(RUN_CLASSES)))
    require(arm in ("persistent", "reset128"),
            "FAIL CLOSED: --arm %r not in (persistent, reset128)" % arm)
    repo = str(audit.repo_root())
    py = python or sys.executable
    argv = [py, "-u", "tools/tier3_scaffolded_evaluation/tier3_evaluator.py",
            "--performance-evaluation" if mode == "performance" else "--interface-smoke",
            "--checkpoint", os.path.relpath(checkpoint, repo),
            "--checkpoint-contract", os.path.relpath(contract, repo),
            "--arm", arm,
            "--out", os.path.relpath(out_tmp, repo)]
    if mode == "smoke":
        argv += ["--scenario", str(scenario)]
        if episodes is not None:
            argv += ["--episodes", str(int(episodes))]
        if max_steps is not None:
            argv += ["--max-steps", str(int(max_steps))]
    if cc2_snapshot_root is not None:
        argv += ["--cc2_snapshot_root", os.path.relpath(cc2_snapshot_root, repo)]
    if cc2_driver_source is not None:
        argv += ["--cc2_driver_source", os.path.relpath(cc2_driver_source, repo)]
    return argv


# ---------------------------------------------------------------------------
# Artifact verification + certificate finalization (task §二/§五)
# ---------------------------------------------------------------------------
def verify_child_artifacts(tmp_dir: str) -> dict:
    """Every engine-stage artifact must exist and be non-empty before finalization."""
    paths = {}
    for name in ENGINE_ARTIFACTS:
        p = os.path.join(tmp_dir, name)
        require(os.path.isfile(p) and os.path.getsize(p) > 0,
                "FAIL CLOSED: engine child temp artifact missing or empty: %s (the "
                "child must have completed its engine stage)" % p)
        paths[name] = p
    return paths


def finalize_certificates(tmp_dir: str, provenance: dict) -> dict:
    """Inject the RUNNER-SUPPLIED exit provenance into every engine certificate and
    run the FULL finalized verification (NEG27/NEG29/NEG37 + NEG24/NEG25).

    Defence in depth: each certificate is first re-asserted at the ENGINE stage — a
    certificate that already carries exit provenance (pre-injection) fails closed,
    because only this runner may bind the literal exit code.
    """
    cert_path = os.path.join(tmp_dir, "evaluation_certificate.json")
    require(os.path.isfile(cert_path) and os.path.getsize(cert_path) > 0,
            "FAIL CLOSED: engine certificate artifact missing or empty: %s (the "
            "child must have completed its engine stage)" % cert_path)
    with open(cert_path, "r", encoding="utf-8") as fh:
        certs = json.load(fh)
    require(isinstance(certs, dict) and certs,
            "FAIL CLOSED: evaluation_certificate.json must be a non-empty "
            "{scenario: certificate} dict")
    prov = dict(provenance)
    prov["evaluation_runner_source_sha256"] = runner_source_sha256()
    for sc, cert in certs.items():
        # 1. engine stage must be CLEAN (no exit provenance yet) — NEG37.
        certmod.assert_engine_binding_complete(cert)
        # 2. inject the runner provenance, then 3. FULL finalized verification.
        binding = dict(cert["eval_binding"])
        binding.update(prov)
        cert["eval_binding"] = binding
        certmod.assert_eval_binding_complete(cert)
        certmod.assert_scaffold_hash_not_global(cert)
        certmod.assert_scaffold_does_not_claim_full_success(cert)
    with open(cert_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(certs, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    return certs


# ---------------------------------------------------------------------------
# run_status.json + SHA256SUMS + atomic rename (task §四)
# ---------------------------------------------------------------------------
def _repo_relative_or_raw(path: str) -> str:
    """Record a repo-relative path when possible (no local absolute D: identity in
    committed evidence); fall back to the raw path across drive boundaries."""
    if not os.path.isabs(path):
        return path
    try:
        return os.path.relpath(path, str(audit.repo_root()))
    except ValueError:
        return path


def run_status_doc(status: str, arm: str, run_class: str, provenance: dict,
                   checkpoint_contract_sha256: str, out_dir: str,
                   temp_dir_promoted: bool, sha256sums_verified: bool,
                   reason: str = None,
                   push_status: str = "NOT_PUSHED_AT_RUN_TIME") -> dict:
    """Every run_status binds the FULL provenance set (总控 §三): the local commit
    SHA, the git tree SHA, the push status, the evaluator source SHA and the runner
    source SHA — in addition to the literal wait() exit provenance."""
    doc = {
        "schema": SCHEMA,
        "run_class": run_class,
        "arm": arm,
        "status": status,
        "output_finalized": status == RUN_STATUS_FINALIZED_PASS,
        "temp_dir_promoted": bool(temp_dir_promoted),
        "sha256sums_verified": bool(sha256sums_verified),
        "child_process_pid": provenance.get("child_process_pid"),
        "child_process_argv": list(provenance.get("child_process_argv") or []),
        "actual_started_at_utc": provenance.get("actual_started_at_utc"),
        "actual_finished_at_utc": provenance.get("actual_finished_at_utc"),
        "literal_exit_code": provenance.get("literal_exit_code"),
        "exit_source": provenance.get("exit_source"),
        "inferred_from_log": provenance.get("inferred_from_log"),
        "local_commit_sha": local_commit_sha(),
        "local_tree_sha": local_tree_sha(),
        "push_status": push_status,
        "evaluator_source_sha256": evaluator_source_sha256(),
        "evaluation_runner_source_sha256": runner_source_sha256(),
        "checkpoint_contract_sha256": checkpoint_contract_sha256,
        "output_dir": _repo_relative_or_raw(out_dir),
    }
    if reason is not None:
        doc["reason"] = reason
    return doc


def _write_json(path: str, doc: dict):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")


def _record_failure(out_dir: str, provenance: dict, arm: str, run_class: str,
                    checkpoint_contract_sha256: str, status: str, reason: str,
                    push_status: str = "NOT_PUSHED_AT_RUN_TIME") -> str:
    """Write a FAIL run_status.json into the FINAL dir (created if needed). The temp
    dir is never promoted on this path — no PASS certificate can exist. The FULL
    provenance set (总控 §三) is bound on the failure path too."""
    os.makedirs(out_dir, exist_ok=True)
    doc = run_status_doc(status, arm, run_class, provenance,
                         checkpoint_contract_sha256, out_dir,
                         temp_dir_promoted=False, sha256sums_verified=False,
                         reason=reason, push_status=push_status)
    p = os.path.join(out_dir, "run_status.json")
    _write_json(p, doc)
    return p


def write_sha256sums(dir_path: str) -> str:
    """SHA256SUMS over SUMMED_FILES (all must exist) in the finalized temp dir."""
    import hashlib
    sums = {}
    for name in SUMMED_FILES:
        p = os.path.join(dir_path, name)
        require(os.path.isfile(p) and os.path.getsize(p) > 0,
                "FAIL CLOSED: cannot hash missing artifact %s" % p)
        with open(p, "rb") as fh:
            sums[name] = hashlib.sha256(fh.read()).hexdigest()
    p = os.path.join(dir_path, "SHA256SUMS")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        for name in sorted(sums):
            fh.write("%s  %s\n" % (sums[name], name))
    return p


def verify_dir_sha256sums(dir_path: str) -> bool:
    """Re-verify every SHA256SUMS entry against the on-disk bytes (fail closed).
    Shared by the runner (post-rename check) and the cross-arm comparator."""
    import hashlib
    sums_path = os.path.join(dir_path, "SHA256SUMS")
    require(os.path.isfile(sums_path),
            "FAIL CLOSED: %s missing — directory not finalized" % sums_path)
    entries = {}
    with open(sums_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            sha, name = line.split(None, 1)
            entries[name.strip()] = sha.strip()
    require(set(entries) == set(SUMMED_FILES),
            "FAIL CLOSED: SHA256SUMS entries %s != frozen set %s"
            % (sorted(entries), list(SUMMED_FILES)))
    for name, sha in entries.items():
        p = os.path.join(dir_path, name)
        with open(p, "rb") as fh:
            actual = hashlib.sha256(fh.read()).hexdigest()
        require(actual == sha,
                "FAIL CLOSED: SHA256SUMS mismatch for %s (recorded %s, actual %s)"
                % (name, sha[:16], actual[:16]))
    return True


def _fsync_best_effort(path: str):
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass            # e.g. directory fsync on Windows; best-effort only


# ---------------------------------------------------------------------------
# The full orchestrated arm run
# ---------------------------------------------------------------------------
def evaluate_arm(arm: str, checkpoint: str, contract: str, out_dir: str,
                 mode: str = "performance", scenario: str = "all",
                 episodes: int = None, max_steps: int = None,
                 cc2_snapshot_root: str = None, cc2_driver_source: str = None,
                 push_status: str = "NOT_PUSHED_AT_RUN_TIME") -> int:
    """Run ONE arm through the parent/child runner (task §二/§四/§十). Sequential by
    contract: never spawn two JAX evaluators at once. `push_status` records the
    remote-push outcome at run time (总控 §三; e.g. BLOCKED_NETWORK — not a code
    failure) into every run_status written on this path."""
    require(arm in ("persistent", "reset128"),
            "FAIL CLOSED: --arm %r not in (persistent, reset128)" % arm)
    require(mode in RUN_CLASSES, "FAIL CLOSED: runner mode %r not in %s"
            % (mode, sorted(RUN_CLASSES)))
    run_class = RUN_CLASSES[mode]

    # task §四: freshness gates FIRST (no rm -rf / overwrite / append / auto-rename).
    evaluator.assert_output_dir_fresh(out_dir)
    tmp = out_dir + ".inprogress"
    require(not os.path.exists(tmp),
            "FAIL CLOSED (EVALUATION_INPROGRESS_DIR_EXISTS): temp dir %s already "
            "exists — a prior run did not finalize; inspect and remove it manually "
            "(never auto-deleted)" % tmp)

    contract_sha = contractmod.load_contract(contract)["checkpoint_contract_sha256"]
    os.makedirs(tmp, exist_ok=True)

    argv = build_engine_argv(mode, arm, checkpoint, contract, tmp,
                             scenario=scenario, episodes=episodes,
                             max_steps=max_steps,
                             cc2_snapshot_root=cc2_snapshot_root,
                             cc2_driver_source=cc2_driver_source)
    repo = str(audit.repo_root())
    print("RUNNER: spawning engine child (arm=%s, run_class=%s, cwd=%s)"
          % (arm, run_class, repo), flush=True)
    prov = run_child(argv, cwd=repo)
    print("RUNNER: child pid=%d literal_exit_code=%d (exit_source=wait_pid, "
          "inferred_from_log=false) started=%s finished=%s"
          % (prov["child_process_pid"], prov["literal_exit_code"],
             prov["actual_started_at_utc"], prov["actual_finished_at_utc"]),
          flush=True)

    # task §二: child failure → FAIL run_status in the final dir; temp NOT promoted.
    if prov["literal_exit_code"] != 0:
        _record_failure(out_dir, prov, arm, run_class, contract_sha,
                        RUN_STATUS_ENGINE_FAILED,
                        "engine child exited with literal code %d (no PASS "
                        "certificate emitted; temp dir not promoted)"
                        % prov["literal_exit_code"], push_status=push_status)
        print("RUNNER: ENGINE_FAILED (arm=%s literal_exit_code=%d) — temp dir %s "
              "NOT promoted" % (arm, prov["literal_exit_code"], tmp), flush=True)
        return 1

    try:
        verify_child_artifacts(tmp)
        certs = finalize_certificates(tmp, prov)
    except (FailClosed, certmod.FailClosed) as exc:
        _record_failure(out_dir, prov, arm, run_class, contract_sha,
                        RUN_STATUS_FINALIZE_FAILED, str(exc)[:500],
                        push_status=push_status)
        print("RUNNER: FINALIZE_FAILED (arm=%s): %s" % (arm, exc), flush=True)
        return 2

    _write_json(os.path.join(tmp, "run_status.json"),
                run_status_doc(RUN_STATUS_FINALIZED_PASS, arm, run_class, prov,
                               contract_sha, out_dir, temp_dir_promoted=False,
                               sha256sums_verified=False,
                               push_status=push_status))
    write_sha256sums(tmp)
    for name in SUMMED_FILES + ("SHA256SUMS",):
        _fsync_best_effort(os.path.join(tmp, name))
    _fsync_best_effort(tmp)
    os.replace(tmp, out_dir)                      # atomic promotion
    require(verify_dir_sha256sums(out_dir),
            "FAIL CLOSED: post-rename SHA256SUMS verification failed")
    print("RUNNER: FINALIZED_PASS (arm=%s, scenarios=%s, "
          "checkpoint_contract_sha256=%s, out=%s)"
          % (arm, sorted(certs), contract_sha, out_dir), flush=True)
    return 0


# ---------------------------------------------------------------------------
# Self-test (pure; runs on any host — no JAX required).
# ---------------------------------------------------------------------------
def self_test() -> int:
    problems = []

    def check(name, cond):
        if not cond:
            problems.append(name)

    import tempfile

    def _prov(**over):
        p = {"child_process_pid": 4242,
             "child_process_argv": ["python", "-u",
                                    "tools/tier3_scaffolded_evaluation/tier3_evaluator.py",
                                    "--performance-evaluation"],
             "actual_started_at_utc": "2026-07-30T00:00:00+00:00",
             "actual_finished_at_utc": "2026-07-30T01:00:00+00:00",
             "literal_exit_code": 0,
             "exit_source": "wait_pid",
             "inferred_from_log": False}
        p.update(over)
        return p

    # a) The literal wait() exit code is captured (task §二): a child exiting 3 is
    #    reported as literal_exit_code=3, exit_source=wait_pid — never 0.
    prov = run_child([sys.executable, "-c", "import sys; sys.exit(3)"],
                     cwd=os.getcwd())
    check("child_literal_rc_captured",
          prov["literal_exit_code"] == 3 and prov["exit_source"] == "wait_pid"
          and prov["inferred_from_log"] is False
          and isinstance(prov["child_process_pid"], int)
          and prov["child_process_pid"] > 0)
    check("child_argv_bound",
          prov["child_process_argv"][-1] == "import sys; sys.exit(3)")
    check("child_times_iso",
          isinstance(prov["actual_started_at_utc"], str)
          and prov["actual_started_at_utc"]
          and isinstance(prov["actual_finished_at_utc"], str))

    # b) task §四 freshness gates (no child spawned — they fire first).
    with tempfile.TemporaryDirectory() as td:
        nonempty = os.path.join(td, "out")
        os.makedirs(nonempty)
        with open(os.path.join(nonempty, "stale.json"), "w") as fh:
            fh.write("{}")
        try:
            evaluate_arm("persistent", "ckpt.pkl", "contract.json", nonempty)
            check("NEG39_runner_nonempty_out_rejected", False)
        except (FailClosed, evaluator.FailClosed):
            check("NEG39_runner_nonempty_out_rejected", True)
        fresh = os.path.join(td, "fresh")
        os.makedirs(fresh + ".inprogress")
        try:
            evaluate_arm("persistent", "ckpt.pkl", "contract.json", fresh)
            check("inprogress_collision_rejected", False)
        except (FailClosed, evaluator.FailClosed):
            check("inprogress_collision_rejected", True)

    # c) Finalizing an empty temp dir fails closed (missing engine artifacts).
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, "t")
        os.makedirs(tmp)
        try:
            finalize_certificates(tmp, _prov())
            check("finalize_empty_tmp_rejected", False)
        except FailClosed:
            check("finalize_empty_tmp_rejected", True)
        try:
            verify_child_artifacts(tmp)
            check("verify_empty_tmp_rejected", False)
        except FailClosed:
            check("verify_empty_tmp_rejected", True)

    # d) Full synthetic finalize: an ENGINE certificate (built by certmod with no
    #    exit provenance) is injected + fully verified, SHA256SUMS written +
    #    re-verified, and the temp dir is atomically promoted.
    FULL, FRONT, BACK = certmod.FULL, certmod.FRONT, certmod.BACK

    def _engine_binding(**over):
        b = {
            "state_bank_hash": "2" + "a" * 63,
            "state_payload_hashes": ["b" * 64, "c" * 64],
            "checkpoint_file_sha256": "d" * 64,
            "cc2_params_sha256": "e" * 64,
            "checkpoint_step": 98304,
            "carry_mode": "persistent",
            "run_class": "PROVISIONAL_STRONG_STUDENT_SELECTION",
            "episode_records_sha256": "f" * 64,
            "cc2_policy_source_sha256": "0" * 64,
            "evaluator_source_sha256": "1" * 64,
            "predicate_code_sha256": "a4fba86b054d20412fc1df2c79e7000d66b0525d"
                                     "ecb1801fa474ee7fb0d25b4c",
            "observation_shape": [8335],
            "action_dim": 43,
            "params_unchanged": True,
            "performance_claim_authorized": False,
            "driver_source_sha256": "9" * 64,
            "checkpoint_contract_sha256": "7" * 64,
            "checkpoint_contract_arm": "persistent",
            "action_mode": "greedy_argmax",
            "max_timesteps": 4096,
            "evaluation_seed_schedule": {
                FULL: {"kind": "canonical_reset_seeds_held_out", "base": 200000,
                       "count": 64, "seeds": [200000 + i for i in range(64)]},
                FRONT: {"kind": "frozen_bank_state_each_once", "seed_base": 10000,
                        "stride": 1, "count": 8, "seeds": [10000 + i for i in range(8)]},
                BACK: {"kind": "frozen_bank_state_each_once", "seed_base": 10000,
                       "stride": 1, "count": 8,
                       "seeds": [1010000 + i for i in range(8)]},
            },
            "state_entry_ids": {
                FULL: ["full-seed%d" % (200000 + i) for i in range(64)],
                FRONT: ["front_l2-bank%d" % i for i in range(8)],
                BACK: ["back_l2-bank%d" % i for i in range(8)]},
            "python_version": "3.11.9",
            "jax_version": "0.4.30",
            "jaxlib_version": "0.4.30",
            "numpy_version": "1.26.4",
            "flax_version": "0.8.5",
            "craftax_version": "1.4.5",
            "evaluator_git_commit": "f67675b87ad98b391f82678bc2f937ab30578145",
            "scientific_claim_authorized": False,
            "single_training_seed": True,
            "provisional_selection_only": True,
        }
        b.update(over)
        return b

    def _result(scenario):
        import tier3_metrics as metrics
        return {
            "schema": "mechanism_UED.tier3_evaluation_result/v1",
            "scenario": scenario,
            "contract": {"observation_schema": "canonical_craftax_symbolic"},
            "metrics": {"primary": {"metric": metrics.PRIMARY_METRIC[scenario],
                                    "value": 0.5, "valid_starts": 8}},
            "terminal_label_counts": {},
            "rollout_status": "TESTED_REAL_ENV_RESET",
        }

    engine_cert = certmod.build_certificate(
        _result(FRONT), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        claims=["PROVISIONAL_SELECTION_ONLY"], has_real_rollout=True,
        student_state={"student_checkpoint_loaded": True,
                       "student_policy_rollout_executed": True,
                       "performance_evaluation_executed": True,
                       "scientific_claim_authorized": False},
        mode="performance_evaluation", eval_binding=_engine_binding(),
        finalized=False)
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, "final.inprogress")
        out = os.path.join(td, "final")
        os.makedirs(tmp)
        with open(os.path.join(tmp, "episode_records.jsonl"), "w") as fh:
            fh.write("{}\n")
        _write_json(os.path.join(tmp, "evaluation_result.json"), {"schema": "x"})
        _write_json(os.path.join(tmp, "evaluation_certificate.json"),
                    {FRONT: engine_cert})
        verify_child_artifacts(tmp)
        certs = finalize_certificates(tmp, _prov())
        fb = certs[FRONT]["eval_binding"]
        check("finalize_runner_provenance_injected",
              fb["literal_exit_code"] == 0 and fb["exit_source"] == "wait_pid"
              and fb["inferred_from_log"] is False and fb["child_process_pid"] == 4242
              and len(fb["evaluation_runner_source_sha256"]) == 64)
        # re-load from disk: the finalized certificate passes FULL verification.
        with open(os.path.join(tmp, "evaluation_certificate.json")) as fh:
            reloaded = json.load(fh)
        try:
            certmod.assert_eval_binding_complete(reloaded[FRONT])
            check("finalize_disk_cert_full_verification", True)
        except certmod.FailClosed:
            check("finalize_disk_cert_full_verification", False)
        _write_json(os.path.join(tmp, "run_status.json"),
                    run_status_doc(RUN_STATUS_FINALIZED_PASS, "persistent",
                                   RUN_CLASSES["performance"], _prov(), "7" * 64,
                                   out, False, False))
        write_sha256sums(tmp)
        os.replace(tmp, out)
        check("finalize_atomic_rename",
              os.path.isdir(out) and not os.path.exists(tmp))
        check("finalize_post_rename_sums_verify", verify_dir_sha256sums(out) is True)
        # tamper one byte after finalize → verification fails closed.
        with open(os.path.join(out, "run_status.json"), "a") as fh:
            fh.write(" ")
        try:
            verify_dir_sha256sums(out)
            check("finalize_tamper_detected", False)
        except FailClosed:
            check("finalize_tamper_detected", True)

    # e) NEG37: a certificate already carrying exit provenance (pre-injection) is
    #    rejected at finalize time — only this runner may bind the exit code.
    pre = certmod.build_certificate(
        _result(FRONT), state_bank_hash_label="FRONT_SCAFFOLD_STATE_BANK_HASH",
        claims=["PROVISIONAL_SELECTION_ONLY"], has_real_rollout=True,
        student_state={"student_checkpoint_loaded": True,
                       "student_policy_rollout_executed": True,
                       "performance_evaluation_executed": True,
                       "scientific_claim_authorized": False},
        mode="performance_evaluation", eval_binding=_engine_binding(),
        finalized=False)
    pre["eval_binding"].update(_prov())
    pre["eval_binding"]["evaluation_runner_source_sha256"] = "8" * 64
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, "t")
        os.makedirs(tmp)
        with open(os.path.join(tmp, "episode_records.jsonl"), "w") as fh:
            fh.write("{}\n")
        _write_json(os.path.join(tmp, "evaluation_result.json"), {"schema": "x"})
        _write_json(os.path.join(tmp, "evaluation_certificate.json"), {FRONT: pre})
        try:
            finalize_certificates(tmp, _prov())
            check("NEG37_preinjected_provenance_rejected", False)
        except (FailClosed, certmod.FailClosed):
            check("NEG37_preinjected_provenance_rejected", True)

    # f) Failure path: a FAIL run_status.json lands in the final dir and declares
    #    the temp dir NOT promoted / not finalized — and still binds the FULL
    #    provenance set (总控 §三).
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "final")
        p = _record_failure(out, _prov(literal_exit_code=3), "reset128",
                            RUN_CLASSES["performance"], "7" * 64,
                            RUN_STATUS_ENGINE_FAILED, "engine child exited 3",
                            push_status="BLOCKED_NETWORK")
        with open(p) as fh:
            doc = json.load(fh)
        check("failure_run_status_written",
              doc["status"] == RUN_STATUS_ENGINE_FAILED
              and doc["output_finalized"] is False
              and doc["temp_dir_promoted"] is False
              and doc["literal_exit_code"] == 3
              and doc["exit_source"] == "wait_pid"
              and doc["schema"] == SCHEMA)
        # 总控 §三: every run_status binds local commit SHA / tree SHA / push
        # status / evaluator source SHA / runner source SHA.
        check("run_status_full_provenance_binding",
              "local_commit_sha" in doc and isinstance(doc["local_commit_sha"], str)
              and "local_tree_sha" in doc and isinstance(doc["local_tree_sha"], str)
              and doc["push_status"] == "BLOCKED_NETWORK"
              and len(doc["evaluator_source_sha256"]) == 64
              and len(doc["evaluation_runner_source_sha256"]) == 64
              and doc["evaluator_source_sha256"] == evaluator_source_sha256()
              and doc["evaluation_runner_source_sha256"] == runner_source_sha256())
        # the PASS-path run_status carries the same binding set.
        ok = run_status_doc(RUN_STATUS_FINALIZED_PASS, "persistent",
                            RUN_CLASSES["performance"], _prov(), "7" * 64,
                            out, True, True,
                            push_status="NOT_PUSHED_AT_RUN_TIME")
        check("pass_run_status_full_provenance_binding",
              "local_commit_sha" in ok and "local_tree_sha" in ok
              and ok["push_status"] == "NOT_PUSHED_AT_RUN_TIME"
              and len(ok["evaluator_source_sha256"]) == 64
              and len(ok["evaluation_runner_source_sha256"]) == 64)

    if problems:
        print("TIER3_EVALUATION_RUNNER_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_EVALUATION_RUNNER_SELF_TEST_PASS (literal wait() provenance; "
          "atomic finalize; NEG37/NEG39 guards live)")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return self_test()

    def _opt(flag, default=None):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                return argv[i + 1]
        return default

    perf = "--performance-evaluation" in argv
    smoke = "--interface-smoke" in argv
    if perf == smoke:                       # neither / both → usage error
        print("usage: tier3_evaluation_runner.py --self-test\n"
              "       tier3_evaluation_runner.py --performance-evaluation "
              "--arm {persistent|reset128} --checkpoint <full_state.pkl> "
              "--checkpoint-contract <PATH> --out <DIR> "
              "[--cc2_snapshot_root <PATH>] [--cc2_driver_source <PATH>]\n"
              "       tier3_evaluation_runner.py --interface-smoke "
              "--arm {persistent|reset128} --checkpoint <full_state.pkl> "
              "--checkpoint-contract <PATH> --out <DIR> "
              "[--scenario {front_l2,back_l2,full,all}] [--episodes N] "
              "[--max-steps M] [--cc2_snapshot_root <PATH>] "
              "[--cc2_driver_source <PATH>] [--push-status <STATUS>]\n"
              "(--performance-evaluation and --interface-smoke are mutually exclusive;\n"
              " --push-status records the remote-push outcome, e.g. BLOCKED_NETWORK /\n"
              " PUSHED / NOT_PUSHED_AT_RUN_TIME — default NOT_PUSHED_AT_RUN_TIME)")
        return 3
    arm = _opt("--arm")
    checkpoint = _opt("--checkpoint")
    contract = _opt("--checkpoint-contract")
    out = _opt("--out")
    if not arm or not checkpoint or not contract or not out:
        print("FAIL CLOSED (usage): --arm, --checkpoint, --checkpoint-contract and "
              "--out are required")
        return 3
    try:
        return evaluate_arm(
            arm, checkpoint, contract, out,
            mode="performance" if perf else "smoke",
            scenario=_opt("--scenario", "all"),
            episodes=int(_opt("--episodes")) if _opt("--episodes") else None,
            max_steps=int(_opt("--max-steps")) if _opt("--max-steps") else None,
            cc2_snapshot_root=_opt("--cc2_snapshot_root"),
            cc2_driver_source=_opt("--cc2_driver_source"),
            push_status=_opt("--push-status", "NOT_PUSHED_AT_RUN_TIME"))
    except (FailClosed, certmod.FailClosed, contractmod.FailClosed,
            evaluator.FailClosed) as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
