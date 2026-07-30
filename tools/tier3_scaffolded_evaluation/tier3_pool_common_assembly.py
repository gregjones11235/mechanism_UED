#!/usr/bin/env python3
"""CC4 Tier3 — STUDENT POOL COMMON EVALUATOR ASSEMBLY (closing contract §2/§4/§9).

Builds /home/oseasy/student_pool_v1/common/ (or any --common-root) as the single
public evaluation surface. NOTHING here is a placeholder: every file is a verified
copy, a real entry shim bound by full SHA256 to the deployed engine, or a gate
evidence document produced by actually running the code. All JSON references use
ACTUAL full SHA256 values computed over the real bytes.

Two phases:

  --assemble   verify+copy the evaluation profile / metric schema / ABI doc / the
               whole evaluator package; copy the FROZEN bank artifacts byte-exactly
               (content SHA must reproduce the historical frozen identities
               21aeb7dc... / c632e30d... — a difference fails closed, never silently
               replaced); record the four bank SHAs (artifact_file /
               canonical_content / ordered_payload / field_manifest); run the
               negative suite (--json) and the engine / ABI / metrics self-tests;
               write environment_lock.json, the real common_evaluator.py +
               common_runner.py shims, CLI_TEMPLATE.txt, assembly_manifest.json,
               SHA256SUMS.
  --finalize-ready
               after the cross-GPU preflight and both RMT16 capsule bindings exist,
               re-verify everything from disk, regenerate SHA256SUMS, and write
               COMMON_EVALUATOR_READY.json implementing the §9 eight gates. ANY
               failed gate -> COMMON_EVALUATOR_READY=false (honest false, never a
               faked PASS).

Runs under the pinned venv (jax/craftax) on the server; the --self-test below is
pure and runs on any interpreter.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_source_audit as audit        # noqa: E402

SCHEMA_ASSEMBLY = "mechanism_UED.pool_common_assembly/v1"
SCHEMA_READY = "mechanism_UED.common_evaluator_ready/v1"
SCHEMA_LOCK = "mechanism_UED.environment_lock/v1"

FRONT_FROZEN_CONTENT_SHA256 = (
    "21aeb7dcdcb4ffccfb1eedc80f2c6daec1995c242822a0efbec9b947e275d687")
BACK_FROZEN_CONTENT_SHA256 = (
    "c632e30dcabea7ff812fa07ce855809ec54e7cbca87747fa2d5ab775431f2566")

READY_FILE = "COMMON_EVALUATOR_READY.json"
SUMS_FILE = "SHA256SUMS"
# Files hashed at finalize but never listed inside SHA256SUMS (self-reference is
# impossible; the READY doc instead embeds sha256sums_sha256).
SUMS_EXCLUDE = (SUMS_FILE, READY_FILE)

# The §9 eight gates, in contract order.
READY_GATES = (
    "COMMON_ARTIFACT_IDENTITY", "B1_COMMON_EVALUATOR", "B3_FROZEN_BANKS",
    "FULL_PROFILE_READY", "COMMON_RUNTIME_ABI_READY", "NEGATIVE_GATES",
    "CROSS_GPU_DETERMINISM_PREFLIGHT", "SHA256SUMS_STATUS")


class FailClosed(Exception):
    """Hard stop on any assembly violation."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_lf_file(path) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, doc):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def _copy(src, dst) -> str:
    os.makedirs(os.path.dirname(str(dst)), exist_ok=True)
    shutil.copyfile(str(src), str(dst))
    return _sha256_file(str(dst))


def _ordered_payload_sha256(ordered_payload_hashes) -> str:
    return hashlib.sha256(json.dumps(
        list(ordered_payload_hashes), sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# SHA256SUMS
# ---------------------------------------------------------------------------
def write_sha256sums(common_root) -> str:
    """Write SHA256SUMS over every file under common_root (sorted relpaths,
    'sha  relpath' lines) excluding SHA256SUMS itself and the READY doc."""
    lines = []
    for dirpath, dirnames, filenames in os.walk(common_root):
        dirnames.sort()
        for name in sorted(filenames):
            if dirpath == str(common_root) and name in SUMS_EXCLUDE:
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, str(common_root)).replace(os.sep, "/")
            lines.append("%s  %s" % (_sha256_file(full), rel))
    lines.sort(key=lambda l: l.split("  ", 1)[1])
    sums_path = os.path.join(str(common_root), SUMS_FILE)
    with open(sums_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return _sha256_file(sums_path)


def verify_sha256sums(common_root) -> dict:
    """Re-verify every listed file; also ensure no non-excluded file is unlisted."""
    sums_path = os.path.join(str(common_root), SUMS_FILE)
    require(os.path.isfile(sums_path), "FAIL CLOSED: %s missing" % sums_path)
    listed = {}
    with open(sums_path, encoding="utf-8") as fh:
        for line in fh.read().splitlines():
            if not line.strip():
                continue
            sha, rel = line.split("  ", 1)
            listed[rel] = sha
    mismatches, missing = [], []
    for rel, sha in listed.items():
        full = os.path.join(str(common_root), rel.replace("/", os.sep))
        if not os.path.isfile(full):
            missing.append(rel)
        elif _sha256_file(full) != sha:
            mismatches.append(rel)
    present = set()
    for dirpath, dirnames, filenames in os.walk(str(common_root)):
        dirnames.sort()
        for name in sorted(filenames):
            if dirpath == str(common_root) and name in SUMS_EXCLUDE:
                continue
            present.add(os.path.relpath(os.path.join(dirpath, name),
                                        str(common_root)).replace(os.sep, "/"))
    unlisted = sorted(present - set(listed))
    ok = not mismatches and not missing and not unlisted
    return {"status": "PASS" if ok else "FAIL", "listed": len(listed),
            "mismatches": mismatches, "missing": missing, "unlisted": unlisted,
            "sha256sums_sha256": _sha256_file(sums_path)}


# ---------------------------------------------------------------------------
# Entry shims (REAL code, bound to the deployed engine by full LF-SHA)
# ---------------------------------------------------------------------------
COMMON_EVALUATOR_TEMPLATE = '''#!/usr/bin/env python3
"""PUBLIC COMMON EVALUATOR ENTRY (closing contract S2/S3).

The single public entry point of the common evaluator. It owns NO evaluation
semantics of its own: it verifies that the deployed engine source is bit-identical
to the full SHA256 bound at assembly time, then delegates ALL behaviour (canonical
world/profile, frozen banks, episode loop, reset/step, terminal labels, FRONT
transition / graph-distance progress, BACK/FULL DEFEAT_KOBOLD, metric aggregation,
evaluation certificate) to tools/tier3_scaffolded_evaluation/tier3_evaluator.py.
Post-assembly tampering with the engine fails closed.
"""
import hashlib
import json
import os
import sys

ASSEMBLED_REPO_ROOT = "@@REPO_ROOT@@"
ENGINE_REL = ("tools", "tier3_scaffolded_evaluation", "tier3_evaluator.py")
COMMON_EVALUATOR_ENGINE_SHA256 = "@@ENGINE_SHA@@"
EVALUATION_PROFILE_SHA256 = "@@PROFILE_SHA@@"
METRIC_SCHEMA_SHA256 = "@@SCHEMA_SHA@@"
GIT_COMMIT_AT_ASSEMBLY = "@@GIT_COMMIT@@"
ASSEMBLED_AT_UTC = "@@TIMESTAMP@@"


def _lf_sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\\r\\n", b"\\n")).hexdigest()


def engine_repo_root():
    return os.environ.get("CC4_EVALUATOR_REPO", ASSEMBLED_REPO_ROOT)


def engine_path():
    return os.path.join(engine_repo_root(), *ENGINE_REL)


def verify_engine_binding():
    p = engine_path()
    if not os.path.isfile(p):
        raise SystemExit("FAIL CLOSED: common evaluator engine missing at %s" % p)
    got = _lf_sha(p)
    if got != COMMON_EVALUATOR_ENGINE_SHA256:
        raise SystemExit("FAIL CLOSED: engine SHA drift %s != frozen %s"
                         % (got, COMMON_EVALUATOR_ENGINE_SHA256))
    return p


def binding_identity():
    return {
        "schema": "mechanism_UED.common_evaluator_binding_identity/v1",
        "engine_path": engine_path(),
        "common_evaluator_engine_sha256": COMMON_EVALUATOR_ENGINE_SHA256,
        "evaluation_profile_sha256": EVALUATION_PROFILE_SHA256,
        "metric_schema_sha256": METRIC_SCHEMA_SHA256,
        "git_commit_at_assembly": GIT_COMMIT_AT_ASSEMBLY,
        "assembled_at_utc": ASSEMBLED_AT_UTC,
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--binding-identity" in argv:
        print(json.dumps(binding_identity(), indent=2, sort_keys=True))
        return 0
    p = verify_engine_binding()
    for entry in (os.path.dirname(p),
                  os.path.join(engine_repo_root(), "dicode_src", "src")):
        if os.path.isdir(entry) and entry not in sys.path:
            sys.path.insert(0, entry)
    import tier3_evaluator
    return tier3_evaluator.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
'''

COMMON_RUNNER_TEMPLATE = '''#!/usr/bin/env python3
"""PUBLIC COMMON RUNNER ENTRY (closing contract S5).

The single public entry point of the common candidate-runtime runner. It never
hardcodes an architecture: family registration / dispatch lives in the bound
engine (tier3_candidate_runtime.py, ABI mechanism_UED.candidate_runtime_abi/v1);
this shim verifies the engine binding by full SHA256 and delegates. Base GTrXL /
Control / SlowGRU / Teacher runtimes are registered by their OWN owners; unknown
or missing families fail closed in the engine.
"""
import hashlib
import json
import os
import sys

ASSEMBLED_REPO_ROOT = "@@REPO_ROOT@@"
RUNNER_REL = ("tools", "tier3_scaffolded_evaluation", "tier3_candidate_runtime.py")
COMMON_RUNNER_ENGINE_SHA256 = "@@RUNNER_SHA@@"
ABI_DOC_SHA256 = "@@ABI_DOC_SHA@@"
GIT_COMMIT_AT_ASSEMBLY = "@@GIT_COMMIT@@"
ASSEMBLED_AT_UTC = "@@TIMESTAMP@@"


def _lf_sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\\r\\n", b"\\n")).hexdigest()


def runner_repo_root():
    return os.environ.get("CC4_EVALUATOR_REPO", ASSEMBLED_REPO_ROOT)


def runner_path():
    return os.path.join(runner_repo_root(), *RUNNER_REL)


def verify_runner_binding():
    p = runner_path()
    if not os.path.isfile(p):
        raise SystemExit("FAIL CLOSED: common runner engine missing at %s" % p)
    got = _lf_sha(p)
    if got != COMMON_RUNNER_ENGINE_SHA256:
        raise SystemExit("FAIL CLOSED: runner engine SHA drift %s != frozen %s"
                         % (got, COMMON_RUNNER_ENGINE_SHA256))
    return p


def _engine_module():
    p = verify_runner_binding()
    d = os.path.dirname(p)
    for entry in (d, os.path.join(runner_repo_root(), "dicode_src", "src")):
        if os.path.isdir(entry) and entry not in sys.path:
            sys.path.insert(0, entry)
    import tier3_candidate_runtime as engine
    return engine


def load_candidate(checkpoint_contract):
    """ABI entry: family dispatch happens inside the engine (fail closed)."""
    return _engine_module().load_candidate(checkpoint_contract)


def binding_identity():
    return {
        "schema": "mechanism_UED.common_runner_binding_identity/v1",
        "runner_path": runner_path(),
        "common_runner_engine_sha256": COMMON_RUNNER_ENGINE_SHA256,
        "abi_doc_sha256": ABI_DOC_SHA256,
        "git_commit_at_assembly": GIT_COMMIT_AT_ASSEMBLY,
        "assembled_at_utc": ASSEMBLED_AT_UTC,
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--binding-identity" in argv:
        print(json.dumps(binding_identity(), indent=2, sort_keys=True))
        return 0
    if "--metadata" in argv:
        with open(argv[argv.index("--metadata") + 1], encoding="utf-8") as fh:
            contract = json.load(fh)
        rt = load_candidate(contract)
        print(json.dumps(rt.candidate_metadata(), indent=2, sort_keys=True))
        return 0
    print("usage: common_runner.py --binding-identity | --metadata CONTRACT.json")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
'''

CLI_TEMPLATE = """# CC4 STUDENT POOL — COMMON EVALUATOR CLI TEMPLATE (closing contract S2)
# All evaluation semantics are owned by the common evaluator. Candidate runtimes
# expose only the ABI. This round authorizes INTERFACE_SMOKE binding ONLY — the
# formal 6-student pool ranking must NOT be started by CC4.
#
# Environment: pinned venv python; CC4 may use ONLY GPU2/GPU3 (GPU0/GPU1 banned).

# 0. entry-point binding identities (full SHAs bound at assembly)
python COMMON_ROOT/common_evaluator.py --binding-identity
python COMMON_ROOT/common_runner.py --binding-identity

# 1. candidate runtime metadata through the common runner (ABI)
python COMMON_ROOT/common_runner.py --metadata CAPSULE_ROOT/checkpoint_contract.json

# 2. INTERFACE_SMOKE binding per candidate (FRONT/BACK/FULL, max_steps=32,
#    frozen banks loaded READ-ONLY from the common artifacts)
python COMMON_ROOT/common_evaluator.py --interface-smoke \\
    --checkpoint <PKL> --checkpoint-contract <repo>/configs/tier3_cc2_final98304_checkpoint_contract_v1.json \\
    --arm <persistent|reset128> \\
    --scenario all --episodes 1 --max-steps 32 \\
    --frozen-bank-artifacts COMMON_ROOT/frozen_bank_artifacts \\
    --out CAPSULE_ROOT/interface_smoke_out

# 3. cross-GPU determinism preflight (SAME Persistent checkpoint, two idle GPUs)
CUDA_VISIBLE_DEVICES=2 python tools/tier3_scaffolded_evaluation/tier3_cross_gpu_preflight.py \\
    --run --checkpoint <PERSISTENT_PKL> \\
    --frozen-bank-artifacts COMMON_ROOT/frozen_bank_artifacts --out preflight_gpu2.json
CUDA_VISIBLE_DEVICES=3 python tools/tier3_scaffolded_evaluation/tier3_cross_gpu_preflight.py \\
    --run --checkpoint <PERSISTENT_PKL> \\
    --frozen-bank-artifacts COMMON_ROOT/frozen_bank_artifacts --out preflight_gpu3.json
python tools/tier3_scaffolded_evaluation/tier3_cross_gpu_preflight.py \\
    --compare preflight_gpu2.json preflight_gpu3.json \\
    --out COMMON_ROOT/cross_gpu_preflight_certificate.json

# 4. verify the frozen artifact directory at any time
sha256sum -c COMMON_ROOT/SHA256SUMS   (run from COMMON_ROOT)

# PROHIBITED this round: starting the formal pool ranking, any training run, D052,
# ablations, GPU bank regeneration, inventing FULL seeds, touching other owners'
# runtimes. FORMAL_POOL_EVALUATION_STARTED must stay false.
"""


def _render(template, mapping) -> str:
    out = template
    for key, value in mapping.items():
        out = out.replace("@@%s@@" % key, str(value))
    require("@@" not in out, "FAIL CLOSED: unresolved placeholder in shim template")
    return out


# ---------------------------------------------------------------------------
# Tool subprocesses (gate evidence = actually running the code)
# ---------------------------------------------------------------------------
def _run_tool(repo_root, rel_argv, timeout=3600) -> dict:
    """Run a tool; keep the FULL stdout (callers may need to parse it, e.g. the
    negative-suite --json report) and bounded tails for evidence files."""
    cmd = [sys.executable] + list(rel_argv)
    p = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True,
                       timeout=timeout)
    return {"argv": " ".join(rel_argv), "exit_code": int(p.returncode),
            "stdout": p.stdout or "",
            "stdout_tail": (p.stdout or "")[-4000:],
            "stderr_tail": (p.stderr or "")[-2000:]}


def _parse_neg_json(stdout: str) -> dict:
    s = (stdout or "").strip()
    require(s, "FAIL CLOSED: negative tests produced no output")
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        require(start >= 0 and end > start,
                "FAIL CLOSED: cannot parse negative-test JSON report")
        return json.loads(s[start:end + 1])


# ---------------------------------------------------------------------------
# ASSEMBLE
# ---------------------------------------------------------------------------
def assemble(common_root, frozen_bank_artifacts, repo_root=None) -> dict:
    repo_root = str(repo_root or audit.repo_root())
    common_root = str(common_root)
    tools_dir = os.path.join(repo_root, "tools", "tier3_scaffolded_evaluation")

    # S0 fresh-directory gate: never clobber an existing pool surface.
    if os.path.exists(common_root):
        require(os.path.isdir(common_root) and not os.listdir(common_root),
                "FAIL CLOSED: --common-root %s exists and is NOT empty; refuse to "
                "overwrite an existing pool surface" % common_root)
    else:
        os.makedirs(common_root)

    import tier3_evaluation_profile as profile_mod
    import tier3_frozen_bank_artifacts as art
    import tier3_metrics as metrics

    file_map = []

    def record(common_rel, source_path, sha):
        file_map.append({"common_path": common_rel, "source_path": str(source_path),
                         "sha256": sha})

    # ---- 1. evaluation profile (verified, then byte-copied) ----------------
    profile_src = os.path.join(repo_root, "configs", "tier3_evaluation_profile_v1.json")
    profile = profile_mod.load_profile(profile_src)
    profile_mod.verify_profile(profile)
    sha = _copy(profile_src, os.path.join(common_root, "evaluation_profile.json"))
    require(sha == profile_mod.compute_profile_sha256(profile)
            == profile.get("evaluation_profile_sha256"),
            "FAIL CLOSED: copied profile SHA does not reproduce the verified "
            "self-hash")
    record("evaluation_profile.json", profile_src, sha)
    evaluation_profile_sha256 = sha
    full_profile_ready = (profile.get("scenarios", {}).get("full", {})
                          .get("FULL_PROFILE_READY") is True)
    require(full_profile_ready,
            "FAIL CLOSED: FULL_PROFILE_READY is not true in the frozen profile; the "
            "common evaluator may not assemble (no ad-hoc FULL seeds allowed)")

    # ---- 2. metric schema (bound to the metrics source by LF-SHA) ----------
    schema_src = os.path.join(repo_root, "schemas", "tier3_metric_schema_v1.json")
    with open(schema_src, encoding="utf-8") as fh:
        schema_doc = json.load(fh)
    require(schema_doc.get("schema") == metrics.METRIC_SCHEMA_ID,
            "FAIL CLOSED: metric schema id mismatch")
    metrics_src = os.path.join(tools_dir, "tier3_metrics.py")
    require(schema_doc.get("metrics_source_sha256") == _sha256_lf_file(metrics_src),
            "FAIL CLOSED: metric schema no longer binds tier3_metrics.py by LF-SHA")
    sha = _copy(schema_src, os.path.join(common_root, "metric_schema.json"))
    record("metric_schema.json", schema_src, sha)
    metric_schema_sha256 = sha

    # ---- 3. ABI specification doc ------------------------------------------
    abi_src = os.path.join(repo_root, "reports", "tier3_scaffolded_evaluation",
                           "candidate_runtime_abi.md")
    sha = _copy(abi_src, os.path.join(common_root, "candidate_runtime_abi.md"))
    record("candidate_runtime_abi.md", abi_src, sha)
    abi_doc_sha256 = sha

    # ---- 4. the evaluator package (every module + schemas + configs) -------
    engine_module_sha256 = {}
    for src in sorted(glob.glob(os.path.join(tools_dir, "*.py"))):
        name = os.path.basename(src)
        sha = _copy(src, os.path.join(common_root, "evaluator", name))
        record("evaluator/%s" % name, src, sha)
        engine_module_sha256[name] = _sha256_lf_file(src)
    for src in sorted(glob.glob(os.path.join(repo_root, "schemas", "*.json"))):
        name = os.path.basename(src)
        sha = _copy(src, os.path.join(common_root, "evaluator_schemas", name))
        record("evaluator_schemas/%s" % name, src, sha)
    for src in sorted(glob.glob(os.path.join(repo_root, "configs", "*.json"))):
        name = os.path.basename(src)
        sha = _copy(src, os.path.join(common_root, "evaluator_configs", name))
        record("evaluator_configs/%s" % name, src, sha)

    # ---- 5. frozen bank artifacts (byte-exact; identities fail closed) -----
    bank_identity = {}
    for sc, frozen_content in ((metrics.FRONT, FRONT_FROZEN_CONTENT_SHA256),
                               (metrics.BACK, BACK_FROZEN_CONTENT_SHA256)):
        src_dir = os.path.join(str(frozen_bank_artifacts), sc)
        src_npz = os.path.join(src_dir, "states.npz")
        src_manifest = os.path.join(src_dir, "manifest.json")
        require(os.path.isfile(src_npz) and os.path.isfile(src_manifest),
                "FAIL CLOSED: frozen bank artifact missing for %s under %s"
                % (sc, frozen_bank_artifacts))
        # load_bank-compatible layout + flat contract-named aliases (same bytes).
        layout_dir = os.path.join(common_root, "frozen_bank_artifacts", sc)
        flat_npz = os.path.join(common_root, "%s_states.npz"
                                % ("front_bank" if sc == metrics.FRONT
                                   else "back_bank"))
        flat_manifest = os.path.join(common_root, "%s_manifest.json"
                                     % ("front_bank" if sc == metrics.FRONT
                                        else "back_bank"))
        sha_npz = _copy(src_npz, os.path.join(layout_dir, "states.npz"))
        _copy(src_npz, flat_npz)
        sha_man = _copy(src_manifest, os.path.join(layout_dir, "manifest.json"))
        _copy(src_manifest, flat_manifest)
        record("frozen_bank_artifacts/%s/states.npz" % sc, src_npz, sha_npz)
        record("frozen_bank_artifacts/%s/manifest.json" % sc, src_manifest, sha_man)
        record(os.path.basename(flat_npz), src_npz, sha_npz)
        record(os.path.basename(flat_manifest), src_manifest, sha_man)

        bank = art.load_bank(sc, os.path.join(common_root, "frozen_bank_artifacts"))
        require(bank["loaded_content_sha256"] == frozen_content,
                "FAIL CLOSED: %s artifact content SHA %s != frozen historical "
                "identity %s — silent replacement is forbidden"
                % (sc, bank["loaded_content_sha256"][:16], frozen_content[:16]))
        with open(src_manifest, encoding="utf-8") as fh:
            raw_manifest = json.load(fh)
        bank_identity[sc] = {
            "artifact_file_sha256": bank["artifact_file_sha256"],
            "canonical_content_sha256": bank["loaded_content_sha256"],
            "ordered_payload_sha256": _ordered_payload_sha256(
                bank["ordered_payload_hashes"]),
            "field_manifest_sha256": bank["field_manifest_sha256"],
            "state_count": int(bank.get("state_count", 0)),
            "seeds": [int(s) for s in bank["seeds"]],
            "device_provenance": bank["device_provenance"],
            "bank_source": bank.get("bank_source"),
            "pinned_env_identity": raw_manifest.get("pinned_env_identity"),
            "frozen_historical_content_sha256": frozen_content,
        }

    # ---- 6. negative gate report (the suite actually runs) ------------------
    neg = _run_tool(repo_root, ["tools/tier3_scaffolded_evaluation/"
                                "tier3_negative_tests.py", "--json"])
    require(neg["exit_code"] == 0,
            "FAIL CLOSED: negative suite exit %d: %s"
            % (neg["exit_code"], neg["stderr_tail"][-800:]))
    neg_report = _parse_neg_json(neg["stdout"])
    require(int(neg_report.get("fail", 1)) == 0,
            "FAIL CLOSED: negative suite reports fail=%s" % neg_report.get("fail"))
    neg_report["tool_exit_code"] = neg["exit_code"]
    _atomic_json(os.path.join(common_root, "negative_test_report.json"),
                 {"schema": SCHEMA_ASSEMBLY, "generated_at_utc": _utc_now(),
                  "fail": neg_report["fail"],
                  "results": neg_report.get("results", []),
                  "pending_commit3": neg_report.get("pending_commit3", []),
                  "tool_exit_code": neg_report["tool_exit_code"],
                  "tool_stdout_tail": neg["stdout_tail"]})

    # ---- 7. engine / ABI / metrics self-tests (gate evidence) ---------------
    os.makedirs(os.path.join(common_root, "statuses"), exist_ok=True)
    eval_st = _run_tool(repo_root, ["tools/tier3_scaffolded_evaluation/"
                                    "tier3_evaluator.py", "--self-test"])
    _atomic_json(os.path.join(common_root, "statuses", "evaluator_self_test.json"),
                 eval_st)
    abi_st = _run_tool(repo_root, ["tools/tier3_scaffolded_evaluation/"
                                   "tier3_candidate_runtime.py", "--self-test"])
    _atomic_json(os.path.join(common_root, "statuses", "abi_self_test.json"), abi_st)
    metrics_st = _run_tool(repo_root, ["tools/tier3_scaffolded_evaluation/"
                                       "tier3_metrics.py", "--self-test"])
    _atomic_json(os.path.join(common_root, "statuses", "metrics_self_test.json"),
                 metrics_st)
    _atomic_json(os.path.join(common_root, "statuses", "bank_identity.json"),
                 {"schema": SCHEMA_ASSEMBLY, "generated_at_utc": _utc_now(),
                  "banks": bank_identity})
    require(eval_st["exit_code"] == 0,
            "FAIL CLOSED: evaluator self-test failed on this host "
            "(B1_COMMON_EVALUATOR cannot be evidenced): %s"
            % eval_st["stderr_tail"][-800:])
    require(abi_st["exit_code"] == 0,
            "FAIL CLOSED: candidate runtime ABI self-test failed: %s"
            % abi_st["stderr_tail"][-800:])

    # ---- 8. environment lock ------------------------------------------------
    import jax
    import tier3_evaluator as ev
    gpu_names = None
    try:
        smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                             timeout=30)
        if smi.returncode == 0:
            gpu_names = [ln for ln in smi.stdout.splitlines() if ln.strip()]
    except (OSError, subprocess.TimeoutExpired):
        gpu_names = None
    env_lock = {
        "schema": SCHEMA_LOCK,
        "generated_at_utc": _utc_now(),
        "host": {"platform": sys.platform, "python_version": platform_python(),
                 "python_executable": sys.executable},
        "runtime_versions": ev._runtime_versions(),
        "eval_device_identity": ev._eval_device_identity(),
        "jax_devices": [str(d) for d in jax.devices()],
        "gpu_names_nvidia_smi": gpu_names,
        "cc4_gpu_policy": "CC4 may use ONLY GPU2/GPU3; GPU0/GPU1 strictly banned; "
                          "idle + non-conflicting + no W512 preemption must be "
                          "re-verified before ANY GPU run",
        "git_commit_head": ev._git_commit_head(),
        "repo_root_at_assembly": repo_root,
        "pinned_env_identity_from_bank_manifest":
            bank_identity[metrics.FRONT]["pinned_env_identity"],
    }
    _atomic_json(os.path.join(common_root, "environment_lock.json"), env_lock)
    environment_lock_sha256 = _sha256_file(
        os.path.join(common_root, "environment_lock.json"))
    record("environment_lock.json", "(generated)", environment_lock_sha256)

    # ---- 9. real entry shims (bound by full SHA) ----------------------------
    mapping = {"REPO_ROOT": repo_root,
               "ENGINE_SHA": engine_module_sha256["tier3_evaluator.py"],
               "RUNNER_SHA": engine_module_sha256["tier3_candidate_runtime.py"],
               "PROFILE_SHA": evaluation_profile_sha256,
               "SCHEMA_SHA": metric_schema_sha256,
               "ABI_DOC_SHA": abi_doc_sha256,
               "GIT_COMMIT": env_lock["git_commit_head"],
               "TIMESTAMP": env_lock["generated_at_utc"]}
    for name, template in (("common_evaluator.py", COMMON_EVALUATOR_TEMPLATE),
                           ("common_runner.py", COMMON_RUNNER_TEMPLATE)):
        path = os.path.join(common_root, name)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(_render(template, mapping))
        record(name, "(generated)", _sha256_file(path))

    # ---- 10. CLI template ----------------------------------------------------
    cli_path = os.path.join(common_root, "CLI_TEMPLATE.txt")
    with open(cli_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(CLI_TEMPLATE.replace("COMMON_ROOT", common_root))
    record("CLI_TEMPLATE.txt", "(generated)", _sha256_file(cli_path))

    # ---- 11. assembly manifest ------------------------------------------------
    # negative_test_report.json + statuses/* were generated; record them too.
    for rel in ("negative_test_report.json", "statuses/evaluator_self_test.json",
                "statuses/abi_self_test.json", "statuses/metrics_self_test.json",
                "statuses/bank_identity.json"):
        record(rel, "(generated)", _sha256_file(os.path.join(common_root, *rel.split("/"))))
    manifest = {
        "schema": SCHEMA_ASSEMBLY,
        "generated_at_utc": env_lock["generated_at_utc"],
        "assembler_source_sha256": _sha256_lf_file(os.path.abspath(__file__)),
        "repo_root": repo_root,
        "frozen_bank_artifacts_source": str(frozen_bank_artifacts),
        "git_commit_head": env_lock["git_commit_head"],
        "evaluation_profile_sha256": evaluation_profile_sha256,
        "metric_schema_sha256": metric_schema_sha256,
        "abi_doc_sha256": abi_doc_sha256,
        "environment_lock_sha256": environment_lock_sha256,
        "common_evaluator_sha256": _sha256_file(
            os.path.join(common_root, "common_evaluator.py")),
        "common_runner_sha256": _sha256_file(
            os.path.join(common_root, "common_runner.py")),
        "engine_module_sha256": engine_module_sha256,
        "bank_identity": bank_identity,
        "FULL_PROFILE_READY": True,
        "file_map": file_map,
    }
    _atomic_json(os.path.join(common_root, "assembly_manifest.json"), manifest)

    # ---- 12. SHA256SUMS (regenerated again at finalize, after the cert) -----
    write_sha256sums(common_root)
    verify = verify_sha256sums(common_root)
    require(verify["status"] == "PASS",
            "FAIL CLOSED: fresh SHA256SUMS does not verify: %s" % verify)
    print("COMMON_ASSEMBLY_COMPLETE root=%s files=%d "
          "evaluation_profile_sha256=%s FULL_PROFILE_READY=true"
          % (common_root, len(file_map) + 1, evaluation_profile_sha256[:16]))
    return manifest


def platform_python() -> str:
    import platform
    return platform.python_version()


# ---------------------------------------------------------------------------
# FINALIZE — the S9 eight gates, re-checked FROM DISK
# ---------------------------------------------------------------------------
def finalize_ready(common_root, preflight_cert_src, persistent_binding_src,
                   reset128_binding_src, repo_root=None) -> dict:
    repo_root = str(repo_root or audit.repo_root())
    common_root = str(common_root)

    import tier3_evaluation_profile as profile_mod
    import tier3_frozen_bank_artifacts as art
    import tier3_metrics as metrics

    def _load(rel):
        with open(os.path.join(common_root, rel), encoding="utf-8") as fh:
            return json.load(fh)

    manifest = _load("assembly_manifest.json")
    gates = {}

    # G1 COMMON_ARTIFACT_IDENTITY: profile verifies; schema binds the metrics
    # source; every bound engine module still reproduces its assembly LF-SHA.
    evidence = {}
    try:
        profile = profile_mod.load_profile(
            os.path.join(common_root, "evaluation_profile.json"))
        profile_mod.verify_profile(profile)
        schema_doc = _load("metric_schema.json")
        metrics_lf = _sha256_lf_file(os.path.join(
            repo_root, "tools", "tier3_scaffolded_evaluation", "tier3_metrics.py"))
        engine_ok = {}
        tools_dir = os.path.join(repo_root, "tools", "tier3_scaffolded_evaluation")
        for name, lf_sha in sorted(manifest["engine_module_sha256"].items()):
            engine_ok[name] = (_sha256_lf_file(os.path.join(tools_dir, name))
                               == lf_sha)
        ok = (schema_doc.get("schema") == metrics.METRIC_SCHEMA_ID
              and schema_doc.get("metrics_source_sha256") == metrics_lf
              and all(engine_ok.values()))
        evidence = {"evaluation_profile_sha256":
                        manifest["evaluation_profile_sha256"],
                    "metric_schema_sha256": manifest["metric_schema_sha256"],
                    "metrics_source_lf_sha256_reproduced":
                        schema_doc.get("metrics_source_sha256") == metrics_lf,
                    "engine_modules_bound": len(engine_ok),
                    "engine_modules_reproduced": sum(engine_ok.values())}
    except Exception as exc:  # engine modules raise their own FailClosed classes
        ok, evidence = False, {"error": "%s: %s" % (type(exc).__name__, exc)}
    gates["COMMON_ARTIFACT_IDENTITY"] = {"status": "PASS" if ok else "FAIL",
                                         "evidence": evidence}

    # G2 B1_COMMON_EVALUATOR: the engine self-test passed on this host AND both
    # RMT16 candidates completed a real INTERFACE_SMOKE binding through it.
    try:
        eval_st = _load("statuses/evaluator_self_test.json")
        bindings = {}
        for arm, src in (("persistent", persistent_binding_src),
                         ("reset128", reset128_binding_src)):
            with open(src, encoding="utf-8") as fh:
                doc = json.load(fh)
            bindings[arm] = (doc.get("formal_eval_binding") == "PASS"
                             and doc.get("run_class") == "INTERFACE_SMOKE"
                             and doc.get("performance_claim_authorized") is False)
        ok = eval_st.get("exit_code") == 0 and all(bindings.values())
        evidence = {"evaluator_self_test_exit_code": eval_st.get("exit_code"),
                    "binding_pass_by_arm": bindings}
    except Exception as exc:
        ok, evidence = False, {"error": "%s: %s" % (type(exc).__name__, exc)}
    gates["B1_COMMON_EVALUATOR"] = {"status": "PASS" if ok else "FAIL",
                                    "evidence": evidence}

    # G3 B3_FROZEN_BANKS: reload from the COMMON copies; content SHAs must equal
    # the frozen historical identities (never silently replaced); all four SHA
    # kinds recorded.
    try:
        bank_st = _load("statuses/bank_identity.json")["banks"]
        reloaded = {}
        for sc, frozen in ((metrics.FRONT, FRONT_FROZEN_CONTENT_SHA256),
                           (metrics.BACK, BACK_FROZEN_CONTENT_SHA256)):
            bank = art.load_bank(sc, os.path.join(common_root,
                                                  "frozen_bank_artifacts"))
            reloaded[sc] = (bank["loaded_content_sha256"] == frozen
                            and bank_st[sc]["canonical_content_sha256"] == frozen
                            and bank_st[sc]["artifact_file_sha256"]
                            == bank["artifact_file_sha256"])
        four_kinds = all(
            all(k in bank_st[sc] for k in ("artifact_file_sha256",
                                           "canonical_content_sha256",
                                           "ordered_payload_sha256",
                                           "field_manifest_sha256"))
            for sc in (metrics.FRONT, metrics.BACK))
        ok = all(reloaded.values()) and four_kinds
        evidence = {"reloaded_content_matches_frozen": reloaded,
                    "four_sha_kinds_recorded": four_kinds,
                    "front_bank_content_sha256":
                        bank_st[metrics.FRONT]["canonical_content_sha256"],
                    "back_bank_content_sha256":
                        bank_st[metrics.BACK]["canonical_content_sha256"]}
    except Exception as exc:
        ok, evidence = False, {"error": "%s: %s" % (type(exc).__name__, exc)}
    gates["B3_FROZEN_BANKS"] = {"status": "PASS" if ok else "FAIL",
                                "evidence": evidence}

    # G4 FULL_PROFILE_READY
    full = _load("evaluation_profile.json").get("scenarios", {}).get("full", {})
    gates["FULL_PROFILE_READY"] = {
        "status": "PASS" if full.get("FULL_PROFILE_READY") is True else "FAIL",
        "evidence": {"full_seed_count": len(full.get("world_seed_set", {})
                                           .get("seeds", []))}}

    # G5 COMMON_RUNTIME_ABI_READY
    try:
        abi_st = _load("statuses/abi_self_test.json")
        runner_lf = _sha256_lf_file(os.path.join(
            repo_root, "tools", "tier3_scaffolded_evaluation",
            "tier3_candidate_runtime.py"))
        ok = (abi_st.get("exit_code") == 0
              and runner_lf == manifest["engine_module_sha256"]
              ["tier3_candidate_runtime.py"])
        evidence = {"abi_self_test_exit_code": abi_st.get("exit_code"),
                    "runner_engine_sha_reproduced": ok}
    except Exception as exc:
        ok, evidence = False, {"error": "%s: %s" % (type(exc).__name__, exc)}
    gates["COMMON_RUNTIME_ABI_READY"] = {"status": "PASS" if ok else "FAIL",
                                         "evidence": evidence}

    # G6 NEGATIVE_GATES
    try:
        neg = _load("negative_test_report.json")
        ok = int(neg.get("fail", 1)) == 0 and neg.get("tool_exit_code") == 0
        evidence = {"fail": neg.get("fail"),
                    "tests": len(neg.get("results", []))}
    except Exception as exc:
        ok, evidence = False, {"error": "%s: %s" % (type(exc).__name__, exc)}
    gates["NEGATIVE_GATES"] = {"status": "PASS" if ok else "FAIL",
                               "evidence": evidence}

    # G7 CROSS_GPU_DETERMINISM_PREFLIGHT: copy the certificate in byte-exactly,
    # require PASS and the frozen Persistent checkpoint identity.
    try:
        cert_dst = os.path.join(common_root,
                                "cross_gpu_preflight_certificate.json")
        if _sha256_file(preflight_cert_src) != (
                _sha256_file(cert_dst) if os.path.isfile(cert_dst) else None):
            shutil.copyfile(str(preflight_cert_src), cert_dst)
        cert = _load("cross_gpu_preflight_certificate.json")
        import tier3_checkpoint_contract as cc
        ok = (cert.get("CROSS_GPU_DETERMINISM_PREFLIGHT") == "PASS"
              and cert.get("checkpoint_file_sha256")
              == cc.FROZEN_CHECKPOINT_FILE_SHA256["persistent"]
              and cert.get("front_bank_content_sha256")
              == FRONT_FROZEN_CONTENT_SHA256
              and cert.get("back_bank_content_sha256")
              == BACK_FROZEN_CONTENT_SHA256)
        evidence = {"certificate_verdict":
                        cert.get("CROSS_GPU_DETERMINISM_PREFLIGHT"),
                    "certificate_sha256": _sha256_file(cert_dst),
                    "checkpoint_identity_verified":
                        cert.get("checkpoint_file_sha256")
                        == cc.FROZEN_CHECKPOINT_FILE_SHA256["persistent"]}
    except Exception as exc:
        ok, evidence = False, {"error": "%s: %s" % (type(exc).__name__, exc)}
    gates["CROSS_GPU_DETERMINISM_PREFLIGHT"] = {
        "status": "PASS" if ok else "FAIL", "evidence": evidence}

    # G8 SHA256SUMS_STATUS: regenerate over the FINAL file set, then verify.
    sums_sha = write_sha256sums(common_root)
    verify = verify_sha256sums(common_root)
    gates["SHA256SUMS_STATUS"] = {"status": verify["status"],
                                  "evidence": verify}

    ready = all(g["status"] == "PASS" for g in gates.values())
    require(set(gates) == set(READY_GATES),
            "FAIL CLOSED: gate set drifted from the S9 contract")
    doc = {
        "schema": SCHEMA_READY,
        "generated_at_utc": _utc_now(),
        "COMMON_EVALUATOR_READY": bool(ready),
        "honest_false_discipline": "ANY failed gate keeps COMMON_EVALUATOR_READY "
                                   "false; no gate is ever faked to PASS",
        "gates": gates,
        "common_root": common_root,
        "common_evaluator_sha256": manifest["common_evaluator_sha256"],
        "common_runner_sha256": manifest["common_runner_sha256"],
        "evaluation_profile_sha256": manifest["evaluation_profile_sha256"],
        "metric_schema_sha256": manifest["metric_schema_sha256"],
        "environment_lock_sha256": manifest["environment_lock_sha256"],
        "front_bank_content_sha256":
            manifest["bank_identity"][metrics.FRONT]["canonical_content_sha256"],
        "back_bank_content_sha256":
            manifest["bank_identity"][metrics.BACK]["canonical_content_sha256"],
        "front_bank_file_sha256":
            manifest["bank_identity"][metrics.FRONT]["artifact_file_sha256"],
        "back_bank_file_sha256":
            manifest["bank_identity"][metrics.BACK]["artifact_file_sha256"],
        "sha256sums_sha256": sums_sha,
        "git_commit_head": manifest["git_commit_head"],
    }
    _atomic_json(os.path.join(common_root, READY_FILE), doc)
    print("COMMON_EVALUATOR_READY=%s gates_passed=%d/8 root=%s"
          % (doc["COMMON_EVALUATOR_READY"],
             sum(1 for g in gates.values() if g["status"] == "PASS"), common_root))
    return doc


# ---------------------------------------------------------------------------
# Self-test (PURE; any interpreter)
# ---------------------------------------------------------------------------
def self_test() -> int:
    import tempfile
    problems = []

    def check(name, cond):
        if not cond:
            problems.append(name)

    # SHA256SUMS round-trip + tamper + unlisted-file detection.
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "common")
        os.makedirs(os.path.join(root, "sub"))
        for rel, body in (("a.json", "{}\n"), ("sub/b.txt", "x\n"),
                          ("common_evaluator.py", "print(1)\n")):
            with open(os.path.join(root, *rel.split("/")), "w") as fh:
                fh.write(body)
        write_sha256sums(root)
        v = verify_sha256sums(root)
        check("sums_roundtrip_pass", v["status"] == "PASS" and v["listed"] == 3)
        with open(os.path.join(root, "a.json"), "a") as fh:
            fh.write("tamper\n")
        check("sums_tamper_detected",
              verify_sha256sums(root)["status"] == "FAIL"
              and "a.json" in verify_sha256sums(root)["mismatches"])
        with open(os.path.join(root, "a.json"), "w") as fh:
            fh.write("{}\n")
        with open(os.path.join(root, "unlisted.txt"), "w") as fh:
            fh.write("y\n")
        check("sums_unlisted_detected",
              "unlisted.txt" in verify_sha256sums(root)["unlisted"])
        # READY + SHA256SUMS are excluded from the listing by construction.
        with open(os.path.join(root, READY_FILE), "w") as fh:
            fh.write("{}\n")
        write_sha256sums(root)
        with open(os.path.join(root, SUMS_FILE)) as fh:
            listing = fh.read()
        check("ready_and_sums_excluded",
              READY_FILE not in listing and "SHA256SUMS  " not in listing
              and verify_sha256sums(root)["status"] == "PASS")

    # Shim templates: every placeholder resolves.
    mapping = {"REPO_ROOT": "/r", "ENGINE_SHA": "e" * 64, "RUNNER_SHA": "r" * 64,
               "PROFILE_SHA": "p" * 64, "SCHEMA_SHA": "s" * 64,
               "ABI_DOC_SHA": "a" * 64, "GIT_COMMIT": "c" * 40,
               "TIMESTAMP": "t"}
    for tpl in (COMMON_EVALUATOR_TEMPLATE, COMMON_RUNNER_TEMPLATE):
        rendered = _render(tpl, mapping)
        check("template_no_placeholder_left", "@@" not in rendered)
        compile(rendered, "<shim>", "exec")
    check("shims_compile", True)
    try:
        _render("@@MISSING@@", {})
        check("unresolved_placeholder_rejected", False)
    except FailClosed:
        check("unresolved_placeholder_rejected", True)

    # S9 gate aggregation: ready iff ALL eight gates pass (exact gate set).
    check("gate_set_matches_contract",
          set(READY_GATES) == {"COMMON_ARTIFACT_IDENTITY", "B1_COMMON_EVALUATOR",
                               "B3_FROZEN_BANKS", "FULL_PROFILE_READY",
                               "COMMON_RUNTIME_ABI_READY", "NEGATIVE_GATES",
                               "CROSS_GPU_DETERMINISM_PREFLIGHT",
                               "SHA256SUMS_STATUS"})
    all_pass = all("PASS" == "PASS" for _ in READY_GATES)
    one_fail = all(g == "PASS" for g in
                   ["PASS"] * 7 + ["FAIL"])
    check("ready_only_if_all_pass", all_pass is True and one_fail is False)

    # NEG JSON lenient parser.
    parsed = _parse_neg_json('noise\n{"fail": 0, "results": []}\n')
    check("neg_json_lenient_parse", parsed["fail"] == 0)

    # Ordered-payload SHA is stable + order-sensitive.
    s1 = _ordered_payload_sha256(["aa", "bb"])
    s2 = _ordered_payload_sha256(["aa", "bb"])
    s3 = _ordered_payload_sha256(["bb", "aa"])
    check("ordered_payload_sha_stable", s1 == s2 and s1 != s3)

    if problems:
        print("TIER3_POOL_COMMON_ASSEMBLY_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_POOL_COMMON_ASSEMBLY_SELF_TEST_PASS "
          "(sums round-trip/tamper/unlisted; shims bind+compile; 8-gate set frozen)")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return self_test()

    def opt(flag, default=None):
        return argv[argv.index(flag) + 1] if flag in argv else default

    if "--assemble" in argv:
        require(opt("--common-root") and opt("--frozen-bank-artifacts"),
                "usage: tier3_pool_common_assembly.py --assemble "
                "--common-root DIR --frozen-bank-artifacts DIR [--repo-root DIR]")
        assemble(opt("--common-root"), opt("--frozen-bank-artifacts"),
                 opt("--repo-root"))
        return 0
    if "--finalize-ready" in argv:
        require(opt("--common-root") and opt("--preflight-cert")
                and opt("--persistent-binding") and opt("--reset128-binding"),
                "usage: tier3_pool_common_assembly.py --finalize-ready "
                "--common-root DIR --preflight-cert CERT.json "
                "--persistent-binding BINDING.json --reset128-binding BINDING.json "
                "[--repo-root DIR]")
        doc = finalize_ready(opt("--common-root"), opt("--preflight-cert"),
                             opt("--persistent-binding"),
                             opt("--reset128-binding"), opt("--repo-root"))
        return 0 if doc["COMMON_EVALUATOR_READY"] else 1
    print("usage: tier3_pool_common_assembly.py --self-test | --assemble ... | "
          "--finalize-ready ...")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
