#!/usr/bin/env python3
"""CC4 Tier3 — V2 dynamic-topology projection binding smoke DRIVER.

Task: CC4_FIX_FRONT_DYNAMIC_TOPOLOGY_METRIC_AND_REBIND_FORMAL_POOL_V2 §六
(re-bind all 6+1 on the V2 common artifacts; old bindings are NOT inherited).

This driver is the V1 binding driver (tier3_projection_binding_smoke.py, frozen
at d0d05ff2) forked for the V2 common assembly. Differences, and ONLY these:
  * binds the V2 engine: tier3_evaluator_v2 + tier3_event_predicates_v2
    (BFS_GRAPH_SOURCE=CURRENT_ENVIRONMENT_STATE_TOPOLOGY — the FRONT dense
    metric now covers legal topology mutation / mining),
  * verifies the V2 common assembly (common_v2/): V2 assembly manifest + V2
    engine LF-SHA pins (the 24 V1 modules re-verified UNCHANGED + the two new
    V2 modules), V2 SHA256SUMS, V2 artifact byte SHAs, the protocol-version
    stamp in the V2 profile, and — preservation proof — the V1 common/ 57/57
    sums still frozen with FORMAL_RANKING_AUTHORIZED=false (V1 status =
    SUPERSEDED_PRE_RANKING; never overwritten, never impersonated),
  * writes V2DT evidence: episode_records.jsonl, projection_record.json,
    common_evaluator_binding_result_v2dt.json, SHA256SUMS_V2DT, READY_V2DT.json
    under <out> (default cc4/<ID>/projection_binding_v2dt/).
Everything else — GPU discipline, dicode resolution pin, frozen bank content
gates, owner capsule triple-SHA verification, owner-protocol params/checkpoint
recomputation, policy adapters, the smoke schedule, NEG23 analog, the honest
run_class=INTERFACE_SMOKE labels — is the identical V1 gate flow.

FailClosed discipline under V2: legal mining is now INSIDE the metric domain,
so the historical CONTROL abort class (position outside the INITIAL walkable
grid) cannot occur. The ONLY tier3_event_predicates.FailClosed verdicts that
remain are genuine corruption-class violations (coordinate out-of-bounds,
non-finite / undecodable coordinates, player position contradicting the
CURRENT map state, frozen-bank baseline unreachable); they are caught and
recorded as structured minimum blocking evidence exactly as before, never
relaxed, never faked. Every other exception still crashes the driver
fail-closed.

This driver performs NO formal performance evaluation, NO ranking, and makes
NO performance claim (run_class=INTERFACE_SMOKE,
performance_claim_authorized=false). This task must await independent
secondary audit before any formal ranking.

Usage (server, locked CC4 venv, GPU2 or GPU3, CWD = repo root):
  CUDA_VISIBLE_DEVICES=GPU-8df11537-ab79-722d-606f-411966196c4c \
  python tier3_projection_binding_smoke_v2.py \
      --candidate-id BASE_GTRXL_ORIGINAL_VTRACE_98304 \
      [--episodes 2] [--max-steps 32] \
      [--common-dir /home/oseasy/student_pool_v1/common_v2] \
      [--v1-common-dir /home/oseasy/student_pool_v1/common] \
      [--frozen-bank-artifacts /home/oseasy/student_pool_v1/common/frozen_bank_artifacts] \
      [--out /home/oseasy/student_pool_v1/cc4/<ID>/projection_binding_v2dt]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import tier3_projection_runtime as proj        # noqa: E402

BINDING_SCHEMA = "mechanism_UED.common_evaluator_binding/v2dt"
PROJECTION_RECORD_SCHEMA = "mechanism_UED.projection_binding_record/v2dt"
READY_V2DT_SCHEMA = "mechanism_UED.projection_ready/v2dt"
COMMON_EVALUATOR_PROTOCOL_VERSION = "V2_DYNAMIC_TOPOLOGY"
SUPERSEDED_V1_DRIVER_COMMIT = "d0d05ff26ffd1ea0bfd80e4c0364edfe6f5616d4"
SUPERSEDED_V1_STATUS = "SUPERSEDED_PRE_RANKING"

# The READY marker + the sums file never list themselves (sums-excluded by
# construction, mirroring V1).
COMMON_V2_SUMS_EXCLUDED = {"SHA256SUMS", "COMMON_EVALUATOR_V2_READY.json"}
SKIP_DIR_NAMES = {"__pycache__"}

# ---------------------------------------------------------------------------
# V2 assembly pins. The engine LF-SHA pins are the committed repo bytes
# (computable anywhere); the common_v2/ artifact pins come from the server
# assembly (common_v2_assembly_gen.py) and were back-filled here after the
# assembly ran. ANY mismatch fails closed.
# ---------------------------------------------------------------------------
FROZEN_V2_ASSEMBLY_MANIFEST_SHA256 = "V2_ASSEMBLY_PENDING_BACKFILL"
FROZEN_V2_SHA256SUMS_SHA256 = "V2_ASSEMBLY_PENDING_BACKFILL"
FROZEN_V2_COMMON_SUMS_ENTRY_COUNT = 0
FROZEN_V2_COMMON_EVALUATOR_SHA256 = "V2_ASSEMBLY_PENDING_BACKFILL"
FROZEN_V2_COMMON_RUNNER_SHA256 = "V2_ASSEMBLY_PENDING_BACKFILL"
FROZEN_V2_EVALUATION_PROFILE_SHA256 = "V2_ASSEMBLY_PENDING_BACKFILL"
FROZEN_V2_METRIC_SCHEMA_SHA256 = "V2_ASSEMBLY_PENDING_BACKFILL"
FROZEN_V2_ENVIRONMENT_LOCK_SHA256 = "V2_ASSEMBLY_PENDING_BACKFILL"
# candidate_runtime_abi.md is byte-copied from V1 (the runner ABI doc is not a
# target of the metric fix) — its SHA is the frozen V1 value.
FROZEN_V2_ABI_DOC_SHA256 = "61e52af6ff64a3071f8b64916c80906275dcb201d37feaa0382ed988d03d7f6a"
FROZEN_V2_ENGINE_LF_SHA256 = {
    "tier3_evaluator_v2.py": "7de07f1e8cec86ff11adb563a217b8695482ede92eaf873f9451744cb7196629",
    "tier3_event_predicates_v2.py": "93e7b3d43450db7a722de863e3b7adbfe34eb859d745d32ea2ecc68ba292e3c1",
}
V2_ENGINE_MODULE_COUNT = len(proj.FROZEN_ENGINE_LF_SHA256) + len(FROZEN_V2_ENGINE_LF_SHA256)


def utc_now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Stage 1 — V2 engine + V2 common identity + V1 preservation proof
# ---------------------------------------------------------------------------
def verify_engine_and_common_v2(common_dir, v1_common_dir, tools_dir):
    ev = {}

    def _require_pins_filled():
        pending = [k for k, v in (
            ("assembly_manifest", FROZEN_V2_ASSEMBLY_MANIFEST_SHA256),
            ("sha256sums", FROZEN_V2_SHA256SUMS_SHA256),
            ("common_evaluator", FROZEN_V2_COMMON_EVALUATOR_SHA256),
            ("common_runner", FROZEN_V2_COMMON_RUNNER_SHA256),
            ("evaluation_profile", FROZEN_V2_EVALUATION_PROFILE_SHA256),
            ("metric_schema", FROZEN_V2_METRIC_SCHEMA_SHA256),
            ("environment_lock", FROZEN_V2_ENVIRONMENT_LOCK_SHA256),
        ) if v == "V2_ASSEMBLY_PENDING_BACKFILL"]
        pending += [k for k, v in FROZEN_V2_ENGINE_LF_SHA256.items()
                    if v == "V2_ASSEMBLY_PENDING_BACKFILL"]
        proj.require(not pending,
                     "FAIL CLOSED (V2_PINS_NOT_BACKFILLED): run the common_v2 "
                     "assembly and back-fill the driver pins first: %s" % pending)
        proj.require(FROZEN_V2_COMMON_SUMS_ENTRY_COUNT > 0,
                     "FAIL CLOSED (V2_PINS_NOT_BACKFILLED): sums entry count")

    _require_pins_filled()

    # 1a. V2 assembly_manifest byte SHA + engine module LF-SHAs.
    manifest_path = os.path.join(common_dir, "assembly_manifest_v2.json")
    am_sha = proj.sha256_file(manifest_path)
    proj.require(am_sha == FROZEN_V2_ASSEMBLY_MANIFEST_SHA256,
                 "FAIL CLOSED (V2_ASSEMBLY_MANIFEST_SHA_MISMATCH): live %s != "
                 "frozen %s" % (am_sha, FROZEN_V2_ASSEMBLY_MANIFEST_SHA256))
    manifest = proj.read_json(manifest_path)
    proj.require(manifest.get("common_evaluator_protocol_version")
                 == COMMON_EVALUATOR_PROTOCOL_VERSION,
                 "FAIL CLOSED: V2 manifest protocol version %r"
                 % manifest.get("common_evaluator_protocol_version"))
    engine_shas = manifest.get("engine_module_sha256")
    proj.require(isinstance(engine_shas, dict)
                 and len(engine_shas) == V2_ENGINE_MODULE_COUNT,
                 "FAIL CLOSED: V2 manifest engine_module_sha256 has %s entries, "
                 "expected %d (24 frozen V1 + 2 new V2)"
                 % (len(engine_shas) if isinstance(engine_shas, dict) else None,
                    V2_ENGINE_MODULE_COUNT))
    engine_verified = {}
    for fname, want in sorted(engine_shas.items()):
        got = proj.lf_sha256_file(os.path.join(tools_dir, fname))
        proj.require(got == want,
                     "FAIL CLOSED (ENGINE_MODULE_LF_SHA_MISMATCH): %s live %s != "
                     "manifest %s" % (fname, got, want))
        engine_verified[fname] = got
    # The 24 V1 modules must be pinned at the FROZEN V1 LF-SHAs (unchanged).
    for fname, want in proj.FROZEN_ENGINE_LF_SHA256.items():
        proj.require(engine_verified.get(fname) == want,
                     "FAIL CLOSED (V1_ENGINE_MODULE_DRIFT): %s in V2 manifest "
                     "%r != frozen V1 %s" % (fname, engine_verified.get(fname), want))
    # The two NEW V2 modules must be pinned at the V2 LF-SHAs.
    for fname, want in FROZEN_V2_ENGINE_LF_SHA256.items():
        proj.require(engine_verified.get(fname) == want,
                     "FAIL CLOSED (V2_ENGINE_MODULE_DRIFT): %s in V2 manifest "
                     "%r != frozen V2 %s" % (fname, engine_verified.get(fname), want))
    abi_sha = manifest.get("abi_doc_sha256")
    proj.require(abi_sha == FROZEN_V2_ABI_DOC_SHA256,
                 "FAIL CLOSED (V2_ABI_DOC_SHA_MISMATCH): manifest %r != frozen %s"
                 % (abi_sha, FROZEN_V2_ABI_DOC_SHA256))
    ev["assembly_manifest_v2_sha256"] = am_sha
    ev["engine_modules_lf_sha_verified"] = len(engine_verified)
    ev["engine_modules"] = engine_verified
    ev["engine_modules_v1_unchanged_count"] = len(proj.FROZEN_ENGINE_LF_SHA256)
    ev["engine_modules_v2_new"] = sorted(FROZEN_V2_ENGINE_LF_SHA256.keys())

    # 1b. V2 SHA256SUMS byte SHA + live N/N + unlisted scan.
    sums_path = os.path.join(common_dir, "SHA256SUMS")
    sums_sha = proj.sha256_file(sums_path)
    proj.require(sums_sha == FROZEN_V2_SHA256SUMS_SHA256,
                 "FAIL CLOSED (V2_SHA256SUMS_SHA_MISMATCH): live %s != frozen %s"
                 % (sums_sha, FROZEN_V2_SHA256SUMS_SHA256))
    sums = proj.parse_sha256sums(sums_path)
    proj.require(len(sums) == FROZEN_V2_COMMON_SUMS_ENTRY_COUNT,
                 "FAIL CLOSED: V2 SHA256SUMS lists %d entries, frozen %d"
                 % (len(sums), FROZEN_V2_COMMON_SUMS_ENTRY_COUNT))
    mismatches, missing = [], []
    for rel, want in sorted(sums.items()):
        p = os.path.join(common_dir, rel)
        if not os.path.isfile(p):
            missing.append(rel)
            continue
        if proj.sha256_file(p) != want:
            mismatches.append(rel)
    proj.require(not missing, "FAIL CLOSED (V2_COMMON_SUMS_MISSING_FILES): %s" % missing)
    proj.require(not mismatches, "FAIL CLOSED (V2_COMMON_SUMS_MISMATCH): %s" % mismatches)
    listed_abs = {os.path.normpath(os.path.join(common_dir, r)) for r in sums}
    unlisted = []
    for dp, dns, fns in os.walk(common_dir):
        dns[:] = [d for d in dns if d not in SKIP_DIR_NAMES]
        for fn in fns:
            full = os.path.normpath(os.path.join(dp, fn))
            if full in listed_abs:
                continue
            if os.path.basename(full) in COMMON_V2_SUMS_EXCLUDED:
                continue
            unlisted.append(os.path.relpath(full, common_dir))
    proj.require(not unlisted, "FAIL CLOSED (V2_COMMON_UNLISTED_FILES): %s" % sorted(unlisted))
    ev["sha256sums_v2_sha256"] = sums_sha
    ev["common_v2_sha256sums_self_check"] = "PASS (%d/%d)" % (
        len(sums), FROZEN_V2_COMMON_SUMS_ENTRY_COUNT)
    ev["common_v2_unlisted_files"] = []

    # 1c. Individual V2 artifact byte SHAs.
    named = {
        "common_evaluator.py": FROZEN_V2_COMMON_EVALUATOR_SHA256,
        "common_runner.py": FROZEN_V2_COMMON_RUNNER_SHA256,
        "evaluation_profile.json": FROZEN_V2_EVALUATION_PROFILE_SHA256,
        "metric_schema.json": FROZEN_V2_METRIC_SCHEMA_SHA256,
        "environment_lock.json": FROZEN_V2_ENVIRONMENT_LOCK_SHA256,
        "candidate_runtime_abi.md": FROZEN_V2_ABI_DOC_SHA256,
    }
    for fn, want in named.items():
        got = proj.sha256_file(os.path.join(common_dir, fn))
        proj.require(got == want,
                     "FAIL CLOSED (V2_COMMON_ARTIFACT_SHA_MISMATCH): %s live %s "
                     "!= frozen %s" % (fn, got, want))
        ev[fn + "_sha256"] = got

    # 1d. V2 profile invariants (identical to V1 — the metric fix touches none
    # of them) + the protocol-version stamp.
    profile = proj.read_json(os.path.join(common_dir, "evaluation_profile.json"))
    inv = profile.get("common_evaluation_invariants", {})
    proj.require(inv.get("max_timesteps") == proj.FROZEN_MAX_TIMESTEPS,
                 "FAIL CLOSED: V2 profile max_timesteps %r" % inv.get("max_timesteps"))
    proj.require(inv.get("action_mode") == proj.FROZEN_ACTION_MODE,
                 "FAIL CLOSED: V2 profile action_mode %r" % inv.get("action_mode"))
    proj.require(tuple(inv.get("observation_shape") or ()) == proj.FROZEN_OBSERVATION_SHAPE,
                 "FAIL CLOSED: V2 profile observation_shape %r" % inv.get("observation_shape"))
    proj.require(inv.get("action_dim") == proj.FROZEN_ACTION_DIM,
                 "FAIL CLOSED: V2 profile action_dim %r" % inv.get("action_dim"))
    sc = profile.get("scenarios", {})
    front_n = sc.get("front_l2", {}).get("n")
    back_n = sc.get("back_l2", {}).get("n")
    full_n = (sc.get("full", {}).get("world_seed_set") or {}).get("count")
    proj.require(front_n == proj.FROZEN_FRONT_EPISODE_COUNT
                 and back_n == proj.FROZEN_BACK_EPISODE_COUNT
                 and full_n == proj.FROZEN_FULL_EPISODE_COUNT,
                 "FAIL CLOSED: V2 profile episode counts %r/%r/%r != frozen 8/8/64"
                 % (front_n, back_n, full_n))
    proj.require(sc.get("front_l2", {}).get("bank_content_sha256")
                 == proj.FROZEN_FRONT_BANK_CONTENT_SHA256,
                 "FAIL CLOSED: V2 profile front bank_content_sha256 drift")
    proj.require(sc.get("back_l2", {}).get("bank_content_sha256")
                 == proj.FROZEN_BACK_BANK_CONTENT_SHA256,
                 "FAIL CLOSED: V2 profile back bank_content_sha256 drift")
    proj.require(profile.get("common_evaluator_protocol_version")
                 == COMMON_EVALUATOR_PROTOCOL_VERSION,
                 "FAIL CLOSED: V2 profile protocol version %r"
                 % profile.get("common_evaluator_protocol_version"))
    ev["profile_episode_counts"] = {
        "front_episode_count": front_n, "back_episode_count": back_n,
        "full_episode_count": full_n,
        "source": "evaluation_profile.json (V2) scenarios.{front_l2.n,back_l2.n,"
                  "full.world_seed_set.count}"}
    ev["common_evaluator_protocol_version"] = COMMON_EVALUATOR_PROTOCOL_VERSION

    # 1e. V2 READY marker cross-check (sums-excluded; presence-tolerant,
    # strict where fields exist; pre-gate the marker carries
    # COMMON_EVALUATOR_V2_READY=false — the binding runs BEFORE the §七 gate
    # flips it, so its truth value is recorded, not required).
    ready_path = os.path.join(common_dir, "COMMON_EVALUATOR_V2_READY.json")
    if os.path.isfile(ready_path):
        ready = proj.read_json(ready_path)
        for key, want in (
                ("common_evaluator_sha256", FROZEN_V2_COMMON_EVALUATOR_SHA256),
                ("common_runner_sha256", FROZEN_V2_COMMON_RUNNER_SHA256),
                ("evaluation_profile_sha256", FROZEN_V2_EVALUATION_PROFILE_SHA256),
                ("metric_schema_sha256", FROZEN_V2_METRIC_SCHEMA_SHA256),
                ("environment_lock_sha256", FROZEN_V2_ENVIRONMENT_LOCK_SHA256),
                ("front_bank_content_sha256", proj.FROZEN_FRONT_BANK_CONTENT_SHA256),
                ("back_bank_content_sha256", proj.FROZEN_BACK_BANK_CONTENT_SHA256),
                ("sha256sums_sha256", FROZEN_V2_SHA256SUMS_SHA256)):
            if key in ready:
                proj.require(ready[key] == want,
                             "FAIL CLOSED: V2 READY marker %s %r != frozen %s"
                             % (key, ready[key], want))
        if ready.get("FORMAL_RANKING_STARTED") is not None:
            proj.require(ready["FORMAL_RANKING_STARTED"] is False,
                         "FAIL CLOSED: V2 READY FORMAL_RANKING_STARTED must be "
                         "false (this task may not start formal ranking)")
        ev["v2_ready_marker_present"] = True
        ev["common_evaluator_v2_ready_at_binding_time"] = bool(
            ready.get("COMMON_EVALUATOR_V2_READY"))
    else:
        ev["v2_ready_marker_present"] = False
        ev["common_evaluator_v2_ready_at_binding_time"] = False

    # 1f. V1 preservation proof: the V1 common/ assembly (d0d05ff2 era) must
    # still be byte-frozen with FORMAL_RANKING_AUTHORIZED=false — V1 is
    # SUPERSEDED_PRE_RANKING, never overwritten, never impersonated.
    v1_sums_path = os.path.join(v1_common_dir, "SHA256SUMS")
    v1_sums_sha = proj.sha256_file(v1_sums_path)
    proj.require(v1_sums_sha == proj.FROZEN_SHA256SUMS_SHA256,
                 "FAIL CLOSED (V1_COMMON_MODIFIED): v1 SHA256SUMS byte SHA %s != "
                 "frozen %s — the V1 assembly must stay untouched"
                 % (v1_sums_sha, proj.FROZEN_SHA256SUMS_SHA256))
    v1_sums = proj.parse_sha256sums(v1_sums_path)
    proj.require(len(v1_sums) == proj.FROZEN_COMMON_SUMS_ENTRY_COUNT,
                 "FAIL CLOSED (V1_COMMON_MODIFIED): v1 sums lists %d entries, "
                 "frozen %d" % (len(v1_sums), proj.FROZEN_COMMON_SUMS_ENTRY_COUNT))
    v1_mismatch = [rel for rel, want in sorted(v1_sums.items())
                   if proj.sha256_file(os.path.join(v1_common_dir, rel)) != want]
    proj.require(not v1_mismatch,
                 "FAIL CLOSED (V1_COMMON_MODIFIED): %s" % v1_mismatch)
    v1_ready_path = os.path.join(v1_common_dir, "COMMON_EVALUATOR_READY.json")
    v1_formal_authorized = None
    if os.path.isfile(v1_ready_path):
        v1_ready = proj.read_json(v1_ready_path)
        v1_formal_authorized = v1_ready.get("FORMAL_RANKING_AUTHORIZED")
        proj.require(v1_formal_authorized is False,
                     "FAIL CLOSED (V1_COMMON_MODIFIED): v1 READY "
                     "FORMAL_RANKING_AUTHORIZED must stay false")
    ev["v1_preservation"] = {
        "v1_common_root": v1_common_dir,
        "v1_sha256sums_sha256": v1_sums_sha,
        "v1_sha256sums_self_check": "PASS (%d/%d)" % (
            len(v1_sums), proj.FROZEN_COMMON_SUMS_ENTRY_COUNT),
        "v1_formal_ranking_authorized": v1_formal_authorized,
        "v1_driver_commit": SUPERSEDED_V1_DRIVER_COMMIT,
        "v1_status": SUPERSEDED_V1_STATUS,
    }
    return ev


# ---------------------------------------------------------------------------
# Stage 2 — GPU discipline (identical to V1)
# ---------------------------------------------------------------------------
def verify_gpu_allowed():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    proj.require(cvd.strip(),
                 "FAIL CLOSED (GPU_DISCIPLINE): CUDA_VISIBLE_DEVICES must be set "
                 "explicitly (GPU2 or GPU3 UUID)")
    try:
        proc = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                              text=True, timeout=30)
    except Exception as exc:
        raise proj.FailClosed("FAIL CLOSED: nvidia-smi -L failed: %r" % exc)
    proj.require(proc.returncode == 0,
                 "FAIL CLOSED: nvidia-smi -L rc=%d: %s"
                 % (proc.returncode, (proc.stderr or "").strip()[:200]))
    by_index = {}
    for line in (proc.stdout or "").splitlines():
        if "(UUID:" not in line:
            continue
        head, uuid_part = line.split("(UUID:", 1)
        uuid = uuid_part.strip().rstrip(")").strip()
        toks = head.strip().split()
        idx = toks[1].rstrip(":") if len(toks) > 1 else None
        if idx is not None and idx.isdigit():
            by_index[int(idx)] = uuid
    entries = [e.strip() for e in cvd.split(",") if e.strip()]
    visible = []
    for e in entries:
        if e.startswith("GPU-"):
            visible.append(e)
        elif e.isdigit():
            proj.require(int(e) in by_index,
                         "FAIL CLOSED: CUDA_VISIBLE_DEVICES index %s not found in "
                         "nvidia-smi -L" % e)
            visible.append(by_index[int(e)])
        else:
            raise proj.FailClosed(
                "FAIL CLOSED: unsupported CUDA_VISIBLE_DEVICES entry %r" % e)
    for u in visible:
        proj.require(u in proj.CC4_GPU_ALLOWED_UUIDS,
                     "FAIL CLOSED (GPU_DISCIPLINE): visible GPU %s is NOT in the "
                     "CC4 allowlist %s" % (u, list(proj.CC4_GPU_ALLOWED_UUIDS)))
        for pre in proj.CC4_GPU_BANNED_UUID_PREFIXES:
            proj.require(not u.startswith(pre),
                         "FAIL CLOSED (GPU_DISCIPLINE): visible GPU %s is BANNED "
                         "(prefix %s)" % (u, pre))
    return {"cuda_visible_devices": cvd, "visible_gpu_uuids": visible,
            "nvidia_smi_L": (proc.stdout or "").strip().splitlines(),
            "gpu_allowlist_cc4": list(proj.CC4_GPU_ALLOWED_UUIDS)}


# ---------------------------------------------------------------------------
# Stage 8 — evidence writers
# ---------------------------------------------------------------------------
def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, sort_keys=True, ensure_ascii=False, indent=2)
        fh.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--episodes", type=int, default=2,
                    help="smoke episodes per scenario [1, 8]; default 2")
    ap.add_argument("--max-steps", type=int, default=32,
                    help="smoke step cap [1, 4096]; default 32")
    ap.add_argument("--common-dir", default="/home/oseasy/student_pool_v1/common_v2")
    ap.add_argument("--v1-common-dir", default="/home/oseasy/student_pool_v1/common")
    ap.add_argument("--frozen-bank-artifacts",
                    default="/home/oseasy/student_pool_v1/common/frozen_bank_artifacts")
    ap.add_argument("--out", default=None,
                    help="output dir; default <pool>/cc4/<ID>/projection_binding_v2dt")
    args = ap.parse_args(argv)

    # --- Stage 0: anti-pollution + registry + freshness ---------------------
    hook = os.environ.get("RMT16_POSTJAX_BINDING_SELFTEST", "")
    proj.require(hook.strip() in ("", "0"),
                 "FAIL CLOSED (anti-pollution): RMT16_POSTJAX_BINDING_SELFTEST=%r "
                 "is set (false-success hook). Unset before any binding." % hook)
    spec = proj.get_spec(args.candidate_id)
    episodes = int(args.episodes)
    max_steps = int(args.max_steps)
    proj.require(1 <= episodes <= proj.FROZEN_BACK_EPISODE_COUNT,
                 "FAIL CLOSED: --episodes %d outside [1, %d]"
                 % (episodes, proj.FROZEN_BACK_EPISODE_COUNT))
    proj.require(1 <= max_steps <= proj.FROZEN_MAX_TIMESTEPS,
                 "FAIL CLOSED: --max-steps %d outside [1, %d]"
                 % (max_steps, proj.FROZEN_MAX_TIMESTEPS))
    out_dir = args.out
    if out_dir is None:
        pool_root = os.path.dirname(os.path.normpath(args.common_dir))
        out_dir = os.path.join(pool_root, "cc4", args.candidate_id,
                               "projection_binding_v2dt")
    tools_dir = HERE
    repo_root = os.path.dirname(os.path.dirname(tools_dir))

    # Launch contract (identical to V1): the frozen engine resolves its
    # audited raw-data extract paths CWD-relative; launch from the repo root.
    cwd_real = os.path.realpath(os.getcwd())
    root_real = os.path.realpath(repo_root)
    proj.require(cwd_real == root_real,
                 "FAIL CLOSED (launch contract): cwd %s != repo root %s."
                 % (cwd_real, root_real))

    print("[stage0] candidate=%s family=%s class=%s out=%s protocol=%s"
          % (args.candidate_id, spec["runtime_family"], spec["candidate_class"],
             out_dir, COMMON_EVALUATOR_PROTOCOL_VERSION), flush=True)

    # --- engine import (V2) ---------------------------------------------------
    import tier3_evaluator_v2 as ev
    import tier3_event_predicates_v2 as predm
    import tier3_state_bank_materializer as mat
    import tier3_state_serializer as ser
    proj.require(predm.COMMON_EVALUATOR_PROTOCOL_VERSION
                 == COMMON_EVALUATOR_PROTOCOL_VERSION,
                 "FAIL CLOSED: predicate module protocol version %r"
                 % predm.COMMON_EVALUATOR_PROTOCOL_VERSION)
    proj.require(ser.have_jax_craftax(),
                 "FAIL CLOSED (BLOCKED_ENVIRONMENT): JAX+craftax required "
                 "(jax=%s, craftax=%s)" % (ser.have_jax(), ser.have_craftax()))
    ev.assert_output_dir_fresh(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # --- Stage 1: V2 engine + V2 common identity + V1 preservation -----------
    print("[stage1] verifying V2 engine + common_v2/ identity + V1 preservation "
          "...", flush=True)
    common_ev = verify_engine_and_common_v2(args.common_dir, args.v1_common_dir,
                                            tools_dir)
    print("[stage1] %s; engine modules LF-SHA %d/%d (24 V1 unchanged + 2 new V2); "
          "profile counts %s; v1 %s"
          % (common_ev["common_v2_sha256sums_self_check"],
             common_ev["engine_modules_lf_sha_verified"],
             common_ev["engine_modules_lf_sha_verified"],
             common_ev["profile_episode_counts"],
             common_ev["v1_preservation"]["v1_status"]), flush=True)

    # --- Stage 2: GPU discipline --------------------------------------------
    gpu_ev = verify_gpu_allowed()
    print("[stage2] visible GPUs %s (allowlist enforced)"
          % gpu_ev["visible_gpu_uuids"], flush=True)

    # --- Stage 3: dicode resolution pin + canonical env ----------------------
    dicode_ev = proj.pin_dicode_resolution(repo_root)
    print("[stage3] dicode resolution pinned to %s (network.py %s == CC1 "
          "policy_source)" % (dicode_ev["dicode_src"],
                              dicode_ev["dicode_network_sha256"]), flush=True)
    import jax
    import jax.numpy as jnp
    print("[stage3] building canonical env ...", flush=True)
    entry = ev.make_canonical_env()
    proj.require(tuple(entry["observation_shape"]) == proj.FROZEN_OBSERVATION_SHAPE,
                 "FAIL CLOSED: observation shape %s != frozen %s"
                 % (entry["observation_shape"], proj.FROZEN_OBSERVATION_SHAPE))
    proj.require(int(entry["action_count"]) == proj.FROZEN_ACTION_DIM,
                 "FAIL CLOSED: action count %d != frozen %d"
                 % (entry["action_count"], proj.FROZEN_ACTION_DIM))

    # --- Stage 4: frozen bank artifacts (bytes unchanged; same content SHAs) -
    import tier3_frozen_bank_artifacts as art
    print("[stage4] loading frozen bank artifacts (read-only) ...", flush=True)
    bindings = {}
    bank_ev = {}
    for sc in (ev.FRONT, ev.BACK):
        b = art.load_bank(sc, args.frozen_bank_artifacts)
        want = (proj.FROZEN_FRONT_BANK_CONTENT_SHA256 if sc == ev.FRONT
                else proj.FROZEN_BACK_BANK_CONTENT_SHA256)
        proj.require(b.get("state_bank_hash") == want,
                     "FAIL CLOSED (FROZEN_BANK_CONTENT_MISMATCH): %s state_bank_hash "
                     "%r != frozen %s" % (sc, b.get("state_bank_hash"), want))
        bindings[sc] = b
        bank_ev[sc] = {"state_bank_hash": b.get("state_bank_hash"),
                       "bank_source": b.get("bank_source"),
                       "artifact_file_sha256": b.get("artifact_file_sha256"),
                       "loaded_content_sha256": b.get("loaded_content_sha256"),
                       "device_provenance": b.get("device_provenance"),
                       "n_states": len(b.get("states", []))}
        proj.require(bank_ev[sc]["n_states"] >= episodes,
                     "FAIL CLOSED: bank %s has %d states < episodes %d"
                     % (sc, bank_ev[sc]["n_states"], episodes))

    # --- Stage 5: capsule verification + owner runtime load ------------------
    print("[stage5] verifying capsule files + loading owner runtime via "
          "projection registry ...", flush=True)
    capsule_ev = proj.verify_capsule_files(spec)
    ctx = proj.load_owner_runtime(spec)

    params_before = proj.recompute_params_sha_owner(ctx)
    declared_params = spec["declared_params_sha256"]["value"]
    proj.require(params_before == declared_params,
                 "FAIL CLOSED (PARAMS_SHA_CC4_RECOMPUTE_MISMATCH): owner-protocol "
                 "recompute %s != owner-declared %s [%s]"
                 % (params_before, declared_params,
                    spec["declared_params_sha256"]["declaration_source"]))
    file_sha = proj.recompute_checkpoint_file_sha_owner(spec, ctx)
    declared_file = spec["declared_checkpoint_file_sha256"]["value"]
    proj.require(file_sha == declared_file,
                 "FAIL CLOSED (CHECKPOINT_FILE_SHA_CC4_RECOMPUTE_MISMATCH): "
                 "owner-protocol recompute %s != owner-declared %s [%s]"
                 % (file_sha, declared_file,
                    spec["declared_checkpoint_file_sha256"]["declaration_source"]))
    print("[stage5] params_sha256(owner protocol)==declared MATCH; "
          "checkpoint_file_sha256(owner protocol)==declared MATCH", flush=True)

    # --- Stage 6: policy adapter + (slowgru) boundary unit check --------------
    print("[stage6] building policy adapter ...", flush=True)
    policy = proj.build_policy(spec, ctx)
    boundary_ev = None
    if spec["loader_kind"] == "cc3_slowgru":
        boundary_ev = proj.slowgru_boundary_unit_check(ctx["module"],
                                                       spec["carry_mode"])
        print("[stage6] slowgru boundary unit check: carry_mode=%s info=%s"
              % (boundary_ev["carry_mode"], boundary_ev["boundary_info"]),
              flush=True)
    batch1_ev = getattr(policy, "batch1_workaround", None)
    if batch1_ev:
        print("[stage6] batch-1 workaround ACTIVE (disclosed): effective_batch=%d "
              "readout_row=%d owner_code_modified=%s"
              % (batch1_ev["effective_batch"], batch1_ev["readout_row"],
                 batch1_ev["owner_code_modified"]), flush=True)

    # --- Stage 7: smoke rollouts (V2 engine library path) ---------------------
    print("[stage7] interface smoke (V2 dynamic topology): %d episodes/scenario, "
          "max_steps=%d ..." % (episodes, max_steps), flush=True)
    scenarios = [ev.FULL, ev.FRONT, ev.BACK]
    schedule = {}
    for sc in scenarios:
        seeds = ([ev.FULL_SMOKE_SEED_BASE + i for i in range(episodes)]
                 if sc == ev.FULL
                 else mat.fixed_seed_schedule(sc, mat.FROZEN_BANK_N,
                                              mat.FROZEN_SEED_BASE,
                                              mat.FROZEN_SEED_STRIDE)[:episodes])
        schedule[sc] = {
            "kind": ("canonical_reset_seeds_smoke" if sc == ev.FULL
                     else "frozen_bank_state_smoke"),
            "count": len(seeds), "seeds": [int(s) for s in seeds]}
    reset_fn = ev._jit_reset(entry)
    records_by_scenario, results_by_scenario = {}, {}
    smoke_abort = None
    for sc in scenarios:
        seeds = [int(s) for s in schedule[sc]["seeds"]]
        entry_ids = ev.state_entry_ids_for(sc, seeds)
        eps = []
        for i, seed in enumerate(seeds):
            policy.reset()
            if sc == ev.FULL:
                _obs0, start_state = reset_fn(jax.random.PRNGKey(int(seed)))
            else:
                start_state = jax.tree.map(jnp.asarray,
                                           bindings[sc]["states"][i])
            try:
                rec = ev.rollout_episode(entry, start_state, sc, policy,
                                         entry_ids[i], int(seed),
                                         max_steps=max_steps)
            except predm.FailClosed as exc:
                # Under V2_DYNAMIC_TOPOLOGY legal mining is INSIDE the metric
                # domain (the graph is rebuilt from the current env state every
                # step), so the historical CONTROL abort class cannot occur.
                # The ONLY remaining predicate FailClosed verdicts are genuine
                # corruption-class violations (coordinate out-of-bounds,
                # non-finite / undecodable coordinates, player position
                # contradicting the CURRENT map state, frozen-bank baseline
                # unreachable). Recorded as structured minimum blocking
                # evidence — never relaxed, never faked; every other exception
                # still crashes the driver fail-closed. SCOPE: ONLY the
                # predicate FailClosed (re-exported unchanged from the frozen
                # V1 module) is caught.
                smoke_abort = {
                    "exception_type": "tier3_event_predicates.FailClosed (via "
                                      "tier3_event_predicates_v2 re-export)",
                    "engine_message": str(exc),
                    "scenario": sc,
                    "episode_index": i,
                    "episodes_planned": len(seeds),
                    "episodes_completed_before_abort": i,
                    "entry_id": entry_ids[i],
                    "seed": int(seed),
                    "scenarios_completed_before_abort":
                        list(records_by_scenario.keys()),
                    "verdict": "ENGINE_PREDICATE_REJECTED_ROLLOUT_V2",
                    "v2_failclosed_class": "CORRUPTION_CLASS_ONLY (out-of-bounds / "
                        "non-finite / player-vs-current-map contradiction / "
                        "baseline-unreachable bank corruption) — legal topology "
                        "mutation is in-domain under V2_DYNAMIC_TOPOLOGY",
                    "authority": "frozen engine predicate re-exported by the V2 "
                                 "module (predicate_code_sha256-bound V1 bytes; "
                                 "not relaxable by CC4)",
                    "formal_evaluation_consequence": "a formal run executes the "
                                 "same V2 rollout_episode code path and reaches "
                                 "the same verdict for this candidate",
                }
                print("  [%s %d/%d %s seed=%d] ENGINE PREDICATE ABORT (recorded, "
                      "NOT swallowed): %s" % (sc, i + 1, len(seeds), entry_ids[i],
                                              seed, exc), flush=True)
                break
            rec["episode_record_sha256"] = proj.sha256_bytes(
                proj.canonical_json_bytes(rec))
            eps.append(rec)
            print("  [%s %d/%d %s seed=%d] steps=%d defeat=%s died=%s "
                  "transition=%s engaged=%s progress=%s"
                  % (sc, i + 1, len(seeds), entry_ids[i], seed, rec["timesteps"],
                     rec["defeat_kobold"], rec["player_died"],
                     rec["front_floor_transition_reached"], rec["kobold_engaged"],
                     rec["graph_distance_progress"]), flush=True)
        if eps:                            # keep honest partial evidence
            lines = [proj.canonical_json_bytes(r).decode("utf-8") for r in eps]
            records_by_scenario[sc] = {
                "seeds": seeds[:len(eps)],
                "entry_ids": entry_ids[:len(eps)],
                "episode_records": eps,
                "partial": bool(smoke_abort and smoke_abort["scenario"] == sc),
                "episode_records_sha256": proj.sha256_bytes(
                    ("\n".join(lines) + "\n").encode("utf-8"))}
            if not (smoke_abort and smoke_abort["scenario"] == sc):
                results_by_scenario[sc] = ev.evaluate(sc, eps)
        if smoke_abort:
            break

    # --- NEG23 analog (owner-protocol params unchanged) -----------------------
    params_after = proj.recompute_params_sha_owner(ctx)
    proj.require(params_after == params_before,
                 "FAIL CLOSED (PARAMS_CHANGED_BY_EVALUATION): before %s != after %s"
                 % (params_before, params_after))
    params_unchanged = True

    # --- provenance -----------------------------------------------------------
    try:
        device_identity = ev._eval_device_identity()
    except Exception as exc:                       # never mask the smoke result
        device_identity = {"error": repr(exc)}
    try:
        git_head = ev._git_commit_head()
    except Exception as exc:
        git_head = "unavailable: %r" % exc
    xla_flags = {k: os.environ.get(k) for k in sorted(os.environ)
                 if k.startswith(("XLA_", "JAX_"))}
    provenance = {
        "pid": os.getpid(),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "host": socket.gethostname(),
        "python_executable": sys.executable,
        "generated_at_utc": utc_now_iso(),
        "git_commit_head": git_head,
        "device_identity": device_identity,
        "gpu": gpu_ev,
        "xla_jax_env": xla_flags,
        "projection_module_sha256": proj.sha256_file(
            os.path.join(tools_dir, "tier3_projection_runtime.py")),
        "driver_module_sha256": proj.sha256_file(os.path.abspath(__file__)),
        "projection_module_lf_sha256": proj.lf_sha256_file(
            os.path.join(tools_dir, "tier3_projection_runtime.py")),
        "driver_module_lf_sha256": proj.lf_sha256_file(os.path.abspath(__file__)),
        "evaluator_v2_module_sha256": proj.sha256_file(
            os.path.join(tools_dir, "tier3_evaluator_v2.py")),
        "predicates_v2_module_sha256": proj.sha256_file(
            os.path.join(tools_dir, "tier3_event_predicates_v2.py")),
    }

    # --- Stage 8: evidence writes ---------------------------------------------
    jsonl_lines = []
    for sc in scenarios:
        if sc not in records_by_scenario:
            continue
        for r in records_by_scenario[sc]["episode_records"]:
            jsonl_lines.append(proj.canonical_json_bytes(r).decode("utf-8"))
    jsonl_path = os.path.join(out_dir, "episode_records.jsonl")
    with open(jsonl_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(("\n".join(jsonl_lines) + "\n") if jsonl_lines else "")
    episode_records_jsonl_sha256 = proj.sha256_file(jsonl_path)

    policy_state = None
    if isinstance(policy, proj.SlowGRUProjectionPolicy):
        policy_state = {"boundary_invocations": policy.boundary_invocations,
                        "boundary_info_log": policy.boundary_info_log,
                        "segment_boundary_steps": policy.segment_boundary_steps}
    projection_record = {
        "schema": PROJECTION_RECORD_SCHEMA,
        "candidate_id": args.candidate_id,
        "runtime_family": spec["runtime_family"],
        "owner": spec["owner"],
        "candidate_class": spec["candidate_class"],
        "common_evaluator_protocol_version": COMMON_EVALUATOR_PROTOCOL_VERSION,
        "bfs_graph_source": predm.BFS_GRAPH_SOURCE,
        "rebind_generation": "V2DT",
        "run_class": "INTERFACE_SMOKE",
        "registration_authority": {
            "NON_RMT_RUNTIME_REGISTRATION_AUTHORITY": "CC4_CAN_REGISTER_PROJECTIONS",
            "qualification": "CONDITIONAL",
            "conditions": ["C1_FROZEN_COMMON_UNTOUCHED", "C2_ZERO_REIMPLEMENTATION",
                           "C3_PROJECTION_ADDENDUM_DOCUMENTED", "C4_HONEST_LABELS",
                           "C5_OWNER_ARTIFACTS_READONLY"],
            "audit_record": "reports/tier3_scaffolded_evaluation/"
                            "non_rmt_abi_binding_closure_20260731/"
                            "NON_RMT_RUNTIME_REGISTRATION_AUTHORITY_AUDIT.md",
            "owner_action_required": False,
        },
        "capsule_verification": capsule_ev,
        "source_capsule_root": spec["source_capsule_root"],
        "owner_runtime_load": {
            "kind": ctx["kind"],
            "checkpoint_path": ctx["checkpoint_path"],
            "wandb_stub": ctx.get("wandb_stub"),
            "import_stubs": ctx.get("import_stubs"),
            "numpy_pickle_compat": ctx.get("numpy_pickle_compat"),
            "frozen_modules_live_sha256": ctx.get("frozen_modules_live_sha256"),
            "shared_runtime": ctx.get("shared_runtime"),
            "slowgru_runtime_path": ctx.get("slowgru_runtime_path"),
            "slowgru_runtime_sha256": ctx.get("slowgru_runtime_sha256"),
            "slowgru_network_sha256": ctx.get("slowgru_network_sha256"),
            "arm_src": ctx.get("arm_src"),
            "obs_dim": ctx.get("obs_dim"),
            "action_dim": ctx.get("action_dim"),
            "carry_mode": (ctx["handle"].get("carry_mode")
                           if ctx["kind"] == "cc3_slowgru" else None),
        },
        "dicode_resolution": dicode_ev,
        "sha_recomputation": {
            "params_sha256_owner_protocol": params_before,
            "params_sha256_owner_declared": declared_params,
            "params_sha256_declaration_source":
                spec["declared_params_sha256"]["declaration_source"],
            "params_hash_protocol": spec["params_hash_protocol"],
            "checkpoint_file_sha256_owner_protocol": file_sha,
            "checkpoint_file_sha256_owner_declared": declared_file,
            "checkpoint_file_sha256_declaration_source":
                spec["declared_checkpoint_file_sha256"]["declaration_source"],
            "checkpoint_file_hash_protocol": spec["checkpoint_file_hash_protocol"],
            "params_unchanged_after_rollouts": params_unchanged,
            "params_sha256_after": params_after,
        },
        "canonical_env": {"observation_shape": list(entry["observation_shape"]),
                          "action_count": int(entry["action_count"])},
        "boundary_unit_check": boundary_ev,
        "batch1_workaround": batch1_ev,
        "policy_adapter": {
            "class": type(policy).__name__,
            "greedy_readout": (
                "owner argmax(logits) inside Candidate.policy_step"
                if ctx["kind"] == "cc2_base_gtrxl"
                else "owner pi.mode() (greedy=True)"
                if ctx["kind"] == "cc1_gtrxl128"
                else "argmax(extras['logits']) — memory update is action-independent; "
                     "faithful greedy readout of the owner's identical forward"),
            "state": policy_state,
        },
        "smoke_schedule": schedule,
        "smoke_episodes": episodes,
        "smoke_max_steps": max_steps,
        "frozen_banks": bank_ev,
        "records_by_scenario": {
            sc: {"seeds": records_by_scenario[sc]["seeds"],
                 "entry_ids": records_by_scenario[sc]["entry_ids"],
                 "episode_records_sha256":
                     records_by_scenario[sc]["episode_records_sha256"],
                 "partial": records_by_scenario[sc].get("partial", False),
                 "episode_records": records_by_scenario[sc]["episode_records"]}
            for sc in records_by_scenario},
        "smoke_abort": smoke_abort,
        "results_by_scenario_smoke_only": results_by_scenario,
        "episode_records_jsonl_sha256": episode_records_jsonl_sha256,
        "common_verification": common_ev,
        "provenance": provenance,
        "performance_claim_authorized": False,
    }
    record_path = os.path.join(out_dir, "projection_record.json")
    write_json(record_path, projection_record)

    # 8c. common_evaluator_binding_result_v2dt.json
    is_teacher = spec["candidate_class"] == "TEACHER_REFERENCE"
    binding = {
        "schema": BINDING_SCHEMA,
        "supersedes": "cc4/%s/projection_binding_v2/ (d0d05ff2-era V1 metric "
                      "binding; retained as history, NOT modified, NOT "
                      "inherited — the common evaluator SHA changed)"
                      % args.candidate_id,
        "generated_at_utc": utc_now_iso(),
        "candidate_id": args.candidate_id,
        "owner": spec["owner"],
        "candidate_class": spec["candidate_class"],
        "runtime_family": spec["runtime_family"],
        "runtime_registered": True,
        "common_evaluator_protocol_version": COMMON_EVALUATOR_PROTOCOL_VERSION,
        "bfs_graph_source": predm.BFS_GRAPH_SOURCE,
        "rebind_generation": "V2DT",
        "source_capsule_root": spec["source_capsule_root"],
        "cc4_projection_path": out_dir,
        # frozen contract (formal scale — NOT the smoke scale; unchanged by V2)
        "action_mode": proj.FROZEN_ACTION_MODE,
        "action_mode_source": "frozen_evaluation_profile (V2 stamp; invariants "
                              "byte-identical to V1)",
        "max_steps": proj.FROZEN_MAX_TIMESTEPS,
        "observation_shape": list(proj.FROZEN_OBSERVATION_SHAPE),
        "action_dim": proj.FROZEN_ACTION_DIM,
        "front_episode_count": proj.FROZEN_FRONT_EPISODE_COUNT,
        "back_episode_count": proj.FROZEN_BACK_EPISODE_COUNT,
        "full_episode_count": proj.FROZEN_FULL_EPISODE_COUNT,
        "episode_count_source": "evaluation_profile.json (V2; live byte-SHA "
            "verified) scenarios.{front_l2.n,back_l2.n,full.world_seed_set.count}",
        "episode_count_status": "PASS (frozen 8/8/64 from V2 profile; smoke "
            "executed %d/scenario at max_steps=%d)" % (episodes, max_steps),
        "run_class": "INTERFACE_SMOKE",
        "smoke_episodes_per_scenario": episodes,
        "smoke_max_steps": max_steps,
        "smoke_schedule": schedule,
        "performance_claim_authorized": False,
        "strong_student_selection_authorized": False,
        "evaluation_certificate_status": "PENDING_FORMAL_EVALUATION",
        "evaluation_certificate_file": None,
        # V2 common SHAs (live-reverified this run) — the SHA set every V2
        # certificate must reference
        "common_root": args.common_dir,
        "common_evaluator_sha256": FROZEN_V2_COMMON_EVALUATOR_SHA256,
        "common_runner_sha256": FROZEN_V2_COMMON_RUNNER_SHA256,
        "evaluation_profile_sha256": FROZEN_V2_EVALUATION_PROFILE_SHA256,
        "metric_schema_sha256": FROZEN_V2_METRIC_SCHEMA_SHA256,
        "environment_lock_sha256": FROZEN_V2_ENVIRONMENT_LOCK_SHA256,
        "candidate_runtime_abi_sha256": FROZEN_V2_ABI_DOC_SHA256,
        "assembly_manifest_v2_sha256": FROZEN_V2_ASSEMBLY_MANIFEST_SHA256,
        "sha256sums_v2_sha256": FROZEN_V2_SHA256SUMS_SHA256,
        "full_profile_sha256": proj.FROZEN_FULL_PROFILE_SHA256,
        "full_profile_status": "FROZEN",
        "front_bank_content_sha256": proj.FROZEN_FRONT_BANK_CONTENT_SHA256,
        "back_bank_content_sha256": proj.FROZEN_BACK_BANK_CONTENT_SHA256,
        "common_v2_sha256sums_self_check": common_ev["common_v2_sha256sums_self_check"],
        "common_sha_match_status": "PASS (live %d/%d + per-artifact byte SHAs + "
            "profile invariants + bank content hashes + protocol version)"
            % (FROZEN_V2_COMMON_SUMS_ENTRY_COUNT, FROZEN_V2_COMMON_SUMS_ENTRY_COUNT),
        "engine_module_lf_sha_verified": common_ev["engine_modules_lf_sha_verified"],
        "engine_modules_v2_new": common_ev["engine_modules_v2_new"],
        "evaluator_v2_module_sha256": provenance["evaluator_v2_module_sha256"],
        "predicates_v2_module_sha256": provenance["predicates_v2_module_sha256"],
        "v1_supersession": {
            "COMMON_EVALUATOR_V1_DRIVER": SUPERSEDED_V1_DRIVER_COMMIT,
            "COMMON_EVALUATOR_V1_STATUS": SUPERSEDED_V1_STATUS,
            "v1_common_evaluator_sha256": proj.FROZEN_COMMON_EVALUATOR_SHA256,
            "v1_common_runner_sha256": proj.FROZEN_COMMON_RUNNER_SHA256,
            "v1_evaluation_profile_sha256": proj.FROZEN_EVALUATION_PROFILE_SHA256,
            "v1_metric_schema_sha256": proj.FROZEN_METRIC_SCHEMA_SHA256,
            "v1_environment_lock_sha256": proj.FROZEN_ENVIRONMENT_LOCK_SHA256,
            "v1_sha256sums_sha256": proj.FROZEN_SHA256SUMS_SHA256,
            "v1_assembly_manifest_sha256": proj.FROZEN_ASSEMBLY_MANIFEST_SHA256,
            "v1_formal_ranking_ever_authorized": False,
            "v1_preservation_reverified": common_ev["v1_preservation"],
            "v1_binding_evidence_retained": "cc4/%s/projection_binding_v2/ "
                "(unmodified)" % args.candidate_id,
        },
        # owner capsule artifact SHAs (live == registry == capsule sums)
        "candidate_runtime_sha256":
            spec["capsule_file_sha256"]["candidate_runtime.py"],
        "evaluate_candidate_sha256":
            spec["capsule_file_sha256"]["evaluate_candidate.py"],
        "candidate_manifest_sha256":
            spec["capsule_file_sha256"]["candidate_manifest.json"],
        "checkpoint_contract_sha256":
            spec["capsule_file_sha256"]["checkpoint_contract.json"],
        "capsule_file_verification": capsule_ev,
        "source_capsule_missing_files": [],
        # params / checkpoint (owner protocol, recomputed this run)
        "params_sha256": params_before,
        "params_sha256_cc4_recomputation": "CC4_RECOMPUTED_MATCH_VIA_OWNER_PROTOCOL",
        "params_hash_protocol": spec["params_hash_protocol"],
        "params_sha256_declaration_source":
            spec["declared_params_sha256"]["declaration_source"],
        "params_unchanged": params_unchanged,
        "checkpoint_file_sha256": file_sha,
        "checkpoint_file_sha256_cc4_recomputed": file_sha,
        "checkpoint_file_sha256_status": "CC4_RECOMPUTED_MATCH_VIA_OWNER_PROTOCOL",
        "checkpoint_file_sha256_verified_by_cc4": True,
        "checkpoint_file_hash_protocol": spec["checkpoint_file_hash_protocol"],
        # projection identity
        "projection_module_sha256": provenance["projection_module_sha256"],
        "driver_module_sha256": provenance["driver_module_sha256"],
        "bound_owner_runtime_sha256": spec["bound_owner_runtime_sha256"],
        "interface_smoke_status": ("PASS" if smoke_abort is None
                                   else "FAIL_CLOSED_ENGINE_PREDICATE"),
        "binding_status": "PASS" if smoke_abort is None else "BLOCKED",
        "formal_eval_binding": ("INTERFACE_SMOKE_PASS_FORMAL_PENDING"
                                if smoke_abort is None
                                else "INTERFACE_SMOKE_BLOCKED_BY_FROZEN_ENGINE_"
                                     "PREDICATE"),
        "remaining_blocker": (None if smoke_abort is None else {
            "kind": "ENGINE_PREDICATE_REJECTED_ROLLOUT_V2",
            "engine_message": smoke_abort["engine_message"],
            "scenario": smoke_abort["scenario"],
            "episode_index": smoke_abort["episode_index"],
            "v2_failclosed_class": smoke_abort["v2_failclosed_class"],
            "why_not_relaxable": "the predicate is predicate_code_sha256-bound "
                "(frozen V1 bytes re-exported by the V2 module); CC4 may not "
                "relax, skip, or reimplement it; a formal evaluation executes "
                "the same V2 code path and reaches the same verdict",
            "minimum_owner_prompt": "[%s → owner %s] V2 动态拓扑引擎在 smoke "
                "rollout 以腐败类裁定拒绝该候选(engine_message=%r)。V2 已覆盖"
                "合法挖掘(该历史阻断类不再存在),剩余 FailClosed 仅为真实状态"
                "腐败类(越界/非有限/玩家-当前地图矛盾/银行基线不可达)。请 "
                "owner 核验该候选的状态完整性。"
                % (args.candidate_id, spec["owner"],
                   smoke_abort["engine_message"]),
        }),
        "episode_records_jsonl_sha256": episode_records_jsonl_sha256,
        "records_sha256_by_scenario": {
            sc: records_by_scenario[sc]["episode_records_sha256"]
            for sc in records_by_scenario},
        "results_by_scenario_smoke_only": results_by_scenario,
        "wandb_stub": ctx.get("wandb_stub"),
        "import_stubs": ctx.get("import_stubs"),
        "dicode_resolution": dicode_ev,
        "gpu": gpu_ev,
        "device_identity": provenance["device_identity"],
        "git_commit_head": provenance["git_commit_head"],
        "provenance_pid": provenance["pid"],
        # eligibility flags
        "formal_student_ranking_eligible": spec["formal_student_ranking_eligible"],
        "strong_student_selection_eligible":
            spec["strong_student_selection_eligible"],
        "reference_only": spec["reference_only"],
        "student_rank": spec["student_rank"],
        "budget_class": spec["budget_class"],
        "training_steps": spec["training_steps"],
        "training_seed": spec["training_seed"],
        "teacher_included_in_student_ranking": False,
        "counts_toward_student_binding_count": (False if is_teacher else True),
        "formal_ranking_started": False,
        "await_independent_secondary_audit": True,
    }
    if spec["loader_kind"] == "cc3_slowgru":
        binding["carry_mode"] = spec["carry_mode"]
        binding["segment_boundary_steps"] = spec["segment_boundary_steps"]
        binding["boundary_semantics"] = spec["boundary_semantics"]
        binding["boundary_unit_check"] = boundary_ev
        binding["numpy_pickle_compat"] = ctx.get("numpy_pickle_compat")
    if batch1_ev:
        binding["batch1_workaround"] = batch1_ev
    if smoke_abort:
        binding["smoke_abort"] = smoke_abort
        binding["episode_count_status"] += (
            " — SMOKE PARTIAL/ABORTED: V2 engine predicate (corruption class) "
            "rejected the rollout at %s episode %d/%d (see smoke_abort; "
            "remaining scenarios not executed)"
            % (smoke_abort["scenario"], smoke_abort["episode_index"] + 1,
               smoke_abort["episodes_planned"]))
    binding_path = os.path.join(out_dir, "common_evaluator_binding_result_v2dt.json")
    write_json(binding_path, binding)

    # 8d. SHA256SUMS_V2DT over the three evidence files
    sums_entries = []
    for fn in ("episode_records.jsonl", "projection_record.json",
               "common_evaluator_binding_result_v2dt.json"):
        sums_entries.append("%s  %s" % (proj.sha256_file(os.path.join(out_dir, fn)), fn))
    sums_v2dt_path = os.path.join(out_dir, "SHA256SUMS_V2DT")
    with open(sums_v2dt_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(sums_entries) + "\n")

    # 8e. READY_V2DT.json (honest gates; sums-excluded like V1 READY)
    gates = {
        "G1_CAPSULE_FILE_SHA_MATCH": all(
            capsule_ev[fn]["match"] for fn in proj.CAPSULE_FILES),
        "G2_CHECKPOINT_FILE_SHA_OWNER_RECOMPUTE_MATCH": file_sha == declared_file,
        "G3_PARAMS_SHA_OWNER_RECOMPUTE_MATCH": params_before == declared_params,
        "G4_INTERFACE_SMOKE_EXECUTED": (smoke_abort is None) and all(
            sc in records_by_scenario
            and len(records_by_scenario[sc]["episode_records"]) == episodes
            for sc in scenarios),
        "G5_PARAMS_UNCHANGED": params_unchanged,
        "G6_COMMON_V2_SUMS": common_ev["common_v2_sha256sums_self_check"]
            == "PASS (%d/%d)" % (FROZEN_V2_COMMON_SUMS_ENTRY_COUNT,
                                 FROZEN_V2_COMMON_SUMS_ENTRY_COUNT),
        "G7_EPISODE_COUNTS_FROM_V2_PROFILE":
            common_ev["profile_episode_counts"]["front_episode_count"] == 8
            and common_ev["profile_episode_counts"]["back_episode_count"] == 8
            and common_ev["profile_episode_counts"]["full_episode_count"] == 64,
        "G8_GPU_ALLOWED": gpu_ev["visible_gpu_uuids"] and all(
            u in proj.CC4_GPU_ALLOWED_UUIDS
            for u in gpu_ev["visible_gpu_uuids"]),
        "G9_V1_FROZEN_PRESERVED":
            common_ev["v1_preservation"]["v1_sha256sums_self_check"]
            == "PASS (%d/%d)" % (proj.FROZEN_COMMON_SUMS_ENTRY_COUNT,
                                 proj.FROZEN_COMMON_SUMS_ENTRY_COUNT)
            and common_ev["v1_preservation"]["v1_formal_ranking_authorized"] is False,
        "G10_PROTOCOL_VERSION_V2":
            common_ev["common_evaluator_protocol_version"]
            == COMMON_EVALUATOR_PROTOCOL_VERSION,
    }
    if spec["loader_kind"] == "cc3_slowgru":
        gates["G11_BOUNDARY_SEMANTICS_UNIT_CHECK"] = bool(boundary_ev)
    ready = {
        "schema": READY_V2DT_SCHEMA,
        "candidate_id": args.candidate_id,
        "runtime_family": spec["runtime_family"],
        "common_evaluator_protocol_version": COMMON_EVALUATOR_PROTOCOL_VERSION,
        "READY_V2DT": all(gates.values()),
        "binding_status": "PASS" if smoke_abort is None else "BLOCKED",
        "smoke_abort": smoke_abort,
        "candidate_class": spec["candidate_class"],
        "counts_toward_student_binding_count": (False if is_teacher else True),
        "teacher_reference_binding": (
            (("PASS" if smoke_abort is None else "BLOCKED") if is_teacher
             else None)),
        "run_class": "INTERFACE_SMOKE",
        "performance_claim_authorized": False,
        "formal_ranking_started": False,
        "formal_ranking_authorized_by_this_file": False,
        "await_independent_secondary_audit": True,
        "gates": gates,
        "generated_at_utc": utc_now_iso(),
        "evidence_files": ["episode_records.jsonl", "projection_record.json",
                           "common_evaluator_binding_result_v2dt.json",
                           "SHA256SUMS_V2DT"],
        "honest_false_discipline": "ANY failed gate keeps READY_V2DT false; no "
            "gate is ever faked to PASS; smoke is not performance",
    }
    write_json(os.path.join(out_dir, "READY_V2DT.json"), ready)

    print("[done] %s binding_status=%s READY_V2DT=%s failed_gates=%s"
          % (args.candidate_id, ready["binding_status"], ready["READY_V2DT"],
             {k: v for k, v in gates.items() if not v} or "none"), flush=True)
    if smoke_abort:
        print("[done] BLOCKED by V2 engine predicate (corruption class): %s @ "
              "%s episode %d/%d (engine_message=%r) — recorded as minimum "
              "blocking evidence; NOT relaxed, NOT skipped"
              % (args.candidate_id, smoke_abort["scenario"],
                 smoke_abort["episode_index"] + 1,
                 smoke_abort["episodes_planned"],
                 smoke_abort["engine_message"]), flush=True)
    print("[done] out=%s" % out_dir, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
