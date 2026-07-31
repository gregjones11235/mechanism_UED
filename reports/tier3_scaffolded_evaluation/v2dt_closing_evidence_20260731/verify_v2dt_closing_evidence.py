#!/usr/bin/env python3
"""Offline re-verification of the V2_DYNAMIC_TOPOLOGY closing evidence bundle.

Run:  python verify_v2dt_closing_evidence.py
Exit 0 iff every check passes. Read-only: touches nothing outside this dir.

Scope of this evidence bundle (all artifacts pulled back from
oseasy@172.25.14.221:/home/oseasy/student_pool_v1, npz/pkl never leave server):

- pool_gate/POOL_BINDING_GATE_V2DT.json (+.sha256): the section-7 pool gate
  rollup (6/6 students + teacher reference + all SHA256SUMS + regression A-F).
- common_v2_ready/COMMON_EVALUATOR_V2_READY.json: status marker flipped to
  READY=true by the gate, pending only INDEPENDENT_SECONDARY_AUDIT.
- bindings/<CANDIDATE_ID>/: 7x {common_evaluator_binding_result_v2dt.json,
  READY_V2DT.json, SHA256SUMS_V2DT, projection_record.json}.
- regression/v2dt_regression_results.json: V2DT regression tests A-F.
- logs/SLOWGRU_PERSISTENT_CANONICAL_98304.r4.log: the last binding rerun that
  brought HEAD uniformity (all 7 at f43d3257...).

This verifier checks internal consistency of the stored bundle ONLY (SHA
sidecars, cross-certificate pin uniformity, gate/READY cross-references,
regression verdicts). Live server re-hashing was done by
cc4 tmp_gen/pool_gate_v2dt_gen.py at gate time; see pool_gate file fields.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V2_HEAD = "f43d3257a915bccc44c3a12e6130f358fb26a74f"
V1_DRIVER = "d0d05ff26ffd1ea0bfd80e4c0364edfe6f5616d4"
STUDENTS = [
    "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
    "RESET128_RMT16_ORIGINAL_VTRACE_98304",
    "BASE_GTRXL_ORIGINAL_VTRACE_98304",
    "CONTROL_CONTINUOUS_98304",
    "SLOWGRU_RESET128_CANONICAL_98304",
    "SLOWGRU_PERSISTENT_CANONICAL_98304",
]
TEACHER = "BASELINE_TEACHER_CKPT17500"
PIN_KEYS = [
    "common_evaluator_sha256", "common_runner_sha256", "evaluation_profile_sha256",
    "metric_schema_sha256", "front_bank_content_sha256", "back_bank_content_sha256",
    "full_profile_sha256", "environment_lock_sha256", "assembly_manifest_v2_sha256",
    "candidate_runtime_abi_sha256",
]
REGRESSION_TESTS = [
    "STATIC_TOPOLOGY_PARITY", "LEGAL_DIG_NO_ABORT", "DYNAMIC_DISTANCE_UPDATE",
    "UNREACHABLE_CONTINUES", "TRUE_INVALID_FAIL_CLOSED", "CONTROL_REPRODUCTION",
]
REGRESSION_LETTERS = "ABCDEF"

failures = []
count = 0


def check(name, cond, detail=""):
    global count
    count += 1
    if not cond:
        failures.append(f"{name}: {detail}")


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


# ---- 1. gate file: sidecar SHA + rollup fields ----------------------------
gate_bytes = (ROOT / "pool_gate/POOL_BINDING_GATE_V2DT.json").read_bytes()
gate_sha = hashlib.sha256(gate_bytes).hexdigest()
sidecar = (ROOT / "pool_gate/POOL_BINDING_GATE_V2DT.json.sha256").read_text().split()[0]
check("GATE_SHA_MATCHES_SIDECAR", gate_sha == sidecar, f"{gate_sha} vs {sidecar}")
gate = json.loads(gate_bytes)
check("GATE_NO_FAILURES", gate["gate_failures"] == [], str(gate["gate_failures"]))
check("GATE_ROLLUP_ALL", bool(
    gate["G_6_STUDENTS_BINDING_PASS"] if "G_6_STUDENTS_BINDING_PASS" in gate
    else gate["gates"]["G_6_STUDENTS_BINDING_PASS"]
) and gate["ALL_SHA256SUMS_PASS"] and gate["REGRESSION_ALL_PASS"])
check("GATE_ALL_INNER_TRUE", all(gate["gates"].values()), str(gate["gates"]))
check("GATE_STUDENTS_6_OF_6", gate["STUDENT_COMMON_BINDING_PASS_COUNT"] == "6/6")
check("GATE_TEACHER_PASS", gate["TEACHER_REFERENCE_BINDING"] == "PASS")
check("GATE_V2_READY", gate["COMMON_EVALUATOR_V2_READY"] is True)
check("GATE_NO_FORMAL_RANKING", gate["FORMAL_RANKING_STARTED"] is False
      and gate["FORMAL_RANKING_AUTHORIZED"] is False)
check("GATE_FORBIDDEN_FLAGS_FALSE",
      gate["CHECKPOINTS_MODIFIED"] is False and gate["CONTROL_RETRAINED"] is False
      and gate["CANDIDATE_EXCEPTION_USED"] is False
      and gate["FROZEN_BANKS_MODIFIED"] is False)
check("GATE_SERVER_HEAD", gate["server_git_head"] == V2_HEAD, gate["server_git_head"])
check("GATE_V1_DRIVER_REF", gate["v1_supersession_references"]["v1_driver_commit"] == V1_DRIVER)
for letter, t in zip(REGRESSION_LETTERS, REGRESSION_TESTS):
    key = f"REGRESSION_{letter}_{t}"
    check(f"GATE_{key}", str(gate[key]).startswith("PASS"), gate[key])

# ---- 2. READY marker -------------------------------------------------------
ready = load("common_v2_ready/COMMON_EVALUATOR_V2_READY.json")
check("READY_TRUE", ready["COMMON_EVALUATOR_V2_READY"] is True)
check("READY_PENDING_ONLY_AUDIT", ready["pending_gates"] == ["INDEPENDENT_SECONDARY_AUDIT"],
      str(ready["pending_gates"]))
check("READY_6_OF_6", ready["STUDENT_COMMON_BINDING_PASS_COUNT"] == "6/6")
check("READY_CITES_GATE_SHA", ready["binding_gate_file_sha256"] == gate_sha)
check("READY_NO_RANKING", ready["FORMAL_RANKING_STARTED"] is False)
check("READY_PROTOCOL", ready["common_evaluator_protocol_version"] == "V2_DYNAMIC_TOPOLOGY")
check("READY_V1_SUPERSEDED",
      ready["supersedes_v1"]["status"] == "SUPERSEDED_PRE_RANKING"
      and ready["supersedes_v1"]["driver_commit"] == V1_DRIVER)

# ---- 3. seven certificates: uniform + complete -----------------------------
pins0 = None
for cid in STUDENTS + [TEACHER]:
    base = f"bindings/{cid}"
    cert = load(f"{base}/common_evaluator_binding_result_v2dt.json")
    rdy = load(f"{base}/READY_V2DT.json")
    check(f"{cid}:BINDING_PASS", cert["binding_status"] == "PASS")
    check(f"{cid}:READY_TRUE", rdy["READY_V2DT"] is True)
    check(f"{cid}:INNER_GATES_ALL_TRUE_GE10", len(rdy["gates"]) >= 10 and all(rdy["gates"].values()),
          str(rdy["gates"]))
    check(f"{cid}:HEAD", cert["git_commit_head"] == V2_HEAD, cert["git_commit_head"])
    check(f"{cid}:PROTOCOL", cert["common_evaluator_protocol_version"] == "V2_DYNAMIC_TOPOLOGY")
    check(f"{cid}:BFS_CURRENT", "CURRENT_ENVIRONMENT_STATE_TOPOLOGY" in cert["bfs_graph_source"],
          cert["bfs_graph_source"])
    check(f"{cid}:NO_ABORT", rdy.get("smoke_abort") is None)
    check(f"{cid}:NO_RANKING_IN_CERT", cert["formal_ranking_started"] is False
          and cert["performance_claim_authorized"] is False)
    check(f"{cid}:PARAMS_UNCHANGED", cert["params_unchanged"] is True)
    check(f"{cid}:RECOMPUTE_STATUS",
          cert["checkpoint_file_sha256_status"] == "CC4_RECOMPUTED_MATCH_VIA_OWNER_PROTOCOL",
          cert["checkpoint_file_sha256_status"])
    check(f"{cid}:COMMON_SHA_PASS", str(cert["common_sha_match_status"]).startswith("PASS"),
          str(cert["common_sha_match_status"]))
    v1 = cert["v1_supersession"]
    check(f"{cid}:V1_STATUS", v1["COMMON_EVALUATOR_V1_STATUS"] == "SUPERSEDED_PRE_RANKING")
    check(f"{cid}:V1_DRIVER", v1["COMMON_EVALUATOR_V1_DRIVER"] == V1_DRIVER)
    check(f"{cid}:V1_NEVER_RANKED", v1["v1_formal_ranking_ever_authorized"] is False)
    pins = {k: cert.get(k) for k in PIN_KEYS}
    missing = [k for k, v in pins.items() if not v]
    check(f"{cid}:PINS_PRESENT", not missing, str(missing))
    if pins0 is None:
        pins0 = pins
    else:
        diff = {k: (pins[k], pins0[k]) for k in PIN_KEYS if pins[k] != pins0[k]}
        check(f"{cid}:PINS_UNIFORM", not diff, str(diff))
    # per-binding SHA256SUMS_V2DT: 3 entries, syntactically well-formed
    sums_lines = [ln for ln in (ROOT / f"{base}/SHA256SUMS_V2DT").read_text().splitlines() if ln.strip()]
    check(f"{cid}:SUMS_3_ENTRIES", len(sums_lines) == 3, str(len(sums_lines)))
    check(f"{cid}:SUMS_HEX64", all(len(ln.split()[0]) == 64 for ln in sums_lines))

teacher_cert = load(f"bindings/{TEACHER}/common_evaluator_binding_result_v2dt.json")
check("TEACHER_REFERENCE_ONLY",
      teacher_cert["counts_toward_student_binding_count"] is False
      and teacher_cert["candidate_class"] == "TEACHER_REFERENCE")
for cid in STUDENTS:
    c = load(f"bindings/{cid}/common_evaluator_binding_result_v2dt.json")
    check(f"{cid}:CLASS_STUDENT_COUNTS",
          c["candidate_class"] == "STUDENT" and c["counts_toward_student_binding_count"] is True)

# ---- 4. regression results file -------------------------------------------
reg = load("regression/v2dt_regression_results.json")
for t in REGRESSION_TESTS:
    entry = reg[t]
    verdict = entry["verdict"] if isinstance(entry, dict) else entry
    check(f"REGFILE_{t}", str(verdict).startswith("PASS"), verdict)
check("REGFILE_PROTOCOL", reg["common_evaluator_protocol_version"] == "V2_DYNAMIC_TOPOLOGY")
f = reg["CONTROL_REPRODUCTION"]
check("REGFILE_CONTROL_REPRODUCED",
      f.get("v1_engine_aborts_with_historical_verdict") is True
      and f.get("instrumented_actions_match_official_record") is True)

# ---- verdict ----------------------------------------------------------------
print(f"checks run: {count}")
if failures:
    print(f"FAIL ({len(failures)}):")
    for f_ in failures:
        print("  -", f_)
    sys.exit(1)
print("V2DT_CLOSING_EVIDENCE_VERIFY_PASS (bundle internally consistent)")
print(f"  V2 driver HEAD : {V2_HEAD}")
print(f"  V1 driver (SUPERSEDED_PRE_RANKING): {V1_DRIVER}")
print("  formal ranking : NOT started; awaiting INDEPENDENT_SECONDARY_AUDIT")
sys.exit(0)
